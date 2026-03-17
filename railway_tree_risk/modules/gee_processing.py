"""
GEE (Google Earth Engine) による衛星データ処理モジュール

使用データソース:
- Sentinel-2 MSI: 植生健全性 (NDVI, nVVI) [10m解像度, 5日周期]
- Sentinel-1 SAR: 土壌水分推定 [10m解像度, 6日周期]
- SRTM DEM: 地形解析 (傾斜, TWI, 地形的風曝露) [30m解像度]
- GEDI: 樹高推定 [25mフットプリント]

参考文献:
- Kamimura et al. (2008) Effects of thinning on wind and snow damage in forests
- Peltola (2006) Mechanical stability of trees under static loading
- Mitchell (2013) Catching the wind in more than a net
- Rüetschi et al. (2019) Rapid detection of windthrows using Sentinel-1 SAR data
- Puliti et al. (2020) nVVI for tree vigour decline assessment
"""

import ee
import json
import math
from datetime import datetime, timedelta
from typing import Dict, Tuple


def initialize_gee(project_id: str = None):
    """Google Earth Engine を初期化する"""
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        print("✓ Google Earth Engine 初期化成功")
    except Exception as e:
        raise RuntimeError(
            f"GEE初期化失敗: {e}\n"
            "以下を確認してください:\n"
            "  1. earthengine authenticate を実行済みか\n"
            "  2. GEEアカウントが有効か (https://earthengine.google.com/)\n"
            "  3. --project オプションでプロジェクトIDを指定しているか"
        )


def get_analysis_dates(sentinel2_days: int, sentinel1_days: int) -> Dict[str, str]:
    """解析対象の日付範囲を計算する"""
    end_date = datetime.now()
    return {
        "s2_start": (end_date - timedelta(days=sentinel2_days)).strftime("%Y-%m-%d"),
        "s2_end": end_date.strftime("%Y-%m-%d"),
        "s1_start": (end_date - timedelta(days=sentinel1_days)).strftime("%Y-%m-%d"),
        "s1_end": end_date.strftime("%Y-%m-%d"),
    }


def geojson_to_ee_geometry(geojson_geometry: dict) -> ee.Geometry:
    """GeoJSON geometry を EE Geometry に変換する"""
    return ee.Geometry(geojson_geometry)


def get_sentinel2_indices(
    roi: ee.Geometry,
    start_date: str,
    end_date: str,
    cloud_threshold: int = 20,
) -> Dict[str, float]:
    """
    Sentinel-2 から植生指標を計算する

    Returns:
        ndvi_mean: 期間平均NDVI (-1〜1, 健全: >0.6)
        ndvi_trend: NDVI変化トレンド (負値 = 衰退傾向)
        nbr_mean: 正規化燃焼比 (植生ストレス指標)
    """
    # Sentinel-2 Surface Reflectance コレクション
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
    )

    def add_ndvi(image):
        ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
        return image.addBands(ndvi)

    def add_nbr(image):
        # NBR = (NIR - SWIR) / (NIR + SWIR)
        # 低NBR = 植生ストレス・立ち枯れリスク
        nbr = image.normalizedDifference(["B8", "B12"]).rename("NBR")
        return image.addBands(nbr)

    def add_ndre(image):
        # NDRE = (RedEdge - Red) / (RedEdge + Red)
        # 葉緑素含量に敏感 → 早期ストレス検出
        ndre = image.normalizedDifference(["B8A", "B4"]).rename("NDRE")
        return image.addBands(ndre)

    s2_with_indices = s2.map(add_ndvi).map(add_nbr).map(add_ndre)
    count = s2_with_indices.size().getInfo()

    if count == 0:
        print("  警告: 対象期間内のS2画像なし。雲量閾値を緩和します。")
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start_date, end_date)
            .filterBounds(roi)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
        )
        s2_with_indices = s2.map(add_ndvi).map(add_nbr).map(add_ndre)

    median_composite = s2_with_indices.median()

    # ROI内の平均値を計算
    stats = median_composite.select(["NDVI", "NBR", "NDRE"]).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=20,
        maxPixels=1e8,
    )

    # NDVI トレンド計算: 直近30日 vs 60-90日前 の変化
    recent_end = datetime.strptime(end_date, "%Y-%m-%d")
    mid_date = (recent_end - timedelta(days=30)).strftime("%Y-%m-%d")
    older_start = (recent_end - timedelta(days=90)).strftime("%Y-%m-%d")

    ndvi_recent = (
        s2_with_indices.filterDate(mid_date, end_date)
        .select("NDVI")
        .mean()
        .reduceRegion(
            reducer=ee.Reducer.mean(), geometry=roi, scale=20, maxPixels=1e8
        )
    )
    ndvi_older = (
        s2_with_indices.filterDate(older_start, mid_date)
        .select("NDVI")
        .mean()
        .reduceRegion(
            reducer=ee.Reducer.mean(), geometry=roi, scale=20, maxPixels=1e8
        )
    )

    result = stats.getInfo()
    recent_val = ndvi_recent.getInfo().get("NDVI")
    older_val = ndvi_older.getInfo().get("NDVI")

    ndvi_trend = 0.0
    if recent_val is not None and older_val is not None:
        ndvi_trend = float(recent_val) - float(older_val)

    return {
        "ndvi_mean": result.get("NDVI") or 0.0,
        "nbr_mean": result.get("NBR") or 0.0,
        "ndre_mean": result.get("NDRE") or 0.0,
        "ndvi_trend": ndvi_trend,
    }


def get_sentinel1_soil_moisture(
    roi: ee.Geometry, start_date: str, end_date: str
) -> Dict[str, float]:
    """
    Sentinel-1 SAR から土壌水分代替指標を計算する

    VV後方散乱: 土壌水分に感度が高い
    VH後方散乱: 植生構造・biomassに感度が高い
    VH/VV比: 植生レーダー植生指標 (RVI) の代替

    参考: Bauer-Marschallinger et al. (2019) SAR soil moisture
    """
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
    )

    def add_rvi(image):
        vv = image.select("VV")
        vh = image.select("VH")
        # RVI4S1: Radar Vegetation Index
        # RVI = (4 * VH) / (VV + VH) → 値が高いほど植生密度高い
        rvi = vh.multiply(4).divide(vv.add(vh)).rename("RVI")
        return image.addBands(rvi)

    s1_processed = s1.map(add_rvi)
    composite = s1_processed.mean()

    stats = composite.select(["VV", "VH", "RVI"]).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=20,
        maxPixels=1e8,
    )

    result = stats.getInfo()
    vv = result.get("VV") or -15.0
    vh = result.get("VH") or -22.0

    # VV dBをリニアに変換して相対土壌水分スコアに正規化
    # 参考値: 乾燥土壌 ~ -20dB, 湿潤土壌 ~ -8dB
    vv_norm = max(0.0, min(1.0, (float(vv) + 20.0) / 12.0))

    return {
        "vv_backscatter_db": float(vv),
        "vh_backscatter_db": float(vh),
        "rvi": result.get("RVI") or 0.5,
        "soil_moisture_proxy": vv_norm,  # 0=乾燥, 1=湿潤
    }


def get_terrain_metrics(roi: ee.Geometry) -> Dict[str, float]:
    """
    SRTM DEMから地形指標を計算する

    指標:
    - slope: 傾斜角 (度)
    - aspect: 斜面向き
    - twi: 地形湿潤指数 (Topographic Wetness Index)
    - tpi: 地形位置指数 (Topographic Position Index): 尾根=正, 谷=負
    - wind_exposure: 地形的風曝露指数

    参考:
    - Beven & Kirkby (1979) TWI formula
    - Wilson & Gallant (2000) Terrain analysis
    """
    dem = ee.Image("USGS/SRTMGL1_003")

    # 傾斜・斜面向き
    terrain = ee.Terrain.products(dem)
    slope = terrain.select("slope")
    aspect = terrain.select("aspect")
    elevation = dem

    # 地形湿潤指数 TWI = ln(A / tan(β))
    # A: 上流集水域面積, β: 傾斜
    # GEE では正確なTWI計算は制約があるため傾斜ベース近似を使用
    slope_rad = slope.multiply(math.pi / 180)
    tan_slope = slope_rad.tan().max(ee.Image(0.001))  # ゼロ除算防止

    # 簡易TWI近似: 傾斜が低い平坦地で高くなる
    twi_approx = tan_slope.pow(-1).log().rename("TWI")

    # 地形位置指数 TPI: 局所的な凸凹 (尾根=高リスク, 谷=低リスク)
    # 近傍平均との差分
    dem_smooth = dem.focal_mean(radius=500, kernelType="circle", units="meters")
    tpi = dem.subtract(dem_smooth).rename("TPI")

    # 地形的風曝露指数: 高標高・正TPI・南西向き斜面が高リスク
    # 日本の主要台風進路を考慮 (南〜南東からの風)
    # 斜面向きスコア: 南東向き(135度)を最高リスクとする
    aspect_rad = aspect.multiply(math.pi / 180)
    wind_dir_rad = ee.Image(135.0 * math.pi / 180)  # 台風主要風向 (SE)
    aspect_score = (
        aspect_rad.subtract(wind_dir_rad).cos().multiply(-1).add(1).divide(2)
    )  # 0〜1

    # 標高正規化 (高いほどリスク大)
    elev_norm = elevation.subtract(0).divide(3000).min(1.0).max(0.0)

    # 傾斜正規化 (45度を1.0とする)
    slope_norm = slope.divide(45).min(1.0)

    # 合成風曝露指数
    wind_exposure = (
        elev_norm.multiply(0.3)
        .add(slope_norm.multiply(0.4))
        .add(aspect_score.multiply(0.3))
        .rename("WindExposure")
    )

    all_bands = (
        slope.rename("slope")
        .addBands(aspect.rename("aspect"))
        .addBands(twi_approx.rename("TWI"))
        .addBands(tpi.rename("TPI"))
        .addBands(wind_exposure.rename("WindExposure"))
        .addBands(elevation.rename("elevation"))
    )

    stats = all_bands.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=30,
        maxPixels=1e8,
    )

    result = stats.getInfo()

    return {
        "slope_deg": result.get("slope") or 0.0,
        "aspect_deg": result.get("aspect") or 180.0,
        "twi": result.get("TWI") or 2.0,
        "tpi": result.get("TPI") or 0.0,
        "wind_exposure": result.get("WindExposure") or 0.5,
        "elevation_m": result.get("elevation") or 0.0,
    }


def get_canopy_height(roi: ee.Geometry) -> Dict[str, float]:
    """
    全球樹冠高さモデルから樹高を推定する

    使用データ:
    1. Lang et al. (2023) ETH Global Canopy Height 2020 [10m] (優先)
    2. GEDI L3 Canopy Height (代替)

    高い樹木は倒木時の到達距離が長くなるため高リスク
    参考: Mitchell (2013) 鉄道への倒木到達距離は樹高の0.7〜1.2倍
    """
    try:
        # ETH Global Canopy Height Map 2020 (Lang et al. 2023)
        # https://gee-community-catalog.org/projects/canopy/
        canopy_height = ee.Image("users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1")
        stats = canopy_height.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                ee.Reducer.percentile([75, 90]), sharedInputs=True
            ),
            geometry=roi,
            scale=30,
            maxPixels=1e8,
        )
        result = stats.getInfo()
        mean_h = result.get("b1_mean") or result.get("mean") or 10.0
        p90_h = result.get("b1_p90") or result.get("p90") or 15.0
    except Exception:
        # GEDI L4B フォールバック
        try:
            gedi = ee.Image("LARSE/GEDI/GEDI04_B_002")
            stats = gedi.select("MU").reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=roi,
                scale=30,
                maxPixels=1e8,
            )
            result = stats.getInfo()
            mean_h = result.get("MU") or 10.0
            p90_h = mean_h * 1.3
        except Exception:
            # データなし: デフォルト値 (日本の山間部平均的樹高)
            mean_h = 12.0
            p90_h = 18.0

    # 樹高リスクスコア: 線路に届く可能性を考慮
    # Mitchell (2013): 倒木到達距離 ≈ 0.9 × 樹高
    # バッファ50m内の樹高が50mを超えると最大リスク
    height_risk = min(1.0, float(p90_h) / 30.0)

    return {
        "canopy_height_mean_m": float(mean_h),
        "canopy_height_p90_m": float(p90_h),
        "height_risk_score": height_risk,
    }


def fetch_all_indices(
    roi_geojson: dict,
    config: dict,
) -> Dict[str, float]:
    """
    ROI (バッファ済み鉄道セグメント) の全衛星指標を取得する

    Args:
        roi_geojson: GeoJSON geometry (Polygon)
        config: 設定辞書

    Returns:
        全衛星指標の辞書
    """
    roi = geojson_to_ee_geometry(roi_geojson)
    dates = get_analysis_dates(
        config.get("sentinel2_days", 90),
        config.get("sentinel1_days", 30),
    )

    # 各データソースから指標を取得
    s2 = get_sentinel2_indices(
        roi,
        dates["s2_start"],
        dates["s2_end"],
        config.get("cloud_cover_threshold", 20),
    )
    s1 = get_sentinel1_soil_moisture(roi, dates["s1_start"], dates["s1_end"])
    terrain = get_terrain_metrics(roi)
    canopy = get_canopy_height(roi)

    return {**s2, **s1, **terrain, **canopy}
