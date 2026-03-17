"""
多要因倒木リスクスコアリングモジュール

アルゴリズム設計根拠:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1] 植生健全性スコア (Vegetation Health Score, VHS)
    - NDVI (Tucker 1979): 健全植生 >0.6, ストレス <0.4
    - NBR (Key & Benson 2006): 立ち枯れ・ストレス検出
    - NDVI衰退トレンド (Puliti et al. 2020 nVVI ベース)
    - 重み: 25% (主要リスク因子)

[2] 風曝露スコア (Wind Exposure Score, WES)
    - 地形的風曝露指数 (TPI + 傾斜 + 斜面向き)
    - 台風主要風向 (南東方向) への曝露
    - Peltola et al. (1999) ForestGALES モデル参照
    - 重み: 25% (最重要因子)

[3] 斜面安定性スコア (Slope Stability Score, SSS)
    - 傾斜角: >30度で根が弱くなる (Kamimura et al. 2008)
    - TWI: 高値 = 土壌水分集積 → 根腐れリスク
    - SAR土壌水分: 降雨後の根の保持力低下
    - 重み: 20%

[4] 樹高スコア (Canopy Height Score, CHS)
    - 高木ほど鉄道への到達距離が長い
    - Mitchell (2013): 倒木到達距離 = 0.7〜1.2 × 樹高
    - 重み: 20%

[5] 暴風歴スコア (Storm History Score, SHS)
    - 標高・地形から台風リスクを代替評価
    - Nakamura et al. (2021) 北日本台風被害パターン参照
    - 重み: 10%

総合リスクスコア = Σ(重み × 各サブスコア) × 100
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import math
from typing import Dict, Tuple


def compute_vegetation_health_score(indices: Dict) -> Tuple[float, Dict]:
    """
    植生健全性スコアを計算する (0=健全/低リスク, 1=衰退/高リスク)

    衰退した樹木は構造的強度が低下し倒木リスクが高まる。
    枯れ木・立ち枯れ木は特に強風時に危険。
    """
    ndvi = float(indices.get("ndvi_mean") or 0.5)
    nbr = float(indices.get("nbr_mean") or 0.4)
    ndre = float(indices.get("ndre_mean") or 0.3)
    ndvi_trend = float(indices.get("ndvi_trend") or 0.0)

    # NDVI スコア: 低NDVI = 高リスク
    # 閾値: 0.7以上=健全(スコア0), 0.2以下=枯死(スコア1)
    ndvi_score = max(0.0, min(1.0, (0.7 - ndvi) / 0.5))

    # NBR スコア: 低NBR = ストレス・燃焼リスク
    # 閾値: 0.5以上=健全(スコア0), -0.1以下=高ストレス(スコア1)
    nbr_score = max(0.0, min(1.0, (0.5 - nbr) / 0.6))

    # NDRE スコア: 葉緑素量 (早期ストレス指標)
    ndre_score = max(0.0, min(1.0, (0.4 - ndre) / 0.4))

    # NDVI トレンドスコア: 減少傾向 = 高リスク
    # -0.1以下(急減): スコア1, +0.05以上(増加): スコア0
    trend_score = max(0.0, min(1.0, (-ndvi_trend + 0.05) / 0.15))

    # 加重平均 (NDVI最重視)
    vhs = 0.40 * ndvi_score + 0.25 * nbr_score + 0.15 * ndre_score + 0.20 * trend_score

    detail = {
        "ndvi_score": round(ndvi_score, 3),
        "nbr_score": round(nbr_score, 3),
        "ndre_score": round(ndre_score, 3),
        "trend_score": round(trend_score, 3),
        "vhs_raw": round(vhs, 3),
    }
    return min(1.0, max(0.0, vhs)), detail


def compute_wind_exposure_score(indices: Dict) -> Tuple[float, Dict]:
    """
    地形的風曝露スコアを計算する (0=遮蔽/低リスク, 1=曝露/高リスク)

    ForestGALES (Gardiner et al. 2000) の知見:
    - 尾根・山頂: 谷よりも2〜3倍の風速
    - 台風は日本に南〜南東から接近
    - 傾斜地は水平面より風荷重を多く受ける
    """
    wind_exposure = float(indices.get("wind_exposure") or 0.5)
    tpi = float(indices.get("tpi") or 0.0)
    slope_deg = float(indices.get("slope_deg") or 0.0)
    aspect_deg = float(indices.get("aspect_deg") or 180.0)
    elevation_m = float(indices.get("elevation_m") or 0.0)

    # 地形位置指数: 尾根(+) → 高リスク
    # TPI > 50m: 明確な尾根 → スコア1
    # TPI < -50m: 谷 → スコア0
    tpi_score = max(0.0, min(1.0, (tpi + 50) / 100))

    # 傾斜スコア: 急傾斜 = 高リスク (根の保持力低下)
    slope_score = max(0.0, min(1.0, slope_deg / 35.0))

    # 斜面向きスコア: 台風風向(SE=135度)に正対する斜面が高リスク
    # 南東向き(135度)= 最高リスク
    aspect_diff = abs(aspect_deg - 135) % 360
    if aspect_diff > 180:
        aspect_diff = 360 - aspect_diff
    aspect_score = max(0.0, min(1.0, 1.0 - aspect_diff / 180.0))

    # 標高スコア: 高標高ほど風速が強い
    # 1000m以上を高リスクとする
    elev_score = max(0.0, min(1.0, elevation_m / 1500.0))

    # 地形的風曝露 (GEEで計算済み) との組み合わせ
    wes = (
        0.30 * wind_exposure
        + 0.25 * tpi_score
        + 0.20 * slope_score
        + 0.15 * aspect_score
        + 0.10 * elev_score
    )

    detail = {
        "tpi_score": round(tpi_score, 3),
        "slope_score": round(slope_score, 3),
        "aspect_score": round(aspect_score, 3),
        "elev_score": round(elev_score, 3),
        "wind_exposure_raw": round(wind_exposure, 3),
        "wes_raw": round(wes, 3),
    }
    return min(1.0, max(0.0, wes)), detail


def compute_slope_stability_score(indices: Dict) -> Tuple[float, Dict]:
    """
    斜面安定性スコアを計算する (0=安定/低リスク, 1=不安定/高リスク)

    根系の保持力を下げる要因:
    - 急傾斜: 根の横方向支持が弱い
    - 高土壌水分: 降雨後に根の保持力低下
    - 高TWI: 水が集積しやすい地形 = 根腐れリスク
    (Kamimura et al. 2008, Peltola et al. 2000)
    """
    slope_deg = float(indices.get("slope_deg") or 0.0)
    twi = float(indices.get("twi") or 2.0)
    soil_moisture = float(indices.get("soil_moisture_proxy") or 0.3)

    # 傾斜スコア
    # 30度以上で根の保持力が急減 (Coutts 1983)
    slope_score = max(0.0, min(1.0, slope_deg / 40.0))

    # TWIスコア: 高TWI = 常に湿潤 = 根腐れ
    # TWI > 8: 常時湿潤地形
    twi_score = max(0.0, min(1.0, (twi - 1.0) / 7.0))

    # 土壌水分スコア (SAR由来)
    # 0=乾燥(安定), 1=湿潤(不安定)
    moisture_score = float(soil_moisture)

    # 組み合わせ: 傾斜と水分の相乗効果を考慮
    sss = 0.40 * slope_score + 0.30 * twi_score + 0.30 * moisture_score

    # 傾斜 × 水分の相乗効果 (急傾斜かつ湿潤 = 最大リスク)
    synergy = slope_score * moisture_score * 0.2
    sss = min(1.0, sss + synergy)

    detail = {
        "slope_score": round(slope_score, 3),
        "twi_score": round(twi_score, 3),
        "moisture_score": round(moisture_score, 3),
        "synergy_bonus": round(synergy, 3),
        "sss_raw": round(sss, 3),
    }
    return min(1.0, max(0.0, sss)), detail


def compute_canopy_height_score(indices: Dict) -> Tuple[float, Dict]:
    """
    樹高スコアを計算する (0=低木/低リスク, 1=高木/高リスク)

    Mitchell (2013) の研究:
    - 鉄道脇50m以内の樹木高さが点検優先度の主要因
    - 高木は倒木時の運動エネルギーが大きい
    - 到達距離 ≈ 0.9 × 樹高 (地上高から)
    """
    mean_h = float(indices.get("canopy_height_mean_m") or 10.0)
    p90_h = float(indices.get("canopy_height_p90_m") or 15.0)
    height_risk = float(indices.get("height_risk_score") or 0.5)

    # P90樹高スコア: 90パーセンタイル樹高を使用 (外れ値の高木を考慮)
    # バッファ50m内で30m超の木が届く可能性は高い
    p90_score = max(0.0, min(1.0, p90_h / 30.0))

    # 平均樹高スコア
    mean_score = max(0.0, min(1.0, mean_h / 25.0))

    chs = 0.60 * p90_score + 0.40 * mean_score

    detail = {
        "mean_height_m": round(mean_h, 1),
        "p90_height_m": round(p90_h, 1),
        "p90_score": round(p90_score, 3),
        "mean_score": round(mean_score, 3),
        "chs_raw": round(chs, 3),
    }
    return min(1.0, max(0.0, chs)), detail


def compute_storm_history_score(indices: Dict) -> Tuple[float, Dict]:
    """
    暴風リスクスコアを計算する (地形代替指標)

    台風頻度と強度が高い地域:
    - 太平洋沿岸・南部: 高リスク
    - 内陸山間部: 局所的な暴風 (フェーン現象等)
    - 高標高: 強風曝露

    完全実装では気象庁台風トラックデータを使用
    """
    elevation_m = float(indices.get("elevation_m") or 0.0)
    slope_deg = float(indices.get("slope_deg") or 0.0)
    wind_exposure = float(indices.get("wind_exposure") or 0.5)

    # 標高ベース台風リスク
    # 低標高(沿岸): 台風直撃リスク高
    # 高標高: 強風リスク高
    if elevation_m < 100:
        elev_storm_score = 0.8  # 沿岸・平野: 台風直撃
    elif elevation_m < 500:
        elev_storm_score = 0.6
    elif elevation_m < 1000:
        elev_storm_score = 0.7  # 山腹: 強風
    else:
        elev_storm_score = 0.9  # 山頂: 最強風

    # 地形的風曝露との組み合わせ
    shs = 0.50 * wind_exposure + 0.50 * elev_storm_score

    detail = {
        "elev_storm_score": round(elev_storm_score, 3),
        "wind_exp_component": round(wind_exposure, 3),
        "shs_raw": round(shs, 3),
    }
    return min(1.0, max(0.0, shs)), detail


def compute_composite_risk_score(
    indices: Dict,
    weights: Dict,
) -> Dict:
    """
    全因子を統合した複合リスクスコアを計算する

    Returns:
        risk_score: 0〜100のリスクスコア
        priority: 点検優先度 (critical/high/medium/low)
        sub_scores: 各因子のスコア詳細
        thresholds: 優先度判定閾値
    """
    vhs, vhs_detail = compute_vegetation_health_score(indices)
    wes, wes_detail = compute_wind_exposure_score(indices)
    sss, sss_detail = compute_slope_stability_score(indices)
    chs, chs_detail = compute_canopy_height_score(indices)
    shs, shs_detail = compute_storm_history_score(indices)

    w = weights
    composite = (
        w.get("vegetation_health", 0.25) * vhs
        + w.get("wind_exposure", 0.25) * wes
        + w.get("slope_stability", 0.20) * sss
        + w.get("canopy_height", 0.20) * chs
        + w.get("storm_history", 0.10) * shs
    )

    risk_score = round(composite * 100, 1)

    # 点検優先度判定
    thresholds = {
        "critical": 75,
        "high": 55,
        "medium": 35,
    }
    if risk_score >= thresholds["critical"]:
        priority = "critical"
        priority_label = "緊急点検"
    elif risk_score >= thresholds["high"]:
        priority = "high"
        priority_label = "高優先度"
    elif risk_score >= thresholds["medium"]:
        priority = "medium"
        priority_label = "中優先度"
    else:
        priority = "low"
        priority_label = "低優先度"

    return {
        "risk_score": risk_score,
        "priority": priority,
        "priority_label": priority_label,
        # サブスコア (0〜100)
        "vegetation_health_score": round(vhs * 100, 1),
        "wind_exposure_score": round(wes * 100, 1),
        "slope_stability_score": round(sss * 100, 1),
        "canopy_height_score": round(chs * 100, 1),
        "storm_history_score": round(shs * 100, 1),
        # 詳細診断
        "detail": {
            "vegetation": vhs_detail,
            "wind": wes_detail,
            "slope": sss_detail,
            "canopy": chs_detail,
            "storm": shs_detail,
        },
        # 生の指標値
        "raw_indices": {
            "ndvi": round(float(indices.get("ndvi_mean") or 0), 3),
            "nbr": round(float(indices.get("nbr_mean") or 0), 3),
            "ndvi_trend": round(float(indices.get("ndvi_trend") or 0), 3),
            "slope_deg": round(float(indices.get("slope_deg") or 0), 1),
            "elevation_m": round(float(indices.get("elevation_m") or 0), 0),
            "canopy_height_p90_m": round(
                float(indices.get("canopy_height_p90_m") or 0), 1
            ),
            "soil_moisture_proxy": round(
                float(indices.get("soil_moisture_proxy") or 0), 3
            ),
            "wind_exposure": round(float(indices.get("wind_exposure") or 0), 3),
        },
    }


def rank_segments(segments: list) -> list:
    """
    全セグメントをリスクスコア降順でランキングする

    Args:
        segments: 各セグメントの result dict のリスト

    Returns:
        inspection_rank 付きで昇順ランキングされたリスト
    """
    sorted_segs = sorted(
        segments, key=lambda x: x.get("risk_score", 0), reverse=True
    )
    for rank, seg in enumerate(sorted_segs, 1):
        seg["inspection_rank"] = rank
    return sorted_segs
