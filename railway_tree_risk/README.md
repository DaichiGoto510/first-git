# 鉄道脇 倒木リスク推定システム

衛星データ（Sentinel-1/2、SRTM DEM、GEDI）を用いて、鉄道路線沿いの倒木リスクを推定し、点検優先順位を出力するCLIツールです。

## アーキテクチャ概要

```
鉄道路線 GeoJSON
      │
      ▼
┌─────────────────┐
│ セグメント分割   │  500m 毎に分割 + 50m バッファ
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│         Google Earth Engine 衛星処理         │
│                                             │
│  Sentinel-2 → NDVI / NBR / NDRE / トレンド  │
│  Sentinel-1 → VV後方散乱 / 土壌水分代替指標  │
│  SRTM DEM   → 傾斜 / TWI / TPI / 風曝露    │
│  GEDI/CHM   → 樹高 (P90)                   │
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│         多因子リスクスコアリング              │
│                                             │
│  植生健全性スコア  × 0.25                    │
│  風曝露スコア      × 0.25                    │
│  斜面安定性スコア  × 0.20                    │
│  樹高スコア        × 0.20                    │
│  暴風歴スコア      × 0.10                    │
│                                             │
│  → 総合リスクスコア 0〜100                   │
└────────┬────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  出力                         │
│  - GeoJSON (QGIS等で表示)     │
│  - CSV (Excel対応)            │
│  - HTML インタラクティブ地図   │
└──────────────────────────────┘
```

## アルゴリズム設計

### リスク因子と根拠文献

| 因子 | 重み | 使用データ | 根拠文献 |
|------|------|-----------|---------|
| **植生健全性** | 25% | Sentinel-2 NDVI/NBR/NDRE + 時系列トレンド | Puliti et al. (2020) nVVI index |
| **風曝露** | 25% | SRTM DEM (TPI + 傾斜 + 斜面向き + 標高) | Gardiner et al. (2000) ForestGALES; Peltola et al. (1999) |
| **斜面安定性** | 20% | SRTM DEM (TWI) + Sentinel-1 土壌水分 | Kamimura et al. (2008); Beven & Kirkby (1979) |
| **樹高** | 20% | GEDI / ETH Global Canopy Height | Mitchell (2013) 倒木到達距離 ≈ 0.9×樹高 |
| **暴風歴** | 10% | 地形代替指標 (標高・風曝露) | Nakamura et al. (2021) 台風被害パターン |

### 点検優先度クラス

| クラス | リスクスコア | 推奨対応 |
|--------|------------|---------|
| 🔴 緊急点検 | 75 以上 | 速やかに現地確認・危険木の処置 |
| 🟠 高優先度 | 55〜75 | 次回巡回時に優先確認 |
| 🟡 中優先度 | 35〜55 | 定期巡回時に確認 |
| 🟢 低優先度 | 35 未満 | 通常の維持管理 |

### 植生健全性スコア詳細

```
VHS = 0.40×NDVI_score + 0.25×NBR_score + 0.15×NDRE_score + 0.20×trend_score

NDVI_score: (0.7 - NDVI) / 0.5  → NDVI<0.2で1.0(最高リスク)
NBR_score : (0.5 - NBR ) / 0.6  → NBR<-0.1で1.0(立ち枯れ)
trend_score: 直近30日 vs 60-90日前のNDVI差分
```

### 風曝露スコア詳細

```
WES = 0.30×wind_exposure + 0.25×TPI_score + 0.20×slope_score
      + 0.15×aspect_score + 0.10×elev_score

wind_exposure: GEEで計算 (標高+傾斜+斜面向きの合成)
TPI_score    : 地形位置指数 (尾根=1.0, 谷=0.0)
aspect_score : 台風主要風向(SE=135°)への曝露度
```

## セットアップ

### 1. 必要要件

- Python 3.9 以上
- Google Earth Engine アカウント（無料）: https://earthengine.google.com/

### 2. インストール

```bash
pip install -r requirements.txt
```

### 3. GEE 認証

```bash
# 初回のみ実行 (ブラウザが開いて認証)
earthengine authenticate
```

### 4. 動作確認 (GEE不要)

```bash
# --dry-run でダミーデータを使って動作確認
python risk_estimator.py \
  --railway data/sample_railway.geojson \
  --output output/ \
  --name "サンプル路線" \
  --dry-run
```

### 5. 実際の解析

```bash
python risk_estimator.py \
  --railway your_railway.geojson \
  --output output/ \
  --name "○○線" \
  --project your-gee-project-id
```

## 鉄道路線 GeoJSON の準備

### OpenStreetMapからダウンロード (Overpass API)

```python
import requests, json

# 例: 富良野線の取得
query = """
[out:json][timeout:60];
(
  relation["name"="富良野線"]["route"="train"];
);
out geom;
"""
r = requests.get(
    "https://overpass-api.de/api/interpreter",
    params={"data": query}
)
# 結果を GeoJSON に変換して使用
```

### または QGIS で作成

1. QGIS > 新規ラインレイヤー作成
2. 対象路線をデジタイズ
3. `エクスポート > GeoJSON` で保存

## 出力ファイル

```
output/
├── risk_YYYYMMDD_HHMMSS.geojson   # GIS読み込み用
├── risk_YYYYMMDD_HHMMSS.csv       # Excel対応 (BOM付きUTF-8)
└── risk_YYYYMMDD_HHMMSS_map.html  # ブラウザで確認
```

### HTML地図の見方

- **赤線**: 緊急点検が必要なセグメント
- **橙線**: 高優先度セグメント
- **黄線**: 中優先度セグメント
- **緑線**: 低優先度セグメント
- セグメントをクリックすると詳細情報が表示されます

## 設定カスタマイズ

`config.yaml` で各種パラメータを変更できます:

```yaml
buffer_distance_m: 50      # 解析対象バッファ幅
segment_length_m: 500      # セグメント長
sentinel2_days: 90         # Sentinel-2 取得期間
cloud_cover_threshold: 20  # 雲量フィルタ

risk_weights:
  vegetation_health: 0.25  # 植生健全性の重み
  wind_exposure: 0.25      # 風曝露の重み
  slope_stability: 0.20    # 斜面安定性の重み
  canopy_height: 0.20      # 樹高の重み
  storm_history: 0.10      # 暴風歴の重み
```

## 参考文献

- Puliti S. et al. (2020). A New Index for Assessing Tree Vigour Decline Based on Sentinel-2. *Remote Sensing Letters*, 12(1).
- Gardiner B. et al. (2000). ForestGALES: A PC-based wind risk model for British forests. Forestry Commission.
- Peltola H. et al. (1999). A mechanistic model for assessing the risk of wind and snow damage to single trees. *Silva Fennica*, 33(1).
- Kamimura K. et al. (2008). Effects of thinning on wind and snow damage in forests. *Forest Ecology and Management*.
- Mitchell S.J. (2013). Predicting the risk of tree fall onto railway lines. *Forest Ecology and Management*.
- Beven K.J. & Kirkby M.J. (1979). A physically based variable contributing area model of basin hydrology. *Hydrological Sciences Bulletin*.
- Rüetschi M. et al. (2019). Rapid detection of windthrows using Sentinel-1 C-band SAR data. *Remote Sensing*, 11(2).
- Nakamura T. et al. (2021). Risk assessment of forest disturbance by typhoons with heavy precipitation in northern Japan. *Forest Ecology and Management*.
