#!/usr/bin/env python3
"""
鉄道脇 倒木リスク推定 CLI ツール
Railway Tree Fall Risk Estimator

使用方法:
    python risk_estimator.py --railway data/sample_railway.geojson --output output/

Google Earth Engine 認証が必要です:
    earthengine authenticate

引数:
    --railway   : 鉄道路線 GeoJSON ファイルパス (LineString/MultiLineString)
    --output    : 出力ディレクトリ
    --config    : 設定ファイル (デフォルト: config.yaml)
    --project   : GEE プロジェクトID (省略可)
    --name      : 路線名 (出力ファイル・地図タイトル用)
    --dry-run   : GEEに接続せずダミーデータで動作確認

出力:
    risk_output.geojson : リスクスコア付きセグメント
    risk_output.csv     : 点検優先順位リスト (Excel対応)
    risk_map.html       : インタラクティブ地図

アルゴリズム詳細:
    modules/risk_calculator.py および modules/gee_processing.py を参照
"""

import os
import sys
import json
import time
import random
import click
import yaml

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.segment_utils import split_railway_to_segments, build_output_geojson
from modules.risk_calculator import compute_composite_risk_score, rank_segments
from modules.visualizer import save_geojson, save_csv, build_html_map, print_summary


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_geojson(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_dummy_indices(segment_id: str, seed: int = 0) -> dict:
    """
    --dry-run 用のダミー衛星指標を生成する
    セグメントごとにランダムシードを変えて多様なリスクパターンを再現
    """
    rng = random.Random(hash(segment_id) + seed)

    # 健全〜衰退の幅を持たせる
    ndvi = rng.uniform(0.15, 0.85)
    nbr = rng.uniform(0.1, 0.65)
    ndre = rng.uniform(0.1, 0.55)
    ndvi_trend = rng.uniform(-0.12, 0.05)

    slope = rng.uniform(0, 40)
    elevation = rng.uniform(10, 1200)
    aspect = rng.uniform(0, 360)

    # 急傾斜は高標高に偏らせる (現実的な地形を模擬)
    if elevation > 600:
        slope = min(40, slope * 1.4)

    twi = rng.uniform(1.0, 9.0)
    vv_db = rng.uniform(-20, -8)
    soil_moisture = max(0.0, min(1.0, (vv_db + 20) / 12))

    canopy_mean = rng.uniform(5, 25)
    canopy_p90 = canopy_mean * rng.uniform(1.2, 1.8)
    height_risk = min(1.0, canopy_p90 / 30.0)

    tpi = rng.uniform(-80, 80)
    wind_exposure = rng.uniform(0.1, 0.9)

    return {
        "ndvi_mean": ndvi,
        "nbr_mean": nbr,
        "ndre_mean": ndre,
        "ndvi_trend": ndvi_trend,
        "vv_backscatter_db": vv_db,
        "vh_backscatter_db": vv_db - 7,
        "rvi": rng.uniform(0.2, 0.8),
        "soil_moisture_proxy": soil_moisture,
        "slope_deg": slope,
        "aspect_deg": aspect,
        "twi": twi,
        "tpi": tpi,
        "wind_exposure": wind_exposure,
        "elevation_m": elevation,
        "canopy_height_mean_m": canopy_mean,
        "canopy_height_p90_m": canopy_p90,
        "height_risk_score": height_risk,
    }


@click.command()
@click.option("--railway", required=True, type=click.Path(exists=True), help="鉄道路線 GeoJSON ファイルパス")
@click.option("--output", default="output", show_default=True, help="出力ディレクトリ")
@click.option("--config", "config_path", default=None, help="設定ファイルパス (デフォルト: config.yaml)")
@click.option("--project", default=None, help="Google Earth Engine プロジェクトID")
@click.option("--name", default="対象路線", show_default=True, help="路線名 (出力タイトル用)")
@click.option("--dry-run", is_flag=True, default=False, help="GEEに接続せずダミーデータで動作確認")
def main(railway, output, config_path, project, name, dry_run):
    """
    鉄道脇の倒木リスクを衛星データから推定し、点検優先順位を出力します。

    Google Earth Engine アカウントが必要です (無料):
    https://earthengine.google.com/

    認証手順:
      pip install earthengine-api
      earthengine authenticate
    """
    # ─── 設定読み込み ───
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)

    print("\n" + "=" * 60)
    print("  🛤️  鉄道脇 倒木リスク推定システム")
    print("=" * 60)
    print(f"  路線名    : {name}")
    print(f"  入力ファイル: {railway}")
    print(f"  出力先    : {output}")
    print(f"  バッファ  : {config['buffer_distance_m']} m")
    print(f"  セグメント: {config['segment_length_m']} m 毎")
    if dry_run:
        print("  モード    : DRY RUN (ダミーデータ)")
    print()

    # ─── GEE 初期化 ───
    if not dry_run:
        from modules.gee_processing import initialize_gee, fetch_all_indices
        initialize_gee(project)
    else:
        print("  [DRY RUN] GEE接続をスキップ")

    # ─── 鉄道路線読み込み ───
    print("  → 鉄道路線データ読み込み中...")
    railway_geojson = load_geojson(railway)

    # ─── セグメント分割 ───
    print("  → セグメント分割中...")
    segments = split_railway_to_segments(
        railway_geojson,
        segment_length_m=config["segment_length_m"],
        buffer_m=config["buffer_distance_m"],
    )
    print(f"  → {len(segments)} セグメントに分割")

    # ─── 衛星データ取得 & リスクスコア計算 ───
    print(f"\n  → 衛星データ取得 & リスクスコア計算開始...")
    results = []
    weights = config.get("risk_weights", {})

    for i, seg in enumerate(segments):
        seg_id = seg["segment_id"]
        progress = f"[{i+1}/{len(segments)}]"
        print(f"  {progress} {seg_id} 処理中...", end="", flush=True)

        try:
            if dry_run:
                indices = generate_dummy_indices(seg_id)
            else:
                from modules.gee_processing import fetch_all_indices
                indices = fetch_all_indices(seg["roi_geojson"], config)

            risk_result = compute_composite_risk_score(indices, weights)

            result = {
                **risk_result,
                "segment_id": seg_id,
                "center_lon": seg["center_lon"],
                "center_lat": seg["center_lat"],
                "coords": seg["coords"],
            }
            results.append(result)
            score = risk_result["risk_score"]
            label = risk_result["priority_label"]
            print(f" スコア={score:5.1f} ({label})")

        except Exception as e:
            print(f" [エラー: {e}]")
            # エラー時はスコア不明として記録
            results.append({
                "segment_id": seg_id,
                "risk_score": -1,
                "priority": "unknown",
                "priority_label": "エラー",
                "center_lon": seg["center_lon"],
                "center_lat": seg["center_lat"],
                "coords": seg["coords"],
            })

        # GEE レート制限対策 (dry-run 時は不要)
        if not dry_run:
            time.sleep(0.5)

    # ─── ランキング ───
    valid_results = [r for r in results if r.get("risk_score", -1) >= 0]
    ranked = rank_segments(valid_results)

    # ─── 出力 ───
    os.makedirs(output, exist_ok=True)
    output_geojson = build_output_geojson(ranked)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(output, f"risk_{timestamp}")

    print(f"\n  → 出力ファイル生成中...")
    if config.get("output", {}).get("geojson", True):
        save_geojson(output_geojson, f"{base}.geojson")

    if config.get("output", {}).get("csv", True):
        save_csv(ranked, f"{base}.csv")

    if config.get("output", {}).get("html_map", True):
        build_html_map(
            output_geojson,
            ranked,
            railway_geojson,
            f"{base}_map.html",
            railway_name=name,
        )

    # ─── サマリー表示 ───
    print_summary(ranked, name)

    return 0


if __name__ == "__main__":
    main()
