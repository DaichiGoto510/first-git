#!/usr/bin/env python3
"""
倒木リスクモデル 精度検証ツール (イベントベース・プロキシ検証)

使用方法:
    python validate.py --railway data/sample_railway.geojson \\
                       --event-date 2019-10-12 \\
                       --event-name "台風Hagibis" \\
                       --output output/validation/

検証の仕組み:
    1. イベント前 (pre-window) の衛星データでリスクスコアを計算
    2. イベント後 (post-window) の NDVI 降下量を「疑似正解」として取得
    3. 相関・ROC/AUC・Precision/Recall を計算し HTML レポートを出力

推奨イベント例:
    --event-date 2019-10-12  --event-name "台風Hagibis" (関東・東北)
    --event-date 2018-09-04  --event-name "台風Jebi"    (近畿・東海)
    --event-date 2019-09-09  --event-name "台風Faxai"   (関東)

引数:
    --railway         : 鉄道路線 GeoJSON ファイルパス
    --event-date      : 台風通過日 (YYYY-MM-DD)
    --event-name      : イベント名 (ラベル用)
    --output          : 出力ディレクトリ
    --config          : 設定ファイル (デフォルト: config.yaml)
    --project         : GEE プロジェクトID
    --damage-threshold: NDVI 降下の被害判定閾値 (デフォルト: 0.05)
    --pre-days        : イベント前データ取得期間・日数 (デフォルト: 60)
    --post-days       : イベント後データ取得期間・日数 (デフォルト: 60)
    --dry-run         : GEE に接続せずダミーデータで動作確認

出力:
    validation_YYYYMMDD_HHMMSS_scatter.png : 散布図
    validation_YYYYMMDD_HHMMSS_roc.png     : ROC 曲線
    validation_YYYYMMDD_HHMMSS_report.html : 検証レポート
    validation_YYYYMMDD_HHMMSS.json        : 指標 JSON
"""

import os
import sys
import json
import time
import random
import click
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.segment_utils import split_railway_to_segments
from modules.risk_calculator import compute_composite_risk_score
from modules.validator import (
    compute_validation_metrics,
    plot_validation_figures,
    generate_html_report,
    fetch_ndvi_for_period,
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_geojson(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dummy_indices(seg_id: str) -> dict:
    """dry-run 用ダミー衛星指標 (risk_estimator.py の generate_dummy_indices と同じ)"""
    rng = random.Random(hash(seg_id))
    ndvi = rng.uniform(0.15, 0.85)
    nbr = rng.uniform(0.1, 0.65)
    ndre = rng.uniform(0.1, 0.55)
    ndvi_trend = rng.uniform(-0.12, 0.05)
    slope = rng.uniform(0, 40)
    elevation = rng.uniform(10, 1200)
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
        "ndvi_mean": ndvi, "nbr_mean": nbr, "ndre_mean": ndre,
        "ndvi_trend": ndvi_trend, "vv_backscatter_db": vv_db,
        "vh_backscatter_db": vv_db - 7, "rvi": rng.uniform(0.2, 0.8),
        "soil_moisture_proxy": soil_moisture, "slope_deg": slope,
        "aspect_deg": rng.uniform(0, 360), "twi": twi, "tpi": tpi,
        "wind_exposure": wind_exposure, "elevation_m": elevation,
        "canopy_height_mean_m": canopy_mean, "canopy_height_p90_m": canopy_p90,
        "height_risk_score": height_risk,
    }


def _dummy_ndvi_drop(risk_score: float, seg_id: str) -> float:
    """
    dry-run 用: リスクスコアに相関したダミー NDVI 降下量を生成する

    実際のデータでは Spearman ρ ≈ 0.4〜0.6 が期待される。
    ダミーでは ρ ≈ 0.55 になるよう相関ノイズを設定。
    """
    rng = random.Random(hash(seg_id) + 9999)
    risk_norm = risk_score / 100.0
    expected = risk_norm * 0.12          # 高リスク区間は最大 12% の NDVI 降下
    noise = rng.gauss(0, 0.028)         # ノイズ (標準偏差 2.8%)
    return max(-0.06, min(0.20, expected + noise))


@click.command()
@click.option("--railway", required=True, type=click.Path(exists=True),
              help="鉄道路線 GeoJSON ファイルパス")
@click.option("--event-date", required=True,
              help="台風通過日 (YYYY-MM-DD 形式)")
@click.option("--event-name", default="台風イベント", show_default=True,
              help="イベント名 (レポートタイトル用)")
@click.option("--output", default="output/validation", show_default=True,
              help="出力ディレクトリ")
@click.option("--config", "config_path", default=None,
              help="設定ファイルパス (デフォルト: config.yaml)")
@click.option("--project", default=None,
              help="Google Earth Engine プロジェクトID")
@click.option("--damage-threshold", default=0.05, show_default=True, type=float,
              help="NDVI 降下の被害判定閾値 (Δ NDVI)")
@click.option("--pre-days", default=60, show_default=True, type=int,
              help="イベント前データ取得期間 (日数)")
@click.option("--post-days", default=60, show_default=True, type=int,
              help="イベント後データ取得期間 (日数)")
@click.option("--dry-run", is_flag=True, default=False,
              help="GEE に接続せずダミーデータで動作確認")
def main(
    railway, event_date, event_name, output, config_path,
    project, damage_threshold, pre_days, post_days, dry_run,
):
    """
    過去の台風イベントを使ってリスクモデルの精度をプロキシ検証します。

    正解データ (現地記録) がなくても、台風前後の NDVI 変化を
    「疑似正解」として利用することで、相関・AUC を計算できます。

    GEE 認証が必要です (--dry-run で認証なし動作確認可):
      earthengine authenticate
    """
    from datetime import datetime, timedelta

    # ─── 設定読み込み ────────────────────────────────────────
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)

    try:
        event_dt = datetime.strptime(event_date, "%Y-%m-%d")
    except ValueError:
        click.echo(f"エラー: --event-date の形式が不正です。YYYY-MM-DD で指定してください。")
        sys.exit(1)

    # イベント前後の日付ウィンドウ
    pre_start = (event_dt - timedelta(days=pre_days)).strftime("%Y-%m-%d")
    pre_end   = (event_dt - timedelta(days=7)).strftime("%Y-%m-%d")
    post_start = (event_dt + timedelta(days=7)).strftime("%Y-%m-%d")
    post_end   = (event_dt + timedelta(days=post_days)).strftime("%Y-%m-%d")

    print("\n" + "=" * 62)
    print("  🔍  倒木リスクモデル 精度検証")
    print("=" * 62)
    print(f"  イベント    : {event_name} ({event_date})")
    print(f"  前ウィンドウ: {pre_start} 〜 {pre_end}")
    print(f"  後ウィンドウ: {post_start} 〜 {post_end}")
    print(f"  被害閾値    : Δ NDVI ≥ {damage_threshold}")
    print(f"  入力ファイル: {railway}")
    print(f"  出力先      : {output}")
    if dry_run:
        print("  モード      : DRY RUN (ダミーデータ)")
    print()

    # ─── GEE 初期化 ──────────────────────────────────────────
    if not dry_run:
        from modules.gee_processing import initialize_gee
        initialize_gee(project)
    else:
        print("  [DRY RUN] GEE 接続をスキップ\n")

    # ─── 路線読み込み & セグメント分割 ───────────────────────
    print("  → 鉄道路線データ読み込み中...")
    railway_geojson = load_geojson(railway)
    segments = split_railway_to_segments(
        railway_geojson,
        segment_length_m=config["segment_length_m"],
        buffer_m=config["buffer_distance_m"],
    )
    print(f"  → {len(segments)} セグメントに分割\n")

    # ─── セグメントごとの処理 ────────────────────────────────
    weights = config.get("risk_weights", {})
    cloud_thresh = config.get("cloud_cover_threshold", 20)
    segment_results = []

    for i, seg in enumerate(segments):
        seg_id = seg["segment_id"]
        roi = seg["roi_geojson"]
        print(f"  [{i+1}/{len(segments)}] {seg_id}", end="", flush=True)

        try:
            # ── イベント前: リスクスコア計算 ─────────────────
            if dry_run:
                indices = _dummy_indices(seg_id)
            else:
                from modules.gee_processing import (
                    geojson_to_ee_geometry,
                    get_sentinel2_indices,
                    get_sentinel1_soil_moisture,
                    get_terrain_metrics,
                    get_canopy_height,
                )
                import ee
                roi_ee = geojson_to_ee_geometry(roi)
                s2 = get_sentinel2_indices(roi_ee, pre_start, pre_end, cloud_thresh)
                s1 = get_sentinel1_soil_moisture(roi_ee, pre_start, pre_end)
                terrain = get_terrain_metrics(roi_ee)
                canopy = get_canopy_height(roi_ee)
                indices = {**s2, **s1, **terrain, **canopy}

            risk_result = compute_composite_risk_score(indices, weights)
            risk_score = risk_result["risk_score"]

            # ── イベント後: NDVI 変化量 ───────────────────────
            if dry_run:
                pre_ndvi = float(indices.get("ndvi_mean", 0.5))
                ndvi_drop = _dummy_ndvi_drop(risk_score, seg_id)
                post_ndvi = pre_ndvi - ndvi_drop
            else:
                pre_ndvi = float(indices.get("ndvi_mean") or 0.5)
                post_ndvi = fetch_ndvi_for_period(roi, post_start, post_end, cloud_thresh)
                if post_ndvi is None:
                    print(f" [後画像なし: スキップ]")
                    continue
                ndvi_drop = pre_ndvi - post_ndvi

            print(
                f" | リスク={risk_score:5.1f}"
                f" | pre NDVI={pre_ndvi:.3f}"
                f" | post NDVI={post_ndvi:.3f}"
                f" | Δ={ndvi_drop:+.3f}"
                f" {'🔴 被害' if ndvi_drop >= damage_threshold else '🟢'}"
            )

            segment_results.append({
                "segment_id": seg_id,
                "risk_score": risk_score,
                "pre_ndvi": round(pre_ndvi, 4),
                "post_ndvi": round(post_ndvi, 4),
                "ndvi_drop": round(ndvi_drop, 4),
                "center_lon": seg["center_lon"],
                "center_lat": seg["center_lat"],
            })

        except Exception as e:
            print(f" [エラー: {e}]")

        if not dry_run:
            time.sleep(0.5)

    # ─── 精度指標の計算 ──────────────────────────────────────
    if len(segment_results) < 5:
        print(f"\n  エラー: 有効セグメント数が少なすぎます ({len(segment_results)} 件)。")
        sys.exit(1)

    risk_scores = [s["risk_score"] for s in segment_results]
    ndvi_drops  = [s["ndvi_drop"]  for s in segment_results]

    print(f"\n  → 精度指標計算中...")
    metrics = compute_validation_metrics(risk_scores, ndvi_drops, damage_threshold)

    print(f"\n{'─' * 44}")
    print(f"  Spearman ρ   : {metrics['spearman_rho']:+.3f}")
    print(f"  Pearson  r   : {metrics['pearson_r']:+.3f}")
    print(f"  AUC          : {metrics['auc']:.3f}")
    print(f"  Precision    : {metrics['precision']:.3f}")
    print(f"  Recall       : {metrics['recall']:.3f}")
    print(f"  F1 スコア    : {metrics['f1_score']:.3f}")
    print(f"  最適閾値     : {metrics['optimal_threshold']:.0f} / 100")
    print(f"  被害率       : {metrics['damage_rate_pct']:.1f}% ({metrics['n_damaged']}/{metrics['n_segments']})")
    print(f"{'─' * 44}\n")

    # ─── 出力 ────────────────────────────────────────────────
    os.makedirs(output, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"validation_{timestamp}"

    print("  → グラフ生成中...")
    scatter_path, roc_path = plot_validation_figures(
        risk_scores, ndvi_drops, metrics, event_name, output, base_name
    )

    print("  → HTML レポート生成中...")
    html_path = os.path.join(output, f"{base_name}_report.html")
    generate_html_report(
        segment_results, metrics, event_name, event_date,
        scatter_path, roc_path, html_path,
    )

    # JSON 保存
    json_path = os.path.join(output, f"{base_name}.json")
    export_metrics = {k: v for k, v in metrics.items()
                      if k not in ("fpr_list", "tpr_list")}
    payload = {
        "event_name": event_name,
        "event_date": event_date,
        "pre_window": {"start": pre_start, "end": pre_end},
        "post_window": {"start": post_start, "end": post_end},
        "damage_threshold": damage_threshold,
        "metrics": export_metrics,
        "segments": segment_results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  → JSON データ: {json_path}")

    if scatter_path:
        print(f"  → 散布図     : {scatter_path}")
    if roc_path:
        print(f"  → ROC 曲線   : {roc_path}")

    print(f"\n  ✅ 検証完了\n")


if __name__ == "__main__":
    main()
