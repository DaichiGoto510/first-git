"""
リスク評価結果の可視化・出力モジュール

出力形式:
- GeoJSON: リスクスコア付き鉄道セグメント (GIS ソフトで読み込み可)
- CSV: 点検優先順位リスト (Excel で開ける)
- HTML: インタラクティブ地図 (ブラウザで確認可)
"""

import json
import csv
import os
from datetime import datetime
from typing import List, Dict


# 優先度別カラーマッピング
PRIORITY_COLORS = {
    "critical": "#d32f2f",   # 赤
    "high": "#f57c00",       # 橙
    "medium": "#fbc02d",     # 黄
    "low": "#388e3c",        # 緑
}

PRIORITY_LABELS_JP = {
    "critical": "緊急点検",
    "high": "高優先度",
    "medium": "中優先度",
    "low": "低優先度",
}


def save_geojson(segments_geojson: dict, output_path: str):
    """リスクスコア付き GeoJSON を保存する"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments_geojson, f, ensure_ascii=False, indent=2)
    print(f"  ✓ GeoJSON 保存: {output_path}")


def save_csv(results: List[Dict], output_path: str):
    """点検優先順位 CSV を保存する"""
    if not results:
        return

    fieldnames = [
        "inspection_rank",
        "segment_id",
        "priority_label",
        "risk_score",
        "vegetation_health_score",
        "wind_exposure_score",
        "slope_stability_score",
        "canopy_height_score",
        "storm_history_score",
        "ndvi",
        "slope_deg",
        "elevation_m",
        "canopy_height_p90_m",
        "soil_moisture_proxy",
        "center_lon",
        "center_lat",
    ]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:  # BOM付きUTF-8 (Excel対応)
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = {
                "inspection_rank": r.get("inspection_rank"),
                "segment_id": r.get("segment_id"),
                "priority_label": r.get("priority_label"),
                "risk_score": r.get("risk_score"),
                "vegetation_health_score": r.get("vegetation_health_score"),
                "wind_exposure_score": r.get("wind_exposure_score"),
                "slope_stability_score": r.get("slope_stability_score"),
                "canopy_height_score": r.get("canopy_height_score"),
                "storm_history_score": r.get("storm_history_score"),
                "ndvi": r.get("raw_indices", {}).get("ndvi"),
                "slope_deg": r.get("raw_indices", {}).get("slope_deg"),
                "elevation_m": r.get("raw_indices", {}).get("elevation_m"),
                "canopy_height_p90_m": r.get("raw_indices", {}).get("canopy_height_p90_m"),
                "soil_moisture_proxy": r.get("raw_indices", {}).get("soil_moisture_proxy"),
                "center_lon": r.get("center_lon"),
                "center_lat": r.get("center_lat"),
            }
            writer.writerow(row)
    print(f"  ✓ CSV 保存: {output_path}")


def build_html_map(
    segments_geojson: dict,
    results: List[Dict],
    railway_geojson: dict,
    output_path: str,
    railway_name: str = "対象路線",
):
    """
    Folium を使ったインタラクティブHTMLマップを生成する

    地図要素:
    - 鉄道路線 (青線)
    - リスクセグメント (色分けライン)
    - 各セグメントのポップアップ (詳細情報)
    - リスクスコアのカラーバー凡例
    """
    try:
        import folium
        from folium.plugins import FloatImage
    except ImportError:
        print("  警告: folium未インストール。HTML地図をスキップします。")
        print("  インストール: pip install folium")
        return

    # 地図の中心座標を計算
    coords = []
    for feature in segments_geojson.get("features", []):
        geom = feature.get("geometry", {})
        if geom.get("type") == "LineString":
            coords.extend(geom.get("coordinates", []))
        elif geom.get("type") == "MultiLineString":
            for line in geom.get("coordinates", []):
                coords.extend(line)

    if not coords:
        print("  警告: 座標データなし。HTML地図をスキップします。")
        return

    center_lon = sum(c[0] for c in coords) / len(coords)
    center_lat = sum(c[1] for c in coords) / len(coords)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="CartoDB positron",
    )

    # 追加タイルオプション
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(
        tiles="https://cyberjapandata.gsi.go.jp/xyz/std/{z}/{x}/{y}.png",
        attr="© 国土地理院",
        name="国土地理院標準地図",
    ).add_to(m)

    # 元の鉄道路線 (青色)
    folium.GeoJson(
        railway_geojson,
        name="鉄道路線",
        style_function=lambda x: {
            "color": "#1565c0",
            "weight": 3,
            "opacity": 0.7,
        },
    ).add_to(m)

    # リスクスコア別セグメント
    for feature in segments_geojson.get("features", []):
        props = feature.get("properties", {})
        priority = props.get("priority", "low")
        color = PRIORITY_COLORS.get(priority, "#388e3c")
        risk_score = props.get("risk_score", 0)
        segment_id = props.get("segment_id", "")

        # ポップアップHTML
        popup_html = f"""
        <div style="font-family: sans-serif; min-width: 250px;">
          <h4 style="margin:0; color:{color}">
            {props.get('priority_label', '')} (#{props.get('inspection_rank', '-')})
          </h4>
          <p style="font-size:1.3em; margin:4px 0;">
            <b>総合リスクスコア: {risk_score}</b>/100
          </p>
          <hr style="margin:4px 0">
          <table style="width:100%; font-size:0.9em;">
            <tr><td>植生健全性</td><td align="right"><b>{props.get('vegetation_health_score', 0)}</b></td></tr>
            <tr><td>風曝露</td><td align="right"><b>{props.get('wind_exposure_score', 0)}</b></td></tr>
            <tr><td>斜面安定性</td><td align="right"><b>{props.get('slope_stability_score', 0)}</b></td></tr>
            <tr><td>樹高リスク</td><td align="right"><b>{props.get('canopy_height_score', 0)}</b></td></tr>
            <tr><td>暴風歴</td><td align="right"><b>{props.get('storm_history_score', 0)}</b></td></tr>
          </table>
          <hr style="margin:4px 0">
          <table style="width:100%; font-size:0.85em; color:#555">
            <tr><td>NDVI</td><td align="right">{props.get('ndvi', '-')}</td></tr>
            <tr><td>傾斜角</td><td align="right">{props.get('slope_deg', '-')}°</td></tr>
            <tr><td>標高</td><td align="right">{props.get('elevation_m', '-')} m</td></tr>
            <tr><td>樹高 P90</td><td align="right">{props.get('canopy_height_p90_m', '-')} m</td></tr>
          </table>
          <p style="font-size:0.8em; color:#888; margin:4px 0">
            セグメントID: {segment_id}
          </p>
        </div>
        """

        # セグメントラインの太さをリスクスコアに比例
        weight = 3 + risk_score / 20

        folium.GeoJson(
            feature,
            name=f"seg_{segment_id}",
            style_function=lambda x, c=color, w=weight: {
                "color": c,
                "weight": w,
                "opacity": 0.85,
            },
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=folium.Tooltip(
                f"リスク: {risk_score} ({props.get('priority_label', '')})",
                sticky=True,
            ),
        ).add_to(m)

    # 凡例
    legend_html = """
    <div style="
        position: fixed;
        bottom: 50px; right: 20px; z-index: 1000;
        background: white; padding: 12px 16px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        font-family: sans-serif; font-size: 13px;
    ">
        <b>倒木リスク 点検優先度</b><br>
        <span style="color:#d32f2f">━━</span> 緊急点検 (スコア 75+)<br>
        <span style="color:#f57c00">━━</span> 高優先度 (55〜75)<br>
        <span style="color:#fbc02d">━━</span> 中優先度 (35〜55)<br>
        <span style="color:#388e3c">━━</span> 低優先度 (0〜35)<br>
        <span style="color:#1565c0">━━</span> 鉄道路線
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # タイトル
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    title_html = f"""
    <div style="
        position: fixed;
        top: 10px; left: 50%; transform: translateX(-50%);
        z-index: 1000;
        background: rgba(255,255,255,0.9); padding: 8px 16px;
        border-radius: 6px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.2);
        font-family: sans-serif;
    ">
        <b>鉄道脇 倒木リスクマップ</b> — {railway_name}
        <span style="font-size:0.85em; color:#888; margin-left:12px">
            解析日時: {now_str}
        </span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    folium.LayerControl().add_to(m)

    m.save(output_path)
    print(f"  ✓ HTML地図 保存: {output_path}")
    print(f"  → ブラウザで開く: file://{os.path.abspath(output_path)}")


def print_summary(results: List[Dict], railway_name: str = "対象路線"):
    """コンソールにサマリーを表示する"""
    if not results:
        print("結果なし")
        return

    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for r in results:
        counts[r.get("priority", "low")] += 1

    total = len(results)
    print("\n" + "=" * 60)
    print(f"  倒木リスク評価 サマリー: {railway_name}")
    print("=" * 60)
    print(f"  解析セグメント数: {total}")
    print(f"  🔴 緊急点検   : {counts['critical']} セグメント")
    print(f"  🟠 高優先度   : {counts['high']} セグメント")
    print(f"  🟡 中優先度   : {counts['medium']} セグメント")
    print(f"  🟢 低優先度   : {counts['low']} セグメント")
    print()
    print("  ─── 上位10セグメント (点検優先順) ───")
    print(f"  {'順位':>4}  {'セグメントID':<15}  {'スコア':>5}  優先度")
    print("  " + "─" * 45)
    for r in results[:10]:
        label = PRIORITY_LABELS_JP.get(r.get("priority", "low"), "")
        print(
            f"  {r.get('inspection_rank', '-'):>4}  "
            f"{str(r.get('segment_id', '-')):<15}  "
            f"{r.get('risk_score', 0):>5.1f}  {label}"
        )
    print("=" * 60)
