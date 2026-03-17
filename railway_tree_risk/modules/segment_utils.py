"""
鉄道路線のセグメント分割・バッファ処理ユーティリティ
"""

import json
import math
from typing import List, Dict, Tuple


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """2点間のハバーサイン距離を計算 (メートル)"""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def interpolate_line(coords: List[Tuple], segment_length_m: float) -> List[List[Tuple]]:
    """
    ラインストリングを指定距離のセグメントに分割する

    Args:
        coords: [(lon, lat), ...] の座標リスト
        segment_length_m: セグメント長さ (メートル)

    Returns:
        セグメントごとの座標リスト
    """
    if len(coords) < 2:
        return [coords]

    segments = []
    current_segment = [coords[0]]
    accumulated = 0.0

    for i in range(1, len(coords)):
        prev = coords[i - 1]
        curr = coords[i]
        dist = haversine_distance(prev[0], prev[1], curr[0], curr[1])

        if accumulated + dist >= segment_length_m:
            # セグメント境界点を補間
            remaining = segment_length_m - accumulated
            ratio = remaining / dist if dist > 0 else 0
            mid_lon = prev[0] + ratio * (curr[0] - prev[0])
            mid_lat = prev[1] + ratio * (curr[1] - prev[1])
            mid = (mid_lon, mid_lat)
            current_segment.append(mid)
            segments.append(current_segment)
            current_segment = [mid, curr]
            accumulated = dist - remaining
        else:
            current_segment.append(curr)
            accumulated += dist

    if len(current_segment) >= 2:
        segments.append(current_segment)

    return segments


def extract_all_coords(geojson: dict) -> List[Tuple]:
    """GeoJSON から全座標を抽出する"""
    coords = []
    features = geojson.get("features", [geojson] if "geometry" in geojson else [])

    for feature in features:
        geom = feature.get("geometry", feature)
        geom_type = geom.get("type", "")

        if geom_type == "LineString":
            coords.extend(tuple(c) for c in geom["coordinates"])
        elif geom_type == "MultiLineString":
            for line in geom["coordinates"]:
                coords.extend(tuple(c) for c in line)
        elif geom_type == "GeometryCollection":
            for g in geom.get("geometries", []):
                if g["type"] == "LineString":
                    coords.extend(tuple(c) for c in g["coordinates"])

    return coords


def approximate_buffer_bbox(
    coords: List[Tuple], buffer_m: float
) -> Dict:
    """
    セグメント座標リストから近似バッファ bbox を GeoJSON Polygon として返す

    注意: 正確な地理バッファはGEE側で処理する。
    ここでは GEE に渡す ROI 用の簡易 bbox を作成。
    """
    if not coords:
        return {}

    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]

    # 1度 ≈ 111km なので buffer_m をdeg変換
    deg_buf = buffer_m / 111000.0

    min_lon = min(lons) - deg_buf
    max_lon = max(lons) + deg_buf
    min_lat = min(lats) - deg_buf
    max_lat = max(lats) + deg_buf

    return {
        "type": "Polygon",
        "coordinates": [[
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat],
        ]],
    }


def segment_center(coords: List[Tuple]) -> Tuple[float, float]:
    """セグメントの中心座標を返す"""
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return sum(lons) / len(lons), sum(lats) / len(lats)


def split_railway_to_segments(
    railway_geojson: dict,
    segment_length_m: float,
    buffer_m: float,
) -> List[Dict]:
    """
    鉄道路線 GeoJSON を解析セグメントに分割する

    Returns:
        各セグメントの辞書リスト:
          - segment_id: セグメント番号
          - coords: セグメント座標
          - center_lon, center_lat: 中心座標
          - roi_geojson: GEE用バッファ bbox
          - line_geojson: セグメントライン (GeoJSON)
    """
    all_coords = extract_all_coords(railway_geojson)
    if not all_coords:
        raise ValueError("GeoJSONから座標を抽出できませんでした。LineStringまたはMultiLineStringを確認してください。")

    raw_segments = interpolate_line(all_coords, segment_length_m)

    result = []
    for i, seg_coords in enumerate(raw_segments):
        if len(seg_coords) < 2:
            continue
        center_lon, center_lat = segment_center(seg_coords)
        roi = approximate_buffer_bbox(seg_coords, buffer_m)

        result.append({
            "segment_id": f"seg_{i+1:04d}",
            "coords": seg_coords,
            "center_lon": round(center_lon, 6),
            "center_lat": round(center_lat, 6),
            "roi_geojson": roi,
            "line_geojson": {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [list(c) for c in seg_coords],
                },
                "properties": {"segment_id": f"seg_{i+1:04d}"},
            },
        })

    return result


def build_output_geojson(results: List[Dict]) -> Dict:
    """リスクスコア付きの GeoJSON FeatureCollection を構築する"""
    features = []
    for r in results:
        props = {
            "segment_id": r.get("segment_id"),
            "inspection_rank": r.get("inspection_rank"),
            "risk_score": r.get("risk_score"),
            "priority": r.get("priority"),
            "priority_label": r.get("priority_label"),
            "vegetation_health_score": r.get("vegetation_health_score"),
            "wind_exposure_score": r.get("wind_exposure_score"),
            "slope_stability_score": r.get("slope_stability_score"),
            "canopy_height_score": r.get("canopy_height_score"),
            "storm_history_score": r.get("storm_history_score"),
            "center_lon": r.get("center_lon"),
            "center_lat": r.get("center_lat"),
        }
        # 生指標値もプロパティに追加
        raw = r.get("raw_indices", {})
        props["ndvi"] = raw.get("ndvi")
        props["slope_deg"] = raw.get("slope_deg")
        props["elevation_m"] = raw.get("elevation_m")
        props["canopy_height_p90_m"] = raw.get("canopy_height_p90_m")
        props["soil_moisture_proxy"] = raw.get("soil_moisture_proxy")

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [list(c) for c in r.get("coords", [])],
            },
            "properties": props,
        }
        features.append(feature)

    return {"type": "FeatureCollection", "features": features}
