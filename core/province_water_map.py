from __future__ import annotations

import html
import json
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

import folium
import numpy as np
import pandas as pd
import requests

from core.maps import format_time
from core.water import STATUS_RANK

PH_CENTER = [12.7, 122.3]
MAP_ZOOM = 5.5

PROVINCE_LAYER_URLS = [
    (
        "DENR Provincial Boundary",
        "https://fmbfsd.denr.gov.ph/server/rest/services/Hosted/"
        "Provincial_Boundary/FeatureServer/0",
    ),
    (
        "GeoRiskPH PSA Provincial Boundary",
        "https://ulap-nga.georisk.gov.ph/arcgis/rest/services/PSA/"
        "Provincial/MapServer/0",
    ),
]


def _json_safe(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return [_json_safe(item) for item in list(value)]
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def normalize_province_key(value: Any) -> str:
    """Normalize station and government-boundary province names for matching."""
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.upper().replace("&", " AND ")
    text = re.sub(r"\bPROVINCE OF\b", " ", text)
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    aliases = {
        "NCR": "METRO MANILA",
        "NATIONAL CAPITAL REGION": "METRO MANILA",
        "METROPOLITAN MANILA": "METRO MANILA",
        "MANILA": "METRO MANILA",
        "NORTH COTABATO": "COTABATO",
        "COMPOSTELA VALLEY": "DAVAO DE ORO",
        "WESTERN SAMAR": "SAMAR",
        "DINAGAT ISLAND": "DINAGAT ISLANDS",
        "MINDORO OCCIDENTAL": "OCCIDENTAL MINDORO",
        "MINDORO ORIENTAL": "ORIENTAL MINDORO",
    }
    return aliases.get(text, text)


def _ring_area_centroid(ring: list) -> tuple[float, float, float] | None:
    points: list[tuple[float, float]] = []
    for pair in ring or []:
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            try:
                points.append((float(pair[0]), float(pair[1])))
            except (TypeError, ValueError):
                continue
    if len(points) < 3:
        return None
    if points[0] != points[-1]:
        points.append(points[0])

    cross_sum = 0.0
    cx_sum = 0.0
    cy_sum = 0.0
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        cross = x0 * y1 - x1 * y0
        cross_sum += cross
        cx_sum += (x0 + x1) * cross
        cy_sum += (y0 + y1) * cross

    if abs(cross_sum) < 1e-12:
        xs = [point[0] for point in points[:-1]]
        ys = [point[1] for point in points[:-1]]
        return 0.0, float(np.mean(xs)), float(np.mean(ys))

    area = cross_sum / 2.0
    centroid_x = cx_sum / (3.0 * cross_sum)
    centroid_y = cy_sum / (3.0 * cross_sum)
    return abs(area), centroid_x, centroid_y


def geometry_representative_point(geometry: dict | None) -> tuple[float, float] | None:
    """Return the centroid of the largest polygon part in a GeoJSON geometry."""
    if not isinstance(geometry, dict):
        return None
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates") or []
    outer_rings: list[list] = []
    if geometry_type == "Polygon":
        if coordinates:
            outer_rings.append(coordinates[0])
    elif geometry_type == "MultiPolygon":
        for polygon in coordinates:
            if polygon:
                outer_rings.append(polygon[0])

    candidates = [result for ring in outer_rings if (result := _ring_area_centroid(ring))]
    if not candidates:
        return None
    _, longitude, latitude = max(candidates, key=lambda item: item[0])
    return latitude, longitude


def fetch_province_reference(timeout_seconds: int = 45) -> tuple[dict, pd.DataFrame, str]:
    """Load province polygons and representative points from public government services."""
    errors: list[str] = []
    for source_name, layer_url in PROVINCE_LAYER_URLS:
        try:
            response = requests.get(
                f"{layer_url}/query",
                params={
                    "where": "1=1",
                    "outFields": "*",
                    "returnGeometry": "true",
                    "outSR": "4326",
                    "f": "geojson",
                },
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            province_geojson = response.json()
            if province_geojson.get("type") != "FeatureCollection":
                raise ValueError("service did not return a GeoJSON FeatureCollection")

            rows: list[dict[str, Any]] = []
            cleaned_features: list[dict] = []
            for feature in province_geojson.get("features", []):
                properties = feature.setdefault("properties", {})
                province = (
                    properties.get("province")
                    or properties.get("PROVINCE")
                    or properties.get("prov_name")
                    or properties.get("prov_name_s")
                    or properties.get("PROV_NAME")
                    or properties.get("province_")
                )
                region = (
                    properties.get("region")
                    or properties.get("REGION")
                    or properties.get("reg_name")
                    or properties.get("REG_NAME")
                    or ""
                )
                if province is None:
                    continue

                province_text = str(province).strip()
                region_text = str(region).strip()
                province_key = normalize_province_key(province_text)
                point = geometry_representative_point(feature.get("geometry"))
                if not province_key or point is None:
                    continue
                latitude, longitude = point
                properties.update(
                    {
                        "province": province_text,
                        "region": region_text,
                        "province_key": province_key,
                        "province_lat": latitude,
                        "province_lon": longitude,
                    }
                )
                rows.append(
                    {
                        "province_ref": province_text,
                        "province_key": province_key,
                        "region_ref": region_text,
                        "province_lat": latitude,
                        "province_lon": longitude,
                    }
                )
                cleaned_features.append(feature)

            centroids = pd.DataFrame(rows)
            if centroids.empty:
                raise ValueError("no province representative points were produced")
            centroids = centroids.drop_duplicates("province_key")
            province_geojson["features"] = cleaned_features

            maguindanao = centroids[
                centroids["province_key"].str.startswith("MAGUINDANAO", na=False)
            ]
            if not maguindanao.empty and "MAGUINDANAO" not in set(centroids["province_key"]):
                centroids = pd.concat(
                    [
                        centroids,
                        pd.DataFrame(
                            [
                                {
                                    "province_ref": "Maguindanao",
                                    "province_key": "MAGUINDANAO",
                                    "region_ref": "BARMM",
                                    "province_lat": maguindanao["province_lat"].mean(),
                                    "province_lon": maguindanao["province_lon"].mean(),
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
            return province_geojson, centroids, source_name
        except Exception as exc:
            errors.append(f"{source_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


def match_station_provinces(
    stations: pd.DataFrame,
    province_centroids: pd.DataFrame,
) -> pd.DataFrame:
    output = stations.copy()
    if output.empty:
        return output
    if "province" not in output.columns:
        output["province"] = ""
    output["province_key"] = output["province"].apply(normalize_province_key)
    reference_keys = province_centroids["province_key"].dropna().astype(str).tolist()
    reference_set = set(reference_keys)

    match_cache: dict[str, str | None] = {}
    for key in output["province_key"].dropna().unique():
        if not key:
            match_cache[key] = None
        elif key in reference_set:
            match_cache[key] = key
        else:
            candidates = [
                (SequenceMatcher(None, key, reference_key).ratio(), reference_key)
                for reference_key in reference_keys
            ]
            best_ratio, best_key = max(candidates, default=(0.0, ""))
            match_cache[key] = best_key if best_ratio >= 0.84 else None

    output["province_ref_key"] = output["province_key"].map(match_cache)
    return output.merge(
        province_centroids,
        left_on="province_ref_key",
        right_on="province_key",
        how="left",
        suffixes=("", "_reference"),
    )


def _trend_symbol(row: pd.Series) -> str:
    if bool(row.get("rapid_rise", False)):
        return "↑"
    if bool(row.get("rapid_fall", False)):
        return "↓"
    rate = row.get("rise_rate_m_hr")
    if pd.notna(rate) and float(rate) > 0.005:
        return "↑"
    if pd.notna(rate) and float(rate) < -0.005:
        return "↓"
    return "→"


def _province_map_summary(stations: pd.DataFrame) -> pd.DataFrame:
    if stations.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    valid = stations.dropna(subset=["province_lat", "province_lon"]).copy()
    for province_key, group in valid.groupby("province_ref_key", dropna=True):
        active = group[~group["water_status"].isin(["Stale", "Offline", "No Data"])]
        status_source = active if not active.empty else group
        ranks = status_source["water_status"].map(STATUS_RANK).fillna(-1)
        worst_status = status_source.loc[ranks.idxmax(), "water_status"] if not ranks.empty else "No Data"
        rates = pd.to_numeric(group["rise_rate_m_hr"], errors="coerce")
        dominant_rate = rates.loc[rates.abs().idxmax()] if rates.notna().any() else np.nan
        rows.append(
            {
                "province_ref_key": province_key,
                "province_ref": group["province_ref"].dropna().iloc[0],
                "province_lat": group["province_lat"].dropna().iloc[0],
                "province_lon": group["province_lon"].dropna().iloc[0],
                "station_count": int(len(group)),
                "active_count": int(len(active)),
                "worst_status": worst_status,
                "rapid_rise_count": int(group.get("rapid_rise", pd.Series(False, index=group.index)).fillna(False).sum()),
                "rapid_fall_count": int(group.get("rapid_fall", pd.Series(False, index=group.index)).fillna(False).sum()),
                "max_rise_m_hr": rates.max() if rates.notna().any() else np.nan,
                "max_fall_m_hr": rates.min() if rates.notna().any() else np.nan,
                "dominant_rate_m_hr": dominant_rate,
                "latest_timestamp": group["timestamp"].max(),
            }
        )
    return pd.DataFrame(rows)


def _station_popup_table(group: pd.DataFrame, max_rows: int = 30) -> str:
    display = group.copy()
    display["_active_sort"] = ~display["water_status"].isin(["Stale", "Offline", "No Data"])
    display["_rate_abs"] = pd.to_numeric(display["rise_rate_m_hr"], errors="coerce").abs()
    display = display.sort_values(
        ["_active_sort", "_rate_abs", "timestamp"],
        ascending=[False, False, False],
    )
    rows: list[str] = []
    for _, row in display.head(max_rows).iterrows():
        rate = row.get("rise_rate_m_hr")
        level = row.get("level_m")
        rate_text = f"{float(rate):+.3f}" if pd.notna(rate) else "-"
        level_text = f"{float(level):.3f}" if pd.notna(level) else "-"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('station_name', '-')))}</td>"
            f"<td>{html.escape(str(row.get('source_name', '-')))}</td>"
            f"<td>{level_text}</td>"
            f"<td>{rate_text}</td>"
            f"<td>{_trend_symbol(row)} {html.escape(str(row.get('trend_label', '-')))}</td>"
            f"<td>{html.escape(str(row.get('water_status', '-')))}</td>"
            f"<td>{html.escape(str(row.get('threshold_status', '-')))}</td>"
            f"<td>{html.escape(format_time(row.get('timestamp')))}</td>"
            "</tr>"
        )
    remaining = max(len(display) - min(len(display), max_rows), 0)
    more_text = (
        f"<div style='margin-top:6px;color:#555'>Plus {remaining} more station(s).</div>"
        if remaining
        else ""
    )
    return (
        "<div style='max-height:330px;overflow:auto'>"
        "<table style='border-collapse:collapse;width:100%;font-size:11px'>"
        "<thead><tr>"
        "<th style='text-align:left;padding:4px;border-bottom:1px solid #bbb'>Station</th>"
        "<th style='padding:4px;border-bottom:1px solid #bbb'>Source</th>"
        "<th style='padding:4px;border-bottom:1px solid #bbb'>Level (m)</th>"
        "<th style='padding:4px;border-bottom:1px solid #bbb'>Δ (m/hr)</th>"
        "<th style='padding:4px;border-bottom:1px solid #bbb'>Trend</th>"
        "<th style='padding:4px;border-bottom:1px solid #bbb'>Status</th>"
        "<th style='padding:4px;border-bottom:1px solid #bbb'>Threshold</th>"
        "<th style='padding:4px;border-bottom:1px solid #bbb'>Latest</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
        + more_text
    )


def _province_trend_category(row: pd.Series) -> str:
    rise_count = int(row.get("rapid_rise_count", 0) or 0)
    fall_count = int(row.get("rapid_fall_count", 0) or 0)
    rate = row.get("dominant_rate_m_hr")
    if rise_count and fall_count:
        return "Mixed rapid change"
    if rise_count:
        return "Rapid rise"
    if fall_count:
        return "Rapid fall"
    if pd.notna(rate) and float(rate) > 0.005:
        return "Rising"
    if pd.notna(rate) and float(rate) < -0.005:
        return "Falling"
    return "Stable"


def _province_trend_color(category: str) -> str:
    return {
        "Rapid rise": "#b91c1c",
        "Rising": "#f97316",
        "Stable": "#6b7280",
        "Falling": "#3b82f6",
        "Rapid fall": "#1d4ed8",
        "Mixed rapid change": "#7e22ce",
        "No data": "#d1d5db",
    }.get(category, "#d1d5db")


def _merge_province_metrics(province_geojson: dict, summary: pd.DataFrame) -> dict:
    merged = json.loads(json.dumps(_json_safe(province_geojson)))
    metric_map = summary.set_index("province_ref_key").to_dict(orient="index") if not summary.empty else {}
    for feature in merged.get("features", []):
        properties = feature.setdefault("properties", {})
        metrics = metric_map.get(normalize_province_key(properties.get("province")), {})
        properties.update(
            {
                "trend_category": metrics.get("trend_category", "No data"),
                "dominant_rate_m_hr": metrics.get("dominant_rate_m_hr"),
                "max_rise_m_hr": metrics.get("max_rise_m_hr"),
                "max_fall_m_hr": metrics.get("max_fall_m_hr"),
                "station_count": metrics.get("station_count", 0),
                "worst_status": metrics.get("worst_status", "No Data"),
            }
        )
    return json.loads(json.dumps(_json_safe(merged), allow_nan=False))


def build_province_water_map(
    stations: pd.DataFrame,
    province_geojson: dict | None,
    include_inactive: bool,
    only_rapid: bool = False,
) -> tuple[folium.Map, pd.DataFrame, pd.DataFrame]:
    """Build the restored province rise/fall map used by dashboard build 3.8."""
    map_object = folium.Map(location=PH_CENTER, zoom_start=MAP_ZOOM, tiles="cartodbpositron")
    required = ["province_ref_key", "province_lat", "province_lon"]
    if stations.empty or any(column not in stations.columns for column in required):
        folium.LayerControl(collapsed=False).add_to(map_object)
        return map_object, pd.DataFrame(), pd.DataFrame()

    mapped = stations.dropna(subset=required).copy()
    if not include_inactive:
        mapped = mapped[~mapped["water_status"].isin(["Stale", "Offline", "No Data"])]
    if only_rapid:
        rapid_rise = mapped.get("rapid_rise", pd.Series(False, index=mapped.index)).fillna(False)
        rapid_fall = mapped.get("rapid_fall", pd.Series(False, index=mapped.index)).fillna(False)
        mapped = mapped[rapid_rise | rapid_fall]

    summary = _province_map_summary(mapped)
    if not summary.empty:
        summary["trend_category"] = summary.apply(_province_trend_category, axis=1)

    if province_geojson:
        merged_provinces = _merge_province_metrics(province_geojson, summary)
        folium.GeoJson(
            merged_provinces,
            name="Province trend shading",
            style_function=lambda feature: {
                "fillColor": _province_trend_color(
                    feature.get("properties", {}).get("trend_category", "No data")
                ),
                "color": "#374151",
                "weight": 0.8,
                "fillOpacity": (
                    0.52
                    if feature.get("properties", {}).get("trend_category") != "No data"
                    else 0.05
                ),
            },
            highlight_function=lambda feature: {"weight": 2.4, "fillOpacity": 0.70},
            tooltip=folium.GeoJsonTooltip(
                fields=[
                    "province",
                    "region",
                    "trend_category",
                    "dominant_rate_m_hr",
                    "station_count",
                    "worst_status",
                ],
                aliases=[
                    "Province",
                    "Region",
                    "Water-level trend",
                    "Strongest change (m/hr)",
                    "Stations",
                    "Worst status",
                ],
                localize=True,
                sticky=False,
            ),
        ).add_to(map_object)

    marker_layer = folium.FeatureGroup(name="Visible water-level change labels", show=True)
    for _, summary_row in summary.iterrows():
        province_key = summary_row["province_ref_key"]
        group = mapped[mapped["province_ref_key"] == province_key].copy()
        category = summary_row["trend_category"]
        dominant_rate = summary_row["dominant_rate_m_hr"]
        symbol = {
            "Rapid rise": "↑↑",
            "Rising": "↑",
            "Stable": "→",
            "Falling": "↓",
            "Rapid fall": "↓↓",
            "Mixed rapid change": "↕",
        }.get(category, "•")
        marker_color = _province_trend_color(category)
        rate_label = f"{float(dominant_rate):+.3f}" if pd.notna(dominant_rate) else "-"
        marker_html = (
            "<div style='width:130px;text-align:center;white-space:nowrap;'>"
            f"<div style='display:inline-block;background:{marker_color};color:white;"
            "border:2px solid #111827;border-radius:18px;padding:5px 10px;"
            "font-weight:800;font-size:13px;box-shadow:0 2px 5px rgba(0,0,0,.45)'>"
            f"{symbol} {rate_label} m/hr"
            "</div>"
            f"<div style='margin-top:2px;font-size:10px;font-weight:700;color:#111827;"
            "background:rgba(255,255,255,.88);border-radius:7px;padding:1px 4px;"
            "display:inline-block'>"
            f"{html.escape(str(summary_row['province_ref']))} · {int(summary_row['station_count'])} gauge(s)"
            "</div></div>"
        )
        popup_header = (
            f"<div style='min-width:650px;max-width:800px'>"
            f"<h4 style='margin:0 0 6px'>{html.escape(str(summary_row['province_ref']))}</h4>"
            f"<b>Visible trend:</b> {html.escape(category)}<br>"
            f"<b>Strongest change:</b> {rate_label} m/hr<br>"
            f"<b>Worst current status:</b> {html.escape(str(summary_row['worst_status']))}<br>"
            f"<b>Stations shown:</b> {int(summary_row['station_count'])} "
            f"({int(summary_row['active_count'])} active)<br>"
            f"<b>Latest province reading:</b> {html.escape(format_time(summary_row['latest_timestamp']))}"
            "<hr style='margin:8px 0'>"
        )
        popup_html = popup_header + _station_popup_table(group) + "</div>"
        tooltip = (
            f"Province: {summary_row['province_ref']} | {category} | "
            f"{rate_label} m/hr | {int(summary_row['station_count'])} station(s)"
        )
        location = [float(summary_row["province_lat"]), float(summary_row["province_lon"])]
        folium.CircleMarker(
            location=location,
            radius=10,
            color="#111827",
            weight=2,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.95,
            tooltip=tooltip,
            popup=folium.Popup(popup_html, max_width=860),
        ).add_to(marker_layer)
        folium.Marker(
            location=location,
            icon=folium.DivIcon(
                html=marker_html,
                icon_size=(130, 48),
                icon_anchor=(65, 24),
                class_name="water-trend-label",
            ),
            tooltip=tooltip,
            popup=folium.Popup(popup_html, max_width=860),
        ).add_to(marker_layer)

    marker_layer.add_to(map_object)
    folium.LayerControl(collapsed=False).add_to(map_object)
    return map_object, mapped, summary
