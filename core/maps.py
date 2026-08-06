from __future__ import annotations

import html
import json
from typing import Any

import folium
import numpy as np
import pandas as pd

PH_CENTER = [12.7, 122.3]
MAP_ZOOM = 5.5

RAIN_COLORS = {
    "Low": "#2ecc71",
    "Moderate": "#f1c40f",
    "High": "#e67e22",
    "Severe": "#e74c3c",
    "Extreme": "#8e0000",
    "No Data": "#9aa0a6",
}
WATER_COLORS = {
    "Normal": "#2ecc71",
    "Alert": "#f1c40f",
    "Alarm": "#e67e22",
    "Critical": "#e74c3c",
    "Stale": "#9aa0a6",
    "Offline": "#333333",
    "No Data": "#c4c7c5",
    "No Threshold": "#64748b",
}


def _safe(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [_safe(item) for item in list(value)]
    try:
        if bool(pd.isna(value)):
            return None
    except Exception:
        pass
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def merge_geojson_metrics(geojson: dict, metrics: pd.DataFrame) -> dict:
    metric_map = metrics.set_index("basin_name").to_dict(orient="index") if not metrics.empty else {}
    merged = json.loads(json.dumps(_safe(geojson)))
    for feature in merged.get("features", []):
        basin = feature.get("properties", {}).get("basin_name")
        feature.setdefault("properties", {}).update(_safe(metric_map.get(basin, {})))
    return json.loads(json.dumps(_safe(merged), allow_nan=False))


def format_time(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("Asia/Manila").strftime("%Y-%m-%d %I:%M %p")


def _station_popup(row: pd.Series) -> str:
    rate = row.get("rise_rate_m_hr")
    rate_text = f"{float(rate):+.3f} m/hr" if pd.notna(rate) else "-"
    thresholds = [row.get("alert_m"), row.get("alarm_m"), row.get("critical_m")]
    threshold_text = " / ".join(f"{float(item):.2f}" for item in thresholds) + " m" if all(pd.notna(item) for item in thresholds) else "Not available"
    source_url = str(row.get("source_url", "") or "")
    source_line = f"<a href='{html.escape(source_url)}' target='_blank'>Open source</a>" if source_url.startswith("http") else "-"
    return (
        f"<div style='min-width:310px'>"
        f"<b>{html.escape(str(row.get('station_name', '-')))}</b><br>"
        f"River: {html.escape(str(row.get('river_system', '-') or '-'))}<br>"
        f"Basin: {html.escape(str(row.get('basin_name', '-') or '-'))}<br>"
        f"Province/LGU: {html.escape(str(row.get('province', '-') or '-'))} / {html.escape(str(row.get('municipality', '-') or '-'))}<br>"
        f"Level: <b>{float(row.get('level_m')):.3f} m</b><br>"
        f"Trend: {html.escape(str(row.get('trend_label', row.get('source_trend', '-'))))} ({rate_text})<br>"
        f"Status: <b>{html.escape(str(row.get('water_status', row.get('threshold_status', '-'))))}</b><br>"
        f"Alert / Alarm / Critical: {threshold_text}<br>"
        f"Observed: {format_time(row.get('timestamp'))}<br>"
        f"Source: {html.escape(str(row.get('source_name', '-')))} · {source_line}<br>"
        f"Data type: {html.escape(str(row.get('data_kind', '-')))}<br>"
        f"Notes: {html.escape(str(row.get('notes', '') or ''))}"
        f"</div>"
    )


def build_monitoring_map(geojson: dict, hazard_df: pd.DataFrame, stations: pd.DataFrame, view: str = "Combined") -> folium.Map:
    map_object = folium.Map(location=PH_CENTER, zoom_start=MAP_ZOOM, tiles="cartodbpositron")
    merged = merge_geojson_metrics(geojson, hazard_df)
    if view == "Rainfall":
        property_name = "hazard_level"
        palette = RAIN_COLORS
    elif view == "Water":
        property_name = "water_status"
        palette = WATER_COLORS
    else:
        property_name = "combined_level"
        palette = RAIN_COLORS

    def style(feature: dict) -> dict:
        level = feature.get("properties", {}).get(property_name, "No Data")
        return {
            "fillColor": palette.get(level, "#9aa0a6"),
            "color": "#333333",
            "weight": 1,
            "fillOpacity": 0.62,
        }

    tooltip_fields = ["basin_name", "hazard_level", "water_status", "combined_level", "effective_rain_mm", "station_count"]
    tooltip_aliases = ["Basin", "Rainfall hazard", "Water status", "Combined level", "Effective rain (mm)", "Mapped stations"]
    folium.GeoJson(
        merged,
        name="River-basin screening",
        style_function=style,
        highlight_function=lambda feature: {"weight": 3, "fillOpacity": 0.8},
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases, localize=True, sticky=False),
    ).add_to(map_object)

    if stations is not None and not stations.empty:
        marker_layer = folium.FeatureGroup(name="Verified-coordinate water stations", show=True)
        mapped = stations.dropna(subset=["lat", "lon"]).copy()
        for _, row in mapped.iterrows():
            status = str(row.get("water_status", "No Data"))
            radius = 8 if bool(row.get("rapid_rise", False) or row.get("rapid_fall", False)) else 6
            folium.CircleMarker(
                location=[float(row["lat"]), float(row["lon"])],
                radius=radius,
                color="#111827",
                weight=2 if radius == 8 else 1,
                fill=True,
                fill_color=WATER_COLORS.get(status, "#64748b"),
                fill_opacity=0.95,
                popup=folium.Popup(_station_popup(row), max_width=430),
                tooltip=f"{row.get('station_name')}: {float(row.get('level_m')):.2f} m ({status})",
            ).add_to(marker_layer)
        marker_layer.add_to(map_object)
    folium.LayerControl(collapsed=False).add_to(map_object)
    return map_object
