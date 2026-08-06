from __future__ import annotations

import html
import re
from collections import OrderedDict
from typing import Any

import folium
import numpy as np
import pandas as pd

from core.maps import WATER_COLORS, format_time

# These are representative display anchors for river-system summaries. They are
# deliberately not presented as exact gauge coordinates. Exact station markers
# are added separately whenever verified latitude/longitude values are available.
TARGETS = OrderedDict(
    [
        (
            "marikina",
            {
                "label": "Marikina River",
                "anchor": (14.6507, 121.1029),
            },
        ),
        (
            "tullahan",
            {
                "label": "Tullahan River",
                "anchor": (14.7140, 121.0380),
            },
        ),
        (
            "meycauayan",
            {
                "label": "Meycauayan/MMORS",
                "anchor": (14.7368, 120.9606),
            },
        ),
        (
            "pampanga",
            {
                "label": "Pampanga River",
                "anchor": (14.9530, 120.7580),
            },
        ),
        (
            "laguna",
            {
                "label": "Laguna rivers/lake system",
                "anchor": (14.2290, 121.3260),
            },
        ),
    ]
)

STATUS_RANK = {
    "No Data": -1,
    "No Threshold": -1,
    "Normal": 0,
    "Stale": 0,
    "Offline": 0,
    "Alert": 1,
    "Alarm": 2,
    "Critical": 3,
}


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def target_key_for_row(row: pd.Series) -> str:
    """Assign a requested-river target using strict river/province metadata.

    Basin names are intentionally excluded. In particular, the generic
    ``Pasig-Laguna`` basin label must not cause every PMT station to be counted as
    a Laguna river station.
    """
    river = _norm(row.get("river_system"))
    station = _norm(row.get("station_name"))
    location = _norm(row.get("location"))
    province = _norm(row.get("province"))
    municipality = _norm(row.get("municipality"))
    combined_name = " ".join([river, station, location, municipality])

    if "marikina river" in river:
        return "marikina"
    if "tullahan river" in river:
        return "tullahan"
    if any(term in combined_name for term in ["meycauayan", "marilao", "obando", "mmors", "northville"]):
        return "meycauayan"
    if "pampanga river" in river or any(term in combined_name for term in ["sulipan", "apalit", "candaba", "paralaya"]):
        return "pampanga"

    laguna_terms = [
        "pagsanjan",
        "san cristobal",
        "santa cruz river",
        "sta cruz river",
        "victoria",
        "molawin",
        "san juan river laguna",
        "cabuyao river",
        "santa rosa river",
        "sta rosa river",
        "bay river",
    ]
    if province == "laguna" or any(term in combined_name for term in laguna_terms):
        return "laguna"
    return ""


def tag_target_stations(stations: pd.DataFrame) -> pd.DataFrame:
    output = stations.copy() if stations is not None else pd.DataFrame()
    if output.empty:
        output["target_key"] = pd.Series(dtype=str)
        output["target_label"] = pd.Series(dtype=str)
        return output
    output["target_key"] = output.apply(target_key_for_row, axis=1)
    output["target_label"] = output["target_key"].map(
        {key: config["label"] for key, config in TARGETS.items()}
    )
    return output


def _reference_row(group: pd.DataFrame) -> pd.Series:
    ranked = group.copy()
    ranked["_status_rank"] = ranked.get("water_status", "No Data").map(STATUS_RANK).fillna(-1)
    ranked["_time"] = pd.to_datetime(ranked.get("timestamp"), errors="coerce", utc=True)
    ranked["_rate_abs"] = pd.to_numeric(ranked.get("rise_rate_m_hr"), errors="coerce").abs().fillna(-1.0)
    ranked["_level"] = pd.to_numeric(ranked.get("level_m"), errors="coerce").fillna(-np.inf)
    ranked = ranked.sort_values(
        ["_status_rank", "_time", "_rate_abs", "_level", "station_name"],
        ascending=[False, False, False, False, True],
        na_position="last",
    )
    return ranked.iloc[0]


def target_river_summary(stations: pd.DataFrame) -> pd.DataFrame:
    tagged = tag_target_stations(stations)
    rows: list[dict[str, Any]] = []
    for key, config in TARGETS.items():
        group = tagged[tagged["target_key"] == key].copy() if not tagged.empty else pd.DataFrame()
        if group.empty:
            rows.append(
                {
                    "target": config["label"],
                    "stations": 0,
                    "reference_station": "-",
                    "reference_level_m": np.nan,
                    "worst_status": "No verified numerical reading",
                    "latest_observation": "-",
                    "data_state": "-",
                    "source": "-",
                }
            )
            continue

        reference = _reference_row(group)
        cached = group.get("is_cached", pd.Series(False, index=group.index)).fillna(False).astype(bool)
        if cached.all():
            state = "Cached"
        elif cached.any():
            state = "Mixed live/cache"
        else:
            state = "Live"
        sources = ", ".join(sorted(set(group.get("source_name", pd.Series(dtype=str)).dropna().astype(str))))
        latest_time = pd.to_datetime(group["timestamp"], errors="coerce", utc=True).max()
        rows.append(
            {
                "target": config["label"],
                "stations": int(group["station_id"].nunique()) if "station_id" in group else len(group),
                "reference_station": reference.get("station_name", "-"),
                "reference_level_m": reference.get("level_m", np.nan),
                "worst_status": reference.get("water_status", reference.get("threshold_status", "No Data")),
                "latest_observation": format_time(latest_time),
                "data_state": state,
                "source": sources or "-",
            }
        )
    return pd.DataFrame(rows)


def _station_table(group: pd.DataFrame) -> str:
    shown = group.copy()
    shown["_status_rank"] = shown.get("water_status", "No Data").map(STATUS_RANK).fillna(-1)
    shown["_time"] = pd.to_datetime(shown.get("timestamp"), errors="coerce", utc=True)
    shown = shown.sort_values(["_status_rank", "_time", "station_name"], ascending=[False, False, True])
    rows = []
    for _, row in shown.iterrows():
        level = pd.to_numeric(pd.Series([row.get("level_m")]), errors="coerce").iloc[0]
        rate = pd.to_numeric(pd.Series([row.get("rise_rate_m_hr")]), errors="coerce").iloc[0]
        level_text = f"{float(level):.3f}" if pd.notna(level) else "-"
        rate_text = f"{float(rate):+.3f}" if pd.notna(rate) else "-"
        cached_text = "Yes" if bool(row.get("is_cached", False)) else "No"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('station_name', '-')))}</td>"
            f"<td>{level_text}</td>"
            f"<td>{rate_text}</td>"
            f"<td>{html.escape(str(row.get('trend_label', row.get('source_trend', '-'))))}</td>"
            f"<td>{html.escape(str(row.get('water_status', row.get('threshold_status', '-'))))}</td>"
            f"<td>{html.escape(format_time(row.get('timestamp')))}</td>"
            f"<td>{cached_text}</td>"
            "</tr>"
        )
    return (
        "<div style='max-height:360px;overflow:auto'>"
        "<table style='border-collapse:collapse;width:100%;font-size:11px'>"
        "<thead><tr>"
        "<th style='text-align:left;padding:4px;border-bottom:1px solid #bbb'>Station</th>"
        "<th style='padding:4px;border-bottom:1px solid #bbb'>Level (m)</th>"
        "<th style='padding:4px;border-bottom:1px solid #bbb'>Δ (m/hr)</th>"
        "<th style='padding:4px;border-bottom:1px solid #bbb'>Trend</th>"
        "<th style='padding:4px;border-bottom:1px solid #bbb'>Status</th>"
        "<th style='padding:4px;border-bottom:1px solid #bbb'>Observed</th>"
        "<th style='padding:4px;border-bottom:1px solid #bbb'>Cached</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _target_bulletins(bulletins: pd.DataFrame, key: str) -> pd.DataFrame:
    if bulletins is None or bulletins.empty:
        return pd.DataFrame()
    text = (
        bulletins[["source_name", "basin_name", "river_system"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )
    if key == "marikina":
        mask = text.str.contains("marikina", na=False)
    elif key == "pampanga":
        mask = text.str.contains("pampanga", na=False)
    elif key == "laguna":
        mask = text.str.contains("laguna de bay", na=False)
    else:
        mask = pd.Series(False, index=bulletins.index)
    return bulletins[mask].copy()


def _bulletin_html(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    items = []
    for _, row in frame.iterrows():
        items.append(
            "<li>"
            f"<b>{html.escape(str(row.get('source_name', '-')))}</b>: "
            f"{html.escape(str(row.get('forecast_water_level', row.get('status', '-')) or '-'))} "
            f"(issued {html.escape(format_time(row.get('issued_at')))})"
            "</li>"
        )
    return "<hr><b>Related qualitative bulletin(s)</b><ul>" + "".join(items) + "</ul>"


def build_target_river_map(
    stations: pd.DataFrame,
    bulletins: pd.DataFrame | None = None,
) -> tuple[folium.Map, pd.DataFrame]:
    tagged = tag_target_stations(stations)
    map_object = folium.Map(location=[14.55, 121.0], zoom_start=8, tiles="cartodbpositron")
    summary_layer = folium.FeatureGroup(name="Requested-river summaries", show=True)
    exact_layer = folium.FeatureGroup(name="Verified exact station coordinates", show=True)
    bulletin_layer = folium.FeatureGroup(name="Qualitative basin bulletins", show=True)

    bounds: list[list[float]] = []
    for key, config in TARGETS.items():
        label = config["label"]
        lat, lon = config["anchor"]
        bounds.append([lat, lon])
        group = tagged[tagged["target_key"] == key].copy() if not tagged.empty else pd.DataFrame()
        related_bulletins = _target_bulletins(bulletins if bulletins is not None else pd.DataFrame(), key)

        if group.empty:
            status = "No Data"
            popup = (
                f"<div style='min-width:340px'><h4 style='margin:0 0 6px'>{html.escape(label)}</h4>"
                "<b>No verified numerical reading is currently available.</b>"
                + _bulletin_html(related_bulletins)
                + "<hr><small>Map position is a river-system display anchor, not a gauge coordinate.</small></div>"
            )
            count = 0
        else:
            reference = _reference_row(group)
            status = str(reference.get("water_status", "No Data"))
            popup = (
                f"<div style='min-width:720px;max-width:860px'>"
                f"<h4 style='margin:0 0 6px'>{html.escape(label)}</h4>"
                f"<b>Stations represented:</b> {group['station_id'].nunique()}<br>"
                f"<b>Most concerning current status:</b> {html.escape(status)}<br>"
                f"<b>Reference station:</b> {html.escape(str(reference.get('station_name', '-')))} — "
                f"{float(reference.get('level_m')):.3f} m<br>"
                f"<b>Latest observation in group:</b> {html.escape(format_time(pd.to_datetime(group['timestamp'], utc=True, errors='coerce').max()))}<br>"
                "<hr>"
                + _station_table(group)
                + _bulletin_html(related_bulletins)
                + "<hr><small>Large marker position is a river-system display anchor. Use the exact-coordinate layer only for verified station locations.</small></div>"
            )
            count = int(group["station_id"].nunique())

            exact = group.dropna(subset=["lat", "lon"]).copy()
            for _, row in exact.iterrows():
                exact_lat = float(row["lat"])
                exact_lon = float(row["lon"])
                bounds.append([exact_lat, exact_lon])
                exact_status = str(row.get("water_status", "No Data"))
                exact_popup = (
                    f"<b>{html.escape(str(row.get('station_name', '-')))}</b><br>"
                    f"River: {html.escape(label)}<br>"
                    f"Level: {float(row.get('level_m')):.3f} m<br>"
                    f"Status: {html.escape(exact_status)}<br>"
                    f"Trend: {html.escape(str(row.get('trend_label', '-')))}<br>"
                    f"Observed: {html.escape(format_time(row.get('timestamp')))}<br>"
                    f"Source: {html.escape(str(row.get('source_name', '-')))}"
                )
                folium.CircleMarker(
                    location=[exact_lat, exact_lon],
                    radius=7,
                    color="#111827",
                    weight=2,
                    fill=True,
                    fill_color=WATER_COLORS.get(exact_status, "#64748b"),
                    fill_opacity=0.95,
                    popup=folium.Popup(exact_popup, max_width=420),
                    tooltip=f"{row.get('station_name')}: {float(row.get('level_m')):.2f} m",
                ).add_to(exact_layer)

        color = WATER_COLORS.get(status, "#64748b")
        marker_text = f"{html.escape(label)} · {count} gauge(s)" if count else f"{html.escape(label)} · no numerical reading"
        marker_html = (
            "<div style='white-space:nowrap;transform:translate(-50%,-50%);'>"
            f"<span style='display:inline-block;background:{color};color:white;border:2px solid #111827;"
            "border-radius:16px;padding:5px 9px;font-size:12px;font-weight:800;"
            "box-shadow:0 2px 5px rgba(0,0,0,.35)'>"
            f"{marker_text}</span></div>"
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=10,
            color="#111827",
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.95,
            popup=folium.Popup(popup, max_width=900),
            tooltip=marker_text,
        ).add_to(summary_layer)
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(html=marker_html, icon_size=(250, 40), icon_anchor=(0, 20)),
            popup=folium.Popup(popup, max_width=900),
            tooltip=marker_text,
        ).add_to(summary_layer)

        if not related_bulletins.empty:
            shifted = [lat - 0.025, lon + 0.025]
            bulletin_popup = (
                f"<div style='min-width:420px'><h4 style='margin:0 0 6px'>{html.escape(label)} — basin bulletin</h4>"
                + _bulletin_html(related_bulletins)
                + "<small>Qualitative bulletin only; this is not a numerical gauge measurement.</small></div>"
            )
            folium.Marker(
                location=shifted,
                icon=folium.Icon(color="blue", icon="info-sign"),
                popup=folium.Popup(bulletin_popup, max_width=600),
                tooltip=f"{label}: qualitative basin bulletin",
            ).add_to(bulletin_layer)

    summary_layer.add_to(map_object)
    exact_layer.add_to(map_object)
    bulletin_layer.add_to(map_object)
    if bounds:
        map_object.fit_bounds(bounds, padding=(25, 25))
    folium.LayerControl(collapsed=False).add_to(map_object)
    return map_object, tagged
