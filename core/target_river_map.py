from __future__ import annotations

import html
import re
from collections import OrderedDict
from typing import Any

import folium
import numpy as np
import pandas as pd

from core.maps import WATER_COLORS, format_time

# Representative display anchors. Exact station markers are plotted separately
# only when a provider supplies verified latitude/longitude values.
TARGETS = OrderedDict(
    [
        (
            "marikina",
            {
                "label": "Marikina River",
                "anchor": (14.6507, 121.1029),
                "bulletin_terms": ["marikina"],
            },
        ),
        (
            "tullahan",
            {
                "label": "Tullahan River",
                "anchor": (14.7140, 121.0380),
                "bulletin_terms": ["tullahan"],
            },
        ),
        (
            "meycauayan",
            {
                "label": "Meycauayan/MMORS",
                "anchor": (14.7368, 120.9606),
                "bulletin_terms": ["meycauayan", "marilao", "obando", "mmors"],
            },
        ),
        (
            "pampanga",
            {
                "label": "Pampanga River",
                "anchor": (15.0460, 120.7300),
                "bulletin_terms": ["pampanga"],
            },
        ),
        (
            "laguna",
            {
                "label": "Laguna de Bay / Laguna area",
                "anchor": (14.2290, 121.3260),
                "bulletin_terms": ["laguna de bay", "pasig laguna"],
            },
        ),
        (
            "abra",
            {
                "label": "Abra River Basin",
                "anchor": (17.5750, 120.6200),
                "bulletin_terms": ["abra"],
            },
        ),
        (
            "samar",
            {
                "label": "Samar rivers",
                "anchor": (11.7800, 125.0000),
                "bulletin_terms": ["samar"],
            },
        ),
        (
            "panay",
            {
                "label": "Panay River Basin",
                "anchor": (11.5500, 122.7600),
                "bulletin_terms": ["panay"],
            },
        ),
        (
            "cagayan_de_oro",
            {
                "label": "Cagayan de Oro River Basin",
                "anchor": (8.4800, 124.6500),
                "bulletin_terms": ["cagayan de oro"],
            },
        ),
        (
            "davao",
            {
                "label": "Davao River Basin",
                "anchor": (7.1800, 125.4300),
                "bulletin_terms": ["davao"],
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

BULLETIN_COLORS = {
    "Flood Watch": "#dc2626",
    "Warning": "#ea580c",
    "Advisory": "#f59e0b",
    "Increasing": "#f97316",
    "Normal": "#16a34a",
    "Non-Flood Watch": "#16a34a",
    "Forecast": "#2563eb",
    "No Data": "#64748b",
}


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def target_key_for_row(row: pd.Series) -> str:
    """Assign strict requested-river groups without using broad basin labels."""
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
    if "pampanga river" in river or any(term in combined_name for term in ["sulipan", "apalit", "candaba", "zaragoza", "arayat", "paralaya"]):
        return "pampanga"
    if "abra" in river or "abra river" in combined_name:
        return "abra"
    if "panay" in river or "panay river" in combined_name:
        return "panay"
    if "cagayan de oro" in river or "cagayan de oro" in combined_name:
        return "cagayan_de_oro"
    if "davao river" in river or "davao river" in combined_name:
        return "davao"
    if "samar" in river or "samar" in province or "samar" in combined_name:
        return "samar"

    laguna_terms = [
        "laguna de bay",
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
        "napindan",
        "laguna de bay outlet",
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


def _bulletin_text(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return ""
    newest = frame.copy()
    newest["_issued"] = pd.to_datetime(newest.get("issued_at"), errors="coerce", utc=True)
    newest = newest.sort_values(["_issued", "scraped_at"], ascending=False, na_position="last")
    row = newest.iloc[0]
    parts = [
        str(row.get("forecast_water_level", "") or "").strip(),
        str(row.get("status", "") or "").strip(),
    ]
    return next((part for part in parts if part), "Official advisory available")


def _target_bulletins(bulletins: pd.DataFrame, key: str) -> pd.DataFrame:
    if bulletins is None or bulletins.empty:
        return pd.DataFrame()
    config = TARGETS[key]
    searchable = (
        bulletins[["source_name", "basin_name", "river_system"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .map(_norm)
    )
    mask = pd.Series(False, index=bulletins.index)
    for term in config.get("bulletin_terms", []):
        mask = mask | searchable.str.contains(_norm(term), regex=False, na=False)
    return bulletins[mask].copy()


def _bulletin_category(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return "No Data"
    text = " ".join(
        frame[["status", "forecast_water_level", "notes"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .tolist()
    ).lower()
    if "non-flood watch" in text:
        return "Non-Flood Watch"
    if "flood watch" in text:
        return "Flood Watch"
    if "critical" in text or "warning" in text:
        return "Warning"
    if "advisory" in text:
        return "Advisory"
    if any(term in text for term in ["increase", "rising", "upward"]):
        return "Increasing"
    if any(term in text for term in ["remain normal", "normal during", "below alert"]):
        return "Normal"
    return "Forecast"


def target_river_summary(
    stations: pd.DataFrame,
    bulletins: pd.DataFrame | None = None,
) -> pd.DataFrame:
    tagged = tag_target_stations(stations)
    bulletins = bulletins if bulletins is not None else pd.DataFrame()
    rows: list[dict[str, Any]] = []

    for key, config in TARGETS.items():
        group = tagged[tagged["target_key"] == key].copy() if not tagged.empty else pd.DataFrame()
        related = _target_bulletins(bulletins, key)
        bulletin_summary = _bulletin_text(related)
        bulletin_issued = "-"
        bulletin_sources: list[str] = []
        if not related.empty:
            issued = pd.to_datetime(related["issued_at"], errors="coerce", utc=True).max()
            bulletin_issued = format_time(issued)
            bulletin_sources = sorted(set(related["source_name"].dropna().astype(str)))

        if group.empty:
            coverage = "Official advisory/forecast" if not related.empty else "No current extracted source"
            rows.append(
                {
                    "target": config["label"],
                    "coverage": coverage,
                    "numerical_stations": 0,
                    "reference_station": "-",
                    "reference_level_m": np.nan,
                    "observed_status": "No verified numerical reading",
                    "official_forecast_or_advisory": bulletin_summary or "-",
                    "issued": bulletin_issued,
                    "data_state": "Cached" if (not related.empty and related.get("is_cached", pd.Series(False, index=related.index)).fillna(False).all()) else "-",
                    "source": ", ".join(bulletin_sources) or "-",
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
        sources = sorted(set(group.get("source_name", pd.Series(dtype=str)).dropna().astype(str)))
        all_sources = sorted(set(sources + bulletin_sources))
        latest_time = pd.to_datetime(group["timestamp"], errors="coerce", utc=True).max()
        coverage = "Numerical measurement + official advisory" if not related.empty else "Numerical measurement"
        rows.append(
            {
                "target": config["label"],
                "coverage": coverage,
                "numerical_stations": int(group["station_id"].nunique()) if "station_id" in group else len(group),
                "reference_station": reference.get("station_name", "-"),
                "reference_level_m": reference.get("level_m", np.nan),
                "observed_status": reference.get("water_status", reference.get("threshold_status", "No Data")),
                "official_forecast_or_advisory": bulletin_summary or "-",
                "issued": bulletin_issued if bulletin_issued != "-" else format_time(latest_time),
                "data_state": state,
                "source": ", ".join(all_sources) or "-",
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
        level_30 = pd.to_numeric(pd.Series([row.get("level_30min_ago_m")]), errors="coerce").iloc[0]
        level_1h = pd.to_numeric(pd.Series([row.get("level_1hr_ago_m")]), errors="coerce").iloc[0]
        level_2h = pd.to_numeric(pd.Series([row.get("level_2hr_ago_m")]), errors="coerce").iloc[0]
        rate = pd.to_numeric(pd.Series([row.get("rise_rate_m_hr")]), errors="coerce").iloc[0]
        level_text = f"{float(level):.3f}" if pd.notna(level) else "-"
        level_30_text = f"{float(level_30):.3f}" if pd.notna(level_30) else "-"
        level_1h_text = f"{float(level_1h):.3f}" if pd.notna(level_1h) else "-"
        level_2h_text = f"{float(level_2h):.3f}" if pd.notna(level_2h) else "-"
        rate_text = f"{float(rate):+.3f}" if pd.notna(rate) else "-"
        cached_text = "Yes" if bool(row.get("is_cached", False)) else "No"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('station_name', '-')))}</td>"
            f"<td>{level_text}</td>"
            f"<td>{level_30_text}</td>"
            f"<td>{level_1h_text}</td>"
            f"<td>{level_2h_text}</td>"
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
        "<th style='padding:4px;border-bottom:1px solid #bbb'>Current level (m)</th>"
        "<th style='padding:4px;border-bottom:1px solid #bbb'>-30 min</th>"
        "<th style='padding:4px;border-bottom:1px solid #bbb'>-1 hr</th>"
        "<th style='padding:4px;border-bottom:1px solid #bbb'>-2 hr</th>"
        "<th style='padding:4px;border-bottom:1px solid #bbb'>Delta (m/hr)</th>"
        "<th style='padding:4px;border-bottom:1px solid #bbb'>Trend</th>"
        "<th style='padding:4px;border-bottom:1px solid #bbb'>Status</th>"
        "<th style='padding:4px;border-bottom:1px solid #bbb'>Observed</th>"
        "<th style='padding:4px;border-bottom:1px solid #bbb'>Cached</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _bulletin_html(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return ""
    display = frame.copy()
    display["_issued"] = pd.to_datetime(display.get("issued_at"), errors="coerce", utc=True)
    display = display.sort_values(["_issued", "scraped_at"], ascending=False, na_position="last")
    items = []
    for _, row in display.head(5).iterrows():
        source_url = str(row.get("source_url", "") or "")
        source_link = (
            f"<a href='{html.escape(source_url)}' target='_blank'>{html.escape(str(row.get('source_name', '-')))}</a>"
            if source_url.startswith("http")
            else html.escape(str(row.get("source_name", "-")))
        )
        details = [
            str(row.get("status", "") or "").strip(),
            str(row.get("forecast_water_level", "") or "").strip(),
            str(row.get("forecast_rainfall", "") or "").strip(),
        ]
        details_text = " | ".join(part for part in details if part) or "Official advisory available"
        items.append(
            "<li>"
            f"<b>{source_link}</b>: {html.escape(details_text)} "
            f"(issued {html.escape(format_time(row.get('issued_at')))})"
            "</li>"
        )
    return (
        "<hr><b>Official qualitative bulletin/advisory</b><ul>"
        + "".join(items)
        + "</ul><small>This section is advisory information, not a numerical gauge reading.</small>"
    )


def _marker_style(group: pd.DataFrame, related: pd.DataFrame) -> tuple[str, str, str]:
    if group is not None and not group.empty:
        reference = _reference_row(group)
        status = str(reference.get("water_status", "No Data"))
        return WATER_COLORS.get(status, "#64748b"), status, "numerical"
    if related is not None and not related.empty:
        category = _bulletin_category(related)
        return BULLETIN_COLORS.get(category, "#2563eb"), category, "advisory"
    return "#64748b", "No Data", "none"


def build_target_river_map(
    stations: pd.DataFrame,
    bulletins: pd.DataFrame | None = None,
) -> tuple[folium.Map, pd.DataFrame]:
    tagged = tag_target_stations(stations)
    bulletins = bulletins if bulletins is not None else pd.DataFrame()
    map_object = folium.Map(location=[12.5, 122.0], zoom_start=5.5, tiles="cartodbpositron")
    summary_layer = folium.FeatureGroup(name="Requested rivers and official source summaries", show=True)
    exact_layer = folium.FeatureGroup(name="Verified numerical station coordinates", show=True)
    advisory_layer = folium.FeatureGroup(name="Official forecast/advisory markers", show=True)

    bounds: list[list[float]] = []
    for key, config in TARGETS.items():
        label = config["label"]
        lat, lon = config["anchor"]
        bounds.append([lat, lon])
        group = tagged[tagged["target_key"] == key].copy() if not tagged.empty else pd.DataFrame()
        related = _target_bulletins(bulletins, key)
        color, display_status, mode = _marker_style(group, related)

        if group.empty:
            numeric_html = "<b>No verified numerical reading is currently available.</b>"
            count = 0
        else:
            reference = _reference_row(group)
            numeric_html = (
                f"<b>Numerical stations:</b> {group['station_id'].nunique()}<br>"
                f"<b>Most concerning observed status:</b> {html.escape(str(reference.get('water_status', 'No Data')))}<br>"
                f"<b>Reference station:</b> {html.escape(str(reference.get('station_name', '-')))} — "
                f"{float(reference.get('level_m')):.3f} m<br>"
                f"<b>Latest observation:</b> {html.escape(format_time(pd.to_datetime(group['timestamp'], utc=True, errors='coerce').max()))}<br>"
                "<hr>" + _station_table(group)
            )
            count = int(group["station_id"].nunique())

            exact = group.dropna(subset=["lat", "lon"]).copy()
            for _, row in exact.iterrows():
                exact_lat = float(row["lat"])
                exact_lon = float(row["lon"])
                bounds.append([exact_lat, exact_lon])
                exact_status = str(row.get("water_status", "No Data"))
                source_url = str(row.get("source_url", "") or "")
                source_line = (
                    f"<a href='{html.escape(source_url)}' target='_blank'>Open source</a>"
                    if source_url.startswith("http")
                    else "-"
                )
                exact_popup = (
                    f"<b>{html.escape(str(row.get('station_name', '-')))}</b><br>"
                    f"Target: {html.escape(label)}<br>"
                    f"Level: {float(row.get('level_m')):.3f} m<br>"
                    f"Status: {html.escape(exact_status)}<br>"
                    f"Trend: {html.escape(str(row.get('trend_label', '-')))}<br>"
                    f"Observed: {html.escape(format_time(row.get('timestamp')))}<br>"
                    f"Source: {html.escape(str(row.get('source_name', '-')))} · {source_line}<br>"
                    f"Notes: {html.escape(str(row.get('notes', '') or ''))}"
                )
                folium.CircleMarker(
                    location=[exact_lat, exact_lon],
                    radius=7,
                    color="#111827",
                    weight=2,
                    fill=True,
                    fill_color=WATER_COLORS.get(exact_status, "#64748b"),
                    fill_opacity=0.95,
                    popup=folium.Popup(exact_popup, max_width=470),
                    tooltip=f"{row.get('station_name')}: {float(row.get('level_m')):.2f} m",
                ).add_to(exact_layer)

        popup = (
            f"<div style='min-width:480px;max-width:900px'>"
            f"<h4 style='margin:0 0 6px'>{html.escape(label)}</h4>"
            f"<b>Map summary:</b> {html.escape(display_status)}<br>"
            + numeric_html
            + _bulletin_html(related)
            + "<hr><small>Large marker position is a river-system display anchor. Exact station locations appear only in the verified-coordinate layer.</small></div>"
        )

        if mode == "numerical":
            marker_text = f"{label} · {count} gauge(s)"
        elif mode == "advisory":
            marker_text = f"{label} · official {display_status}"
        else:
            marker_text = f"{label} · no extracted source"

        marker_html = (
            "<div style='white-space:nowrap;transform:translate(-50%,-50%);'>"
            f"<span style='display:inline-block;background:{color};color:white;border:2px solid #111827;"
            "border-radius:15px;padding:4px 8px;font-size:11px;font-weight:800;"
            "box-shadow:0 2px 5px rgba(0,0,0,.35)'>"
            f"{html.escape(marker_text)}</span></div>"
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=9,
            color="#111827",
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.95,
            popup=folium.Popup(popup, max_width=930),
            tooltip=marker_text,
        ).add_to(summary_layer)
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(html=marker_html, icon_size=(225, 36), icon_anchor=(112, 18)),
            popup=folium.Popup(popup, max_width=930),
            tooltip=marker_text,
        ).add_to(summary_layer)

        if not related.empty:
            advisory_popup = (
                f"<div style='min-width:430px'><h4 style='margin:0 0 6px'>{html.escape(label)} — official advisory</h4>"
                + _bulletin_html(related)
                + "</div>"
            )
            folium.Marker(
                location=[lat - 0.04, lon + 0.04],
                icon=folium.Icon(color="blue", icon="info-sign"),
                popup=folium.Popup(advisory_popup, max_width=650),
                tooltip=f"{label}: official bulletin/advisory",
            ).add_to(advisory_layer)

    summary_layer.add_to(map_object)
    exact_layer.add_to(map_object)
    advisory_layer.add_to(map_object)
    if bounds:
        map_object.fit_bounds(bounds, padding=(20, 20))
    folium.LayerControl(collapsed=False).add_to(map_object)
    return map_object, tagged
