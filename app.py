from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

# Ensure the repository root is importable on Streamlit Cloud.
APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from core.geospatial import assign_basins
from core.maps import build_monitoring_map, format_time
from core.province_water_map import (
    build_province_water_map,
    fetch_province_reference,
    match_station_provinces,
)
from core.target_river_map import build_target_river_map, target_river_summary
from core.rainfall import (
    basin_dataframe,
    compute_hazard,
    fetch_openmeteo_basin_forecast,
    load_geojson,
    sample_basin_forecast,
)
from core.registry import apply_supplementary_registry, load_registry
from core.schema import ProviderResult, combine_provider_results
from core.water import STATUS_RANK, aggregate_by_basin, combined_level, compute_station_state
from providers import (
    bulacan_pdrrmo,
    llda_water_level,
    official_reports_csv,
    pagasa_bulletins,
    pagasa_pmt,
    philsensors,
)

st.set_page_config(
    page_title="PH Multi-Source River Monitor",
    page_icon="🌊",
    layout="wide",
)

BUILD = "5.4.0"
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / ".cache"
GEOJSON_PATH = DATA_DIR / "major_river_basins_simplified.geojson"
SAMPLE_RAIN_PATH = DATA_DIR / "sample_basin_rainfall.csv"
PHILSENSORS_REGISTRY_PATH = DATA_DIR / "philsensors_station_registry.csv"
OFFICIAL_REPORT_TEMPLATE = DATA_DIR / "official_report_template.csv"
SUPPLEMENTARY_REGISTRY_PATH = DATA_DIR / "supplementary_station_registry.csv"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def refresh_bucket(minutes: int) -> int:
    return int(pd.Timestamp.now(tz="UTC").timestamp() // (max(minutes, 1) * 60))


def read_registry(uploaded) -> pd.DataFrame:
    if uploaded is not None:
        return pd.read_csv(uploaded)
    if PHILSENSORS_REGISTRY_PATH.exists():
        try:
            return pd.read_csv(PHILSENSORS_REGISTRY_PATH)
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=60 * 30)
def cached_rain_forecast(master_json: str) -> tuple[pd.DataFrame, list[str]]:
    return fetch_openmeteo_basin_forecast(pd.read_json(io.StringIO(master_json)))


@st.cache_data(show_spinner=False, max_entries=4)
def cached_philsensors(bucket: int, timeout_ms: int, registry_csv: str, load_metadata: bool) -> ProviderResult:
    del bucket
    registry = pd.read_csv(io.StringIO(registry_csv)) if registry_csv.strip() else pd.DataFrame()
    return philsensors.fetch(CACHE_DIR, registry=registry, timeout_ms=timeout_ms, load_station_metadata=load_metadata)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_pmt(bucket: int, timeout_ms: int) -> ProviderResult:
    del bucket
    return pagasa_pmt.fetch(CACHE_DIR, timeout_ms=timeout_ms)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_bulacan(bucket: int) -> ProviderResult:
    del bucket
    return bulacan_pdrrmo.fetch(CACHE_DIR)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_llda(bucket: int, timeout_ms: int) -> ProviderResult:
    del bucket
    return llda_water_level.fetch(CACHE_DIR, timeout_ms=timeout_ms)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_bulletins(bucket: int) -> ProviderResult:
    del bucket
    return pagasa_bulletins.fetch(CACHE_DIR)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def cached_province_reference() -> tuple[dict, pd.DataFrame, str]:
    return fetch_province_reference()


def source_health_frame(results: list[ProviderResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "provider": result.provider,
                "mode": result.mode,
                "readings": len(result.readings) if result.readings is not None else 0,
                "bulletins": len(result.bulletins) if result.bulletins is not None else 0,
                "fetched_at": format_time(result.fetched_at),
                "message": result.message,
                "error": result.error,
            }
        )
    return pd.DataFrame(rows)


st.title("🌊 Philippine Multi-Source River Monitoring Dashboard")
st.caption(
    "Independent providers for PhilSensors, PAGASA NCR–Rizal/PMT gauges, LLDA Laguna de Bay, "
    "Bulacan/Pampanga stations, PAGASA basin forecasts for Abra, Panay, Cagayan de Oro and Davao, "
    "Samar regional advisories, and optional official LGU report imports."
)

with st.sidebar:
    st.header("Controls")
    st.caption(f"Clean package build {BUILD}")
    rainfall_source = st.radio("Rainfall source", ["Live Open-Meteo", "Sample data"], index=0)
    rain_window = st.radio("Rainfall window", ["24h", "72h"], horizontal=True)
    refresh_minutes = st.selectbox("Provider refresh", [5, 10, 15, 30, 60], index=1)
    map_view = st.radio("Basin map", ["Combined", "Rainfall", "Water"], index=0)

    st.subheader("Water providers")
    use_philsensors = st.checkbox("DOST-ASTI PhilSensors", value=True)
    use_pmt = st.checkbox("PAGASA NCR–Rizal live gauges", value=True)
    use_bulacan = st.checkbox("Bulacan PDRRMO river stations", value=True)
    use_llda = st.checkbox("LLDA Laguna de Bay water level", value=True)
    use_bulletins = st.checkbox("PAGASA basin forecasts and regional advisories", value=True)

    load_metadata = st.checkbox(
        "Load PhilSensors station coordinates",
        value=True,
        disabled=not use_philsensors,
    )
    browser_timeout = st.number_input("Browser timeout (seconds)", 30, 180, 90, 10)
    registry_upload = st.file_uploader("Optional PhilSensors registry CSV", type=["csv"])
    supplementary_registry_upload = st.file_uploader(
        "Optional supplementary-source registry CSV",
        type=["csv"],
        help="Adds verified coordinates or corrected river/basin metadata for PAGASA PMT and Bulacan stations.",
    )
    official_upload = st.file_uploader(
        "Optional official LGU report CSV",
        type=["csv"],
        help="Use this for targeted public DRRMO reports collected during an alerted event.",
    )

    stale_minutes = st.number_input("Stale after (minutes)", 30, 2880, 240, 30)
    offline_minutes = st.number_input("Offline after (minutes)", 60, 10080, 1440, 60)
    rapid_rise = st.number_input("Rapid rise (m/hour)", 0.01, 5.0, 0.30, 0.05)
    rapid_fall = st.number_input("Rapid fall (m/hour)", 0.01, 5.0, 0.30, 0.05)

    st.subheader("Requested-river map")
    show_target_river_map = st.checkbox(
        "Show requested-river watch map",
        value=True,
        help="Shows numerical gauges and official forecasts/advisories for Marikina, Tullahan, Meycauayan, Pampanga, Laguna, Abra, Samar, Panay, Cagayan de Oro, and Davao.",
    )

    st.subheader("Province rise/fall map")
    show_province_trend_map = st.checkbox(
        "Show province water-level trend map",
        value=True,
        help="Restores the large province labels and trend shading from dashboard build 3.8.",
    )
    include_inactive_province_map = st.checkbox(
        "Include stale/offline gauges when a change value is available",
        value=True,
        disabled=not show_province_trend_map,
    )
    only_rapid_province_map = st.checkbox(
        "Show only rapid-rise or rapid-fall provinces",
        value=False,
        disabled=not show_province_trend_map,
    )

    if st.button("Force all providers to refresh"):
        cached_philsensors.clear()
        cached_pmt.clear()
        cached_bulacan.clear()
        cached_llda.clear()
        cached_bulletins.clear()
        cached_province_reference.clear()
        st.rerun()

geojson = load_geojson(GEOJSON_PATH)
basin_master = basin_dataframe(geojson)
if basin_master.empty:
    st.error("The basin GeoJSON has no features.")
    st.stop()

if rainfall_source == "Live Open-Meteo":
    with st.spinner("Loading rainfall forecasts..."):
        rain_values, rain_errors = cached_rain_forecast(basin_master.to_json(orient="records"))
else:
    rain_values = sample_basin_forecast(SAMPLE_RAIN_PATH)
    rain_errors = []

hazard_df = compute_hazard(basin_master, rain_values, window=rain_window)

providers: list[ProviderResult] = []
bucket = refresh_bucket(int(refresh_minutes))
registry = read_registry(registry_upload)
registry_csv = registry.to_csv(index=False) if not registry.empty else ""

if use_philsensors:
    with st.spinner("Loading PhilSensors..."):
        providers.append(cached_philsensors(bucket, int(browser_timeout * 1000), registry_csv, load_metadata))
if use_pmt:
    with st.spinner("Loading PAGASA NCR–Rizal water-level table..."):
        providers.append(cached_pmt(bucket, int(browser_timeout * 1000)))
if use_bulacan:
    with st.spinner("Loading Bulacan PDRRMO river stations..."):
        providers.append(cached_bulacan(bucket))
if use_llda:
    with st.spinner("Loading the official LLDA Laguna de Bay level..."):
        providers.append(cached_llda(bucket, int(browser_timeout * 1000)))
if use_bulletins:
    with st.spinner("Loading PAGASA basin forecasts and regional advisories..."):
        providers.append(cached_bulletins(bucket))
if official_upload is not None:
    providers.append(official_reports_csv.from_dataframe(pd.read_csv(official_upload)))

water_readings, bulletin_df = combine_provider_results(providers)
supplementary_registry = (
    pd.read_csv(supplementary_registry_upload)
    if supplementary_registry_upload is not None
    else load_registry(SUPPLEMENTARY_REGISTRY_PATH)
)
water_readings = apply_supplementary_registry(water_readings, supplementary_registry)
water_readings = assign_basins(water_readings, geojson)
station_df = compute_station_state(
    water_readings,
    stale_minutes=int(stale_minutes),
    offline_minutes=int(offline_minutes),
    rapid_rise_m_hr=float(rapid_rise),
    rapid_fall_m_hr=float(rapid_fall),
)

basin_water = aggregate_by_basin(station_df)
water_defaults = {
    "water_status": "No Data",
    "station_count": 0,
    "active_station_count": 0,
    "max_level_m": np.nan,
    "max_rise_rate_m_hr": np.nan,
    "rapid_rise_count": 0,
    "latest_water_timestamp": pd.NaT,
}
if not basin_water.empty:
    hazard_df = hazard_df.merge(basin_water, on="basin_name", how="left")
for column, default in water_defaults.items():
    if column not in hazard_df:
        hazard_df[column] = default
    hazard_df[column] = hazard_df[column].fillna(default)
hazard_df["combined_level"] = hazard_df.apply(
    lambda row: combined_level(row["hazard_level"], row["water_status"]), axis=1
)

metrics = st.columns(6)
metrics[0].metric("Basins", len(hazard_df))
metrics[1].metric("Providers enabled", len(providers))
metrics[2].metric("Water stations", len(station_df))
metrics[3].metric("Mapped stations", int(station_df[["lat", "lon"]].notna().all(axis=1).sum()) if not station_df.empty else 0)
metrics[4].metric("Alarm/Critical", int(station_df["water_status"].isin(["Alarm", "Critical"]).sum()) if not station_df.empty else 0)
metrics[5].metric("Rapid rise", int(station_df.get("rapid_rise", pd.Series(dtype=bool)).fillna(False).sum()) if not station_df.empty else 0)

if rain_errors:
    with st.expander("Rainfall retrieval issues"):
        st.write(rain_errors)

st.subheader("Requested river coverage")
target_summary = target_river_summary(station_df, bulletin_df)
if "reference_level_m" in target_summary:
    target_summary["reference_level_m"] = pd.to_numeric(
        target_summary["reference_level_m"], errors="coerce"
    ).round(3)
st.dataframe(target_summary, use_container_width=True, hide_index=True)
st.caption(
    "Numerical readings and qualitative forecasts are kept separate. The reference level belongs to the named station; "
    "levels from different gauges are never averaged because their datums and thresholds can differ. "
    "LLDA is treated as a lake-wide Laguna de Bay measurement, while Abra, Samar, Panay, Cagayan de Oro and Davao may show official forecasts/advisories without a numerical gauge."
)

if show_target_river_map:
    st.markdown("---")
    st.subheader("🗺️ National Requested-River and Official-Source Map")
    st.caption(
        "Large labels summarize each requested river or area. Click a label to see numerical stations and related official forecasts/advisories. "
        "The exact-coordinate layer plots only verified numerical locations; qualitative markers are clearly labelled as forecasts or advisories."
    )
    target_map, tagged_target_stations = build_target_river_map(station_df, bulletin_df)
    st_folium(
        target_map,
        height=650,
        use_container_width=True,
        key="requested_river_watch_map_v54",
    )
    target_count = int(tagged_target_stations["target_key"].ne("").sum()) if not tagged_target_stations.empty else 0
    exact_count = int(
        tagged_target_stations.loc[tagged_target_stations["target_key"].ne(""), ["lat", "lon"]]
        .notna()
        .all(axis=1)
        .sum()
    ) if not tagged_target_stations.empty else 0
    st.caption(
        f"Requested-river station rows represented: {target_count}. "
        f"Rows with verified exact coordinates: {exact_count}."
    )

st.subheader("Data-source health")
health = source_health_frame(providers)
st.dataframe(health, use_container_width=True, hide_index=True)
for result in providers:
    partial = result.details.get("partial_errors") or result.details.get("retrieval_errors") or result.details.get("live_errors")
    if partial:
        with st.expander(f"{result.provider}: retrieval details"):
            st.write(partial)
    elif result.error:
        with st.expander(f"{result.provider}: error details"):
            st.code(result.error)

if show_province_trend_map:
    st.markdown("---")
    st.subheader("💧 Observed Water-Level Rise/Fall Map")
    st.caption(
        "Province shading and large labels show the strongest measured hourly change. "
        "Red/orange means rising; blue means falling; grey means nearly stable. "
        "The map combines compatible readings from all enabled providers, not only PhilSensors."
    )
    st.markdown(
        """
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin:4px 0 12px">
          <span style="background:#b91c1c;color:white;padding:4px 8px;border-radius:12px">↑↑ Rapid rise</span>
          <span style="background:#f97316;color:white;padding:4px 8px;border-radius:12px">↑ Rising</span>
          <span style="background:#6b7280;color:white;padding:4px 8px;border-radius:12px">→ Stable</span>
          <span style="background:#3b82f6;color:white;padding:4px 8px;border-radius:12px">↓ Falling</span>
          <span style="background:#1d4ed8;color:white;padding:4px 8px;border-radius:12px">↓↓ Rapid fall</span>
          <span style="background:#7e22ce;color:white;padding:4px 8px;border-radius:12px">↕ Mixed rapid change</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if station_df.empty:
        st.info("No water-level station readings are available for the province trend map.")
    else:
        try:
            with st.spinner("Loading province boundaries and restoring water-level trend labels..."):
                province_geojson, province_centroids, province_source_name = cached_province_reference()
                province_station_df = match_station_provinces(station_df, province_centroids)
                province_map, mapped_province_stations, province_summary_df = build_province_water_map(
                    province_station_df,
                    province_geojson,
                    include_inactive=include_inactive_province_map,
                    only_rapid=only_rapid_province_map,
                )
            st_folium(
                province_map,
                height=760,
                use_container_width=True,
                key="province_water_level_map_v51",
            )
            matched_count = int(mapped_province_stations["province_ref_key"].notna().sum()) if not mapped_province_stations.empty else 0
            st.caption(
                f"Province boundary source: {province_source_name} · "
                f"Province-matched gauges displayed: {matched_count} of {len(station_df)}."
            )
            if province_summary_df.empty:
                st.warning(
                    "No province labels were produced. Keep 'Include stale/offline gauges' enabled "
                    "or confirm that the stations include province names and numeric rise/fall values."
                )
        except Exception as exc:
            st.error(f"The province water-level map could not be rendered: {type(exc).__name__}: {exc}")
            st.caption(
                "The combined basin map and station tables remain available. The province map "
                "depends on public Philippine government boundary services."
            )

st.markdown("---")
st.subheader("Combined rainfall and water-level map")
monitor_map = build_monitoring_map(geojson, hazard_df, station_df, view=map_view)
st_folium(monitor_map, height=720, use_container_width=True, key="clean_v54_monitor_map")

station_tab, history_tab, bulletin_tab, basin_tab = st.tabs(
    ["Latest water readings", "Reading history", "Basin bulletins", "Basin screening"]
)

with station_tab:
    if station_df.empty:
        st.info("No verified numerical water-level readings are currently available from the enabled providers.")
    else:
        display = station_df.copy()
        display["observed_at"] = display["timestamp"].apply(format_time)
        for numeric_column in [
            "level_m", "level_30min_ago_m", "level_1hr_ago_m",
            "level_2hr_ago_m", "rise_rate_m_hr",
        ]:
            if numeric_column in display.columns:
                display[numeric_column] = pd.to_numeric(
                    display[numeric_column], errors="coerce"
                ).round(3)
        display["mapped"] = display[["lat", "lon"]].notna().all(axis=1)
        columns = [
            "source_name", "station_name", "river_system", "basin_name", "province", "municipality",
            "level_m", "level_30min_ago_m", "level_1hr_ago_m", "level_2hr_ago_m",
            "rise_rate_m_hr", "trend_label", "water_status", "threshold_status",
            "alert_m", "alarm_m", "critical_m", "observed_at", "mapped", "is_cached", "source_url", "notes",
        ]
        st.dataframe(
            display[columns].sort_values(
                by="water_status",
                key=lambda series: series.map(STATUS_RANK),
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )

with history_tab:
    if water_readings.empty:
        st.info("No water-level history is loaded.")
    else:
        choices = (
            water_readings[["station_id", "station_name", "source_name"]]
            .drop_duplicates()
            .assign(label=lambda frame: frame["station_name"] + " — " + frame["source_name"])
        )
        selected_label = st.selectbox("Station", choices["label"].tolist())
        station_id = choices.loc[choices["label"] == selected_label, "station_id"].iloc[0]
        history = water_readings[water_readings["station_id"] == station_id].sort_values("timestamp")
        st.line_chart(history.set_index("timestamp")[["level_m"]])
        st.dataframe(history.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)

with bulletin_tab:
    if bulletin_df.empty:
        st.info("No qualitative hydrological bulletin is currently available.")
    else:
        display = bulletin_df.copy()
        display["issued"] = display["issued_at"].apply(format_time)
        st.dataframe(
            display[[
                "source_name", "basin_name", "river_system", "issued", "observed_rainfall",
                "forecast_rainfall", "forecast_water_level", "status", "is_cached", "source_url", "notes",
            ]],
            use_container_width=True,
            hide_index=True,
        )
        st.warning("Bulletins are qualitative basin forecasts. They are not numerical gauge readings.")

with basin_tab:
    display = hazard_df.copy()
    for column in ["forecast_rain_24h_mm", "forecast_rain_72h_mm", "antecedent_rain_72h_mm", "effective_rain_mm", "hazard_ratio", "max_level_m", "max_rise_rate_m_hr"]:
        if column in display:
            display[column] = pd.to_numeric(display[column], errors="coerce").round(2)
    st.dataframe(
        display[[
            "basin_name", "region", "forecast_rain_24h_mm", "forecast_rain_72h_mm",
            "antecedent_rain_72h_mm", "effective_rain_mm", "hazard_ratio", "hazard_level",
            "water_status", "station_count", "max_level_m", "max_rise_rate_m_hr", "combined_level",
        ]],
        use_container_width=True,
        hide_index=True,
    )

st.markdown("---")
col_a, col_b = st.columns(2)
with col_a:
    if OFFICIAL_REPORT_TEMPLATE.exists():
        st.download_button(
            "Download official-report CSV template",
            data=OFFICIAL_REPORT_TEMPLATE.read_bytes(),
            file_name="official_report_template.csv",
            mime="text/csv",
        )
with col_b:
    if not station_df.empty:
        st.download_button(
            "Download normalized current readings",
            data=station_df.to_csv(index=False).encode("utf-8"),
            file_name="normalized_current_water_readings.csv",
            mime="text/csv",
        )

st.markdown(
    """
**Operational limitations**

- PhilSensors and the PAGASA PMT integration read public webpages and may require parser maintenance when a site changes.
- Marikina and Tullahan values are station elevation readings; compare them only against thresholds for the same station.
- Bulacan PDRRMO sometimes publishes `No Record`; cached values are clearly marked and must not be treated as current.
- The LLDA provider represents the lake-wide Laguna de Bay water-surface level. It must not be relabelled as a Victoria, Pagsanjan, San Juan, or other tributary-river measurement.
- Abra, Panay, Cagayan de Oro and Davao are shown as PAGASA basin forecasts unless an explicit numerical station is available. Samar is shown from an extracted PAGASA regional advisory when one mentioning Samar is visible.
- Official LGU reports imported from CSV remain a separate report type and are not silently treated as instrument measurements.
- This is an academic screening dashboard, not a replacement for official evacuation or flood-warning instructions.
"""
)
