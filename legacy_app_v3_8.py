import html
import io
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

from philsensors_scraper import (
    DEFAULT_URL as PHILSENSORS_URL,
    fetch_philsensors_readings,
    merge_station_registry,
    registry_template_from_readings,
)

st.set_page_config(
    page_title="PH River Basin Rainfall + Water-Level Monitor",
    page_icon="🌊",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_GEOJSON = DATA_DIR / "major_river_basins_simplified.geojson"
DEFAULT_SAMPLE = DATA_DIR / "sample_basin_rainfall.csv"
DEFAULT_TYPHOON = DATA_DIR / "sample_typhoon_track.csv"
DEFAULT_WATER_SAMPLE = DATA_DIR / "sample_water_levels.csv"
DEFAULT_STATION_REGISTRY = DATA_DIR / "philsensors_station_registry.csv"
PHILSENSORS_BACKUP = Path(__file__).parent / ".cache" / "philsensors_last_success.csv"

PH_CENTER = [12.7, 122.3]
MAP_ZOOM = 5.5
MANILA_TZ = "Asia/Manila"
REQUEST_TIMEOUT = 25
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
DASHBOARD_BUILD = "3.8.0"

RAIN_LEVELS = ["Low", "Moderate", "High", "Severe", "Extreme", "No Data"]
WATER_LEVELS = ["Normal", "Alert", "Alarm", "Critical", "Stale", "Offline", "No Data"]


# -----------------------------
# GENERIC HELPERS
# -----------------------------
def secret_value(name: str, default: str = "") -> str:
    """Read an optional Streamlit secret without failing when secrets.toml is absent."""
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    return str(value) if value is not None else default


def fetch_json(
    url: str,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
) -> Any:
    response = requests.get(
        url,
        headers=headers or {},
        params=params or {},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def format_timestamp(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(MANILA_TZ).strftime("%Y-%m-%d %I:%M %p")


@st.cache_data(show_spinner=False, max_entries=4)
def fetch_philsensors_cached(
    url: str,
    refresh_bucket: int,
    backup_path: str,
    timeout_ms: int,
):
    # refresh_bucket intentionally changes every selected refresh interval.
    del refresh_bucket
    return fetch_philsensors_readings(
        url=url,
        backup_path=backup_path,
        timeout_ms=timeout_ms,
    )


def current_refresh_bucket(minutes: int) -> int:
    interval_seconds = max(int(minutes), 1) * 60
    return int(pd.Timestamp.now(tz="UTC").timestamp() // interval_seconds)


# -----------------------------
# BASIN DATA LOADING
# -----------------------------
@st.cache_data(show_spinner=False)
def load_geojson(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def normalize_geojson(uploaded_file) -> dict:
    if uploaded_file is None:
        return load_geojson(str(DEFAULT_GEOJSON))
    return json.load(uploaded_file)


@st.cache_data(show_spinner=False)
def geojson_to_basin_df(geojson_dict: dict) -> pd.DataFrame:
    rows = []
    for feature in geojson_dict.get("features", []):
        properties = feature.get("properties", {})
        rows.append(
            {
                "basin_name": properties.get("basin_name", "Unknown Basin"),
                "region": properties.get("region", ""),
                "lat": pd.to_numeric(properties.get("lat"), errors="coerce"),
                "lon": pd.to_numeric(properties.get("lon"), errors="coerce"),
                "threshold24_mm": pd.to_numeric(
                    properties.get("threshold24_mm", 100), errors="coerce"
                ),
                "threshold72_mm": pd.to_numeric(
                    properties.get("threshold72_mm", 200), errors="coerce"
                ),
            }
        )

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return pd.DataFrame(
            columns=[
                "basin_name",
                "region",
                "lat",
                "lon",
                "threshold24_mm",
                "threshold72_mm",
            ]
        )

    dataframe["threshold24_mm"] = dataframe["threshold24_mm"].fillna(100.0)
    dataframe["threshold72_mm"] = dataframe["threshold72_mm"].fillna(200.0)
    return dataframe


# -----------------------------
# RAINFALL DATA
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_openmeteo_forecast(lat: float, lon: float) -> dict:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "precipitation",
        "forecast_days": 7,
        "timezone": MANILA_TZ,
    }
    return fetch_json(url, params=params)


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_openmeteo_historical(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> dict:
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "precipitation",
        "timezone": MANILA_TZ,
    }
    return fetch_json(url, params=params)


def summarize_precip_from_hourly(payload: dict) -> tuple[float, float]:
    hourly = payload.get("hourly", {})
    values = pd.to_numeric(
        pd.Series(hourly.get("precipitation", [])), errors="coerce"
    ).fillna(0.0)
    array = values.to_numpy()
    if array.size == 0:
        return np.nan, np.nan
    return float(np.nansum(array[:24])), float(np.nansum(array[:72]))


def summarize_historical_72h(payload: dict) -> float:
    hourly = payload.get("hourly", {})
    values = pd.to_numeric(
        pd.Series(hourly.get("precipitation", [])), errors="coerce"
    ).fillna(0.0)
    array = values.to_numpy()
    if array.size == 0:
        return np.nan
    return float(np.nansum(array[-72:]))


@st.cache_data(show_spinner=False, ttl=60 * 30)
def get_live_forecast_for_basins(basin_df_json: str):
    basin_df = pd.read_json(io.StringIO(basin_df_json))
    if basin_df.empty:
        return pd.DataFrame(), pd.Timestamp.now(tz=MANILA_TZ), []

    output = []
    failures = []
    today = pd.Timestamp.now(tz=MANILA_TZ).date()
    start_date = str(today - pd.Timedelta(days=3))
    end_date = str(today)

    for _, row in basin_df.iterrows():
        basin = row["basin_name"]
        lat = row.get("lat")
        lon = row.get("lon")

        if pd.isna(lat) or pd.isna(lon):
            failures.append(f"{basin}: missing centroid coordinates")
            output.append(
                {
                    "basin_name": basin,
                    "forecast_rain_24h_mm": np.nan,
                    "forecast_rain_72h_mm": np.nan,
                    "antecedent_rain_72h_mm": np.nan,
                }
            )
            continue

        try:
            forecast = fetch_openmeteo_forecast(float(lat), float(lon))
            historical = fetch_openmeteo_historical(
                float(lat), float(lon), start_date, end_date
            )
            rain24, rain72 = summarize_precip_from_hourly(forecast)
            antecedent72 = summarize_historical_72h(historical)
            output.append(
                {
                    "basin_name": basin,
                    "forecast_rain_24h_mm": rain24,
                    "forecast_rain_72h_mm": rain72,
                    "antecedent_rain_72h_mm": antecedent72,
                }
            )
        except Exception as exc:
            failures.append(f"{basin}: {type(exc).__name__}: {exc}")
            output.append(
                {
                    "basin_name": basin,
                    "forecast_rain_24h_mm": np.nan,
                    "forecast_rain_72h_mm": np.nan,
                    "antecedent_rain_72h_mm": np.nan,
                }
            )

    return pd.DataFrame(output), pd.Timestamp.now(tz=MANILA_TZ), failures


def prepare_sample_data(sample_df: pd.DataFrame) -> pd.DataFrame:
    renamed = sample_df.copy()
    expected = {
        "forecast_rain_24h_mm": 0.0,
        "forecast_rain_72h_mm": 0.0,
        "antecedent_rain_72h_mm": 0.0,
        "river_stage_factor": 1.0,
        "dam_release_factor": 1.0,
    }
    renamed = renamed.rename(
        columns={
            "rain_mm_24h": "forecast_rain_24h_mm",
            "rain_mm_72h": "forecast_rain_72h_mm",
        }
    )
    for column, default in expected.items():
        if column not in renamed.columns:
            renamed[column] = default
    return renamed


def safe_merge_master_with_values(
    basin_master: pd.DataFrame,
    basin_values: pd.DataFrame,
) -> pd.DataFrame:
    master_columns = [
        "basin_name",
        "region",
        "lat",
        "lon",
        "threshold24_mm",
        "threshold72_mm",
    ]
    value_columns = [
        column for column in basin_values.columns if column not in master_columns[1:]
    ]
    merged = basin_master[master_columns].merge(
        basin_values[value_columns],
        on="basin_name",
        how="left",
    )

    defaults = {
        "forecast_rain_24h_mm": np.nan,
        "forecast_rain_72h_mm": np.nan,
        "antecedent_rain_72h_mm": 0.0,
        "river_stage_factor": 1.0,
        "dam_release_factor": 1.0,
    }
    for column, default in defaults.items():
        if column not in merged.columns:
            merged[column] = default
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    return merged


# -----------------------------
# RAINFALL HAZARD LOGIC
# -----------------------------
def rain_grade_color(level: str) -> str:
    return {
        "Low": "#2ecc71",
        "Moderate": "#f1c40f",
        "High": "#e67e22",
        "Severe": "#e74c3c",
        "Extreme": "#8e0000",
        "No Data": "#9aa0a6",
    }.get(level, "#9aa0a6")


def compute_effective_rain(row: pd.Series, window: str = "24h"):
    rain = (
        row["forecast_rain_24h_mm"]
        if window == "24h"
        else row["forecast_rain_72h_mm"]
    )
    threshold = (
        row["threshold24_mm"] if window == "24h" else row["threshold72_mm"]
    )

    if pd.isna(rain) or pd.isna(threshold) or threshold <= 0:
        return np.nan, np.nan, threshold

    antecedent72 = row.get("antecedent_rain_72h_mm", 0.0)
    river_factor = row.get("river_stage_factor", 1.0)
    dam_factor = row.get("dam_release_factor", 1.0)

    antecedent_factor = 1 + min(
        max((0 if pd.isna(antecedent72) else antecedent72) / 300.0, 0),
        0.35,
    )
    river_factor = 1.0 if pd.isna(river_factor) else river_factor
    dam_factor = 1.0 if pd.isna(dam_factor) else dam_factor

    effective_rain = (
        float(rain)
        * float(antecedent_factor)
        * float(river_factor)
        * float(dam_factor)
    )
    ratio = effective_rain / float(threshold)
    return effective_rain, ratio, threshold


def classify_rain_ratio(ratio: float) -> str:
    if pd.isna(ratio):
        return "No Data"
    if ratio >= 1.50:
        return "Extreme"
    if ratio >= 1.00:
        return "Severe"
    if ratio >= 0.70:
        return "High"
    if ratio >= 0.40:
        return "Moderate"
    return "Low"


def add_hazard_columns(df: pd.DataFrame, window: str = "24h") -> pd.DataFrame:
    output = df.copy()
    computed = output.apply(
        lambda row: compute_effective_rain(row, window=window),
        axis=1,
        result_type="expand",
    )
    output["effective_rain_mm"] = computed[0]
    output["hazard_ratio"] = computed[1]
    output["threshold_selected_mm"] = computed[2]
    output["selected_window"] = window
    output["hazard_level"] = output["hazard_ratio"].apply(classify_rain_ratio)
    return output


def advisory_text(row: pd.Series) -> str:
    basin = row["basin_name"]
    level = row["hazard_level"]
    window = row["selected_window"]
    rain = row["effective_rain_mm"]
    threshold = row["threshold_selected_mm"]

    if level in ["Extreme", "Severe"]:
        return (
            f"{level} rainfall hazard for {basin} River Basin. Effective {window} "
            f"rainfall is {rain:.1f} mm versus threshold {threshold:.1f} mm. "
            "Possible flooding in low-lying and flood-prone communities. "
            "Coordinate with LGUs and monitor official advisories."
        )
    if level == "High":
        return (
            f"High rainfall hazard for {basin} River Basin. Effective {window} "
            f"rainfall is {rain:.1f} mm. Localized flooding is possible in "
            "vulnerable areas."
        )
    if level == "Moderate":
        return (
            f"Moderate rainfall hazard for {basin} River Basin. Continue "
            "monitoring rainfall and observed river conditions."
        )
    if level == "No Data":
        return f"No forecast data available for {basin} River Basin at the moment."
    return f"Low rainfall hazard currently indicated for {basin} River Basin."


# -----------------------------
# WATER-LEVEL DATA
# -----------------------------
def make_demo_water_level_data() -> pd.DataFrame:
    """Fallback data so Version 3 can run even before an API or CSV is connected."""
    now = pd.Timestamp.now(tz="UTC").floor("10min")
    records = [
        ("AGNO-001", "Agno River Station 1", "Agno", 16.05, 120.37, 2.75, 3.0, 4.0, 5.0),
        ("ABRA-001", "Abra River Station 1", "Abra", 17.20, 120.48, 2.20, 3.0, 4.0, 5.0),
        ("PAMP-001", "Pampanga River Station 1", "Pampanga", 15.10, 120.95, 1.40, 2.5, 3.5, 4.5),
        ("BICOL-001", "Bicol River Station 1", "Bicol", 13.55, 123.20, 1.80, 2.5, 3.5, 4.5),
        ("CAG-001", "Cagayan River Station 1", "Cagayan", 17.65, 121.75, 2.10, 3.0, 4.0, 5.0),
    ]
    rows = []
    for station_id, station_name, basin, lat, lon, latest, alert, alarm, critical in records:
        rows.extend(
            [
                {
                    "station_id": station_id,
                    "station_name": station_name,
                    "basin_name": basin,
                    "lat": lat,
                    "lon": lon,
                    "timestamp": now - pd.Timedelta(hours=1),
                    "level_m": max(latest - 0.12, 0),
                    "alert_m": alert,
                    "alarm_m": alarm,
                    "critical_m": critical,
                },
                {
                    "station_id": station_id,
                    "station_name": station_name,
                    "basin_name": basin,
                    "lat": lat,
                    "lon": lon,
                    "timestamp": now,
                    "level_m": latest,
                    "alert_m": alert,
                    "alarm_m": alarm,
                    "critical_m": critical,
                },
            ]
        )
    return pd.DataFrame(rows)


def extract_api_records(payload: Any) -> list[dict]:
    """Accept common JSON layouts: list, data, results, readings, or stations."""
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    if isinstance(payload, dict):
        for key in ["data", "results", "readings", "stations", "items"]:
            records = payload.get(key)
            if isinstance(records, list):
                return [record for record in records if isinstance(record, dict)]
        if all(not isinstance(value, (list, dict)) for value in payload.values()):
            return [payload]
    raise ValueError(
        "Unsupported API JSON. Expected a list or an object containing data/results/readings/stations/items."
    )


@st.cache_data(show_spinner=False, ttl=60)
def fetch_water_level_api(
    url: str,
    api_key: str = "",
    api_key_header: str = "Authorization",
    bearer_prefix: str = "Bearer",
) -> pd.DataFrame:
    headers: dict[str, str] = {}
    if api_key:
        header_value = f"{bearer_prefix} {api_key}".strip() if bearer_prefix else api_key
        headers[api_key_header] = header_value
    payload = fetch_json(url, headers=headers)
    return pd.DataFrame(extract_api_records(payload))


def normalize_water_level_columns(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize observations while allowing unmapped PhilSensors stations.

    Coordinates and basin names are optional at this stage. Unmapped stations
    remain visible in the station table but are excluded from basin aggregation
    and map markers until a station registry supplies those fields.
    """
    dataframe = raw_df.copy()
    normalized_names = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in dataframe.columns
    }

    aliases = {
        "station_id": ["station_id", "stationid", "id", "sensor_id", "device_id"],
        "station_name": ["station_name", "station", "name", "site_name", "location_name"],
        "basin_name": ["basin_name", "basin", "river_basin", "riverbasin"],
        "lat": ["lat", "latitude", "station_lat", "y"],
        "lon": ["lon", "lng", "longitude", "station_lon", "x"],
        "timestamp": ["timestamp", "datetime", "date_time", "observed_at", "time", "date"],
        "level_m": ["level_m", "water_level_m", "water_level", "stage_m", "river_stage_m", "value"],
        "rise_rate_m_hr": ["rise_rate_m_hr", "rate_m_hr", "rise_rate", "change_m_hr"],
        "alert_m": ["alert_m", "alert_level_m", "alert_level"],
        "alarm_m": ["alarm_m", "alarm_level_m", "alarm_level"],
        "critical_m": ["critical_m", "critical_level_m", "critical_level"],
        "threshold_status": ["threshold_status", "source_status", "official_status"],
        "source_trend": ["source_trend", "trend"],
        "region": ["region"],
        "province": ["province"],
        "location": ["location"],
        "source_name": ["source_name", "data_source"],
        "scraped_at": ["scraped_at", "retrieved_at"],
    }

    rename_map: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            original = normalized_names.get(candidate)
            if original is not None:
                rename_map[original] = canonical
                break
    dataframe = dataframe.rename(columns=rename_map)

    if "station_id" not in dataframe.columns:
        if "station_name" in dataframe.columns:
            dataframe["station_id"] = dataframe["station_name"].astype(str)
        else:
            dataframe["station_id"] = [
                f"STATION-{index + 1:03d}" for index in range(len(dataframe))
            ]

    if "station_name" not in dataframe.columns:
        dataframe["station_name"] = dataframe["station_id"].astype(str)

    required = ["timestamp", "level_m"]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError("Missing required water-level columns: " + ", ".join(missing))

    for column in [
        "lat",
        "lon",
        "level_m",
        "rise_rate_m_hr",
        "alert_m",
        "alarm_m",
        "critical_m",
    ]:
        if column not in dataframe.columns:
            dataframe[column] = np.nan
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    for column in [
        "basin_name",
        "threshold_status",
        "source_trend",
        "region",
        "province",
        "location",
        "source_name",
    ]:
        if column not in dataframe.columns:
            dataframe[column] = np.nan

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"], errors="coerce", utc=True
    )
    if "scraped_at" in dataframe.columns:
        dataframe["scraped_at"] = pd.to_datetime(
            dataframe["scraped_at"], errors="coerce", utc=True
        )
    else:
        dataframe["scraped_at"] = pd.NaT

    dataframe["station_id"] = dataframe["station_id"].astype(str).str.strip()
    dataframe["station_name"] = dataframe["station_name"].astype(str).str.strip()
    dataframe["basin_name"] = dataframe["basin_name"].where(
        dataframe["basin_name"].notna(), np.nan
    )
    dataframe.loc[
        dataframe["basin_name"].astype(str).str.strip().str.lower().isin(["", "nan", "none"]),
        "basin_name",
    ] = np.nan

    return dataframe.dropna(subset=["station_id", "timestamp", "level_m"])


def apply_threshold_defaults(
    dataframe: pd.DataFrame,
    default_alert: float,
    default_alarm: float,
    default_critical: float,
) -> pd.DataFrame:
    output = dataframe.copy()
    output["alert_m"] = output["alert_m"].fillna(default_alert)
    output["alarm_m"] = output["alarm_m"].fillna(default_alarm)
    output["critical_m"] = output["critical_m"].fillna(default_critical)

    invalid = ~(
        (output["alert_m"] < output["alarm_m"])
        & (output["alarm_m"] < output["critical_m"])
    )
    output.loc[invalid, "alert_m"] = default_alert
    output.loc[invalid, "alarm_m"] = default_alarm
    output.loc[invalid, "critical_m"] = default_critical
    return output


def compute_latest_station_state(
    all_readings: pd.DataFrame,
    stale_minutes: int,
    offline_minutes: int,
    rapid_rise_m_hr: float,
    rapid_fall_m_hr: float,
) -> pd.DataFrame:
    if all_readings.empty:
        return pd.DataFrame()

    dataframe = all_readings.sort_values(["station_id", "timestamp"]).copy()
    latest_rows = []
    now_utc = pd.Timestamp.now(tz="UTC")

    for _, group in dataframe.groupby("station_id", sort=False):
        group = group.sort_values("timestamp")
        latest = group.iloc[-1].copy()

        rate = latest.get("rise_rate_m_hr", np.nan)
        if pd.isna(rate) and len(group) >= 2:
            previous = group.iloc[-2]
            elapsed_hours = (
                latest["timestamp"] - previous["timestamp"]
            ).total_seconds() / 3600.0
            if elapsed_hours > 0:
                rate = (latest["level_m"] - previous["level_m"]) / elapsed_hours
        latest["rise_rate_m_hr"] = rate

        age_minutes = (now_utc - latest["timestamp"]).total_seconds() / 60.0
        latest["age_minutes"] = max(age_minutes, 0.0)

        source_status = str(latest.get("threshold_status", "")).strip().title()
        if latest["age_minutes"] > offline_minutes:
            status = "Offline"
        elif latest["age_minutes"] > stale_minutes:
            status = "Stale"
        elif source_status in ["Normal", "Alert", "Alarm", "Critical"]:
            # Prefer the status displayed by PhilSensors when it is available.
            status = source_status
        elif (
            pd.notna(latest.get("alert_m"))
            and pd.notna(latest.get("alarm_m"))
            and pd.notna(latest.get("critical_m"))
        ):
            if latest["level_m"] >= latest["critical_m"]:
                status = "Critical"
            elif latest["level_m"] >= latest["alarm_m"]:
                status = "Alarm"
            elif latest["level_m"] >= latest["alert_m"]:
                status = "Alert"
            else:
                status = "Normal"
        else:
            status = "No Data"

        latest["water_status"] = status
        usable_for_trend = status not in ["Stale", "Offline", "No Data"]
        latest["rapid_rise"] = bool(
            pd.notna(rate)
            and float(rate) >= rapid_rise_m_hr
            and usable_for_trend
        )
        latest["rapid_fall"] = bool(
            pd.notna(rate)
            and float(rate) <= -abs(rapid_fall_m_hr)
            and usable_for_trend
        )
        if latest["rapid_rise"]:
            latest["trend_label"] = "Rapid rise"
        elif latest["rapid_fall"]:
            latest["trend_label"] = "Rapid fall"
        elif pd.notna(rate) and float(rate) > 0.005:
            latest["trend_label"] = "Rising"
        elif pd.notna(rate) and float(rate) < -0.005:
            latest["trend_label"] = "Falling"
        else:
            latest["trend_label"] = "Stable"
        latest_rows.append(latest)

    return pd.DataFrame(latest_rows).reset_index(drop=True)


def water_status_rank(status: str) -> int:
    return {
        "No Data": -1,
        "Normal": 0,
        "Stale": 0,
        "Offline": 0,
        "Alert": 1,
        "Alarm": 2,
        "Critical": 3,
    }.get(status, -1)


def water_status_color(status: str) -> str:
    return {
        "Normal": "#2ecc71",
        "Alert": "#f1c40f",
        "Alarm": "#e67e22",
        "Critical": "#e74c3c",
        "Stale": "#9aa0a6",
        "Offline": "#333333",
        "No Data": "#c4c7c5",
    }.get(status, "#c4c7c5")


def aggregate_water_by_basin(stations: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "basin_name",
        "water_status",
        "station_count",
        "active_station_count",
        "max_level_m",
        "max_rise_rate_m_hr",
        "rapid_rise_count",
        "latest_water_timestamp",
    ]
    if stations.empty:
        return pd.DataFrame(columns=columns)

    mapped = stations.dropna(subset=["basin_name"]).copy()
    mapped = mapped[mapped["basin_name"].astype(str).str.strip() != ""]
    if mapped.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for basin, group in mapped.groupby("basin_name"):
        active = group[~group["water_status"].isin(["Stale", "Offline"])]
        if not active.empty:
            worst_row = active.loc[
                active["water_status"].map(water_status_rank).idxmax()
            ]
            basin_status = worst_row["water_status"]
        elif (group["water_status"] == "Stale").any():
            basin_status = "Stale"
        elif (group["water_status"] == "Offline").any():
            basin_status = "Offline"
        else:
            basin_status = "No Data"

        rows.append(
            {
                "basin_name": basin,
                "water_status": basin_status,
                "station_count": int(len(group)),
                "active_station_count": int(len(active)),
                "max_level_m": pd.to_numeric(group["level_m"], errors="coerce").max(),
                "max_rise_rate_m_hr": pd.to_numeric(
                    group["rise_rate_m_hr"], errors="coerce"
                ).max(),
                "rapid_rise_count": int(group["rapid_rise"].sum()),
                "latest_water_timestamp": group["timestamp"].max(),
            }
        )
    return pd.DataFrame(rows)


def combined_level(rain_level: str, water_status: str) -> str:
    rain_rank = {
        "No Data": -1,
        "Low": 0,
        "Moderate": 1,
        "High": 2,
        "Severe": 3,
        "Extreme": 4,
    }.get(rain_level, -1)
    water_rank = {
        "No Data": -1,
        "Normal": 0,
        "Stale": -1,
        "Offline": -1,
        "Alert": 1,
        "Alarm": 3,
        "Critical": 4,
    }.get(water_status, -1)
    rank = max(rain_rank, water_rank)
    return {
        -1: "No Data",
        0: "Low",
        1: "Moderate",
        2: "High",
        3: "Severe",
        4: "Extreme",
    }[rank]



# -----------------------------
# PROVINCE REFERENCE + WATER MAP
# -----------------------------
def normalize_province_key(value: Any) -> str:
    """Normalize PhilSensors and DENR province names for resilient matching."""
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
        "NORTH COTABATO": "COTABATO",
        "COMPOSTELA VALLEY": "DAVAO DE ORO",
        "WESTERN SAMAR": "SAMAR",
        "DINAGAT ISLAND": "DINAGAT ISLANDS",
        "MINDORO OCCIDENTAL": "OCCIDENTAL MINDORO",
        "MINDORO ORIENTAL": "ORIENTAL MINDORO",
    }
    return aliases.get(text, text)


def _ring_area_centroid(ring: list) -> tuple[float, float, float] | None:
    points = []
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


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_province_reference() -> tuple[dict, pd.DataFrame, str]:
    """Load province polygons and derive one representative pin per province.

    Two public Philippine government ArcGIS services are tried in sequence so the
    water-level trend map does not disappear when one service is temporarily down.
    """
    errors: list[str] = []
    for source_name, layer_url in PROVINCE_LAYER_URLS:
        try:
            query_url = f"{layer_url}/query"
            params = {
                "where": "1=1",
                "outFields": "*",
                "returnGeometry": "true",
                "outSR": "4326",
                "f": "geojson",
            }
            response = requests.get(query_url, params=params, timeout=45)
            response.raise_for_status()
            province_geojson = response.json()
            if province_geojson.get("type") != "FeatureCollection":
                raise ValueError("service did not return a GeoJSON FeatureCollection")

            rows = []
            cleaned_features = []
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

                properties["province"] = str(province).strip()
                properties["region"] = str(region).strip()
                properties["province_key"] = normalize_province_key(province)
                point = geometry_representative_point(feature.get("geometry"))
                if point is None:
                    continue
                latitude, longitude = point
                properties["province_lat"] = latitude
                properties["province_lon"] = longitude
                rows.append(
                    {
                        "province_ref": str(province).strip(),
                        "province_key": normalize_province_key(province),
                        "region_ref": str(region).strip(),
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

            # Older PhilSensors records can still use the undivided Maguindanao name.
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
    station_df: pd.DataFrame,
    province_centroids: pd.DataFrame,
) -> pd.DataFrame:
    output = station_df.copy()
    if output.empty:
        return output
    output["province_key"] = output["province"].apply(normalize_province_key)
    reference_keys = province_centroids["province_key"].dropna().astype(str).tolist()
    reference_set = set(reference_keys)

    match_cache: dict[str, str | None] = {}
    for key in output["province_key"].dropna().unique():
        if not key:
            match_cache[key] = None
            continue
        if key in reference_set:
            match_cache[key] = key
            continue
        candidates = [
            (SequenceMatcher(None, key, reference_key).ratio(), reference_key)
            for reference_key in reference_keys
        ]
        best_ratio, best_key = max(candidates, default=(0.0, ""))
        match_cache[key] = best_key if best_ratio >= 0.84 else None

    output["province_ref_key"] = output["province_key"].map(match_cache)
    output = output.merge(
        province_centroids,
        left_on="province_ref_key",
        right_on="province_key",
        how="left",
        suffixes=("", "_reference"),
    )
    return output


def trend_symbol(row: pd.Series) -> str:
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


def province_map_summary(stations: pd.DataFrame) -> pd.DataFrame:
    if stations.empty:
        return pd.DataFrame()
    rows = []
    valid = stations.dropna(subset=["province_lat", "province_lon"]).copy()
    for province_key, group in valid.groupby("province_ref_key", dropna=True):
        active = group[~group["water_status"].isin(["Stale", "Offline", "No Data"])]
        status_source = active if not active.empty else group
        worst_index = status_source["water_status"].map(water_status_rank).idxmax()
        worst_status = status_source.loc[worst_index, "water_status"]
        rates = pd.to_numeric(group["rise_rate_m_hr"], errors="coerce")
        max_rise = rates.max() if rates.notna().any() else np.nan
        max_fall = rates.min() if rates.notna().any() else np.nan
        dominant_rate = np.nan
        if rates.notna().any():
            dominant_rate = rates.loc[rates.abs().idxmax()]
        rows.append(
            {
                "province_ref_key": province_key,
                "province_ref": group["province_ref"].dropna().iloc[0],
                "province_lat": group["province_lat"].dropna().iloc[0],
                "province_lon": group["province_lon"].dropna().iloc[0],
                "station_count": len(group),
                "active_count": len(active),
                "worst_status": worst_status,
                "rapid_rise_count": int(
                    group["rapid_rise"].fillna(False).sum()
                    if "rapid_rise" in group.columns
                    else 0
                ),
                "rapid_fall_count": int(
                    group["rapid_fall"].fillna(False).sum()
                    if "rapid_fall" in group.columns
                    else 0
                ),
                "max_rise_m_hr": max_rise,
                "max_fall_m_hr": max_fall,
                "dominant_rate_m_hr": dominant_rate,
                "latest_timestamp": group["timestamp"].max(),
            }
        )
    return pd.DataFrame(rows)


def station_popup_table(group: pd.DataFrame, max_rows: int = 30) -> str:
    display = group.copy()
    display["_active_sort"] = ~display["water_status"].isin(["Stale", "Offline", "No Data"])
    display["_rate_abs"] = pd.to_numeric(display["rise_rate_m_hr"], errors="coerce").abs()
    display = display.sort_values(
        ["_active_sort", "_rate_abs", "timestamp"],
        ascending=[False, False, False],
    )
    shown = display.head(max_rows)
    rows = []
    for _, row in shown.iterrows():
        rate = row.get("rise_rate_m_hr")
        rate_text = f"{float(rate):+.3f}" if pd.notna(rate) else "-"
        level = row.get("level_m")
        level_text = f"{float(level):.3f}" if pd.notna(level) else "-"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('station_name', '-')))}</td>"
            f"<td>{level_text}</td>"
            f"<td>{rate_text}</td>"
            f"<td>{trend_symbol(row)} {html.escape(str(row.get('trend_label', '-')))}</td>"
            f"<td>{html.escape(str(row.get('water_status', '-')))}</td>"
            f"<td>{html.escape(str(row.get('threshold_status', '-')))}</td>"
            f"<td>{html.escape(format_timestamp(row.get('timestamp')))}</td>"
            "</tr>"
        )
    remaining = max(len(display) - len(shown), 0)
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


def province_trend_category(summary_row: pd.Series) -> str:
    rise_count = int(summary_row.get("rapid_rise_count", 0) or 0)
    fall_count = int(summary_row.get("rapid_fall_count", 0) or 0)
    rate = summary_row.get("dominant_rate_m_hr")
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


def province_trend_color(category: str) -> str:
    return {
        "Rapid rise": "#b91c1c",
        "Rising": "#f97316",
        "Stable": "#6b7280",
        "Falling": "#3b82f6",
        "Rapid fall": "#1d4ed8",
        "Mixed rapid change": "#7e22ce",
        "No data": "#d1d5db",
    }.get(category, "#d1d5db")


def merge_province_metrics(province_geojson: dict, summary: pd.DataFrame) -> dict:
    merged = json.loads(json.dumps(to_json_safe(province_geojson)))
    metric_map = (
        summary.set_index("province_ref_key").to_dict(orient="index")
        if not summary.empty
        else {}
    )
    for feature in merged.get("features", []):
        properties = feature.setdefault("properties", {})
        key = normalize_province_key(properties.get("province"))
        metrics = metric_map.get(key, {})
        category = metrics.get("trend_category", "No data")
        properties["trend_category"] = category
        properties["dominant_rate_m_hr"] = metrics.get("dominant_rate_m_hr")
        properties["max_rise_m_hr"] = metrics.get("max_rise_m_hr")
        properties["max_fall_m_hr"] = metrics.get("max_fall_m_hr")
        properties["station_count"] = metrics.get("station_count", 0)
        properties["worst_status"] = metrics.get("worst_status", "No Data")
    return json.loads(json.dumps(to_json_safe(merged), allow_nan=False))


def build_province_water_map(
    stations: pd.DataFrame,
    province_geojson: dict | None,
    include_inactive: bool,
    only_rapid: bool = False,
):
    map_object = folium.Map(
        location=PH_CENTER,
        zoom_start=MAP_ZOOM,
        tiles="cartodbpositron",
    )

    mapped = stations.dropna(
        subset=["province_ref_key", "province_lat", "province_lon"]
    ).copy()
    if not include_inactive:
        mapped = mapped[~mapped["water_status"].isin(["Stale", "Offline", "No Data"])]
    if only_rapid:
        mapped = mapped[
            mapped.get("rapid_rise", False).fillna(False)
            | mapped.get("rapid_fall", False).fillna(False)
        ]

    summary = province_map_summary(mapped)
    if not summary.empty:
        summary["trend_category"] = summary.apply(province_trend_category, axis=1)

    if province_geojson:
        merged_provinces = merge_province_metrics(province_geojson, summary)
        folium.GeoJson(
            merged_provinces,
            name="Province trend shading",
            style_function=lambda feature: {
                "fillColor": province_trend_color(
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
        marker_color = province_trend_color(category)
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
            f"<div style='min-width:650px;max-width:780px'>"
            f"<h4 style='margin:0 0 6px'>{html.escape(str(summary_row['province_ref']))}</h4>"
            f"<b>Visible trend:</b> {html.escape(category)}<br>"
            f"<b>Strongest change:</b> {rate_label} m/hr<br>"
            f"<b>Worst current status:</b> {html.escape(str(summary_row['worst_status']))}<br>"
            f"<b>Stations shown:</b> {int(summary_row['station_count'])} "
            f"({int(summary_row['active_count'])} active)<br>"
            f"<b>Latest province reading:</b> {html.escape(format_timestamp(summary_row['latest_timestamp']))}"
            "<hr style='margin:8px 0'>"
        )
        popup_html = popup_header + station_popup_table(group) + "</div>"
        tooltip = (
            f"Province: {summary_row['province_ref']} | {category} | "
            f"{rate_label} m/hr | {int(summary_row['station_count'])} station(s)"
        )

        # A solid circle remains visible even if the HTML label is temporarily clipped.
        folium.CircleMarker(
            location=[summary_row["province_lat"], summary_row["province_lon"]],
            radius=10,
            color="#111827",
            weight=2,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.95,
            tooltip=tooltip,
            popup=folium.Popup(popup_html, max_width=840),
        ).add_to(marker_layer)
        folium.Marker(
            location=[summary_row["province_lat"], summary_row["province_lon"]],
            icon=folium.DivIcon(
                html=marker_html,
                icon_size=(130, 48),
                icon_anchor=(65, 24),
                class_name="water-trend-label",
            ),
            tooltip=tooltip,
            popup=folium.Popup(popup_html, max_width=840),
        ).add_to(marker_layer)

    marker_layer.add_to(map_object)
    folium.LayerControl(collapsed=False).add_to(map_object)
    return map_object, mapped, summary


def get_clicked_province(map_state: dict | None) -> str | None:
    if not map_state:
        return None
    tooltip = map_state.get("last_object_clicked_tooltip")
    if isinstance(tooltip, str) and tooltip.startswith("Province:"):
        return tooltip.split("|", 1)[0].replace("Province:", "", 1).strip()
    return None

# -----------------------------
# MAPS
# -----------------------------
def to_json_safe(value: Any) -> Any:
    """Convert pandas/numpy values to types Folium can serialize to JSON."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, np.generic):
        return to_json_safe(value.item())

    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()

    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return [to_json_safe(item) for item in list(value)]

    # Handle scalar NaN/NaT values without applying pd.isna to containers.
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (str, int, float, bool)):
        return value

    # Last-resort conversion prevents an unexpected extension dtype from
    # crashing the entire Folium map.
    return str(value)


def merge_geojson_with_metrics(geojson_dict: dict, metrics_df: pd.DataFrame) -> dict:
    metric_map = metrics_df.set_index("basin_name").to_dict(orient="index")
    merged = to_json_safe(json.loads(json.dumps(geojson_dict)))
    for feature in merged.get("features", []):
        basin = feature.get("properties", {}).get("basin_name")
        metrics = metric_map.get(basin, {})
        properties = feature.setdefault("properties", {})
        for key, value in metrics.items():
            properties[str(key)] = to_json_safe(value)

    # Validate now so any remaining problem is caught before st_folium renders.
    return json.loads(json.dumps(to_json_safe(merged), allow_nan=False))


def add_typhoon_overlay(map_object, track_df: pd.DataFrame | None):
    if track_df is None or track_df.empty:
        return

    clean = track_df.copy()
    clean["lat"] = pd.to_numeric(clean.get("lat"), errors="coerce")
    clean["lon"] = pd.to_numeric(clean.get("lon"), errors="coerce")
    clean = clean.dropna(subset=["lat", "lon"])
    if clean.empty:
        return

    points = clean[["lat", "lon"]].values.tolist()
    if len(points) >= 2:
        folium.PolyLine(points, weight=3, tooltip="Typhoon Track").add_to(map_object)

    for _, row in clean.iterrows():
        popup = (
            f"<b>{row.get('name', 'Typhoon')}</b><br>"
            f"{row.get('datetime', '')}<br>"
            f"Wind: {row.get('wind_kph', '')} kph"
        )
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            weight=1,
            fill=True,
            fill_opacity=0.9,
            popup=popup,
        ).add_to(map_object)


def add_station_markers(map_object, station_df: pd.DataFrame):
    if station_df is None or station_df.empty:
        return

    mapped = station_df.dropna(subset=["lat", "lon"]).copy()
    if mapped.empty:
        return

    station_layer = folium.FeatureGroup(name="Water-level stations", show=True)
    for _, row in mapped.iterrows():
        rate_text = (
            f"{row['rise_rate_m_hr']:+.3f} m/hr"
            if pd.notna(row.get("rise_rate_m_hr"))
            else "-"
        )
        if row.get("rapid_rise", False):
            trend_text = "Rapid rise detected"
        elif row.get("rapid_fall", False):
            trend_text = "Rapid fall detected"
        else:
            trend_text = str(row.get("trend_label", ""))
        basin_text = row.get("basin_name") if pd.notna(row.get("basin_name")) else "Unmapped"
        threshold_values = [row.get("alert_m"), row.get("alarm_m"), row.get("critical_m")]
        threshold_text = (
            " / ".join(f"{float(value):.2f}" for value in threshold_values) + " m"
            if all(pd.notna(value) for value in threshold_values)
            else "Not loaded"
        )
        source_status = row.get("threshold_status")
        source_status_text = (
            str(source_status) if pd.notna(source_status) else "Not reported"
        )
        popup = (
            f"<b>{row['station_name']}</b><br>"
            f"Station ID: {row['station_id']}<br>"
            f"Basin: {basin_text}<br>"
            f"Water level: {row['level_m']:.3f} m<br>"
            f"Rate of change: {rate_text}<br>"
            f"Dashboard status: <b>{row['water_status']}</b><br>"
            f"PhilSensors threshold class: {source_status_text}<br>"
            f"Alert / Alarm / Critical: {threshold_text}<br>"
            f"Last reading: {format_timestamp(row['timestamp'])}<br>"
            f"Data age: {row['age_minutes']:.0f} min<br>"
            f"<b>{trend_text}</b>"
        )
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=7 if row.get("rapid_rise", False) or row.get("rapid_fall", False) else 6,
            color="#111111",
            weight=2 if row.get("rapid_rise", False) or row.get("rapid_fall", False) else 1,
            fill=True,
            fill_color=water_status_color(row["water_status"]),
            fill_opacity=1.0,
            popup=folium.Popup(popup, max_width=350),
            tooltip=f"{row['station_name']}: {row['level_m']:.2f} m ({row['water_status']})",
        ).add_to(station_layer)
    station_layer.add_to(map_object)


def build_map(
    geojson_merged: dict,
    map_view: str,
    station_df: pd.DataFrame | None = None,
    track_df: pd.DataFrame | None = None,
):
    map_object = folium.Map(
        location=PH_CENTER,
        zoom_start=MAP_ZOOM,
        tiles="cartodbpositron",
    )

    if map_view.startswith("Observed water level"):
        style_property = "water_status"
        color_function = water_status_color
        tooltip_fields = [
            "basin_name",
            "region",
            "water_status",
            "station_count",
            "active_station_count",
            "max_level_m",
            "max_rise_rate_m_hr",
        ]
        tooltip_aliases = [
            "River Basin",
            "Region",
            "Water Status",
            "Stations",
            "Active Stations",
            "Highest Level (m)",
            "Maximum Rise (m/hr)",
        ]
    elif map_view == "Combined monitoring":
        style_property = "combined_level"
        color_function = rain_grade_color
        tooltip_fields = [
            "basin_name",
            "region",
            "combined_level",
            "hazard_level",
            "water_status",
            "effective_rain_mm",
            "max_level_m",
        ]
        tooltip_aliases = [
            "River Basin",
            "Region",
            "Combined Level",
            "Rainfall Hazard",
            "Water Status",
            "Effective Rain (mm)",
            "Highest Water Level (m)",
        ]
    else:
        style_property = "hazard_level"
        color_function = rain_grade_color
        tooltip_fields = [
            "basin_name",
            "region",
            "hazard_level",
            "effective_rain_mm",
            "threshold_selected_mm",
        ]
        tooltip_aliases = [
            "River Basin",
            "Region",
            "Rainfall Hazard",
            "Effective Rain (mm)",
            "Threshold (mm)",
        ]

    def style_function(feature):
        level = feature.get("properties", {}).get(style_property, "No Data")
        return {
            "fillColor": color_function(level),
            "color": "#333333",
            "weight": 1.0,
            "fillOpacity": 0.65,
        }

    folium.GeoJson(
        geojson_merged,
        name="River basins",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            localize=True,
            sticky=False,
        ),
        highlight_function=lambda feature: {"weight": 3, "fillOpacity": 0.8},
    ).add_to(map_object)

    add_station_markers(map_object, station_df if station_df is not None else pd.DataFrame())
    add_typhoon_overlay(map_object, track_df)
    folium.LayerControl().add_to(map_object)
    return map_object


def get_clicked_basin(map_state: dict | None):
    if not map_state:
        return None
    for key in [
        "last_active_drawing",
        "last_object_clicked_tooltip",
        "last_object_clicked_popup",
    ]:
        value = map_state.get(key)
        if isinstance(value, dict):
            properties = value.get("properties", {})
            if properties.get("basin_name"):
                return properties.get("basin_name")
    return None


# -----------------------------
# UI
# -----------------------------
st.title("🌊 Philippine River Basin Rainfall + Water-Level Monitor")
st.caption(
    "Near-real-time basin rainfall screening with optional observed water-level stations."
)

with st.sidebar:
    st.header("Controls")
    st.caption(f"Dashboard build {DASHBOARD_BUILD}")
    app_mode = st.radio("Mode", ["Version 1.1", "Version 2", "Version 3"], index=2)
    rainfall_source = st.radio(
        "Rainfall source",
        ["Sample data", "Live Open-Meteo"],
        index=1,
    )
    rain_window = st.radio("Rainfall window", ["24h", "72h"], horizontal=True)
    auto_refresh_minutes = st.selectbox(
        "Auto-refresh",
        [0, 5, 10, 15, 30, 60],
        index=4,
        format_func=lambda value: "Off" if value == 0 else f"Every {value} minutes",
    )
    uploaded_geojson = st.file_uploader(
        "Optional: upload basin GeoJSON",
        type=["geojson", "json"],
    )

if auto_refresh_minutes:
    st.markdown(
        f"<meta http-equiv='refresh' content='{auto_refresh_minutes * 60}'>",
        unsafe_allow_html=True,
    )

geojson_data = normalize_geojson(uploaded_geojson)
basin_master = geojson_to_basin_df(geojson_data)
if basin_master.empty:
    st.error("No basin features were found in the GeoJSON.")
    st.stop()

rain_last_updated = None
rain_failures: list[str] = []
if rainfall_source == "Sample data":
    basin_values = prepare_sample_data(load_csv(str(DEFAULT_SAMPLE)))
    rain_last_updated = pd.Timestamp.now(tz=MANILA_TZ)
else:
    with st.spinner("Downloading live rainfall forecast by basin centroid..."):
        basin_values, rain_last_updated, rain_failures = get_live_forecast_for_basins(
            basin_master.to_json(orient="records")
        )

basin_data = safe_merge_master_with_values(basin_master, basin_values)
use_typhoon_overlay = False
typhoon_track_df = None

if app_mode in ["Version 2", "Version 3"]:
    with st.sidebar:
        st.subheader("Rainfall modifiers")
        if app_mode == "Version 2":
            apply_stage = st.checkbox("Apply elevated river-stage factor", value=False)
            river_stage_default = st.slider(
                "Manual river-stage factor",
                1.0,
                1.5,
                1.05,
                0.05,
                disabled=not apply_stage,
            )
        else:
            apply_stage = False
            river_stage_default = 1.0
            st.caption(
                "Version 3 keeps observed gauge status separate from the rainfall "
                "calculation instead of using a manual river-stage multiplier."
            )
        apply_dam = st.checkbox("Apply dam-release factor", value=False)
        dam_release_default = st.slider(
            "Manual dam-release factor",
            1.0,
            1.4,
            1.0,
            0.05,
            disabled=not apply_dam,
        )
        use_typhoon_overlay = st.checkbox("Show typhoon-track overlay", value=False)
        upload_typhoon = st.file_uploader("Upload typhoon-track CSV", type=["csv"])

    if apply_stage:
        basin_data["river_stage_factor"] = river_stage_default
    if apply_dam:
        basin_data["dam_release_factor"] = dam_release_default

    if use_typhoon_overlay:
        if upload_typhoon is not None:
            typhoon_track_df = pd.read_csv(upload_typhoon)
        elif DEFAULT_TYPHOON.exists():
            typhoon_track_df = pd.read_csv(DEFAULT_TYPHOON)

hazard_df = add_hazard_columns(basin_data, window=rain_window)
hazard_df["alert_text"] = hazard_df.apply(advisory_text, axis=1)

# Water-level defaults for Versions 1/2
water_readings = pd.DataFrame()
station_df = pd.DataFrame()
basin_water_df = pd.DataFrame()
water_last_updated = None
water_error = None
water_source = ""
map_view = "Rainfall hazard"
philsensors_metadata: dict[str, Any] = {}
philsensors_registry_template = pd.DataFrame()
use_fallback_thresholds = True
overlay_exact_station_markers = False
rapid_fall_m_hr = 0.30

if app_mode == "Version 3":
    with st.sidebar:
        st.subheader("Observed water level")
        water_source = st.radio(
            "Water-level source",
            [
                "PhilSensors public webpage",
                "Sample water-level data",
                "Upload CSV",
                "REST API (JSON)",
            ],
            index=0,
        )

        upload_water = None
        upload_registry = None
        api_url = ""
        philsensors_refresh_minutes = 10
        philsensors_timeout_seconds = 90

        if water_source == "PhilSensors public webpage":
            st.caption(
                "Unofficial automated reading of the public DOST-ASTI table. "
                "No login, cookie, or API key is used."
            )
            philsensors_refresh_minutes = st.selectbox(
                "PhilSensors retrieval interval",
                [5, 10, 15, 30, 60],
                index=1,
                format_func=lambda value: f"Every {value} minutes",
            )
            philsensors_timeout_seconds = st.number_input(
                "Browser timeout (seconds)", 30, 180, 90, 10
            )
            if st.button("Force PhilSensors refresh"):
                fetch_philsensors_cached.clear()

            upload_registry = st.file_uploader(
                "Optional station registry CSV",
                type=["csv"],
                help=(
                    "Adds latitude, longitude, river-basin assignment, and optional "
                    "verified station thresholds. A fillable template is available below."
                ),
            )
            use_fallback_thresholds = st.checkbox(
                "Use generic thresholds when PhilSensors has no threshold class",
                value=False,
                help=(
                    "Keep this off unless the fallback levels are technically justified. "
                    "Gauge elevations are station-specific."
                ),
            )
        elif water_source == "Upload CSV":
            upload_water = st.file_uploader(
                "Upload water-level CSV",
                type=["csv"],
                help=(
                    "Required: timestamp and level_m. Recommended: station_id, "
                    "station_name, basin_name, lat, lon, and station thresholds."
                ),
            )
        elif water_source == "REST API (JSON)":
            api_url = st.text_input(
                "API endpoint",
                value=secret_value("WATER_LEVEL_API_URL", ""),
                placeholder="https://example.gov.ph/api/water-levels",
            )
            st.caption("Store API keys in .streamlit/secrets.toml, not in the code.")

        threshold_controls_disabled = (
            water_source == "PhilSensors public webpage" and not use_fallback_thresholds
        )
        default_alert = st.number_input(
            "Fallback alert level (m)",
            0.0,
            50.0,
            3.0,
            0.1,
            disabled=threshold_controls_disabled,
        )
        default_alarm = st.number_input(
            "Fallback alarm level (m)",
            0.0,
            50.0,
            4.0,
            0.1,
            disabled=threshold_controls_disabled,
        )
        default_critical = st.number_input(
            "Fallback critical level (m)",
            0.0,
            50.0,
            5.0,
            0.1,
            disabled=threshold_controls_disabled,
        )

        stale_default = 90 if water_source == "PhilSensors public webpage" else 30
        offline_default = 240 if water_source == "PhilSensors public webpage" else 120
        stale_minutes = st.number_input(
            "Mark stale after (minutes)", 5, 1440, stale_default, 5
        )
        offline_minutes = st.number_input(
            "Mark offline after (minutes)", 10, 2880, offline_default, 10
        )
        rapid_rise_m_hr = st.number_input(
            "Rapid-rise threshold (m/hour)", 0.01, 5.0, 0.30, 0.05
        )
        rapid_fall_m_hr = st.number_input(
            "Rapid-fall threshold (m/hour)", 0.01, 5.0, 0.30, 0.05
        )
        map_view = st.radio(
            "Basin map view",
            [
                "Rainfall hazard",
                "Observed water level (basin-matched only)",
                "Combined monitoring",
            ],
            index=2,
        )
        overlay_exact_station_markers = st.checkbox(
            "Overlay exact station markers on basin map",
            value=False,
            help=(
                "Keep this off to avoid crowding. The separate province water-level "
                "map below displays all stations with a province name."
            ),
        )

    try:
        if water_source == "PhilSensors public webpage":
            refresh_bucket = current_refresh_bucket(int(philsensors_refresh_minutes))
            scraped_readings, philsensors_metadata = fetch_philsensors_cached(
                PHILSENSORS_URL,
                refresh_bucket,
                str(PHILSENSORS_BACKUP),
                int(philsensors_timeout_seconds * 1000),
            )
            philsensors_registry_template = registry_template_from_readings(
                scraped_readings
            )

            registry_df = pd.DataFrame()
            if upload_registry is not None:
                registry_df = pd.read_csv(upload_registry)
            elif DEFAULT_STATION_REGISTRY.exists():
                registry_df = load_csv(str(DEFAULT_STATION_REGISTRY))

            water_readings = merge_station_registry(scraped_readings, registry_df)
        elif water_source == "Sample water-level data":
            if DEFAULT_WATER_SAMPLE.exists():
                water_readings = load_csv(str(DEFAULT_WATER_SAMPLE))
            else:
                water_readings = make_demo_water_level_data()
        elif water_source == "Upload CSV":
            if upload_water is not None:
                water_readings = pd.read_csv(upload_water)
        else:
            if api_url:
                water_readings = fetch_water_level_api(
                    api_url,
                    api_key=secret_value("WATER_LEVEL_API_KEY", ""),
                    api_key_header=secret_value(
                        "WATER_LEVEL_API_KEY_HEADER", "Authorization"
                    ),
                    bearer_prefix=secret_value(
                        "WATER_LEVEL_API_KEY_PREFIX", "Bearer"
                    ),
                )

        if not water_readings.empty:
            water_readings = normalize_water_level_columns(water_readings)
            if water_source != "PhilSensors public webpage" or use_fallback_thresholds:
                water_readings = apply_threshold_defaults(
                    water_readings,
                    default_alert,
                    default_alarm,
                    default_critical,
                )

            station_df = compute_latest_station_state(
                water_readings,
                int(stale_minutes),
                int(offline_minutes),
                float(rapid_rise_m_hr),
                float(rapid_fall_m_hr),
            )
            basin_water_df = aggregate_water_by_basin(station_df)
            if not station_df.empty:
                water_last_updated = station_df["timestamp"].max()
    except Exception as exc:
        water_error = f"{type(exc).__name__}: {exc}"

water_defaults = {
    "water_status": "No Data",
    "station_count": 0,
    "active_station_count": 0,
    "max_level_m": np.nan,
    "max_rise_rate_m_hr": np.nan,
    "rapid_rise_count": 0,
    "latest_water_timestamp": pd.NaT,
}

if not basin_water_df.empty:
    hazard_df = hazard_df.merge(basin_water_df, on="basin_name", how="left")
else:
    for column, default in water_defaults.items():
        hazard_df[column] = default

for column, default in water_defaults.items():
    if column not in hazard_df.columns:
        hazard_df[column] = default
    hazard_df[column] = hazard_df[column].fillna(default)

hazard_df["combined_level"] = hazard_df.apply(
    lambda row: combined_level(row["hazard_level"], row["water_status"]), axis=1
)

# Header metrics
if app_mode == "Version 3":
    metric1, metric2, metric3, metric4, metric5 = st.columns(5)
    metric1.metric("Basins monitored", f"{len(hazard_df)}")
    metric2.metric("Water-level stations", f"{len(station_df)}")
    mapped_count = (
        int(station_df[["lat", "lon", "basin_name"]].notna().all(axis=1).sum())
        if not station_df.empty
        else 0
    )
    metric3.metric("Mapped stations", mapped_count)
    active_count = (
        int((~station_df["water_status"].isin(["Stale", "Offline"])).sum())
        if not station_df.empty
        else 0
    )
    metric4.metric("Active stations", active_count)
    urgent_count = (
        int(station_df["water_status"].isin(["Alarm", "Critical"]).sum())
        if not station_df.empty
        else 0
    )
    metric5.metric("Alarm/Critical", urgent_count)
    st.caption(
        f"Rainfall updated: {format_timestamp(rain_last_updated)} | "
        f"Latest water reading: {format_timestamp(water_last_updated)}"
    )
else:
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Basins monitored", f"{len(hazard_df)}")
    metric2.metric("Rainfall source", rainfall_source)
    metric3.metric("Last updated", format_timestamp(rain_last_updated))

if rain_failures and rainfall_source == "Live Open-Meteo":
    with st.expander("Show rainfall fetch issues"):
        st.write(rain_failures)

if water_error:
    st.error(f"Water-level data could not be processed: {water_error}")
elif app_mode == "Version 3" and station_df.empty:
    st.warning(
        "No water-level station readings are loaded. Try PhilSensors again, "
        "use the sample, upload a CSV, or configure a REST endpoint."
    )

if app_mode == "Version 3" and water_source == "PhilSensors public webpage" and philsensors_metadata:
    mode = philsensors_metadata.get("mode", "unknown")
    scraped_at = philsensors_metadata.get("scraped_at")
    if mode == "live":
        st.success(
            f"PhilSensors webpage retrieval succeeded. Retrieved: {format_timestamp(scraped_at)}"
        )
    else:
        st.warning(
            "The live PhilSensors page could not be read, so the app is showing the last "
            f"successful cache from {format_timestamp(scraped_at)}."
        )
    if philsensors_metadata.get("error"):
        with st.expander("Show PhilSensors retrieval details"):
            st.code(philsensors_metadata["error"])

    if not philsensors_registry_template.empty:
        st.download_button(
            "Download station-registry template",
            data=philsensors_registry_template.to_csv(index=False).encode("utf-8"),
            file_name="philsensors_station_registry.csv",
            mime="text/csv",
            help=(
                "Fill in latitude, longitude, basin_name, and any verified station-specific "
                "thresholds, then upload the CSV from the sidebar."
            ),
        )
        if not station_df.empty:
            unmapped = station_df[
                ~station_df[["lat", "lon", "basin_name"]].notna().all(axis=1)
            ]
            if not unmapped.empty:
                st.info(
                    f"{len(unmapped)} station(s) were read successfully but are not plotted "
                    "on the basin map because their coordinates or basin assignment are missing. They can still appear on the separate province water-level map."
                )

critical_rain = hazard_df[
    hazard_df["hazard_level"].isin(["Extreme", "Severe"])
].sort_values("hazard_ratio", ascending=False)
critical_water = (
    station_df[station_df["water_status"].isin(["Alarm", "Critical"])]
    if not station_df.empty
    else pd.DataFrame()
)

rapid_rise_stations = (
    station_df[station_df.get("rapid_rise", False).fillna(False)]
    if not station_df.empty and "rapid_rise" in station_df.columns
    else pd.DataFrame()
)
rapid_fall_stations = (
    station_df[station_df.get("rapid_fall", False).fillna(False)]
    if not station_df.empty and "rapid_fall" in station_df.columns
    else pd.DataFrame()
)
if not rapid_rise_stations.empty:
    names = ", ".join(rapid_rise_stations["station_name"].head(6).tolist())
    st.error(
        f"Rapid water-level rise: {len(rapid_rise_stations)} station(s) exceed "
        f"+{float(rapid_rise_m_hr):.2f} m/hr: {names}"
    )
if not rapid_fall_stations.empty:
    names = ", ".join(rapid_fall_stations["station_name"].head(6).tolist())
    st.info(
        f"Rapid water-level decrease: {len(rapid_fall_stations)} station(s) exceed "
        f"-{float(rapid_fall_m_hr):.2f} m/hr: {names}"
    )

if not critical_water.empty:
    names = ", ".join(critical_water["station_name"].head(6).tolist())
    st.error(
        f"Observed water-level alert: {len(critical_water)} station(s) are at Alarm/Critical: {names}"
    )
elif not critical_rain.empty:
    names = ", ".join(critical_rain["basin_name"].head(6).tolist())
    st.error(
        f"Rainfall alert: {len(critical_rain)} basin(s) are at Severe/Extreme in the selected "
        f"{rain_window} view: {names}"
    )
else:
    st.success("No Alarm/Critical station or Severe/Extreme rainfall alert is currently shown.")

# Put the water-level trend map before the basin map so it is immediately visible.
if app_mode == "Version 3" and not station_df.empty:
    st.markdown("---")
    st.subheader("💧 Observed Water-Level Rise/Fall Map")
    st.caption(
        "Province shading and large labels show the strongest measured hourly change. "
        "Red/orange means rising; blue means falling; grey means nearly stable. "
        "Click a province to see every gauge, current level, and rate of change."
    )

    legend_html = """
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin:4px 0 12px">
      <span style="background:#b91c1c;color:white;padding:4px 8px;border-radius:12px">↑↑ Rapid rise</span>
      <span style="background:#f97316;color:white;padding:4px 8px;border-radius:12px">↑ Rising</span>
      <span style="background:#6b7280;color:white;padding:4px 8px;border-radius:12px">→ Stable</span>
      <span style="background:#3b82f6;color:white;padding:4px 8px;border-radius:12px">↓ Falling</span>
      <span style="background:#1d4ed8;color:white;padding:4px 8px;border-radius:12px">↓↓ Rapid fall</span>
      <span style="background:#7e22ce;color:white;padding:4px 8px;border-radius:12px">↕ Mixed rapid change</span>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)

    control_a, control_b = st.columns(2)
    with control_a:
        include_inactive_province_map = st.checkbox(
            "Include stale/offline gauges when they still have a measured change",
            value=True,
            key="include_inactive_province_map_v38",
        )
    with control_b:
        only_rapid_province_map = st.checkbox(
            "Show only rapid-rise or rapid-fall provinces",
            value=False,
            key="only_rapid_province_map_v38",
        )

    try:
        with st.spinner("Loading province boundaries and water-level trend labels..."):
            province_geojson, province_centroids, province_source_name = (
                fetch_province_reference()
            )
        province_station_df = match_station_provinces(station_df, province_centroids)
        province_map, mapped_province_stations, province_summary_df = (
            build_province_water_map(
                province_station_df,
                province_geojson,
                include_inactive=include_inactive_province_map,
                only_rapid=only_rapid_province_map,
            )
        )

        trend_left, trend_right = st.columns([1.9, 1.0])
        with trend_left:
            st_folium(
                province_map,
                width=None,
                height=760,
                key="province_water_level_map_v38",
                returned_objects=[
                    "last_object_clicked_tooltip",
                    "last_object_clicked_popup",
                ],
            )
            st.caption(f"Province boundary source: {province_source_name}")

        with trend_right:
            st.markdown("#### Fastest measured changes")
            ranked = mapped_province_stations.copy()
            ranked["abs_rate"] = pd.to_numeric(
                ranked["rise_rate_m_hr"], errors="coerce"
            ).abs()
            ranked = ranked.dropna(subset=["abs_rate"]).sort_values(
                "abs_rate", ascending=False
            )
            ranked["level_m"] = pd.to_numeric(
                ranked["level_m"], errors="coerce"
            ).round(3)
            ranked["rise_rate_m_hr"] = pd.to_numeric(
                ranked["rise_rate_m_hr"], errors="coerce"
            ).round(3)
            st.dataframe(
                ranked[
                    [
                        "province_ref",
                        "station_name",
                        "level_m",
                        "rise_rate_m_hr",
                        "trend_label",
                        "water_status",
                    ]
                ].head(20),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "The map label and this table use the same rise_rate_m_hr value from the "
                "scraped PhilSensors hourly readings."
            )

        matched_count = int(mapped_province_stations["province_ref_key"].notna().sum())
        unmatched_count = int(len(station_df) - matched_count)
        st.info(
            f"Province-matched gauges: {matched_count} of {len(station_df)}. "
            f"Unmatched province names: {unmatched_count}."
        )
        if province_summary_df.empty:
            st.warning(
                "No province labels were produced. Turn on 'Include stale/offline gauges' "
                "or confirm that rise_rate_m_hr contains numeric values."
            )
    except Exception as exc:
        st.error(
            "The water-level rise/fall map could not be rendered: "
            f"{type(exc).__name__}: {exc}"
        )
        st.caption(
            "The station table remains available below. The province map uses two "
            "government ArcGIS services and automatically tries the second when the first fails."
        )

st.markdown("---")
st.subheader("🌧️ River-Basin Rainfall and Combined-Hazard Map")

left, right = st.columns([1.8, 1.0])
with left:
    merged_geojson = merge_geojson_with_metrics(geojson_data, hazard_df)
    folium_map = build_map(
        merged_geojson,
        map_view=map_view,
        station_df=(
            station_df
            if app_mode == "Version 3" and overlay_exact_station_markers
            else None
        ),
        track_df=typhoon_track_df if use_typhoon_overlay else None,
    )
    map_state = st_folium(
        folium_map,
        width=None,
        height=680,
        key="basin_hazard_map",
        returned_objects=[
            "last_active_drawing",
            "last_object_clicked_tooltip",
            "last_object_clicked_popup",
        ],
    )

with right:
    st.subheader("Highest-risk basins")
    if map_view.startswith("Observed water level"):
        top = hazard_df.copy()
        top["_sort"] = top["water_status"].map(water_status_rank)
        top = top.sort_values(
            ["_sort", "max_level_m"], ascending=[False, False]
        ).head(10)
        st.dataframe(
            top[
                [
                    "basin_name",
                    "region",
                    "water_status",
                    "station_count",
                    "max_level_m",
                    "max_rise_rate_m_hr",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
    elif map_view == "Combined monitoring":
        combined_rank = {
            "No Data": -1,
            "Low": 0,
            "Moderate": 1,
            "High": 2,
            "Severe": 3,
            "Extreme": 4,
        }
        top = hazard_df.copy()
        top["_sort"] = top["combined_level"].map(combined_rank)
        top = top.sort_values(
            ["_sort", "hazard_ratio"], ascending=[False, False]
        ).head(10)
        st.dataframe(
            top[
                [
                    "basin_name",
                    "region",
                    "combined_level",
                    "hazard_level",
                    "water_status",
                    "effective_rain_mm",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        top = hazard_df.sort_values("hazard_ratio", ascending=False).head(10).copy()
        top["hazard_ratio"] = (
            (top["hazard_ratio"] * 100).round(0).astype("Int64").astype(str) + "%"
        )
        st.dataframe(
            top[
                [
                    "basin_name",
                    "region",
                    "hazard_level",
                    "effective_rain_mm",
                    "threshold_selected_mm",
                    "hazard_ratio",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    options = ["All"] + hazard_df.sort_values("basin_name")["basin_name"].tolist()
    selected = st.selectbox("Basin details", options, index=0)
    clicked_basin = get_clicked_basin(map_state)
    if clicked_basin:
        st.info(f"Clicked basin: {clicked_basin}")
        selected = clicked_basin

    if selected != "All":
        row = hazard_df.loc[hazard_df["basin_name"] == selected].iloc[0]
        metric_a, metric_b = st.columns(2)
        metric_a.metric("Rainfall hazard", row["hazard_level"])
        metric_b.metric("Water status", row["water_status"])
        metric_a.metric(
            "Forecast 24h",
            f"{row['forecast_rain_24h_mm']:.1f} mm"
            if pd.notna(row["forecast_rain_24h_mm"])
            else "-",
        )
        metric_b.metric(
            "Forecast 72h",
            f"{row['forecast_rain_72h_mm']:.1f} mm"
            if pd.notna(row["forecast_rain_72h_mm"])
            else "-",
        )
        metric_a.metric("Water stations", int(row["station_count"]))
        metric_b.metric(
            "Highest level",
            f"{row['max_level_m']:.2f} m" if pd.notna(row["max_level_m"]) else "-",
        )
        metric_a.metric(
            "Maximum rise",
            f"{row['max_rise_rate_m_hr']:+.3f} m/hr"
            if pd.notna(row["max_rise_rate_m_hr"])
            else "-",
        )
        metric_b.metric("Combined level", row["combined_level"])
        st.text_area("Suggested rainfall advisory", row["alert_text"], height=125)
    else:
        counts = hazard_df["combined_level"].value_counts().reindex(
            ["Extreme", "Severe", "High", "Moderate", "Low", "No Data"],
            fill_value=0,
        )
        st.bar_chart(counts)

if False and app_mode == "Version 3" and not station_df.empty:
    st.markdown("---")
    st.subheader("Observed Water-Level Province Map")
    st.caption(
        "This is a separate administrative-location map, so water-level values do not "
        "crowd or falsely recolor the river-basin rainfall polygons. Each province pin "
        "opens a table of its PhilSensors stations, levels, change rates, trends, and statuses."
    )
    control_a, control_b = st.columns([1, 2])
    with control_a:
        include_inactive_province_map = st.checkbox(
            "Include stale, offline, and no-data stations",
            value=False,
            key="include_inactive_province_map",
        )
    with control_b:
        st.caption(
            "Pin label = strongest station rate in m/hr. Badge = number of stations represented."
        )

    try:
        with st.spinner("Loading official province boundaries for water-level pins..."):
            province_geojson, province_centroids = fetch_province_reference()
        province_station_df = match_station_provinces(station_df, province_centroids)

        water_left, water_right = st.columns([1.8, 1.0])
        with water_left:
            province_map, mapped_province_stations, province_summary_df = (
                build_province_water_map(
                    province_station_df,
                    province_geojson,
                    include_inactive=include_inactive_province_map,
                )
            )
            province_map_state = st_folium(
                province_map,
                width=None,
                height=690,
                key="province_water_level_map",
                returned_objects=[
                    "last_object_clicked_tooltip",
                    "last_object_clicked_popup",
                ],
            )

        with water_right:
            st.markdown("#### Province station details")
            clicked_province = get_clicked_province(province_map_state)
            province_options = sorted(
                mapped_province_stations["province_ref"].dropna().unique().tolist()
            )
            if province_options:
                selected_index = (
                    province_options.index(clicked_province)
                    if clicked_province in province_options
                    else 0
                )
                selected_province = st.selectbox(
                    "Province",
                    province_options,
                    index=selected_index,
                    key="water_map_selected_province",
                )
                province_rows = mapped_province_stations[
                    mapped_province_stations["province_ref"] == selected_province
                ].copy()
                province_rows["last_reading"] = province_rows["timestamp"].apply(
                    format_timestamp
                )
                province_rows["level_m"] = province_rows["level_m"].round(3)
                province_rows["rise_rate_m_hr"] = province_rows[
                    "rise_rate_m_hr"
                ].round(3)
                st.dataframe(
                    province_rows[
                        [
                            "station_name",
                            "level_m",
                            "rise_rate_m_hr",
                            "trend_label",
                            "water_status",
                            "threshold_status",
                            "last_reading",
                        ]
                    ].sort_values(
                        "rise_rate_m_hr",
                        key=lambda values: values.abs(),
                        ascending=False,
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                fastest_rise = pd.to_numeric(
                    province_rows["rise_rate_m_hr"], errors="coerce"
                ).max()
                fastest_fall = pd.to_numeric(
                    province_rows["rise_rate_m_hr"], errors="coerce"
                ).min()
                metric_a, metric_b, metric_c = st.columns(3)
                metric_a.metric("Stations", len(province_rows))
                metric_b.metric(
                    "Fastest rise",
                    f"{fastest_rise:+.3f} m/hr" if pd.notna(fastest_rise) else "-",
                )
                metric_c.metric(
                    "Fastest fall",
                    f"{fastest_fall:+.3f} m/hr" if pd.notna(fastest_fall) else "-",
                )
            else:
                st.info("No station province names matched the DENR province reference.")

        unmatched_provinces = province_station_df[
            province_station_df["province_ref_key"].isna()
            & province_station_df["province"].notna()
        ]
        if not unmatched_provinces.empty:
            names = ", ".join(
                sorted(unmatched_provinces["province"].astype(str).unique())[:12]
            )
            st.warning(
                f"{len(unmatched_provinces)} station(s) have province names that were not "
                f"matched to the DENR reference: {names}"
            )
    except Exception as exc:
        st.warning(
            "The separate province water-level map could not load the DENR province "
            f"reference: {type(exc).__name__}: {exc}"
        )

if app_mode == "Version 3":
    st.markdown("---")
    station_tab, history_tab = st.tabs(
        ["Latest water-level stations", "Recent station readings"]
    )

    with station_tab:
        if station_df.empty:
            st.info("No station data loaded.")
        else:
            display_station = station_df.copy()
            display_station["last_reading"] = display_station["timestamp"].apply(
                format_timestamp
            )
            display_station["rise_rate_m_hr"] = display_station[
                "rise_rate_m_hr"
            ].round(3)
            display_station["level_m"] = display_station["level_m"].round(3)
            display_station["mapped"] = display_station[
                ["lat", "lon", "basin_name"]
            ].notna().all(axis=1)
            st.dataframe(
                display_station[
                    [
                        "station_id",
                        "station_name",
                        "region",
                        "province",
                        "basin_name",
                        "mapped",
                        "threshold_status",
                        "water_status",
                        "level_m",
                        "rise_rate_m_hr",
                        "rapid_rise",
                        "rapid_fall",
                        "trend_label",
                        "alert_m",
                        "alarm_m",
                        "critical_m",
                        "age_minutes",
                        "last_reading",
                    ]
                ].sort_values(
                    by="water_status",
                    key=lambda series: series.map(water_status_rank),
                    ascending=False,
                ),
                use_container_width=True,
                hide_index=True,
            )

    with history_tab:
        if water_readings.empty:
            st.info("No historical station readings loaded.")
        else:
            station_choices = sorted(water_readings["station_id"].dropna().unique())
            selected_station = st.selectbox("Station", station_choices)
            history = water_readings[
                water_readings["station_id"] == selected_station
            ].sort_values("timestamp")
            chart_df = history.set_index("timestamp")[["level_m"]]
            st.line_chart(chart_df)
            st.dataframe(
                history[
                    [
                        "timestamp",
                        "level_m",
                        "alert_m",
                        "alarm_m",
                        "critical_m",
                    ]
                ].sort_values("timestamp", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

st.markdown("---")
st.markdown("### Basin forecast table")
show_columns = [
    "basin_name",
    "region",
    "forecast_rain_24h_mm",
    "forecast_rain_72h_mm",
    "antecedent_rain_72h_mm",
    "effective_rain_mm",
    "hazard_level",
    "water_status",
    "station_count",
    "max_level_m",
    "max_rise_rate_m_hr",
    "combined_level",
]
st.dataframe(
    hazard_df[show_columns].sort_values(
        ["combined_level", "effective_rain_mm"],
        ascending=[True, False],
        na_position="last",
    ),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")
st.markdown(
    """
**Important notes**

- Open-Meteo data are rainfall inputs; they are not observed river-gauge measurements.
- The PhilSensors option is an unofficial automated retrieval from the public monitoring webpage. It does not use a DOST-ASTI API credential.
- The app refreshes the public page conservatively, caches successful results, and falls back to the most recent saved readings when retrieval fails.
- Stations without verified basin coordinates are kept off the basin polygon map, but can be summarized at province level on the separate water-level map. Province pins are administrative-location summaries and do not assign a station to a river basin.
- Water-level status is only operationally meaningful when the displayed PhilSensors threshold class or verified station-specific Alert/Alarm/Critical levels are available.
- This dashboard is a screening and monitoring aid. It does not replace PAGASA, DOST-ASTI, LGU, DRRMO, dam-operator, or evacuation advisories.
"""
)
