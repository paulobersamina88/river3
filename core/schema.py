from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

MANILA_TZ = "Asia/Manila"

READING_COLUMNS = [
    "station_id",
    "station_name",
    "river_system",
    "basin_name",
    "region",
    "province",
    "municipality",
    "location",
    "lat",
    "lon",
    "timestamp",
    "level_m",
    "level_30min_ago_m",
    "level_1hr_ago_m",
    "level_2hr_ago_m",
    "rise_rate_m_hr",
    "alert_m",
    "alarm_m",
    "critical_m",
    "threshold_status",
    "source_trend",
    "source_name",
    "source_url",
    "data_kind",
    "coordinate_basis",
    "scraped_at",
    "is_cached",
    "notes",
]

BULLETIN_COLUMNS = [
    "source_name",
    "source_url",
    "basin_name",
    "river_system",
    "issued_at",
    "valid_until",
    "observed_rainfall",
    "forecast_rainfall",
    "forecast_water_level",
    "status",
    "scraped_at",
    "is_cached",
    "notes",
]


@dataclass
class ProviderResult:
    provider: str
    readings: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=READING_COLUMNS))
    bulletins: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=BULLETIN_COLUMNS))
    mode: str = "empty"
    fetched_at: pd.Timestamp | None = None
    message: str = ""
    error: str = ""
    source_url: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not bool(self.error)


def empty_readings() -> pd.DataFrame:
    return pd.DataFrame(columns=READING_COLUMNS)


def empty_bulletins() -> pd.DataFrame:
    return pd.DataFrame(columns=BULLETIN_COLUMNS)


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", clean_text(value).lower()).strip("-")


def parse_number(value: Any) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value) if not pd.isna(value) else np.nan
    text = clean_text(value).replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else np.nan


def parse_level_metres(value: Any, unit_hint: str = "") -> float:
    number = parse_number(value)
    if pd.isna(number):
        return np.nan
    text = f"{clean_text(value)} {unit_hint}".lower()
    if re.search(r"\b(?:ft|feet|foot)\b", text):
        return float(number) * 0.3048
    if re.search(r"\bcm\b|centimet", text):
        return float(number) / 100.0
    if re.search(r"\bmm\b|millimet", text):
        return float(number) / 1000.0
    return float(number)


def ensure_utc(value: Any, default: pd.Timestamp | None = None) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default if default is not None else pd.Timestamp.now(tz="UTC")
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return default if default is not None else pd.Timestamp.now(tz="UTC")
    if pd.isna(ts):
        return default if default is not None else pd.Timestamp.now(tz="UTC")
    if ts.tzinfo is None:
        ts = ts.tz_localize(MANILA_TZ)
    return ts.tz_convert("UTC")


def classify_threshold(level: Any, alert: Any, alarm: Any, critical: Any) -> str:
    values = [parse_number(item) for item in [level, alert, alarm, critical]]
    level_num, alert_num, alarm_num, critical_num = values
    if pd.isna(level_num):
        return "No Data"
    if any(pd.isna(item) for item in [alert_num, alarm_num, critical_num]):
        return "No Threshold"
    if not (alert_num < alarm_num < critical_num):
        return "No Threshold"
    if level_num >= critical_num:
        return "Critical"
    if level_num >= alarm_num:
        return "Alarm"
    if level_num >= alert_num:
        return "Alert"
    return "Normal"


def normalize_readings(raw: pd.DataFrame, provider_name: str = "") -> pd.DataFrame:
    if raw is None or raw.empty:
        return empty_readings()
    output = raw.copy()
    for column in READING_COLUMNS:
        if column not in output.columns:
            if column in ["lat", "lon", "level_m", "level_30min_ago_m", "level_1hr_ago_m", "level_2hr_ago_m", "rise_rate_m_hr", "alert_m", "alarm_m", "critical_m"]:
                output[column] = np.nan
            elif column in ["is_cached"]:
                output[column] = False
            elif column in ["timestamp", "scraped_at"]:
                output[column] = pd.NaT
            else:
                output[column] = ""

    for column in ["lat", "lon", "level_m", "level_30min_ago_m", "level_1hr_ago_m", "level_2hr_ago_m", "rise_rate_m_hr", "alert_m", "alarm_m", "critical_m"]:
        output[column] = pd.to_numeric(output[column], errors="coerce")

    output["timestamp"] = pd.to_datetime(output["timestamp"], errors="coerce", utc=True)
    output["scraped_at"] = pd.to_datetime(output["scraped_at"], errors="coerce", utc=True)
    now = pd.Timestamp.now(tz="UTC")
    output["timestamp"] = output["timestamp"].fillna(now)
    output["scraped_at"] = output["scraped_at"].fillna(now)
    output["source_name"] = output["source_name"].replace("", provider_name)
    output["station_name"] = output["station_name"].fillna("").astype(str).str.strip()
    output["station_id"] = output["station_id"].fillna("").astype(str).str.strip()
    missing_id = output["station_id"].eq("")
    output.loc[missing_id, "station_id"] = output.loc[missing_id].apply(
        lambda row: f"{slug(row.get('source_name') or provider_name).upper()}-{slug(row.get('station_name') or row.get('location')).upper()}",
        axis=1,
    )
    output["is_cached"] = output["is_cached"].fillna(False).astype(bool)
    return output[READING_COLUMNS].dropna(subset=["station_id", "timestamp", "level_m"]).reset_index(drop=True)


def normalize_bulletins(raw: pd.DataFrame, provider_name: str = "") -> pd.DataFrame:
    if raw is None or raw.empty:
        return empty_bulletins()
    output = raw.copy()
    for column in BULLETIN_COLUMNS:
        if column not in output.columns:
            output[column] = False if column == "is_cached" else (pd.NaT if column in ["issued_at", "valid_until", "scraped_at"] else "")
    for column in ["issued_at", "valid_until", "scraped_at"]:
        output[column] = pd.to_datetime(output[column], errors="coerce", utc=True)
    output["scraped_at"] = output["scraped_at"].fillna(pd.Timestamp.now(tz="UTC"))
    output["source_name"] = output["source_name"].replace("", provider_name)
    output["is_cached"] = output["is_cached"].fillna(False).astype(bool)
    return output[BULLETIN_COLUMNS].reset_index(drop=True)


def combine_provider_results(results: list[ProviderResult]) -> tuple[pd.DataFrame, pd.DataFrame]:
    reading_frames = [normalize_readings(result.readings, result.provider) for result in results if result.readings is not None and not result.readings.empty]
    bulletin_frames = [normalize_bulletins(result.bulletins, result.provider) for result in results if result.bulletins is not None and not result.bulletins.empty]
    readings = pd.concat(reading_frames, ignore_index=True) if reading_frames else empty_readings()
    bulletins = pd.concat(bulletin_frames, ignore_index=True) if bulletin_frames else empty_bulletins()
    if not readings.empty:
        readings = readings.sort_values(["station_id", "timestamp"]).drop_duplicates(
            subset=["station_id", "timestamp", "source_name"], keep="last"
        )
    return readings.reset_index(drop=True), bulletins.reset_index(drop=True)
