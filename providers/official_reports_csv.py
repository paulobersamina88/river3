from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from core.schema import (
    ProviderResult,
    classify_threshold,
    clean_text,
    ensure_utc,
    normalize_readings,
    parse_level_metres,
    parse_number,
    slug,
)

PROVIDER_NAME = "Official LGU report import"

ALIASES = {
    "station_name": ["station_name", "monitoring_point", "location", "site"],
    "river_system": ["river_system", "river_name", "river"],
    "basin_name": ["basin_name", "basin"],
    "region": ["region"],
    "province": ["province"],
    "municipality": ["municipality", "city", "lgu"],
    "timestamp": ["timestamp", "observed_at", "observation_time", "date_time"],
    "level": ["level_m", "level", "water_level", "value"],
    "unit": ["unit", "level_unit"],
    "status": ["status", "threshold_status", "official_status"],
    "alert_m": ["alert_m", "alert"],
    "alarm_m": ["alarm_m", "alarm"],
    "critical_m": ["critical_m", "critical"],
    "lat": ["lat", "latitude"],
    "lon": ["lon", "lng", "longitude"],
    "source_name": ["source_name", "source_page", "office"],
    "source_url": ["source_url", "post_url", "url"],
    "notes": ["notes", "source_wording", "exact_source_wording"],
}


def _normalized_lookup(columns: list[Any]) -> dict[str, Any]:
    return {re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_"): column for column in columns}


def _find(lookup: dict[str, Any], names: list[str]) -> Any | None:
    for name in names:
        if name in lookup:
            return lookup[name]
    return None


def parse(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    lookup = _normalized_lookup(list(frame.columns))
    cols = {key: _find(lookup, values) for key, values in ALIASES.items()}
    if not cols["station_name"] or not cols["timestamp"] or not cols["level"]:
        raise ValueError("CSV requires a station/location column, timestamp, and numerical water level")
    rows = []
    for _, record in frame.iterrows():
        station = clean_text(record.get(cols["station_name"]))
        if not station:
            continue
        unit = clean_text(record.get(cols["unit"])) if cols["unit"] else "m"
        level = parse_level_metres(record.get(cols["level"]), unit)
        if pd.isna(level):
            continue
        alert = parse_level_metres(record.get(cols["alert_m"]), "m") if cols["alert_m"] else np.nan
        alarm = parse_level_metres(record.get(cols["alarm_m"]), "m") if cols["alarm_m"] else np.nan
        critical = parse_level_metres(record.get(cols["critical_m"]), "m") if cols["critical_m"] else np.nan
        reported_status = clean_text(record.get(cols["status"])).title() if cols["status"] else ""
        status = reported_status if reported_status in {"Normal", "Alert", "Alarm", "Critical"} else classify_threshold(level, alert, alarm, critical)
        source_name = clean_text(record.get(cols["source_name"])) if cols["source_name"] else PROVIDER_NAME
        source_url = clean_text(record.get(cols["source_url"])) if cols["source_url"] else ""
        rows.append(
            {
                "station_id": f"LGU-{slug(source_name).upper()}-{slug(station).upper()}",
                "station_name": station,
                "river_system": clean_text(record.get(cols["river_system"])) if cols["river_system"] else "",
                "basin_name": clean_text(record.get(cols["basin_name"])) if cols["basin_name"] else "",
                "region": clean_text(record.get(cols["region"])) if cols["region"] else "",
                "province": clean_text(record.get(cols["province"])) if cols["province"] else "",
                "municipality": clean_text(record.get(cols["municipality"])) if cols["municipality"] else "",
                "location": station,
                "lat": parse_number(record.get(cols["lat"])) if cols["lat"] else np.nan,
                "lon": parse_number(record.get(cols["lon"])) if cols["lon"] else np.nan,
                "timestamp": ensure_utc(record.get(cols["timestamp"])),
                "level_m": level,
                "rise_rate_m_hr": np.nan,
                "alert_m": alert,
                "alarm_m": alarm,
                "critical_m": critical,
                "threshold_status": status,
                "source_trend": "",
                "source_name": source_name or PROVIDER_NAME,
                "source_url": source_url,
                "data_kind": "official_social_report",
                "scraped_at": pd.Timestamp.now(tz="UTC"),
                "is_cached": False,
                "notes": clean_text(record.get(cols["notes"])) if cols["notes"] else "Imported official public report; not an instrument feed unless explicitly stated by the source.",
            }
        )
    return pd.DataFrame(rows)


def from_dataframe(frame: pd.DataFrame) -> ProviderResult:
    try:
        readings = normalize_readings(parse(frame), PROVIDER_NAME)
        return ProviderResult(
            provider=PROVIDER_NAME,
            readings=readings,
            mode="uploaded",
            fetched_at=pd.Timestamp.now(tz="UTC"),
            message=f"Imported {len(readings)} official-report row(s).",
        )
    except Exception as exc:
        return ProviderResult(
            provider=PROVIDER_NAME,
            mode="error",
            fetched_at=pd.Timestamp.now(tz="UTC"),
            message="The uploaded official-report CSV could not be normalized.",
            error=f"{type(exc).__name__}: {exc}",
        )
