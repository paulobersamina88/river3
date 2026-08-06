from __future__ import annotations

import numpy as np
import pandas as pd

from core.schema import classify_threshold, normalize_readings

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


def compute_station_state(
    readings: pd.DataFrame,
    stale_minutes: int = 180,
    offline_minutes: int = 1440,
    rapid_rise_m_hr: float = 0.30,
    rapid_fall_m_hr: float = 0.30,
) -> pd.DataFrame:
    readings = normalize_readings(readings)
    if readings.empty:
        return readings
    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for _, group in readings.sort_values(["station_id", "timestamp"]).groupby("station_id", sort=False):
        latest = group.iloc[-1].copy()
        rate = latest.get("rise_rate_m_hr", np.nan)
        if pd.isna(rate) and len(group) >= 2:
            previous = group.iloc[-2]
            hours = (latest["timestamp"] - previous["timestamp"]).total_seconds() / 3600
            if hours > 0:
                rate = (float(latest["level_m"]) - float(previous["level_m"])) / hours
        latest["rise_rate_m_hr"] = rate
        age = max((now - latest["timestamp"]).total_seconds() / 60, 0)
        latest["age_minutes"] = age
        source_status = str(latest.get("threshold_status", "")).strip().title()
        if age > offline_minutes:
            status = "Offline"
        elif age > stale_minutes:
            status = "Stale"
        elif source_status in {"Normal", "Alert", "Alarm", "Critical"}:
            status = source_status
        else:
            status = classify_threshold(latest["level_m"], latest.get("alert_m"), latest.get("alarm_m"), latest.get("critical_m"))
        latest["water_status"] = status
        latest["rapid_rise"] = bool(pd.notna(rate) and float(rate) >= rapid_rise_m_hr and status not in {"Stale", "Offline"})
        latest["rapid_fall"] = bool(pd.notna(rate) and float(rate) <= -abs(rapid_fall_m_hr) and status not in {"Stale", "Offline"})
        if latest["rapid_rise"]:
            trend = "Rapid rise"
        elif latest["rapid_fall"]:
            trend = "Rapid fall"
        elif pd.notna(rate) and float(rate) > 0.005:
            trend = "Rising"
        elif pd.notna(rate) and float(rate) < -0.005:
            trend = "Falling"
        else:
            trend = str(latest.get("source_trend", "")).strip() or "Stable"
        latest["trend_label"] = trend
        rows.append(latest)
    return pd.DataFrame(rows).reset_index(drop=True)


def aggregate_by_basin(stations: pd.DataFrame) -> pd.DataFrame:
    columns = ["basin_name", "water_status", "station_count", "active_station_count", "max_level_m", "max_rise_rate_m_hr", "rapid_rise_count", "latest_water_timestamp"]
    if stations is None or stations.empty:
        return pd.DataFrame(columns=columns)
    mapped = stations[stations["basin_name"].fillna("").astype(str).str.strip().ne("")].copy()
    rows = []
    for basin, group in mapped.groupby("basin_name"):
        active = group[~group["water_status"].isin(["Stale", "Offline"])]
        source = active if not active.empty else group
        status = source.loc[source["water_status"].map(STATUS_RANK).idxmax(), "water_status"]
        rows.append({
            "basin_name": basin,
            "water_status": status,
            "station_count": len(group),
            "active_station_count": len(active),
            "max_level_m": pd.to_numeric(group["level_m"], errors="coerce").max(),
            "max_rise_rate_m_hr": pd.to_numeric(group["rise_rate_m_hr"], errors="coerce").max(),
            "rapid_rise_count": int(group["rapid_rise"].fillna(False).sum()),
            "latest_water_timestamp": group["timestamp"].max(),
        })
    return pd.DataFrame(rows)


def combined_level(rain_level: str, water_status: str) -> str:
    rain_rank = {"No Data": -1, "Low": 0, "Moderate": 1, "High": 2, "Severe": 3, "Extreme": 4}.get(rain_level, -1)
    water_rank = {"No Data": -1, "No Threshold": -1, "Normal": 0, "Stale": -1, "Offline": -1, "Alert": 1, "Alarm": 3, "Critical": 4}.get(water_status, -1)
    return {-1: "No Data", 0: "Low", 1: "Moderate", 2: "High", 3: "Severe", 4: "Extreme"}[max(rain_rank, water_rank)]
