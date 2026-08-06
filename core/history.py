from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.schema import normalize_readings


def append_history_and_compute_rates(
    current: pd.DataFrame,
    history_path: str | Path,
    max_age_days: int = 14,
) -> pd.DataFrame:
    """Append provider snapshots and compute rates from the previous distinct reading.

    This is mainly for sources that publish only one current value. The result is
    retrieval-to-retrieval change, not necessarily the sensor's native sampling rate.
    """
    current = normalize_readings(current)
    if current.empty:
        return current

    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame()
    if path.exists():
        try:
            history = normalize_readings(pd.read_csv(path))
        except Exception:
            history = pd.DataFrame()

    combined = pd.concat([history, current], ignore_index=True) if not history.empty else current.copy()
    combined = combined.sort_values(["station_id", "timestamp"]).drop_duplicates(
        subset=["station_id", "timestamp", "level_m"], keep="last"
    )
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=max_age_days)
    combined = combined[combined["timestamp"] >= cutoff]

    updated_rows = []
    for _, row in current.iterrows():
        record = row.copy()
        if pd.isna(record.get("rise_rate_m_hr")):
            candidates = combined[
                (combined["station_id"] == record["station_id"])
                & (combined["timestamp"] < record["timestamp"])
            ].sort_values("timestamp")
            if not candidates.empty:
                previous = candidates.iloc[-1]
                elapsed_hours = (record["timestamp"] - previous["timestamp"]).total_seconds() / 3600.0
                if elapsed_hours > 0:
                    record["rise_rate_m_hr"] = (
                        float(record["level_m"]) - float(previous["level_m"])
                    ) / elapsed_hours
                    if not str(record.get("notes", "")).strip():
                        record["notes"] = "Rate is retrieval-to-retrieval change."
        rate = record.get("rise_rate_m_hr", np.nan)
        if pd.notna(rate):
            if float(rate) > 0.005:
                record["source_trend"] = "Rising"
            elif float(rate) < -0.005:
                record["source_trend"] = "Falling"
            else:
                record["source_trend"] = "Stable"
        updated_rows.append(record)

    try:
        combined.to_csv(path, index=False)
    except Exception:
        pass
    return normalize_readings(pd.DataFrame(updated_rows))
