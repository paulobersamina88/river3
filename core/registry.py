from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def load_registry(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def apply_supplementary_registry(readings: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Apply exact station-id/name patterns without changing measured values.

    Registry fields are treated as user-verified metadata. A row may match by exact
    station_id, or by a case-insensitive substring in station_name_pattern.
    """
    if readings is None or readings.empty or registry is None or registry.empty:
        return readings
    output = readings.copy()
    fields = [
        "river_system", "basin_name", "region", "province", "municipality",
        "lat", "lon", "notes",
    ]
    for index, row in output.iterrows():
        provider_key = _norm(row.get("source_name"))
        station_id_key = _norm(row.get("station_id"))
        station_name = str(row.get("station_name", ""))
        candidates = registry.copy()
        if "source_name" in candidates.columns:
            source_keys = candidates["source_name"].map(_norm)
            candidates = candidates[(source_keys == "") | (source_keys == provider_key)]
        matched = None
        if "station_id" in candidates.columns and station_id_key:
            exact = candidates[candidates["station_id"].map(_norm) == station_id_key]
            if not exact.empty:
                matched = exact.iloc[-1]
        if matched is None and "station_name_pattern" in candidates.columns:
            for _, candidate in candidates.iterrows():
                pattern = str(candidate.get("station_name_pattern", "") or "").strip()
                if pattern and pattern.lower() in station_name.lower():
                    matched = candidate
                    break
        if matched is None:
            continue
        for field in fields:
            if field not in matched.index:
                continue
            value = matched.get(field)
            if pd.isna(value) or str(value).strip() == "":
                continue
            if field in {"lat", "lon"}:
                value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                if pd.isna(value):
                    continue
            if field == "notes" and str(output.at[index, "notes"] or "").strip():
                output.at[index, field] = f"{output.at[index, field]} {value}".strip()
            else:
                output.at[index, field] = value
    return output
