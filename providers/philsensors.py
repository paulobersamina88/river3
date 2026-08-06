from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.schema import ProviderResult, normalize_readings
from philsensors_scraper import (
    DEFAULT_URL,
    STATION_METADATA_URL,
    fetch_philsensors_readings,
    fetch_philsensors_station_metadata,
    merge_station_metadata,
    merge_station_registry,
)

PROVIDER_NAME = "DOST-ASTI PhilSensors"


def fetch(
    cache_dir: str | Path,
    registry: pd.DataFrame | None = None,
    timeout_ms: int = 90_000,
    load_station_metadata: bool = True,
) -> ProviderResult:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    try:
        readings, reading_meta = fetch_philsensors_readings(
            url=DEFAULT_URL,
            backup_path=cache / "philsensors_last_success.csv",
            timeout_ms=timeout_ms,
        )
        metadata_meta: dict[str, Any] = {}
        if load_station_metadata:
            try:
                station_meta, metadata_meta = fetch_philsensors_station_metadata(
                    url=STATION_METADATA_URL,
                    backup_path=cache / "philsensors_station_metadata.csv",
                    timeout_ms=timeout_ms,
                )
                readings = merge_station_metadata(readings, station_meta)
            except Exception as exc:
                metadata_meta = {"mode": "error", "error": f"{type(exc).__name__}: {exc}"}
        readings = merge_station_registry(readings, registry)
        readings = readings.copy()
        readings["river_system"] = readings.get("river_system", "")
        readings["municipality"] = readings.get("municipality", "")
        readings["source_name"] = PROVIDER_NAME
        readings["source_url"] = DEFAULT_URL
        readings["data_kind"] = "instrument"
        readings["is_cached"] = reading_meta.get("mode") != "live"
        readings["notes"] = readings.get("notes", "")
        if "source_trend" not in readings:
            readings["source_trend"] = readings.get("trend", "")
        result = normalize_readings(readings, PROVIDER_NAME)
        mode = "live" if reading_meta.get("mode") == "live" else "cache"
        message = reading_meta.get("message", "")
        errors = [reading_meta.get("error", ""), metadata_meta.get("error", "")]
        error = "; ".join(item for item in errors if item)
        return ProviderResult(
            provider=PROVIDER_NAME,
            readings=result,
            mode=mode,
            fetched_at=pd.Timestamp(reading_meta.get("scraped_at", pd.Timestamp.now(tz="UTC"))),
            message=message,
            error=error if mode == "empty" else "",
            source_url=DEFAULT_URL,
            details={"reading_metadata": reading_meta, "station_metadata": metadata_meta},
        )
    except Exception as exc:
        return ProviderResult(
            provider=PROVIDER_NAME,
            mode="error",
            fetched_at=pd.Timestamp.now(tz="UTC"),
            message="PhilSensors provider failed without a usable cache.",
            error=f"{type(exc).__name__}: {exc}",
            source_url=DEFAULT_URL,
        )
