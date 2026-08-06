from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.schema import (
    ProviderResult,
    clean_text,
    ensure_utc,
    normalize_bulletins,
    normalize_readings,
    parse_level_metres,
    parse_number,
    slug,
)

PROVIDER_NAME = "ChatGPT Work / manual official report import"

ALIASES = {
    "requested_source": ["requested_source", "requested page", "requested_page", "search_target"],
    "reporting_source": [
        "reporting_source", "source_name", "source_page", "office", "reporting office",
    ],
    "station_name": [
        "river_or_site", "station_name", "monitoring_point", "location", "site",
        "river_system", "river_name", "river",
    ],
    "river_system": ["river_or_site", "river_system", "river_name", "river"],
    "basin_name": ["basin_name", "basin"],
    "region": ["region"],
    "province": ["province"],
    "municipality": ["municipality", "city", "lgu"],
    "timestamp": ["observed_at", "timestamp", "observation_time", "date_time"],
    "level": ["level", "level_m", "water_level", "value"],
    "unit": ["unit", "level_unit"],
    "status": ["status", "reported_status", "threshold_status", "official_status"],
    "alert_m": ["alert_m", "alert"],
    "alarm_m": ["alarm_m", "alarm"],
    "critical_m": ["critical_m", "critical"],
    "lat": ["lat", "latitude"],
    "lon": ["lon", "lng", "longitude"],
    "source_url": ["source_url", "post_url", "url"],
    "notes": ["notes", "source_wording", "exact_source_wording"],
}

REPORT_COLUMNS = [
    "report_id",
    "requested_source",
    "reporting_source",
    "river_or_site",
    "reported_level",
    "reported_unit",
    "level_m",
    "reported_status",
    "standard_status",
    "observed_at",
    "source_url",
    "notes",
    "region",
    "province",
    "municipality",
    "basin_name",
    "lat",
    "lon",
    "coordinate_basis",
    "report_kind",
]

VALID_UNITS = {
    "m", "meter", "meters", "metre", "metres", "el.m", "el m", "masl",
    "ft", "feet", "foot", "cm", "mm",
}


def _normalized_lookup(columns: list[Any]) -> dict[str, Any]:
    return {
        re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_"): column
        for column in columns
    }


def _find(lookup: dict[str, Any], names: list[str]) -> Any | None:
    for name in names:
        key = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
        if key in lookup:
            return lookup[key]
    return None


def _status_class(value: Any) -> str:
    """Map only clear official words; preserve ambiguous local terms as No Threshold."""
    text = clean_text(value).lower()
    if not text:
        return "No Threshold"
    if any(term in text for term in ["above normal level", "(anl)", "middle level", "spilling"]):
        return "No Threshold"
    if "critical" in text:
        return "Critical"
    if "below alarm" not in text and re.search(r"\balarm\b", text):
        return "Alarm"
    if "below alert" not in text and re.search(r"\balert\b", text):
        return "Alert"
    normal_terms = [
        "normal", "safe level", "code green", "below normal", "way below normal",
        "below alert", "below alarm",
    ]
    if any(term in text for term in normal_terms):
        return "Normal"
    return "No Threshold"


def _source_trend(value: Any) -> str:
    text = clean_text(value).lower()
    if any(term in text for term in ["receding", "falling", "decreasing", "downward"]):
        return "Falling"
    if any(term in text for term in ["rising", "increasing", "upward"]):
        return "Rising"
    if any(term in text for term in ["stable", "no significant change", "unchanged"]):
        return "Stable"
    if "spilling" in text:
        return "Spilling"
    return ""


def _clean_unit(value: Any) -> str:
    unit = clean_text(value).lower().lstrip(".")
    return unit if unit in VALID_UNITS else ""


def _load_registry(registry: pd.DataFrame | str | Path | None) -> pd.DataFrame:
    if isinstance(registry, pd.DataFrame):
        return registry.copy()
    if registry is None:
        return pd.DataFrame()
    path = Path(registry)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _registry_match(report: dict[str, Any], registry: pd.DataFrame) -> dict[str, Any]:
    if registry.empty:
        return report
    searchable = " ".join(
        clean_text(report.get(field))
        for field in ["requested_source", "reporting_source", "river_or_site", "notes"]
    ).lower()
    candidates = registry.copy()
    if "priority" in candidates.columns:
        candidates["priority"] = pd.to_numeric(candidates["priority"], errors="coerce").fillna(9999)
        candidates = candidates.sort_values("priority")
    for _, row in candidates.iterrows():
        pattern = clean_text(row.get("match_pattern"))
        if not pattern:
            continue
        try:
            matched = re.search(pattern, searchable, flags=re.IGNORECASE) is not None
        except re.error:
            matched = pattern.lower() in searchable
        if not matched:
            continue
        for field in [
            "region", "province", "municipality", "basin_name", "river_system",
            "lat", "lon", "coordinate_basis",
        ]:
            current = report.get(field)
            value = row.get(field)
            if (current is None or clean_text(current) == "" or (field in {"lat", "lon"} and pd.isna(current))) and pd.notna(value) and clean_text(value):
                report[field] = value
        registry_note = clean_text(row.get("notes"))
        if registry_note:
            report["notes"] = f"{clean_text(report.get('notes'))} {registry_note}".strip()
        break
    return report


def _row_value(record: pd.Series, column: Any | None, default: Any = "") -> Any:
    return record.get(column, default) if column is not None else default


def parse_all(frame: pd.DataFrame, registry: pd.DataFrame | str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Return numerical readings, qualitative bulletins, all normalized reports, and warnings."""
    if frame is None or frame.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(columns=REPORT_COLUMNS), []

    lookup = _normalized_lookup(list(frame.columns))
    cols = {key: _find(lookup, values) for key, values in ALIASES.items()}
    if not cols["station_name"] or not cols["timestamp"]:
        raise ValueError(
            "Input requires river_or_site/station/location and observed_at/timestamp columns."
        )

    location_registry = _load_registry(registry)
    readings: list[dict[str, Any]] = []
    bulletins: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    warnings: list[str] = []
    now = pd.Timestamp.now(tz="UTC")

    for source_index, (_, record) in enumerate(frame.iterrows(), start=1):
        river_or_site = clean_text(_row_value(record, cols["station_name"]))
        if not river_or_site:
            warnings.append(f"Row {source_index}: skipped because river_or_site/station is blank.")
            continue

        requested_source = clean_text(_row_value(record, cols["requested_source"]))
        reporting_source = clean_text(_row_value(record, cols["reporting_source"])) or requested_source or PROVIDER_NAME
        raw_level = _row_value(record, cols["level"], "")
        raw_unit = _clean_unit(_row_value(record, cols["unit"], ""))
        raw_status = clean_text(_row_value(record, cols["status"], ""))
        observed = ensure_utc(_row_value(record, cols["timestamp"]))
        source_url = clean_text(_row_value(record, cols["source_url"], ""))
        notes = clean_text(_row_value(record, cols["notes"], ""))
        lat = parse_number(_row_value(record, cols["lat"], np.nan))
        lon = parse_number(_row_value(record, cols["lon"], np.nan))
        coordinate_basis = "User-supplied coordinates" if pd.notna(lat) and pd.notna(lon) else ""
        level_m = parse_level_metres(raw_level, raw_unit)

        report = {
            "report_id": f"WORK-{slug(reporting_source).upper()}-{slug(river_or_site).upper()}-{source_index:03d}",
            "requested_source": requested_source,
            "reporting_source": reporting_source,
            "river_or_site": river_or_site,
            "reported_level": clean_text(raw_level),
            "reported_unit": raw_unit,
            "level_m": level_m,
            "reported_status": raw_status,
            "standard_status": _status_class(raw_status),
            "observed_at": observed,
            "source_url": source_url,
            "notes": notes,
            "region": clean_text(_row_value(record, cols["region"], "")),
            "province": clean_text(_row_value(record, cols["province"], "")),
            "municipality": clean_text(_row_value(record, cols["municipality"], "")),
            "basin_name": clean_text(_row_value(record, cols["basin_name"], "")),
            "river_system": clean_text(_row_value(record, cols["river_system"], "")) or river_or_site,
            "lat": lat,
            "lon": lon,
            "coordinate_basis": coordinate_basis,
            "report_kind": "Numerical report" if pd.notna(level_m) else "Qualitative report",
        }
        report = _registry_match(report, location_registry)
        report["lat"] = parse_number(report.get("lat"))
        report["lon"] = parse_number(report.get("lon"))
        if not report.get("coordinate_basis") and pd.notna(report["lat"]) and pd.notna(report["lon"]):
            report["coordinate_basis"] = "Representative registry anchor"
        reports.append(report)

        source_notes = [
            f"Requested source: {requested_source}" if requested_source else "",
            f"Original report: {clean_text(raw_level)} {raw_unit}".strip() if pd.notna(level_m) else "",
            f"Reported status: {raw_status}" if raw_status else "",
            notes,
            f"Coordinate basis: {report.get('coordinate_basis')}" if report.get("coordinate_basis") else "",
            "Imported from a supervised ChatGPT Work/manual search result; not an instrument feed unless the source explicitly says so.",
        ]
        combined_notes = " ".join(part for part in source_notes if part).strip()

        if pd.notna(level_m):
            alert = parse_level_metres(_row_value(record, cols["alert_m"], ""), "m") if cols["alert_m"] else np.nan
            alarm = parse_level_metres(_row_value(record, cols["alarm_m"], ""), "m") if cols["alarm_m"] else np.nan
            critical = parse_level_metres(_row_value(record, cols["critical_m"], ""), "m") if cols["critical_m"] else np.nan
            readings.append(
                {
                    "station_id": report["report_id"],
                    "station_name": river_or_site,
                    "river_system": report.get("river_system") or river_or_site,
                    "basin_name": report.get("basin_name", ""),
                    "region": report.get("region", ""),
                    "province": report.get("province", ""),
                    "municipality": report.get("municipality", ""),
                    "location": river_or_site,
                    "lat": report.get("lat", np.nan),
                    "lon": report.get("lon", np.nan),
                    "timestamp": observed,
                    "level_m": level_m,
                    "rise_rate_m_hr": np.nan,
                    "alert_m": alert,
                    "alarm_m": alarm,
                    "critical_m": critical,
                    "threshold_status": report["standard_status"],
                    "source_trend": _source_trend(raw_status),
                    "source_name": reporting_source,
                    "source_url": source_url,
                    "data_kind": "manual_work_numeric_report",
                    "coordinate_basis": report.get("coordinate_basis", ""),
                    "scraped_at": now,
                    "is_cached": False,
                    "notes": combined_notes,
                }
            )
        else:
            bulletin_status = raw_status or "Official qualitative river report"
            bulletins.append(
                {
                    "source_name": reporting_source,
                    "source_url": source_url,
                    "basin_name": report.get("basin_name", ""),
                    "river_system": report.get("river_system") or river_or_site,
                    "issued_at": observed,
                    "valid_until": pd.NaT,
                    "observed_rainfall": "",
                    "forecast_rainfall": "",
                    "forecast_water_level": raw_status,
                    "status": report["standard_status"] if report["standard_status"] != "No Threshold" else bulletin_status,
                    "scraped_at": now,
                    "is_cached": False,
                    "notes": combined_notes,
                }
            )

    report_frame = pd.DataFrame(reports)
    for column in REPORT_COLUMNS:
        if column not in report_frame.columns:
            report_frame[column] = np.nan if column in {"level_m", "lat", "lon"} else ""
    report_frame = report_frame[REPORT_COLUMNS]
    return pd.DataFrame(readings), pd.DataFrame(bulletins), report_frame, warnings


def parse(frame: pd.DataFrame, registry: pd.DataFrame | str | Path | None = None) -> pd.DataFrame:
    """Backward-compatible numerical parser used by the original tests and callers."""
    readings, _, _, _ = parse_all(frame, registry=registry)
    return readings


def read_delimited_text(text: str) -> pd.DataFrame:
    cleaned = str(text or "").strip("\ufeff\n\r ")
    if not cleaned:
        return pd.DataFrame()
    try:
        dialect = csv.Sniffer().sniff(cleaned[:5000], delimiters="\t,;|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = "\t" if "\t" in cleaned.splitlines()[0] else ","
    return pd.read_csv(io.StringIO(cleaned), sep=delimiter, dtype=str, keep_default_na=False)


def from_dataframe(
    frame: pd.DataFrame,
    registry: pd.DataFrame | str | Path | None = None,
) -> ProviderResult:
    try:
        raw_readings, raw_bulletins, reports, warnings = parse_all(frame, registry=registry)
        readings = normalize_readings(raw_readings, PROVIDER_NAME)
        bulletins = normalize_bulletins(raw_bulletins, PROVIDER_NAME)
        return ProviderResult(
            provider=PROVIDER_NAME,
            readings=readings,
            bulletins=bulletins,
            mode="manual-import",
            fetched_at=pd.Timestamp.now(tz="UTC"),
            message=(
                f"Imported {len(reports)} report(s): {len(readings)} numerical and "
                f"{len(bulletins)} qualitative."
            ),
            details={"reports": reports, "warnings": warnings},
        )
    except Exception as exc:
        return ProviderResult(
            provider=PROVIDER_NAME,
            mode="error",
            fetched_at=pd.Timestamp.now(tz="UTC"),
            message="The ChatGPT Work/manual report input could not be normalized.",
            error=f"{type(exc).__name__}: {exc}",
            details={"reports": pd.DataFrame(columns=REPORT_COLUMNS), "warnings": []},
        )


def from_text(
    text: str,
    registry: pd.DataFrame | str | Path | None = None,
) -> ProviderResult:
    try:
        frame = read_delimited_text(text)
    except Exception as exc:
        return ProviderResult(
            provider=PROVIDER_NAME,
            mode="error",
            fetched_at=pd.Timestamp.now(tz="UTC"),
            message="The pasted text could not be read as TSV or CSV.",
            error=f"{type(exc).__name__}: {exc}",
            details={"reports": pd.DataFrame(columns=REPORT_COLUMNS), "warnings": []},
        )
    return from_dataframe(frame, registry=registry)
