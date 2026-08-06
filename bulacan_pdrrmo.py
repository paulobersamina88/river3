from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from core.history import append_history_and_compute_rates
from core.schema import (
    MANILA_TZ,
    ProviderResult,
    classify_threshold,
    clean_text,
    normalize_readings,
    parse_level_metres,
    parse_number,
    slug,
)

PROVIDER_NAME = "Bulacan PDRRMO River Status"
SOURCE_URL = "https://pdrrmo.bulacan.gov.ph/"


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if isinstance(output.columns, pd.MultiIndex):
        output.columns = [
            " ".join(str(part).strip() for part in column if str(part).strip() and not str(part).lower().startswith("unnamed")).strip()
            for column in output.columns
        ]
    else:
        output.columns = [str(column).strip() for column in output.columns]
    return output


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean_text(value).lower()).strip()


def _find_column(columns: list[str], term: str) -> str | None:
    term = _key(term)
    for column in columns:
        if term in _key(column):
            return column
    return None


def _page_timestamp(html: str) -> pd.Timestamp:
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    patterns = [
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\s+\d{1,2}:\d{2}\s*(?:am|pm)",
        r"\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}\s*(?:am|pm)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            try:
                ts = pd.Timestamp(match.group(0))
                return ts.tz_localize(MANILA_TZ).tz_convert("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            except Exception:
                pass
    return pd.Timestamp.now(tz="UTC")


def infer_station(station: str) -> tuple[str, str, str, str, str, str]:
    text = _key(station)
    if any(term in text for term in ["meycauayan", "marilao", "northville", "obando"]):
        municipality = ""
        for candidate in ["Meycauayan", "Marilao", "Obando"]:
            if candidate.lower() in text:
                municipality = candidate
                break
        return "Meycauayan-Marilao-Obando River System", "Pasig-Laguna", "Region III", "Bulacan", municipality, "MMORS"
    if any(term in text for term in ["sulipan", "apalit", "candaba", "paralaya", "pampanga river"]):
        municipality = "Apalit" if "apalit" in text or "sulipan" in text else ("Candaba" if "candaba" in text or "paralaya" in text else "")
        return "Pampanga River", "Pampanga", "Region III", "Pampanga", municipality, "Pampanga River"
    if any(term in text for term in ["calumpit", "hagonoy", "paombong"]):
        municipality = next((name for name in ["Calumpit", "Hagonoy", "Paombong"] if name.lower() in text), "")
        return "Lower Pampanga/Angat river system", "Pampanga", "Region III", "Bulacan", municipality, "Lower Pampanga"
    if "bustos" in text or "alejo bridge" in text:
        return "Angat River", "Pampanga", "Region III", "Bulacan", "Bustos", "Angat"
    return "Bulacan river station", "", "Region III", "Bulacan", "", "Unclassified"


def parse_html(html: str) -> pd.DataFrame:
    timestamp = _page_timestamp(html)
    try:
        tables = [_flatten_columns(table) for table in pd.read_html(io.StringIO(html))]
    except Exception:
        tables = []
    rows = []
    for table in tables:
        columns = list(table.columns)
        station_col = _find_column(columns, "station")
        actual_col = _find_column(columns, "actual level")
        alert_col = _find_column(columns, "alert")
        alarm_col = _find_column(columns, "alarm")
        critical_col = _find_column(columns, "critical")
        date_col = _find_column(columns, "date")
        if not station_col or not actual_col or not alert_col or not alarm_col or not critical_col:
            continue
        for _, row in table.iterrows():
            station = clean_text(row.get(station_col))
            if not station or station.lower() in {"no record!", "no record", "nan", "station"}:
                continue
            level = parse_level_metres(row.get(actual_col))
            if pd.isna(level):
                continue
            alert = parse_level_metres(row.get(alert_col))
            alarm = parse_level_metres(row.get(alarm_col))
            critical = parse_level_metres(row.get(critical_col))
            observed_at = timestamp
            if date_col:
                date_text = clean_text(row.get(date_col))
                try:
                    day = pd.Timestamp(date_text)
                    if day.tzinfo is None:
                        day = day.tz_localize(MANILA_TZ)
                    page_local = timestamp.tz_convert(MANILA_TZ)
                    observed_at = day.replace(hour=page_local.hour, minute=page_local.minute).tz_convert("UTC")
                except Exception:
                    pass
            river, basin, region, province, municipality, classification = infer_station(station)
            status = classify_threshold(level, alert, alarm, critical)
            rows.append(
                {
                    "station_id": f"BUL-{slug(station).upper()}",
                    "station_name": station,
                    "river_system": river,
                    "basin_name": basin,
                    "region": region,
                    "province": province,
                    "municipality": municipality,
                    "location": station,
                    "lat": np.nan,
                    "lon": np.nan,
                    "timestamp": observed_at,
                    "level_m": level,
                    "rise_rate_m_hr": np.nan,
                    "alert_m": alert,
                    "alarm_m": alarm,
                    "critical_m": critical,
                    "threshold_status": status,
                    "source_trend": "",
                    "source_name": PROVIDER_NAME,
                    "source_url": SOURCE_URL,
                    "data_kind": "official_lgu_gauge",
                    "scraped_at": pd.Timestamp.now(tz="UTC"),
                    "is_cached": False,
                    "notes": f"Station classified as {classification} from its published name. Rate is computed only after a later retrieval.",
                }
            )
        if rows:
            break
    return pd.DataFrame(rows)


def fetch(cache_dir: str | Path) -> ProviderResult:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    last_success = cache / "bulacan_pdrrmo_last_success.csv"
    history_path = cache / "bulacan_pdrrmo_history.csv"
    try:
        response = requests.get(SOURCE_URL, timeout=35, headers={"User-Agent": "Mozilla/5.0 river-monitor-academic"})
        response.raise_for_status()
        readings = parse_html(response.text)
        if readings.empty:
            raise ValueError("the River Status Stations table currently contains no numerical records")
        readings = normalize_readings(readings, PROVIDER_NAME)
        readings = append_history_and_compute_rates(readings, history_path)
        readings.to_csv(last_success, index=False)
        return ProviderResult(
            provider=PROVIDER_NAME,
            readings=readings,
            mode="live",
            fetched_at=pd.Timestamp.now(tz="UTC"),
            message=f"Loaded {len(readings)} official river-station row(s).",
            source_url=SOURCE_URL,
        )
    except Exception as exc:
        if last_success.exists():
            cached = pd.read_csv(last_success)
            cached["is_cached"] = True
            return ProviderResult(
                provider=PROVIDER_NAME,
                readings=normalize_readings(cached, PROVIDER_NAME),
                mode="cache",
                fetched_at=pd.Timestamp.fromtimestamp(last_success.stat().st_mtime, tz="UTC"),
                message="The live table has no usable rows; showing the last successful cache.",
                source_url=SOURCE_URL,
                details={"live_error": f"{type(exc).__name__}: {exc}"},
            )
        return ProviderResult(
            provider=PROVIDER_NAME,
            mode="empty",
            fetched_at=pd.Timestamp.now(tz="UTC"),
            message="The official Bulacan river table currently has no numerical records.",
            error=f"{type(exc).__name__}: {exc}",
            source_url=SOURCE_URL,
        )
