from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from core.schema import (
    MANILA_TZ,
    ProviderResult,
    clean_text,
    normalize_readings,
    parse_number,
)

PROVIDER_NAME = "LLDA Laguna de Bay water level"
SOURCE_URLS = [
    "https://llda.gov.ph/water-level/",
    "https://www.llda.gov.ph/water-level/",
]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; PH-River-Monitor/5.4; academic research)",
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
}


def _parse_timestamp(value: Any, now_utc: pd.Timestamp | None = None) -> pd.Timestamp:
    now_utc = now_utc or pd.Timestamp.now(tz="UTC")
    text = clean_text(value).strip(" ,:-")
    if not text:
        return pd.NaT
    text = re.sub(r"\bPST\b|\bPHT\b|\bLT\b", "", text, flags=re.I).strip()
    try:
        ts = pd.Timestamp(text)
    except Exception:
        # A page may show only a clock time. Attach it to today's Manila date.
        time_match = re.search(r"(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM))", text, flags=re.I)
        if not time_match:
            return pd.NaT
        local_now = now_utc.tz_convert(MANILA_TZ)
        parsed = pd.Timestamp(time_match.group(1))
        ts = local_now.replace(hour=parsed.hour, minute=parsed.minute, second=parsed.second, microsecond=0)
    if pd.isna(ts):
        return pd.NaT
    if ts.tzinfo is None:
        ts = ts.tz_localize(MANILA_TZ)
    return ts.tz_convert("UTC")


def _status_from_text(text: str) -> str:
    lowered = text.lower()
    # Prefer explicit current status wording over threshold labels appearing in
    # legends or explanatory text.
    match = re.search(r"(?:current\s+status|status)\s*[:\-]?\s*(normal|alert|alarm|critical)", lowered, flags=re.I)
    if match:
        return match.group(1).title()
    return "No Threshold"


def _record(level: float, timestamp: pd.Timestamp, source_url: str, status: str, notes: str) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    return pd.DataFrame(
        [
            {
                "station_id": "LLDA-LAGUNA-DE-BAY",
                "station_name": "Laguna de Bay (LLDA lake level)",
                "river_system": "Laguna de Bay",
                "basin_name": "Pasig-Laguna",
                "region": "CALABARZON / NCR",
                "province": "Laguna",
                "municipality": "Lake-wide",
                "location": "Laguna de Bay",
                "lat": 14.2720,
                "lon": 121.2250,
                "timestamp": timestamp if not pd.isna(timestamp) else now,
                "level_m": level,
                "level_30min_ago_m": np.nan,
                "level_1hr_ago_m": np.nan,
                "level_2hr_ago_m": np.nan,
                "rise_rate_m_hr": np.nan,
                "alert_m": np.nan,
                "alarm_m": np.nan,
                "critical_m": np.nan,
                "threshold_status": status,
                "source_trend": "",
                "source_name": PROVIDER_NAME,
                "source_url": source_url,
                "data_kind": "official lake-level measurement",
                "scraped_at": now,
                "is_cached": False,
                "notes": notes,
            }
        ]
    )


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if isinstance(output.columns, pd.MultiIndex):
        output.columns = [clean_text(" ".join(str(part) for part in column if str(part) != "nan")) for column in output.columns]
    else:
        output.columns = [clean_text(column) for column in output.columns]
    return output


def _parse_tables(html: str, source_url: str) -> pd.DataFrame:
    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception:
        return pd.DataFrame()
    candidates: list[tuple[pd.Timestamp, float, str, str]] = []
    for table in tables:
        frame = _flatten_columns(table)
        if frame.empty:
            continue
        columns = {column: column.lower() for column in frame.columns}
        level_columns = [
            column for column, lowered in columns.items()
            if ("water" in lowered and "level" in lowered) or "lake level" in lowered or "current level" in lowered
        ]
        date_columns = [column for column, lowered in columns.items() if any(term in lowered for term in ["date", "time", "as of", "updated", "observation"])]
        status_columns = [column for column, lowered in columns.items() if "status" in lowered]
        if not level_columns:
            # Some LLDA tables can be two-column label/value layouts.
            for _, row in frame.iterrows():
                row_text = " ".join(clean_text(value) for value in row.tolist())
                match = re.search(r"(?:current\s+(?:lake\s+)?level|laguna\s+de\s+bay\s+water\s+level)\s*[:\-]?\s*(\d{1,2}(?:\.\d+)?)\s*(?:m|meters?)", row_text, flags=re.I)
                if match:
                    timestamp = _parse_timestamp(row_text)
                    candidates.append((timestamp, float(match.group(1)), _status_from_text(row_text), row_text))
            continue
        for _, row in frame.iterrows():
            for level_column in level_columns:
                level = parse_number(row.get(level_column))
                if pd.isna(level) or not (0 < float(level) < 30):
                    continue
                timestamp = pd.NaT
                for date_column in date_columns:
                    timestamp = _parse_timestamp(row.get(date_column))
                    if not pd.isna(timestamp):
                        break
                status = _status_from_text(clean_text(row.get(status_columns[0]))) if status_columns else "No Threshold"
                candidates.append((timestamp, float(level), status, clean_text(" ".join(str(value) for value in row.tolist()))))
    if not candidates:
        return pd.DataFrame()
    candidates.sort(key=lambda item: item[0].value if not pd.isna(item[0]) else -1)
    timestamp, level, status, row_text = candidates[-1]
    return _record(
        level,
        timestamp,
        source_url,
        status,
        "Lake-wide Laguna de Bay level parsed from the official LLDA page; it is not a tributary-river measurement. " + row_text[:300],
    )


def _parse_text(text: str, source_url: str) -> pd.DataFrame:
    normalized = clean_text(text)
    patterns = [
        r"(?:current\s+(?:lake\s+)?level|current\s+water\s+level)\s*[:\-]?\s*(\d{1,2}(?:\.\d+)?)\s*(?:m|meters?)",
        r"laguna\s+de\s+bay\s+water\s+level(?:\s+update)?\s*(?:as\s+of[^0-9]{0,80})?\s*[:\-]?\s*(\d{1,2}(?:\.\d+)?)\s*(?:m|meters?)",
        r"water\s+level\s*[:\-]?\s*(\d{1,2}(?:\.\d+)?)\s*(?:m|meters?)",
    ]
    level = np.nan
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.I)
        if match:
            level = float(match.group(1))
            break
    if pd.isna(level) or not (0 < float(level) < 30):
        return pd.DataFrame()

    timestamp = pd.NaT
    timestamp_patterns = [
        r"(?:as\s+of|last\s+update|updated|observation(?:\s+time)?)\s*[:\-]?\s*([A-Za-z0-9,/:\- ]{5,60})",
        r"(\d{1,2}\s+[A-Za-z]+\s+20\d{2}\s+\d{1,2}:\d{2}\s*(?:AM|PM)?)",
        r"([A-Za-z]+\s+\d{1,2},?\s+20\d{2}\s*,?\s*\d{1,2}:\d{2}\s*(?:AM|PM)?)",
    ]
    for pattern in timestamp_patterns:
        match = re.search(pattern, normalized, flags=re.I)
        if match:
            timestamp = _parse_timestamp(match.group(1))
            if not pd.isna(timestamp):
                break

    return _record(
        float(level),
        timestamp,
        source_url,
        _status_from_text(normalized),
        "Lake-wide Laguna de Bay level parsed from the official LLDA page; it must not be relabelled as a Victoria, Pagsanjan, San Juan, or other tributary-river gauge.",
    )


def _parse_embedded_json(html: str, source_url: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    blobs = [script.string or script.get_text(" ", strip=True) for script in soup.find_all("script")]
    key_pattern = re.compile(
        r'["\'](?:current[_\- ]?(?:lake[_\- ]?)?level|water[_\- ]?level|lake[_\- ]?level)["\']\s*:\s*["\']?(\d{1,2}(?:\.\d+)?)["\']?',
        flags=re.I,
    )
    for blob in blobs:
        match = key_pattern.search(blob or "")
        if not match:
            continue
        level = float(match.group(1))
        if not (0 < level < 30):
            continue
        timestamp = pd.NaT
        date_match = re.search(r'["\'](?:observed[_\- ]?at|updated[_\- ]?at|timestamp|date[_\- ]?time)["\']\s*:\s*["\']([^"\']+)', blob, flags=re.I)
        if date_match:
            timestamp = _parse_timestamp(date_match.group(1))
        return _record(
            level,
            timestamp,
            source_url,
            "No Threshold",
            "Lake-wide Laguna de Bay level parsed from structured data embedded in the official LLDA page.",
        )
    return pd.DataFrame()


def parse_llda_html(html: str, source_url: str = SOURCE_URLS[0]) -> pd.DataFrame:
    for parser in (_parse_tables, _parse_embedded_json):
        frame = parser(html, source_url)
        if frame is not None and not frame.empty:
            return normalize_readings(frame, PROVIDER_NAME)
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    frame = _parse_text(text, source_url)
    return normalize_readings(frame, PROVIDER_NAME) if frame is not None and not frame.empty else pd.DataFrame()


def _rendered_html(url: str, timeout_ms: int) -> str:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1200})
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        try:
            page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 30000))
        except Exception:
            pass
        content = page.content()
        browser.close()
        return content


def fetch(cache_dir: str | Path, timeout_ms: int = 90000) -> ProviderResult:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    for url in SOURCE_URLS:
        try:
            response = requests.get(url, timeout=35, headers=HEADERS)
            response.raise_for_status()
            readings = parse_llda_html(response.text, response.url)
            if readings.empty:
                raise ValueError("no current Laguna de Bay level was found in the static page")
            cache_path = cache / "llda_laguna_de_bay_last_success.csv"
            readings.to_csv(cache_path, index=False)
            return ProviderResult(
                provider=PROVIDER_NAME,
                readings=readings,
                mode="live",
                fetched_at=pd.Timestamp.now(tz="UTC"),
                message="Loaded the official Laguna de Bay lake-level reading.",
                source_url=response.url,
            )
        except Exception as exc:
            errors.append(f"requests {url}: {type(exc).__name__}: {exc}")

        try:
            rendered = _rendered_html(url, timeout_ms=timeout_ms)
            readings = parse_llda_html(rendered, url)
            if readings.empty:
                raise ValueError("no current Laguna de Bay level was found after browser rendering")
            cache_path = cache / "llda_laguna_de_bay_last_success.csv"
            readings.to_csv(cache_path, index=False)
            return ProviderResult(
                provider=PROVIDER_NAME,
                readings=readings,
                mode="live-browser",
                fetched_at=pd.Timestamp.now(tz="UTC"),
                message="Loaded the official Laguna de Bay lake-level reading after browser rendering.",
                source_url=url,
            )
        except Exception as exc:
            errors.append(f"browser {url}: {type(exc).__name__}: {exc}")

    cache_path = cache / "llda_laguna_de_bay_last_success.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        cached["is_cached"] = True
        return ProviderResult(
            provider=PROVIDER_NAME,
            readings=normalize_readings(cached, PROVIDER_NAME),
            mode="cache",
            fetched_at=pd.Timestamp.fromtimestamp(cache_path.stat().st_mtime, tz="UTC"),
            message="The LLDA page could not be read; showing the last successful Laguna de Bay cache.",
            source_url=SOURCE_URLS[0],
            details={"live_errors": errors},
        )

    return ProviderResult(
        provider=PROVIDER_NAME,
        mode="error",
        fetched_at=pd.Timestamp.now(tz="UTC"),
        message="No official Laguna de Bay lake-level reading could be loaded.",
        error="; ".join(errors),
        source_url=SOURCE_URLS[0],
    )
