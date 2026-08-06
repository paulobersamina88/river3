from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

from core.schema import (
    MANILA_TZ,
    ProviderResult,
    clean_text,
    empty_bulletins,
    normalize_bulletins,
)

PROVIDER_NAME = "PAGASA hydrological bulletins"
BULLETIN_SOURCES = [
    {
        "source_name": "PAGASA NCR/Pasig-Marikina-Laguna de Bay bulletin",
        "source_url": "https://www.pagasa.dost.gov.ph/flood/ncr-pasig-marikina-laguna-de-bay",
        "basin_name": "Pasig-Laguna",
        "river_system": "Pasig-Marikina-Laguna de Bay system",
    },
    {
        "source_name": "PAGASA Abra River Basin bulletin",
        "source_url": "https://www.pagasa.dost.gov.ph/flood/abra",
        "basin_name": "Abra",
        "river_system": "Abra River",
    },
]
PRFFWC_URLS = [
    "https://prffwc.synthasite.com/hydro-forecast-1.php",
    "http://prffwc.synthasite.com/hydro-forecast-1.php",
]


def _extract_between(text: str, start: str, end_terms: list[str]) -> str:
    start_match = re.search(re.escape(start), text, flags=re.I)
    if not start_match:
        return ""
    remainder = text[start_match.end():]
    end_positions = []
    for term in end_terms:
        match = re.search(re.escape(term), remainder, flags=re.I)
        if match:
            end_positions.append(match.start())
    end = min(end_positions) if end_positions else len(remainder)
    return clean_text(remainder[:end].strip(" :-"))


def _parse_pagasa_page(html: str, config: dict[str, str]) -> pd.DataFrame:
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    issued_text = ""
    valid_text = ""
    match = re.search(r"ISSUED AT\s+(.+?)(?=VALID UNTIL|OBSERVED 24-HR RAINFALL)", text, flags=re.I)
    if match:
        issued_text = clean_text(match.group(1))
    match = re.search(r"VALID UNTIL\s+(.+?)(?=OBSERVED 24-HR RAINFALL)", text, flags=re.I)
    if match:
        valid_text = clean_text(match.group(1))
    observed = _extract_between(text, "OBSERVED 24-HR RAINFALL", ["FORECAST 24-HR RAINFALL", "FORECAST AVERAGE BASIN RAINFALL"])
    forecast = _extract_between(text, "FORECAST 24-HR RAINFALL", ["FORECAST WATER LEVEL", "PREPARED BY"])
    if not forecast:
        forecast = _extract_between(text, "FORECAST AVERAGE BASIN RAINFALL", ["FORECAST WATER LEVEL", "PREPARED BY"])
    water = _extract_between(text, "FORECAST WATER LEVEL", ["PREPARED BY", "CHECKED BY"])
    if not any([issued_text, observed, forecast, water]):
        return empty_bulletins()
    issued_at = pd.NaT
    try:
        cleaned = re.sub(r"\bTODAY\b|\bTOMORROW\b", "", issued_text, flags=re.I).strip(" ,")
        issued_at = pd.Timestamp(cleaned)
        if issued_at.tzinfo is None:
            issued_at = issued_at.tz_localize(MANILA_TZ)
        issued_at = issued_at.tz_convert("UTC")
    except Exception:
        pass
    return pd.DataFrame(
        [
            {
                **config,
                "issued_at": issued_at,
                "valid_until": pd.NaT,
                "observed_rainfall": observed,
                "forecast_rainfall": forecast,
                "forecast_water_level": water,
                "status": "Official basin forecast",
                "scraped_at": pd.Timestamp.now(tz="UTC"),
                "is_cached": False,
                "notes": "Qualitative basin forecast; not a numerical station measurement.",
            }
        ]
    )


def _parse_prffwc(html: str) -> pd.DataFrame:
    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception:
        tables = []
    rows = []
    for table in tables:
        frame = table.copy()
        frame.columns = [clean_text(column) for column in frame.columns]
        joined = " ".join(frame.columns).lower()
        if "present river status" not in joined or "trend" not in joined:
            continue
        area_col = next((column for column in frame.columns if "area" in column.lower()), frame.columns[0])
        status_col = next((column for column in frame.columns if "present river status" in column.lower()), None)
        trend_col = next((column for column in frame.columns if "trend" in column.lower()), None)
        weather_col = next((column for column in frame.columns if "weather" in column.lower()), None)
        if not status_col or not trend_col:
            continue
        for _, record in frame.iterrows():
            area = clean_text(record.get(area_col))
            if not area or area.lower() == "nan":
                continue
            status = clean_text(record.get(status_col))
            trend = clean_text(record.get(trend_col))
            rows.append(
                {
                    "source_name": "Pampanga River Flood Forecasting and Warning Center",
                    "source_url": PRFFWC_URLS[0],
                    "basin_name": "Pampanga",
                    "river_system": area,
                    "issued_at": pd.NaT,
                    "valid_until": pd.NaT,
                    "observed_rainfall": clean_text(record.get(weather_col)) if weather_col else "",
                    "forecast_rainfall": "",
                    "forecast_water_level": f"Present status: {status}; Trend: {trend}",
                    "status": trend or status,
                    "scraped_at": pd.Timestamp.now(tz="UTC"),
                    "is_cached": False,
                    "notes": "Qualitative PRFFWC sub-basin status; not a numerical gauge reading.",
                }
            )
        if rows:
            break
    return pd.DataFrame(rows)


def fetch(cache_dir: str | Path) -> ProviderResult:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    errors = []
    for config in BULLETIN_SOURCES:
        try:
            response = requests.get(config["source_url"], timeout=30, headers={"User-Agent": "Mozilla/5.0 river-monitor-academic"})
            response.raise_for_status()
            frame = _parse_pagasa_page(response.text, config)
            if frame.empty:
                raise ValueError("no bulletin fields were found")
            frames.append(frame)
        except Exception as exc:
            errors.append(f"{config['source_name']}: {type(exc).__name__}: {exc}")

    prffwc_loaded = False
    for url in PRFFWC_URLS:
        try:
            response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0 river-monitor-academic"})
            response.raise_for_status()
            frame = _parse_prffwc(response.text)
            if frame.empty:
                raise ValueError("no PRFFWC status table was found")
            frames.append(frame)
            prffwc_loaded = True
            break
        except Exception as exc:
            errors.append(f"PRFFWC {url}: {type(exc).__name__}: {exc}")
    if not prffwc_loaded:
        pass

    last_success = cache / "pagasa_bulletins_last_success.csv"
    if frames:
        bulletins = normalize_bulletins(pd.concat(frames, ignore_index=True), PROVIDER_NAME)
        bulletins.to_csv(last_success, index=False)
        return ProviderResult(
            provider=PROVIDER_NAME,
            bulletins=bulletins,
            mode="live",
            fetched_at=pd.Timestamp.now(tz="UTC"),
            message=f"Loaded {len(bulletins)} official qualitative bulletin row(s).",
            error="",
            source_url="https://www.pagasa.dost.gov.ph/flood",
            details={"partial_errors": errors},
        )

    if last_success.exists():
        cached = pd.read_csv(last_success)
        cached["is_cached"] = True
        return ProviderResult(
            provider=PROVIDER_NAME,
            bulletins=normalize_bulletins(cached, PROVIDER_NAME),
            mode="cache",
            fetched_at=pd.Timestamp.fromtimestamp(last_success.stat().st_mtime, tz="UTC"),
            message="Live bulletin pages failed; showing the last successful bulletin cache.",
            source_url="https://www.pagasa.dost.gov.ph/flood",
            details={"live_errors": errors},
        )

    return ProviderResult(
        provider=PROVIDER_NAME,
        mode="error",
        fetched_at=pd.Timestamp.now(tz="UTC"),
        message="No hydrological bulletin could be loaded.",
        error="; ".join(errors),
        source_url="https://www.pagasa.dost.gov.ph/flood",
    )
