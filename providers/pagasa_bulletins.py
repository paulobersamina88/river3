from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

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

try:  # PDF links are used by PAGASA during active Flood Watch periods.
    from pypdf import PdfReader
except Exception:  # pragma: no cover - handled as a provider error at runtime
    PdfReader = None

PROVIDER_NAME = "PAGASA hydrological bulletins"
FLOOD_INDEX_URL = "https://www.pagasa.dost.gov.ph/flood"

# The main Flood Information page can link either to these HTML detail pages or
# to a current PDF bulletin. The provider discovers the linked URL first and
# then falls back to the stable detail page below.
BULLETIN_SOURCES = [
    {
        "key": "ncr_laguna",
        "index_names": ["NCR/Pasig Marikina Laguna de Bay", "NCR Pasig Marikina Laguna de Bay"],
        "source_name": "PAGASA NCR/Pasig-Marikina-Laguna de Bay bulletin",
        "source_url": "https://www.pagasa.dost.gov.ph/flood/ncr-pasig-marikina-laguna-de-bay",
        "basin_name": "Pasig-Laguna",
        "river_system": "Pasig-Marikina-Laguna de Bay system",
    },
    {
        "key": "abra",
        "index_names": ["Abra"],
        "source_name": "PAGASA Abra River Basin bulletin",
        "source_url": "https://www.pagasa.dost.gov.ph/flood/abra",
        "basin_name": "Abra",
        "river_system": "Abra River Basin",
    },
    {
        "key": "panay",
        "index_names": ["Panay"],
        "source_name": "PAGASA Panay River Basin bulletin",
        "source_url": "https://www.pagasa.dost.gov.ph/flood/panay",
        "basin_name": "Panay",
        "river_system": "Panay River Basin",
    },
    {
        "key": "cagayan_de_oro",
        "index_names": ["Cagayan De Oro", "Cagayan de Oro"],
        "source_name": "PAGASA Cagayan de Oro River Basin bulletin",
        "source_url": "https://www.pagasa.dost.gov.ph/flood/cagayan-de-oro",
        "basin_name": "Cagayan de Oro",
        "river_system": "Cagayan de Oro River Basin",
    },
    {
        "key": "davao",
        "index_names": ["Davao"],
        "source_name": "PAGASA Davao River Basin bulletin",
        "source_url": "https://www.pagasa.dost.gov.ph/flood/davao",
        "basin_name": "Davao",
        "river_system": "Davao River Basin",
    },
]

SAMAR_REGIONAL_URLS = [
    "https://www.pagasa.dost.gov.ph/regional-forecast/visprsd",
    "https://bagong.pagasa.dost.gov.ph/regional-forecast/visprsd",
]

PRFFWC_URLS = [
    "https://prffwc.synthasite.com/hydro-forecast-1.php",
    "http://prffwc.synthasite.com/hydro-forecast-1.php",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; PH-River-Monitor/5.4; academic research)",
    "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
}


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean_text(value).lower()).strip()


def _parse_datetime_fragment(value: str, issued_at: pd.Timestamp | None = None) -> pd.Timestamp:
    text = clean_text(value).strip(" ,:-")
    if not text:
        return pd.NaT
    text = re.sub(r"\bPST\b|\bPHT\b|\bLT\b", "", text, flags=re.I).strip()
    if re.search(r"\bTOMORROW\b", text, flags=re.I) and issued_at is not None and not pd.isna(issued_at):
        time_match = re.search(r"(\d{1,2}:\d{2}\s*(?:AM|PM))", text, flags=re.I)
        tomorrow = issued_at.tz_convert(MANILA_TZ) + pd.Timedelta(days=1)
        if time_match:
            parsed_time = pd.Timestamp(time_match.group(1))
            tomorrow = tomorrow.replace(
                hour=parsed_time.hour,
                minute=parsed_time.minute,
                second=0,
                microsecond=0,
            )
        return tomorrow.tz_convert("UTC")
    text = re.sub(r"\bTODAY\b|\bTOMORROW\b", "", text, flags=re.I).strip(" ,")
    try:
        ts = pd.Timestamp(text)
    except Exception:
        return pd.NaT
    if pd.isna(ts):
        return pd.NaT
    if ts.tzinfo is None:
        ts = ts.tz_localize(MANILA_TZ)
    return ts.tz_convert("UTC")


def _extract_between(text: str, start_terms: list[str], end_terms: list[str]) -> str:
    start_positions: list[tuple[int, int]] = []
    for term in start_terms:
        match = re.search(re.escape(term), text, flags=re.I)
        if match:
            start_positions.append((match.start(), match.end()))
    if not start_positions:
        return ""
    _, start_end = min(start_positions, key=lambda item: item[0])
    remainder = text[start_end:]
    end_positions = []
    for term in end_terms:
        match = re.search(re.escape(term), remainder, flags=re.I)
        if match:
            end_positions.append(match.start())
    end = min(end_positions) if end_positions else len(remainder)
    return clean_text(remainder[:end].strip(" :-"))


def _pdf_text(content: bytes) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf is not installed; current PAGASA PDF bulletins cannot be parsed")
    reader = PdfReader(io.BytesIO(content))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return clean_text(" ".join(parts))


def _response_text(response: requests.Response) -> str:
    content_type = response.headers.get("content-type", "").lower()
    if "application/pdf" in content_type or response.url.lower().endswith(".pdf") or response.content[:4] == b"%PDF":
        return _pdf_text(response.content)
    return BeautifulSoup(response.text, "html.parser").get_text(" ", strip=True)


def _parse_flood_index(html: str) -> dict[str, dict[str, str]]:
    """Return normalized basin name -> {status, url} from PAGASA's index table."""
    soup = BeautifulSoup(html, "html.parser")
    result: dict[str, dict[str, str]] = {}
    for row in soup.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if len(cells) < 2:
            continue
        basin = clean_text(cells[0].get_text(" ", strip=True))
        status = clean_text(cells[1].get_text(" ", strip=True))
        if not basin or "flood watch" not in status.lower():
            continue
        link = cells[1].find("a") or cells[0].find("a")
        href = urljoin(FLOOD_INDEX_URL, link.get("href")) if link and link.get("href") else ""
        result[_norm(basin)] = {"status": status, "url": href}
    return result


def _index_record(index: dict[str, dict[str, str]], config: dict[str, Any]) -> dict[str, str]:
    for name in config.get("index_names", []):
        key = _norm(name)
        if key in index:
            return index[key]
    # Resilient fuzzy containment for punctuation changes in the PAGASA table.
    candidates = [_norm(name) for name in config.get("index_names", [])]
    for key, record in index.items():
        if any(candidate and (candidate in key or key in candidate) for candidate in candidates):
            return record
    return {"status": "", "url": ""}


def _parse_pagasa_text(
    text: str,
    config: dict[str, str],
    source_url: str,
    watch_status: str = "",
) -> pd.DataFrame:
    issued_match = re.search(
        r"ISSUED\s+AT\s*:?\s*(.+?)(?=VALID\s+UNTIL|OBSERVED(?:\s+24[- ]?HR)?\s+RAINFALL|FORECAST\s+AVERAGE\s+BASIN\s+RAINFALL)",
        text,
        flags=re.I,
    )
    issued_text = clean_text(issued_match.group(1)) if issued_match else ""
    issued_at = _parse_datetime_fragment(issued_text)

    valid_match = re.search(
        r"VALID\s+UNTIL\s*:?\s*(.+?)(?=OBSERVED(?:\s+24[- ]?HR)?\s+RAINFALL|FORECAST\s+AVERAGE\s+BASIN\s+RAINFALL)",
        text,
        flags=re.I,
    )
    valid_text = clean_text(valid_match.group(1)) if valid_match else ""
    valid_until = _parse_datetime_fragment(valid_text, issued_at=issued_at)

    observed = _extract_between(
        text,
        ["OBSERVED 24-HR RAINFALL", "OBSERVED 24 HR RAINFALL", "OBSERVED RAINFALL"],
        ["FORECAST 24-HR RAINFALL", "FORECAST 24 HR RAINFALL", "FORECAST AVERAGE BASIN RAINFALL", "FORECAST WATER LEVEL"],
    )
    forecast = _extract_between(
        text,
        ["FORECAST 24-HR RAINFALL", "FORECAST 24 HR RAINFALL", "FORECAST AVERAGE BASIN RAINFALL", "FORECAST RAINFALL"],
        ["FORECAST WATER LEVEL", "WATER LEVEL FORECAST", "PREPARED BY", "POSSIBLE IMPACT"],
    )
    water = _extract_between(
        text,
        ["FORECAST WATER LEVEL", "WATER LEVEL FORECAST"],
        ["PREPARED BY", "CHECKED BY", "POSSIBLE IMPACT", "DUTY HYDROLOGIST"],
    )

    if not any([issued_text, observed, forecast, water, watch_status]):
        return empty_bulletins()

    notes = "Official PAGASA basin forecast; this is not a numerical river-gauge measurement."
    if not any([observed, forecast, water]):
        notes += " Only the Flood Information index status was available during this retrieval."

    return pd.DataFrame(
        [
            {
                "source_name": config["source_name"],
                "source_url": source_url or config["source_url"],
                "basin_name": config["basin_name"],
                "river_system": config["river_system"],
                "issued_at": issued_at,
                "valid_until": valid_until,
                "observed_rainfall": observed,
                "forecast_rainfall": forecast,
                "forecast_water_level": water,
                "status": watch_status or "Official basin forecast",
                "scraped_at": pd.Timestamp.now(tz="UTC"),
                "is_cached": False,
                "notes": notes,
            }
        ]
    )




def _is_fresh(frame: pd.DataFrame, max_hours: int = 72) -> bool:
    if frame is None or frame.empty or "issued_at" not in frame.columns:
        return False
    issued = pd.to_datetime(frame["issued_at"], errors="coerce", utc=True).max()
    if pd.isna(issued):
        return False
    age_hours = (pd.Timestamp.now(tz="UTC") - issued).total_seconds() / 3600.0
    return -6 <= age_hours <= max_hours

def _parse_samar_advisory(html: str, source_url: str) -> pd.DataFrame:
    """Extract the newest visible PAGASA Visayas advisory that mentions Samar."""
    text = BeautifulSoup(html, "html.parser").get_text("\n", strip=True)
    heading_pattern = re.compile(
        r"(?im)^(General Flood Advisory[^\n]*|Heavy Rainfall Warning[^\n]*|Rainfall Warning[^\n]*|Thunderstorm Advisory[^\n]*|Thunderstorm Watch[^\n]*)$"
    )
    matches = list(heading_pattern.finditer(text))
    segments: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segment = clean_text(text[match.start():end])
        if re.search(r"\b(?:Northern\s+Samar|Eastern\s+Samar|Samar)\b", segment, flags=re.I):
            segments.append((clean_text(match.group(1)), segment))
    if not segments:
        return empty_bulletins()

    heading, segment = segments[0]
    issued_match = re.search(r"Issued\s+at\s*:?\s*(.+?)(?=(?:Valid\s+until|Moderate|Heavy|Light|The above|All are advised|$))", segment, flags=re.I)
    issued_at = _parse_datetime_fragment(clean_text(issued_match.group(1))) if issued_match else pd.NaT

    # Keep a compact excerpt around the first Samar mention instead of storing a
    # very long regional page dump.
    samar_match = re.search(r"\b(?:Northern\s+Samar|Eastern\s+Samar|Samar)\b", segment, flags=re.I)
    start = max((samar_match.start() if samar_match else 0) - 180, 0)
    excerpt = clean_text(segment[start:start + 900])

    return pd.DataFrame(
        [
            {
                "source_name": "PAGASA Visayas regional advisory for Samar",
                "source_url": source_url,
                "basin_name": "Samar",
                "river_system": "Samar rivers (regional advisory)",
                "issued_at": issued_at,
                "valid_until": pd.NaT,
                "observed_rainfall": excerpt,
                "forecast_rainfall": "",
                "forecast_water_level": "",
                "status": heading,
                "scraped_at": pd.Timestamp.now(tz="UTC"),
                "is_cached": False,
                "notes": "Official regional rainfall/flood advisory mentioning Samar; no numerical river-gauge measurement is implied.",
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
    frames: list[pd.DataFrame] = []
    errors: list[str] = []

    index_html = ""
    flood_index: dict[str, dict[str, str]] = {}
    try:
        response = requests.get(FLOOD_INDEX_URL, timeout=35, headers=HEADERS)
        response.raise_for_status()
        index_html = response.text
        flood_index = _parse_flood_index(index_html)
    except Exception as exc:
        errors.append(f"PAGASA Flood Information index: {type(exc).__name__}: {exc}")

    for config in BULLETIN_SOURCES:
        index_record = _index_record(flood_index, config)
        watch_status = index_record.get("status", "")
        candidate_urls = []
        if index_record.get("url"):
            candidate_urls.append(index_record["url"])
        candidate_urls.append(config["source_url"])
        seen: set[str] = set()
        loaded = False
        for url in candidate_urls:
            if not url or url in seen:
                continue
            seen.add(url)
            try:
                response = requests.get(url, timeout=40, headers=HEADERS)
                response.raise_for_status()
                text = _response_text(response)
                frame = _parse_pagasa_text(text, config, response.url, watch_status)
                if frame.empty:
                    raise ValueError("no bulletin fields were found")
                if not _is_fresh(frame, max_hours=96):
                    raise ValueError("bulletin detail is missing a fresh issue time")
                frames.append(frame)
                loaded = True
                break
            except Exception as exc:
                errors.append(f"{config['source_name']} {url}: {type(exc).__name__}: {exc}")
        if not loaded and watch_status:
            frames.append(_parse_pagasa_text("", config, index_record.get("url", "") or config["source_url"], watch_status))

    samar_loaded = False
    for url in SAMAR_REGIONAL_URLS:
        try:
            response = requests.get(url, timeout=40, headers=HEADERS)
            response.raise_for_status()
            frame = _parse_samar_advisory(response.text, response.url)
            if frame.empty:
                raise ValueError("no current visible Samar advisory was found")
            if not _is_fresh(frame, max_hours=48):
                raise ValueError("the visible Samar advisory is older than 48 hours or has no issue time")
            frames.append(frame)
            samar_loaded = True
            break
        except Exception as exc:
            errors.append(f"Samar regional advisory {url}: {type(exc).__name__}: {exc}")
    if not samar_loaded:
        # No placeholder is fabricated. The target map will explicitly show that
        # no current Samar advisory was extracted.
        pass

    for url in PRFFWC_URLS:
        try:
            response = requests.get(url, timeout=30, headers=HEADERS)
            response.raise_for_status()
            frame = _parse_prffwc(response.text)
            if frame.empty:
                raise ValueError("no PRFFWC qualitative status table was found")
            frames.append(frame)
            break
        except Exception as exc:
            errors.append(f"PRFFWC {url}: {type(exc).__name__}: {exc}")

    last_success = cache / "pagasa_bulletins_last_success.csv"
    if frames:
        nonempty = [frame for frame in frames if frame is not None and not frame.empty]
        bulletins = normalize_bulletins(pd.concat(nonempty, ignore_index=True), PROVIDER_NAME)
        # Prefer the newest duplicate for the same source/river/status category.
        bulletins = bulletins.sort_values("scraped_at").drop_duplicates(
            subset=["source_name", "basin_name", "river_system"], keep="last"
        )
        bulletins.to_csv(last_success, index=False)
        return ProviderResult(
            provider=PROVIDER_NAME,
            bulletins=bulletins,
            mode="live" if not errors else "partial",
            fetched_at=pd.Timestamp.now(tz="UTC"),
            message=f"Loaded {len(bulletins)} official bulletin/advisory row(s).",
            error="",
            source_url=FLOOD_INDEX_URL,
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
            source_url=FLOOD_INDEX_URL,
            details={"live_errors": errors},
        )

    return ProviderResult(
        provider=PROVIDER_NAME,
        mode="error",
        fetched_at=pd.Timestamp.now(tz="UTC"),
        message="No hydrological bulletin could be loaded.",
        error="; ".join(errors),
        source_url=FLOOD_INDEX_URL,
    )
