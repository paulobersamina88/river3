from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageOps

from core.history import append_history_and_compute_rates
from core.schema import MANILA_TZ, ProviderResult, normalize_readings, slug

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None

PROVIDER_NAME = "PRFFWC Pampanga numerical water levels"
HOME_URLS = [
    "https://prffwc.synthasite.com/",
    "http://prffwc.synthasite.com/",
]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; PH-River-Monitor/5.5; academic research)",
    "Accept": "text/html,application/xhtml+xml,image/avif,image/webp,image/apng,*/*;q=0.8",
}

# The PRFFWC hydrological infographic uses a stable station order. Rows without
# a water-level value are safely ignored. Coordinates are municipality anchors,
# not surveyed gauge coordinates, so they are left blank and the national target
# map uses its explicit Pampanga display anchor.
STATIONS: list[dict[str, Any]] = [
    {"name": "Muñoz", "river": "Pampanga River Basin", "province": "Nueva Ecija", "municipality": "Muñoz"},
    {"name": "Sapang Buho", "river": "Pampanga River", "province": "Nueva Ecija", "municipality": "Palayan"},
    {"name": "Calaanan", "river": "Pampanga River Basin", "province": "Nueva Ecija", "municipality": "Bongabon"},
    {"name": "Mayapyap", "river": "Pampanga River", "province": "Nueva Ecija", "municipality": "Cabanatuan"},
    {"name": "Gabaldon", "river": "Pampanga River Basin", "province": "Nueva Ecija", "municipality": "Gabaldon"},
    {"name": "Palali", "river": "Pampanga River Basin", "province": "Nueva Ecija", "municipality": "General Tinio"},
    {"name": "Zaragoza", "river": "Rio Chico River / Pampanga River Basin", "province": "Tarlac", "municipality": "La Paz"},
    {"name": "Peñaranda", "river": "Peñaranda River / Pampanga River Basin", "province": "Nueva Ecija", "municipality": "Peñaranda"},
    {"name": "San Isidro", "river": "Pampanga River", "province": "Nueva Ecija", "municipality": "San Isidro"},
    {"name": "Sibul Spring", "river": "Pampanga River Basin", "province": "Bulacan", "municipality": "San Miguel"},
    {"name": "Arayat", "river": "Pampanga River", "province": "Pampanga", "municipality": "Arayat"},
    {"name": "Candaba", "river": "Candaba Swamp / Pampanga River", "province": "Pampanga", "municipality": "Candaba"},
    {"name": "Porac", "river": "Pasac-Guagua allied basin", "province": "Pampanga", "municipality": "Porac"},
    {"name": "Mexico", "river": "Abacan River / Pasac-Guagua allied basin", "province": "Pampanga", "municipality": "Mexico"},
    {"name": "San Rafael", "river": "Pampanga River Basin", "province": "Bulacan", "municipality": "San Rafael"},
    {"name": "Sasmuan", "river": "Guagua River / Pasac-Guagua allied basin", "province": "Pampanga", "municipality": "Sasmuan"},
    {"name": "Sulipan", "river": "Pampanga River", "province": "Pampanga", "municipality": "Apalit"},
    {"name": "PRFFWC", "river": "Pampanga River Basin", "province": "Pampanga", "municipality": "San Fernando"},
]


def _ocr(image: Image.Image, config: str = "--psm 6") -> str:
    if pytesseract is None:
        raise RuntimeError("pytesseract is not installed")
    enlarged = image.resize((max(image.width * 4, 1), max(image.height * 4, 1)))
    gray = ImageOps.grayscale(enlarged)
    return pytesseract.image_to_string(gray, config=config).strip()


def _parse_datetime(text: str) -> pd.Timestamp:
    match = re.search(
        r"(\d{1,2}\s+(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+20\d{2}\s+\d{1,2}:\d{2}\s*(?:AM|PM))",
        text,
        flags=re.I,
    )
    if not match:
        return pd.NaT
    ts = pd.Timestamp(match.group(1))
    if ts.tzinfo is None:
        ts = ts.tz_localize(MANILA_TZ)
    return ts.tz_convert("UTC")


def _observation_time(text: str, issued_at: pd.Timestamp) -> pd.Timestamp:
    match = re.search(
        r"WATER\s*LEVEL\s*:?\s*INSTANTANEOUS\s+DATA\s+AT\s+(\d{1,2}:\d{2}\s*(?:AM|PM))",
        text,
        flags=re.I,
    )
    if not match:
        # The infographic is commonly issued one hour after the instantaneous
        # water-level observation. Preserve that convention only as a fallback.
        return issued_at - pd.Timedelta(hours=1) if not pd.isna(issued_at) else pd.NaT
    parsed = pd.Timestamp(match.group(1))
    local_issue = issued_at.tz_convert(MANILA_TZ)
    observed = local_issue.replace(
        hour=parsed.hour,
        minute=parsed.minute,
        second=0,
        microsecond=0,
    )
    return observed.tz_convert("UTC")


def _cell_level(image: Image.Image, row_index: int) -> float:
    # Ratios are based on PRFFWC's stable 1075x629 infographic template.
    width, height = image.size
    x0, x1 = int(width * 0.697), int(width * 0.793)
    y0 = int(height * (0.2305 + row_index * 0.0321))
    y1 = int(height * (0.2305 + (row_index + 1) * 0.0321))
    crop = image.crop((x0, y0, x1, y1))
    text = _ocr(crop, "--psm 7 -c tessedit_char_whitelist=0123456789.*")
    if "*" in text and not re.search(r"\d", text):
        return np.nan
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return np.nan
    token = match.group(0)
    # OCR occasionally drops the decimal point in three-digit values such as
    # 435. The infographic reports values with two decimal places.
    if "." not in token and len(token) == 3:
        token = f"{token[0]}.{token[1:]}"
    try:
        value = float(token)
    except ValueError:
        return np.nan
    return value if 0 <= value <= 30 else np.nan


def _status_and_trend(image: Image.Image, row_index: int) -> tuple[str, str]:
    width, height = image.size
    x0, x1 = int(width * 0.792), int(width * 0.883)
    y0 = int(height * (0.2305 + row_index * 0.0321))
    y1 = int(height * (0.2305 + (row_index + 1) * 0.0321))
    crop = np.asarray(image.crop((x0, y0, x1, y1)).convert("RGB"))
    if crop.size == 0:
        return "No Threshold", ""
    r, g, b = crop[..., 0], crop[..., 1], crop[..., 2]
    green = (g > 160) & (r < 130) & (b < 130)
    yellow = (r > 180) & (g > 180) & (b < 130)
    orange = (r > 180) & (g > 70) & (g <= 190) & (b < 130)
    red = (r > 170) & (g < 100) & (b < 100)
    masks = {
        "Critical": red,
        "Alarm": orange,
        "Alert": yellow,
        "Normal": green,
    }
    status, mask = max(masks.items(), key=lambda item: int(item[1].sum()))
    if int(mask.sum()) < 5:
        return "No Threshold", ""

    ys, _ = np.where(mask)
    if len(ys) < 5:
        return status, ""
    top = int(mask[: max(crop.shape[0] // 3, 1)].sum())
    bottom = int(mask[-max(crop.shape[0] // 3, 1) :].sum())
    total = int(mask.sum())
    # Filled circles have similar top and bottom areas, triangles do not.
    if abs(top - bottom) <= max(int(total * 0.15), 2):
        trend = "No significant change"
    elif bottom > top:
        trend = "Rising"
    else:
        trend = "Receding"
    return status, trend


def parse_infographic(image_bytes: bytes, source_url: str) -> pd.DataFrame:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if image.width < 700 or image.height < 350:
        return pd.DataFrame()
    # Resize to the reference aspect ratio to make row crops deterministic.
    image = image.resize((1075, 629))
    header_text = _ocr(image.crop((0, 0, 960, 155)), "--psm 6")
    normalized_header = re.sub(r"\s+", " ", header_text).upper()
    if "PAMPANGA RIVER BASIN" not in normalized_header or "WATER" not in normalized_header:
        return pd.DataFrame()
    issued_at = _parse_datetime(normalized_header)
    if pd.isna(issued_at):
        return pd.DataFrame()
    observed_at = _observation_time(normalized_header, issued_at)

    rows = []
    for index, station in enumerate(STATIONS):
        level = _cell_level(image, index)
        if pd.isna(level):
            continue
        status, trend = _status_and_trend(image, index)
        rows.append(
            {
                "station_id": f"PRFFWC-{slug(station['name']).upper()}",
                "station_name": station["name"],
                "river_system": station["river"],
                "basin_name": "Pampanga",
                "region": "Region III",
                "province": station["province"],
                "municipality": station["municipality"],
                "location": f"{station['name']}, {station['municipality']}",
                "lat": np.nan,
                "lon": np.nan,
                "timestamp": observed_at,
                "level_m": level,
                "level_30min_ago_m": np.nan,
                "level_1hr_ago_m": np.nan,
                "level_2hr_ago_m": np.nan,
                "rise_rate_m_hr": np.nan,
                "alert_m": np.nan,
                "alarm_m": np.nan,
                "critical_m": np.nan,
                "threshold_status": status,
                "source_trend": trend,
                "source_name": PROVIDER_NAME,
                "source_url": source_url,
                "data_kind": "official PRFFWC infographic gauge",
                "scraped_at": pd.Timestamp.now(tz="UTC"),
                "is_cached": False,
                "notes": (
                    "Numerical value, official threshold color, and trend symbol were extracted from "
                    "the latest PRFFWC hydrological infographic. Gauge datums differ by station."
                ),
            }
        )
    return normalize_readings(pd.DataFrame(rows), PROVIDER_NAME)


def _image_candidates(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []
    for image in soup.find_all("img"):
        src = image.get("src") or image.get("data-src") or image.get("data-original")
        if not src:
            continue
        url = urljoin(base_url, src)
        if url not in urls:
            urls.append(url)
    # The hydrological infographic is usually a large resource image. Prioritize
    # likely filenames but still inspect every image if names change.
    return sorted(
        urls,
        key=lambda url: (
            0 if any(term in url.lower() for term in ["hydro", "forecast", "prffwc", "waterlevel", "water-level"]) else 1,
            url,
        ),
    )


def _from_static_page(url: str) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []
    response = requests.get(url, timeout=35, headers=HEADERS, allow_redirects=True)
    response.raise_for_status()
    for image_url in _image_candidates(response.text, response.url):
        try:
            image_response = requests.get(image_url, timeout=35, headers=HEADERS)
            image_response.raise_for_status()
            frame = parse_infographic(image_response.content, image_response.url)
            if not frame.empty:
                return frame, errors
        except Exception as exc:
            errors.append(f"image {image_url}: {type(exc).__name__}: {exc}")
    return pd.DataFrame(), errors


def _from_browser(url: str, timeout_ms: int) -> tuple[pd.DataFrame, list[str]]:
    from playwright.sync_api import sync_playwright

    errors: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = browser.new_page(viewport={"width": 1600, "height": 1400})
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        try:
            page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 30000))
        except Exception:
            pass
        images = page.locator("img")
        for index in range(images.count()):
            locator = images.nth(index)
            try:
                box = locator.bounding_box()
                if not box or box["width"] < 650 or box["height"] < 300:
                    continue
                data = locator.screenshot(type="png")
                frame = parse_infographic(data, page.url)
                if not frame.empty:
                    browser.close()
                    return frame, errors
            except Exception as exc:
                errors.append(f"browser image {index}: {type(exc).__name__}: {exc}")
        browser.close()
    return pd.DataFrame(), errors


def fetch(cache_dir: str | Path, timeout_ms: int = 90000) -> ProviderResult:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    last_success = cache / "prffwc_water_levels_last_success.csv"
    history_path = cache / "prffwc_water_levels_history.csv"
    errors: list[str] = []

    for url in HOME_URLS:
        try:
            readings, partial = _from_static_page(url)
            errors.extend(partial)
            if not readings.empty:
                readings = append_history_and_compute_rates(readings, history_path)
                readings.to_csv(last_success, index=False)
                return ProviderResult(
                    provider=PROVIDER_NAME,
                    readings=readings,
                    mode="live-image",
                    fetched_at=pd.Timestamp.now(tz="UTC"),
                    message=f"Loaded {len(readings)} current PRFFWC gauge reading(s) from the official infographic.",
                    source_url=url,
                    details={"partial_errors": errors},
                )
            raise ValueError("no current PRFFWC hydrological infographic was parsed")
        except Exception as exc:
            errors.append(f"static {url}: {type(exc).__name__}: {exc}")

        try:
            readings, partial = _from_browser(url, timeout_ms=timeout_ms)
            errors.extend(partial)
            if not readings.empty:
                readings = append_history_and_compute_rates(readings, history_path)
                readings.to_csv(last_success, index=False)
                return ProviderResult(
                    provider=PROVIDER_NAME,
                    readings=readings,
                    mode="live-browser-image",
                    fetched_at=pd.Timestamp.now(tz="UTC"),
                    message=f"Loaded {len(readings)} current PRFFWC gauge reading(s) after browser rendering.",
                    source_url=url,
                    details={"partial_errors": errors},
                )
            raise ValueError("no rendered PRFFWC infographic was parsed")
        except Exception as exc:
            errors.append(f"browser {url}: {type(exc).__name__}: {exc}")

    if last_success.exists():
        cached = pd.read_csv(last_success)
        cached["is_cached"] = True
        return ProviderResult(
            provider=PROVIDER_NAME,
            readings=normalize_readings(cached, PROVIDER_NAME),
            mode="cache",
            fetched_at=pd.Timestamp.fromtimestamp(last_success.stat().st_mtime, tz="UTC"),
            message="The current PRFFWC infographic could not be read; showing the last successful Pampanga cache.",
            source_url=HOME_URLS[0],
            details={"live_errors": errors},
        )

    return ProviderResult(
        provider=PROVIDER_NAME,
        mode="error",
        fetched_at=pd.Timestamp.now(tz="UTC"),
        message="No current PRFFWC numerical water-level readings could be extracted.",
        error="; ".join(errors),
        source_url=HOME_URLS[0],
    )
