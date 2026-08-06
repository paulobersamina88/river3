from __future__ import annotations

import io
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from core.schema import (
    MANILA_TZ,
    ProviderResult,
    classify_threshold,
    clean_text,
    normalize_readings,
    parse_number,
    slug,
)

PROVIDER_NAME = "PAGASA Pasig-Marikina-Tullahan FFWS"
SOURCE_URL = "https://pasig-marikina-tullahanffws.pagasa.dost.gov.ph/water/table.do"

MARIKINA_TERMS = [
    "marikina", "sto nino", "sto. nino", "st nino", "nangka", "montalban",
    "rosario", "san mateo", "batasan", "tumana",
]
TULLAHAN_TERMS = [
    "tullahan", "quirino", "novaliches", "fairview", "valenzuela", "malabon",
    "navotas", "caloocan", "dam site", "la mesa",
]


def _find_browser() -> str | None:
    explicit = __import__("os").getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE", "").strip()
    if explicit and Path(explicit).exists():
        return explicit
    for name in ["chromium", "chromium-browser", "google-chrome", "google-chrome-stable"]:
        found = shutil.which(name)
        if found:
            return found
    for path in ["/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/google-chrome"]:
        if Path(path).exists():
            return path
    return None


def _render_html(url: str, timeout_ms: int) -> str:
    from playwright.sync_api import sync_playwright

    executable = _find_browser()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            executable_path=executable,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1200})
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        try:
            page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 30_000))
        except Exception:
            pass
        page.wait_for_timeout(3000)
        html = page.content()
        browser.close()
        return html


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if isinstance(output.columns, pd.MultiIndex):
        output.columns = [
            " ".join(
                str(part).strip()
                for part in column
                if str(part).strip() and not str(part).lower().startswith("unnamed")
            ).strip()
            for column in output.columns
        ]
    else:
        output.columns = [str(column).strip() for column in output.columns]
    return output


def _column_key(value: Any) -> str:
    text = clean_text(value).lower()
    text = text.replace("−", "-")
    return re.sub(r"[^a-z0-9+-]+", " ", text).strip()


def _pick_column(columns: list[str], terms: list[str]) -> str | None:
    keyed = {column: _column_key(column) for column in columns}
    for term in terms:
        term_key = _column_key(term)
        for column, key in keyed.items():
            if term_key and term_key in key:
                return column
    return None


def _candidate_tables(html: str) -> list[pd.DataFrame]:
    try:
        return [_flatten_columns(table) for table in pd.read_html(io.StringIO(html))]
    except Exception:
        return []


def _extract_page_time(html: str) -> pd.Timestamp:
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    patterns = [
        r"\bTime\s*:\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{2})",
        r"\bTime\s*:\s*(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2})",
        r"\bTime\s*:\s*(\d{1,2}:\d{2})",
    ]
    now = pd.Timestamp.now(tz=MANILA_TZ)
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if not match:
            continue
        value = match.group(1)
        try:
            if re.fullmatch(r"\d{1,2}:\d{2}", value):
                ts = pd.Timestamp(f"{now.date()} {value}")
            else:
                ts = pd.Timestamp(value)
            if ts.tzinfo is None:
                ts = ts.tz_localize(MANILA_TZ)
            return ts.tz_convert("UTC")
        except Exception:
            continue
    return now.tz_convert("UTC")


def infer_river_system(station_name: str) -> tuple[str, str, str, str, str]:
    text = clean_text(station_name).lower()
    if any(term in text for term in MARIKINA_TERMS):
        return "Marikina River", "Pasig-Laguna", "NCR", "Metro Manila", ""
    if any(term in text for term in TULLAHAN_TERMS):
        return "Tullahan River", "Pasig-Laguna", "NCR", "Metro Manila", ""
    return "Pasig-Marikina-Tullahan system", "Pasig-Laguna", "NCR", "Metro Manila", ""


def parse_html(html: str) -> pd.DataFrame:
    page_time = _extract_page_time(html)
    rows: list[dict[str, Any]] = []
    for table in _candidate_tables(html):
        if table.empty:
            continue
        columns = list(table.columns)
        station_col = _pick_column(columns, ["station"])
        current_col = _pick_column(columns, ["current", "observed wl current"])
        alert_col = _pick_column(columns, ["alert"])
        alarm_col = _pick_column(columns, ["alarm"])
        critical_col = _pick_column(columns, ["critical"])
        minus30_col = _pick_column(columns, ["-30 min", "30 min"])
        minus1h_col = _pick_column(columns, ["-1 hr", "1 hr"])
        minus2h_col = _pick_column(columns, ["-2 hr", "2 hr"])
        if not station_col or not current_col:
            continue
        for _, row in table.iterrows():
            station = clean_text(row.get(station_col))
            if not station or station.lower() in {"no data", "no data.", "nan", "station"}:
                continue
            current = parse_number(row.get(current_col))
            if pd.isna(current):
                continue
            previous_1h = parse_number(row.get(minus1h_col)) if minus1h_col else np.nan
            previous_30 = parse_number(row.get(minus30_col)) if minus30_col else np.nan
            rate = current - previous_1h if pd.notna(previous_1h) else (
                (current - previous_30) * 2 if pd.notna(previous_30) else np.nan
            )
            alert = parse_number(row.get(alert_col)) if alert_col else np.nan
            alarm = parse_number(row.get(alarm_col)) if alarm_col else np.nan
            critical = parse_number(row.get(critical_col)) if critical_col else np.nan
            river, basin, region, province, municipality = infer_river_system(station)
            status = classify_threshold(current, alert, alarm, critical)
            trend = "Stable"
            if pd.notna(rate) and rate > 0.005:
                trend = "Rising"
            elif pd.notna(rate) and rate < -0.005:
                trend = "Falling"
            rows.append(
                {
                    "station_id": f"PMT-{slug(station).upper()}",
                    "station_name": station,
                    "river_system": river,
                    "basin_name": basin,
                    "region": region,
                    "province": province,
                    "municipality": municipality,
                    "location": station,
                    "lat": np.nan,
                    "lon": np.nan,
                    "timestamp": page_time,
                    "level_m": current,
                    "rise_rate_m_hr": rate,
                    "alert_m": alert,
                    "alarm_m": alarm,
                    "critical_m": critical,
                    "threshold_status": status,
                    "source_trend": trend,
                    "source_name": PROVIDER_NAME,
                    "source_url": SOURCE_URL,
                    "data_kind": "instrument",
                    "scraped_at": pd.Timestamp.now(tz="UTC"),
                    "is_cached": False,
                    "notes": "PAGASA values are elevation levels (EL.m); compare only with the same station thresholds.",
                    "level_30min_ago_m": previous_30,
                    "level_1hr_ago_m": previous_1h,
                    "level_2hr_ago_m": parse_number(row.get(minus2h_col)) if minus2h_col else np.nan,
                }
            )
        if rows:
            break
    return pd.DataFrame(rows)


def fetch(cache_dir: str | Path, timeout_ms: int = 70_000) -> ProviderResult:
    cache_path = Path(cache_dir) / "pagasa_pmt_last_success.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    html = ""
    try:
        response = requests.get(SOURCE_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0 river-monitor-academic"})
        response.raise_for_status()
        html = response.text
        readings = parse_html(html)
        if readings.empty:
            raise ValueError("the direct page contained no numerical station rows")
    except Exception as exc:
        errors.append(f"direct request: {type(exc).__name__}: {exc}")
        try:
            html = _render_html(SOURCE_URL, timeout_ms=timeout_ms)
            readings = parse_html(html)
            if readings.empty:
                raise ValueError("the rendered page contained no numerical station rows")
        except Exception as browser_exc:
            errors.append(f"browser: {type(browser_exc).__name__}: {browser_exc}")
            if cache_path.exists():
                cached = pd.read_csv(cache_path)
                cached["is_cached"] = True
                return ProviderResult(
                    provider=PROVIDER_NAME,
                    readings=normalize_readings(cached, PROVIDER_NAME),
                    mode="cache",
                    fetched_at=pd.Timestamp.fromtimestamp(cache_path.stat().st_mtime, tz="UTC"),
                    message="Live PAGASA PMT table had no usable data; showing the last successful cache.",
                    error="",
                    source_url=SOURCE_URL,
                    details={"retrieval_errors": errors},
                )
            return ProviderResult(
                provider=PROVIDER_NAME,
                mode="empty",
                fetched_at=pd.Timestamp.now(tz="UTC"),
                message="PAGASA PMT currently returned no numerical station rows.",
                error="; ".join(errors),
                source_url=SOURCE_URL,
            )

    normalized = normalize_readings(readings, PROVIDER_NAME)
    normalized.to_csv(cache_path, index=False)
    return ProviderResult(
        provider=PROVIDER_NAME,
        readings=normalized,
        mode="live",
        fetched_at=pd.Timestamp.now(tz="UTC"),
        message=f"Loaded {len(normalized)} PAGASA PMT station reading(s).",
        error="",
        source_url=SOURCE_URL,
        details={"retrieval_errors": errors},
    )
