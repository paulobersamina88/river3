"""Utilities for reading the public DOST-ASTI PhilSensors water-level table.

This is an unofficial webpage integration. It intentionally uses a slow refresh
interval, a last-successful-data fallback, and no login/session credentials.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MANILA_TZ = "Asia/Manila"
DEFAULT_URL = "https://philsensors.asti.dost.gov.ph/site/waterlevel"
SCRAPER_VERSION = "3.5.0"
STATION_METADATA_URL = "https://philsensors.asti.dost.gov.ph/site/data"

_REQUIRED_HEADERS = {"region", "province", "location"}
_OBSERVATION_RE = re.compile(r"^(?:current\s*hour|\d{1,2}:\d{2}(?:\s*[ap]m)?)$", re.I)
_LEVEL_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:m|meter|metre)\b", re.I)
_PLAIN_LEVEL_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*$")
_RGB_RE = re.compile(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)")

# Approximate fallback colors seen in the PhilSensors threshold legend.
_FALLBACK_STATUS_COLORS: dict[str, list[tuple[int, int, int]]] = {
    "Normal": [(152, 190, 211), (159, 198, 220), (173, 216, 230)],
    "Alert": [(238, 229, 141), (245, 232, 112), (255, 235, 59)],
    "Alarm": [(250, 183, 94), (255, 183, 77), (255, 152, 0)],
    "Critical": [(255, 139, 124), (255, 128, 112), (244, 67, 54)],
    "No Threshold": [(211, 214, 217), (207, 210, 213), (224, 224, 224)],
}


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _norm(value: Any) -> str:
    text = _clean_text(value).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def stable_station_id(region: Any, province: Any, location: Any) -> str:
    key = "|".join([_clean_text(region), _clean_text(province), _clean_text(location)]).lower()
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12].upper()
    return f"PHIL-{digest}"


def find_chromium_executable() -> str | None:
    """Find a system Chromium/Chrome executable, or use an explicit override."""
    explicit = os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE", "").strip()
    if explicit and Path(explicit).exists():
        return explicit

    for candidate in [
        "chromium",
        "chromium-browser",
        "google-chrome",
        "google-chrome-stable",
        "microsoft-edge",
        "msedge",
    ]:
        found = shutil.which(candidate)
        if found:
            return found

    common_paths = [
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/usr/bin/google-chrome",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for candidate in common_paths:
        if Path(candidate).exists():
            return candidate
    return None


def _render_table_payload(url: str, timeout_ms: int = 90_000) -> dict[str, Any]:
    """Render the PhilSensors DataTable and extract its underlying cell values.

    PhilSensors uses a JavaScript/DataTables view. Depending on the browser build,
    the visible table can be split into cloned header/body tables, and some values
    may live in DataTables' internal row data or element attributes rather than in
    ``td.textContent``. This extractor checks all three representations.
    """
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - depends on deployment image
        raise RuntimeError(
            "Playwright is not installed. Add 'playwright' to requirements.txt."
        ) from exc

    script = r"""
    () => {
      const clean = (value) => String(value ?? '').replace(/\s+/g, ' ').trim();
      const norm = (value) => clean(value).toLowerCase().replace(/[^a-z0-9]+/g, '');
      const levelRe = /-?\d+(?:\.\d+)?\s*(?:m|meter|metre)\b/i;
      const unique = (values) => [...new Set(values.map(clean).filter(Boolean))];

      const usefulColor = (element) => {
        if (!element) return '';
        const style = getComputedStyle(element);
        const color = style.backgroundColor || '';
        if (!color || color === 'transparent' || color === 'rgba(0, 0, 0, 0)') return '';
        return color;
      };

      const pseudoText = (element, pseudo) => {
        try {
          const value = getComputedStyle(element, pseudo).content || '';
          return clean(value.replace(/^['\"]|['\"]$/g, ''));
        } catch (_) {
          return '';
        }
      };

      const htmlInfo = (rawValue) => {
        const raw = rawValue == null ? '' : String(rawValue);
        const holder = document.createElement('div');
        holder.innerHTML = raw;
        const first = holder.firstElementChild;
        const descendants = Array.from(holder.querySelectorAll('*'));
        const attributeValues = descendants.flatMap((el) =>
          Array.from(el.attributes || []).map((attr) => attr.value)
        );
        const inputValues = descendants.flatMap((el) => {
          const values = [];
          if ('value' in el && el.value != null) values.push(el.value);
          if (el.dataset) values.push(...Object.values(el.dataset));
          return values;
        });
        const candidates = unique([
          holder.innerText,
          holder.textContent,
          raw,
          ...attributeValues,
          ...inputValues,
        ]);
        const preferred = candidates.find((value) => levelRe.test(value)) || candidates[0] || '';
        return {
          text: preferred,
          innerText: clean(holder.innerText),
          rawHtml: raw,
          candidates,
          className: first ? clean(first.className) : '',
          style: first ? (first.getAttribute('style') || '') : '',
          backgroundColor: first ? usefulColor(first) : '',
          title: first ? (first.getAttribute('title') || '') : '',
          ariaLabel: first ? (first.getAttribute('aria-label') || '') : '',
          dataset: first && first.dataset ? {...first.dataset} : {},
          pseudoBefore: '',
          pseudoAfter: '',
        };
      };

      const domCell = (cell) => {
        const descendants = Array.from(cell.querySelectorAll('*'));
        const attributeValues = [cell, ...descendants].flatMap((el) =>
          Array.from(el.attributes || []).map((attr) => attr.value)
        );
        const inputValues = [cell, ...descendants].flatMap((el) => {
          const values = [];
          if ('value' in el && el.value != null) values.push(el.value);
          if (el.dataset) values.push(...Object.values(el.dataset));
          return values;
        });
        const before = pseudoText(cell, '::before');
        const after = pseudoText(cell, '::after');
        const candidates = unique([
          cell.innerText,
          cell.textContent,
          cell.getAttribute('data-order'),
          cell.getAttribute('data-sort'),
          cell.getAttribute('data-search'),
          cell.getAttribute('data-filter'),
          cell.getAttribute('data-value'),
          cell.getAttribute('value'),
          cell.getAttribute('title'),
          cell.getAttribute('aria-label'),
          before,
          after,
          ...attributeValues,
          ...inputValues,
          cell.innerHTML,
        ]);
        const preferred = candidates.find((value) => levelRe.test(value)) || candidates[0] || '';
        return {
          text: preferred,
          innerText: clean(cell.innerText),
          rawHtml: cell.innerHTML || '',
          candidates,
          className: clean(cell.className),
          style: cell.getAttribute('style') || '',
          backgroundColor: usefulColor(cell),
          title: cell.getAttribute('title') || '',
          ariaLabel: cell.getAttribute('aria-label') || '',
          dataset: cell.dataset ? {...cell.dataset} : {},
          pseudoBefore: before,
          pseudoAfter: after,
        };
      };

      const findLegendColor = (label) => {
        const all = Array.from(document.querySelectorAll('body *'));
        const target = all.find((el) => clean(el.textContent).toLowerCase() === label.toLowerCase());
        if (!target) return '';
        const candidates = [
          target,
          target.previousElementSibling,
          target.nextElementSibling,
          ...(target.parentElement ? Array.from(target.parentElement.children) : []),
        ];
        for (const candidate of candidates) {
          const color = usefulColor(candidate);
          if (color && color !== 'rgb(255, 255, 255)') return color;
        }
        return '';
      };

      const headersFor = (table) => {
        let cells = Array.from(table.querySelectorAll('thead tr:last-child th'));
        if (!cells.length) {
          const firstRow = table.querySelector('tr');
          cells = firstRow ? Array.from(firstRow.querySelectorAll('th, td')) : [];
        }
        return cells.map((cell) => clean(cell.innerText || cell.textContent));
      };

      const isTargetHeaders = (headers) => {
        const normalized = headers.map(norm);
        return normalized.includes('region') &&
               normalized.includes('province') &&
               normalized.includes('location') &&
               normalized.some((value) => value === 'currenthour');
      };

      const domRows = (table) => Array.from(table.querySelectorAll('tbody tr'))
        .map((row) => Array.from(row.cells || row.querySelectorAll('td')).map(domCell))
        .filter((row) => row.length > 0);

      const getNested = (object, path) => {
        if (object == null || path == null) return '';
        if (typeof path === 'number') return object[path];
        if (typeof path !== 'string') return '';
        return path.split('.').reduce((value, key) => value == null ? '' : value[key], object);
      };

      const dataTableRows = (table) => {
        try {
          if (!(window.jQuery && jQuery.fn && jQuery.fn.dataTable &&
                jQuery.fn.dataTable.isDataTable(table))) return [];
          const dt = jQuery(table).DataTable();
          const settings = dt.settings()[0];
          const sources = (settings.aoColumns || []).map((column, index) =>
            column.mData == null ? index : column.mData
          );
          return dt.rows({search: 'applied'}).data().toArray().map((record) => {
            let values;
            if (Array.isArray(record)) {
              values = record;
            } else if (record && typeof record === 'object') {
              values = sources.map((source, index) => {
                if (typeof source === 'function') {
                  try { return source(record, 'display', undefined, {row: 0, col: index}); }
                  catch (_) { return ''; }
                }
                return getNested(record, source);
              });
            } else {
              values = [record];
            }
            return values.map(htmlInfo);
          }).filter((row) => row.length > 0);
        } catch (_) {
          return [];
        }
      };

      const wrapperRows = (table) => {
        const wrapper = table.closest('.dataTables_wrapper, .dt-container') || table.parentElement;
        if (!wrapper) return [];
        const result = [];
        wrapper.querySelectorAll('table tbody tr').forEach((row) => {
          const cells = Array.from(row.cells || row.querySelectorAll('td'));
          if (cells.length) result.push(cells.map(domCell));
        });
        return result;
      };

      const numericScore = (rows, headers) => {
        const observationIndexes = headers.map((header, index) =>
          /^(?:current\s*hour|\d{1,2}:\d{2}(?:\s*[ap]m)?)$/i.test(clean(header)) ? index : -1
        ).filter((index) => index >= 0);
        let levels = 0;
        let locations = 0;
        const locationIndex = headers.map(norm).indexOf('location');
        rows.forEach((row) => {
          if (locationIndex >= 0 && row[locationIndex] && clean(row[locationIndex].text)) locations += 1;
          observationIndexes.forEach((index) => {
            const cell = row[index] || {};
            const values = [cell.text, ...(cell.candidates || []), cell.rawHtml];
            if (values.some((value) => levelRe.test(clean(value)))) levels += 1;
          });
        });
        return levels * 1000 + locations;
      };

      const tables = Array.from(document.querySelectorAll('table'));
      const targetTables = tables.map((table) => ({table, headers: headersFor(table)}))
        .filter((entry) => isTargetHeaders(entry.headers));
      if (!targetTables.length) {
        throw new Error('Water-level table with Region/Province/Location/Current Hour headers was not found.');
      }

      let best = null;
      for (const entry of targetTables) {
        const candidates = [
          {source: 'same-table DOM', rows: domRows(entry.table)},
          {source: 'DataTables internal data', rows: dataTableRows(entry.table)},
          {source: 'DataTables wrapper DOM', rows: wrapperRows(entry.table)},
          {source: 'all visible table rows', rows: Array.from(document.querySelectorAll('tbody tr')).map((row) => Array.from(row.cells || row.querySelectorAll('td')).map(domCell)).filter((row) => row.length === entry.headers.length)},
        ];
        for (const candidate of candidates) {
          const score = numericScore(candidate.rows, entry.headers);
          if (!best || score > best.score) {
            best = {...entry, ...candidate, score};
          }
        }
      }

      return {
        headers: best.headers,
        rows: best.rows,
        extractionSource: best.source,
        extractionScore: best.score,
        pageTitle: document.title,
        pageUrl: location.href,
        bodyHasMeterValue: /-?\d+(?:\.\d+)?\s*m\b/i.test(document.body.innerText || ''),
        bodyTextSample: clean(document.body.innerText || '').slice(0, 1000),
        legendColors: {
          Normal: findLegendColor('Normal'),
          Alert: findLegendColor('Alert'),
          Alarm: findLegendColor('Alarm'),
          Critical: findLegendColor('Critical'),
          'No Threshold': findLegendColor('No Threshold'),
        },
      };
    }
    """

    failed_requests: list[str] = []
    response_diagnostics: list[str] = []
    network_payloads: list[dict[str, Any]] = []

    with sync_playwright() as playwright:
        launch_kwargs: dict[str, Any] = {
            "headless": True,
            "args": [
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-blink-features=AutomationControlled",
            ],
        }
        executable = find_chromium_executable()
        if executable:
            launch_kwargs["executable_path"] = executable

        browser = playwright.chromium.launch(**launch_kwargs)
        try:
            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                ),
                locale="en-PH",
                timezone_id=MANILA_TZ,
                java_script_enabled=True,
                ignore_https_errors=True,
            )
            context.add_init_script(
                """
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                Object.defineProperty(navigator, 'languages', {get: () => ['en-PH', 'en-US', 'en']});
                Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                window.chrome = window.chrome || {runtime: {}};
                """
            )
            page = context.new_page()
            page.on(
                "requestfailed",
                lambda request: failed_requests.append(
                    f"{request.resource_type}: {request.url} :: {request.failure or 'failed'}"
                ),
            )
            def handle_response(response):
                if response.request.resource_type not in {"xhr", "fetch"}:
                    return
                response_diagnostics.append(
                    f"{response.status} {response.request.resource_type} {response.url}"
                )
                try:
                    content_type = response.headers.get("content-type", "").lower()
                    if response.status != 200 or not any(
                        token in content_type for token in ["json", "text", "html", "javascript"]
                    ):
                        return
                    text = response.text()
                    if text and len(text) <= 5_000_000:
                        network_payloads.append(
                            {
                                "url": response.url,
                                "content_type": content_type,
                                "text": text,
                            }
                        )
                except Exception:
                    pass

            page.on("response", handle_response)

            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 30_000))
            except PlaywrightTimeoutError:
                pass
            page.wait_for_selector("table", timeout=timeout_ms)

            page.wait_for_function(
                """
                () => Array.from(document.querySelectorAll('table')).some((table) => {
                    const headers = Array.from(table.querySelectorAll('th')).map(
                        (cell) => (cell.innerText || cell.textContent || '').toLowerCase()
                    );
                    return headers.some((x) => x.includes('region')) &&
                           headers.some((x) => x.includes('province')) &&
                           headers.some((x) => x.includes('location')) &&
                           headers.some((x) => x.includes('current hour'));
                })
                """,
                timeout=timeout_ms,
            )

            # Ask DataTables for all rows, then wait for actual meter values. If the
            # visual DOM never exposes them, the evaluator still reads internal data.
            page.evaluate(
                """
                () => {
                  try {
                    if (window.jQuery && jQuery.fn && jQuery.fn.dataTable) {
                      document.querySelectorAll('table').forEach((table) => {
                        if (jQuery.fn.dataTable.isDataTable(table)) {
                          const dt = jQuery(table).DataTable();
                          dt.page.len(-1).draw(false);
                        }
                      });
                    }
                  } catch (_) {}
                }
                """
            )
            try:
                page.wait_for_function(
                    r"""
                    () => {
                      const meter = /-?\d+(?:\.\d+)?\s*m\b/i;
                      if (meter.test(document.body.innerText || '')) return true;
                      try {
                        if (window.jQuery && jQuery.fn && jQuery.fn.dataTable) {
                          return Array.from(document.querySelectorAll('table')).some((table) => {
                            if (!jQuery.fn.dataTable.isDataTable(table)) return false;
                            const data = jQuery(table).DataTable().rows().data().toArray();
                            return meter.test(JSON.stringify(data));
                          });
                        }
                      } catch (_) {}
                      return false;
                    }
                    """,
                    timeout=min(timeout_ms, 60_000),
                )
            except PlaywrightTimeoutError:
                # Continue and return rich diagnostics rather than failing before
                # DataTables' internal representation can be inspected.
                page.wait_for_timeout(3_000)

            page.wait_for_timeout(5_000)

            frame_payloads: list[dict[str, Any]] = []
            for frame in page.frames:
                try:
                    candidate = frame.evaluate(script)
                    candidate["frameUrl"] = frame.url
                    frame_payloads.append(candidate)
                except Exception:
                    continue

            if not frame_payloads:
                raise RuntimeError("No PhilSensors water-level table could be extracted from any browser frame.")

            payload = max(
                frame_payloads,
                key=lambda item: float(item.get("extractionScore", 0) or 0),
            )
            payload["scrapedAt"] = pd.Timestamp.now(tz=MANILA_TZ).isoformat()
            payload["failedRequests"] = failed_requests[-20:]
            payload["xhrResponses"] = response_diagnostics[-30:]
            payload["networkPayloads"] = network_payloads[-20:]
            payload["scraperVersion"] = SCRAPER_VERSION
            return payload
        finally:
            browser.close()

def _parse_rgb(value: str) -> tuple[int, int, int] | None:
    match = _RGB_RE.search(value or "")
    if not match:
        return None
    return tuple(int(match.group(index)) for index in range(1, 4))


def _color_distance(first: tuple[int, int, int], second: tuple[int, int, int]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def _strip_html(value: Any) -> str:
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", str(value or ""), flags=re.I | re.S)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return _clean_text(text)


def _cell_candidates(cell: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in [
        "text",
        "innerText",
        "title",
        "ariaLabel",
        "pseudoBefore",
        "pseudoAfter",
        "rawHtml",
        "className",
        "style",
    ]:
        values.append(cell.get(key, ""))

    candidates = cell.get("candidates", [])
    if isinstance(candidates, (list, tuple)):
        values.extend(candidates)

    dataset = cell.get("dataset", {})
    if isinstance(dataset, dict):
        values.extend(dataset.values())

    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        raw = _clean_text(value)
        for candidate in [raw, _strip_html(raw)]:
            if candidate and candidate not in seen:
                seen.add(candidate)
                output.append(candidate)
    return output


def _detect_threshold_status(
    cell: dict[str, Any], legend_colors: dict[str, str] | None = None
) -> str:
    combined = " ".join(_cell_candidates(cell)).lower()

    explicit_patterns = {
        "Critical": ["critical", "danger", "bg-critical"],
        "Alarm": ["alarm", "warning-orange", "bg-alarm"],
        "Alert": ["alert", "warning-yellow", "bg-alert"],
        "Normal": ["normal", "safe", "bg-normal"],
        "No Threshold": [
            "no-threshold",
            "nothreshold",
            "no threshold",
            "secondary",
            "bg-default",
        ],
    }
    for status, patterns in explicit_patterns.items():
        if any(pattern in combined for pattern in patterns):
            return status

    observed = _parse_rgb(str(cell.get("backgroundColor", "")))
    if not observed:
        # Inline styles from DataTables' raw HTML can still carry the color.
        observed = _parse_rgb(str(cell.get("style", "")))
    if not observed:
        return "No Threshold"

    candidates: dict[str, list[tuple[int, int, int]]] = {
        key: list(values) for key, values in _FALLBACK_STATUS_COLORS.items()
    }
    for status, color in (legend_colors or {}).items():
        parsed = _parse_rgb(color)
        if parsed:
            candidates.setdefault(status, []).insert(0, parsed)

    best_status = "No Threshold"
    best_distance = float("inf")
    for status, colors in candidates.items():
        for color in colors:
            distance = _color_distance(observed, color)
            if distance < best_distance:
                best_status = status
                best_distance = distance

    return best_status if best_distance <= 95 else "No Threshold"


def _parse_level_and_trend_from_cell(cell: dict[str, Any]) -> tuple[float | None, str]:
    candidates = _cell_candidates(cell)
    matched_text = ""
    level: float | None = None

    # Prefer values explicitly carrying the metre unit so hours, station numbers,
    # RGB values, and CSS dimensions cannot be mistaken for a water level.
    for candidate in candidates:
        match = _LEVEL_RE.search(candidate)
        if match:
            level = float(match.group(1))
            matched_text = candidate
            break

    # Some DataTables configurations store only a bare numeric display value.
    if level is None:
        for candidate in candidates:
            match = _PLAIN_LEVEL_RE.match(candidate)
            if match:
                value = float(match.group(1))
                if -50.0 <= value <= 100.0:
                    level = value
                    matched_text = candidate
                    break

    if level is None:
        return None, "Unknown"

    combined = " ".join(candidates).lower()
    if (
        "↑" in combined
        or "▲" in combined
        or re.search(r"(?:arrow|caret|chevron)[-_ ]?up", combined)
        or re.search(r"\b(?:rise|rising|upward)\b", combined)
    ):
        trend = "Rise"
    elif (
        "↓" in combined
        or "▼" in combined
        or re.search(r"(?:arrow|caret|chevron)[-_ ]?down", combined)
        or re.search(r"\b(?:fall|falling|downward)\b", combined)
    ):
        trend = "Fall"
    else:
        # Keep matched_text referenced for easier debugging and future extension.
        _ = matched_text
        trend = "Steady/Unknown"
    return level, trend

def _timestamp_for_header(header: str, now: pd.Timestamp) -> pd.Timestamp:
    clean = _clean_text(header).lower()
    if "current" in clean:
        return now.floor("h")

    parsed = pd.to_datetime(header, errors="coerce")
    if pd.isna(parsed):
        return now.floor("h")

    candidate = now.normalize() + pd.Timedelta(
        hours=int(parsed.hour), minutes=int(parsed.minute)
    )
    if candidate > now + pd.Timedelta(minutes=5):
        candidate -= pd.Timedelta(days=1)
    return candidate



def _simple_cell(value: Any) -> dict[str, Any]:
    raw = "" if value is None else str(value)
    return {
        "text": _strip_html(raw),
        "innerText": _strip_html(raw),
        "rawHtml": raw,
        "candidates": [raw, _strip_html(raw)],
        "className": "",
        "style": "",
        "backgroundColor": "",
        "title": "",
        "ariaLabel": "",
        "dataset": {},
        "pseudoBefore": "",
        "pseudoAfter": "",
    }


def _network_fallback_rows(payload: dict[str, Any]) -> tuple[list[list[dict[str, Any]]], str]:
    """Recover DataTables rows directly from captured XHR/fetch responses."""
    headers = [_clean_text(value) for value in payload.get("headers", [])]
    if not headers:
        return [], ""
    expected = len(headers)
    observation_indexes = [
        index for index, header in enumerate(headers)
        if _OBSERVATION_RE.match(_clean_text(header))
    ]
    location_index = next((i for i, h in enumerate(headers) if _norm(h) == "location"), -1)

    def score(rows: list[list[dict[str, Any]]]) -> int:
        levels = 0
        locations = 0
        for row in rows:
            if location_index >= 0 and location_index < len(row):
                if _clean_text(row[location_index].get("text", "")):
                    locations += 1
            for index in observation_indexes:
                if index < len(row):
                    if any(_LEVEL_RE.search(candidate) for candidate in _cell_candidates(row[index])):
                        levels += 1
        return levels * 1000 + locations

    best_rows: list[list[dict[str, Any]]] = []
    best_source = ""
    best_score = 0

    def consider(raw_rows: Any, source: str) -> None:
        nonlocal best_rows, best_source, best_score
        if not isinstance(raw_rows, list):
            return
        converted: list[list[dict[str, Any]]] = []
        for raw_row in raw_rows:
            values: list[Any] | None = None
            if isinstance(raw_row, (list, tuple)):
                values = list(raw_row)
            elif isinstance(raw_row, dict):
                normalized = {_norm(key): value for key, value in raw_row.items()}
                mapped = []
                found_identity = False
                for header in headers:
                    key = _norm(header)
                    value = normalized.get(key, "")
                    if key in {"region", "province", "location"} and value not in {"", None}:
                        found_identity = True
                    mapped.append(value)
                if found_identity:
                    values = mapped
                elif len(raw_row) == expected:
                    values = list(raw_row.values())
            if values is None or len(values) < 4:
                continue
            if len(values) < expected:
                values += [""] * (expected - len(values))
            elif len(values) > expected:
                values = values[:expected]
            converted.append([_simple_cell(value) for value in values])
        current_score = score(converted)
        if current_score > best_score:
            best_rows = converted
            best_source = source
            best_score = current_score

    def walk_json(obj: Any, source: str, depth: int = 0) -> None:
        if depth > 6:
            return
        if isinstance(obj, dict):
            for key in ["data", "aaData", "rows", "results", "result", "items", "records"]:
                if key in obj:
                    consider(obj[key], f"{source} JSON key '{key}'")
                    walk_json(obj[key], source, depth + 1)
            for value in obj.values():
                if isinstance(value, (dict, list)):
                    walk_json(value, source, depth + 1)
        elif isinstance(obj, list):
            consider(obj, source)
            for value in obj[:100]:
                if isinstance(value, (dict, list)):
                    walk_json(value, source, depth + 1)

    for index, item in enumerate(payload.get("networkPayloads", []) or []):
        text = str(item.get("text", "") or "")
        if not text:
            continue
        source = f"XHR/fetch #{index + 1}: {item.get('url', '')}"
        try:
            walk_json(json.loads(text), source)
        except Exception:
            pass

        if "<table" in text.lower() or "<tr" in text.lower():
            try:
                tables = pd.read_html(io.StringIO(text))
                for table in tables:
                    raw_rows = table.astype(object).where(pd.notna(table), "").values.tolist()
                    consider(raw_rows, f"{source} HTML table")
            except Exception:
                pass

        # Many DataTables endpoints return only a sequence of <tr> elements.
        row_matches = re.findall(r"<tr\b[^>]*>(.*?)</tr>", text, flags=re.I | re.S)
        if row_matches:
            raw_rows = []
            for row_html in row_matches:
                cells = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", row_html, flags=re.I | re.S)
                if cells:
                    raw_rows.append(cells)
            consider(raw_rows, f"{source} HTML rows")

    return best_rows, best_source


def parse_rendered_table_payload(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert the rendered PhilSensors table payload to normalized hourly rows."""
    headers = [_clean_text(value) for value in payload.get("headers", [])]
    rows = payload.get("rows", [])
    normalized_headers = [_norm(value) for value in headers]

    if not _REQUIRED_HEADERS.issubset(set(normalized_headers)):
        raise ValueError("Rendered table does not contain Region, Province, and Location columns.")

    index_map = {name: normalized_headers.index(name) for name in _REQUIRED_HEADERS}
    observation_indexes = [
        index
        for index, header in enumerate(headers)
        if _OBSERVATION_RE.match(_clean_text(header))
    ]
    if not observation_indexes:
        raise ValueError("No Current Hour or hourly water-level columns were found.")

    scraped_at = pd.Timestamp(payload.get("scrapedAt", pd.Timestamp.now(tz=MANILA_TZ)))
    if scraped_at.tzinfo is None:
        scraped_at = scraped_at.tz_localize(MANILA_TZ)
    else:
        scraped_at = scraped_at.tz_convert(MANILA_TZ)

    output: list[dict[str, Any]] = []
    for row in rows:
        if len(row) < len(headers):
            row = list(row) + [{} for _ in range(len(headers) - len(row))]

        region = _clean_text(row[index_map["region"]].get("text", ""))
        province = _clean_text(row[index_map["province"]].get("text", ""))
        location = _clean_text(row[index_map["location"]].get("text", ""))
        if not location:
            continue

        station_id = stable_station_id(region, province, location)
        for index in observation_indexes:
            cell = row[index]
            level, trend = _parse_level_and_trend_from_cell(cell)
            if level is None:
                continue
            output.append(
                {
                    "station_id": station_id,
                    "station_name": location,
                    "location": location,
                    "region": region,
                    "province": province,
                    "timestamp": _timestamp_for_header(headers[index], scraped_at),
                    "level_m": level,
                    "threshold_status": _detect_threshold_status(
                        cell, payload.get("legendColors", {})
                    ),
                    "source_trend": trend,
                    "source_column": headers[index],
                    "scraped_at": scraped_at,
                    "source_url": payload.get("pageUrl", DEFAULT_URL),
                    "source_name": "DOST-ASTI PhilSensors public webpage",
                }
            )

    dataframe = pd.DataFrame(output)
    if dataframe.empty:
        network_rows, network_source = _network_fallback_rows(payload)
        if network_rows:
            retry_payload = dict(payload)
            retry_payload["rows"] = network_rows
            retry_payload["extractionSource"] = network_source
            retry_payload["networkPayloads"] = []
            return parse_rendered_table_payload(retry_payload)

        sample_cells: list[str] = []
        for row in rows[:3]:
            for index in observation_indexes[:4]:
                if index < len(row):
                    sample_cells.extend(_cell_candidates(row[index])[:3])
        source = payload.get("extractionSource", "unknown")
        body_has_value = payload.get("bodyHasMeterValue", False)
        failed = "; ".join(payload.get("failedRequests", [])[-5:]) or "none"
        xhr = "; ".join(payload.get("xhrResponses", [])[-5:]) or "none"
        raise ValueError(
            f"The PhilSensors table loaded, but no numeric water-level readings were parsed (scraper {SCRAPER_VERSION}). "
            f"Extraction source: {source}. Browser body contained a meter value: {body_has_value}. "
            f"Sample observation candidates: {sample_cells[:12] or ['none']}. "
            f"Recent failed requests: {failed}. Recent XHR/fetch responses: {xhr}."
        )

    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], utc=True)
    dataframe["scraped_at"] = pd.to_datetime(dataframe["scraped_at"], utc=True)
    dataframe = dataframe.drop_duplicates(
        subset=["station_id", "timestamp"], keep="last"
    ).sort_values(["station_id", "timestamp"])
    return dataframe.reset_index(drop=True)


def save_backup(dataframe: pd.DataFrame, path: str | Path) -> None:
    backup = Path(path)
    try:
        backup.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(backup, index=False)
    except OSError:
        # Read-only or ephemeral deployments should still display the live result.
        pass


def load_backup(path: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    for column in ["timestamp", "scraped_at"]:
        if column in dataframe.columns:
            dataframe[column] = pd.to_datetime(dataframe[column], errors="coerce", utc=True)
    return dataframe


def fetch_philsensors_readings(
    url: str = DEFAULT_URL,
    backup_path: str | Path = ".cache/philsensors_last_success.csv",
    timeout_ms: int = 90_000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch live readings, falling back to the last successfully saved CSV."""
    try:
        payload = _render_table_payload(url, timeout_ms=timeout_ms)
        dataframe = parse_rendered_table_payload(payload)
        save_backup(dataframe, backup_path)
        metadata = {
            "mode": "live",
            "message": "Live PhilSensors public webpage",
            "scraped_at": dataframe["scraped_at"].max(),
            "error": "",
        }
        return dataframe, metadata
    except Exception as exc:
        backup = Path(backup_path)
        if backup.exists():
            dataframe = load_backup(backup)
            metadata = {
                "mode": "backup",
                "message": "Last successful PhilSensors cache",
                "scraped_at": dataframe.get("scraped_at", pd.Series(dtype="datetime64[ns, UTC]")).max(),
                "error": f"{type(exc).__name__}: {exc}",
            }
            return dataframe, metadata
        raise RuntimeError(f"PhilSensors retrieval failed and no backup exists: {exc}") from exc



def _first_record_value(record: dict[str, Any], aliases: list[str]) -> Any:
    normalized = {_norm(key): value for key, value in record.items()}
    for alias in aliases:
        key = _norm(alias)
        if key in normalized and normalized[key] not in (None, ""):
            return normalized[key]
    return None


def _canonical_location(value: Any) -> str:
    """Create a conservative key for matching the same station across pages."""
    text = _clean_text(value).lower()
    text = re.sub(r"^\s*\d+\s*[.\-:)]+\s*", "", text)
    text = re.sub(
        r"\b(?:water\s*level\s*monitoring\s*system|waterlevel\s*monitoring\s*system|"
        r"wlms\s*with\s*arg|wlms|slms|automated\s*rain\s*gauge|arg)\b",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(r"\b(?:sensor|station)\b", " ", text, flags=re.I)
    return re.sub(r"[^a-z0-9]+", "", text)


def _metadata_record_from_mapping(record: dict[str, Any], source: str) -> dict[str, Any] | None:
    location = _first_record_value(
        record,
        [
            "location", "station_name", "stationname", "site_name", "sitename",
            "station", "name", "display_name", "displayname", "title",
        ],
    )
    lat = _first_record_value(
        record,
        ["latitude", "lat", "station_lat", "stationlatitude", "y", "ycoord"],
    )
    lon = _first_record_value(
        record,
        ["longitude", "long", "lng", "lon", "station_lon", "stationlongitude", "x", "xcoord"],
    )
    lat_num = pd.to_numeric(pd.Series([lat]), errors="coerce").iloc[0]
    lon_num = pd.to_numeric(pd.Series([lon]), errors="coerce").iloc[0]
    if pd.isna(lat_num) or pd.isna(lon_num):
        return None
    if not (-90 <= float(lat_num) <= 90 and -180 <= float(lon_num) <= 180):
        return None
    location_text = _clean_text(location)
    if not location_text or len(location_text) > 240:
        return None

    region = _clean_text(_first_record_value(record, ["region", "region_name", "regionname"]))
    province = _clean_text(
        _first_record_value(record, ["province", "province_name", "provincename"])
    )
    sensor_type = _clean_text(
        _first_record_value(
            record,
            ["sensor_type", "sensortype", "station_type", "stationtype", "type", "device_type"],
        )
    )
    operational_status = _clean_text(
        _first_record_value(record, ["status", "operational_status", "station_status", "active"])
    )
    official_id = _clean_text(
        _first_record_value(
            record,
            ["station_id", "stationid", "sensor_id", "sensorid", "device_id", "deviceid", "id"],
        )
    )
    elevation = pd.to_numeric(
        pd.Series([
            _first_record_value(record, ["elevation", "elevation_m", "altitude", "altitude_m"])
        ]),
        errors="coerce",
    ).iloc[0]

    return {
        "station_id": stable_station_id(region, province, location_text),
        "official_station_id": official_id or np.nan,
        "station_name": location_text,
        "location": location_text,
        "region": region or np.nan,
        "province": province or np.nan,
        "lat": float(lat_num),
        "lon": float(lon_num),
        "elevation_m": float(elevation) if pd.notna(elevation) else np.nan,
        "sensor_type": sensor_type or np.nan,
        "operational_status": operational_status or np.nan,
        "metadata_source": source,
    }


def _walk_json_dicts(value: Any, depth: int = 0):
    if depth > 8:
        return
    if isinstance(value, dict):
        yield value
        for child in value.values():
            if isinstance(child, (dict, list)):
                yield from _walk_json_dicts(child, depth + 1)
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, (dict, list)):
                yield from _walk_json_dicts(child, depth + 1)


def _render_station_metadata_payload(
    url: str = STATION_METADATA_URL,
    timeout_ms: int = 90_000,
) -> dict[str, Any]:
    """Render the PhilSensors Data Visualization page and capture station metadata."""
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Playwright is not installed. Add 'playwright' to requirements.txt."
        ) from exc

    network_payloads: list[dict[str, Any]] = []
    response_diagnostics: list[str] = []
    failed_requests: list[str] = []

    script = r"""
    () => {
      const clean = (value) => String(value ?? '').replace(/\s+/g, ' ').trim();
      const norm = (value) => clean(value).toLowerCase().replace(/[^a-z0-9]+/g, '');
      const getNested = (object, path) => {
        if (object == null || path == null) return '';
        if (typeof path === 'number') return object[path];
        if (typeof path !== 'string') return '';
        return path.split('.').reduce((value, key) => value == null ? '' : value[key], object);
      };
      const htmlText = (value) => {
        const holder = document.createElement('div');
        holder.innerHTML = value == null ? '' : String(value);
        return clean(holder.innerText || holder.textContent || value);
      };
      const tableRows = (table) => {
        const headers = Array.from(table.querySelectorAll('thead tr:last-child th'))
          .map((cell) => clean(cell.innerText || cell.textContent));
        let rows = Array.from(table.querySelectorAll('tbody tr')).map((row) =>
          Array.from(row.cells || row.querySelectorAll('td')).map((cell) => clean(cell.innerText || cell.textContent))
        ).filter((row) => row.length);
        try {
          if (window.jQuery && jQuery.fn && jQuery.fn.dataTable && jQuery.fn.dataTable.isDataTable(table)) {
            const dt = jQuery(table).DataTable();
            const settings = dt.settings()[0];
            const dtHeaders = (settings.aoColumns || []).map((column) => clean(column.sTitle || column.ariaTitle || ''));
            const sources = (settings.aoColumns || []).map((column, index) => column.mData == null ? index : column.mData);
            const dtRows = dt.rows().data().toArray().map((record) => {
              const values = Array.isArray(record)
                ? record
                : sources.map((source) => {
                    if (typeof source === 'function') {
                      try { return source(record, 'display'); } catch (_) { return ''; }
                    }
                    return getNested(record, source);
                  });
              return values.map(htmlText);
            });
            if (dtRows.length > rows.length) rows = dtRows;
            return {headers: dtHeaders.some(Boolean) ? dtHeaders : headers, rows, source: 'DataTables'};
          }
        } catch (_) {}
        return {headers, rows, source: 'DOM'};
      };

      const tables = Array.from(document.querySelectorAll('table')).map(tableRows);
      const dataElements = Array.from(document.querySelectorAll('*')).map((element) => {
        const dataset = element.dataset ? {...element.dataset} : {};
        const attrs = {};
        Array.from(element.attributes || []).forEach((attribute) => { attrs[attribute.name] = attribute.value; });
        return {text: clean(element.innerText || element.textContent), dataset, attrs};
      }).filter((entry) => {
        const keys = Object.keys({...entry.dataset, ...entry.attrs}).map(norm);
        return keys.some((key) => ['latitude','lat','longitude','lng','lon'].includes(key));
      }).slice(0, 5000);

      return {
        pageUrl: location.href,
        pageTitle: document.title,
        tables,
        dataElements,
        bodyTextSample: clean(document.body.innerText || '').slice(0, 1500),
      };
    }
    """

    with sync_playwright() as playwright:
        launch_kwargs: dict[str, Any] = {
            "headless": True,
            "args": [
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-blink-features=AutomationControlled",
            ],
        }
        executable = find_chromium_executable()
        if executable:
            launch_kwargs["executable_path"] = executable

        browser = playwright.chromium.launch(**launch_kwargs)
        try:
            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                ),
                locale="en-PH",
                timezone_id=MANILA_TZ,
                java_script_enabled=True,
                ignore_https_errors=True,
            )
            context.add_init_script(
                """
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                Object.defineProperty(navigator, 'languages', {get: () => ['en-PH', 'en-US', 'en']});
                Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                window.chrome = window.chrome || {runtime: {}};
                """
            )
            page = context.new_page()
            page.on(
                "requestfailed",
                lambda request: failed_requests.append(
                    f"{request.resource_type}: {request.url} :: {request.failure or 'failed'}"
                ),
            )

            def handle_response(response):
                if response.request.resource_type not in {"xhr", "fetch"}:
                    return
                response_diagnostics.append(
                    f"{response.status} {response.request.resource_type} {response.url}"
                )
                try:
                    content_type = response.headers.get("content-type", "").lower()
                    if response.status != 200 or not any(
                        token in content_type for token in ["json", "text", "html", "javascript"]
                    ):
                        return
                    raw = response.text()
                    if raw and len(raw) <= 8_000_000:
                        network_payloads.append(
                            {
                                "url": response.url,
                                "content_type": content_type,
                                "text": raw,
                            }
                        )
                except Exception:
                    pass

            page.on("response", handle_response)
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 30_000))
            except PlaywrightTimeoutError:
                pass
            try:
                page.wait_for_selector("table, [data-lat], [data-latitude]", timeout=min(timeout_ms, 45_000))
            except PlaywrightTimeoutError:
                pass

            page.evaluate(
                """
                () => {
                  try {
                    if (window.jQuery && jQuery.fn && jQuery.fn.dataTable) {
                      jQuery.fn.dataTable.tables().forEach((table) => {
                        try { jQuery(table).DataTable().page.len(-1).draw(false); } catch (_) {}
                      });
                    }
                  } catch (_) {}
                }
                """
            )
            page.wait_for_timeout(2500)
            payload = page.evaluate(script)
            payload["networkPayloads"] = network_payloads
            payload["xhrResponses"] = response_diagnostics[-30:]
            payload["failedRequests"] = failed_requests[-30:]
            payload["scrapedAt"] = pd.Timestamp.now(tz=MANILA_TZ).isoformat()
            return payload
        finally:
            browser.close()


def parse_station_metadata_payload(payload: dict[str, Any]) -> pd.DataFrame:
    """Parse station coordinates from DOM tables, data attributes, and XHR JSON."""
    records: list[dict[str, Any]] = []

    for table_index, table in enumerate(payload.get("tables", []) or []):
        headers = [_clean_text(value) for value in table.get("headers", [])]
        normalized = [_norm(value) for value in headers]
        if not headers:
            continue
        has_location = any(value in normalized for value in ["location", "stationname", "name", "station"])
        has_lat = any(value in normalized for value in ["latitude", "lat", "stationlat"])
        has_lon = any(value in normalized for value in ["longitude", "long", "lng", "lon", "stationlon"])
        if not (has_location and has_lat and has_lon):
            continue
        for row in table.get("rows", []) or []:
            mapping = {headers[index]: row[index] if index < len(row) else "" for index in range(len(headers))}
            parsed = _metadata_record_from_mapping(
                mapping,
                f"PhilSensors station table #{table_index + 1} ({table.get('source', 'DOM')})",
            )
            if parsed:
                records.append(parsed)

    for element_index, element in enumerate(payload.get("dataElements", []) or []):
        mapping: dict[str, Any] = {}
        mapping.update(element.get("attrs", {}) or {})
        mapping.update(element.get("dataset", {}) or {})
        if element.get("text"):
            mapping.setdefault("location", element.get("text"))
        parsed = _metadata_record_from_mapping(
            mapping,
            f"PhilSensors page data attributes #{element_index + 1}",
        )
        if parsed:
            records.append(parsed)

    for network_index, item in enumerate(payload.get("networkPayloads", []) or []):
        raw = str(item.get("text", "") or "")
        if not raw:
            continue
        source = f"PhilSensors XHR/fetch #{network_index + 1}: {item.get('url', '')}"
        try:
            decoded = json.loads(raw)
            for mapping in _walk_json_dicts(decoded):
                parsed = _metadata_record_from_mapping(mapping, source)
                if parsed:
                    records.append(parsed)
        except Exception:
            pass

        if "<table" in raw.lower():
            try:
                for table in pd.read_html(io.StringIO(raw)):
                    for mapping in table.to_dict(orient="records"):
                        parsed = _metadata_record_from_mapping(mapping, source + " HTML")
                        if parsed:
                            records.append(parsed)
            except Exception:
                pass

    dataframe = pd.DataFrame(records)
    if dataframe.empty:
        xhr = "; ".join(payload.get("xhrResponses", [])[-8:]) or "none"
        failed = "; ".join(payload.get("failedRequests", [])[-8:]) or "none"
        raise ValueError(
            "The PhilSensors station metadata page loaded, but no location/latitude/longitude "
            f"records were parsed. Recent XHR/fetch: {xhr}. Failed requests: {failed}."
        )

    dataframe["lat"] = pd.to_numeric(dataframe["lat"], errors="coerce")
    dataframe["lon"] = pd.to_numeric(dataframe["lon"], errors="coerce")
    dataframe = dataframe.dropna(subset=["location", "lat", "lon"])
    dataframe = dataframe[
        dataframe["lat"].between(4.0, 22.5) & dataframe["lon"].between(115.0, 128.5)
    ]
    dataframe["location_key"] = dataframe["location"].map(_canonical_location)
    dataframe["region_key"] = dataframe["region"].map(_norm)
    dataframe["province_key"] = dataframe["province"].map(_norm)
    dataframe = dataframe.sort_values(
        ["location_key", "region_key", "province_key", "metadata_source"]
    ).drop_duplicates(
        subset=["location_key", "region_key", "province_key", "lat", "lon"], keep="first"
    )
    return dataframe.drop(columns=["location_key", "region_key", "province_key"]).reset_index(drop=True)


def fetch_philsensors_station_metadata(
    url: str = STATION_METADATA_URL,
    backup_path: str | Path = ".cache/philsensors_station_metadata.csv",
    timeout_ms: int = 90_000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch station coordinates, falling back to the last successful metadata CSV."""
    try:
        payload = _render_station_metadata_payload(url=url, timeout_ms=timeout_ms)
        dataframe = parse_station_metadata_payload(payload)
        save_backup(dataframe, backup_path)
        return dataframe, {
            "mode": "live",
            "message": "Live PhilSensors station metadata",
            "scraped_at": pd.Timestamp(payload.get("scrapedAt", pd.Timestamp.now(tz=MANILA_TZ))),
            "error": "",
            "records": int(len(dataframe)),
        }
    except Exception as exc:
        backup = Path(backup_path)
        if backup.exists():
            dataframe = load_backup(backup)
            return dataframe, {
                "mode": "backup",
                "message": "Last successful PhilSensors station metadata cache",
                "scraped_at": pd.Timestamp.fromtimestamp(backup.stat().st_mtime, tz="UTC"),
                "error": f"{type(exc).__name__}: {exc}",
                "records": int(len(dataframe)),
            }
        raise RuntimeError(
            f"PhilSensors station-metadata retrieval failed and no backup exists: {exc}"
        ) from exc


def merge_station_metadata(
    readings: pd.DataFrame,
    metadata: pd.DataFrame | None,
) -> pd.DataFrame:
    """Conservatively attach coordinates from the PhilSensors station catalogue."""
    output = readings.copy()
    metadata_columns = [
        "lat", "lon", "elevation_m", "sensor_type", "operational_status",
        "official_station_id", "metadata_source", "metadata_match_method", "metadata_match_score",
    ]
    for column in metadata_columns:
        if column not in output.columns:
            output[column] = np.nan
    if metadata is None or metadata.empty or output.empty:
        return output

    meta = metadata.copy()
    for column in ["station_name", "location", "region", "province"]:
        if column not in meta.columns:
            meta[column] = np.nan
    if "location" not in output.columns:
        output["location"] = output.get("station_name", pd.Series(index=output.index, dtype=object))
    for column in ["region", "province"]:
        if column not in output.columns:
            output[column] = np.nan

    def keys(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result["_location_key"] = result["location"].fillna(result.get("station_name")).map(_canonical_location)
        result["_region_key"] = result["region"].map(_norm)
        result["_province_key"] = result["province"].map(_norm)
        result["_strict_key"] = result["_region_key"] + "|" + result["_province_key"] + "|" + result["_location_key"]
        result["_region_location_key"] = result["_region_key"] + "|" + result["_location_key"]
        result["_province_location_key"] = result["_province_key"] + "|" + result["_location_key"]
        return result

    unique_readings = output.sort_values("timestamp").drop_duplicates("station_id", keep="last")
    read_keys = keys(unique_readings)
    meta_keys = keys(meta)

    def unique_lookup(key: str) -> dict[str, pd.Series]:
        lookup: dict[str, pd.Series] = {}
        for value, group in meta_keys.groupby(key, dropna=False):
            value_text = str(value)
            if value_text and not value_text.endswith("|") and len(group) == 1:
                lookup[value_text] = group.iloc[0]
        return lookup

    strict_lookup = unique_lookup("_strict_key")
    region_lookup = unique_lookup("_region_location_key")
    province_lookup = unique_lookup("_province_location_key")
    location_lookup = unique_lookup("_location_key")

    match_rows: list[dict[str, Any]] = []
    for _, reading in read_keys.iterrows():
        candidate = None
        method = ""
        score = np.nan
        for key_name, lookup, label in [
            ("_strict_key", strict_lookup, "region+province+location"),
            ("_region_location_key", region_lookup, "region+location"),
            ("_province_location_key", province_lookup, "province+location"),
            ("_location_key", location_lookup, "unique location"),
        ]:
            value = str(reading.get(key_name, ""))
            if value in lookup:
                candidate = lookup[value]
                method = label
                score = 1.0
                break

        if candidate is None:
            pool = meta_keys
            region_key = str(reading.get("_region_key", ""))
            province_key = str(reading.get("_province_key", ""))
            if region_key:
                region_pool = pool[pool["_region_key"] == region_key]
                if not region_pool.empty:
                    pool = region_pool
            if province_key:
                province_pool = pool[pool["_province_key"] == province_key]
                if not province_pool.empty:
                    pool = province_pool
            target = str(reading.get("_location_key", ""))
            scored = []
            if target:
                for index, metadata_row in pool.iterrows():
                    candidate_key = str(metadata_row.get("_location_key", ""))
                    if candidate_key:
                        scored.append((SequenceMatcher(None, target, candidate_key).ratio(), index))
            scored.sort(reverse=True)
            if scored and scored[0][0] >= 0.94 and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.03):
                score, index = scored[0]
                candidate = meta_keys.loc[index]
                method = "conservative fuzzy location"

        result = {"station_id": reading["station_id"]}
        if candidate is not None:
            for column in [
                "lat", "lon", "elevation_m", "sensor_type", "operational_status",
                "official_station_id", "metadata_source",
            ]:
                result[column] = candidate.get(column, np.nan)
            result["metadata_match_method"] = method
            result["metadata_match_score"] = score
        match_rows.append(result)

    matches = pd.DataFrame(match_rows)
    merged = output.merge(matches, on="station_id", how="left", suffixes=("", "_auto"))
    for column in metadata_columns:
        auto_column = f"{column}_auto"
        if auto_column in merged.columns:
            merged[column] = merged[column].where(merged[column].notna(), merged[auto_column])
            merged = merged.drop(columns=[auto_column])
    return merged

def _normalize_registry_columns(registry: pd.DataFrame) -> pd.DataFrame:
    dataframe = registry.copy()
    lookup = {
        re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_"): column
        for column in dataframe.columns
    }
    aliases = {
        "station_id": ["station_id", "stationid", "id"],
        "station_name": ["station_name", "name"],
        "location": ["location", "site", "location_name"],
        "region": ["region"],
        "province": ["province"],
        "lat": ["lat", "latitude"],
        "lon": ["lon", "lng", "longitude"],
        "basin_name": ["basin_name", "basin", "river_basin"],
        "alert_m": ["alert_m", "alert_level_m"],
        "alarm_m": ["alarm_m", "alarm_level_m"],
        "critical_m": ["critical_m", "critical_level_m"],
    }
    rename_map: dict[Any, str] = {}
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            if candidate in lookup:
                rename_map[lookup[candidate]] = canonical
                break
    dataframe = dataframe.rename(columns=rename_map)

    for column in aliases:
        if column not in dataframe.columns:
            dataframe[column] = np.nan

    missing_id = dataframe["station_id"].isna() | (dataframe["station_id"].astype(str).str.strip() == "")
    dataframe.loc[missing_id, "station_id"] = dataframe.loc[missing_id].apply(
        lambda row: stable_station_id(row.get("region"), row.get("province"), row.get("location")),
        axis=1,
    )

    dataframe["station_id"] = dataframe["station_id"].astype(str).str.strip()
    for column in ["lat", "lon", "alert_m", "alarm_m", "critical_m"]:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    return dataframe[
        [
            "station_id",
            "station_name",
            "location",
            "region",
            "province",
            "lat",
            "lon",
            "basin_name",
            "alert_m",
            "alarm_m",
            "critical_m",
        ]
    ].drop_duplicates(subset=["station_id"], keep="last")


def merge_station_registry(
    readings: pd.DataFrame, registry: pd.DataFrame | None
) -> pd.DataFrame:
    """Apply a user-verified registry, taking priority over auto metadata."""
    output = readings.copy()
    fill_columns = [
        "station_name", "location", "region", "province", "lat", "lon",
        "basin_name", "alert_m", "alarm_m", "critical_m",
    ]
    for column in fill_columns:
        if column not in output.columns:
            output[column] = np.nan
    if registry is None or registry.empty:
        return output

    clean_registry = _normalize_registry_columns(registry)
    merged = output.merge(clean_registry, on="station_id", how="left", suffixes=("", "_registry"))
    for column in fill_columns:
        registry_column = f"{column}_registry"
        if registry_column in merged.columns:
            # A non-empty registry value is treated as user-verified and overrides auto metadata.
            registry_values = merged[registry_column]
            usable = registry_values.notna() & (registry_values.astype(str).str.strip() != "")
            merged.loc[usable, column] = registry_values.loc[usable]
            merged = merged.drop(columns=[registry_column])
    merged["registry_matched"] = merged.get("lat", pd.Series(index=merged.index)).notna() & merged.get(
        "lon", pd.Series(index=merged.index)
    ).notna()
    return merged


def registry_template_from_readings(readings: pd.DataFrame) -> pd.DataFrame:
    """Create a fillable registry containing every currently observed station."""
    columns = [
        "station_id",
        "station_name",
        "location",
        "region",
        "province",
        "lat",
        "lon",
        "basin_name",
        "alert_m",
        "alarm_m",
        "critical_m",
    ]
    if readings.empty:
        return pd.DataFrame(columns=columns)

    template = readings.sort_values("timestamp").drop_duplicates("station_id", keep="last")
    template = template[["station_id", "station_name", "location", "region", "province"]].copy()
    for column in ["lat", "lon", "basin_name", "alert_m", "alarm_m", "critical_m"]:
        template[column] = ""
    return template[columns].sort_values(["region", "province", "location"]).reset_index(drop=True)


def metadata_as_json(metadata: dict[str, Any]) -> str:
    serializable = {
        key: value.isoformat() if isinstance(value, pd.Timestamp) else value
        for key, value in metadata.items()
    }
    return json.dumps(serializable, indent=2)
