# Philippine Multi-Source River Monitoring Dashboard — Build 5.6.0

This is a separate Streamlit application package. The original legacy app remains saved as `legacy_app_v3_8.py`.

## Build 5.4 additions

- Added an **LLDA Laguna de Bay provider** using the official LLDA water-level page. It tries a normal HTTP read first and a rendered Chromium page second, then falls back to the last successful cache.
- Expanded PAGASA hydrological bulletins to include:
  - Abra River Basin
  - Panay River Basin
  - Cagayan de Oro River Basin
  - Davao River Basin
  - NCR/Pasig–Marikina–Laguna de Bay
  - the newest visible PAGASA Visayas regional advisory that mentions Samar
- Added PAGASA Flood Information index parsing so each basin can display `Flood Watch` or `Non-Flood Watch` when the main page publishes that status.
- Added PDF text parsing for current PAGASA bulletin links using `pypdf`.
- Expanded the requested-river map to a national map covering:
  - Marikina River
  - Tullahan River
  - Meycauayan/MMORS
  - Pampanga River
  - Laguna de Bay / Laguna area
  - Abra River Basin
  - Samar rivers
  - Panay River Basin
  - Cagayan de Oro River Basin
  - Davao River Basin
- Numerical measurements and qualitative forecasts/advisories are displayed separately and clearly labelled.

## Numerical providers

1. DOST-ASTI PhilSensors
2. PAGASA NCR–Rizal / Pasig–Marikina–Tullahan gauges
3. Bulacan PDRRMO river stations
4. LLDA Laguna de Bay lake level
5. ChatGPT Work / manual official report CSV or TSV import

## Qualitative providers

1. PAGASA Flood Information and basin hydrological forecasts
2. PAGASA Visayas regional advisory for Samar when a current visible advisory mentions Samar
3. PRFFWC qualitative Pampanga sub-basin status when the HTML status table is available

A PAGASA basin forecast is not treated as a measured level in metres. LLDA is treated as a lake-wide Laguna de Bay level and is not relabelled as a Victoria, Pagsanjan, San Juan, or another tributary-river reading.

## GitHub structure

```text
app.py
requirements.txt
packages.txt
philsensors_scraper.py
core/
providers/
data/
tests/
.streamlit/
```

Upload the **contents** of the extracted package to the repository root. Keep the Streamlit main file path as `app.py`.

## Local run

```bash
python -m pip install -r requirements.txt
playwright install chromium
streamlit run app.py
```

## Validation completed

- Python compilation for all application, core, and provider modules
- Twelve parser, manual-report, and target-classification tests
- Offline Folium map rendering test

Live government-site access was unavailable from the build environment. The Data-source health table will show whether each provider is live, cached, partial, or unavailable after deployment.

## Build 5.5 additions

- PRFFWC Pampanga numerical water levels extracted from the official daily infographic.
- LLDA official-image/page OCR fallback for Laguna de Bay lake level.
- Direct PAGASA Abra basin forecast retained.


## Build 5.6 additions

- Paste or upload ChatGPT Work results as TSV, CSV, or text.
- Flexible support for `requested_source`, `reporting_source`, `river_or_site`, `level`, `unit`, `status`, `observed_at`, `source_url`, and `notes`.
- Numerical reports are normalized to metres while retaining the original value and unit.
- Qualitative rows such as `CODE GREEN / Safe Level` remain non-numerical official reports.
- Ambiguous local terms such as `Above Normal Level`, `middle level`, and `spilling` are preserved rather than forced into PAGASA Alert/Alarm/Critical classes.
- Added a dedicated `ChatGPT Work / manual official reports` map layer.
- Added a downloadable location registry; representative coordinates are clearly labelled and are not presented as exact gauge positions.
- Manual one-off reports do not silently alter instrument-based basin hazard scores or province rise/fall trends.
