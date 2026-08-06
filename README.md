# Philippine Multi-Source River Monitoring Dashboard — Build 5.0.0

This is a **new, independent Streamlit package** created from the uploaded `river2-main` repository. The original working application is preserved as `legacy_app_v3_8.py`; the clean deployment entry point is the new `app.py`.

## Build 5.2 corrections

- Corrected the requested-river summary so it no longer treats the generic `Pasig-Laguna` basin name as a Laguna river reading.
- Marikina, Tullahan, Meycauayan/MMORS, Pampanga, and Laguna now use strict river/province matching.
- Added a dedicated requested-river watch map with clickable station tables.
- River-system summary markers use clearly labelled representative display anchors; exact station markers appear only when verified coordinates are available.
- The summary reports a named reference station instead of choosing an arbitrary last row from stations that share the same timestamp.

## Included monitoring sources

### Numerical water-level providers

1. **DOST-ASTI PhilSensors**
   - Uses the existing `philsensors_scraper.py` from the uploaded repository.
   - Reads public water-level readings with Playwright.
   - Optionally loads the public station catalogue for coordinates.
   - Uses a last-successful local cache.

2. **PAGASA Pasig–Marikina–Tullahan FFWS**
   - Source: `https://pasig-marikina-tullahanffws.pagasa.dost.gov.ph/water/table.do`
   - Attempts a direct HTML read first, then a Playwright-rendered page.
   - Extracts current, 30-minute, one-hour, and two-hour levels when available.
   - Preserves station-specific Alert, Alarm, and Critical elevation thresholds.
   - Classifies station names conservatively as Marikina, Tullahan, or the wider PMT system.

3. **Bulacan PDRRMO River Status Stations**
   - Source: `https://pdrrmo.bulacan.gov.ph/`
   - Reads the public River Status Stations table.
   - Identifies clearly named Pampanga and Meycauayan–Marilao–Obando system stations.
   - Stores successive retrievals to estimate retrieval-to-retrieval change.
   - When the official page says `No Record`, no numerical value is invented.

4. **Official LGU report CSV import**
   - Intended for targeted DRRMO reports collected during a Severe/Extreme event.
   - Supports metres, feet, centimetres, and millimetres.
   - Keeps social/public-report records separate from instrument feeds.

### Qualitative hydrological bulletins

- PAGASA NCR/Pasig–Marikina–Laguna de Bay basin bulletin
- PAGASA Abra River Basin bulletin
- Pampanga River Flood Forecasting and Warning Center status table when reachable

These bulletins are displayed separately because they are basin forecasts, **not numerical station measurements**.

## Requested river coverage

The dashboard has a dedicated coverage table for:

- Marikina River
- Tullahan River
- Meycauayan–Marilao–Obando River System
- Pampanga River
- Available Laguna rivers or Laguna de Bay system observations

For Laguna, the app uses numerical PhilSensors gauges whose station metadata identify Laguna. It also displays the PAGASA NCR/Pasig–Marikina–Laguna de Bay qualitative bulletin. It does not claim a Victoria or Pagsanjan numerical level without an identified official gauge.

## Package structure

```text
ph_river_monitor_v5/
├── app.py                         # clean Streamlit entry point
├── legacy_app_v3_8.py             # untouched reference copy of the old app
├── philsensors_scraper.py         # uploaded working PhilSensors scraper
├── requirements.txt
├── packages.txt
├── .streamlit/
│   └── config.toml
├── core/
│   ├── geospatial.py
│   ├── history.py
│   ├── maps.py
│   ├── rainfall.py
│   ├── registry.py
│   ├── schema.py
│   └── water.py
├── providers/
│   ├── bulacan_pdrrmo.py
│   ├── official_reports_csv.py
│   ├── pagasa_bulletins.py
│   ├── pagasa_pmt.py
│   └── philsensors.py
├── data/
│   ├── major_river_basins_simplified.geojson
│   ├── sample_basin_rainfall.csv
│   ├── sample_typhoon_track.csv
│   ├── philsensors_station_registry.csv
│   ├── supplementary_station_registry.csv
│   └── official_report_template.csv
└── tests/
    └── test_parsers.py
```

## Deploy as a new GitHub repository

1. Extract this ZIP.
2. Create a new GitHub repository, for example `ph-river-monitor-v5`.
3. Upload **the contents inside** the extracted folder, not the outer folder itself.
4. Confirm that `app.py`, `requirements.txt`, and `packages.txt` are in the repository root.
5. Create a new Streamlit Community Cloud app.
6. Select the new repository and set the entry file to `app.py`.
7. Deploy.

Do not overwrite your existing `river2` deployment until this new app has been tested.

## Local run

```bash
python -m pip install -r requirements.txt
playwright install chromium
streamlit run app.py
```

On Streamlit Community Cloud, `packages.txt` installs system Chromium. The code automatically checks common Chromium paths.

## Updating station coordinates

PAGASA PMT and Bulacan readings can appear in the station table even when exact coordinates are not available. They are placed on the map only after verified coordinates are supplied.

Edit:

```text
data/supplementary_station_registry.csv
```

Fill in `lat` and `lon` only after checking the official station map or station page. Avoid using a municipality centre as though it were the actual gauge location.

## Source isolation

Each provider runs independently. A failure in one provider does not stop the others. The Data-source health table reports:

- `live`
- `cache`
- `empty`
- `error`
- `uploaded`

Cached values are explicitly marked and should not be treated as current observations.

## Tests completed

- Python syntax compilation for the application, provider modules, core modules, and PhilSensors scraper
- Simulated PAGASA PMT multi-row table parsing
- Simulated Bulacan PDRRMO river-table parsing
- Official LGU CSV unit conversion from feet to metres

The external government webpages cannot be reached from the build container, so final live-source validation must occur after deployment.

## Important technical limitations

- Public webpage structure may change and require parser maintenance.
- PAGASA PMT readings are elevation values (`EL.m`); compare only with thresholds for the same station.
- Bulacan PDRRMO may temporarily publish `No Record`.
- The included basin GeoJSON contains simplified placeholder polygons inherited from the uploaded repository. Replace it with validated basin boundaries before operational use.
- This dashboard is an academic screening tool and does not replace PAGASA, DOST-ASTI, LLDA, LGU, DRRMO, dam-operator, or evacuation advisories.

## Build 5.2.0 — restored province rise/fall map

The province-level observed water-level map from dashboard build 3.8 is restored. It includes:

- province trend shading;
- large visible rise/fall labels in m/hour;
- per-province gauge counts;
- a popup table with station, source, current level, trend, status, and timestamp;
- controls for including stale/offline readings and showing only rapid changes.

The map combines compatible numerical readings from all enabled providers. It remains separate from the river-basin rainfall/combined-hazard map.
