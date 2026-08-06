# Build 5.6 — ChatGPT Work / Manual Official Report Import

This build adds a supervised import workflow for river updates collected through ChatGPT Work or another manual public-source search.

## Accepted format

Paste or upload a CSV/TSV with these main columns:

- `requested_source`
- `reporting_source`
- `river_or_site`
- `level`
- `unit`
- `status`
- `observed_at`
- `source_url`
- `notes`

Optional fields include province, municipality, latitude, and longitude.

## Behavior

- Numerical rows are converted to metres for consistent display.
- The original value and unit remain visible in the manual-report table and map popup.
- Rows without a numerical value remain qualitative official reports.
- Ambiguous local classifications such as `Above Normal Level`, `middle level`, and `spilling` are preserved and are not silently converted to PAGASA Alert/Alarm/Critical classes.
- A separate national-map layer displays all mapped manual reports.
- Registry coordinates are explicitly labelled as representative anchors and are not presented as exact gauge locations.
- Manual one-off reports do not alter the instrument-based river-basin hazard score or the province rise/fall map.

## New files

- `data/chatgpt_work_report_template.tsv`
- `data/chatgpt_work_sample.tsv`
- `data/manual_report_location_registry.csv`
- `tests/test_manual_work_reports.py`

## Updated runtime files

- `app.py`
- `providers/official_reports_csv.py`
- `core/target_river_map.py`
- `core/maps.py`
- `core/schema.py`
