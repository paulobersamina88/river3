# Build 5.5.0 — PRFFWC Pampanga + LLDA image extraction

This build keeps the V5.4 multi-source dashboard and adds two official-image extraction fallbacks.

## Pampanga River

A new provider reads the latest hydrological infographic from the PRFFWC support site:

- discovers the current infographic on the homepage;
- downloads the image or captures the rendered image element;
- extracts the issue/observation time;
- extracts numerical water levels for every row that publishes a value;
- reads the official Below/Alert/Alarm/Critical color and rising/receding symbol;
- keeps station values separate because their gauge datums differ;
- caches the last successful result.

The daily station list can include Zaragoza, Arayat, Candaba, Sulipan, and other PRFFWC stations when their water-level cell is populated.

## Laguna de Bay

The LLDA provider still attempts HTML tables and page text first. It now also:

- downloads official images embedded on the LLDA water-level page;
- captures rendered image elements and the rendered page when necessary;
- extracts a clearly labelled Laguna de Bay current lake level and observation time;
- rejects images that do not explicitly mention Laguna de Bay/Laguna Lake;
- labels the value as a lake-wide measurement, not a tributary river gauge.

## Abra

The app continues to use the direct official PAGASA Abra page. Abra is shown as an official basin forecast/advisory unless PAGASA publishes an explicit numerical station measurement.

## Required dependencies

`requirements.txt` adds Pillow and pytesseract. `packages.txt` adds tesseract-ocr.
