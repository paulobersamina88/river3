from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

MANILA_TZ = "Asia/Manila"


def load_geojson(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def basin_dataframe(geojson: dict) -> pd.DataFrame:
    rows = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        rows.append(
            {
                "basin_name": props.get("basin_name", "Unknown"),
                "region": props.get("region", ""),
                "lat": pd.to_numeric(props.get("lat"), errors="coerce"),
                "lon": pd.to_numeric(props.get("lon"), errors="coerce"),
                "threshold24_mm": pd.to_numeric(props.get("threshold24_mm", 100), errors="coerce"),
                "threshold72_mm": pd.to_numeric(props.get("threshold72_mm", 200), errors="coerce"),
            }
        )
    return pd.DataFrame(rows)


def _fetch_json(url: str, params: dict[str, Any], timeout: int = 25) -> dict:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_openmeteo_basin_forecast(master: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    errors = []
    today = pd.Timestamp.now(tz=MANILA_TZ).date()
    start_date = str(today - pd.Timedelta(days=3))
    end_date = str(today)
    for _, basin in master.iterrows():
        name = basin["basin_name"]
        lat, lon = basin.get("lat"), basin.get("lon")
        record = {
            "basin_name": name,
            "forecast_rain_24h_mm": np.nan,
            "forecast_rain_72h_mm": np.nan,
            "antecedent_rain_72h_mm": np.nan,
        }
        if pd.isna(lat) or pd.isna(lon):
            errors.append(f"{name}: missing centroid")
            rows.append(record)
            continue
        try:
            forecast = _fetch_json(
                "https://api.open-meteo.com/v1/forecast",
                {
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "hourly": "precipitation",
                    "forecast_days": 7,
                    "timezone": MANILA_TZ,
                },
            )
            historical = _fetch_json(
                "https://historical-forecast-api.open-meteo.com/v1/forecast",
                {
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "start_date": start_date,
                    "end_date": end_date,
                    "hourly": "precipitation",
                    "timezone": MANILA_TZ,
                },
            )
            values = pd.to_numeric(pd.Series(forecast.get("hourly", {}).get("precipitation", [])), errors="coerce").fillna(0)
            history = pd.to_numeric(pd.Series(historical.get("hourly", {}).get("precipitation", [])), errors="coerce").fillna(0)
            record["forecast_rain_24h_mm"] = float(values.iloc[:24].sum()) if len(values) else np.nan
            record["forecast_rain_72h_mm"] = float(values.iloc[:72].sum()) if len(values) else np.nan
            record["antecedent_rain_72h_mm"] = float(history.iloc[-72:].sum()) if len(history) else np.nan
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
        rows.append(record)
    return pd.DataFrame(rows), errors


def sample_basin_forecast(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.rename(columns={"rain_mm_24h": "forecast_rain_24h_mm", "rain_mm_72h": "forecast_rain_72h_mm"})
    if "antecedent_rain_72h_mm" not in frame:
        frame["antecedent_rain_72h_mm"] = 0.0
    return frame


def classify_ratio(ratio: float) -> str:
    if pd.isna(ratio):
        return "No Data"
    if ratio >= 1.5:
        return "Extreme"
    if ratio >= 1.0:
        return "Severe"
    if ratio >= 0.7:
        return "High"
    if ratio >= 0.4:
        return "Moderate"
    return "Low"


def compute_hazard(master: pd.DataFrame, values: pd.DataFrame, window: str = "24h") -> pd.DataFrame:
    output = master.merge(values, on="basin_name", how="left")
    rain_col = "forecast_rain_24h_mm" if window == "24h" else "forecast_rain_72h_mm"
    threshold_col = "threshold24_mm" if window == "24h" else "threshold72_mm"
    antecedent = pd.to_numeric(output.get("antecedent_rain_72h_mm", 0), errors="coerce").fillna(0)
    factor = 1 + (antecedent / 300).clip(lower=0, upper=0.35)
    output["effective_rain_mm"] = pd.to_numeric(output[rain_col], errors="coerce") * factor
    output["threshold_selected_mm"] = pd.to_numeric(output[threshold_col], errors="coerce")
    output["hazard_ratio"] = output["effective_rain_mm"] / output["threshold_selected_mm"]
    output["hazard_level"] = output["hazard_ratio"].apply(classify_ratio)
    output["selected_window"] = window
    return output
