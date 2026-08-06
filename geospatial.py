from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _point_in_ring(lon: float, lat: float, ring: list) -> bool:
    inside = False
    if not ring or len(ring) < 3:
        return False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        intersects = ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-15) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def point_in_geometry(lon: float, lat: float, geometry: dict[str, Any]) -> bool:
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates") or []
    polygons = [coordinates] if kind == "Polygon" else coordinates if kind == "MultiPolygon" else []
    for polygon in polygons:
        if not polygon:
            continue
        if _point_in_ring(lon, lat, polygon[0]):
            if any(_point_in_ring(lon, lat, hole) for hole in polygon[1:]):
                continue
            return True
    return False


def assign_basins(readings: pd.DataFrame, geojson: dict) -> pd.DataFrame:
    if readings is None or readings.empty:
        return readings
    output = readings.copy()
    features = geojson.get("features", [])
    for index, row in output.iterrows():
        basin = str(row.get("basin_name", "") or "").strip()
        if basin:
            continue
        lat = pd.to_numeric(pd.Series([row.get("lat")]), errors="coerce").iloc[0]
        lon = pd.to_numeric(pd.Series([row.get("lon")]), errors="coerce").iloc[0]
        if pd.isna(lat) or pd.isna(lon):
            continue
        for feature in features:
            if point_in_geometry(float(lon), float(lat), feature.get("geometry", {})):
                output.at[index, "basin_name"] = feature.get("properties", {}).get("basin_name", "")
                break
    return output
