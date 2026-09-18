from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import requests

from .constants import SOIL_PROPERTIES

SOILGRID_URLS = [
    "https://rest.isric.org/soilgrids/v2.0/properties/query",
    "https://dev-rest.isric.org/soilgrids/v2.0/properties/query",
]
WEATHERAPI_URL = "https://api.weatherapi.com/v1/current.json"
PLANETARY_COMPUTER_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def fetch_weather_features(latitude: float, longitude: float, api_key: str) -> dict[str, Any]:
    response = requests.get(
        WEATHERAPI_URL,
        params={"key": api_key, "q": f"{latitude},{longitude}", "aqi": "no"},
        timeout=30,
    )
    response.raise_for_status()
    current = response.json()["current"]
    return {
        "temperature_c": current.get("temp_c"),
        "humidity": current.get("humidity"),
        "precip_mm": current.get("precip_mm"),
        "wind_kph": current.get("wind_kph"),
        "cloud": current.get("cloud"),
        "uv": current.get("uv"),
    }


def fetch_soil_features(latitude: float, longitude: float) -> dict[str, float]:
    params = {
        "lat": latitude,
        "lon": longitude,
        "property": SOIL_PROPERTIES,
        "depth": ["0-5cm", "5-15cm"],
        "value": "mean",
    }
    payload = None
    last_error: Exception | None = None
    for url in SOILGRID_URLS:
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            break
        except requests.RequestException as exc:
            last_error = exc

    if payload is None:
        if last_error is not None:
            raise last_error
        return {}

    layers = payload.get("properties", {}).get("layers", []) or payload.get("layers", [])

    features: dict[str, float] = {}
    for layer in layers:
        name = layer.get("name")
        means: list[float] = []
        for depth in layer.get("depths", []):
            mean_value = depth.get("values", {}).get("mean")
            if mean_value is not None:
                means.append(float(mean_value))
        if name and means:
            features[f"soil_{name}_mean"] = float(np.mean(means))
    return features


def fetch_latest_ndvi(latitude: float, longitude: float, days_back: int = 60) -> dict[str, Any]:
    try:
        import planetary_computer
        import rasterio
        from pystac_client import Client
        from rasterio.warp import transform
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install the project with the [geo] extra to enable NDVI support.") from exc

    end_date = datetime.now(UTC).date()
    start_date = end_date - timedelta(days=days_back)

    catalog = Client.open(PLANETARY_COMPUTER_STAC_URL)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=[longitude - 0.01, latitude - 0.01, longitude + 0.01, latitude + 0.01],
        datetime=f"{start_date.isoformat()}/{end_date.isoformat()}",
        query={"eo:cloud_cover": {"lt": 20}},
    )
    items = list(search.items())
    if not items:
        return {"ndvi": np.nan}

    item = min(items, key=lambda candidate: candidate.properties.get("eo:cloud_cover", 100))
    signed_item = planetary_computer.sign(item)

    with rasterio.open(signed_item.assets["B08"].href) as nir_dataset, rasterio.open(
        signed_item.assets["B04"].href
    ) as red_dataset:
        xs, ys = transform("EPSG:4326", nir_dataset.crs, [longitude], [latitude])
        nir = next(nir_dataset.sample(list(zip(xs, ys)), indexes=1))[0]
        red = next(red_dataset.sample(list(zip(xs, ys)), indexes=1))[0]
        denominator = float(nir + red)
        ndvi = float((nir - red) / denominator) if denominator else np.nan

    return {
        "ndvi": ndvi,
        "satellite_scene_id": item.id,
        "satellite_scene_date": item.datetime.date().isoformat() if item.datetime else None,
    }
