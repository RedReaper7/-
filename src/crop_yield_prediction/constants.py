TARGET_COLUMN = "yield_tonnes_per_hectare"

GOVERNMENT_COLUMN_ALIASES = {
    "state_name": "state",
    "state": "state",
    "state/ut": "state",
    "district_name": "district",
    "district": "district",
    "crop_year": "crop_year",
    "year": "crop_year",
    "crop": "crop",
    "season": "season",
    "area": "area_ha",
    "area_ha": "area_ha",
    "area_hectare": "area_ha",
    "production": "production_tonnes",
    "production_tonnes": "production_tonnes",
    "yield": TARGET_COLUMN,
    "yield_tonnes_per_hectare": TARGET_COLUMN,
    "latitude": "latitude",
    "longitude": "longitude",
    "lat": "latitude",
    "lon": "longitude",
    "lng": "longitude",
}

KAGGLE_COLUMN_ALIASES = {
    "state_name": "state",
    "state": "state",
    "crop_year": "crop_year",
    "year": "crop_year",
    "crop": "crop",
    "average_rainfall": "rainfall_mm",
    "average_rain_fall_mm_per_year": "rainfall_mm",
    "annual_rainfall": "rainfall_mm",
    "rainfall": "rainfall_mm",
    "rainfall_mm": "rainfall_mm",
    "temperature": "temperature_c",
    "avg_temp": "temperature_c",
    "temperature_c": "temperature_c",
}

SOIL_PROPERTIES = ["bdod", "cec", "clay", "nitrogen", "phh2o", "sand", "silt", "soc"]

