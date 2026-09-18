from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import load

from .config import Settings
from .constants import TARGET_COLUMN
from .data_sources import (
    ensure_directory,
    load_government_dataset,
    load_kaggle_dataset,
    merge_training_sources,
)
from .features import add_feature_engineering, align_feature_frame, build_reference_history, feature_columns
from .live_features import fetch_latest_ndvi, fetch_soil_features, fetch_weather_features
from .modeling import save_model, train_and_compare_models


def train_project(settings: Settings) -> dict[str, Any]:
    ensure_directory(settings.model_dir)
    ensure_directory(settings.metrics_dir)
    ensure_directory(settings.metadata_dir)

    government_df = load_government_dataset(settings.gov_dataset_path)
    kaggle_df = None
    try:
        kaggle_df = load_kaggle_dataset(settings.kaggle_dataset_path)
    except FileNotFoundError:
        kaggle_df = None

    training_frame = merge_training_sources(government_df, kaggle_df)
    training_frame = add_feature_engineering(training_frame)
    training_frame = training_frame.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    artifacts = train_and_compare_models(training_frame)
    reference_history = build_reference_history(training_frame)

    model_path = settings.model_dir / "best_model.joblib"
    metrics_path = settings.metrics_dir / "model_comparison.csv"
    feature_path = settings.metadata_dir / "feature_columns.json"
    reference_path = settings.metadata_dir / "reference_history.csv"

    save_model(artifacts.best_estimator, str(model_path))
    artifacts.metrics.to_csv(metrics_path, index=False)
    reference_history.to_csv(reference_path, index=False)
    feature_path.write_text(
        json.dumps(
            {
                "best_model": artifacts.best_model_name,
                "feature_columns": feature_columns(training_frame),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "best_model": artifacts.best_model_name,
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "reference_path": str(reference_path),
    }


def _load_reference_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _merge_reference_features(row: dict[str, Any], reference_history: pd.DataFrame) -> dict[str, Any]:
    if reference_history.empty:
        return row

    mask = pd.Series(True, index=reference_history.index)
    for key in ["state", "district", "crop", "season"]:
        if row.get(key) is not None and key in reference_history.columns:
            mask &= reference_history[key].fillna("").astype(str) == str(row[key])

    matches = reference_history.loc[mask]
    if matches.empty:
        looser_mask = pd.Series(True, index=reference_history.index)
        for key in ["state", "crop"]:
            if row.get(key) is not None and key in reference_history.columns:
                looser_mask &= reference_history[key].fillna("").astype(str) == str(row[key])
        matches = reference_history.loc[looser_mask]

    if matches.empty:
        return row

    aggregate = matches.mean(numeric_only=True)
    for column, value in aggregate.items():
        row[column] = value

    if "historical_avg_yield" in row:
        row.setdefault("lag_yield_1", row["historical_avg_yield"])
        row.setdefault("rolling_yield_3", row["historical_avg_yield"])

    return row


def predict_live(
    settings: Settings,
    payload: dict[str, Any],
    include_ndvi: bool = True,
) -> dict[str, Any]:
    model = load(settings.model_dir / "best_model.joblib")
    metadata = json.loads((settings.metadata_dir / "feature_columns.json").read_text(encoding="utf-8"))
    reference_history = _load_reference_history(settings.metadata_dir / "reference_history.csv")

    row = {
        "crop": payload["crop"],
        "state": payload.get("state"),
        "district": payload.get("district"),
        "season": payload.get("season"),
        "area_ha": payload.get("area_ha"),
        "crop_year": payload.get("crop_year"),
        "latitude": payload["latitude"],
        "longitude": payload["longitude"],
    }
    row = _merge_reference_features(row, reference_history)

    live_features: dict[str, Any] = {}
    if settings.weatherapi_key:
        try:
            live_features.update(
                fetch_weather_features(
                    latitude=payload["latitude"],
                    longitude=payload["longitude"],
                    api_key=settings.weatherapi_key,
                )
            )
        except Exception:
            live_features["temperature_c"] = None

    try:
        live_features.update(fetch_soil_features(payload["latitude"], payload["longitude"]))
    except Exception:
        live_features["soil_service_available"] = False

    if include_ndvi:
        try:
            live_features.update(fetch_latest_ndvi(payload["latitude"], payload["longitude"]))
        except Exception:
            live_features["ndvi"] = None

    row.update(live_features)
    feature_frame = add_feature_engineering(pd.DataFrame([row]))
    aligned = align_feature_frame(feature_frame, metadata["feature_columns"])

    prediction = float(model.predict(aligned)[0])
    return {
        "predicted_yield_tonnes_per_hectare": prediction,
        "features_used": row,
    }
