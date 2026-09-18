from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

from .constants import TARGET_COLUMN

DEFAULT_MAX_TRAINING_ROWS = 5_000
EXCLUDED_FEATURE_COLUMNS = {TARGET_COLUMN, "production_tonnes"}

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None


@dataclass
class TrainingArtifacts:
    best_model_name: str
    best_estimator: Any
    metrics: pd.DataFrame


def build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric_features = frame.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [
        column for column in frame.columns if column not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def model_specs(preprocessor: ColumnTransformer) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    specs: dict[str, tuple[Pipeline, dict[str, list[Any]]]] = {
        "linear_regression": (
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("regressor", LinearRegression()),
                ]
            ),
            {
                "regressor__fit_intercept": [True, False],
            },
        ),
        "random_forest": (
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "regressor",
                        RandomForestRegressor(
                            random_state=42,
                            n_estimators=40,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            {
                "regressor__n_estimators": [40],
                "regressor__max_depth": [20],
                "regressor__min_samples_split": [2],
            },
        ),
    }

    if XGBRegressor is not None:
        specs["xgboost"] = (
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "regressor",
                        XGBRegressor(
                            objective="reg:squarederror",
                            random_state=42,
                            n_estimators=80,
                            learning_rate=0.05,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            {
                "regressor__max_depth": [4],
                "regressor__subsample": [0.9],
                "regressor__colsample_bytree": [0.9],
            },
        )

    return specs


def _metric_row(model_name: str, estimator: Any, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | str]:
    predictions = estimator.predict(x_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    return {
        "model": model_name,
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": rmse,
        "r2": float(r2_score(y_test, predictions)),
    }


def train_and_compare_models(
    frame: pd.DataFrame,
    max_training_rows: int = DEFAULT_MAX_TRAINING_ROWS,
) -> TrainingArtifacts:
    if len(frame) > max_training_rows:
        frame = frame.sample(n=max_training_rows, random_state=42).reset_index(drop=True)

    x = frame.drop(columns=[column for column in EXCLUDED_FEATURE_COLUMNS if column in frame.columns])
    y = frame[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    preprocessor = build_preprocessor(x_train)
    comparison_rows: list[dict[str, float | str]] = []
    best_score = float("-inf")
    best_name = ""
    best_estimator = None

    for model_name, (pipeline, grid) in model_specs(preprocessor).items():
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            cv=2,
            n_jobs=1,
            scoring="r2",
            verbose=0,
        )
        search.fit(x_train, y_train)
        comparison_rows.append(_metric_row(model_name, search.best_estimator_, x_test, y_test))

        if search.best_score_ > best_score:
            best_score = float(search.best_score_)
            best_name = model_name
            best_estimator = search.best_estimator_

    metrics = pd.DataFrame(comparison_rows).sort_values("r2", ascending=False).reset_index(drop=True)
    if best_estimator is None:
        raise RuntimeError("No model could be trained.")

    return TrainingArtifacts(best_model_name=best_name, best_estimator=best_estimator, metrics=metrics)


def save_model(estimator: Any, output_path: str) -> None:
    dump(estimator, output_path)
