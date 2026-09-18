from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .constants import TARGET_COLUMN


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["crop_year"] = pd.to_numeric(frame["crop_year"], errors="coerce")
    if "area_ha" not in frame.columns:
        frame["area_ha"] = np.nan
    frame["area_ha"] = pd.to_numeric(frame["area_ha"], errors="coerce")

    frame["area_log"] = np.log1p(frame["area_ha"].clip(lower=0))
    frame["year_sin"] = np.sin(2 * np.pi * frame["crop_year"] / 12.0)
    frame["year_cos"] = np.cos(2 * np.pi * frame["crop_year"] / 12.0)

    if "rainfall_mm" in frame.columns:
        frame["rainfall_mm"] = pd.to_numeric(frame["rainfall_mm"], errors="coerce")
        frame["rainfall_x_area"] = frame["rainfall_mm"] * frame["area_ha"]
        frame["rainfall_per_hectare"] = frame["rainfall_mm"] / frame["area_ha"].replace(0, np.nan)

    group_keys = [key for key in ["state", "district", "crop", "season"] if key in frame.columns]
    if group_keys and TARGET_COLUMN in frame.columns:
        frame = frame.sort_values(group_keys + ["crop_year"])
        grouped = frame.groupby(group_keys, dropna=False)[TARGET_COLUMN]
        frame["lag_yield_1"] = grouped.shift(1)
        frame["rolling_yield_3"] = grouped.transform(
            lambda values: values.shift(1).rolling(window=3, min_periods=1).mean()
        )
        frame["historical_avg_yield"] = grouped.transform("mean")

    return frame


def build_reference_history(df: pd.DataFrame) -> pd.DataFrame:
    keys = [key for key in ["state", "district", "crop", "season"] if key in df.columns]
    groupby_target = df.groupby(keys, dropna=False) if keys else df.groupby(lambda _: 0)

    reference = (
        groupby_target.agg(
            historical_avg_yield=(TARGET_COLUMN, "mean"),
            historical_median_yield=(TARGET_COLUMN, "median"),
            observation_count=(TARGET_COLUMN, "count"),
        ).reset_index()
    )

    if "rainfall_mm" in df.columns:
        rainfall_reference = (
            groupby_target["rainfall_mm"]
            .mean()
            .rename("historical_avg_rainfall")
            .reset_index()
        )
        reference = reference.merge(rainfall_reference, on=keys, how="left") if keys else reference.assign(
            historical_avg_rainfall=rainfall_reference["historical_avg_rainfall"].iloc[0]
        )

    return reference


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        TARGET_COLUMN,
        "production_tonnes",
    }
    return [column for column in df.columns if column not in excluded]


def align_feature_frame(df: pd.DataFrame, expected_columns: Iterable[str]) -> pd.DataFrame:
    aligned = df.copy()
    for column in expected_columns:
        if column not in aligned.columns:
            aligned[column] = np.nan
    return aligned[list(expected_columns)]
