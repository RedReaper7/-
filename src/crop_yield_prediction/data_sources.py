from __future__ import annotations

import os
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .constants import GOVERNMENT_COLUMN_ALIASES, KAGGLE_COLUMN_ALIASES


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url: str, output_path: Path, timeout: int = 60) -> Path:
    ensure_directory(output_path.parent)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)
    return output_path


def download_kaggle_dataset(dataset_slug: str, output_dir: Path, config_dir: Path | None = None) -> None:
    ensure_directory(output_dir)
    environment = os.environ.copy()
    if config_dir is not None:
        ensure_directory(config_dir)
        environment["KAGGLE_CONFIG_DIR"] = str(config_dir.resolve())
    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset_slug,
        "-p",
        str(output_dir),
        "--unzip",
    ]
    subprocess.run(command, check=True, env=environment)


def read_first_csv(path: Path) -> pd.DataFrame:
    if path.is_file():
        return pd.read_csv(path)
    csv_files = sorted(path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {path}")
    return pd.read_csv(csv_files[0])


def normalize_columns(df: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    normalized = {column: column.strip().lower().replace(" ", "_") for column in df.columns}
    df = df.rename(columns=normalized)
    rename_map = {column: aliases[column] for column in df.columns if column in aliases}
    return df.rename(columns=rename_map)


def load_government_dataset(path: Path) -> pd.DataFrame:
    df = read_first_csv(path)
    df = normalize_columns(df, GOVERNMENT_COLUMN_ALIASES)
    required = {"state", "district", "crop_year", "season", "crop"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Government dataset is missing required columns: {sorted(missing)}")

    if "yield_tonnes_per_hectare" not in df.columns:
        if {"area_ha", "production_tonnes"} - set(df.columns):
            raise ValueError(
                "Government dataset needs either yield_tonnes_per_hectare or both area_ha and "
                "production_tonnes."
            )
        df["yield_tonnes_per_hectare"] = df["production_tonnes"] / df["area_ha"].replace(0, pd.NA)

    return df


def load_kaggle_dataset(path: Path) -> pd.DataFrame:
    df = read_first_csv(path)
    df = normalize_columns(df, KAGGLE_COLUMN_ALIASES)

    join_columns = [column for column in ["state", "crop_year", "crop"] if column in df.columns]
    value_columns = [column for column in ["rainfall_mm", "temperature_c"] if column in df.columns]

    if not join_columns or not value_columns:
        raise ValueError(
            "Kaggle dataset must contain at least one join column (state/crop_year/crop) "
            "and one value column (rainfall_mm/temperature_c)."
        )

    return df[join_columns + value_columns].drop_duplicates()


def merge_training_sources(government_df: pd.DataFrame, kaggle_df: pd.DataFrame | None) -> pd.DataFrame:
    merged = government_df.copy()
    if kaggle_df is None:
        return merged

    common_keys = [column for column in ["state", "crop_year", "crop"] if column in kaggle_df.columns]
    if not common_keys:
        return merged

    supplemental = kaggle_df.groupby(common_keys, dropna=False).mean(numeric_only=True).reset_index()
    return merged.merge(supplemental, on=common_keys, how="left")


def write_json(data: dict[str, Any], output_path: Path) -> None:
    ensure_directory(output_path.parent)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
