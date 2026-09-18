from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crop_yield_prediction.config import get_settings
from crop_yield_prediction.data_sources import download_file, download_kaggle_dataset, ensure_directory


def main() -> None:
    settings = get_settings()
    ensure_directory(settings.gov_dataset_path)
    ensure_directory(settings.kaggle_dataset_path)

    try:
        download_kaggle_dataset(
            settings.kaggle_dataset,
            settings.kaggle_dataset_path,
            settings.kaggle_config_dir,
        )
        print(f"Kaggle dataset downloaded to {settings.kaggle_dataset_path}")
    except Exception as exc:
        print(
            "Kaggle download was skipped or failed. "
            "Make sure the Kaggle CLI is installed and .kaggle/kaggle.json is configured."
        )
        print(f"Reason: {exc}")

    if settings.gov_dataset_url:
        target_path = Path(settings.gov_dataset_path) / "government_dataset.csv"
        download_file(settings.gov_dataset_url, target_path)
        print(f"Government dataset downloaded to {target_path}")
    else:
        print(
            "No GOV_DATASET_URL set. Place the government CSV/ZIP-extracted CSV inside "
            f"{settings.gov_dataset_path}"
        )


if __name__ == "__main__":
    main()
