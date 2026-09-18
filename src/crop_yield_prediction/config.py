from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    weatherapi_key: str | None = None
    kaggle_dataset: str = "chinmaynagesh/crop-yield-per-state-and-rainfall-data-of-india"
    gov_dataset_url: str | None = None
    gov_dataset_path: Path = Path("data/raw/government")
    kaggle_dataset_path: Path = Path("data/raw/kaggle")
    kaggle_config_dir: Path = Path(".kaggle")
    model_artifact_dir: Path = Path("artifacts")

    @property
    def model_dir(self) -> Path:
        return self.model_artifact_dir / "models"

    @property
    def metrics_dir(self) -> Path:
        return self.model_artifact_dir / "metrics"

    @property
    def metadata_dir(self) -> Path:
        return self.model_artifact_dir / "metadata"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
