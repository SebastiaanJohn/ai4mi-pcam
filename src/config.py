"""Configuration file for the project."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration settings for the project."""

    root_dir: Path = Path(__file__).resolve().parent.parent
    experiments_dir: Path = root_dir / "experiments"
    data_dir: Path = root_dir / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"

    wandb_project: str = "pcam-classification"
    wandb_entity: str = "ai4mi"


settings = Settings()
