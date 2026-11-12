from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    predictor: Literal["local", "sagemaker"] = "local"
    artifacts_dir: Path = ROOT / "artifacts"
    static_dir: Path | None = None  # built SPA (set in Docker)
    cors_origins: list[str] = ["http://localhost:5173"]
    aws_region: str = "eu-central-1"
    sagemaker_rent_endpoint: str | None = None
    sagemaker_sale_endpoint: str | None = None
