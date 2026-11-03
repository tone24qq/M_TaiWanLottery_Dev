"""Application configuration using Pydantic Settings."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    DATABASE_URL: str = (
        "postgresql+asyncpg://postgres:password@localhost:5432/taiwan_lottery"
    )
    DATABASE_URL_SYNC: str = (
        "postgresql+psycopg2://postgres:password@localhost:5432/taiwan_lottery"
    )

    # App
    APP_NAME: str = "賓果研究系統"
    APP_ENV: str = "development"
    DEBUG: bool = False
    SECRET_KEY: str = "change-me"

    # ML
    MODEL_ARTIFACTS_DIR: Path = Path("./model_artifacts")
    TORCH_DEVICE: str = "cpu"

    # Scraper
    SCRAPER_ENABLED: bool = True
    SYNC_INTERVAL_SECONDS: int = Field(45, ge=5, le=300)
    CPU_THREADS: int = Field(2, ge=1, le=8)
    TRAIN_WINDOW: int = Field(1000, ge=100, le=10000)
    RETRAIN_MIN_DRAWS: int = Field(36, ge=1)
    RETRAIN_INTERVAL_SECONDS: int = Field(10800, ge=60)
    HISTORY_MIN_DRAWS: int = Field(100, ge=10, le=10000)
    MAX_BACKFILL_DAYS: int = Field(31, ge=3, le=366)
    AUTO_TRAIN: bool = True
    ADAPTIVE_ENABLED: bool = True
    AUTO_VALIDATE: bool = True
    ADAPTIVE_SHADOW_BUDGET: int = Field(3, ge=2, le=3)
    VALIDATION_COOLDOWN_SECONDS: int = Field(86400, ge=3600)
    VALIDATION_COUNT: int = Field(100, ge=100, le=1000)
    BINGO_SCRAPE_URL: str = (
        "https://www.taiwanlottery.com/lotto/bingobingo/history.aspx"
    )


settings = Settings()
