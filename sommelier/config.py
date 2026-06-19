"""Application configuration for the AI sommelier assistant."""

from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SOMMELIER_",
        extra="ignore",
    )

    project_root: Path = Field(default_factory=lambda: Path.cwd())
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-5.4-mini"
    data_dir: Path = Path("data")
    catalog_dir: Path = Path("data/catalog")
    index_dir: Path = Path("data/indexes")


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()
