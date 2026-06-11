"""Load application settings from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    """Settings shared by scripts and services."""

    api_key: str
    database_uri: str | None
    db_host: str | None
    db_port: int
    db_name: str | None
    db_user: str | None
    db_password: str | None
    upstox_access_token: str | None
    newsapi_key: str | None
    gnews_key: str | None
    guardian_key: str | None

    data_folder: str | None
    sweep_mode: str
    lookback_hours: int
    news_request_timeout_seconds: int
    newsapi_max_pages_per_query: int


def _get_optional(name: str) -> str | None:
    """Read text and treat a blank value as missing."""
    value = os.getenv(name, "").strip()
    return value or None


def _get_int(name: str, default: int) -> int:
    """Read an integer setting with a default value."""
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc


def get_settings() -> Settings:
    """Build the current application settings."""
    return Settings(
        api_key=os.getenv("NIFTY_API_KEY", "").strip(),
        database_uri=_get_optional("DATABASE_URI"),
        db_host=_get_optional("DB_HOST"),
        db_port=_get_int("DB_PORT", 5432),
        db_name=_get_optional("DB_NAME"),
        db_user=_get_optional("DB_USER"),
        db_password=_get_optional("DB_PASSWORD"),
        upstox_access_token=_get_optional("UPSTOX_ACCESS_TOKEN"),
        newsapi_key=_get_optional("NEWSAPI_KEY"),
        gnews_key=_get_optional("GNEWS_KEY"),
        guardian_key=_get_optional("GUARDIAN_KEY"),
        data_folder=_get_optional("DATA_FOLDER"),
        sweep_mode=os.getenv("SWEEP_MODE", "incremental").strip().lower(),
        lookback_hours=_get_int("LOOKBACK_HOURS", 2),
        news_request_timeout_seconds=_get_int("NEWS_REQUEST_TIMEOUT_SECONDS", 30),
        newsapi_max_pages_per_query=_get_int("NEWSAPI_MAX_PAGES_PER_QUERY", 2),
    )
