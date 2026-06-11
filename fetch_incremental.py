"""
fetch_incremental.py - incremental market-impact news fetcher

This script appends newly fetched news into the latest local news_gap CSV and
updates news_data/last_run.json so repeated runs only pull the missing window.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from src.settings import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
NEWS_DIR = BASE_DIR / "news_data"
NEWS_DIR.mkdir(parents=True, exist_ok=True)
LAST_RUN_FILE = NEWS_DIR / "last_run.json"

SETTINGS = get_settings()

NEWSAPI_KEY = SETTINGS.newsapi_key or ""
GNEWS_KEY = SETTINGS.gnews_key or ""
SWEEP_MODE = SETTINGS.sweep_mode
LOOKBACK_HOURS = SETTINGS.lookback_hours
REQUEST_TIMEOUT_SECONDS = SETTINGS.news_request_timeout_seconds
NEWSAPI_MAX_PAGES_PER_QUERY = SETTINGS.newsapi_max_pages_per_query

QUERIES = [
    "India stock market",
    "Nifty 50",
    "NSE India",
    "RBI policy India",
    "Indian corporate earnings",
]


def latest_output_file(today: date) -> Path:
    csv_files = sorted(NEWS_DIR.glob("news_gap_*.csv"))
    if csv_files:
        return csv_files[-1]
    return NEWS_DIR / f"news_gap_{today.strftime('%Y_%m_%d')}.csv"


def load_last_run_time() -> datetime | None:
    if not LAST_RUN_FILE.exists():
        return None
    try:
        with open(LAST_RUN_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    raw = payload.get("last_run")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        try:
            return datetime.combine(date.fromisoformat(raw), datetime.min.time())
        except ValueError:
            return None


def save_last_run_time(ts: datetime) -> None:
    with open(LAST_RUN_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_run": ts.replace(microsecond=0).isoformat()}, f)


def utc_now_naive() -> datetime:
    """Return the current UTC timestamp without tzinfo for consistent local storage."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def to_api_utc(ts: datetime) -> str:
    """Format a naive-or-aware datetime as an ISO UTC timestamp for news APIs."""
    aware = ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def normalize_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["Archive", "Date", "Headline", "Description", "Headline link", "Source", "Query"])
    df = pd.DataFrame(rows)
    for col in ["Archive", "Date", "Headline", "Description", "Headline link", "Source", "Query"]:
        if col not in df.columns:
            df[col] = ""
    return df[["Archive", "Date", "Headline", "Description", "Headline link", "Source", "Query"]]


def _format_article_date(raw_value: Any) -> str:
    ts = pd.to_datetime(raw_value, errors="coerce")
    if pd.isna(ts):
        return ""
    return ts.strftime("%d-%m-%Y")


def fetch_newsapi(from_dt: datetime, to_dt: datetime) -> list[dict[str, Any]]:
    if not NEWSAPI_KEY:
        logger.warning("Skipping NewsAPI: missing key")
        return []

    endpoint = "https://newsapi.org/v2/everything"
    rows: list[dict[str, Any]] = []
    for query in QUERIES:
        for page in range(1, NEWSAPI_MAX_PAGES_PER_QUERY + 1):
            params = {
                "q": query,
                "from": to_api_utc(from_dt),
                "to": to_api_utc(to_dt),
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 100,
                "page": page,
                "apiKey": NEWSAPI_KEY,
            }
            try:
                resp = requests.get(endpoint, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
                if resp.status_code != 200:
                    logger.error(f"NewsAPI error {resp.status_code}: {resp.text[:180]}")
                    break
                payload = resp.json()
            except requests.RequestException as exc:
                logger.exception(f"NewsAPI request failed: {exc}")
                break

            articles = payload.get("articles", []) or []
            if not articles:
                break
            for item in articles:
                rows.append(
                    {
                        "Archive": "NewsAPI",
                        "Date": _format_article_date(item.get("publishedAt")),
                        "Headline": item.get("title") or "",
                        "Description": item.get("description") or "",
                        "Headline link": item.get("url") or "",
                        "Source": item.get("source", {}).get("name") or "NewsAPI",
                        "Query": query,
                    }
                )
    return rows


def fetch_gnews(from_dt: datetime, to_dt: datetime) -> list[dict[str, Any]]:
    if not GNEWS_KEY:
        logger.warning("Skipping GNews: missing key")
        return []

    endpoint = "https://gnews.io/api/v4/search"
    rows: list[dict[str, Any]] = []
    for query in QUERIES:
        params = {
            "q": query,
            "from": to_api_utc(from_dt),
            "to": to_api_utc(to_dt),
            "lang": "en",
            "max": 10,
            "token": GNEWS_KEY,
        }
        try:
            resp = requests.get(endpoint, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            if resp.status_code != 200:
                logger.error(f"GNews error {resp.status_code}: {resp.text[:180]}")
                continue
            payload = resp.json()
        except requests.RequestException as exc:
            logger.exception(f"GNews request failed: {exc}")
            continue

        for item in payload.get("articles", []) or []:
            rows.append(
                {
                    "Archive": "GNews",
                    "Date": _format_article_date(item.get("publishedAt")),
                    "Headline": item.get("title") or "",
                    "Description": item.get("description") or "",
                    "Headline link": item.get("url") or "",
                    "Source": item.get("source", {}).get("name") or "GNews",
                    "Query": query,
                }
            )
    return rows


def deduplicate_csv(path: Path) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    for col in ["Headline link", "Headline", "Date"]:
        if col not in df.columns:
            df[col] = ""
    key = np.where(
        df["Headline link"].fillna("").astype(str).str.strip().ne(""),
        df["Headline link"].fillna("").astype(str).str.strip().str.lower(),
        df["Date"].fillna("").astype(str) + "|" + df["Headline"].fillna("").astype(str).str.strip().str.lower(),
    )
    df["_dedupe_key"] = key
    df = df.drop_duplicates(subset=["_dedupe_key"], keep="last").drop(columns=["_dedupe_key"])
    df.to_csv(path, index=False)


def main() -> None:
    today = utc_now_naive()
    if SWEEP_MODE == "overnight":
        since = today - timedelta(hours=LOOKBACK_HOURS)
        logger.info(f"Overnight sweep mode: fetching since {since.isoformat()}")
    else:
        since = load_last_run_time()
        if since is None:
            since = today - timedelta(days=1)
            logger.info(f"No last_run found. Defaulting to {since.date()}")
        else:
            logger.info(f"Incremental mode: last run was {since.date()}")

    output_file = latest_output_file(today.date())
    rows = fetch_newsapi(since, today) + fetch_gnews(since, today)
    df_new = normalize_rows(rows)
    if df_new.empty:
        logger.info("No new articles fetched.")
        save_last_run_time(today)
        return

    write_header = not output_file.exists()
    df_new.to_csv(output_file, mode="a", index=False, header=write_header)
    deduplicate_csv(output_file)
    save_last_run_time(today)
    logger.info(f"Appended {len(df_new)} rows to {output_file}")


if __name__ == "__main__":
    main()
