"""Backfill missing 5-minute Nifty 50 candles from Upstox."""

import os
import sys
import json
import argparse
import yaml
import time
import requests
import logging
import pandas as pd
from pathlib import Path
from sqlalchemy import text
from datetime import datetime, timedelta
from urllib.parse import quote


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.database_manager import get_db
from src.settings import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

UPSTOX_ACCESS_TOKEN = get_settings().upstox_access_token


config_path = Path(__file__).resolve().parent.parent / 'config.yaml'
with open(config_path, 'r') as f:
    _cfg = yaml.safe_load(f)

ALL_TICKERS = _cfg['tickers']
TICKER_MAP = _cfg.get('ticker_map', {})
BACKFILL_CFG = _cfg.get('backfill', {})
UPSTOX_CFG = BACKFILL_CFG.get('upstox', {})
UPSTOX_INSTRUMENT_MAP = UPSTOX_CFG.get('instrument_map', {})
UPSTOX_INSTRUMENT_FILE = UPSTOX_CFG.get('instrument_file')
CHUNK_DAYS = int(BACKFILL_CFG.get('chunk_days', 20))
SLEEP_SECONDS = float(BACKFILL_CFG.get('sleep_between_chunks', 1))
DEFAULT_LOOKBACK_DAYS = int(BACKFILL_CFG.get('default_lookback_days', 365))
REQUEST_TIMEOUT_SECONDS = int(UPSTOX_CFG.get('request_timeout_seconds', 30))
ALLOW_SYMBOL_FALLBACK = bool(UPSTOX_CFG.get('allow_symbol_fallback', False))
INSTRUMENT_KEYS_PATH = (
    Path(__file__).resolve().parent.parent / "output" / "nifty50_instrument_keys.csv"
)

_instrument_lookup = None
_nifty50_instrument_keys = None


def _db():
    return get_db()


def _normalize_symbol(value):
    if value is None:
        return ""
    return str(value).strip().upper().replace("&", "AND").replace("-", "").replace(" ", "")


def load_nifty50_instrument_keys():
    """Load the validated Upstox key for each model ticker."""
    global _nifty50_instrument_keys
    if _nifty50_instrument_keys is not None:
        return _nifty50_instrument_keys
    if not INSTRUMENT_KEYS_PATH.exists():
        _nifty50_instrument_keys = {}
        return _nifty50_instrument_keys

    frame = pd.read_csv(INSTRUMENT_KEYS_PATH, dtype=str)
    required = {"ticker", "instrument_key"}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{INSTRUMENT_KEYS_PATH} is missing columns: {sorted(missing_columns)}"
        )

    frame["ticker"] = frame["ticker"].str.strip().str.upper()
    frame["instrument_key"] = frame["instrument_key"].str.strip()
    if frame["ticker"].duplicated().any():
        duplicates = sorted(frame.loc[frame["ticker"].duplicated(), "ticker"].unique())
        raise ValueError(f"Duplicate ticker mappings found: {duplicates}")

    _nifty50_instrument_keys = dict(
        zip(frame["ticker"], frame["instrument_key"])
    )
    logger.info(
        "Loaded %d Nifty 50 instrument keys from %s",
        len(_nifty50_instrument_keys),
        INSTRUMENT_KEYS_PATH,
    )
    return _nifty50_instrument_keys


def load_instrument_lookup():
    """Load an optional Upstox instrument master file."""
    global _instrument_lookup
    if _instrument_lookup is not None:
        return _instrument_lookup

    if not UPSTOX_INSTRUMENT_FILE:
        _instrument_lookup = {}
        return _instrument_lookup

    instrument_path = Path(UPSTOX_INSTRUMENT_FILE)
    if not instrument_path.is_absolute():
        instrument_path = Path(__file__).resolve().parent.parent / instrument_path

    if not instrument_path.exists():
        raise FileNotFoundError(
            f"Configured Upstox instrument file not found: {instrument_path}"
        )

    suffix = instrument_path.suffix.lower()
    if suffix == ".csv":
        raw = pd.read_csv(instrument_path)
    elif suffix == ".json":
        with open(instrument_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        raw = pd.DataFrame(payload)
    else:
        raise ValueError(
            "Unsupported upstox instrument file format. Use CSV or JSON."
        )

    if raw.empty:
        _instrument_lookup = {}
        return _instrument_lookup

    column_map = {c.lower(): c for c in raw.columns}
    key_col = next((column_map[c] for c in ("instrument_key", "instrumentkey") if c in column_map), None)
    symbol_col = next((column_map[c] for c in ("trading_symbol", "tradingsymbol", "symbol", "name") if c in column_map), None)
    isin_col = next((column_map[c] for c in ("isin",) if c in column_map), None)

    if not key_col or not symbol_col:
        raise ValueError(
            "Instrument file must contain instrument_key and a trading symbol column."
        )

    lookup = {}
    for _, row in raw.iterrows():
        instrument_key = row.get(key_col)
        symbol = row.get(symbol_col)
        if pd.isna(instrument_key) or pd.isna(symbol):
            continue

        normalized_symbol = _normalize_symbol(symbol)
        if normalized_symbol:
            lookup[normalized_symbol] = str(instrument_key).strip()

        if isin_col:
            isin_value = row.get(isin_col)
            if pd.notna(isin_value):
                lookup[str(isin_value).strip().upper()] = str(instrument_key).strip()

    _instrument_lookup = lookup
    logger.info(f"Loaded {len(_instrument_lookup)} Upstox instrument aliases from {instrument_path}")
    return _instrument_lookup


def resolve_instrument_key(ticker):
    """Find the Upstox instrument key for one ticker."""
    csv_mapping = load_nifty50_instrument_keys()
    if ticker in csv_mapping:
        return csv_mapping[ticker]

    if ticker in UPSTOX_INSTRUMENT_MAP:
        return UPSTOX_INSTRUMENT_MAP[ticker]

    mapped_symbol = TICKER_MAP.get(ticker, ticker)
    lookup = load_instrument_lookup()

    candidates = [
        ticker,
        mapped_symbol,
        f"NSE_EQ|{mapped_symbol}",
    ]
    for candidate in candidates:
        normalized = _normalize_symbol(candidate)
        if normalized in lookup:
            return lookup[normalized]
        candidate_upper = str(candidate).strip().upper()
        if candidate_upper in lookup:
            return lookup[candidate_upper]

    if ALLOW_SYMBOL_FALLBACK:
        return f"NSE_EQ|{mapped_symbol}"

    raise KeyError(
        f"No Upstox instrument key found for ticker '{ticker}'. "
        "Add it under backfill.upstox.instrument_map or configure backfill.upstox.instrument_file."
    )


def get_last_date(ticker):
    """Return the latest stored candle for one ticker."""
    query = text("SELECT MAX(bucket_time) FROM min_5 WHERE ticker = :ticker")
    with _db().engine.connect() as conn:
        result = conn.execute(query, {"ticker": ticker}).scalar()

    if result is None:
        return None
    return pd.to_datetime(result)


def fetch_and_resample_5min(ticker, instrument_key, from_date, to_date, headers):
    """Fetch 1-minute candles and combine them into 5-minute bars."""
    encoded_key = quote(instrument_key, safe="")
    url = (
        f"https://api.upstox.com/v2/historical-candle/{encoded_key}/1minute/"
        f"{to_date.strftime('%Y-%m-%d')}/{from_date.strftime('%Y-%m-%d')}"
    )

    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.RequestException as exc:
        logger.exception("Upstox request failed for %s: %s", ticker, exc)
        return pd.DataFrame()

    if response.status_code != 200:
        logger.error("API error %s for %s: %s", response.status_code, ticker, response.text[:800])
        return pd.DataFrame()

    candles = response.json().get("data", {}).get("candles", [])
    if not candles:
        return pd.DataFrame()

    frame = pd.DataFrame(
        candles,
        columns=["bucket_time", "open", "high", "low", "close", "volume", "oi"],
    )
    frame["bucket_time"] = pd.to_datetime(frame["bucket_time"])
    frame = frame.set_index("bucket_time").sort_index()
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])

    frame_5min = (
        frame[["open", "high", "low", "close", "volume"]]
        .resample("5min", origin="start_day", offset="15min")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
        .between_time("09:15", "15:30")
    )
    frame_5min = frame_5min[frame_5min.index.dayofweek < 5]
    frame_5min["ticker"] = ticker
    return frame_5min.reset_index()


def fetch_chunk(ticker, start, end):
    """Download and clean one date chunk."""
    if not UPSTOX_ACCESS_TOKEN:
        logger.error("UPSTOX_ACCESS_TOKEN is not in .env")
        return pd.DataFrame()

    try:
        instrument_key = resolve_instrument_key(ticker)
    except Exception as exc:
        logger.exception("Instrument key resolution failed for %s: %s", ticker, exc)
        return pd.DataFrame()

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
    }
    df = fetch_and_resample_5min(ticker, instrument_key, start, end, headers)
    if df.empty:
        return df

    if df["bucket_time"].dt.tz is not None:
        df["bucket_time"] = (
            df["bucket_time"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        )

    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float32")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("int64")
    return df.dropna()


def delete_overlap(ticker, start, end):
    """Delete overlapping rows before inserting refreshed candles."""
    query = text("""
        DELETE FROM min_5
        WHERE ticker = :ticker
        AND bucket_time >= :start
        AND bucket_time <= :end
    """)
    with _db().engine.begin() as conn:
        conn.execute(query, {"ticker": ticker, "start": start, "end": end})


def save_to_db(df):
    """Append cleaned candles to the 5-minute table."""
    df.to_sql('min_5', _db().engine, if_exists='append', index=False, method='multi')


def backfill_ticker(ticker, from_date=None, to_date=None):
    """Backfill one ticker over the requested date range."""
    last_date = get_last_date(ticker)
    today = pd.to_datetime(to_date).to_pydatetime() if to_date else datetime.now()

    if from_date:
        start_date = pd.to_datetime(from_date).date()
        logger.info(f"{ticker} | Explicit range: {start_date} -> {today.date()}")
    elif last_date is None:
        start_date = (today - timedelta(days=DEFAULT_LOOKBACK_DAYS)).date()
        logger.info(f"{ticker} | No DB data, starting from: {start_date} | Today: {today.date()}")
    else:
        start_date = last_date.date()
        logger.info(f"{ticker} | DB ends: {last_date.date()} | Today: {today.date()}")


    chunk_size = CHUNK_DAYS
    total_inserted = 0

    while start_date <= today.date():
        end_date = min(start_date + timedelta(days=chunk_size - 1), today.date())
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())
        logger.info(f"fetching {start_date} -> {end_date} for {ticker}")

        df = fetch_chunk(ticker, start=start_dt, end=end_dt)

        if not df.empty:
            delete_overlap(ticker, start=start_dt, end=end_dt)
            save_to_db(df)
            total_inserted += len(df)
            logger.info(f"inserted {len(df)} rows for {ticker}")
        else:
            logger.warning(f"no data for chunk {start_date} -> {end_date} for {ticker}")

        start_date = end_date + timedelta(days=1)
        time.sleep(SLEEP_SECONDS)

    if total_inserted == 0:
        logger.warning("FAILED - 0 rows inserted for %s (check API response)", ticker)
    else:
        logger.info("SUCCESS - %d rows inserted for %s", total_inserted, ticker)
    return total_inserted


def backfill_all(selected_tickers=None, from_date=None, to_date=None):
    """Backfill selected tickers and report any failures."""
    if not UPSTOX_ACCESS_TOKEN:
        logger.error("Please set UPSTOX_ACCESS_TOKEN in the environment before running.")
        sys.exit(1)

    logger.info(f"Upstox Backfill started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    success, failed = [], []
    tickers_to_run = selected_tickers if selected_tickers else ALL_TICKERS

    for ticker in tickers_to_run:
        try:
            rows_inserted = backfill_ticker(
                ticker,
                from_date=from_date,
                to_date=to_date,
            )
            if rows_inserted > 0:
                success.append(ticker)
            else:
                failed.append(ticker)
        except Exception as e:
            logger.exception(f"{ticker} FAILED: {e}")
            failed.append(ticker)

    logger.info(f"Done. Success: {len(success)} | Failed: {len(failed)}")
    if failed:
        logger.warning(f"Failed tickers: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill 5-minute stock data from Upstox.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Optional list of tickers to backfill, e.g. --tickers HDFCBANK INFY",
    )
    parser.add_argument("--ticker", help="Optional single ticker to backfill.")
    parser.add_argument("--from-date", help="Inclusive start date in YYYY-MM-DD format.")
    parser.add_argument("--to-date", help="Inclusive end date in YYYY-MM-DD format.")
    args = parser.parse_args()

    if args.ticker and args.tickers:
        parser.error("Use either --ticker or --tickers, not both.")

    selected = None
    if args.ticker:
        selected = [str(args.ticker).strip().upper()]
    elif args.tickers:
        selected = [str(t).strip().upper() for t in args.tickers]
    if selected:
        unknown = [t for t in selected if t not in ALL_TICKERS]
        if unknown:
            logger.error(f"Unknown tickers requested: {unknown}")
            sys.exit(1)

    backfill_all(
        selected_tickers=selected,
        from_date=args.from_date,
        to_date=args.to_date,
    )
