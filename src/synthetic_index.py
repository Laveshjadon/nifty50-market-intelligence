"""Build and save a weighted synthetic Nifty 50 index."""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml
from sqlalchemy import bindparam, text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
OHLC_COLS = ["open", "high", "low", "close"]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.database_manager import get_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str | Path = CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_weights(config: dict) -> pd.Series:
    tickers = [str(t).strip().upper() for t in config["tickers"]]
    raw_weights = config.get("synthetic_index", {}).get("weights", {}) or {}

    if not raw_weights:
        raise ValueError("No synthetic_index.weights found in config.yaml.")

    weights = pd.Series(
        {str(t).strip().upper(): float(w) for t, w in raw_weights.items()},
        dtype="float64",
    )

    missing = [t for t in tickers if t not in weights.index]
    extra   = [t for t in weights.index if t not in tickers]
    if missing:
        raise ValueError("Missing synthetic weights for tickers: " + ", ".join(missing))
    if extra:
        raise ValueError("Synthetic weights include unknown tickers: " + ", ".join(extra))
    if (weights <= 0).any():
        bad = weights[weights <= 0].index.tolist()
        raise ValueError("Synthetic weights must be positive for: " + ", ".join(bad))

    weights    = weights.loc[tickers]
    normalized = weights / float(weights.sum())


    logger.debug("Normalized weights:\n%s", normalized.to_string())
    logger.info(
        "Loaded weights for %d tickers (min=%.4f, max=%.4f, sum=%.6f)",
        len(normalized),
        float(normalized.min()),
        float(normalized.max()),
        float(normalized.sum()),
    )
    return normalized


def load_constituent_data(tickers: list[str]) -> pd.DataFrame:
    """Load OHLCV rows for the configured constituents."""
    if not tickers:
        raise ValueError("At least one ticker is required to load constituent data.")

    query = text("""
        SELECT ticker, bucket_time, open, high, low, close, volume
        FROM   min_5
        WHERE  ticker IN :tickers
    """).bindparams(bindparam("tickers", expanding=True))

    db = get_db()
    df = pd.read_sql_query(query, db.engine, params={"tickers": tickers})

    if df.empty:
        raise ValueError("No constituent data found in min_5 for the requested tickers.")

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()


    df["bucket_time"] = pd.to_datetime(df["bucket_time"])
    if df["bucket_time"].dt.tz is not None:
        df["bucket_time"] = df["bucket_time"].dt.tz_localize(None)

    df = (
        df
        .sort_values(["ticker", "bucket_time"])
        .drop_duplicates(subset=["ticker", "bucket_time"], keep="last")
    )
    return df


def filter_complete_timestamps(
    df: pd.DataFrame,
    expected_tickers: list[str],
    require_full: bool,
    min_coverage: float,
) -> pd.DataFrame:
    counts        = df.groupby("bucket_time")["ticker"].nunique()
    required_count = len(expected_tickers)

    if require_full:
        valid_times = counts[counts == required_count].index
    else:
        min_count   = max(1, math.ceil(required_count * min_coverage))
        valid_times = counts[counts >= min_count].index

    filtered = df[df["bucket_time"].isin(valid_times)].copy()
    if filtered.empty:
        raise ValueError(
            "No timestamps met the constituent coverage rule. "
            "Try lowering synthetic_index.min_constituent_coverage_ratio."
        )

    dropped = len(df["bucket_time"].unique()) - len(valid_times)
    if dropped:
        logger.warning(
            "Dropped %d timestamps that failed constituent coverage check.", dropped
        )
    return filtered


def build_synthetic_index_frame(config: dict) -> pd.DataFrame:
    tickers       = [str(t).strip().upper() for t in config["tickers"]]
    synthetic_cfg = config.get("synthetic_index", {})
    base_value    = float(synthetic_cfg.get("base_value", 1000.0))
    require_full  = bool(synthetic_cfg.get("require_full_constituent_history", True))
    min_coverage  = float(synthetic_cfg.get("min_constituent_coverage_ratio", 1.0))
    volume_method = str(synthetic_cfg.get("volume_method", "sum")).lower()

    if base_value <= 0:
        raise ValueError(f"synthetic_index.base_value must be positive, got: {base_value}")
    if not 0 < min_coverage <= 1:
        raise ValueError("synthetic_index.min_constituent_coverage_ratio must be in (0, 1].")
    if volume_method not in ("sum", "weight"):
        raise ValueError("synthetic_index.volume_method must be 'sum' or 'weight'.")

    weights = load_weights(config)
    raw     = load_constituent_data(tickers)

    pivot = raw.pivot(index="bucket_time", columns="ticker", values="close")
    if require_full:

        full_pivot = pivot.dropna(axis=0, how="any")
        if full_pivot.empty:
            raise ValueError(
                "No single timestamp exists where all constituents have data. "
                "Check for missing history or set synthetic_index.require_full_constituent_history=false."
            )
        first_common_ts = full_pivot.index.min()
        first_close = full_pivot.loc[first_common_ts].astype("float64")
        logger.info(
            "Rebasing all constituents from first common timestamp: %s", first_common_ts
        )
    else:
        first_close = raw.sort_values("bucket_time").groupby("ticker")["close"].first().astype("float64")
        missing_base = [ticker for ticker in tickers if ticker not in first_close.index]
        if missing_base:
            raise ValueError(
                "Could not determine a base close for: " + ", ".join(missing_base)
            )
        logger.warning(
            "Using per-ticker base closes because require_full_constituent_history=false."
        )


    raw = filter_complete_timestamps(
        raw,
        expected_tickers=tickers,
        require_full=require_full,
        min_coverage=min_coverage,
    )


    weighted_frames: list[pd.DataFrame] = []
    coverage_weights = raw.assign(weight=raw["ticker"].map(weights.to_dict()))
    weight_sum_by_time = coverage_weights.groupby("bucket_time")["weight"].sum()

    for ticker in tickers:
        ticker_df  = raw[raw["ticker"] == ticker].copy()
        if ticker_df.empty:
            continue
        base_close = float(first_close.loc[ticker])
        base_weight = float(weights.loc[ticker])
        if require_full:
            ticker_df["effective_weight"] = base_weight
        else:
            ticker_df["effective_weight"] = (
                base_weight / ticker_df["bucket_time"].map(weight_sum_by_time).astype("float64")
            )

        close_f        = ticker_df["close"].astype("float64")
        weighted_close = (close_f / base_close) * ticker_df["effective_weight"] * base_value


        ticker_df["w_open"]  = weighted_close * (ticker_df["open"].astype("float64")  / close_f)
        ticker_df["w_high"]  = weighted_close * (ticker_df["high"].astype("float64")  / close_f)
        ticker_df["w_low"]   = weighted_close * (ticker_df["low"].astype("float64")   / close_f)
        ticker_df["w_close"] = weighted_close

        if volume_method == "sum":
            ticker_df["w_volume"] = ticker_df["volume"].astype("float64")
        else:
            ticker_df["w_volume"] = ticker_df["volume"].astype("float64") * ticker_df["effective_weight"]

        weighted_frames.append(
            ticker_df[["bucket_time", "w_open", "w_high", "w_low", "w_close", "w_volume"]]
        )

    combined = pd.concat(weighted_frames, ignore_index=True)

    synthetic = (
        combined
        .groupby("bucket_time", as_index=False)
        .agg(
            open   = ("w_open",   "sum"),
            high   = ("w_high",   "sum"),
            low    = ("w_low",    "sum"),
            close  = ("w_close",  "sum"),
            volume = ("w_volume", "sum"),
        )
        .sort_values("bucket_time")
        .reset_index(drop=True)
    )


    near_zero_vol = (synthetic["volume"] > 0) & (synthetic["volume"] < 0.5)
    if volume_method == "weight" and near_zero_vol.any():
        logger.warning(
            "%d candles have weighted volume < 0.5 and will round to 0. "
            "Consider using volume_method='sum' for low-volume constituents.",
            near_zero_vol.sum(),
        )

    synthetic["volume"] = synthetic["volume"].round().astype("int64")
    for col in OHLC_COLS:
        synthetic[col] = synthetic[col].astype("float64").round(4)


    _validate_ohlc(synthetic)

    return synthetic


def _validate_ohlc(df: pd.DataFrame) -> None:
    """Reject synthetic candles with invalid OHLC relationships."""
    bad_high_close = df[df["high"] < df["close"]]
    if not bad_high_close.empty:
        raise ValueError(
            f"Index high < close at {len(bad_high_close)} timestamps "
            f"(first: {bad_high_close['bucket_time'].iloc[0]})"
        )
    bad_high_open = df[df["high"] < df["open"]]
    if not bad_high_open.empty:
        raise ValueError(
            f"Index high < open at {len(bad_high_open)} timestamps "
            f"(first: {bad_high_open['bucket_time'].iloc[0]})"
        )
    bad_low_close = df[df["low"] > df["close"]]
    if not bad_low_close.empty:
        raise ValueError(
            f"Index low > close at {len(bad_low_close)} timestamps "
            f"(first: {bad_low_close['bucket_time'].iloc[0]})"
        )
    bad_low_open = df[df["low"] > df["open"]]
    if not bad_low_open.empty:
        raise ValueError(
            f"Index low > open at {len(bad_low_open)} timestamps "
            f"(first: {bad_low_open['bucket_time'].iloc[0]})"
        )


def save_synthetic_index(config: dict, synthetic_df: pd.DataFrame) -> int:
    index_name = config["index"].get("synthetic_name", "NIFTY50_SYNTHETIC")

    rows = get_db().replace_index_data(index_name, synthetic_df)
    logger.info("Saved %d rows to index '%s'.", rows, index_name)
    return rows


def build_and_save_synthetic_index(
    config_path: str | Path = CONFIG_PATH,
) -> Tuple[pd.DataFrame, int]:
    config       = load_config(config_path)
    synthetic_df = build_synthetic_index_frame(config)
    inserted     = save_synthetic_index(config, synthetic_df)
    return synthetic_df, inserted


def main() -> None:
    try:
        synthetic_df, inserted_rows = build_and_save_synthetic_index()
    except Exception as exc:
        logger.error("Failed to build synthetic index: %s", exc)
        sys.exit(1)

    logger.info("Built %d synthetic candles, saved %d rows.", len(synthetic_df), inserted_rows)

    if not synthetic_df.empty:
        logger.info(
            "Synthetic range: %s -> %s",
            synthetic_df["bucket_time"].min(),
            synthetic_df["bucket_time"].max(),
        )
        logger.info("Latest synthetic close: %.4f", float(synthetic_df["close"].iloc[-1]))


if __name__ == "__main__":
    main()
