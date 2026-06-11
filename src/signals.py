"""
signals.py - market signals engine for the Nifty 50 universe

What it does:
- loads recent 5-minute OHLCV data for all configured tickers
- computes per-stock momentum, volume spike, and high/low distance signals
- ranks strongest/weakest stocks
- builds sector strength rankings from the configured sector map
- saves a single JSON snapshot for downstream report use
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.database_manager import get_db
from src.feature_engineering import CONFIG_PATH, PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bucket_time"] = pd.to_datetime(out["bucket_time"])
    if out["bucket_time"].dt.tz is not None:
        out["bucket_time"] = out["bucket_time"].dt.tz_localize(None)
    return out


def load_recent_market_data(config: dict[str, Any]) -> pd.DataFrame:
    tickers = config["tickers"]
    bars = int(config["signals"]["recent_bars"])
    df = get_db().get_recent_stocks_data(tickers, bars_per_ticker=bars)
    if df.empty:
        raise ValueError("No recent stock data found for signal generation.")
    return _normalize_time(df)


def _safe_return(close: pd.Series, periods: int) -> float:
    if len(close) <= periods:
        return float("nan")
    prev = float(close.iloc[-(periods + 1)])
    curr = float(close.iloc[-1])
    if prev <= 0 or curr <= 0:
        return float("nan")
    return float(np.log(curr / prev))


def build_stock_signal_frame(config: dict[str, Any], market_df: pd.DataFrame) -> pd.DataFrame:
    sig_cfg = config["signals"]
    windows = sig_cfg["momentum_windows"]
    vol_window = int(sig_cfg["volume_spike_window"])
    high_low_window = int(sig_cfg["high_low_window"])

    rows: list[dict[str, Any]] = []

    for ticker, frame in market_df.groupby("ticker", sort=True):
        frame = frame.sort_values("bucket_time").drop_duplicates("bucket_time", keep="last")
        if len(frame) < max(int(windows["long"]) + 1, vol_window, high_low_window):
            logger.warning("Skipping %s: not enough rows for full signal windows.", ticker)
            continue

        close = frame["close"].astype("float64")
        high = frame["high"].astype("float64")
        low = frame["low"].astype("float64")
        volume = frame["volume"].astype("float64")

        last_close = float(close.iloc[-1])
        last_volume = float(volume.iloc[-1])
        avg_volume = float(volume.tail(vol_window).mean())
        rolling_high = float(high.tail(high_low_window).max())
        rolling_low = float(low.tail(high_low_window).min())

        rows.append(
            {
                "ticker": ticker,
                "latest_timestamp": str(frame["bucket_time"].iloc[-1]),
                "last_close": last_close,
                "momentum_short": _safe_return(close, int(windows["short"])),
                "momentum_medium": _safe_return(close, int(windows["medium"])),
                "momentum_long": _safe_return(close, int(windows["long"])),
                "volume_spike": float(last_volume / avg_volume) if avg_volume > 0 else float("nan"),
                "distance_from_high": float((last_close / rolling_high) - 1.0) if rolling_high > 0 else float("nan"),
                "distance_from_low": float((last_close / rolling_low) - 1.0) if rolling_low > 0 else float("nan"),
            }
        )

    signals = pd.DataFrame(rows)
    if signals.empty:
        raise ValueError("No per-stock signals could be computed from the recent market data.")

    for col in ["momentum_short", "momentum_medium", "momentum_long", "volume_spike", "distance_from_high", "distance_from_low"]:
        signals[col] = pd.to_numeric(signals[col], errors="coerce")

    signals["composite_score"] = (
        0.20 * signals["momentum_short"].fillna(0.0)
        + 0.35 * signals["momentum_medium"].fillna(0.0)
        + 0.35 * signals["momentum_long"].fillna(0.0)
        + 0.10 * np.log1p(signals["volume_spike"].clip(lower=0).fillna(0.0))
    )
    return signals.sort_values("composite_score", ascending=False).reset_index(drop=True)


def build_sector_rankings(config: dict[str, Any], stock_signals: pd.DataFrame) -> list[dict[str, Any]]:
    sector_map = config["signals"]["sector_map"]
    signal_lookup = stock_signals.set_index("ticker")
    rankings: list[dict[str, Any]] = []

    for sector, tickers in sector_map.items():
        available = [ticker for ticker in tickers if ticker in signal_lookup.index]
        if not available:
            continue

        sector_frame = signal_lookup.loc[available]
        rankings.append(
            {
                "sector": sector,
                "n_stocks": int(len(available)),
                "avg_composite_score": float(sector_frame["composite_score"].mean()),
                "avg_momentum_medium": float(sector_frame["momentum_medium"].mean()),
                "avg_volume_spike": float(sector_frame["volume_spike"].mean()),
                "leaders": sector_frame["composite_score"].sort_values(ascending=False).head(3).index.tolist(),
            }
        )

    return sorted(rankings, key=lambda item: item["avg_composite_score"], reverse=True)


def _clean_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    return value


def _frame_to_records(
    df: pd.DataFrame,
    columns: list[str],
    top_n: int,
    *,
    sort_by: str,
    ascending: bool = False,
) -> list[dict[str, Any]]:
    subset = df.sort_values(sort_by, ascending=ascending).head(top_n)
    records = subset[columns].to_dict(orient="records")
    return records


def build_signal_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    market_df = load_recent_market_data(config)
    stock_signals = build_stock_signal_frame(config, market_df)
    top_n = int(config["signals"]["top_n"])

    snapshot = {
        "generated_at": str(pd.Timestamp.utcnow()),
        "latest_market_timestamp": str(stock_signals["latest_timestamp"].max()),
        "top_strongest_stocks": _frame_to_records(
            stock_signals,
            ["ticker", "composite_score", "momentum_short", "momentum_medium", "momentum_long", "volume_spike"],
            top_n=top_n,
            sort_by="composite_score",
            ascending=False,
        ),
        "top_weakest_stocks": _frame_to_records(
            stock_signals,
            ["ticker", "composite_score", "momentum_short", "momentum_medium", "momentum_long", "volume_spike"],
            top_n=top_n,
            sort_by="composite_score",
            ascending=True,
        ),
        "sector_strength_ranking": build_sector_rankings(config, stock_signals),
        "closest_to_highs": _frame_to_records(
            stock_signals,
            ["ticker", "distance_from_high", "momentum_medium", "volume_spike"],
            top_n=top_n,
            sort_by="distance_from_high",
            ascending=False,
        ),
        "closest_to_lows": _frame_to_records(
            stock_signals.assign(distance_to_low=-stock_signals["distance_from_low"]),
            ["ticker", "distance_to_low", "momentum_medium", "volume_spike"],
            top_n=top_n,
            sort_by="distance_to_low",
            ascending=False,
        ),
        "top_volume_spikes": _frame_to_records(
            stock_signals,
            ["ticker", "volume_spike", "momentum_short", "momentum_medium"],
            top_n=top_n,
            sort_by="volume_spike",
            ascending=False,
        ),
        "momentum_leaders": _frame_to_records(
            stock_signals,
            ["ticker", "momentum_long", "momentum_medium", "composite_score"],
            top_n=top_n,
            sort_by="momentum_long",
            ascending=False,
        ),
    }
    return snapshot


def save_signal_snapshot(snapshot: dict[str, Any], config: dict[str, Any]) -> Path:
    output_path = PROJECT_ROOT / config["signals"]["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_clean_json(snapshot), f, indent=2, allow_nan=False)
    return output_path


def main() -> None:
    config = load_config()
    snapshot = build_signal_snapshot(config)
    output_path = save_signal_snapshot(snapshot, config)

    logger.info("Saved signal snapshot to %s", output_path)
    if snapshot["top_strongest_stocks"]:
        strongest = snapshot["top_strongest_stocks"][0]
        weakest = snapshot["top_weakest_stocks"][0]
        logger.info(
            "Signal leaders: strongest=%s weakest=%s",
            strongest["ticker"],
            weakest["ticker"],
        )


if __name__ == "__main__":
    main()
