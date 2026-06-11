import pytest
import pandas as pd
import numpy as np
from datetime import datetime

@pytest.fixture
def mock_config():
    """Return a small configuration shared by unit tests."""
    return {
        "signals": {
            "recent_bars": 100,
            "momentum_windows": {"short": 3, "medium": 12, "long": 78},
            "volume_spike_window": 20,
            "high_low_window": 50,
            "top_n": 5,
            "sector_map": {"tech": ["TICK1", "TICK2"]}
        },
        "lstm": {
            "sequence_length": 5,
            "training": {
                "hidden_size": 32,
                "num_layers": 1,
                "dropout": 0.0,
                "compressed_size": 16,
                "batch_size": 2,
                "epochs": 20,
                "learning_rate": 0.01,
            }
        },
        "modeling": {
            "target_column": "target",
            "validation_target_column": "val_target",
        }
    }

@pytest.fixture
def synthetic_market_data():
    """Create repeatable OHLCV data for five test tickers."""
    np.random.seed(42)
    dates = pd.date_range("2026-01-01 09:15:00", periods=100, freq="5min")
    tickers = ["TICK1", "TICK2", "TICK3", "TICK4", "TICK5"]

    rows = []
    for ticker in tickers:
        base_price = 100.0
        for dt in dates:
            open_p = base_price * (1 + np.random.normal(0, 0.001))
            close_p = open_p * (1 + np.random.normal(0, 0.001))
            high_p = max(open_p, close_p) * 1.001
            low_p = min(open_p, close_p) * 0.999
            vol = int(np.random.uniform(1000, 50000))

            rows.append({
                "bucket_time": dt,
                "ticker": ticker,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": vol
            })
            base_price = close_p

    df = pd.DataFrame(rows)

    df["log_ret"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)
    return df
