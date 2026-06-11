import pytest
import pandas as pd
import numpy as np

from src import signals

def test_signal_generation_happy_path(mock_config, synthetic_market_data):
    """Check the signal snapshot structure and value types."""
    df = signals.build_stock_signal_frame(mock_config, synthetic_market_data)

    assert not df.empty, "Signal dataframe should not be empty"
    assert "composite_score" in df.columns
    assert "momentum_short" in df.columns
    assert "volume_spike" in df.columns
    assert "distance_from_high" in df.columns


    assert df.shape[0] == 5, f"Expected 5 tickers in signal output, got {df.shape[0]}"

    assert pd.api.types.is_numeric_dtype(df["composite_score"]), "composite_score must be numeric"
    assert pd.api.types.is_numeric_dtype(df["volume_spike"]), "volume_spike must be numeric"

def test_edge_case_all_nans(mock_config, synthetic_market_data):
    """Check that missing recent values do not crash signal generation."""
    target_idx = synthetic_market_data["ticker"] == "TICK1"
    synthetic_market_data.loc[target_idx, ["close", "high", "low", "volume"]] = np.nan

    df = signals.build_stock_signal_frame(mock_config, synthetic_market_data)
    assert not df.empty, "DataFrame should process successfully even with NaNs"


def test_edge_case_zero_volume(mock_config, synthetic_market_data):
    """Check that zero volume does not cause division errors."""
    synthetic_market_data.loc[synthetic_market_data["ticker"] == "TICK1", "volume"] = 0.0

    df = signals.build_stock_signal_frame(mock_config, synthetic_market_data)

    tick1_row = df[df["ticker"] == "TICK1"].iloc[0]

    assert pd.isna(tick1_row["volume_spike"]), "Volume spike should be NaN when avg volume is exactly 0"

def test_threshold_logic(mock_config):
    """Check price and volume signals for a large final-bar spike."""
    dates = pd.date_range("2026-01-01", periods=80, freq="5min")
    rows = []

    for i, dt in enumerate(dates):
        close_p = 100.0 if i < 79 else 200.0
        vol = 1000 if i < 79 else 10000

        rows.append({
            "bucket_time": dt,
            "ticker": "SPIKE",
            "open": 100.0,
            "high": close_p,
            "low": 100.0,
            "close": close_p,
            "volume": vol
        })

    df = pd.DataFrame(rows)

    signals_df = signals.build_stock_signal_frame(mock_config, df)

    assert not signals_df.empty
    row = signals_df.iloc[0]


    assert row["momentum_short"] >= 0.6, f"Expected massive short momentum, got {row['momentum_short']}"


    assert row["volume_spike"] > 5.0, f"Expected massive volume spike, got {row['volume_spike']}"
