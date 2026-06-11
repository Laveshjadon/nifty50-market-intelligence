import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock

from src import backfill

@pytest.fixture
def mock_upstox_response():
    """Return a small Upstox candle response for tests."""
    return {
        "status": "success",
        "data": {
            "candles": [
                ["2026-06-01T09:15:00+05:30", 100.0, 101.0, 99.0, 100.5, 5000, 0],
                ["2026-06-01T09:20:00+05:30", 100.5, 102.0, 100.0, 101.5, 6000, 0]
            ]
        }
    }

@patch("src.backfill.requests.get")
def test_successful_backfill(mock_get, mock_upstox_response):
    """Check that valid API candles become a clean DataFrame."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_upstox_response
    mock_get.return_value = mock_response

    start_dt = datetime(2026, 6, 1, 9, 15)
    end_dt = datetime(2026, 6, 1, 15, 30)

    df = backfill.fetch_chunk("RELIANCE", start=start_dt, end=end_dt)

    assert not df.empty, "DataFrame should not be empty on successful backfill"
    assert set(df.columns) == {"bucket_time", "ticker", "open", "high", "low", "close", "volume"}
    assert df.shape == (2, 7), f"Expected shape (2, 7), got {df.shape}"
    assert df.iloc[0]["open"] == 100.5
    assert df.iloc[1]["volume"] == 5000

@patch("src.backfill.requests.get")
@patch("src.backfill.time.sleep")
def test_rate_limit_handling(mock_sleep, mock_get, mock_upstox_response):
    """Check that an API rate-limit response fails safely."""
    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.text = "Too Many Requests"

    resp_success = MagicMock()
    resp_success.status_code = 200
    resp_success.json.return_value = mock_upstox_response

    mock_get.return_value = resp_429

    df = backfill.fetch_chunk("RELIANCE", datetime(2026, 6, 1), datetime(2026, 6, 2))

    assert df.empty, "Should return empty dataframe on unrecoverable 429"

@patch("src.backfill.requests.get")
def test_empty_response(mock_get):
    """Check that an empty API response returns no candles."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "success", "data": {"candles": []}}
    mock_get.return_value = mock_response

    df = backfill.fetch_chunk("RELIANCE", datetime(2026, 6, 1), datetime(2026, 6, 2))

    assert isinstance(df, pd.DataFrame), "Should return a pandas DataFrame"
    assert df.empty, "DataFrame should be empty when no candles are returned"

@patch("src.backfill.fetch_chunk")
@patch("src.backfill.get_last_date")
@patch("src.backfill.delete_overlap")
@patch("src.backfill.save_to_db")
@patch("src.backfill.time.sleep")
def test_chunking_logic(mock_sleep, mock_save, mock_delete, mock_last_date, mock_fetch):
    """Check that a long date range is split into API-sized chunks."""
    mock_last_date.return_value = datetime.now() - pd.Timedelta(days=30)

    dummy_df = pd.DataFrame([{"bucket_time": datetime.now(), "ticker": "RELIANCE", "close": 100}])
    mock_fetch.return_value = dummy_df

    original_chunk_days = backfill.CHUNK_DAYS
    backfill.CHUNK_DAYS = 10

    try:
        backfill.backfill_ticker("RELIANCE")


        assert mock_fetch.call_count >= 3, f"Expected at least 3 chunk fetches, got {mock_fetch.call_count}"
        assert mock_save.call_count == mock_fetch.call_count, "Should save to DB for each successful chunk"
        assert mock_sleep.called, "Should sleep between chunk requests"
    finally:
        backfill.CHUNK_DAYS = original_chunk_days
