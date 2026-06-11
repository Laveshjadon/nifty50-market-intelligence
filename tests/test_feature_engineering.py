from __future__ import annotations

import pandas as pd

from src.feature_engineering import (
    assign_news_to_next_candle_start,
    validate_news_lookahead_guard,
)


def test_news_timestamps_shift_to_next_candle_without_lookahead() -> None:
    """Check that news is assigned only to later market candles."""
    news_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-05-06 09:15:00",
                    "2026-05-06 09:17:30",
                    "2026-05-06 09:19:59",
                ]
            ),
            "title": ["open", "mid-candle", "pre-next-candle"],
        }
    )

    news_frame["bucket_time"] = assign_news_to_next_candle_start(news_frame["date"])

    expected_bucket_times = pd.to_datetime(
        [
            "2026-05-06 09:20:00",
            "2026-05-06 09:20:00",
            "2026-05-06 09:20:00",
        ]
    )
    pd.testing.assert_series_equal(
        news_frame["bucket_time"],
        pd.Series(expected_bucket_times, name="bucket_time"),
    )

    validate_news_lookahead_guard(news_frame)
    assert (news_frame["date"] < news_frame["bucket_time"]).all()
