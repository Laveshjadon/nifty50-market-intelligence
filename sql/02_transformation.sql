INSERT INTO min_5 (ticker, bucket_time, open, high, low, close, volume)
WITH raw_buckets AS (
    SELECT 
        ticker,
        date_bin('5 minutes', timestamp, '2020-01-01') AS b_time,
        open, high, low, close, volume,
        FIRST_VALUE(open) OVER (
            PARTITION BY ticker, date_bin('5 minutes', timestamp, '2020-01-01') 
            ORDER BY timestamp ASC
        ) as first_p,
        
        LAST_VALUE(close) OVER (
            PARTITION BY ticker, date_bin('5 minutes', timestamp, '2020-01-01') 
            ORDER BY timestamp ASC 
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) as last_p
    FROM nifty_ticks
)

SELECT 
    ticker,
    b_time,
    MAX(first_p) as open,
    MAX(high) as high,
    MIN(low) as low,
    MAX(last_p) as close,
    SUM(volume) as volume
FROM raw_buckets
GROUP BY ticker, b_time
ORDER BY ticker, b_time;
