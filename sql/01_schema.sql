-- 1. Raw Data Table (1-minute ticks)
CREATE TABLE nifty_ticks (
    ticker VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    open NUMERIC(15, 2) NOT NULL,
    high NUMERIC(15, 2) NOT NULL,
    low NUMERIC(15, 2) NOT NULL,
    close NUMERIC(15, 2) NOT NULL,
    volume NUMERIC(18, 0) NOT NULL,
    PRIMARY KEY (ticker, timestamp)
);

CREATE INDEX nifty_ticks_ticker_idx ON nifty_ticks(ticker);
CREATE INDEX nifty_ticks_timestamp_idx ON nifty_ticks(timestamp);

-- 2.  Data Table (5-minute resampled)
CREATE TABLE min_5 (
    ticker VARCHAR(20) NOT NULL,
    bucket_time TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    open NUMERIC(15, 2) NOT NULL,
    high NUMERIC(15, 2) NOT NULL,
    low NUMERIC(15, 2) NOT NULL,
    close NUMERIC(15, 2) NOT NULL,
    volume NUMERIC(18, 0) NOT NULL,
    PRIMARY KEY (ticker, bucket_time)
);

CREATE INDEX min_5_ticker_idx ON min_5(ticker);
CREATE INDEX min_5_bucket_time_idx ON min_5(bucket_time);

-- 3. Nifty 50 index 5-minute data
CREATE TABLE nifty50_index_5m (
    index_name VARCHAR(50) NOT NULL,
    bucket_time TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    open NUMERIC(15, 2) NOT NULL,
    high NUMERIC(15, 2) NOT NULL,
    low NUMERIC(15, 2) NOT NULL,
    close NUMERIC(15, 2) NOT NULL,
    volume NUMERIC(18, 0) NOT NULL,
    PRIMARY KEY (index_name, bucket_time)
);

CREATE INDEX nifty50_index_5m_name_idx ON nifty50_index_5m(index_name);
CREATE INDEX nifty50_index_5m_bucket_time_idx ON nifty50_index_5m(bucket_time);

-- Note: nifty_ticks is kept as a logged (durable) table to preserve tick data across crashes.
-- If unlogged performance is needed, ensure an automated reload/reconciliation process is in place.






-- In 01_schema.sql
CREATE TABLE IF NOT EXISTS refined_news (
    news_id         SERIAL PRIMARY KEY,   -- optional auto‑increment ID
    date            DATE NOT NULL,
    title           TEXT,
    url             TEXT,
    source_file     TEXT,
    categories      TEXT,
    relevance_score INTEGER,
    has_negation    BOOLEAN,
    impact_tier     TEXT
    -- sentiment column can be added later
);

CREATE INDEX IF NOT EXISTS idx_refined_news_date ON refined_news (date);
