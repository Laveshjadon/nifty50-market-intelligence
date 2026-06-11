"""Provide shared PostgreSQL access for the project."""

import logging
import os
import pandas as pd
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import OperationalError, ProgrammingError
from contextlib import contextmanager
from src.settings import get_settings

logger = logging.getLogger(__name__)


INDEX_TABLE = os.getenv('INDEX_TABLE', 'nifty50_index_5m')


class DatabaseManager:

    def __init__(self):
        settings = get_settings()

        if settings.database_uri:
            connection_url = settings.database_uri
            log_target = "DATABASE_URI"
        else:
            missing = [
                name
                for name, value in {
                    "DB_USER": settings.db_user,
                    "DB_PASSWORD": settings.db_password,
                    "DB_HOST": settings.db_host,
                    "DB_NAME": settings.db_name,
                }.items()
                if not value
            ]
            if missing:
                raise ValueError(f"these are missing from .env: {', '.join(missing)}")

            connection_url = URL.create(
                drivername="postgresql",
                username=settings.db_user,
                password=settings.db_password,
                host=settings.db_host,
                port=settings.db_port,
                database=settings.db_name,
            )
            log_target = f"{settings.db_host}:{settings.db_port}/{settings.db_name}"


        self.engine = create_engine(
            connection_url,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_pre_ping=True,
        )
        logger.info("connected to %s", log_target)

    def _read_sql(self, query, params=None, conn=None):

        target = conn if conn is not None else self.engine
        return pd.read_sql_query(query, target, params=params)

    def _coerce_ohlcv(self, df, allow_zero_volume=False):

        if df.empty:
            return df

        cleaned = df.copy()
        for col in ('open', 'high', 'low', 'close', 'volume'):
            cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce').astype('float32')

        mask = (
            (cleaned['open']  > 0) &
            (cleaned['high']  > 0) &
            (cleaned['low']   > 0) &
            (cleaned['close'] > 0)
        )
        if not allow_zero_volume:
            mask &= cleaned['volume'] > 0

        dropped = (~mask).sum()
        if dropped:
            logger.debug("dropped %d bad rows", dropped)

        return cleaned[mask].copy()

    def _clean_index_like_frame(self, df):

        return self._coerce_ohlcv(df, allow_zero_volume=True)

    def _ensure_index_table_exists(self, conn=None):
        ddl = text(f"""
            CREATE TABLE IF NOT EXISTS {INDEX_TABLE} (
                index_name VARCHAR(50) NOT NULL,
                bucket_time TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                open NUMERIC(15, 2) NOT NULL,
                high NUMERIC(15, 2) NOT NULL,
                low NUMERIC(15, 2) NOT NULL,
                close NUMERIC(15, 2) NOT NULL,
                volume NUMERIC(18, 0) NOT NULL,
                PRIMARY KEY (index_name, bucket_time)
            )
        """)
        idx_name = text(
            f"CREATE INDEX IF NOT EXISTS {INDEX_TABLE}_name_idx ON {INDEX_TABLE}(index_name)"
        )
        idx_time = text(
            f"CREATE INDEX IF NOT EXISTS {INDEX_TABLE}_bucket_time_idx ON {INDEX_TABLE}(bucket_time)"
        )
        target = conn if conn is not None else self.engine
        with target.begin() if conn is None else _NullContext(conn) as active_conn:
            active_conn.execute(ddl)
            active_conn.execute(idx_name)
            active_conn.execute(idx_time)

    @contextmanager
    def advisory_lock(self, lock_id: int):
        """Hold a PostgreSQL advisory lock for one report job."""
        with self.engine.connect() as conn:
            try:

                query = text("SELECT pg_try_advisory_lock(:lock_id)")
                locked = conn.execute(query, {"lock_id": lock_id}).scalar()

                if not locked:
                    logger.info("Advisory lock %s is already held by another process.", lock_id)
                    yield False
                else:
                    logger.info("Acquired advisory lock %s.", lock_id)
                    try:
                        yield True
                    finally:
                        conn.execute(text("SELECT pg_advisory_unlock(:lock_id)"), {"lock_id": lock_id})
                        logger.info("Released advisory lock %s.", lock_id)
            except Exception as e:
                logger.error("Error managing advisory lock %s: %s", lock_id, e)
                yield False

    def get_all_stocks(self, limit=None):

        if limit is not None:
            if not isinstance(limit, int) or limit <= 0:
                raise ValueError(f"limit must be a positive int, got {limit!r}")
            query = text("SELECT * FROM min_5 ORDER BY ticker, bucket_time ASC LIMIT :limit")
            df = self._read_sql(query, params={"limit": limit})
            logger.warning(
                "get_all_stocks called with limit=%d — returned data may be incomplete", limit
            )
        else:
            df = self._read_sql("SELECT * FROM min_5 ORDER BY ticker, bucket_time ASC")

        logger.info("got %d rows (limit=%s)", len(df), limit)
        return self._coerce_ohlcv(df)

    def get_stocks_data(self, tickers):

        if not tickers:
            logger.warning("tickers list is empty")
            return pd.DataFrame(columns=['ticker', 'bucket_time', 'open', 'high', 'low', 'close', 'volume'])

        query = text("""
            SELECT * FROM min_5
            WHERE ticker IN :tickers
            ORDER BY ticker, bucket_time ASC
        """)

        query = query.bindparams(bindparam("tickers", expanding=True))

        df = self._read_sql(query, params={"tickers": list(tickers)})
        logger.info("got %d rows for %d tickers", len(df), len(tickers))
        return self._coerce_ohlcv(df)

    def get_recent_stocks_data(self, tickers, bars_per_ticker):
        if not tickers:
            logger.warning("tickers list is empty")
            return pd.DataFrame(columns=['ticker', 'bucket_time', 'open', 'high', 'low', 'close', 'volume'])
        if not isinstance(bars_per_ticker, int) or bars_per_ticker <= 0:
            raise ValueError(f"bars_per_ticker must be a positive int, got {bars_per_ticker!r}")

        query = text("""
            SELECT ticker, bucket_time, open, high, low, close, volume
            FROM (
                SELECT
                    ticker,
                    bucket_time,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    ROW_NUMBER() OVER (
                        PARTITION BY ticker
                        ORDER BY bucket_time DESC
                    ) AS rn
                FROM min_5
                WHERE ticker IN :tickers
            ) ranked
            WHERE rn <= :bars_per_ticker
            ORDER BY ticker, bucket_time ASC
        """)

        query = query.bindparams(bindparam("tickers", expanding=True))

        df = self._read_sql(
            query,
            params={"tickers": list(tickers), "bars_per_ticker": bars_per_ticker},
        )
        logger.info(
            "got %d recent rows for %d tickers (bars_per_ticker=%d)",
            len(df),
            len(tickers),
            bars_per_ticker,
        )
        return self._coerce_ohlcv(df)

    def get_stock_data(self, ticker):
        query = text("SELECT * FROM min_5 WHERE ticker = :ticker ORDER BY bucket_time ASC")
        df = self._read_sql(query, params={"ticker": ticker})
        logger.info("got %d rows for %s", len(df), ticker)
        return self._coerce_ohlcv(df)

    def get_index_data(self, index_name="Nifty 50"):

        query = text(f"""
            SELECT * FROM {INDEX_TABLE}
            WHERE index_name = :index_name
            ORDER BY bucket_time ASC
        """)

        try:
            df = pd.read_sql_query(query, self.engine, params={"index_name": index_name})
            if not df.empty:
                logger.info("found %d rows in %s", len(df), INDEX_TABLE)
                return self._clean_index_like_frame(df)
        except ProgrammingError:

            logger.warning("%s not found, trying min_5 fallback", INDEX_TABLE)

        for candidate in [index_name, index_name.upper(), "NIFTY 50", "NIFTY50", "^NSEI"]:
            df = self.get_stock_like_series(candidate)
            if not df.empty:
                logger.info("fallback matched '%s' in min_5 (%d rows)", candidate, len(df))
                return self._clean_index_like_frame(df)

        logger.error("no data found for '%s' anywhere", index_name)
        return pd.DataFrame(columns=["bucket_time", "open", "high", "low", "close", "volume"])

    def get_stock_like_series(self, ticker):

        query = text("SELECT * FROM min_5 WHERE ticker = :ticker ORDER BY bucket_time ASC")
        return self._read_sql(query, params={"ticker": ticker})

    def replace_index_data(self, index_name, df):

        required_cols = ['bucket_time', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"df is missing columns: {', '.join(missing_cols)}")

        payload = df[required_cols].copy()
        if payload.empty:
            logger.warning("nothing to write for '%s'", index_name)
            return 0

        payload['bucket_time'] = pd.to_datetime(payload['bucket_time'])
        payload['index_name']  = index_name

        start_time = payload['bucket_time'].min().to_pydatetime()
        end_time   = payload['bucket_time'].max().to_pydatetime()

        delete_query = text(f"""
            DELETE FROM {INDEX_TABLE}
            WHERE index_name = :index_name
            AND bucket_time >= :start_time
            AND bucket_time <= :end_time
        """)

        try:
            with self.engine.begin() as conn:
                self._ensure_index_table_exists(conn=conn)
                result = conn.execute(delete_query, {
                    "index_name": index_name,
                    "start_time": start_time,
                    "end_time":   end_time,
                })
                payload.to_sql(INDEX_TABLE, conn, if_exists='append', index=False, method='multi')

            logger.info("deleted %d, wrote %d rows for '%s'", result.rowcount, len(payload), index_name)
            return len(payload)

        except OperationalError as exc:
            logger.exception("replace_index_data failed: %s", exc)
            raise




_db_manager = None


class _NullContext:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, exc_type, exc, tb):
        return False

def get_db() -> DatabaseManager:
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
