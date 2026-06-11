"""Load historical Nifty 50 CSV data into PostgreSQL."""

import csv
import glob
import os
import sys
import logging
from io import StringIO

import psycopg2
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.settings import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()

DB_CONFIG = {
    "host": settings.db_host,
    "database": settings.db_name,
    "user": settings.db_user,
    "password": settings.db_password
}

DATA_FOLDER = settings.data_folder
ARCHIVE_5M_GLOB = os.path.join('archive', '*_5minute.csv')

with open('config.yaml', 'r') as f:
    _cfg = yaml.safe_load(f)
NIFTY_50_TICKERS = _cfg['tickers']
INDEX_NAME = _cfg['index']['name']
TABLES = _cfg['database']['tables']


def ingest_raw_minute_csv(cur, file_path, ticker):
    """Load one raw 1-minute stock file."""
    with open(file_path, 'r') as f:
        next(f)

        mem_file = StringIO()
        for line in f:
            cleaned_line = line.replace('.0\n', '\n').replace('.0,', ',')
            mem_file.write(f"{ticker},{cleaned_line}")

        mem_file.seek(0)
        cur.copy_from(
            mem_file, TABLES['raw_ticks'], sep=',',
            columns=('ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume')
        )


def ingest_clean_5m_stock_csv(cur, file_path, ticker):
    """Load one cleaned 5-minute stock file."""
    with open(file_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cur.execute(f"""
                INSERT INTO {TABLES['min_5']} (ticker, bucket_time, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                ticker,
                row['date'],
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                row['volume'],
            ))


def ingest_clean_5m_index_csv(cur, file_path):
    """Load the real Nifty 50 index file."""
    with open(file_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cur.execute(f"""
                INSERT INTO {TABLES['index_min_5']} (index_name, bucket_time, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                INDEX_NAME,
                row['date'],
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                row['volume'],
            ))


def ingest_data():
    """Find configured CSV files and load each supported format."""
    conn = None
    cur = None
    try:
        required_env = {
            'DB_HOST': DB_CONFIG['host'],
            'DB_NAME': DB_CONFIG['database'],
            'DB_USER': DB_CONFIG['user'],
            'DB_PASSWORD': DB_CONFIG['password'],
            'DATA_FOLDER': DATA_FOLDER,
        }
        missing_env = [name for name, value in required_env.items() if not value]
        if missing_env:
            raise EnvironmentError(
                f"Missing required environment variable(s): {', '.join(missing_env)}"
            )

        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        logger.info("Successfully connected to 'nifty 50' database!")

        if not DATA_FOLDER:
             logger.warning("DATA_FOLDER is empty or not configured. Skipping 1-minute files scan.")
             files = []
        else:
             files = glob.glob(DATA_FOLDER)

        logger.info(f"Scanning {len(files)} files for Nifty 50 constituents...")

        nifty_count = 0
        for file_path in files:
            file_name = os.path.basename(file_path)
            ticker = file_name.split('_')[0].upper()

            if ticker in NIFTY_50_TICKERS:
                nifty_count += 1
                ingest_raw_minute_csv(cur, file_path, ticker)
                logger.info(f"[{nifty_count}/50] SUCCESS: Loaded {ticker}")

        clean_5m_files = sorted(glob.glob(ARCHIVE_5M_GLOB))
        if clean_5m_files:
            logger.info(f"Found {len(clean_5m_files)} clean 5-minute csv file(s)...")

        for file_path in clean_5m_files:
            file_name = os.path.basename(file_path)
            ticker = file_name.replace('_5minute.csv', '')

            if ticker == 'NIFTY 50':
                ingest_clean_5m_index_csv(cur, file_path)
                logger.info(f"INDEX SUCCESS: Loaded {INDEX_NAME}")
                continue

            if ticker in NIFTY_50_TICKERS:
                ingest_clean_5m_stock_csv(cur, file_path, ticker)
                logger.info(f"5M SUCCESS: Loaded {ticker}")

        conn.commit()
        logger.info(f"DONE! Loaded {nifty_count} raw minute stock file(s).")

    except Exception as e:
        if conn is not None:
            conn.rollback()
        logger.exception("An error occurred during data ingestion")
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    ingest_data()
