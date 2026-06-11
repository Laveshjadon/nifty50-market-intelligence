import os
import sys
import time
import shutil
import logging
from pathlib import Path
import json
import yaml
import requests
import psycopg2
from sqlalchemy import text
from datetime import datetime


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.settings import get_settings
from src.database_manager import get_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
settings = get_settings()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def fetch_nse_nifty50():
    """Fetch the current Nifty 50 constituents and weights."""
    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    }

    session = requests.Session()
    session.headers.update(headers)

    try:
        logger.info("Fetching cookies from NSE homepage...")
        session.get("https://www.nseindia.com", timeout=15)
        time.sleep(2)

        logger.info(f"Querying NSE API: {url}")
        response = session.get(url, timeout=15)

        if response.status_code == 403:
            logger.error("NSE API returned 403 Forbidden. The request was blocked by anti-bot protections.")
            return None

        response.raise_for_status()
        data = response.json()

        items = data.get("data", [])
        if not items:
            logger.error("NSE API returned empty data array.")
            return None

        results = {}
        total_ffmc = 0.0

        for item in items:
            symbol = item.get("symbol")
            if not symbol or symbol == "NIFTY 50":
                continue


            clean_symbol = str(symbol).strip().upper().replace("&", "").replace("-", "")
            if clean_symbol == "MM":
                pass


            weight = item.get("weight")
            ffmc = item.get("ffmc", 0.0)

            if weight is not None:
                results[symbol] = float(weight)
            elif ffmc > 0:
                results[symbol] = float(ffmc)
                total_ffmc += float(ffmc)
            else:

                results[symbol] = 0.0


        if total_ffmc > 0 and all(v > 1.0 for v in results.values()):
            results = {k: v / total_ffmc for k, v in results.items()}
        elif sum(results.values()) > 1.5:
            results = {k: v / 100.0 for k, v in results.items()}

        if len(results) < 45:
            logger.error(f"Extracted only {len(results)} constituents. This seems incorrect.")
            return None

        return results

    except requests.RequestException as e:
        logger.exception(f"Network error while fetching from NSE: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error parsing NSE data: {e}")
        return None


def fetch_yfinance_fallback():
    """Explain how to recover when the NSE request fails."""
    logger.critical("NSE API failed or is blocked.")
    logger.critical("WARNING: yfinance does not reliably provide exact index constituent weights for ^NSEI.")
    logger.critical("MANUAL UPDATE REQUIRED: Please manually download the latest 'Nifty 50 Weightage' CSV from NSE website and update config.yaml.")
    return None


def update_database(weights_dict: dict):
    """Store the latest constituent weights in PostgreSQL."""
    db = get_db()

    create_table_sql = text("""
        CREATE TABLE IF NOT EXISTS nifty50_constituents (
            ticker VARCHAR(50) PRIMARY KEY,
            weight FLOAT NOT NULL,
            last_updated TIMESTAMP WITHOUT TIME ZONE NOT NULL
        )
    """)

    upsert_sql = text("""
        INSERT INTO nifty50_constituents (ticker, weight, last_updated)
        VALUES (:ticker, :weight, :last_updated)
        ON CONFLICT (ticker) DO UPDATE
        SET weight = EXCLUDED.weight, last_updated = EXCLUDED.last_updated
    """)

    now = datetime.now()
    try:
        with db.engine.begin() as conn:
            conn.execute(create_table_sql)

            for ticker, weight in weights_dict.items():
                conn.execute(upsert_sql, {
                    "ticker": ticker,
                    "weight": weight,
                    "last_updated": now
                })
        logger.info(f"Successfully updated database table 'nifty50_constituents' with {len(weights_dict)} records.")
        return True
    except Exception as e:
        logger.exception(f"Failed to update database: {e}")
        return False


def update_config_yaml(weights_dict: dict):
    """Back up and update ticker weights in config.yaml."""
    if not CONFIG_PATH.exists():
        logger.error(f"config.yaml not found at {CONFIG_PATH}")
        return False


    backup_path = CONFIG_PATH.with_suffix(f".yaml.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(CONFIG_PATH, backup_path)
    logger.info(f"Created config backup at {backup_path}")

    try:
        with open(CONFIG_PATH, 'r') as f:
            config_data = yaml.safe_load(f)

        sorted_tickers = sorted(list(weights_dict.keys()))
        config_data['tickers'] = sorted_tickers

        if 'synthetic_index' not in config_data:
            config_data['synthetic_index'] = {}
        if 'weights' not in config_data['synthetic_index']:
            config_data['synthetic_index']['weights'] = {}

        config_data['synthetic_index']['weights'] = {
            t: round(w, 4) for t, w in weights_dict.items()
        }


        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Successfully updated {CONFIG_PATH} with latest constituents and weights.")
        return True
    except Exception as e:
        logger.exception(f"Failed to update config.yaml: {e}")

        shutil.copy2(backup_path, CONFIG_PATH)
        logger.info("Rolled back config.yaml from backup.")
        return False


def main():
    logger.info("Starting Nifty 50 weights update process...")

    weights = fetch_nse_nifty50()

    if not weights:
        weights = fetch_yfinance_fallback()

    if not weights:
        logger.error("Failed to retrieve constituents. Exiting with non-zero status.")
        sys.exit(1)

    logger.info(f"Successfully retrieved {len(weights)} constituents.")


    total_weight = sum(weights.values())
    if not (0.99 <= total_weight <= 1.01):
        logger.warning(f"Weights sum to {total_weight}, which is not 1.0. Re-normalizing...")
        weights = {k: v / total_weight for k, v in weights.items()}

    db_success = update_database(weights)
    cfg_success = update_config_yaml(weights)

    if not db_success or not cfg_success:
        logger.error("Update completed with errors in DB or Config.")
        sys.exit(1)

    logger.info("Nifty 50 weights update completed successfully.")


if __name__ == "__main__":
    main()
