"""
generate_7day_report.py 
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import yaml

from src.modeling import build_or_load_dataset, load_config as load_model_config, run_latest_inference, save_outputs, train_and_evaluate
from src.news_processor import process_recent_news, save_news_outputs
from src.report_agent import build_report_from_outputs
from src.signals import build_signal_snapshot, save_signal_snapshot
from src.synthetic_index import build_and_save_synthetic_index

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(argv: Sequence[str] | None = None) -> None:
    """Generate the report, parsing CLI args only when argv is explicitly provided."""
    parser = argparse.ArgumentParser(description="Generate Nifty 50 7-day intelligence report.")
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Force rebuild the modeling dataset from the database (includes news features).",
    )
    args = parser.parse_args(list(argv) if argv is not None else [])

    config = load_config()
    logger.info("Step 1/5 - Updating synthetic index...")
    synthetic_df, inserted = build_and_save_synthetic_index(CONFIG_PATH)
    logger.info("Synthetic index rows built=%d inserted=%d", len(synthetic_df), inserted)

    logger.info("Step 2/5 - Generating signal snapshot...")
    signal_snapshot = build_signal_snapshot(config)
    signal_path = save_signal_snapshot(signal_snapshot, config)
    logger.info("Saved signal snapshot to %s", signal_path)

    logger.info("Step 3/5 - Training model and running latest inference...")
    model_config = load_model_config(CONFIG_PATH)
    dataset, dataset_summary = build_or_load_dataset(model_config, force_rebuild=args.rebuild_dataset)
    model, metrics, _ = train_and_evaluate(model_config, dataset)
    if dataset_summary is not None:
        metrics["dataset_summary"] = dataset_summary
    latest_prediction = run_latest_inference(model, model_config, dataset)
    _, metadata_path, metrics_path, latest_path = save_outputs(
        model,
        metrics,
        latest_prediction,
        model_config,
        dataset,
    )
    logger.info(
        "Saved model outputs to %s, %s and %s",
        metadata_path,
        metrics_path,
        latest_path,
    )

    logger.info("Step 4/5 - Processing recent news...")
    scored_df, news_summary = process_recent_news(config)
    news_csv_path, news_summary_path = save_news_outputs(scored_df, news_summary, config)
    logger.info("Saved news outputs to %s and %s", news_csv_path, news_summary_path)

    logger.info("Step 5/5 - Rendering markdown report...")
    _, report_path = build_report_from_outputs(config)
    logger.info("Saved final report to %s", report_path)


if __name__ == "__main__":
    main(sys.argv[1:])
