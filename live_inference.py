"""
live_inference.py - load trained LSTM weights and score the latest live dataset
"""

from __future__ import annotations

import argparse
import json
import logging

from src.modeling import (
    CONFIG_PATH,
    PROJECT_ROOT,
    load_config,
    run_live_inference,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run live inference using saved LSTM weights and the latest market dataset."
    )
    parser.add_argument(
        "--use-saved-dataset",
        action="store_true",
        help="Use the last saved dataset CSV instead of rebuilding the dataset from the database.",
    )
    args = parser.parse_args()

    config = load_config(CONFIG_PATH)
    prediction, dataset = run_live_inference(
        config,
        rebuild_dataset=not args.use_saved_dataset,
    )

    output_path = PROJECT_ROOT / config["modeling"]["output_latest_prediction_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prediction, f, indent=2)

    logger.info("Live inference used %d rows from dataset.", len(dataset))
    logger.info("Saved live prediction to %s", output_path)
    logger.info(
        "Live inference: ts=%s predicted_next_5m_return=%.6f direction=%s implied_next_basket_close=%.4f",
        prediction["latest_timestamp"],
        prediction["predicted_next_5m_return"],
        prediction["predicted_direction"],
        prediction["implied_next_basket_close"],
    )


if __name__ == "__main__":
    main()
