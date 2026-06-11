"""Build the aligned multi-stock dataset used by the model."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
logger = logging.getLogger(__name__)


def _get_db_manager():
    from src.database_manager import get_db

    return get_db()


def load_config(config_path: str | Path = CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_json(value):
    """Replace invalid float values before saving JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    return value


def _clean_stock_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["bucket_time"] = pd.to_datetime(cleaned["bucket_time"])
    cleaned = cleaned.sort_values("bucket_time").drop_duplicates("bucket_time", keep="last")

    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").astype("float32")

    cleaned["volume"] = pd.to_numeric(cleaned["volume"], errors="coerce")
    cleaned = cleaned.dropna(subset=["bucket_time", "open", "high", "low", "close", "volume"])
    cleaned["volume"] = cleaned["volume"].astype("int64")

    mask = (
        (cleaned["open"] > 0)
        & (cleaned["high"] > 0)
        & (cleaned["low"] > 0)
        & (cleaned["close"] > 0)
        & (cleaned["volume"] > 0)
    )
    cleaned = cleaned.loc[mask].copy()
    return cleaned.set_index("bucket_time")


def _clean_index_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["bucket_time"] = pd.to_datetime(cleaned["bucket_time"])
    cleaned = cleaned.sort_values("bucket_time").drop_duplicates("bucket_time", keep="last")

    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").astype("float32")

    cleaned["volume"] = pd.to_numeric(cleaned["volume"], errors="coerce").fillna(0).astype("int64")
    cleaned = cleaned.dropna(subset=["bucket_time", "open", "high", "low", "close"])

    mask = (
        (cleaned["open"] > 0)
        & (cleaned["high"] > 0)
        & (cleaned["low"] > 0)
        & (cleaned["close"] > 0)
    )
    cleaned = cleaned.loc[mask].copy()
    return cleaned.set_index("bucket_time")


def load_stock_history(tickers: list[str]) -> tuple[dict[str, pd.DataFrame], dict]:
    db_manager = _get_db_manager()
    stock_frames: dict[str, pd.DataFrame] = {}
    load_summary: dict[str, dict] = {
        ticker: {"rows": 0, "status": "empty"} for ticker in tickers
    }
    raw_all = db_manager.get_stocks_data(tickers)
    if raw_all.empty:
        return stock_frames, load_summary

    raw_all = raw_all.copy()
    raw_all["ticker"] = raw_all["ticker"].astype(str).str.strip().str.upper()

    for ticker, raw in raw_all.groupby("ticker", sort=False):
        if ticker not in load_summary:
            continue
        if raw.empty:
            continue

        frame = _clean_stock_frame(raw)
        if frame.empty:
            load_summary[ticker] = {"rows": 0, "status": "empty_after_cleaning"}
            continue

        stock_frames[ticker] = frame
        load_summary[ticker] = {
            "rows": int(len(frame)),
            "start": str(frame.index.min()),
            "end": str(frame.index.max()),
            "status": "loaded",
        }

    return stock_frames, load_summary


def load_index_history(index_name: str) -> pd.DataFrame:
    db_manager = _get_db_manager()
    index_df = db_manager.get_index_data(index_name=index_name)
    if index_df.empty:
        raise ValueError(
            f"Could not find Nifty index data in the database for '{index_name}'. "
            "Checked the dedicated index table and fallback names in min_5."
        )
    return _clean_index_frame(index_df)


def _normalize_weight_series(weights: pd.Series, tickers: list[str], source_name: str) -> pd.Series:
    missing_tickers = [ticker for ticker in tickers if ticker not in weights.index]
    extra_tickers = [ticker for ticker in weights.index if ticker not in tickers]

    if missing_tickers:
        raise ValueError(f"{source_name} is missing tickers: " + ", ".join(missing_tickers))
    if extra_tickers:
        raise ValueError(f"{source_name} has unknown tickers: " + ", ".join(extra_tickers))

    weights = weights.loc[tickers].astype("float64")
    if weights.isna().any():
        bad = weights[weights.isna()].index.tolist()
        raise ValueError(f"{source_name} has invalid weights for: " + ", ".join(bad))

    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        raise ValueError(f"{source_name} weight sum must be positive.")

    return weights / weight_sum


def load_weights(config: dict, tickers: list[str]) -> pd.Series:
    fe_cfg = config["feature_engineering"]
    yaml_weights = (
        fe_cfg.get("weights", {})
        or config.get("synthetic_index", {}).get("weights", {})
        or {}
    )

    if not yaml_weights:
        raise ValueError(
            "No weights found. Add feature_engineering.weights or synthetic_index.weights in config.yaml."
        )

    weights = pd.Series(
        {str(ticker).strip().upper(): value for ticker, value in yaml_weights.items()},
        dtype="float64",
    )
    return _normalize_weight_series(weights, tickers, source_name="YAML weights")


def build_stock_features(
    df: pd.DataFrame,
    volume_window: int,
    clip_lower: float,
    clip_upper: float,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    close = df["close"].replace(0, np.nan)
    open_ = df["open"].replace(0, np.nan)
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    rolling_volume = (
        df["volume"].rolling(volume_window, min_periods=1).mean().replace(0, np.nan)
    )

    out["log_ret"] = np.log(close / close.shift(1))
    out["log_ret_o"] = np.log(open_ / close.shift(1))
    out["hl_ratio"] = hl / close
    out["co_ratio"] = (df["close"] - df["open"]) / open_
    out["vol_ratio"] = df["volume"] / rolling_volume


    high_low = (df["high"] - df["low"]).replace(0, np.nan)
    open_p = df["open"].replace(0, np.nan)

    out["upper_wick_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / high_low
    out["lower_wick_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / high_low
    out["body_ratio"] = (df["close"] - df["open"]).abs() / high_low
    out["buying_pressure"] = (df["close"] - df["low"]) / high_low
    out["target_ret"] = np.log(close.shift(-1) / close)

    horizon = 6
    threshold = 0.001
    future_return = np.log(close.shift(-horizon) / close)
    out["target_binary"] = (future_return > threshold).astype(int)

    out = out.replace([np.inf, -np.inf], np.nan).dropna()

    for col in ["log_ret", "log_ret_o", "hl_ratio", "co_ratio", "vol_ratio"]:
        lo = out[col].quantile(clip_lower)
        hi = out[col].quantile(clip_upper)
        out[col] = out[col].clip(lo, hi)

    return out


def _make_wide_feature_block(
    feature_frames: dict[str, pd.DataFrame],
    base_cols: list[str],
    lags: int,
) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []

    for ticker, frame in feature_frames.items():
        cols: dict[str, pd.Series] = {}
        for feature in base_cols:
            cols[f"{ticker}__{feature}"] = frame[feature]
            for lag in range(1, lags + 1):
                cols[f"{ticker}__{feature}_lag{lag}"] = frame[feature].shift(lag)

        ticker_block = pd.DataFrame(cols, index=frame.index)
        blocks.append(ticker_block)

    wide = pd.concat(blocks, axis=1, join="inner").sort_index()
    return wide.dropna()


def _make_weighted_feature_block(
    feature_frames: dict[str, pd.DataFrame],
    weights: pd.Series,
    base_cols: list[str],
    lags: int,
) -> pd.DataFrame:
    weighted = pd.DataFrame(index=next(iter(feature_frames.values())).index)

    for feature in base_cols:
        current_series = []
        for ticker, frame in feature_frames.items():
            current_series.append(frame[feature] * float(weights.loc[ticker]))

        weighted[f"basket__{feature}"] = pd.concat(current_series, axis=1, join="inner").sum(axis=1)

    for feature in base_cols:
        for lag in range(1, lags + 1):
            weighted[f"basket__{feature}_lag{lag}"] = weighted[f"basket__{feature}"].shift(lag)

    return weighted.dropna().sort_index()


def build_index_validation_frame(
    index_df: pd.DataFrame,
    clip_lower: float,
    clip_upper: float,
) -> pd.DataFrame:
    validation = pd.DataFrame(index=index_df.index)
    validation["index_close"] = index_df["close"]
    validation["index_ret"] = np.log(index_df["close"] / index_df["close"].shift(1))
    validation["index_target_ret"] = np.log(index_df["close"].shift(-1) / index_df["close"])
    validation = validation.replace([np.inf, -np.inf], np.nan).dropna()

    return validation


def assign_news_to_next_candle_start(news_dates: pd.Series) -> pd.Series:
    """
    Assign each news date to the next trading day open (09:15 AM IST).
    News published on any day becomes effective the following trading day.
    This prevents same-day look-ahead bias.
    """

    next_day = news_dates.dt.normalize() + pd.Timedelta(days=1)
    next_open = next_day + pd.Timedelta(hours=9, minutes=15)
    return next_open


def validate_news_lookahead_guard(df_news: pd.DataFrame) -> None:
    """Reject news assignments that could leak future information."""
    if "date" not in df_news.columns or "bucket_time" not in df_news.columns:
        raise ValueError("News frame must contain date and bucket_time columns.")

    invalid = df_news[df_news["date"] >= df_news["bucket_time"]]
    if not invalid.empty:
        first_bad = invalid[["date", "bucket_time"]].iloc[0].to_dict()
        raise ValueError(
            "Look-ahead bias detected in news integration: "
            f"news timestamp {first_bad['date']} is not strictly before "
            f"assigned candle start {first_bad['bucket_time']}."
        )


def build_multi_stock_dataset(config: dict) -> tuple[pd.DataFrame, dict]:
    tickers = config["tickers"]
    index_name = config["index"]["name"]
    feature_cfg = config.get("model_features") or config["xgboost"]["features"]
    base_cols = feature_cfg["base_cols"]
    lags = int(feature_cfg["lags"])
    fe_cfg = config["feature_engineering"]

    clip_lower = float(fe_cfg["clip_quantiles"]["lower"])
    clip_upper = float(fe_cfg["clip_quantiles"]["upper"])
    volume_window = int(fe_cfg["volume_window"])
    min_rows_per_ticker = int(fe_cfg["min_rows_per_ticker"])
    strict_common_timestamps = bool(fe_cfg["strict_common_timestamps"])

    weights = load_weights(config, tickers)
    stock_frames, load_summary = load_stock_history(tickers)

    missing_tickers = [ticker for ticker in tickers if ticker not in stock_frames]
    if missing_tickers:
        raise ValueError("Missing stock data for tickers: " + ", ".join(missing_tickers))

    feature_frames: dict[str, pd.DataFrame] = {}
    dropped_short = []

    for ticker in tickers:
        features = build_stock_features(
            stock_frames[ticker],
            volume_window=volume_window,
            clip_lower=clip_lower,
            clip_upper=clip_upper,
        )
        if len(features) < min_rows_per_ticker:
            dropped_short.append(ticker)
            continue
        feature_frames[ticker] = features

    if dropped_short:
        raise ValueError(
            "Some tickers do not have enough feature rows after cleaning: "
            + ", ".join(dropped_short)
        )

    if strict_common_timestamps:
        common_index = None
        for ticker in tickers:
            ticker_index = feature_frames[ticker].index
            common_index = ticker_index if common_index is None else common_index.intersection(ticker_index)

        if common_index is None or len(common_index) == 0:
            raise ValueError("No common timestamps found across all stock feature frames.")

        for ticker in tickers:
            feature_frames[ticker] = feature_frames[ticker].loc[common_index].copy()

    wide_feature_block = _make_wide_feature_block(feature_frames, base_cols=base_cols, lags=lags)
    weighted_feature_block = _make_weighted_feature_block(
        feature_frames,
        weights=weights,
        base_cols=base_cols,
        lags=lags,
    )

    aligned_targets = pd.DataFrame(index=next(iter(feature_frames.values())).index)
    aligned_closes = pd.DataFrame(index=next(iter(feature_frames.values())).index)
    for ticker in tickers:
        aligned_targets[ticker] = feature_frames[ticker]["target_binary"]
        aligned_closes[ticker] = stock_frames[ticker].loc[aligned_targets.index, "close"]

    aligned_targets = aligned_targets.dropna().sort_index()
    aligned_closes = aligned_closes.loc[aligned_targets.index].dropna().sort_index()
    weighted_target = aligned_targets.mul(weights, axis=1).sum(axis=1)
    weighted_target = weighted_target.rename("basket_target_ret")
    weighted_close = aligned_closes.mul(weights, axis=1).sum(axis=1).rename("basket_close")

    index_validation = build_index_validation_frame(
        load_index_history(index_name),
        clip_lower=clip_lower,
        clip_upper=clip_upper,
    )

    dataset = pd.concat(
        [wide_feature_block, weighted_feature_block, weighted_close, weighted_target, index_validation],
        axis=1,
        join="inner",
    ).dropna()

    dataset["tracking_error_ret"] = dataset["basket_target_ret"] - dataset["index_target_ret"]
    dataset["same_direction"] = (
        np.sign(dataset["basket_target_ret"]) == np.sign(dataset["index_target_ret"])
    ).astype("int8")


    try:
        csv_names = [
            "filtered_news_with_finbert.csv",
            "filtered_news_2016_to_today.csv",
            "news_unified.csv"
        ]
        csv_path = None
        for name in csv_names:
            candidate = PROJECT_ROOT / "notebooks" / name
            if candidate.exists():
                csv_path = candidate
                break

        if csv_path:
            logger.info("Integrating news features from %s", csv_path.name)
            df_news = pd.read_csv(csv_path)
            df_news["date"] = pd.to_datetime(df_news["date"])
            if df_news["date"].dt.tz is not None:
                df_news["date"] = df_news["date"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)


            df_news["day"] = df_news["date"].dt.date

            cat_columns = []
            if "categories" in df_news.columns:

                all_tags = set()
                for val in df_news["categories"].dropna().unique():
                    for tag in str(val).split("|"):
                        tag = tag.strip()
                        if tag:
                            all_tags.add(tag)

                for tag in sorted(all_tags):
                    col_name = f"cat_{tag}"
                    df_news[col_name] = df_news["categories"].fillna("").str.contains(tag, regex=False).astype(int)
                    cat_columns.append(col_name)

                logger.info(
                    "Created %d clean category columns from %d raw category combinations.",
                    len(cat_columns),
                    len(df_news["categories"].unique()),
                )

            if "has_negation" in df_news.columns:
                df_news["has_negation"] = df_news["has_negation"].astype(int)
            if "impact_tier" in df_news.columns:
                tier_mapping = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
                df_news["impact_tier"] = df_news["impact_tier"].map(tier_mapping).fillna(1)

            agg_dict = {}
            if "sentiment" in df_news.columns: agg_dict["sentiment"] = "mean"
            if "relevance_score" in df_news.columns: agg_dict["relevance_score"] = "max"
            if "impact_tier" in df_news.columns: agg_dict["impact_tier"] = "max"
            if "has_negation" in df_news.columns: agg_dict["has_negation"] = "max"
            for i in range(1, 9):
                if f"embed_dim_{i}" in df_news.columns: agg_dict[f"embed_dim_{i}"] = "mean"
            for col in cat_columns: agg_dict[col] = "max"

            if agg_dict:
                df_news_agg = df_news.groupby("day").agg(agg_dict)


                df_news_agg["has_news"] = 1.0

                from pandas.tseries.offsets import BDay
                df_news_agg.index = pd.to_datetime(df_news_agg.index) + BDay(1)
                df_news_agg.index = df_news_agg.index.date

                dataset["day"] = dataset.index.date
                dataset = dataset.join(df_news_agg, on="day", how="left")
                dataset = dataset.drop(columns=["day"])

                dataset["has_news"] = dataset["has_news"].fillna(0.0)
                for col in agg_dict.keys():
                    dataset[col] = dataset[col].fillna(0.0)
                logger.info("Integrated %d advanced news features + 'has_news' flag.", len(agg_dict))
    except (FileNotFoundError, KeyError, ValueError, pd.errors.ParserError) as exc:
        logger.warning("Failed to integrate news features: %s", exc, exc_info=True)

    rmse = float(mean_squared_error(dataset["index_target_ret"], dataset["basket_target_ret"]) ** 0.5)
    mae = float(mean_absolute_error(dataset["index_target_ret"], dataset["basket_target_ret"]))
    corr = float(dataset["basket_target_ret"].corr(dataset["index_target_ret"]))
    direction_acc = float(dataset["same_direction"].mean())

    feature_columns = [
        col
        for col in dataset.columns
        if col not in {"basket_close", "basket_target_ret", "index_close", "index_ret", "index_target_ret", "tracking_error_ret", "same_direction"}
    ]

    summary = {
        "tickers": tickers,
        "weights": {ticker: float(weights.loc[ticker]) for ticker in tickers},
        "load_summary": load_summary,
        "n_rows": int(len(dataset)),
        "n_feature_columns": int(len(feature_columns)),
        "start": str(dataset.index.min()) if not dataset.empty else None,
        "end": str(dataset.index.max()) if not dataset.empty else None,
        "basket_vs_index": {
            "correlation": corr,
            "rmse": rmse,
            "mae": mae,
            "directional_accuracy": direction_acc,
        },
    }

    return dataset.sort_index(), summary


def get_dataset_output_path(config: dict) -> Path:
    """Get the saved modeling dataset path."""
    fe_cfg = config["feature_engineering"]
    configured_path = fe_cfg.get("output_dataset_path")
    if not configured_path:
        raise KeyError("feature_engineering.output_dataset_path is required.")
    return PROJECT_ROOT / configured_path


def save_dataset_outputs(dataset: pd.DataFrame, summary: dict, config: dict) -> tuple[Path, Path]:
    """Save the dataset and its summary."""
    fe_cfg = config["feature_engineering"]
    dataset_path = get_dataset_output_path(config)
    summary_path = PROJECT_ROOT / fe_cfg["output_summary_json"]

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_to_save = dataset.reset_index().rename(columns={"index": "bucket_time"})
    dataset_to_save.to_parquet(
        dataset_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(clean_json(summary), f, indent=2, allow_nan=False)

    return dataset_path, summary_path


def load_saved_dataset(config: dict) -> pd.DataFrame:
    """Load the saved modeling dataset."""
    dataset_path = get_dataset_output_path(config)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}. Run build_model_dataset.py first."
        )

    dataset = pd.read_parquet(dataset_path, engine="pyarrow")
    if "bucket_time" not in dataset.columns:
        raise ValueError("Saved dataset must contain a bucket_time column.")

    dataset["bucket_time"] = pd.to_datetime(dataset["bucket_time"])
    return dataset.sort_values("bucket_time").set_index("bucket_time")


def get_model_feature_columns(config: dict, dataset: pd.DataFrame) -> list[str]:
    target_col = config["modeling"]["target_column"]
    validation_target_col = config["modeling"]["validation_target_column"]
    excluded = {
        target_col,
        validation_target_col,
        "basket_close",
        "index_close",
        "index_ret",
        "tracking_error_ret",
        "same_direction",
    }
    return [col for col in dataset.columns if col not in excluded]


def split_dataset_by_time(config: dict, dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_ratio = float(config["modeling"]["train_ratio"])
    if not 0 < train_ratio < 1:
        raise ValueError("modeling.train_ratio must be between 0 and 1.")

    split_idx = int(len(dataset) * train_ratio)
    if split_idx <= 0 or split_idx >= len(dataset):
        raise ValueError("Dataset is too small for the configured train/test split.")

    train_df = dataset.iloc[:split_idx].copy()
    test_df = dataset.iloc[split_idx:].copy()
    return train_df, test_df
