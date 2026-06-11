"""Train, evaluate, save, and run the Nifty 50 LSTM classifier."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import logging
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.feature_engineering import (
    CONFIG_PATH,
    PROJECT_ROOT,
    build_multi_stock_dataset,
    get_dataset_output_path,
    get_model_feature_columns,
    load_saved_dataset,
    save_dataset_outputs,
    split_dataset_by_time,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        ticker_weights: torch.Tensor,
        global_input_size: int,
    ) -> None:
        super().__init__()

        weights = ticker_weights / ticker_weights.sum()
        self.register_buffer("ticker_weights", weights)

        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=8,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size + global_input_size, 1)

    def forward(self, x: torch.Tensor, x_global: torch.Tensor) -> torch.Tensor:
        assert x.shape[2] == 50
        assert x.shape[3] == 8

        batch, seq, tickers, feats = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch * 50, seq, 8)

        lstm_out, _ = self.lstm(x)
        out = lstm_out.mean(dim=1)

        out = out.reshape(batch, 50, -1)
        agg = (out * self.ticker_weights.view(1, 50, 1)).sum(dim=1)

        global_context = x_global.mean(dim=1)
        combined = torch.cat([agg, global_context], dim=-1)

        return self.head(combined).squeeze(-1)


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        ticker_weights: torch.Tensor,
        global_input_size: int,
    ) -> None:
        super().__init__()

        weights = ticker_weights / ticker_weights.sum()
        self.register_buffer("ticker_weights", weights)

        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=8,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size + global_input_size, 1)

    def forward(self, x: torch.Tensor, x_global: torch.Tensor) -> torch.Tensor:
        assert x.shape[2] == 50
        assert x.shape[3] == 8

        batch, seq, tickers, feats = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch * 50, seq, 8)

        lstm_out, _ = self.lstm(x)
        out = lstm_out.mean(dim=1)

        out = out.reshape(batch, 50, -1)
        agg = (out * self.ticker_weights.view(1, 50, 1)).sum(dim=1)

        global_context = x_global.mean(dim=1)
        combined = torch.cat([agg, global_context], dim=-1)

        return self.head(combined).squeeze(-1)


def load_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def clean_json(value: Any) -> Any:
    """Replace invalid float values before saving JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    return value


def get_model_state_path(config: dict[str, Any]) -> Path:
    """Get the path used to save model weights."""
    return PROJECT_ROOT / config["modeling"]["output_model_path"]


def get_model_metadata_path(config: dict[str, Any]) -> Path:
    """Get the path used to save model settings and feature statistics."""
    configured = config["modeling"].get("output_model_metadata_path")
    if configured:
        return PROJECT_ROOT / configured
    return get_model_state_path(config).with_name(
        f"{get_model_state_path(config).stem}_metadata.json"
    )


def get_comparison_report_path(config: dict[str, Any]) -> Path:
    """Get the path used to save model comparison results."""
    configured = config["modeling"].get("output_comparison_report_path")
    if configured:
        return PROJECT_ROOT / configured
    return PROJECT_ROOT / "output/models/comparison_report.json"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def build_or_load_dataset(
    config: dict[str, Any],
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Load the saved dataset, or build it when needed."""
    if force_rebuild or bool(config["modeling"].get("force_rebuild_dataset", False)):
        logger.info("Building modeling dataset from database...")
        dataset, summary = build_multi_stock_dataset(config)
        dataset_path, summary_path = save_dataset_outputs(dataset, summary, config)
        logger.info("Saved dataset to %s", dataset_path)
        logger.info("Saved dataset summary to %s", summary_path)
        return dataset, summary

    try:
        logger.info("Loading modeling dataset from %s", get_dataset_output_path(config))
        dataset = load_saved_dataset(config)
        logger.info("Loaded saved dataset with %d rows.", len(dataset))
        return dataset, None
    except FileNotFoundError:
        logger.info("Saved dataset not found; building from database instead.")
        dataset, summary = build_multi_stock_dataset(config)
        dataset_path, summary_path = save_dataset_outputs(dataset, summary, config)
        logger.info("Saved dataset to %s", dataset_path)
        logger.info("Saved dataset summary to %s", summary_path)
        return dataset, summary


def get_lstm_training_config(config: dict[str, Any]) -> dict[str, Any]:
    return config["lstm"]["training"]


def compute_feature_normalization(
    feature_frame: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    means = feature_frame.mean(axis=0)
    stds = feature_frame.std(axis=0).replace(0, 1.0).fillna(1.0)
    return means, stds


def normalize_feature_frame(
    feature_frame: pd.DataFrame,
    means: pd.Series,
    stds: pd.Series,
) -> pd.DataFrame:
    return ((feature_frame - means) / stds).fillna(0.0)


def build_sequence_arrays(
    dataset: pd.DataFrame,
    tickers: list[str],
    global_cols: list[str],
    target_col: str,
    validation_target_col: str,
    sequence_length: int,
    means: pd.Series,
    stds: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    ticker_features = [
        "log_ret", "log_ret_lag1", "vol_ratio", "vol_ratio_lag1",
        "hl_ratio", "hl_ratio_lag1", "co_ratio", "co_ratio_lag1"
    ]

    if len(tickers) != 50:
        raise ValueError(f"Expected 50 tickers, got {len(tickers)}")

    all_req_cols = []
    for ticker in tickers:
        for f in ticker_features:
            col = f"{ticker}__{f}"
            if col not in dataset.columns:
                raise ValueError(f"Missing required ticker feature: {col}")
            all_req_cols.append(col)

    feature_frame = normalize_feature_frame(dataset[all_req_cols + global_cols], means, stds)

    if len(feature_frame) <= sequence_length:
        raise ValueError(
            f"Need more than {sequence_length} rows to build LSTM sequences; got {len(feature_frame)}."
        )

    X_ticker_flat = feature_frame[all_req_cols].to_numpy(dtype=np.float32, copy=True)
    X_global_flat = feature_frame[global_cols].to_numpy(dtype=np.float32, copy=True)
    y_values = dataset[target_col].to_numpy(dtype=np.float32, copy=True)
    y_values = (y_values > 0.5).astype(np.float32)

    from numpy.lib.stride_tricks import sliding_window_view

    X_ticker_strided = sliding_window_view(X_ticker_flat, window_shape=sequence_length, axis=0)[:-1]
    X = X_ticker_strided.transpose(0, 2, 1).reshape(-1, sequence_length, 50, 8).copy()

    X_global_strided = sliding_window_view(X_global_flat, window_shape=sequence_length, axis=0)[:-1]
    X_global = X_global_strided.transpose(0, 2, 1).reshape(-1, sequence_length, len(global_cols)).copy()

    y = y_values[sequence_length:]

    basket_close_vals = dataset["basket_close"].to_numpy()
    index_close_vals = dataset["index_close"].to_numpy()
    index_ret_vals = dataset["index_ret"].to_numpy()
    target_vals = dataset[target_col].to_numpy()
    val_target_vals = dataset[validation_target_col].to_numpy()
    timestamps = dataset.index.to_numpy()

    meta = pd.DataFrame({
        "timestamp": timestamps[sequence_length:],
        "basket_close": basket_close_vals[sequence_length:].astype(float),
        "index_close": index_close_vals[sequence_length:].astype(float),
        "index_ret": index_ret_vals[sequence_length:].astype(float),
        "basket_target_ret": target_vals[sequence_length:].astype(float),
        "index_target_ret": val_target_vals[sequence_length:].astype(float),
    }).set_index("timestamp")
    return X, X_global, y, meta


def build_model(config: dict[str, Any], ticker_weights: torch.Tensor, global_input_size: int) -> LSTMClassifier:
    params = get_lstm_training_config(config)
    return LSTMClassifier(
        input_size=8,
        hidden_size=int(params["hidden_size"]),
        num_layers=int(params["num_layers"]),
        dropout=float(params.get("dropout", 0.0)),
        ticker_weights=ticker_weights,
        global_input_size=global_input_size,
    )


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred_logits: np.ndarray,
    label: str,
) -> dict[str, float]:
    """Convert model logits to probabilities and calculate classification scores."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score,
        recall_score, roc_auc_score
    )
    import scipy.special
    y_prob = scipy.special.expit(y_pred_logits)
    y_pred_binary = (y_prob > 0.5).astype(int)
    y_true_binary = y_true.astype(int)

    from sklearn.metrics import precision_recall_curve
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true_binary, y_prob)

    logger.info("Precision-Recall Threshold Analysis:")
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        idx = np.searchsorted(thresholds, thresh)
        if idx < len(precision_vals):
            logger.info(
                "  threshold=%.2f → precision=%.3f, recall=%.3f",
                thresh, precision_vals[idx], recall_vals[idx]
            )

    return {
        f"{label}_accuracy": float(accuracy_score(y_true_binary, y_pred_binary)),
        f"{label}_f1": float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
        f"{label}_precision": float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
        f"{label}_recall": float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
        f"{label}_roc_auc": float(roc_auc_score(y_true_binary, y_prob)),
    }


def predict_sequences(
    model: LSTMRegressor,
    X: np.ndarray,
    X_global: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    batch_size = 128
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i+batch_size]).to(device)
            batch_global = torch.from_numpy(X_global[i:i+batch_size]).to(device)
            pred = model(batch, batch_global).detach().cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds).astype(np.float64, copy=False)


def train_and_evaluate(
    config: dict[str, Any],
    dataset: pd.DataFrame,
) -> tuple[LSTMRegressor, dict[str, Any], pd.DataFrame]:
    target_col = config["modeling"]["target_column"]
    validation_target_col = config["modeling"]["validation_target_column"]
    sequence_length = int(config["lstm"]["sequence_length"])

    feature_cols = get_model_feature_columns(config, dataset)
    train_df, test_df = split_dataset_by_time(config, dataset)

    train_df[target_col] = train_df[target_col].shift(-1)
    train_df = train_df.dropna(subset=[target_col])

    test_df[target_col] = test_df[target_col].shift(-1)
    test_df = test_df.dropna(subset=[target_col])

    dev_mode = config.get("modeling", {}).get("dev_mode", False)
    if dev_mode:
        dev_rows = config.get("modeling", {}).get("dev_rows", 15000)
        train_df = train_df.iloc[-dev_rows:]
        test_df = test_df.iloc[-int(dev_rows * 0.2):]
        logger.info("DEV MODE: using %d train rows and %d test rows", len(train_df), len(test_df))

    tickers = list(config["synthetic_index"]["weights"].keys())
    if len(tickers) != 50:
        raise ValueError(f"Expected 50 tickers in config, got {len(tickers)}")

    weights_arr = np.array([config["synthetic_index"]["weights"][t] for t in tickers], dtype=np.float32)
    weights_tensor = torch.tensor(weights_arr)

    GLOBAL_FEATURE_PREFIXES = ("cat_", "sentiment_", "embed_dim_")
    GLOBAL_FEATURE_EXACT = {
        "has_news", "has_negation", "impact_tier",
        "high_impact_count", "medium_impact_count",
        "has_high_risk_negative", "negation_count",
        "sentiment",
        "upper_wick_ratio", "lower_wick_ratio",
        "body_ratio", "buying_pressure"
    }
    global_cols = [
        c for c in dataset.columns
        if any(c.startswith(p) for p in GLOBAL_FEATURE_PREFIXES)
        or c in GLOBAL_FEATURE_EXACT
    ]
    logger.info("Global sentiment cols selected: %s", global_cols)
    ticker_features = [
        "log_ret", "log_ret_lag1", "vol_ratio", "vol_ratio_lag1",
        "hl_ratio", "hl_ratio_lag1", "co_ratio", "co_ratio_lag1"
    ]
    all_req_cols = [f"{t}__{f}" for t in tickers for f in ticker_features]
    feature_cols = all_req_cols + global_cols

    means, stds = compute_feature_normalization(train_df[feature_cols])

    cache_key = hashlib.md5(
        json.dumps({
            "target_col": target_col,
            "sequence_length": sequence_length,
            "train_ratio": config["modeling"]["train_ratio"],
            "features": all_req_cols + global_cols
        }, sort_keys=True).encode()
    ).hexdigest()[:8]
    cache_prefix = f"output/modeling/cache_{cache_key}"
    X_train_path = f"{cache_prefix}_X_train.npy"

    if os.path.exists(X_train_path) and not config["modeling"].get("force_rebuild_dataset", False) and not dev_mode:
        logger.info("Loading cached sequence arrays from %s...", cache_prefix)
        X_train = np.load(f"{cache_prefix}_X_train.npy")
        X_global_train = np.load(f"{cache_prefix}_X_global_train.npy")
        y_train = np.load(f"{cache_prefix}_y_train.npy")
        X_test = np.load(f"{cache_prefix}_X_test.npy")
        X_global_test = np.load(f"{cache_prefix}_X_global_test.npy")
        y_test = np.load(f"{cache_prefix}_y_test.npy")
        test_meta = pd.read_parquet(f"{cache_prefix}_test_meta.parquet")
    else:
        logger.info("Building sequence arrays...")
        X_train, X_global_train, y_train, _ = build_sequence_arrays(
            train_df, tickers, global_cols, target_col, validation_target_col, sequence_length, means, stds
        )
        X_test, X_global_test, y_test, test_meta = build_sequence_arrays(
            test_df, tickers, global_cols, target_col, validation_target_col, sequence_length, means, stds
        )
        if not dev_mode:
            np.save(f"{cache_prefix}_X_train.npy", X_train)
            np.save(f"{cache_prefix}_X_global_train.npy", X_global_train)
            np.save(f"{cache_prefix}_y_train.npy", y_train)
            np.save(f"{cache_prefix}_X_test.npy", X_test)
            np.save(f"{cache_prefix}_X_global_test.npy", X_global_test)
            np.save(f"{cache_prefix}_y_test.npy", y_test)
            test_meta.to_parquet(f"{cache_prefix}_test_meta.parquet")

    train_1s = int(y_train.sum())
    train_total = len(y_train)
    logger.info(f"Target Binary Train Distribution: 1s={train_1s} ({train_1s/train_total:.2%}), 0s={train_total-train_1s} ({(train_total-train_1s)/train_total:.2%})")

    device = get_device()
    model = build_model(config, weights_tensor, len(global_cols)).to(device)
    params = get_lstm_training_config(config)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params.get("weight_decay", 0.0)),
    )
    n_pos = int((y_train > 0.5).sum())
    n_neg = int((y_train <= 0.5).sum())
    logger.info("Class distribution — 0: %d (%.1f%%), 1: %d (%.1f%%)",
                n_neg, 100*n_neg/len(y_train),
                n_pos, 100*n_pos/len(y_train))
    pos_weight_val = n_neg / n_pos if n_pos > 0 else 1.0
    logger.info("Recommended pos_weight: %.2f", pos_weight_val)
    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(X_global_train),
        torch.from_numpy(y_train),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    val_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(X_global_test),
        torch.from_numpy(y_test),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    epochs = int(params["epochs"])
    epoch_losses: list[float] = []

    patience = int(params.get("early_stopping_patience", 5))
    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for X_batch, X_global_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            X_global_batch = X_global_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type=device.type):
                pred = model(X_batch, X_global_batch)
                loss = criterion(pred, y_batch)

            scaler.scale(loss).backward()

            clip_grad = float(params.get("gradient_clip_norm", 0.0))
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            scaler.step(optimizer)
            scaler.update()

            batch_size = int(y_batch.size(0))
            running_loss += float(loss.item()) * batch_size
            sample_count += batch_size

        epoch_loss = running_loss / max(sample_count, 1)
        epoch_losses.append(epoch_loss)

        model.eval()
        val_loss_sum = 0.0
        val_samples = 0
        with torch.no_grad():
            for v_batch, vg_batch, vy_batch in val_loader:
                v_batch = v_batch.to(device)
                vg_batch = vg_batch.to(device)
                vy_batch = vy_batch.to(device)
                val_pred = model(v_batch, vg_batch)
                batch_loss = criterion(val_pred, vy_batch)
                batch_sz = int(vy_batch.size(0))
                val_loss_sum += float(batch_loss.item()) * batch_sz
                val_samples += batch_sz
        val_loss = val_loss_sum / max(val_samples, 1)

        logger.info(
            "Epoch %d/%d - train_loss=%.8f - val_loss=%.8f",
            epoch + 1,
            epochs,
            epoch_loss,
            val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    logger.info("Ticker Weight Contributions:")
    for ticker, weight in zip(tickers, model.ticker_weights.detach().cpu().numpy()):
        logger.info("  %s: %.4f", ticker, weight)

    test_pred = predict_sequences(model, X_test, X_global_test, device=device)
    prediction_frame = pd.DataFrame(
        {
            "predicted_basket_target_ret": test_pred,
            "actual_basket_target_ret": y_test.astype(np.float64),
            "actual_index_target_ret": test_meta["index_target_ret"].to_numpy(dtype=np.float64),
        },
        index=test_meta.index,
    )

    metrics: dict[str, Any] = {
        "model_type": "lstm",
        "device": str(device),
        "sequence_length": sequence_length,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_sequences": int(len(X_train)),
        "test_sequences": int(len(X_test)),
        "feature_count": int(len(feature_cols)),
        "train_start": str(train_df.index.min()),
        "train_end": str(train_df.index.max()),
        "test_start": str(test_df.index.min()),
        "test_end": str(test_df.index.max()),
        "final_train_loss": float(epoch_losses[-1]) if epoch_losses else None,
    }
    metrics.update(evaluate_predictions(y_test, test_pred, label="basket"))

    y_test_binary = (y_test > 0.5).astype(int)
    y_train_binary = (y_train > 0.5).astype(int)
    X_test_flat = X_global_test[:, -1, :]
    X_train_flat = X_global_train[:, -1, :]

    majority_class = int(y_train_binary.mean() > 0.5)
    majority_pred = np.full(len(y_test_binary), majority_class, dtype=int)
    majority_roc = roc_auc_score(y_test_binary, majority_pred)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_flat, y_train_binary)
    lr_probs = lr.predict_proba(X_test_flat)[:, 1]
    lr_roc = roc_auc_score(y_test_binary, lr_probs)

    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )
    xgb_model.fit(X_train_flat, y_train_binary)
    xgb_probs = xgb_model.predict_proba(X_test_flat)[:, 1]
    xgb_roc = roc_auc_score(y_test_binary, xgb_probs)

    lstm_probs = torch.sigmoid(torch.from_numpy(test_pred)).numpy()
    lstm_roc = roc_auc_score(y_test_binary, lstm_probs)

    logger.info("=" * 50)
    logger.info("BASELINE COMPARISON:")
    logger.info("  Majority classifier ROC-AUC: %.4f", majority_roc)
    logger.info("  Logistic Regression ROC-AUC: %.4f", lr_roc)
    logger.info("  XGBoost (global only) ROC-AUC: %.4f", xgb_roc)
    logger.info("  LSTM (full model) ROC-AUC:   %.4f", lstm_roc)
    logger.info("  LSTM lift over XGBoost: %+.4f", lstm_roc - xgb_roc)
    logger.info("=" * 50)

    metrics["baseline_majority_roc"] = float(majority_roc)
    metrics["baseline_lr_roc"] = float(lr_roc)
    metrics["baseline_xgb_roc"] = float(xgb_roc)
    metrics["lstm_roc_auc"] = float(lstm_roc)
    metrics["lstm_lift_over_xgb"] = float(lstm_roc - xgb_roc)

    y_pred_t = torch.tensor(test_pred)
    y_test_t = torch.tensor(y_test)
    correct_direction = ((y_pred_t > 0) == (y_test_t > 0)).float().mean()
    metrics["directional_accuracy"] = float(correct_direction.item())

    metrics["xgboost_baseline"] = None
    return model, metrics, prediction_frame


def build_latest_sequence(
    dataset: pd.DataFrame,
    tickers: list[str],
    global_cols: list[str],
    means: pd.Series,
    stds: pd.Series,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    ticker_features = [
        "log_ret", "log_ret_lag1", "vol_ratio", "vol_ratio_lag1",
        "hl_ratio", "hl_ratio_lag1", "co_ratio", "co_ratio_lag1"
    ]
    all_req_cols = [f"{t}__{f}" for t in tickers for f in ticker_features]

    normalized = normalize_feature_frame(dataset[all_req_cols + global_cols], means, stds)
    if len(normalized) < sequence_length:
        raise ValueError(
            f"Need at least {sequence_length} rows for live inference; got {len(normalized)}."
        )

    tail = normalized.tail(sequence_length)
    X_ticker_flat = tail[all_req_cols].to_numpy(dtype=np.float32, copy=True)
    X_ticker_3d = X_ticker_flat.reshape(sequence_length, 50, 8)

    X_global_flat = tail[global_cols].to_numpy(dtype=np.float32, copy=True)

    return X_ticker_3d, X_global_flat


def _get_required_features(config: dict[str, Any], dataset: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    tickers = list(config["synthetic_index"]["weights"].keys())
    GLOBAL_FEATURE_PREFIXES = ("cat_", "sentiment_", "embed_dim_")
    GLOBAL_FEATURE_EXACT = {
        "has_news", "has_negation", "impact_tier",
        "high_impact_count", "medium_impact_count",
        "has_high_risk_negative", "negation_count",
        "sentiment",
        "upper_wick_ratio", "lower_wick_ratio",
        "body_ratio", "buying_pressure"
    }
    global_cols = [
        c for c in dataset.columns
        if any(c.startswith(p) for p in GLOBAL_FEATURE_PREFIXES)
        or c in GLOBAL_FEATURE_EXACT
    ]
    ticker_features = [
        "log_ret", "log_ret_lag1", "vol_ratio", "vol_ratio_lag1",
        "hl_ratio", "hl_ratio_lag1", "co_ratio", "co_ratio_lag1"
    ]
    all_req_cols = [f"{t}__{f}" for t in tickers for f in ticker_features]
    return tickers, global_cols, all_req_cols + global_cols


def run_latest_inference(
    model: LSTMRegressor,
    config: dict[str, Any],
    dataset: pd.DataFrame,
    *,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sequence_length = int(
        artifact["sequence_length"] if artifact else config["lstm"]["sequence_length"]
    )
    tickers, global_cols, feature_cols = _get_required_features(config, dataset)
    if artifact:
        feature_cols = artifact["feature_columns"]

    if artifact:
        means = pd.Series(artifact["feature_means"], index=feature_cols, dtype="float64")
        stds = pd.Series(artifact["feature_stds"], index=feature_cols, dtype="float64")
    else:
        train_df, _ = split_dataset_by_time(config, dataset)
        means, stds = compute_feature_normalization(train_df[feature_cols])

    latest_sequence, latest_global = build_latest_sequence(
        dataset, tickers, global_cols, means, stds, sequence_length
    )
    device = next(model.parameters()).device
    predicted_ret = float(
        predict_sequences(
            model,
            latest_sequence[np.newaxis, :, :, :],
            latest_global[np.newaxis, :, :],
            device=device
        )[0]
    )

    latest_row = dataset.iloc[-1]
    current_basket_close = float(latest_row["basket_close"])
    implied_next_close = float(current_basket_close * np.exp(predicted_ret))

    result = {
        "model_type": "lstm",
        "latest_timestamp": str(dataset.index[-1]),
        "current_basket_close": current_basket_close,
        "predicted_next_5m_return": predicted_ret,
        "predicted_direction": "UP" if predicted_ret > 0 else ("DOWN" if predicted_ret < 0 else "FLAT"),
        "implied_next_basket_close": implied_next_close,
        "latest_real_index_close": float(latest_row["index_close"]),
        "latest_real_index_return": float(latest_row["index_ret"]),
        "sequence_length": sequence_length,
    }
    return result


def build_model_artifact(
    model: LSTMRegressor,
    config: dict[str, Any],
    feature_cols: list[str],
    means: pd.Series,
    stds: pd.Series,
    global_input_size: int,
) -> dict[str, Any]:
    """Store the settings needed to load this model again."""
    params = get_lstm_training_config(config)
    return {
        "model_type": "lstm",
        "sequence_length": int(config["lstm"]["sequence_length"]),
        "feature_columns": list(feature_cols),
        "feature_means": [float(means.loc[col]) for col in feature_cols],
        "feature_stds": [float(stds.loc[col]) for col in feature_cols],
        "model_kwargs": {
            "hidden_size": int(params["hidden_size"]),
            "num_layers": int(params["num_layers"]),
            "dropout": float(params.get("dropout", 0.0)),
            "global_input_size": global_input_size,
        },
    }


def save_model_metadata(metadata: dict[str, Any], config: dict[str, Any]) -> Path:
    """Save model settings and feature statistics as JSON."""
    metadata_path = get_model_metadata_path(config)
    ensure_parent(metadata_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(clean_json(metadata), f, indent=2, allow_nan=False)
    return metadata_path


def load_model_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Load the settings saved with the trained model."""
    metadata_path = get_model_metadata_path(config)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Model metadata not found: {metadata_path}. Train the model first."
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not isinstance(metadata, dict) or metadata.get("model_type") != "lstm":
        raise TypeError(f"Metadata at {metadata_path} is not a valid LSTM artifact.")
    return metadata


def load_model_state_dict(weights_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load model weights on the selected device."""
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}. Train the model first.")

    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        logger.warning(
            "Installed PyTorch does not support weights_only=True; loading state_dict without that flag."
        )
        state_dict = torch.load(weights_path, map_location=device)

    if not isinstance(state_dict, dict):
        raise TypeError(f"Weights at {weights_path} did not contain a PyTorch state_dict.")
    return state_dict


def load_saved_model(config: dict[str, Any]) -> tuple[LSTMRegressor, dict[str, Any]]:
    """Rebuild the LSTM and load its saved weights."""
    metadata = load_model_metadata(config)
    device = get_device()

    weights = config["synthetic_index"]["weights"]
    tickers = list(weights.keys())
    weights_arr = np.array([weights[ticker] for ticker in tickers], dtype=np.float32)
    weights_tensor = torch.tensor(weights_arr)

    kwargs = dict(metadata["model_kwargs"])
    kwargs.setdefault("input_size", 8)
    kwargs["ticker_weights"] = weights_tensor

    model = LSTMRegressor(**kwargs)
    state_dict = load_model_state_dict(get_model_state_path(config), device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, metadata


def run_live_inference(
    config: dict[str, Any],
    *,
    rebuild_dataset: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model, artifact = load_saved_model(config)
    dataset, _ = build_or_load_dataset(config, force_rebuild=rebuild_dataset)
    latest_prediction = run_latest_inference(model, config, dataset, artifact=artifact)
    return latest_prediction, dataset


def save_outputs(
    model: LSTMRegressor,
    metrics: dict[str, Any],
    latest_prediction: dict[str, Any],
    config: dict[str, Any],
    dataset: pd.DataFrame,
) -> tuple[Path, Path, Path, Path]:
    """Save the trained model, metrics, and latest prediction."""
    tickers, global_cols, feature_cols = _get_required_features(config, dataset)
    train_df, _ = split_dataset_by_time(config, dataset)
    means, stds = compute_feature_normalization(train_df[feature_cols])

    model_state_path = get_model_state_path(config)
    metadata_path = get_model_metadata_path(config)
    metrics_path = PROJECT_ROOT / config["modeling"]["output_metrics_path"]
    comparison_path = get_comparison_report_path(config)
    latest_path = PROJECT_ROOT / config["modeling"]["output_latest_prediction_path"]

    for path in (model_state_path, metadata_path, metrics_path, comparison_path, latest_path):
        ensure_parent(path)

    torch.save(model.cpu().state_dict(), model_state_path)
    metadata = build_model_artifact(model, config, feature_cols, means, stds, global_input_size=len(global_cols))
    save_model_metadata(metadata, config)
    model.to(get_device())

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(clean_json(metrics), f, indent=2, allow_nan=False)

    comparison_report = metrics.get("comparison_report", {})
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(clean_json(comparison_report), f, indent=2, allow_nan=False)

    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(clean_json(latest_prediction), f, indent=2, allow_nan=False)

    return model_state_path, metadata_path, metrics_path, latest_path


def configure_challenger_paths(
    config: dict[str, Any], run_id: str
) -> tuple[dict[str, Any], Path]:
    """Point a challenger run to its own output folder."""
    challenger_config = copy.deepcopy(config)
    challenger_dir = PROJECT_ROOT / "output" / "models" / "challengers" / run_id
    modeling = challenger_config["modeling"]
    modeling["output_model_path"] = str(
        (challenger_dir / "nifty50_lstm_state.pt").relative_to(PROJECT_ROOT)
    )
    modeling["output_model_metadata_path"] = str(
        (challenger_dir / "nifty50_lstm_metadata.json").relative_to(PROJECT_ROOT)
    )
    modeling["output_metrics_path"] = str(
        (challenger_dir / "nifty50_lstm_metrics.json").relative_to(PROJECT_ROOT)
    )
    modeling["output_comparison_report_path"] = str(
        (challenger_dir / "comparison_report.json").relative_to(PROJECT_ROOT)
    )
    modeling["output_latest_prediction_path"] = str(
        (challenger_dir / "nifty50_lstm_latest_prediction.json").relative_to(
            PROJECT_ROOT
        )
    )
    return challenger_config, challenger_dir


def snapshot_champion(config: dict[str, Any], run_id: str) -> Path:
    """Back up the current champion before training a challenger."""
    snapshot_dir = PROJECT_ROOT / "output" / "models" / "champions" / run_id
    paths = [
        get_model_state_path(config),
        get_model_metadata_path(config),
        PROJECT_ROOT / config["modeling"]["output_metrics_path"],
        get_comparison_report_path(config),
        PROJECT_ROOT / config["modeling"]["output_latest_prediction_path"],
    ]
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            shutil.copy2(path, snapshot_dir / path.name)
    return snapshot_dir


def promote_challenger(
    champion_config: dict[str, Any],
    challenger_config: dict[str, Any],
) -> None:
    """Promote an accepted challenger without leaving partial files."""
    path_pairs = [
        (get_model_state_path(challenger_config), get_model_state_path(champion_config)),
        (
            get_model_metadata_path(challenger_config),
            get_model_metadata_path(champion_config),
        ),
        (
            PROJECT_ROOT / challenger_config["modeling"]["output_metrics_path"],
            PROJECT_ROOT / champion_config["modeling"]["output_metrics_path"],
        ),
        (
            get_comparison_report_path(challenger_config),
            get_comparison_report_path(champion_config),
        ),
        (
            PROJECT_ROOT
            / challenger_config["modeling"]["output_latest_prediction_path"],
            PROJECT_ROOT / champion_config["modeling"]["output_latest_prediction_path"],
        ),
    ]
    for source, destination in path_pairs:
        ensure_parent(destination)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination)


def challenger_beats_champion(
    champion: dict[str, Any], challenger: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    """Accept a challenger only when it improves the required metrics."""
    champion_auc = float(champion.get("basket_roc_auc", 0.0))
    challenger_auc = float(challenger.get("basket_roc_auc", 0.0))
    champion_f1 = float(champion.get("basket_f1", 0.0))
    challenger_f1 = float(challenger.get("basket_f1", 0.0))
    challenger_lift = float(challenger.get("lstm_lift_over_xgb", 0.0))
    accepted = (
        challenger_auc > champion_auc
        and challenger_f1 >= champion_f1
        and challenger_lift > 0.0
    )
    comparison = {
        "accepted": accepted,
        "criteria": {
            "roc_auc_strictly_improves": challenger_auc > champion_auc,
            "f1_does_not_decline": challenger_f1 >= champion_f1,
            "positive_lift_over_xgboost": challenger_lift > 0.0,
        },
        "champion": {
            "basket_roc_auc": champion_auc,
            "basket_f1": champion_f1,
            "lstm_lift_over_xgb": champion.get("lstm_lift_over_xgb"),
        },
        "challenger": {
            "basket_roc_auc": challenger_auc,
            "basket_f1": challenger_f1,
            "lstm_lift_over_xgb": challenger_lift,
        },
    }
    return accepted, comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM model for synthetic Nifty 50 next-bar returns.")
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Rebuild the modeling dataset from the database before training.",
    )
    parser.add_argument(
        "--challenger",
        action="store_true",
        help="Train in isolation and promote only when champion criteria pass.",
    )
    args = parser.parse_args()

    config = load_config()
    champion_config = config
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    champion_snapshot = None
    champion_metrics: dict[str, Any] = {}
    if args.challenger:
        champion_metrics_path = (
            PROJECT_ROOT / champion_config["modeling"]["output_metrics_path"]
        )
        if not champion_metrics_path.exists():
            raise FileNotFoundError(
                f"Champion metrics not found: {champion_metrics_path}"
            )
        with open(champion_metrics_path, encoding="utf-8") as handle:
            champion_metrics = json.load(handle)
        champion_snapshot = snapshot_champion(champion_config, run_id)
        config, challenger_dir = configure_challenger_paths(config, run_id)
        logger.info("Protected champion snapshot at %s", champion_snapshot)
        logger.info("Challenger artifacts will be written to %s", challenger_dir)

    dataset, dataset_summary = build_or_load_dataset(
        config, force_rebuild=args.rebuild_dataset
    )
    logger.info("Modeling dataset shape: %s", dataset.shape)

    model, metrics, _ = train_and_evaluate(config, dataset)
    if dataset_summary is not None:
        metrics["dataset_summary"] = dataset_summary

    latest_prediction = run_latest_inference(model, config, dataset)
    model_path, metadata_path, metrics_path, latest_path = save_outputs(
        model,
        metrics,
        latest_prediction,
        config,
        dataset,
    )

    logger.info("Saved LSTM state dict to %s", model_path)
    logger.info("Saved model metadata to %s", metadata_path)
    logger.info("Saved metrics to %s", metrics_path)
    logger.info("Saved latest prediction to %s", latest_path)
    if args.challenger:
        accepted, comparison = challenger_beats_champion(
            champion_metrics, metrics
        )
        comparison["run_id"] = run_id
        comparison["champion_snapshot"] = str(champion_snapshot)
        comparison_path = metrics_path.parent / "champion_comparison.json"
        with open(comparison_path, "w", encoding="utf-8") as handle:
            json.dump(clean_json(comparison), handle, indent=2, allow_nan=False)
        if accepted:
            promote_challenger(champion_config, config)
            logger.info("CHALLENGER ACCEPTED and promoted to champion.")
        else:
            logger.info("CHALLENGER REJECTED; existing champion remains active.")
        logger.info("Champion comparison saved to %s", comparison_path)
    logger.info(
        "Latest inference: ts=%s predicted_next_5m_return=%.6f implied_next_basket_close=%.4f",
        latest_prediction["latest_timestamp"],
        latest_prediction["predicted_next_5m_return"],
        latest_prediction["implied_next_basket_close"],
    )


if __name__ == "__main__":
    main()
