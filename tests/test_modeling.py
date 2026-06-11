import numpy as np
import pandas as pd
import pytest
import torch

from src.modeling import LSTMRegressor, build_sequence_arrays


def test_model_forward_pass_shapes():
    """Check the model output shape and gradient flow."""
    ticker_weights = torch.ones(50) / 50
    global_input_size = 3

    model = LSTMRegressor(
        input_size=8,
        hidden_size=32,
        num_layers=1,
        dropout=0.0,
        ticker_weights=ticker_weights,
        global_input_size=global_input_size
    )

    x = torch.randn(2, 12, 50, 8)
    x_global = torch.randn(2, 12, 3)

    output = model(x, x_global)

    assert output.shape == (2,)
    assert not torch.isnan(output).any()

    output.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"


def test_dummy_overfit():
    """Check that the model can learn a simple synthetic pattern."""
    torch.manual_seed(42)

    ticker_weights = torch.ones(50) / 50
    global_input_size = 3

    model = LSTMRegressor(
        input_size=8,
        hidden_size=32,
        num_layers=1,
        dropout=0.0,
        ticker_weights=ticker_weights,
        global_input_size=global_input_size
    )

    x = torch.randn(16, 12, 50, 8)
    x_global = torch.randn(16, 12, 3)
    y = x[:, -1, 0, 0]

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    initial_loss = None
    for epoch in range(50):
        optimizer.zero_grad()
        pred = model(x, x_global)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        if epoch == 0:
            initial_loss = loss.item()

    final_loss = loss.item()
    assert final_loss < initial_loss * 0.5


def test_sequence_building():
    """Check sequence shapes and chronological order."""
    tickers = [
        "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK",
        "BAJAJ-AUTO","BAJAJFINSV","BAJFINANCE","BHARTIARTL","BPCL",
        "BRITANNIA","CIPLA","COALINDIA","DIVISLAB","DRREDDY",
        "EICHERMOT","GRASIM","HCLTECH","HDFCBANK","HDFCLIFE",
        "HEROMOTOCO","HINDALCO","HINDUNILVR","ICICIBANK","INDUSINDBK",
        "INFY","ITC","JSWSTEEL","KOTAKBANK","LT","LTIM","MARUTI",
        "MM","NESTLEIND","NTPC","ONGC","POWERGRID","RELIANCE",
        "SBILIFE","SBIN","SUNPHARMA","TATACONSUM","TATAMOTORS",
        "TATASTEEL","TCS","TECHM","TITAN","ULTRACEMCO","UPL","WIPRO"
    ]

    ticker_features = [
        "log_ret", "log_ret_lag1", "vol_ratio", "vol_ratio_lag1",
        "hl_ratio", "hl_ratio_lag1", "co_ratio", "co_ratio_lag1"
    ]

    feature_cols = []
    for t in tickers:
        for f in ticker_features:
            feature_cols.append(f"{t}__{f}")

    global_cols = ["sentiment_score", "has_news"]
    all_cols = feature_cols + global_cols

    num_rows = 100
    seq_len = 12

    data = np.random.rand(num_rows, len(all_cols))
    df = pd.DataFrame(data, columns=all_cols)

    df["target"] = np.random.rand(num_rows)
    df["val_target"] = np.random.randint(0, 2, num_rows)
    df["basket_close"] = np.random.rand(num_rows) * 100
    df["index_close"] = np.random.rand(num_rows) * 100
    df["index_ret"] = np.random.rand(num_rows)
    df["bucket_time"] = pd.date_range("2026-01-01", periods=num_rows, freq="5min")

    df.set_index("bucket_time", inplace=True)

    means = pd.Series(0.0, index=all_cols)
    stds = pd.Series(1.0, index=all_cols)

    X, X_global, y, df_meta = build_sequence_arrays(
        dataset=df,
        tickers=tickers,
        global_cols=global_cols,
        target_col="target",
        validation_target_col="val_target",
        sequence_length=seq_len,
        means=means,
        stds=stds
    )

    assert X.shape == (88, 12, 50, 8)
    assert X_global.shape == (88, 12, 2)
    assert y.shape == (88,)


    np.testing.assert_array_almost_equal(X[1, 0], X[0, 1])
