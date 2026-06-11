    """Streamlit showcase dashboard for the Nifty 50 ML trading system."""

    from __future__ import annotations

    import json
    from pathlib import Path
    from typing import Any

    import pandas as pd
    import plotly.graph_objects as go
    import streamlit as st


    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    METRICS_PATH = PROJECT_ROOT / "output/models/nifty50_lstm_metrics.json"
    PREDICTION_PATH = PROJECT_ROOT / "output/models/nifty50_lstm_latest_prediction.json"
    DATASET_PATH = PROJECT_ROOT / "output/modeling/nifty50_wide_dataset.parquet"
    NEWS_PATH = PROJECT_ROOT / "output/news/latest_news_scored.csv"

    PR_RESULTS = pd.DataFrame(
        {
            "Threshold": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
            "Precision": [0.320, 0.341, 0.367, 0.400, 0.467, 0.503],
            "Recall": [0.403, 0.282, 0.178, 0.092, 0.042, 0.010],
        }
    )

    st.set_page_config(
        page_title="Nifty 50 ML Trading System",
        page_icon="chart_with_upwards_trend",
        layout="wide",
    )


    @st.cache_data(show_spinner=False)
    def load_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)


    @st.cache_data(show_spinner=False)
    def load_sentiment_dataset(path: Path) -> pd.DataFrame:
        columns = [
            "bucket_time",
            "sentiment",
            "impact_tier",
            "has_negation",
            "has_news",
        ]
        parquet_columns = pd.read_parquet(path, columns=None).columns
        columns.extend(column for column in parquet_columns if column.startswith("cat_"))
        frame = pd.read_parquet(path, columns=columns)
        frame["bucket_time"] = pd.to_datetime(frame["bucket_time"])
        return frame.set_index("bucket_time")


    @st.cache_data(show_spinner=False)
    def load_scored_news(path: Path) -> pd.DataFrame:
        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["sentiment_score"] = pd.to_numeric(
            frame["sentiment_score"], errors="coerce"
        ).fillna(0.0)
        return frame.dropna(subset=["date"])


    def require_artifact(data: dict[str, Any], path: Path, label: str) -> None:
        if data:
            return
        st.error(f"{label} is unavailable. Expected artifact: `{path.relative_to(PROJECT_ROOT)}`")
        st.stop()


    def metric_value(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
        value = metrics.get(key, default)
        return float(value) if value is not None else default


    def render_model_performance(metrics: dict[str, Any]) -> None:
        st.title("Model Performance Dashboard")
        st.caption(
            "Binary classifier: predicts whether the weighted Nifty 50 basket rises "
            "more than 0.1% during the next 30 minutes."
        )

        lstm_auc = metric_value(metrics, "basket_roc_auc")
        xgb_auc = metric_value(metrics, "baseline_xgb_roc", 0.5)
        lift = lstm_auc - xgb_auc

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ROC-AUC", f"{lstm_auc:.4f}", f"{lift:+.4f} vs XGBoost")
        col2.metric("F1 Score", f"{metric_value(metrics, 'basket_f1'):.4f}")
        col3.metric("Precision", f"{metric_value(metrics, 'basket_precision'):.4f}")
        col4.metric("Recall", f"{metric_value(metrics, 'basket_recall'):.4f}")

        st.divider()
        st.subheader("Baseline Comparison")

        baseline_data = pd.DataFrame(
            {
                "Model": [
                    "Majority Classifier",
                    "Logistic Regression",
                    "XGBoost (global only)",
                    "Shared LSTM (ours)",
                ],
                "ROC-AUC": [
                    metric_value(metrics, "baseline_majority_roc", 0.500),
                    metric_value(metrics, "baseline_lr_roc", 0.522),
                    metric_value(metrics, "baseline_xgb_roc", 0.500),
                    metric_value(metrics, "basket_roc_auc", 0.637),
                ],
            }
        )
        colors = ["#888888", "#4488ff", "#ff8844", "#44cc44"]
        figure = go.Figure(
            go.Bar(
                x=baseline_data["Model"],
                y=baseline_data["ROC-AUC"],
                marker_color=colors,
                text=[f"{value:.4f}" for value in baseline_data["ROC-AUC"]],
                textposition="outside",
            )
        )
        figure.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="#d9534f",
            annotation_text="Random baseline",
        )
        figure.update_layout(
            yaxis={"range": [0.45, max(0.70, lstm_auc + 0.04)], "title": "ROC-AUC"},
            height=420,
            title="Full-run ROC-AUC comparison",
        )
        st.plotly_chart(figure, use_container_width=True)

        st.divider()
        st.subheader("Precision-Recall Tradeoff")
        st.caption(
            "Threshold values below are retained experimental results. They should be "
            "recomputed and saved automatically for each new model run."
        )

        left, right = st.columns(2)
        with left:
            pr_figure = go.Figure()
            pr_figure.add_trace(
                go.Scatter(
                    x=PR_RESULTS["Threshold"],
                    y=PR_RESULTS["Precision"],
                    name="Precision",
                    mode="lines+markers",
                    line={"color": "#2ca02c", "width": 2},
                )
            )
            pr_figure.add_trace(
                go.Scatter(
                    x=PR_RESULTS["Threshold"],
                    y=PR_RESULTS["Recall"],
                    name="Recall",
                    mode="lines+markers",
                    line={"color": "#4c78a8", "width": 2},
                )
            )
            pr_figure.update_layout(
                title="Precision and recall by decision threshold",
                xaxis_title="Decision threshold",
                yaxis_title="Score",
                height=360,
            )
            st.plotly_chart(pr_figure, use_container_width=True)

        with right:
            threshold = st.select_slider(
                "Select an inference threshold",
                options=PR_RESULTS["Threshold"].tolist(),
                value=0.65,
            )
            row = PR_RESULTS.loc[PR_RESULTS["Threshold"] == threshold].iloc[0]
            st.metric("Precision", f"{row['Precision']:.1%}")
            st.metric("Recall", f"{row['Recall']:.1%}")
            st.info(
                "A classification threshold is an operating choice, not proof of "
                "profitability. Transaction costs and walk-forward results are still required."
            )

        st.divider()
        st.subheader("Training Details")
        col1, col2, col3 = st.columns(3)
        col1.metric("Training Sequences", f"{int(metrics.get('train_sequences', 0)):,}")
        col2.metric("Evaluation Sequences", f"{int(metrics.get('test_sequences', 0)):,}")
        col3.metric("Feature Count", f"{int(metrics.get('feature_count', 0)):,}")

        col1, col2 = st.columns(2)
        col1.metric(
            "Train Period",
            f"{str(metrics.get('train_start', 'N/A'))[:10]} to "
            f"{str(metrics.get('train_end', 'N/A'))[:10]}",
        )
        col2.metric(
            "Evaluation Period",
            f"{str(metrics.get('test_start', 'N/A'))[:10]} to "
            f"{str(metrics.get('test_end', 'N/A'))[:10]}",
        )


    def render_sentiment_analysis() -> None:
        st.title("News Sentiment Analysis")
        st.caption("FinBERT-derived market context joined to the modeling dataset.")

        if not DATASET_PATH.exists():
            st.error(f"Dataset not found: `{DATASET_PATH.relative_to(PROJECT_ROOT)}`")
            st.stop()

        with st.spinner("Loading sentiment features..."):
            frame = load_sentiment_dataset(DATASET_PATH)

        col1, col2, col3 = st.columns(3)
        col1.metric("Modeling Rows", f"{len(frame):,}")
        col2.metric("Rows With News", f"{frame['has_news'].mean():.1%}")
        col3.metric("News Features", f"{len(frame.columns):,}")

        frame.index = pd.to_datetime(frame.index)
        monthly = (
            frame[["has_news", "sentiment"]]
            .resample("ME")
            .agg(has_news_pct=("has_news", "mean"), avg_sentiment=("sentiment", "mean"))
            .reset_index()
        )
        monthly = monthly.rename(columns={monthly.columns[0]: "date"})

        coverage_figure = go.Figure(
            go.Bar(
                x=monthly["date"],
                y=monthly["has_news_pct"],
                marker_color="#4c78a8",
                name="News coverage",
            )
        )
        coverage_figure.update_layout(
            title="Monthly news-feature coverage",
            yaxis_title="Share of market rows",
            yaxis_tickformat=".0%",
            height=350,
        )
        st.plotly_chart(coverage_figure, use_container_width=True)

        sentiment_figure = go.Figure(
            go.Scatter(
                x=monthly["date"],
                y=monthly["avg_sentiment"],
                mode="lines",
                line={"color": "#2ca02c", "width": 2},
                fill="tozeroy",
                fillcolor="rgba(44, 160, 44, 0.12)",
            )
        )
        sentiment_figure.add_hline(y=0, line_dash="dash", line_color="#8a94a6")
        sentiment_figure.update_layout(
            title="Average monthly sentiment",
            yaxis_title="Sentiment score",
            height=320,
        )
        st.plotly_chart(sentiment_figure, use_container_width=True)

        st.subheader("News Category Coverage")
        category_columns = [column for column in frame.columns if column.startswith("cat_")]
        category_means = frame[category_columns].mean().sort_values()
        category_names = [
            column.removeprefix("cat_").replace("_", " ").title()
            for column in category_means.index
        ]
        category_figure = go.Figure(
            go.Bar(
                x=category_means.values,
                y=category_names,
                orientation="h",
                marker_color="#4c78a8",
            )
        )
        category_figure.update_layout(
            xaxis_title="Share of modeling rows",
            height=540,
        )
        st.plotly_chart(category_figure, use_container_width=True)


    def render_seven_day_report() -> None:
        st.title("7-Day Trading Report")
        st.caption("Last 7 trading days: model features, sentiment, and news activity.")

        if not DATASET_PATH.exists():
            st.error(f"Dataset not found: `{DATASET_PATH.relative_to(PROJECT_ROOT)}`")
            st.stop()

        with st.spinner("Loading dataset..."):
            try:
                frame = load_sentiment_dataset(DATASET_PATH)
                frame.index = pd.to_datetime(frame.index)
                news = load_scored_news(NEWS_PATH) if NEWS_PATH.exists() else pd.DataFrame()
            except Exception as exc:
                st.error(f"Could not load dataset: {exc}")
                st.stop()

        all_dates = sorted(frame.index.normalize().unique())
        if len(all_dates) < 7:
            st.warning("Fewer than 7 trading days are available in the dataset.")
            st.stop()

        last_seven_dates = all_dates[-7:]
        week = frame.loc[frame.index.normalize().isin(last_seven_dates)].copy()

        if week.empty:
            st.warning("No data found for the last 7 trading days.")
            st.stop()

        trading_dates = pd.DatetimeIndex(last_seven_dates)
        week_news = news[news["date"].isin(trading_dates)].copy()
        articles_by_day = (
            week_news.assign(report_date=week_news["date"].dt.date)
            .groupby("report_date")
            .size()
            if not week_news.empty
            else pd.Series(dtype="int64")
        )
        sentiment_by_day = (
            week_news.assign(report_date=week_news["date"].dt.date)
            .groupby("report_date")["sentiment_score"]
            .mean()
            if not week_news.empty
            else pd.Series(dtype="float64")
        )

        st.caption(
            f"Period: {last_seven_dates[0].date()} to {last_seven_dates[-1].date()}"
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Trading Days", len(last_seven_dates))
        col2.metric("Total Candles", f"{len(week):,}")
        col3.metric("News Articles", f"{len(week_news):,}")
        col4.metric(
            "Avg Sentiment",
            f"{week_news['sentiment_score'].mean():.4f}" if not week_news.empty else "N/A",
        )

        daily = (
            week.groupby(week.index.normalize())
            .agg(candles=("sentiment", "count"))
            .reset_index()
        )
        daily = daily.rename(columns={daily.columns[0]: "date"})
        daily["date"] = pd.to_datetime(daily["date"])
        report_dates = daily["date"].dt.date
        daily["news_articles"] = report_dates.map(articles_by_day).fillna(0).astype(int)
        daily["avg_sentiment"] = report_dates.map(sentiment_by_day).fillna(0.0)
        day_labels = daily["date"].dt.strftime("%a %d %b")

        st.divider()
        st.subheader("Daily Sentiment Trend")
        sentiment_figure = go.Figure(
            go.Bar(
                x=day_labels,
                y=daily["avg_sentiment"],
                marker_color=[
                    "#44bb44" if value >= 0 else "#ff4444"
                    for value in daily["avg_sentiment"]
                ],
                name="Average sentiment",
            )
        )
        sentiment_figure.add_hline(y=0, line_dash="dash", line_color="gray")
        sentiment_figure.update_layout(
            title="Average FinBERT sentiment by day",
            yaxis_title="Sentiment score",
            height=300,
        )
        st.plotly_chart(sentiment_figure, use_container_width=True)

        st.divider()
        st.subheader("News Category Activity This Week")
        category_columns = [
            column for column in frame.columns if column.startswith("cat_")
        ]
        category_counts = pd.Series(0, index=category_columns, dtype="int64")
        if not week_news.empty:
            categories = (
                week_news["sectors_matched"]
                .fillna("")
                .str.split(r"[,;|]")
                .explode()
                .str.strip()
            )
            categories = categories[categories.ne("")]
            live_counts = categories.value_counts()
            for category, count in live_counts.items():
                feature_name = f"cat_sector_{category}"
                if feature_name in category_counts.index:
                    category_counts.loc[feature_name] = int(count)

        if not category_counts.empty:
            category_counts = category_counts.sort_values()
            category_figure = go.Figure(
                go.Bar(
                    x=category_counts.values,
                    y=[
                        category.removeprefix("cat_").replace("_", " ").title()
                        for category in category_counts.index
                    ],
                    orientation="h",
                    marker_color="#4488ff",
                )
            )
            category_figure.update_layout(
                title="News category mentions: last 7 trading days",
                xaxis_title="Articles",
                height=560,
            )
            st.plotly_chart(category_figure, use_container_width=True)
        else:
            st.info("No sector-tagged articles were found for this period.")

        st.divider()
        st.subheader("Daily News Coverage")
        coverage_figure = go.Figure(
            go.Bar(
                x=day_labels,
                y=daily["news_articles"],
                marker_color="#4488ff",
                name="News articles",
            )
        )
        coverage_figure.update_layout(
            title="Scored news articles by day",
            yaxis_title="Articles",
            height=280,
        )
        st.plotly_chart(coverage_figure, use_container_width=True)

        st.divider()
        st.subheader("Daily Summary Table")
        display_frame = daily.copy()
        display_frame["date"] = display_frame["date"].dt.strftime("%A, %d %b %Y")
        display_frame["avg_sentiment"] = display_frame["avg_sentiment"].round(4)
        display_frame = display_frame[
            ["date", "candles", "news_articles", "avg_sentiment"]
        ]
        display_frame = display_frame.rename(
            columns={
                "date": "Date",
                "avg_sentiment": "Avg Sentiment",
                "news_articles": "News Articles",
                "candles": "5-min Candles",
            }
        )
        st.dataframe(display_frame, use_container_width=True, hide_index=True)
        st.caption(
            "Market candles come from the modeling dataset; news and sentiment come "
            "from the latest scored-news artifact."
        )


    def render_project_summary(metrics: dict[str, Any], prediction: dict[str, Any]) -> None:
        st.title("Project Summary")
        st.markdown(
            """
            ## Nifty 50 ML Trading System

            An end-to-end machine-learning research pipeline for short-horizon Indian
            equity-market classification.

            **Research target:** Will the weighted Nifty 50 basket rise by more than
            0.1% during the next 30 minutes?

            - Shared LSTM encoder across 50 constituent stocks
            - Market-cap-style weighted stock aggregation
            - Price action, volume pressure, and FinBERT-derived news context
            - Chronological train/evaluation split and train-only normalization
            - PostgreSQL ingestion, FastAPI delivery, and Streamlit visualization
            """
        )

        st.divider()
        st.subheader("Latest Saved Inference")
        if prediction:
            col1, col2, col3 = st.columns(3)
            col1.metric("Direction", str(prediction.get("predicted_direction", "N/A")))
            col2.metric(
                "Model Output",
                f"{float(prediction.get('predicted_next_5m_return', 0.0)):.4f}",
            )
            col3.metric("Timestamp", str(prediction.get("latest_timestamp", "N/A")))
            st.caption(
                "The legacy prediction artifact still uses return-oriented field names; "
                "the current classifier output is a raw logit and should be exposed as a "
                "probability in a future artifact revision."
            )
        else:
            st.info("No latest-prediction artifact is available.")

        st.divider()
        st.subheader("Model vs Baselines")
        results = pd.DataFrame(
            {
                "Model": [
                    "Majority Classifier",
                    "Logistic Regression",
                    "XGBoost (global only)",
                    "Shared LSTM",
                ],
                "ROC-AUC": [
                    metric_value(metrics, "baseline_majority_roc", 0.5),
                    metric_value(metrics, "baseline_lr_roc", 0.5),
                    metric_value(metrics, "baseline_xgb_roc", 0.5),
                    metric_value(metrics, "basket_roc_auc"),
                ],
                "Uses Sequences": ["No", "No", "No", "Yes"],
            }
        )
        st.dataframe(results, use_container_width=True, hide_index=True)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                **Market Data**

                - 500M+ rows in the raw archive
                - 8.7M selected Nifty 50 database rows
                - 50 constituent stocks
                - 5-minute OHLCV candles
                - PostgreSQL storage
                """
            )
        with col2:
            st.markdown(
                """
                **Modeling**

                - 48-bar input sequence
                - 421 model features in the latest full run
                - Weighted binary cross-entropy
                - GPU training with mixed precision
                - Logistic regression and XGBoost baselines
                """
            )

        st.divider()
        st.subheader("Honest Limitations")
        st.markdown(
            """
            - Current-constituent selection introduces survivorship bias.
            - The evaluation partition is reused for early stopping.
            - Walk-forward validation has not yet been completed.
            - Transaction costs, slippage, and market impact are not modeled.
            - Classification quality does not establish deployable trading alpha.
            """
        )


    st.sidebar.title("Nifty 50 ML System")
    page = st.sidebar.radio(
        "Navigate",
        ["Model Performance", "Sentiment Analysis", "7-Day Report", "Project Summary"],
    )
    st.sidebar.caption("Research dashboard. Not financial advice.")

    metrics = load_json(METRICS_PATH)
    prediction = load_json(PREDICTION_PATH)
    require_artifact(metrics, METRICS_PATH, "Model metrics")

    if page == "Model Performance":
        render_model_performance(metrics)
    elif page == "Sentiment Analysis":
        render_sentiment_analysis()
    elif page == "7-Day Report":
        render_seven_day_report()
    else:
        render_project_summary(metrics, prediction)
