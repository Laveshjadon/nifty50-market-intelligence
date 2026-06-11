"""
report_agent.py - assemble a markdown market report from signals, model output, and news
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_pct(value: Any, decimals: int = 3) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except Exception:
        return "N/A"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _fmt_number(value: Any, decimals: int = 6) -> str:
    """Format a numeric metric safely for markdown output."""
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def _headline_list(records: list[dict[str, Any]], limit: int = 5) -> str:
    if not records:
        return "- None\n"
    lines = []
    for item in records[:limit]:
        title = str(item.get("title", "Untitled")).strip()
        source = str(item.get("source", "")).strip()
        score = _fmt_pct(item.get("sentiment_score", 0.0), decimals=2)
        lines.append(f"- {title} ({source or 'Unknown source'}, sentiment {score})")
    return "\n".join(lines) + "\n"


def build_recommendation(model_pred: dict[str, Any], news_summary: dict[str, Any]) -> tuple[str, str]:
    predicted_ret = _safe_float(model_pred.get("predicted_next_5m_return"))
    macro_sentiment = _safe_float(news_summary.get("macro_sentiment"))
    high_risk_count = len(news_summary.get("high_risk_negative_news", []))

    score = predicted_ret + (0.25 * macro_sentiment) - (0.05 * high_risk_count)
    if score >= 0.0025:
        return "Bullish bias", "Momentum and news backdrop lean constructive."
    if score <= -0.0025:
        return "Defensive bias", "Model direction or risk headlines argue for caution."
    return "Neutral / watchlist", "Signals are mixed, so waiting for confirmation is sensible."


def render_report(
    config: dict[str, Any],
    signals: dict[str, Any],
    model_pred: dict[str, Any],
    model_metrics: dict[str, Any],
    news_summary: dict[str, Any],
    comparison_report: dict[str, Any],
) -> str:
    recommendation, rationale = build_recommendation(model_pred, news_summary)
    strongest = signals.get("top_strongest_stocks", [])
    weakest = signals.get("top_weakest_stocks", [])
    sectors = signals.get("sector_strength_ranking", [])

    top_sector = sectors[0]["sector"] if sectors else "N/A"
    weak_sector = sectors[-1]["sector"] if sectors else "N/A"
    lstm_comparison = comparison_report.get("lstm", {})
    xgboost_comparison = comparison_report.get("xgboost", {})
    comparison_delta = comparison_report.get("delta_lstm_minus_xgboost", {})

    lines = [
        "# Nifty 50 Market Intelligence Report",
        "",
        f"Generated at: {signals.get('generated_at') or model_pred.get('latest_timestamp') or news_summary.get('latest_article_timestamp') or 'N/A'}",
        "",
        "## Summary",
        f"- Recommendation: **{recommendation}**",
        f"- Outlook: {rationale}",
        f"- Model next 5-minute direction: **{model_pred.get('predicted_direction', 'N/A')}** ({_fmt_pct(model_pred.get('predicted_next_5m_return'))})",
        f"- Macro news sentiment: {_fmt_pct(news_summary.get('macro_sentiment'))}",
        f"- High-risk negative news count: {len(news_summary.get('high_risk_negative_news', []))}",
        "",
        "## Market Outlook",
        f"- Current synthetic basket close: {model_pred.get('current_basket_close', 'N/A')}",
        f"- Implied next basket close: {model_pred.get('implied_next_basket_close', 'N/A')}",
        f"- Latest real index close: {model_pred.get('latest_real_index_close', 'N/A')}",
        f"- Basket directional accuracy on test set: {_fmt_pct(model_metrics.get('basket_directional_accuracy'))}",
        f"- Basket RMSE on test set: {model_metrics.get('basket_rmse', 'N/A')}",
        "",
        "## Baseline Comparison",
        f"- LSTM directional accuracy: {_fmt_pct(lstm_comparison.get('directional_accuracy'))}",
        f"- XGBoost directional accuracy: {_fmt_pct(xgboost_comparison.get('directional_accuracy'))}",
        f"- LSTM R2: {_fmt_number(lstm_comparison.get('r2'))}",
        f"- XGBoost R2: {_fmt_number(xgboost_comparison.get('r2'))}",
        f"- LSTM MSE: {_fmt_number(lstm_comparison.get('mse'), decimals=10)}",
        f"- XGBoost MSE: {_fmt_number(xgboost_comparison.get('mse'), decimals=10)}",
        f"- Directional accuracy delta: {_fmt_pct(comparison_delta.get('directional_accuracy'))}",
        "",
        "## Signals",
        f"- Strongest stocks: {', '.join(item.get('ticker', '') for item in strongest[:5]) or 'N/A'}",
        f"- Weakest stocks: {', '.join(item.get('ticker', '') for item in weakest[:5]) or 'N/A'}",
        f"- Strongest sector: {top_sector}",
        f"- Weakest sector: {weak_sector}",
        "",
        "## News",
        f"- Articles analyzed in lookback window: {news_summary.get('n_articles', 0)}",
        f"- Latest article timestamp: {news_summary.get('latest_article_timestamp', 'N/A')}",
        "",
        "### High-Risk Negative News",
        _headline_list(news_summary.get("high_risk_negative_news", []), limit=5),
        "### Most Negative Headlines",
        _headline_list(news_summary.get("top_negative_headlines", []), limit=5),
    ]
    return "\n".join(lines).rstrip() + "\n"


def save_report(markdown: str, config: dict[str, Any]) -> Path:
    output_path = PROJECT_ROOT / config["report"]["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def build_report_from_outputs(config: dict[str, Any] | None = None) -> tuple[str, Path]:
    config = config or load_config()
    signals = _load_json(PROJECT_ROOT / config["signals"]["output_path"])
    model_pred = _load_json(PROJECT_ROOT / config["modeling"]["output_latest_prediction_path"])
    model_metrics = _load_json(PROJECT_ROOT / config["modeling"]["output_metrics_path"])
    comparison_report = _load_json(
        PROJECT_ROOT / config["modeling"].get(
            "output_comparison_report_path",
            "output/models/comparison_report.json",
        )
    )
    news_summary = _load_json(PROJECT_ROOT / config["news"]["output_summary_json"])

    markdown = render_report(
        config,
        signals,
        model_pred,
        model_metrics,
        news_summary,
        comparison_report,
    )
    output_path = save_report(markdown, config)
    return markdown, output_path


if __name__ == "__main__":
    cfg = load_config()
    _, path = build_report_from_outputs(cfg)
    print(f"Saved report to: {path}")
