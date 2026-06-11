"""Score and categorize financial news for the model and report."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

POSITIVE_WORDS = {
    "beat", "beats", "growth", "surge", "strong", "gain", "gains", "rally",
    "bullish", "upgrade", "expansion", "profit", "profits", "record", "rise",
    "rises", "recovery", "improves", "improvement", "outperform", "wins",
}
NEGATIVE_WORDS = {
    "miss", "misses", "fall", "falls", "drop", "drops", "weak", "loss", "losses",
    "downgrade", "cut", "cuts", "crash", "crashes", "slump", "risk", "risks",
    "fraud", "probe", "lawsuit", "default", "decline", "warning", "warns",
    "negative", "poor", "unexpectedly", "npa", "inflation", "layoffs",
}



COMPANY_NAMES = {
    "ADANIENT":    ["adani enterprises", "adani group"],
    "ADANIPORTS":  ["adani ports", "adani port"],
    "APOLLOHOSP":  ["apollo hospital", "apollo health"],
    "ASIANPAINT":  ["asian paints", "asian paint"],
    "AXISBANK":    ["axis bank"],
    "BAJAJ-AUTO":  ["bajaj auto"],
    "BAJAJFINSV":  ["bajaj finserv"],
    "BAJFINANCE":  ["bajaj finance"],
    "BHARTIARTL":  ["bharti airtel", "airtel"],
    "BPCL":        ["bharat petroleum", "bpcl"],
    "BRITANNIA":   ["britannia"],
    "CIPLA":       ["cipla"],
    "COALINDIA":   ["coal india"],
    "DIVISLAB":    ["divi's lab", "divis lab"],
    "DRREDDY":     ["dr. reddy", "dr reddy", "drreddys"],
    "EICHERMOT":   ["eicher motors", "royal enfield"],
    "GRASIM":      ["grasim"],
    "HCLTECH":     ["hcl tech", "hcltech", "hcl technologies"],
    "HDFCBANK":    ["hdfc bank"],
    "HDFCLIFE":    ["hdfc life"],
    "HEROMOTOCO":  ["hero motocorp", "hero moto"],
    "HINDALCO":    ["hindalco"],
    "HINDUNILVR":  ["hindustan unilever", "hul"],
    "ICICIBANK":   ["icici bank"],
    "INDUSINDBK":  ["indusind bank"],
    "INFY":        ["infosys"],
    "ITC":         ["itc ltd", "itc limited"],
    "JSWSTEEL":    ["jsw steel"],
    "KOTAKBANK":   ["kotak bank", "kotak mahindra bank"],
    "LT":          ["larsen & toubro", "l&t"],
    "LTIM":        ["ltimindtree", "lti mindtree"],
    "MARUTI":      ["maruti suzuki", "maruti"],
    "MM":          ["mahindra & mahindra", "m&m"],
    "NESTLEIND":   ["nestle india", "nestle"],
    "NTPC":        ["ntpc"],
    "ONGC":        ["ongc"],
    "POWERGRID":   ["power grid", "powergrid"],
    "RELIANCE":    ["reliance industries", "reliance jio", "jio"],
    "SBILIFE":     ["sbi life"],
    "SBIN":        ["state bank of india", "sbi"],
    "SUNPHARMA":   ["sun pharma", "sun pharmaceutical"],
    "TATACONSUM":  ["tata consumer", "tata tea"],
    "TATAMOTORS":  ["tata motors", "jaguar land rover", "jlr"],
    "TATASTEEL":   ["tata steel"],
    "TCS":         ["tata consultancy", "tcs"],
    "TECHM":       ["tech mahindra"],
    "TITAN":       ["titan company", "tanishq"],
    "ULTRACEMCO":  ["ultratech cement"],
    "UPL":         ["upl limited", "upl ltd"],
    "WIPRO":       ["wipro"],
}

SECTOR_KEYWORDS = {
    "banking_finance": ["rbi", "repo rate", "monetary policy", "npa", "banking sector", "nbfc"],
    "it_tech": ["it sector", "software exports", "tech layoffs", "digital transformation", "ai deal"],
    "pharma_health": ["pharma", "fda approval", "usfda", "drug approval", "healthcare"],
    "auto": ["auto sales", "electric vehicle", "fame scheme", "automobile", "ev policy"],
    "energy_oil": ["crude oil", "brent", "petrol price", "renewable energy", "power tariff"],
    "metals_mining": ["steel price", "iron ore", "metal sector", "mining"],
    "fmcg_consumer": ["fmcg", "consumer goods", "rural demand", "inflation impact consumer"],
    "cement_infra": ["cement", "infrastructure", "highways", "capital expenditure"],
    "telecom": ["5g rollout", "telecom sector", "arpu", "telecom tariff"],
}

MACRO_KEYWORDS = ["union budget", "fiscal deficit", "inflation", "gdp", "cpi", "wpi", "fii", "dii"]


def clean_json(value: Any) -> Any:
    """Replace invalid float values before saving JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    return value


class NewsSignalProcessor:
    def __init__(self, use_gpu: bool = True):
        self.use_finbert = False
        self.sentiment_pipe = None
        finbert_enabled = os.getenv("USE_FINBERT", "").strip().lower() in {"1", "true", "yes"}

        if use_gpu and torch.cuda.is_available():
            self.device = 0
            logger.info(f"Using GPU for FinBERT: {torch.cuda.get_device_name(0)}")
        else:
            self.device = -1
            logger.info("Using CPU for FinBERT.")

        if finbert_enabled:
            try:
                logger.info("Loading ProsusAI/finbert sentiment pipeline...")
                self.sentiment_pipe = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    device=self.device,
                    truncation=True,
                    max_length=512,
                    local_files_only=True,
                )
                self.use_finbert = True
            except Exception as exc:
                logger.warning(
                    "Falling back to keyword sentiment because FinBERT is unavailable: %s",
                    exc,
                )
        else:
            logger.info("USE_FINBERT is disabled; using keyword sentiment fallback.")

    def load_config(self) -> dict:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def categorize_news(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds category tags for ticker, sector, and macro.
        """
        logger.info(f"Categorizing {len(df)} news items...")
        df = df.copy()

        def _text_col(name: str) -> pd.Series:
            if name in df.columns:
                return df[name].fillna("").astype(str)
            return pd.Series("", index=df.index, dtype="object")


        text_series = (_text_col("title") + " " + _text_col("description")).str.lower()

        def get_matches(keyword_dict: Dict[str, List[str]], text_series: pd.Series) -> pd.Series:
            matches = pd.Series([[]] * len(text_series), index=text_series.index)
            for category, words in keyword_dict.items():
                pattern = "|".join(rf"\b{re.escape(w)}\b" for w in words)
                matched_idx = text_series[text_series.str.contains(pattern, regex=True)].index
                for idx in matched_idx:
                    matches.loc[idx] = matches.loc[idx] + [category]
            return matches.apply(lambda x: ",".join(set(x)) if x else "")

        df["tickers_matched"] = get_matches(COMPANY_NAMES, text_series)
        df["sectors_matched"] = get_matches(SECTOR_KEYWORDS, text_series)

        macro_pattern = "|".join(rf"\b{re.escape(w)}\b" for w in MACRO_KEYWORDS)
        df["is_macro"] = text_series.str.contains(macro_pattern, regex=True)

        return df

    def compute_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate probability of positive, negative, and neutral sentiment using FinBERT.
        Converts it into a continuous score: P(pos) - P(neg).
        """
        logger.info(
            "Computing sentiment via %s...",
            "FinBERT" if self.use_finbert else "keyword fallback",
        )
        df = df.copy()


        inference_texts = df["title"].fillna(df["description"]).fillna("").tolist()

        if not inference_texts:
            return df

        if self.use_finbert and self.sentiment_pipe is not None:
            results = self.sentiment_pipe(inference_texts, batch_size=32)

            sentiments = []
            scores = []
            derived_scores = []

            for res in results:
                label = res["label"].lower()
                conf = res["score"]
                sentiments.append(label)
                scores.append(conf)

                if label == "positive":
                    derived_scores.append(conf)
                elif label == "negative":
                    derived_scores.append(-conf)
                else:
                    derived_scores.append(0.0)
        else:
            sentiments = []
            scores = []
            derived_scores = []
            for text in inference_texts:
                label, conf, score = self._keyword_sentiment(text)
                sentiments.append(label)
                scores.append(conf)
                derived_scores.append(score)

        df["sentiment_label"] = sentiments
        df["sentiment_confidence"] = scores
        df["sentiment_score"] = derived_scores

        return df

    def _keyword_sentiment(self, text: str) -> tuple[str, float, float]:
        tokens = re.findall(r"[a-zA-Z']+", str(text).lower())
        if not tokens:
            return "neutral", 0.0, 0.0

        pos = sum(token in POSITIVE_WORDS for token in tokens)
        neg = sum(token in NEGATIVE_WORDS for token in tokens)
        total = pos + neg
        if total == 0:
            return "neutral", 0.15, 0.0

        raw = (pos - neg) / total
        score = float(np.clip(raw, -1.0, 1.0))
        confidence = float(min(0.95, 0.35 + (total * 0.08)))
        if score > 0.05:
            return "positive", confidence, score
        if score < -0.05:
            return "negative", confidence, score
        return "neutral", confidence, 0.0

    def aggregate_signals(self, df: pd.DataFrame) -> dict:
        """
        Aggregates recent news into a concise 'News Signal'
        for Report Generation / Modeling.
        """
        logger.info("Aggregating sentiment signals...")

        stock_scores = {}
        for idx, row in df.iterrows():
            if row["tickers_matched"]:
                for t in row["tickers_matched"].split(","):
                    stock_scores.setdefault(t, []).append(row["sentiment_score"])

        stock_avg_sentiment = {k: float(np.mean(v)) for k, v in stock_scores.items()}

        sector_scores = {}
        for idx, row in df.iterrows():
            if row["sectors_matched"]:
                for s in row["sectors_matched"].split(","):
                    sector_scores.setdefault(s, []).append(row["sentiment_score"])

        sector_avg_sentiment = {k: float(np.mean(v)) for k, v in sector_scores.items()}

        macro_df = df[df["is_macro"]]
        macro_sentiment = float(macro_df["sentiment_score"].mean()) if not macro_df.empty else 0.0

        return {
            "stock_sentiment": stock_avg_sentiment,
            "sector_sentiment": sector_avg_sentiment,
            "macro_sentiment": macro_sentiment
        }

    def process(self, raw_news_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Full pipeline to convert raw news into sentiment-scored df and aggregate signals.
        """
        df = self.categorize_news(raw_news_df)
        df = self.compute_sentiment(df)
        signals = self.aggregate_signals(df)
        return df, signals


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _pick_latest_existing_file(config: dict[str, Any]) -> Path:
    news_cfg = config.get("news", {})
    data_dir = PROJECT_ROOT / news_cfg.get("data_dir", "news_data")
    preferred_files = news_cfg.get("preferred_files", [])

    for filename in preferred_files:
        candidate = data_dir / filename
        if candidate.exists():
            return candidate

    csv_candidates = sorted(data_dir.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not csv_candidates:
        raise FileNotFoundError(f"No CSV news files found in {data_dir}")
    return csv_candidates[0]


def load_latest_news_frame(config: dict[str, Any]) -> pd.DataFrame:
    path = _pick_latest_existing_file(config)
    logger.info("Loading news data from %s", path)
    df = pd.read_csv(path)
    if df.empty:
        return df

    column_map = {c.lower().strip(): c for c in df.columns}
    rename_map = {}
    if "headline" in column_map:
        rename_map[column_map["headline"]] = "title"
    if "headline link" in column_map:
        rename_map[column_map["headline link"]] = "url"
    if "link" in column_map:
        rename_map[column_map["link"]] = "url"
    if "archive" in column_map and "source" not in column_map:
        rename_map[column_map["archive"]] = "source"
    if "source" in column_map:
        rename_map[column_map["source"]] = "source"
    if "date" in column_map:
        rename_map[column_map["date"]] = "date"
    if "description" in column_map:
        rename_map[column_map["description"]] = "description"
    if "query" in column_map:
        rename_map[column_map["query"]] = "query"

    df = df.rename(columns=rename_map)

    for required in ["title", "description", "source", "url"]:
        if required not in df.columns:
            df[required] = ""

    if "date" not in df.columns:
        if "fetched_at" in df.columns:
            df["date"] = df["fetched_at"]
        else:
            df["date"] = pd.Timestamp.utcnow()

    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str).str.strip()
    df["source"] = df["source"].fillna("").astype(str).str.strip()
    df["url"] = df["url"].fillna("").astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["date"])

    dedupe_key = np.where(
        df["url"].str.len() > 0,
        df["url"].str.lower(),
        (df["date"].dt.strftime("%Y-%m-%d") + "|" + df["title"].str.lower()),
    )
    df["_dedupe_key"] = dedupe_key
    df = df.drop_duplicates(subset=["_dedupe_key"], keep="last").drop(columns=["_dedupe_key"])
    return df.sort_values("date").reset_index(drop=True)


def summarize_recent_news(scored_df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    news_cfg = config.get("news", {})
    threshold = float(news_cfg.get("high_risk_negative_threshold", -0.35))
    top_negative_count = int(news_cfg.get("top_negative_count", 8))

    if scored_df.empty:
        return {
            "lookback_days": int(news_cfg.get("lookback_days", 7)),
            "n_articles": 0,
            "stock_sentiment": {},
            "sector_sentiment": {},
            "macro_sentiment": 0.0,
            "high_risk_negative_news": [],
            "top_negative_headlines": [],
            "latest_article_timestamp": None,
        }

    processor = NewsSignalProcessor(use_gpu=False)
    aggregate = processor.aggregate_signals(scored_df)

    negative_df = (
        scored_df[scored_df["sentiment_score"] <= threshold]
        .sort_values(["sentiment_score", "date"], ascending=[True, False])
        .copy()
    )

    top_negative_df = (
        scored_df.sort_values(["sentiment_score", "date"], ascending=[True, False])
        .head(top_negative_count)
        .copy()
    )

    def to_records(frame: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
        if frame.empty:
            return []
        cols = ["date", "title", "source", "sentiment_score", "tickers_matched", "sectors_matched", "url"]
        present = [c for c in cols if c in frame.columns]
        out = frame[present].head(limit).copy()
        out["date"] = out["date"].astype(str)
        return out.to_dict(orient="records")

    return {
        "lookback_days": int(news_cfg.get("lookback_days", 7)),
        "n_articles": int(len(scored_df)),
        "stock_sentiment": aggregate["stock_sentiment"],
        "sector_sentiment": aggregate["sector_sentiment"],
        "macro_sentiment": aggregate["macro_sentiment"],
        "high_risk_negative_news": to_records(negative_df, top_negative_count),
        "top_negative_headlines": to_records(top_negative_df, top_negative_count),
        "latest_article_timestamp": str(scored_df["date"].max()),
    }


def process_recent_news(config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = config or load_config()
    news_cfg = config.get("news", {})
    lookback_days = int(news_cfg.get("lookback_days", 7))

    raw_df = load_latest_news_frame(config)
    if raw_df.empty:
        summary = summarize_recent_news(raw_df, config)
        return raw_df, summary

    cutoff = raw_df["date"].max() - pd.Timedelta(days=lookback_days)
    recent_df = raw_df[raw_df["date"] >= cutoff].copy()

    processor = NewsSignalProcessor(use_gpu=False)
    scored_df, _ = processor.process(recent_df)
    summary = summarize_recent_news(scored_df, config)
    return scored_df, summary


def save_news_outputs(
    scored_df: pd.DataFrame,
    summary: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    config = config or load_config()
    news_cfg = config.get("news", {})
    processed_path = PROJECT_ROOT / news_cfg.get("output_processed_csv", "output/news/latest_news_scored.csv")
    summary_path = PROJECT_ROOT / news_cfg.get("output_summary_json", "output/news/latest_news_summary.json")

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    scored_df.to_csv(processed_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(clean_json(summary), f, indent=2, allow_nan=False)

    return processed_path, summary_path

if __name__ == "__main__":
    cfg = load_config()
    scored_df, summary = process_recent_news(cfg)
    processed_path, summary_path = save_news_outputs(scored_df, summary, cfg)
    print(f"Saved scored news to: {processed_path}")
    print(f"Saved news summary to: {summary_path}")
