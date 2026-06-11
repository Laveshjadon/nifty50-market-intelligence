import os

import requests
import streamlit as st

from src.settings import get_settings


st.set_page_config(
    page_title="Nifty 50 Intelligence",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 15
SETTINGS = get_settings()

st.title("Nifty 50 Market Intelligence Dashboard")
st.markdown(
    "A real-time, comprehensive view of the Nifty 50 using machine learning and sentiment analysis."
)

st.sidebar.title("Agent Controls")
if st.sidebar.button("Generate New Report"):
    with st.spinner("Instructing the agent to build the report..."):
        try:
            headers = {"X-API-Key": SETTINGS.api_key} if SETTINGS.api_key else {}
            res = requests.post(
                f"{API_URL}/generate",
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if res.status_code == 200:
                st.sidebar.success(
                    "Started! This might take a few minutes. Refresh the page to see updates."
                )
            else:
                st.sidebar.error("Failed to start generation.")
        except requests.exceptions.ConnectionError:
            st.sidebar.error("Cannot connect to FastAPI backend. Is it running?")
        except requests.exceptions.Timeout:
            st.sidebar.error("The backend did not respond in time.")

st.sidebar.markdown("---")
st.sidebar.info(
    "The intelligence report aggregates the synthetic index, signals spanning 50 stocks, ML direction inference, and macroeconomic news sentiment."
)


@st.cache_data(ttl=60)
def fetch_report_markdown():
    try:
        r = requests.get(f"{API_URL}/report/markdown", timeout=REQUEST_TIMEOUT_SECONDS)
        if r.status_code == 200:
            return r.text
        return None
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return None


@st.cache_data(ttl=60)
def fetch_report_data():
    try:
        r = requests.get(f"{API_URL}/report/data", timeout=REQUEST_TIMEOUT_SECONDS)
        if r.status_code == 200:
            return r.json()
        return None
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return None


markdown_content = fetch_report_markdown()
json_data = fetch_report_data()

if not markdown_content and not json_data:
    st.warning(
        "Cannot connect to backend or report not generated yet. Start the FastAPI server and generate your first report."
    )
    st.stop()

tab1, tab2, tab3 = st.tabs(
    ["Quick Dashboard", "Full Markdown Report", "Raw Data (JSON)"]
)

with tab1:
    if json_data:
        pred = json_data.get("model_prediction", {})
        signals = json_data.get("signals", {})
        news = json_data.get("news_summary", {})

        direction = pred.get("predicted_direction", "N/A")
        confidence = pred.get("predicted_next_5m_return", 0.0)
        basket_close = pred.get("current_basket_close", "N/A")

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Direction (Next 5m)", direction, f"{confidence * 100:.3f}%")
        col2.metric(
            "Synthetic Basket Close",
            f"{basket_close:,.2f}" if isinstance(basket_close, float) else basket_close,
        )
        col3.metric(
            "Macro Sentiment Score",
            f"{news.get('macro_sentiment', 0.0) * 100:.2f}%",
        )

        st.markdown("---")

        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.subheader("Top Strongest Stocks")
            strongest = signals.get("top_strongest_stocks", [])
            for item in strongest[:5]:
                score = item.get("composite_score", item.get("strength_score", 0.0))
                st.write(f"**{item.get('ticker')}** - Score: {score:.2f}")

        with row1_col2:
            st.subheader("Top Weakest Stocks")
            weakest = signals.get("top_weakest_stocks", [])
            for item in weakest[:5]:
                score = item.get("composite_score", item.get("strength_score", 0.0))
                st.write(f"**{item.get('ticker')}** - Score: {score:.2f}")

        st.markdown("---")
        st.subheader("High-Risk Market News")
        high_risk = news.get("high_risk_negative_news", [])
        if not high_risk:
            st.success("No high-risk negative news detected currently.")
        else:
            for item in high_risk[:5]:
                st.warning(
                    f"**{item.get('title')}** _(Sentiment: {item.get('sentiment_score', 0.0):.2f})_"
                )
    else:
        st.info("Still waiting for raw JSON data structure over the network.")

with tab2:
    if markdown_content:
        st.markdown(markdown_content)
    else:
        st.info("The markdown report file wasn't found.")

with tab3:
    if json_data:
        st.json(json_data, expanded=False)
    else:
        st.info("No raw JSON data to display.")
