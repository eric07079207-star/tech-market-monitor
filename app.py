from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:  # pragma: no cover - optional local dependency
    st_autorefresh = None

try:
    from src.ai_summary import ai_summary_quality, latest_ai_history_entry, load_ai_summary_history, load_cached_ai_summary, openai_configuration_status
except ImportError:  # pragma: no cover - protects Streamlit Cloud during partial redeploys
    def load_cached_ai_summary(path=None) -> dict:
        return {}

    def ai_summary_quality(payload: dict) -> dict:
        return {
            "quality_score": 0,
            "quality_label": "資料同步中",
            "text_length": 0,
            "section_count": 0,
            "required_sections": 6,
            "missing_sections": "等待模組同步",
        }
    def openai_configuration_status() -> dict:
        return {"configured": False, "status": "missing_key", "model": "n/a", "api_key_preview": "n/a"}
    def load_ai_summary_history(path=None) -> pd.DataFrame:
        return pd.DataFrame()

    def latest_ai_history_entry(history: pd.DataFrame | None = None) -> dict:
        return {}
from src.config import ETF_TICKERS, NEWS_QUERIES, STOCK_TICKERS, default_start_date
try:
    from src.config import ANNUAL_PICK_TICKERS
except ImportError:  # pragma: no cover - protects Streamlit Cloud during partial redeploys
    ANNUAL_PICK_TICKERS = ["TSLA", "PLTR", "CRWD", "VST", "RKLB", "IONQ", "OKLO", "SOFI", "HOOD", "TMDX"]
try:
    from src.annual_picks import annual_picks_summary, annual_picks_table
except ImportError:  # pragma: no cover - protects Streamlit Cloud during partial redeploys
    def annual_picks_table(prices: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"ticker": ANNUAL_PICK_TICKERS})

    def annual_picks_summary(table: pd.DataFrame) -> dict:
        return {"avg_return": np.nan, "win_rate": np.nan, "best": "n/a", "worst": "n/a", "avg_rel_qqq": np.nan}
try:
    from src.data import MACRO_CACHE, PRICE_CACHE, cache_path, load_metadata
except ImportError:  # pragma: no cover - protects Streamlit Cloud during partial redeploys
    from pathlib import Path

    def cache_path(name: str) -> Path:
        return Path("data/cache") / name

    PRICE_CACHE = cache_path("prices.parquet")
    MACRO_CACHE = cache_path("macro.parquet")

    def load_metadata(path=None) -> dict:
        return {}
try:
    from src.discovery import discovery_performance_summary, load_discovery_history, load_discovery_performance, summarize_discovery_history
except ImportError:  # pragma: no cover - protects Streamlit Cloud during partial redeploys
    def load_discovery_history(path=None) -> pd.DataFrame:
        return pd.DataFrame()

    def summarize_discovery_history(history: pd.DataFrame, days: int, top_n: int = 15) -> pd.DataFrame:
        return pd.DataFrame()

    def load_discovery_performance(path=None) -> pd.DataFrame:
        return pd.DataFrame()

    def discovery_performance_summary(performance: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()
try:
    from src.kg import kg_summary, load_knowledge_graph
    from src.kg_predictions import kg_prediction_summary, load_kg_prediction_log
    from src.kg_predictions_v2 import kg_prediction_v2_summary, load_kg_prediction_v2_log
except ImportError:  # pragma: no cover - protects Streamlit Cloud during partial redeploys
    def load_knowledge_graph():
        from collections import namedtuple

        KGOutput = namedtuple("KGOutput", "facts narratives reactions links")
        empty = pd.DataFrame()
        return KGOutput(empty, empty, empty, empty)

    def kg_summary(payload) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"層級": "事實層", "筆數": 0, "最新時間": "n/a", "說明": "等待模組同步"},
                {"層級": "敘事層", "筆數": 0, "最新時間": "n/a", "說明": "等待模組同步"},
                {"層級": "反應層", "筆數": 0, "最新時間": "n/a", "說明": "等待模組同步"},
                {"層級": "連結層", "筆數": 0, "最新時間": "n/a", "說明": "等待模組同步"},
            ]
        )

    def load_kg_prediction_log() -> pd.DataFrame:
        return pd.DataFrame()

    def kg_prediction_summary(log: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()

    def load_kg_prediction_v2_log() -> pd.DataFrame:
        return pd.DataFrame()

    def kg_prediction_v2_summary(log: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()
try:
    from src.historical_backtest import historical_backtest_summary, load_stratified_market_samples
except ImportError:  # pragma: no cover - protects Streamlit Cloud during partial redeploys
    def load_stratified_market_samples() -> pd.DataFrame:
        return pd.DataFrame()

    def historical_backtest_summary(samples: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()
try:
    from src.factor_effectiveness import factor_effectiveness_summary, load_factor_effectiveness
    from src.kg_backtest import kg_backtest_readiness_summary, load_kg_backtest_readiness
except ImportError:  # pragma: no cover - protects Streamlit Cloud during partial redeploys
    def load_factor_effectiveness() -> pd.DataFrame:
        return pd.DataFrame()

    def factor_effectiveness_summary(report: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()

    def load_kg_backtest_readiness() -> pd.DataFrame:
        return pd.DataFrame()

    def kg_backtest_readiness_summary(report: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()
try:
    from src.fundamentals import load_fundamental_observations
except ImportError:  # pragma: no cover - protects Streamlit Cloud during partial redeploys
    def load_fundamental_observations() -> pd.DataFrame:
        return pd.DataFrame()
try:
    from src.governance import governance_alerts
except ImportError:  # pragma: no cover - protects Streamlit Cloud during partial redeploys
    def governance_alerts(summary: pd.DataFrame) -> list[dict[str, str]]:
        return []
try:
    from src.project_memory import MEMORY_DOCX_FILE, load_memory_bundle
except ImportError:  # pragma: no cover - protects Streamlit Cloud during partial redeploys
    from pathlib import Path

    MEMORY_DOCX_FILE = Path("data/觀察版資料/13_專案記憶/專案記憶與討論摘要.docx")

    def load_memory_bundle():
        class _FallbackMemoryBundle:
            project_memory = "專案記憶模組同步中。"
            conversation_log = "討論摘要模組同步中。"
            active_context = "當前上下文模組同步中。"
            decision_register = pd.DataFrame()
            memory_changelog = pd.DataFrame()
            status_table = pd.DataFrame([{"項目": "專案記憶", "數值": "同步中", "說明": "等待雲端部署完成"}])
            latest_updates = pd.DataFrame()
            directory = Path("data/觀察版資料/13_專案記憶")

        return _FallbackMemoryBundle()
try:
    from src.health import data_health_report, missing_price_symbols
except ImportError:  # pragma: no cover - protects Streamlit Cloud during partial redeploys
    def data_health_report(*args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame([{"資料項目": "資料同步中", "筆數": 0, "最新日期": "n/a", "說明": "等待雲端部署完成"}])

    def missing_price_symbols(prices: pd.DataFrame, symbols: list[str]) -> list[str]:
        return []
try:
    from src.sentiment import EVENT_WINDOWS_CACHE, SENTIMENT_CACHE
except ImportError:  # pragma: no cover - protects Streamlit Cloud during partial redeploys
    SENTIMENT_CACHE = cache_path("sentiment.parquet")
    EVENT_WINDOWS_CACHE = cache_path("market_event_windows.parquet")
try:
    from src.emotion import emotion_alerts, emotion_components, emotion_divergence, emotion_trend, fear_greed_analysis, latest_emotion
except ImportError:  # pragma: no cover - protects Streamlit Cloud during partial redeploys
    def latest_emotion(sentiment: pd.DataFrame) -> dict:
        return {}

    def emotion_components(row: dict) -> pd.DataFrame:
        return pd.DataFrame()

    def emotion_trend(sentiment: pd.DataFrame, days: int = 120) -> pd.DataFrame:
        return pd.DataFrame()

    def emotion_divergence(prices: pd.DataFrame, sentiment: pd.DataFrame, symbols: list[str] | None = None) -> pd.DataFrame:
        return pd.DataFrame()

    def emotion_alerts(row: dict) -> list[dict]:
        return []

    def fear_greed_analysis(row: dict) -> dict:
        return {"score": np.nan, "label": "資料不足", "confidence": 0.0, "components": pd.DataFrame()}
try:
    from src.edge import summarize_quality_frame
except ImportError:  # pragma: no cover - protects Streamlit Cloud during partial redeploys
    def summarize_quality_frame(data: pd.DataFrame, score_column: str = "quality_score") -> dict:
        return {"rows": 0, "avg_quality": np.nan, "median_quality": np.nan, "low_quality_rows": 0, "unique_sources": 0, "dup_ratio": np.nan}
try:
    from src.lstm import build_lstm_status_from_artifacts, load_lstm_backtest, load_lstm_predictions, load_lstm_status, summarize_lstm_status
except ImportError:  # pragma: no cover - protects Streamlit Cloud during partial redeploys
    def load_lstm_status(path=None) -> dict:
        return {
            "enabled": False,
            "mode": "scaffold",
            "status": "LSTM 模組尚未同步",
            "model_version": "n/a",
            "feature_version": "n/a",
            "last_train_at_utc": "",
            "last_predict_at_utc": "",
            "last_backtest_at_utc": "",
            "prediction_rows": 0,
            "backtest_rows": 0,
            "updated_at_utc": "",
        }

    def load_lstm_predictions(path=None) -> pd.DataFrame:
        return pd.DataFrame()

    def load_lstm_backtest(path=None) -> pd.DataFrame:
        return pd.DataFrame()

    def build_lstm_status_from_artifacts(*args, **kwargs) -> dict:
        return load_lstm_status()

    def summarize_lstm_status(status: dict) -> pd.DataFrame:
        return pd.DataFrame([{"項目": "LSTM 模組", "值": "等待同步"}])
from src.indicators import (
    add_price_indicators,
    analog_stats,
    breadth_table,
    detect_anomalies,
    historical_analogs,
    latest_snapshot,
    regime_summary,
)
try:
    from src.indicators import analog_interpretation, categorize_anomalies, conclusion_cards, risk_clue_table, today_conclusion
except ImportError:  # pragma: no cover - protects Streamlit Cloud during partial redeploys
    def analog_interpretation(stats: pd.DataFrame) -> str:
        return "雲端模組正在同步，暫時顯示基礎相似情境。"

    def categorize_anomalies(anomalies: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return {"價格異常": anomalies, "成交量異常": pd.DataFrame(), "趨勢異常": pd.DataFrame()}

    def conclusion_cards(regime: dict, conclusion: dict, market_prediction: dict, anomalies: pd.DataFrame) -> list[dict]:
        return [
            {"title": "市場狀態", "value": conclusion.get("label", "資料同步中"), "detail": "等待雲端部署完成"},
            {"title": "建議行為", "value": "先觀察", "detail": "基礎模式"},
            {"title": "信心等級", "value": conclusion.get("confidence", "低"), "detail": ""},
            {"title": "最大風險", "value": "資料同步", "detail": ""},
        ]

    def risk_clue_table(indicators: pd.DataFrame, macro: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "indicator": "資料同步中",
                    "current": "n/a",
                    "risk_threshold": "等待雲端部署完成",
                    "status": "未觸發",
                    "implication": "Streamlit Cloud 正在更新模組，稍後會恢復完整風險線索。",
                }
            ]
        )

    def today_conclusion(regime: dict, snapshot: pd.DataFrame, anomalies: pd.DataFrame) -> dict:
        return {
            "label": regime.get("label", "資料同步中"),
            "sentence": "雲端模組正在同步，先顯示基礎市場狀態。",
            "confidence": "低",
        }

try:
    from src.news import (
        DEFAULT_TSLA_KEYWORDS,
        fetch_news_batch,
        international_news_selection,
        portfolio_news_impact,
        summarize_keyword_news,
    )
except ImportError:  # pragma: no cover - protects Streamlit Cloud during partial redeploys
    DEFAULT_TSLA_KEYWORDS = [
        "TSLA",
        "Tesla",
        "Elon Musk",
        "Robotaxi",
        "FSD",
        "Autopilot",
        "Cybertruck",
        "Model 3",
        "Model Y",
        "Dojo",
        "Optimus",
        "Gigafactory",
        "Megapack",
        "energy storage",
        "deliveries",
        "price cut",
        "margin pressure",
        "recall",
        "investigation",
        "lawsuit",
    ]

    def fetch_news_batch(symbols: list[str] | None = None, days: int = 7, limit_per_symbol: int = 6) -> pd.DataFrame:
        return pd.DataFrame(columns=["symbol", "title", "source", "source_domain", "source_reliability_score", "published", "tags", "link", "quality_score"])

    def summarize_keyword_news(news: pd.DataFrame, symbol: str = "TSLA") -> dict:
        return {"symbol": symbol, "headline_count": 0, "top_keywords": "", "top_groups": "", "risk_keywords": "", "latest_published": "", "summary": f"近期沒有抓到 {symbol} 關鍵字命中的新聞。"}

    def international_news_selection(news: pd.DataFrame, random_count: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
        return news.copy(), news.copy()

    def portfolio_news_impact(news: pd.DataFrame, portfolio_view: pd.DataFrame | None = None, max_items: int = 8) -> pd.DataFrame:
        return pd.DataFrame(columns=["ticker", "impact_level", "impact", "headline_count", "key_tags", "sample_headline"])

from src.portfolio import bucket_guidelines, bucket_summary, build_portfolio_view, fetch_portfolio_prices, load_portfolio_config
try:
    from src.portfolio import attention_positions
except ImportError:  # pragma: no cover - protects Streamlit Cloud during partial redeploys
    def attention_positions(view: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
        return pd.DataFrame()
try:
    from src.predictions import build_market_prediction, load_prediction_log, prediction_scorecard, prediction_validation_summary, recent_prediction_table
except ImportError:  # pragma: no cover - protects Streamlit Cloud during partial redeploys
    def build_market_prediction(regime: dict, conclusion: dict, snapshot: pd.DataFrame) -> dict:
        return {"target": "QQQ", "prediction_direction": "資料同步中"}

    def load_prediction_log() -> pd.DataFrame:
        return pd.DataFrame()

    def prediction_validation_summary(log: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()

    def prediction_scorecard(log: pd.DataFrame) -> dict:
        return {"validated": 0, "success_rate": np.nan, "avg_return": np.nan, "best_segment": "n/a", "weak_segment": "n/a"}

    def recent_prediction_table(log: pd.DataFrame, limit: int = 30) -> pd.DataFrame:
        return pd.DataFrame()


st.set_page_config(page_title="Tech Market Monitor", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #17212b;
        --muted: #66788a;
        --line: #d6e0e8;
        --canvas: #f3f6f8;
        --panel: #ffffff;
        --navy: #142b3c;
        --blue: #1976b9;
        --teal: #0c8578;
        --green: #16803b;
        --amber: #b76909;
        --red: #c0284b;
    }
    .stApp {background: var(--canvas); color: var(--ink);}
    .block-container {max-width: 1520px; padding-top: 1.15rem; padding-bottom: 2.5rem;}
    h1, h2, h3 {color: var(--ink); letter-spacing: 0 !important;}
    h2 {font-size: 1.45rem; margin-top: 1.75rem; border-bottom: 1px solid var(--line); padding-bottom: 0.55rem;}
    h3 {font-size: 1.08rem; margin-top: 1.35rem;}
    [data-testid="stSidebar"] {background: var(--navy); border-right: 1px solid #29465a;}
    [data-testid="stSidebar"] * {color: #edf5f8;}
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {color: #b9cbd6;}
    [data-testid="stSidebar"] hr {border-color: #385469;}
    [data-testid="stSidebar"] button[kind="secondary"] {background: #215571; border-color: #4c829e; color: #ffffff;}
    [data-testid="stSidebar"] button[kind="secondary"]:hover {background: #2d6c8d; border-color: #88b7cb;}
    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-top: 4px solid var(--blue);
        border-radius: 7px;
        padding: 13px 15px;
        min-height: 104px;
        box-shadow: 0 1px 2px rgba(20, 43, 60, 0.05);
    }
    [data-testid="stMetricLabel"] {color: var(--muted); font-size: 0.79rem; font-weight: 700;}
    [data-testid="stMetricValue"] {color: var(--ink); font-weight: 760;}
    .terminal-hero {background: var(--navy); border: 1px solid #294b61; border-radius: 8px; padding: 1.35rem 1.5rem; margin-bottom: 1rem; box-shadow: 0 3px 10px rgba(20, 43, 60, 0.16);}
    .terminal-eyebrow {color: #89cce1; font-size: 0.75rem; font-weight: 750; letter-spacing: 0.08em; margin-bottom: 0.4rem;}
    .terminal-title {color: #ffffff; font-size: 2rem; font-weight: 780; line-height: 1.15; margin: 0;}
    .terminal-subtitle {color: #c6d7e0; font-size: 0.9rem; margin: 0.45rem 0 0;}
    .terminal-badges {display: flex; gap: 0.45rem; flex-wrap: wrap; justify-content: flex-end; align-content: center; height: 100%;}
    .terminal-badge {background: #21475e; color: #dceef4; border: 1px solid #416b81; border-radius: 999px; font-size: 0.76rem; font-weight: 650; padding: 0.34rem 0.6rem; white-space: nowrap;}
    .terminal-badge.live {background: #0d5d56; border-color: #32988d; color: #d9fffa;}
    .small-muted {color: var(--muted); font-size: 0.82rem;}
    .insight-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-left: 4px solid var(--teal);
        border-radius: 7px;
        padding: 14px 16px;
        min-height: 122px;
        box-shadow: 0 1px 2px rgba(20, 43, 60, 0.05);
    }
    .insight-card .label {color: var(--muted); font-size: 0.76rem; font-weight: 750; margin-bottom: 8px; text-transform: uppercase;}
    .insight-card .value {font-size: 1.1rem; font-weight: 760; line-height: 1.28; color: var(--ink);}
    .insight-card .detail {color: #496074; font-size: 0.83rem; margin-top: 8px; line-height: 1.4;}
    .stTabs [data-baseweb="tab-list"] {gap: 0.2rem; border-bottom: 1px solid var(--line); overflow-x: auto;}
    .stTabs [data-baseweb="tab"] {height: 2.8rem; padding: 0 0.82rem; color: #536b7b; font-size: 0.86rem; font-weight: 700; white-space: nowrap;}
    .stTabs [aria-selected="true"] {color: var(--blue);}
    .stTabs [data-baseweb="tab-highlight"] {background-color: var(--blue); height: 3px;}
    [data-testid="stDataFrame"] {border: 1px solid var(--line); border-radius: 7px; overflow: hidden; background: var(--panel);}
    [data-testid="stDataFrame"] [role="columnheader"] {background: #eaf1f5; color: #324e62; font-weight: 750;}
    [data-testid="stExpander"] {background: var(--panel); border: 1px solid var(--line); border-radius: 7px;}
    [data-testid="stAlert"] {border-radius: 7px;}
    button[kind="secondary"] {border-radius: 6px; border-color: #b7c8d4; color: #294b61; font-weight: 700;}
    button[kind="secondary"]:hover {border-color: var(--blue); color: var(--blue);}
    .light-green {color: var(--green); font-weight: 700;}
    .light-yellow {color: var(--amber); font-weight: 700;}
    .light-red {color: var(--red); font-weight: 700;}
    @media (max-width: 760px) {
        .block-container {padding: 0.75rem 0.7rem 1.75rem;}
        .terminal-hero {padding: 1.05rem 1rem;}
        .terminal-title {font-size: 1.55rem;}
        .terminal-badges {justify-content: flex-start; margin-top: 0.9rem;}
        div[data-testid="stMetric"] {min-height: 88px; padding: 10px 11px;}
        [data-testid="stMetricValue"] {font-size: 1.35rem;}
        .stTabs [data-baseweb="tab"] {padding: 0 0.62rem; font-size: 0.8rem;}
        .insight-card {min-height: 105px; padding: 12px 13px;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def load_market(start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PRICE_CACHE.exists():
        return pd.DataFrame(), pd.DataFrame()
    prices = pd.read_parquet(PRICE_CACHE)
    prices["date"] = pd.to_datetime(prices["date"])
    if start:
        prices = prices[prices["date"] >= pd.to_datetime(start)]
    macro = pd.read_parquet(MACRO_CACHE) if MACRO_CACHE.exists() else pd.DataFrame(columns=["date", "series", "label", "value"])
    if not macro.empty and "date" in macro:
        macro["date"] = pd.to_datetime(macro["date"])
    return prices, macro


@st.cache_data(show_spinner=False, ttl=60 * 60 * 3)
def load_news(days: int) -> pd.DataFrame:
    news_path = cache_path("news.parquet")
    if not news_path.exists():
        return pd.DataFrame(columns=["symbol", "title", "source", "published", "tags", "link"])
    news = pd.read_parquet(news_path)
    news["published"] = pd.to_datetime(news["published"], utc=True, errors="coerce")
    return news


@st.cache_data(show_spinner=False, ttl=60 * 60 * 3)
def load_international_news(days: int) -> pd.DataFrame:
    news_path = cache_path("international_news.parquet")
    if not news_path.exists():
        return pd.DataFrame(columns=["symbol", "title", "source", "published", "tags", "link", "is_major", "priority"])
    news = pd.read_parquet(news_path)
    news["published"] = pd.to_datetime(news["published"], utc=True, errors="coerce")
    return news


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def load_discovery() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    news_path = cache_path("discovery_news.parquet")
    candidates_path = cache_path("discovery_candidates.parquet")
    mentions_path = cache_path("discovery_mentions.parquet")
    discovery_news = pd.read_parquet(news_path) if news_path.exists() else pd.DataFrame()
    candidates = pd.read_parquet(candidates_path) if candidates_path.exists() else pd.DataFrame()
    mentions = pd.read_parquet(mentions_path) if mentions_path.exists() else pd.DataFrame()
    history = load_discovery_history()
    for data, column in [(discovery_news, "published"), (mentions, "published")]:
        if not data.empty and column in data:
            data[column] = pd.to_datetime(data[column], utc=True, errors="coerce")
    return discovery_news, mentions, candidates, history


@st.cache_data(show_spinner=False, ttl=60 * 60 * 3)
def load_tsla_keyword_news() -> pd.DataFrame:
    path = cache_path("tsla_keyword_news.parquet")
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_parquet(path)
    if "published" in data:
        data["published"] = pd.to_datetime(data["published"], utc=True, errors="coerce")
    return data


@st.cache_data(show_spinner=False, ttl=60 * 60 * 3)
def load_governance_summary() -> pd.DataFrame:
    path = cache_path("governance_summary.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def load_sentiment_layer() -> pd.DataFrame:
    if not SENTIMENT_CACHE.exists():
        return pd.DataFrame()
    data = pd.read_parquet(SENTIMENT_CACHE)
    if "date" in data:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return data


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def load_market_event_windows() -> pd.DataFrame:
    if not EVENT_WINDOWS_CACHE.exists():
        return pd.DataFrame()
    data = pd.read_parquet(EVENT_WINDOWS_CACHE)
    for column in ["start_date", "end_date"]:
        if column in data:
            data[column] = pd.to_datetime(data[column], errors="coerce")
    return data


@st.cache_data(show_spinner=False, ttl=60 * 10)
def load_project_memory_data():
    return load_memory_bundle()


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def load_discovery_perf() -> pd.DataFrame:
    return load_discovery_performance()


@st.cache_data(show_spinner=False, ttl=60 * 5)
def load_portfolio_prices(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_portfolio_prices(list(tickers), period="1y")


@st.cache_data(show_spinner=False, ttl=60 * 10)
def load_portfolio_news(tickers: tuple[str, ...], days: int) -> pd.DataFrame:
    return fetch_news_batch(symbols=list(tickers), days=days, limit_per_symbol=8)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def load_lstm_evaluation() -> dict:
    path = cache_path("lstm/lstm_evaluation.json")
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.1%}"


def _display_utc(value: object) -> str:
    if value in {None, "", "n/a"}:
        return "n/a"
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    return "n/a" if pd.isna(parsed) else parsed.strftime("%Y-%m-%d %H:%M UTC")


def _display_date(value: object) -> str:
    if value in {None, "", "n/a"}:
        return "n/a"
    parsed = pd.to_datetime(value, errors="coerce")
    return "n/a" if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


def num(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"


def render_insight_card(container, title: str, value: str, detail: str = "") -> None:
    container.markdown(
        f"""
        <div class="insight-card">
            <div class="label">{title}</div>
            <div class="value">{value}</div>
            <div class="detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def light_class(value: str) -> str:
    if value == "紅":
        return "light-red"
    if value == "黃":
        return "light-yellow"
    return "light-green"


def _show_anomaly_table(data: pd.DataFrame) -> None:
    if data.empty:
        st.info("這一類目前沒有觸發異常。")
        return
    anomaly_display = data[
        ["symbol", "name", "group", "flags", "ret_1d", "ret_z_20d", "volume_ratio_20d", "gap_pct", "dist_ma_200"]
    ].rename(
        columns={
            "symbol": "代號",
            "name": "名稱",
            "group": "類別",
            "flags": "異常",
            "ret_1d": "1D",
            "ret_z_20d": "報酬z",
            "volume_ratio_20d": "量/20日均量",
            "gap_pct": "跳空",
            "dist_ma_200": "距200DMA",
        }
    )
    st.dataframe(
        anomaly_display,
        hide_index=True,
        width="stretch",
        column_config={
            "1D": st.column_config.NumberColumn(format="%.2%"),
            "報酬z": st.column_config.NumberColumn(format="%.2f"),
            "量/20日均量": st.column_config.NumberColumn(format="%.2fx"),
            "跳空": st.column_config.NumberColumn(format="%.2%"),
            "距200DMA": st.column_config.NumberColumn(format="%.2%"),
        },
    )


def latest_value(data: pd.DataFrame, column: str) -> str:
    if data.empty or column not in data:
        return "n/a"
    value = pd.to_datetime(data[column], errors="coerce").max()
    if pd.isna(value):
        return "n/a"
    return str(value.date())


def summary_freshness_status(ai_summary: dict, metadata: dict) -> tuple[str, bool]:
    generated = pd.to_datetime(ai_summary.get("generated_at_utc"), errors="coerce", utc=True)
    cache_updated = pd.to_datetime(metadata.get("updated_at_utc"), errors="coerce", utc=True)
    if pd.isna(generated):
        return "摘要時間未知", False
    if pd.isna(cache_updated):
        return "摘要時間正常", False
    gap_hours = (cache_updated - generated).total_seconds() / 3600
    if gap_hours > 36:
        return f"摘要比最新資料落後約 {gap_hours:.0f} 小時，建議等待下一輪雲端摘要。", True
    return "摘要時間正常", False


def build_health_report(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    news: pd.DataFrame,
    international_news: pd.DataFrame,
    prediction_log: pd.DataFrame,
    metadata: dict,
    ai_history: pd.DataFrame,
    lstm_status: dict,
    discovery_news: pd.DataFrame,
    discovery_candidates: pd.DataFrame,
    discovery_history: pd.DataFrame,
    tsla_keyword_news: pd.DataFrame,
    governance_summary: pd.DataFrame,
    sentiment: pd.DataFrame,
    market_event_windows: pd.DataFrame,
    kg_payload,
) -> pd.DataFrame:
    health_inputs = {
        "prices": prices,
        "macro": macro,
        "news": news,
        "international_news": international_news,
        "prediction_log": prediction_log,
        "metadata": metadata,
        "ai_summary_history": ai_history,
        "lstm_status": lstm_status,
        "discovery_news": discovery_news,
        "discovery_candidates": discovery_candidates,
        "discovery_history": discovery_history,
        "focus_news": tsla_keyword_news,
        "governance": governance_summary,
        "sentiment": sentiment,
        "market_event_windows": market_event_windows,
        "kg_fact_events": kg_payload.facts,
        "kg_narratives": kg_payload.narratives,
        "kg_reactions": kg_payload.reactions,
    }
    signature = inspect.signature(data_health_report)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return data_health_report(**health_inputs)
    supported = {name for name in signature.parameters if name in health_inputs}
    return data_health_report(**{name: health_inputs[name] for name in supported})


def pipeline_status_explainer(metadata: dict) -> tuple[str, str]:
    status = str(metadata.get("pipeline_status", "") or "")
    success_count = int(metadata.get("pipeline_success_count", 0) or 0)
    fallback_count = int(metadata.get("pipeline_fallback_count", 0) or 0)
    failure_count = int(metadata.get("pipeline_failure_count", 0) or 0)

    if status == "success":
        return (
            "success",
            f"本輪更新正常完成，{success_count} 個模組已更新，沒有啟用保護或失敗模組。",
        )
    if status == "partial" and fallback_count > 0 and failure_count == 0:
        return (
            "info",
            "本輪有部分來源較慢，但系統已自動保留較新的舊快取，資料沒有倒退，整體仍可正常閱讀。",
        )
    if status == "partial":
        return (
            "warning",
            f"本輪更新有部分模組需要留意：成功 {success_count}、保護 {fallback_count}、失敗 {failure_count}。",
        )
    if status == "failed":
        return ("error", "本輪更新失敗，前台目前顯示的是先前快取，建議優先檢查更新流程。")
    return ("warning", "目前無法完整判讀這輪更新狀態，前台先顯示現有快取資料。")


def _news_keywords_path():
    return cache_path("news_keywords.txt")


def load_news_keywords() -> list[str]:
    path = _news_keywords_path()
    if not path.exists():
        return DEFAULT_TSLA_KEYWORDS.copy()
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_news_keywords(keywords: list[str]) -> None:
    path = _news_keywords_path()
    path.write_text("\n".join([kw.strip() for kw in keywords if kw.strip()]), encoding="utf-8")


def filter_table_by_keywords(data: pd.DataFrame, keywords: list[str], columns: list[str]) -> pd.DataFrame:
    if data.empty or not keywords:
        return data.copy()
    terms = [term.strip().lower() for term in keywords if term and str(term).strip()]
    if not terms:
        return data.copy()
    present_columns = [col for col in columns if col in data.columns]
    if not present_columns:
        return data.copy()
    mask = pd.Series(False, index=data.index)
    for col in present_columns:
        text = data[col].fillna("").astype(str).str.lower()
        for term in terms:
            mask |= text.str.contains(term, regex=False)
    return data[mask].copy().reset_index(drop=True)


def render_discovery_rank_table(data: pd.DataFrame, title: str) -> None:
    st.markdown(f"#### {title}")
    if data.empty:
        st.info("目前歷史資料還不足，累積幾天後會自動產生排行。")
        return
    display = data.rename(
        columns={
            "ticker": "股票",
            "rank_score": "潛力分數",
            "appearance_days": "入榜天數",
            "avg_candidate_score": "平均候選分",
            "max_candidate_score": "最高候選分",
            "topic_count": "主題數",
            "topics": "主要主題",
            "headline_count": "新聞數",
            "avg_rel_qqq": "平均相對QQQ",
            "risk_count": "風險次數",
            "latest_reason": "最新理由",
            "latest_risk": "最新風險",
            "sample_headline": "代表新聞",
        }
    )
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "潛力分數": st.column_config.NumberColumn(format="%.0f"),
            "平均候選分": st.column_config.NumberColumn(format="%.0f"),
            "最高候選分": st.column_config.NumberColumn(format="%.0f"),
            "平均相對QQQ": st.column_config.NumberColumn(format="%.2%"),
        },
    )


with st.sidebar:
    st.header("設定")
    news_days = st.slider("新聞回看天數", 3, 30, 14)
    show_health = st.button("顯示資料健康檢查", width="stretch")
    st.caption("資料由 GitHub Actions 每 6 小時自動更新；前台只讀快取，避免人為刷新造成偏差。")
    st.caption("AI 摘要每日 07:00（台灣時間）由 OpenAI 自動生成；前台只讀取摘要快取。")
    st.caption("就算 Streamlit 頁面因閒置睡著，雲端資料更新仍會照常進行；重新打開頁面後會讀取最新快取。")

with st.spinner("讀取市場資料..."):
    prices, macro = load_market(str(default_start_date()))

if prices.empty:
    st.error("目前沒有市場資料。請等待 GitHub Actions 完成下一次資料更新。")
    st.stop()

news = load_news(news_days)
international_news = load_international_news(min(news_days, 7))
discovery_news, discovery_mentions, discovery_candidates, discovery_history = load_discovery()
tsla_keyword_news = load_tsla_keyword_news()
governance = load_governance_summary()
sentiment = load_sentiment_layer()
market_event_windows = load_market_event_windows()
emotion_row = latest_emotion(sentiment)
fear_greed = fear_greed_analysis(emotion_row)
emotion_components_table = emotion_components(emotion_row)
emotion_trend_table = emotion_trend(sentiment)
emotion_divergence_table = emotion_divergence(prices, sentiment)
emotion_alert_table = pd.DataFrame(emotion_alerts(emotion_row))
discovery_performance = load_discovery_perf()
kg_payload = load_knowledge_graph()
kg_health = kg_summary(kg_payload)
kg_prediction_log = load_kg_prediction_log()
kg_prediction_v2_log = load_kg_prediction_v2_log()
historical_market_samples = load_stratified_market_samples()
fundamental_observations = load_fundamental_observations()
factor_effectiveness = load_factor_effectiveness()
kg_backtest_readiness = load_kg_backtest_readiness()
metadata = load_metadata()
ai_summary = load_cached_ai_summary()
ai_quality = ai_summary_quality(ai_summary) if ai_summary else {}
ai_freshness_text, ai_is_stale = summary_freshness_status(ai_summary, metadata) if ai_summary else ("尚未產生摘要", True)
ai_history = load_ai_summary_history()
latest_ai_entry = latest_ai_history_entry(ai_history)
openai_status = openai_configuration_status()
indicators = add_price_indicators(prices)
snapshot = latest_snapshot(indicators)
anomalies = detect_anomalies(snapshot)
regime = regime_summary(indicators, macro)
conclusion = today_conclusion(regime, snapshot, anomalies)
market_prediction = build_market_prediction(regime, conclusion, snapshot)
prediction_log = load_prediction_log()
annual_picks = annual_picks_table(prices)
lstm_status = build_lstm_status_from_artifacts()
lstm_predictions = load_lstm_predictions()
lstm_backtest = load_lstm_backtest()
lstm_evaluation = load_lstm_evaluation()
active_news_keywords = load_news_keywords()
keyword_discovery_news = tsla_keyword_news.copy()
tsla_keyword_summary = summarize_keyword_news(keyword_discovery_news, "TSLA")

last_date = pd.to_datetime(snapshot["date"]).max().date() if not snapshot.empty else None
updated_at = metadata.get("updated_at_utc", "尚未寫入")
pipeline_status = str(metadata.get("pipeline_status", "") or "")
pipeline_badge = "資料已同步" if pipeline_status == "success" else "快取保護中"
st.markdown(
    f"""
    <div class="terminal-hero">
      <div style="display:flex; gap:1rem; justify-content:space-between; flex-wrap:wrap;">
        <div>
          <div class="terminal-eyebrow">MARKET RESEARCH TERMINAL</div>
          <div class="terminal-title">科技股量化監控儀表板</div>
          <div class="terminal-subtitle">以價格、敘事、情緒與資金反應整合市場研究</div>
        </div>
        <div class="terminal-badges">
          <span class="terminal-badge live">{pipeline_badge}</span>
          <span class="terminal-badge">市場資料 {last_date}</span>
          <span class="terminal-badge">快取 {updated_at}</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("最後更新")
    st.caption(f"市場價格：{last_date}")
    st.caption(f"標的新聞：{latest_value(news, 'published')}")
    st.caption(f"國際新聞：{latest_value(international_news, 'published')}")
    st.caption(f"新聞探索：{latest_value(discovery_news, 'published')}")
    st.caption(f"探索歷史：{latest_value(discovery_history, 'date')}")
    st.caption(f"AI 摘要：{ai_summary.get('generated_at_utc', '尚未產生')}")
    st.caption(f"LSTM：{lstm_status.get('status', 'n/a')}")
    st.caption(f"快取寫入 UTC：{updated_at}")
    if ai_summary.get("used_ai"):
        st.success(f"AI 摘要已由雲端 OpenAI 產生｜模型：{ai_summary.get('model', 'n/a')}")
    elif openai_status.get("configured"):
        st.success(f"OpenAI 已就緒｜模型：{openai_status.get('model', 'n/a')}")
    else:
        st.warning("前台未讀到 OpenAI API key；目前顯示規則備援摘要。")
    if latest_ai_entry and not ai_summary.get("used_ai"):
        st.caption(
            "最近一次 OpenAI 摘要："
            f"{latest_ai_entry.get('generated_at_utc', 'n/a')}｜模型：{latest_ai_entry.get('model', 'n/a')}"
        )
    if ai_summary:
        (st.warning if ai_is_stale else st.caption)(ai_freshness_text)
    st.divider()
    st.caption("TSLA 關鍵字設定與分析結果已移到「重點個股追蹤」，避免影響主新聞觀看。")

top = st.columns([1.2, 1, 1, 1])
top[0].metric("Regime Score", pct(regime["score"] / 100 if pd.notna(regime["score"]) else np.nan))
top[0].caption(regime["label"])

qqq = snapshot[snapshot["symbol"] == "QQQ"].squeeze()
smh = snapshot[snapshot["symbol"] == "SMH"].squeeze()
vix = snapshot[snapshot["symbol"] == "^VIX"].squeeze()
top[1].metric("QQQ 1M", pct(qqq.get("ret_20d") if isinstance(qqq, pd.Series) else np.nan), pct(qqq.get("ret_1d") if isinstance(qqq, pd.Series) else np.nan))
top[2].metric("SMH 1M", pct(smh.get("ret_20d") if isinstance(smh, pd.Series) else np.nan), pct(smh.get("ret_1d") if isinstance(smh, pd.Series) else np.nan))
top[3].metric("VIX", num(vix.get("close") if isinstance(vix, pd.Series) else np.nan, 1), pct(vix.get("ret_1d") if isinstance(vix, pd.Series) else np.nan))

if regime["drivers"]:
    st.caption(" / ".join(regime["drivers"]))

st.markdown("### 今日結論")
st.info(conclusion["sentence"])
card_cols = st.columns(4)
for card_col, card in zip(card_cols, conclusion_cards(regime, conclusion, market_prediction, anomalies)):
    render_insight_card(card_col, card["title"], card["value"], card.get("detail", ""))

if show_health:
    st.subheader("資料健康檢查")
    health_level, health_message = pipeline_status_explainer(metadata)
    getattr(st, health_level)(health_message)
    st.dataframe(
        build_health_report(
            prices,
            macro,
            news,
            international_news,
            prediction_log,
            metadata,
            ai_history,
            lstm_status,
            discovery_news,
            discovery_candidates,
            discovery_history,
            tsla_keyword_news,
            governance,
            sentiment,
            market_event_windows,
            kg_payload,
        ),
        hide_index=True,
        width="stretch",
    )
    if not governance.empty:
        st.markdown("#### 資料治理分層摘要")
        governance_display = governance.rename(
            columns={
                "dataset": "資料流",
                "rows": "總筆數",
                "official": "正式",
                "pending_short": "短期待確認",
                "pending_medium": "中期待確認",
                "pending_long": "長期待確認",
                "rejected": "拒收",
                "top_reasons": "主要原因",
            }
        )
        st.dataframe(governance_display, hide_index=True, width="stretch")
        st.markdown("#### 資料治理告警")
        for alert in governance_alerts(governance):
            getattr(st, alert.get("level", "info"))(alert.get("message", ""))
    missing = missing_price_symbols(prices, ETF_TICKERS + STOCK_TICKERS + ANNUAL_PICK_TICKERS + ["SPY", "QQQ"])
    if missing:
        st.warning("缺少價格資料：" + ", ".join(missing[:20]))
    else:
        st.success("主要追蹤標的價格資料完整。")

project_memory = load_project_memory_data()

tab_overview, tab_anomaly, tab_analog, tab_news, tab_prediction, tab_discovery, tab_focus, tab_emotion, tab_kg, tab_memory, tab_charts, tab_portfolio, tab_quant = st.tabs(
    ["總覽", "異常雷達", "歷史相似情境", "新聞與摘要", "預測驗證", "新聞探索", "重點個股追蹤", "市場情緒", "金融知識圖譜", "專案記憶", "走勢圖", "我的持倉", "量化數據中心"]
)

with tab_overview:
    chart_df = snapshot[snapshot["symbol"].isin(ETF_TICKERS + STOCK_TICKERS)].copy()
    fig = px.scatter(
        chart_df,
        x="ret_20d",
        y="drawdown_52w",
        size="volume_ratio_20d",
        color="group",
        hover_name="symbol",
        labels={"ret_20d": "1M return", "drawdown_52w": "Drawdown from 52W high"},
        height=360,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=25, b=10), legend_title_text="")
    st.plotly_chart(fig, width="stretch")

    with st.expander("名詞說明"):
        st.markdown(
            """
            - **50DMA**：50 日移動平均線，常用來看中短期趨勢。
            - **200DMA**：200 日移動平均線，常用來看長期多空分界。
            - **VIX**：市場預期波動率，越高通常代表避險情緒越強。
            - **HYG / 高收益債利差**：信用市場風險溫度計，走弱或利差擴大通常代表風險偏好下降。
            - **相對強弱**：例如 QQQ 相對 SPY，代表科技股是否比大盤更強。
            - **最差 10% 均值**：歷史相似樣本中最差那一批結果的平均，用來估計壞情境。
            """
        )

    st.subheader("Codex 年度十大高成長觀察股")
    st.caption("獨立研究名單，不影響 QQQ 市場預測或你的持倉建議。選入日以 2026-05-18 作為追蹤基準。")
    annual_summary = annual_picks_summary(annual_picks)
    pick_cols = st.columns(5)
    pick_cols[0].metric("平均報酬", pct(annual_summary["avg_return"]))
    pick_cols[1].metric("勝率", pct(annual_summary["win_rate"]))
    pick_cols[2].metric("相對 QQQ", pct(annual_summary["avg_rel_qqq"]))
    pick_cols[3].metric("最佳", annual_summary["best"])
    pick_cols[4].metric("最弱", annual_summary["worst"])
    annual_display = annual_picks.rename(
        columns={
            "ticker": "股票",
            "theme": "主題",
            "risk_level": "風險",
            "selected_date": "選入日",
            "selected_price": "選入價",
            "current_price": "現價",
            "return_since_selected": "選入後報酬",
            "relative_spy": "相對 SPY",
            "relative_qqq": "相對 QQQ",
            "max_drawdown": "最大回撤",
            "status": "狀態",
            "reason": "選入理由",
            "risk": "主要風險",
        }
    )
    annual_columns = ["股票", "主題", "風險", "選入價", "現價", "選入後報酬", "相對 QQQ", "最大回撤", "狀態", "選入理由", "主要風險"]
    st.dataframe(
        annual_display[[col for col in annual_columns if col in annual_display.columns]],
        hide_index=True,
        width="stretch",
        column_config={
            "選入價": st.column_config.NumberColumn(format="$%.2f"),
            "現價": st.column_config.NumberColumn(format="$%.2f"),
            "選入後報酬": st.column_config.NumberColumn(format="%.2%"),
            "相對 QQQ": st.column_config.NumberColumn(format="%.2%"),
            "最大回撤": st.column_config.NumberColumn(format="%.2%"),
        },
    )

with tab_anomaly:
    st.subheader("今日異常訊號")
    if anomalies.empty:
        st.info("目前 watchlist 沒有觸發主要異常規則。")
    else:
        category_tabs = st.tabs(["價格異常", "成交量異常", "趨勢異常", "消息異常"])
        categorized = categorize_anomalies(anomalies)
        with category_tabs[0]:
            _show_anomaly_table(categorized.get("價格異常", pd.DataFrame()))
        with category_tabs[1]:
            _show_anomaly_table(categorized.get("成交量異常", pd.DataFrame()))
        with category_tabs[2]:
            _show_anomaly_table(categorized.get("趨勢異常", pd.DataFrame()))
        with category_tabs[3]:
            news_anomaly = pd.DataFrame()
            if not news.empty and "tags" in news:
                news_anomaly = news[
                    news["tags"].astype(str).str.contains("大盤風險|監管/訴訟|財報/財測|國際", na=False)
                ].copy()
            if news_anomaly.empty:
                st.info("近期沒有明顯消息異常。")
            else:
                for row in news_anomaly.head(20).itertuples():
                    published = row.published.strftime("%Y-%m-%d %H:%M") if pd.notna(row.published) else ""
                    st.markdown(f"**{row.symbol}** · `{row.tags}` · {row.source} · {published}  \n[{row.title}]({row.link})")

    st.subheader("下跌前風險線索")
    risk_table = risk_clue_table(indicators, macro, snapshot)
    light_counts = risk_table["light"].value_counts() if "light" in risk_table else pd.Series(dtype=int)
    st.markdown(
        "｜".join(
            [
                f"<span class='{light_class('紅')}'>紅 {int(light_counts.get('紅', 0))}</span>",
                f"<span class='{light_class('黃')}'>黃 {int(light_counts.get('黃', 0))}</span>",
                f"<span class='{light_class('綠')}'>綠 {int(light_counts.get('綠', 0))}</span>",
            ]
        ),
        unsafe_allow_html=True,
    )
    st.dataframe(
        risk_table.rename(
            columns={
                "indicator": "線索",
                "light": "燈號",
                "current": "目前數值",
                "risk_threshold": "風險門檻",
                "status": "狀態",
                "implication": "解讀",
            }
        ),
        hide_index=True,
        width="stretch",
    )

with tab_analog:
    st.subheader("QQQ 歷史相似情境")
    sample_choice = st.segmented_control("樣本層級", options=["核心 50 筆", "參考 100 筆"], default="核心 50 筆")
    analogs_core = historical_analogs(indicators, target="QQQ", top_n=50)
    analogs_broad = historical_analogs(indicators, target="QQQ", top_n=100)
    analogs = analogs_core if sample_choice == "核心 50 筆" else analogs_broad
    if analogs.empty:
        st.info("資料量不足，暫時無法計算歷史相似情境。")
    else:
        stats = analog_stats(analogs)
        st.success(f"一句話解讀：{analog_interpretation(stats)}")
        st.caption(
            "核心 50 筆用來看最接近目前的歷史劇本；參考 100 筆用來看更穩定的背景分布。"
            "這是歷史條件相似度，不是未來保證。"
        )
        stat_cols = st.columns(len(stats) if len(stats) else 1)
        for i, row in enumerate(stats.itertuples()):
            stat_cols[i].metric(
                f"後 {row.horizon}",
                pct(row.avg_return),
                f"中位數 {pct(row.median_return)}",
            )
            stat_cols[i].caption(
                f"N={row.sample}｜原始上漲 {pct(row.win_rate)}｜保守勝率 {pct(row.win_rate_conservative)}｜"
                f"最差10%均值 {pct(row.worst_decile_avg)}｜信心 {row.confidence}"
            )

        st.markdown(
            """
            **指標怎麼看：** 平均報酬代表歷史期望值，中位數代表比較典型的結果，保守勝率會修正小樣本過度樂觀，
            最差 10% 均值用來看壞情境。若樣本數少、分布很分散或平均與中位數互相矛盾，信心等級會下降。
            """
        )

with tab_news:
    from src.news import rule_based_news_summary

    st.subheader("每日 AI 市場摘要")
    if ai_summary:
        used_ai = bool(ai_summary.get("used_ai"))
        status = ai_summary.get("status", "")
        provider = ai_summary.get("provider", "n/a")
        model = ai_summary.get("model", "n/a")
        generated = ai_summary.get("generated_at_utc", "n/a")
        (st.success if used_ai else st.warning)(status)
        st.caption(f"來源：{provider}｜模型：{model}｜生成 UTC：{generated}｜排程：每日 07:00 台灣時間")
        if ai_is_stale:
            st.warning(ai_freshness_text)
        if used_ai:
            st.caption("OpenAI 狀態：本摘要已由雲端排程使用 OpenAI 產生。")
        else:
            st.caption(
                f"OpenAI 狀態：{openai_status.get('status', 'n/a')}｜"
                f"模型：{openai_status.get('model', 'n/a')}｜"
                f"Key：{openai_status.get('api_key_preview', 'n/a')}"
            )
        qcols = st.columns(4)
        qcols[0].metric("摘要品質", ai_quality.get("quality_label", "n/a"))
        qcols[1].metric("完整度", f"{ai_quality.get('section_count', 0)}/{ai_quality.get('required_sections', 6)}")
        qcols[2].metric("字數", f"{ai_quality.get('text_length', 0)}")
        qcols[3].metric("品質分", num(ai_quality.get("quality_score"), 0))
        if ai_quality.get("missing_sections") not in {None, "", "無"}:
            st.caption("缺少章節：" + str(ai_quality.get("missing_sections")))
        st.caption(
            f"edge 品質：{num(ai_summary.get('quality_score'), 0)}｜"
            f"來源可靠度：{num(ai_summary.get('source_reliability_score'), 2)}｜"
            f"來源網域：{ai_summary.get('source_domain', 'n/a')}｜"
            f"去重鍵：{ai_summary.get('dedup_key', 'n/a')}"
        )
        st.markdown(ai_summary.get("text", ""))
        if not ai_history.empty:
            ai_quality_summary = summarize_quality_frame(ai_history, "quality_score")
            hist_cols = st.columns(4)
            hist_cols[0].metric("歷史摘要數", f"{ai_quality_summary['rows']}")
            hist_cols[1].metric("平均品質", num(ai_quality_summary["avg_quality"], 0))
            hist_cols[2].metric("低品質筆數", f"{ai_quality_summary['low_quality_rows']}")
            hist_cols[3].metric("來源數", f"{ai_quality_summary['unique_sources']}")
            with st.expander("查看 AI 摘要歷史"):
                history_display = ai_history.rename(
                    columns={
                        "summary_date": "日期",
                        "generated_at_utc": "產生時間",
                        "provider": "來源",
                        "model": "模型",
                        "used_ai": "使用AI",
                        "status": "狀態",
                        "quality_score": "品質分",
                        "quality_label": "品質標籤",
                        "text_length": "文字長度",
                        "section_count": "章節數",
                        "missing_sections": "缺漏章節",
                        "source_domain": "來源網域",
                        "source_reliability_score": "來源可靠度",
                        "text_density_score": "文字密度",
                        "structure_score": "結構分",
                        "quality_score_edge": "edge品質分",
                        "prompt_version": "提示版本",
                    }
                )
                st.dataframe(
                    history_display[[c for c in [
                        "日期",
                        "產生時間",
                        "來源",
                        "模型",
                        "使用AI",
                        "狀態",
                        "品質分",
                        "品質標籤",
                        "來源網域",
                        "來源可靠度",
                        "edge品質分",
                        "章節數",
                        "缺漏章節",
                        "提示版本",
                    ] if c in history_display.columns]],
                    hide_index=True,
                    width="stretch",
                )
    else:
        st.warning("尚未產生每日 AI 摘要，先顯示規則摘要。GitHub Actions 會在每日 07:00 台灣時間自動生成。")
        st.markdown(rule_based_news_summary(news))

    st.subheader("新聞標籤與連結")
    if news.empty:
        st.info("目前沒有抓到近期新聞。")
    else:
        st.subheader("新聞對持倉影響")
        news_portfolio_config = load_portfolio_config(st.secrets)
        if news_portfolio_config is not None and not news_portfolio_config.positions.empty:
            impact_base = news_portfolio_config.positions.rename(columns={"ticker": "ticker"})
            impact = portfolio_news_impact(news, impact_base, max_items=8)
            if impact.empty:
                st.caption("近期新聞尚未直接命中你的持倉代號。")
            else:
                st.dataframe(
                    impact.rename(
                        columns={
                            "ticker": "股票代號",
                            "impact_level": "影響等級",
                            "impact": "可能影響",
                            "headline_count": "新聞數",
                            "key_tags": "主要標籤",
                            "sample_headline": "代表新聞",
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
        else:
            st.caption("尚未設定持倉 Secrets，因此先顯示一般新聞摘要。")

        filtered_symbols = st.multiselect("標的", options=list(NEWS_QUERIES), default=list(NEWS_QUERIES)[:6])
        news_view = news[news["symbol"].isin(filtered_symbols)] if filtered_symbols else news
        for row in news_view.head(80).itertuples():
            published = row.published.strftime("%Y-%m-%d %H:%M") if pd.notna(row.published) else ""
            st.markdown(f"**{row.symbol}** · `{row.tags}` · {row.source} · {published}  \n[{row.title}]({row.link})")

    st.subheader("今日國際新聞")
    major_news, random_news = international_news_selection(international_news, random_count=3)
    if international_news.empty:
        st.info("目前沒有抓到國際新聞。")
    else:
        if not major_news.empty:
            st.markdown("**重大新聞（戰爭、貿易談判、制裁、出口管制等優先，不計入下方 3 則隨機新聞）**")
            for row in major_news.head(8).itertuples():
                published = row.published.strftime("%Y-%m-%d %H:%M") if pd.notna(row.published) else ""
                st.markdown(f"`{row.tags}` · {row.source} · {published}  \n[{row.title}]({row.link})")
        st.markdown("**隨機 3 則一般國際新聞**")
        if random_news.empty:
            st.caption("一般國際新聞不足 3 則。")
        for row in random_news.itertuples():
            published = row.published.strftime("%Y-%m-%d %H:%M") if pd.notna(row.published) else ""
            st.markdown(f"`{row.tags}` · {row.source} · {published}  \n[{row.title}]({row.link})")

with tab_prediction:
    st.subheader("市場預測驗證")
    st.caption(f"LSTM 狀態：{lstm_status.get('status', 'n/a')}｜模式：{lstm_status.get('mode', 'n/a')}｜模型版本：{lstm_status.get('model_version', 'n/a')}")
    st.markdown("#### LSTM 即時方向預測")
    lstm_time_cols = st.columns(3)
    lstm_time_cols[0].metric("模型訓練日", _display_date(lstm_status.get("last_train_at_utc")))
    lstm_time_cols[1].metric("預測生成日", _display_date(lstm_status.get("last_predict_at_utc")))
    lstm_time_cols[2].metric("預測目標日", _display_date(lstm_status.get("prediction_target_date")))
    st.caption(
        f"完整時間：訓練 {_display_utc(lstm_status.get('last_train_at_utc'))}｜"
        f"預測 {_display_utc(lstm_status.get('last_predict_at_utc'))}"
    )
    lstm_quality_cols = st.columns(3)
    lstm_quality_cols[0].metric("訓練/驗證/測試", f"{lstm_status.get('train_rows', 0)}/{lstm_status.get('valid_rows', 0)}/{lstm_status.get('test_rows', 0)}")
    lstm_quality_cols[1].metric("回測正確率", pct(lstm_status.get("backtest_accuracy")))
    lstm_quality_cols[2].metric("信心等級", str(lstm_status.get("prediction_confidence_level", "低信心")))
    if lstm_predictions.empty:
        st.info("目前沒有即時推論列；下一次模型訓練完成後會產生。")
    else:
        current = lstm_predictions.copy()
        display_columns = [column for column in ["symbol", "prediction_date", "target_date", "target_date_type", "predicted_prob_up", "prediction_direction", "prediction_type"] if column in current]
        st.dataframe(
            current[display_columns].rename(
                columns={
                    "symbol": "標的", "prediction_date": "預測日", "target_date": "目標日（估計）",
                    "target_date_type": "目標日類型", "predicted_prob_up": "上漲機率",
                    "prediction_direction": "方向", "prediction_type": "推論類型",
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={"上漲機率": st.column_config.NumberColumn(format="%.2%")},
        )
        st.caption("即時推論不等待未來報酬標籤；目標日是依交易日推估，待期滿後才會進入回測驗證。")
    evaluation = lstm_evaluation.get("lstm", {}) if isinstance(lstm_evaluation, dict) else {}
    if evaluation:
        st.markdown("#### 預測品質評估")
        eval_cols = st.columns(5)
        eval_cols[0].metric("LSTM 測試準確率", pct(evaluation.get("accuracy")))
        eval_cols[1].metric("多數基準", pct(evaluation.get("baseline_accuracy")))
        eval_cols[2].metric("持續偏多基準", pct(evaluation.get("always_long_accuracy")))
        eval_cols[3].metric("Balanced Accuracy", pct(evaluation.get("balanced_accuracy")))
        eval_cols[4].metric("F1（上漲）", pct(evaluation.get("f1_up")))
        st.caption(
            f"測試樣本 {evaluation.get('rows', 0)} 筆｜Walk-forward 多數基準 {pct(evaluation.get('walk_forward_majority_accuracy'))}｜"
            f"相對持續偏多準確率 {pct(evaluation.get('accuracy_edge_vs_always_long'))}｜"
            f"方向報酬相對持續偏多 {pct(evaluation.get('directional_return_edge_vs_always_long'))}｜"
            f"評估警示：{evaluation.get('confidence_warning', 'n/a')}"
        )
        calibration = pd.DataFrame(evaluation.get("probability_calibration", []))
        if not calibration.empty:
            st.caption("機率校準：模型說上漲機率較高的區間，實際上漲率是否也相對提高。")
            st.dataframe(calibration.rename(columns={"bucket": "預測機率區間", "rows": "樣本數", "actual_up_rate": "實際上漲率"}), hide_index=True, width="stretch", column_config={"實際上漲率": st.column_config.NumberColumn(format="%.2%")})
        audit = lstm_evaluation.get("leakage_audit", {})
        if audit.get("issues"):
            st.error("資料洩漏檢查：" + "；".join(audit["issues"]))
        else:
            st.success("資料洩漏檢查：目前沒有偵測到即時預測日早於目標日的異常。")
        rule_eval = lstm_evaluation.get("rule_predictions", {})
        by_horizon = rule_eval.get("by_horizon", []) if isinstance(rule_eval, dict) else []
        if by_horizon:
            st.dataframe(
                pd.DataFrame(by_horizon).rename(
                    columns={"horizon": "週期", "rows": "總筆數", "validated_rows": "已驗證筆數", "accuracy": "成功率", "avg_return": "平均實際報酬"}
                ),
                hide_index=True,
                width="stretch",
                column_config={"成功率": st.column_config.NumberColumn(format="%.2%"), "平均實際報酬": st.column_config.NumberColumn(format="%.2%")},
            )
    scorecard = prediction_scorecard(prediction_log)
    score_cols = st.columns(5)
    score_cols[0].metric("已驗證樣本", f"{scorecard['validated']}")
    score_cols[1].metric("整體成功率", pct(scorecard["success_rate"]))
    score_cols[2].metric("平均後續報酬", pct(scorecard["avg_return"]))
    score_cols[3].metric("最佳方向", scorecard["best_segment"])
    score_cols[4].metric("待改善方向", scorecard["weak_segment"])

    validation = prediction_validation_summary(prediction_log)
    if validation.empty:
        st.info("目前預測紀錄仍在累積，等 5D / 20D / 60D 週期走完後會自動出現成功率。")
    else:
        st.dataframe(
            validation.rename(
                columns={
                    "horizon": "驗證週期",
                    "prediction_direction": "預測方向",
                    "sample": "樣本數",
                    "success_rate": "成功率",
                    "avg_return": "平均實際報酬",
                    "avg_max_drawdown": "平均最大回撤",
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "成功率": st.column_config.NumberColumn(format="%.2%"),
                "平均實際報酬": st.column_config.NumberColumn(format="%.2%"),
                "平均最大回撤": st.column_config.NumberColumn(format="%.2%"),
            },
        )

with tab_discovery:
    st.subheader("新聞探索候選股")
    st.caption("每天用日期作為隨機種子抽取市場主題，從新聞中找 ticker，再用量價規則評分；系統每日記錄 Top 15，這是觀察清單，不是買入建議。")

    topic_summary = (
        discovery_news.groupby("topic").size().reset_index(name="新聞數").sort_values("新聞數", ascending=False)
        if not discovery_news.empty else pd.DataFrame(columns=["topic", "新聞數"])
    )
    left_disc, right_disc = st.columns([1, 1.6])
    with left_disc:
        st.markdown("#### 今日隨機探索主題")
        if topic_summary.empty:
            st.info("目前沒有探索新聞。")
        else:
            st.dataframe(topic_summary.rename(columns={"topic": "主題"}), hide_index=True, width="stretch")

    with right_disc:
        st.markdown("#### 今日候選觀察股 Top 5")
        if discovery_candidates.empty:
            st.info("目前沒有從探索新聞抽到可驗證的候選股。")
        else:
            daily_display = discovery_candidates.rename(
                columns={
                    "ticker": "股票",
                    "topic": "相關主題",
                    "candidate_score": "候選分數",
                    "candidate_label": "分數解讀",
                    "current_price": "現價",
                    "ret_5d": "5日",
                    "ret_20d": "20日",
                    "volume_ratio_20d": "量/20日均量",
                    "dist_ma_50": "距50DMA",
                    "dist_ma_200": "距200DMA",
                    "rel_spy_20d": "相對SPY",
                    "rel_qqq_20d": "相對QQQ",
                    "risk_flags": "風險標籤",
                    "observation_reason": "觀察理由",
                    "sample_headline": "代表新聞",
                }
            )
            daily_columns = ["股票", "相關主題", "候選分數", "分數解讀", "現價", "5日", "20日", "量/20日均量", "距50DMA", "相對QQQ", "風險標籤", "觀察理由", "代表新聞"]
            st.dataframe(
                daily_display[[col for col in daily_columns if col in daily_display.columns]].head(5),
                hide_index=True,
                width="stretch",
                column_config={
                    "候選分數": st.column_config.NumberColumn(format="%.0f"),
                    "現價": st.column_config.NumberColumn(format="$%.2f"),
                    "5日": st.column_config.NumberColumn(format="%.2%"),
                    "20日": st.column_config.NumberColumn(format="%.2%"),
                    "量/20日均量": st.column_config.NumberColumn(format="%.2fx"),
                    "距50DMA": st.column_config.NumberColumn(format="%.2%"),
                    "相對QQQ": st.column_config.NumberColumn(format="%.2%"),
                },
            )
            with st.expander("查看今日 Top 15"):
                st.dataframe(
                    daily_display[[col for col in daily_columns if col in daily_display.columns]].head(15),
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "候選分數": st.column_config.NumberColumn(format="%.0f"),
                        "現價": st.column_config.NumberColumn(format="$%.2f"),
                        "5日": st.column_config.NumberColumn(format="%.2%"),
                        "20日": st.column_config.NumberColumn(format="%.2%"),
                        "量/20日均量": st.column_config.NumberColumn(format="%.2fx"),
                        "距50DMA": st.column_config.NumberColumn(format="%.2%"),
                        "相對QQQ": st.column_config.NumberColumn(format="%.2%"),
                    },
                )

with tab_focus:
    st.subheader("重點個股追蹤")
    st.caption("這裡是獨立的 TSLA 專題資料流，只讀取專題關鍵字新聞，不會回寫或污染主新聞、新聞探索與候選觀察股。")

    focus_left, focus_right = st.columns([0.95, 1.45])
    with focus_left:
        st.markdown("#### TSLA 關鍵字設定")
        keyword_text = st.text_area(
            "每行一個關鍵字",
            value="\n".join(active_news_keywords),
            height=240,
            placeholder="TSLA\nTesla\nRobotaxi\nFSD\nCybertruck",
            key="focus_keyword_text",
        )
        if st.button("儲存 TSLA 關鍵字", use_container_width=True):
            keywords = [line.strip() for line in keyword_text.splitlines() if line.strip()]
            save_news_keywords(keywords)
            st.success(f"已儲存 {len(keywords)} 個關鍵字；下一次資料更新會套用到 TSLA 專題新聞。")
            st.rerun()
        st.caption("這些關鍵字只影響 TSLA 專題追蹤，不會縮窄全市場新聞流。")

    with focus_right:
        st.markdown("#### TSLA 關鍵字分析總結")
        if not active_news_keywords:
            st.info("目前尚未設定 TSLA 專題關鍵字。")
        else:
            summary_cols = st.columns(4)
            summary_cols[0].metric("命中新聞數", f"{tsla_keyword_summary.get('headline_count', 0)}")
            summary_cols[1].metric("主要分類", tsla_keyword_summary.get("top_groups", "n/a") or "n/a")
            summary_cols[2].metric("風險關鍵字", tsla_keyword_summary.get("risk_keywords", "無") or "無")
            latest_kw = tsla_keyword_summary.get("latest_published", "")
            latest_kw_text = str(pd.to_datetime(latest_kw).date()) if latest_kw else "n/a"
            summary_cols[3].metric("最近新聞", latest_kw_text)
            st.caption("主要命中關鍵字：" + (tsla_keyword_summary.get("top_keywords", "") or "n/a"))
            st.info(tsla_keyword_summary.get("summary", "目前沒有 TSLA 關鍵字分析結果。"))

    st.markdown("#### TSLA 關鍵字命中新聞")
    if keyword_discovery_news.empty:
        st.caption("目前沒有抓到 TSLA 專題新聞。")
    else:
        for row in keyword_discovery_news.head(25).itertuples():
            published = row.published.strftime("%Y-%m-%d %H:%M") if pd.notna(row.published) else ""
            matched = getattr(row, "matched_keywords", "")
            keyword_group = getattr(row, "keyword_group", "")
            meta_bits = [f"`{row.tags}`", row.source, published]
            if matched:
                meta_bits.append(f"命中：`{matched}`")
            if keyword_group:
                meta_bits.append(f"分類：`{keyword_group}`")
            st.markdown(" · ".join([bit for bit in meta_bits if bit]) + f"  \n[{row.title}]({row.link})")
            if getattr(row, "analysis_note", ""):
                st.caption(getattr(row, "analysis_note"))

    weekly_candidates = summarize_discovery_history(discovery_history, days=7, top_n=15)
    monthly_candidates = summarize_discovery_history(discovery_history, days=30, top_n=15)
    perf_summary = discovery_performance_summary(discovery_performance)
    weekly_tab, monthly_tab = st.tabs(["本週潛力排行", "本月潛力排行"])
    with weekly_tab:
        render_discovery_rank_table(weekly_candidates.head(5), "預設 Top 5")
        with st.expander("展開本週 Top 15"):
            render_discovery_rank_table(weekly_candidates, "本週 Top 15")
    with monthly_tab:
        render_discovery_rank_table(monthly_candidates.head(5), "預設 Top 5")
        with st.expander("展開本月 Top 15"):
            render_discovery_rank_table(monthly_candidates, "本月 Top 15")

    st.markdown("#### 候選股入榜後績效追蹤")
    if perf_summary.empty:
        st.info("候選股績效資料仍在累積；入榜後滿 5D / 20D / 60D 會自動驗證。")
    else:
        st.dataframe(
            perf_summary.rename(
                columns={
                    "horizon": "驗證週期",
                    "sample": "樣本數",
                    "success_rate": "上漲率",
                    "avg_return": "平均報酬",
                    "avg_relative_qqq": "平均相對QQQ",
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "上漲率": st.column_config.NumberColumn(format="%.2%"),
                "平均報酬": st.column_config.NumberColumn(format="%.2%"),
                "平均相對QQQ": st.column_config.NumberColumn(format="%.2%"),
            },
        )
    if not discovery_performance.empty:
        with st.expander("查看候選股驗證明細"):
            perf_view = discovery_performance.sort_values(["date", "horizon_days"], ascending=[False, True]).head(80)
            st.dataframe(
                perf_view.rename(
                    columns={
                        "date": "入榜日",
                        "ticker": "股票",
                        "horizon": "週期",
                        "candidate_score": "入榜分數",
                        "entry_price": "入榜價",
                        "actual_return": "後續報酬",
                        "qqq_return": "QQQ同期",
                        "relative_qqq_return": "相對QQQ",
                        "success": "上漲",
                        "validated_at": "驗證日",
                        "observation_reason": "入榜理由",
                    }
                )[["入榜日", "股票", "週期", "入榜分數", "入榜價", "後續報酬", "QQQ同期", "相對QQQ", "上漲", "驗證日", "入榜理由"]],
                hide_index=True,
                width="stretch",
                column_config={
                    "入榜分數": st.column_config.NumberColumn(format="%.0f"),
                    "入榜價": st.column_config.NumberColumn(format="$%.2f"),
                    "後續報酬": st.column_config.NumberColumn(format="%.2%"),
                    "QQQ同期": st.column_config.NumberColumn(format="%.2%"),
                    "相對QQQ": st.column_config.NumberColumn(format="%.2%"),
                },
            )

    st.markdown("#### 探索新聞")
    if discovery_news.empty:
        st.info("目前沒有探索新聞可顯示。")
    else:
        for row in discovery_news.head(30).itertuples():
            published = row.published.strftime("%Y-%m-%d %H:%M") if pd.notna(row.published) else ""
            tickers_text = f"｜tickers: `{row.tickers}`" if getattr(row, "tickers", "") else ""
            st.markdown(f"**{row.topic}** · `{row.tags}` · {row.source} · {published} {tickers_text}  \n[{row.title}]({row.link})")

with tab_emotion:
    st.subheader("市場情緒分析")
    st.caption("本頁使用既有價格、總經、新聞與事件窗資料，另行計算展示，不會修改原始資料或市場預測模型。")
    if not emotion_row:
        st.warning("目前沒有可用的市場情緒資料，等待下一次資料更新。")
    else:
        mood_score = emotion_row.get("market_mood_score")
        mood_label = emotion_row.get("market_mood_label", "資料不足")
        mood_change = emotion_row.get("mood_5d_change")
        risk_pressure = emotion_row.get("fear_pressure")
        emotion_cols = st.columns(4)
        emotion_cols[0].metric("市場情緒總分", num(mood_score, 1), f"{mood_label}")
        emotion_cols[1].metric("恐慌壓力", num(risk_pressure, 1), "0-100，越高壓力越大")
        emotion_cols[2].metric("5日情緒變化", num(mood_change, 1), "分數變化")
        emotion_cols[3].metric("新聞信心", num(emotion_row.get("news_sentiment_confidence"), 1), "來源與覆蓋品質")

        st.markdown("#### 恐懼貪婪指數")
        fg_cols = st.columns(4)
        fg_cols[0].metric("內部指數", num(fear_greed.get("score"), 1), fear_greed.get("label", "資料不足"))
        fg_cols[1].metric("模型信心", num(fear_greed.get("confidence"), 0), "資料覆蓋度")
        fg_cols[2].metric("資料來源", fear_greed.get("source", "內部規則模型"), "可離線運作")
        fg_cols[3].metric("外部指數", "未連接", "不影響內部模型")
        st.caption("恐懼貪婪指數是本系統的可重現內部模型，不直接複製任何外部網站指數；網路中斷時仍可用最近一次成功更新的資料計算。")
        fear_greed_components = fear_greed.get("components", pd.DataFrame())

        st.markdown("#### 今日情緒結論")
        if pd.notna(mood_score):
            if mood_score < 25:
                emotion_sentence = "市場情緒偏防守，恐慌或信用壓力已達需要降低追價的區間。"
            elif mood_score < 40:
                emotion_sentence = "市場情緒正在升溫，尚未等同全面轉空，但需要觀察壓力是否擴散。"
            elif mood_score < 60:
                emotion_sentence = "市場情緒中性，方向仍需搭配價格趨勢與資金反應判斷。"
            elif mood_score < 75:
                emotion_sentence = "市場情緒偏樂觀，風險尚可控，但不宜只依新聞熱度追價。"
            else:
                emotion_sentence = "市場情緒偏熱，若價格與情緒出現背離，應留意高檔降溫。"
            st.info(emotion_sentence)

        alert_cols = st.columns(max(len(emotion_alert_table), 1))
        for column, alert in zip(alert_cols, emotion_alert_table.to_dict("records")):
            tone = "error" if alert.get("燈號") == "🔴" else "warning" if alert.get("燈號") == "🟡" else "success"
            getattr(column, tone)(f"{alert.get('燈號', '')} {alert.get('項目', '')}\n\n{alert.get('說明', '')}")

        st.markdown("#### 情緒趨勢")
        if emotion_trend_table.empty:
            st.caption("目前沒有足夠的情緒歷史資料。")
        else:
            trend_chart = px.line(
                emotion_trend_table,
                x="date",
                y=["市場情緒", "恐慌壓力"],
                labels={"date": "日期", "value": "分數", "variable": "指標"},
                color_discrete_map={"市場情緒": "#1f77b4", "恐慌壓力": "#d62728"},
            )
            trend_chart.update_yaxes(range=[0, 100])
            st.plotly_chart(trend_chart, width="stretch")

        if not emotion_divergence_table.empty:
            st.caption("價格上漲但情緒下降，代表上漲信心減弱；價格下跌但情緒改善，代表可能進入修復觀察期。這是研究訊號，不是單獨交易指令。")
        st.caption(f"資料日期：{pd.to_datetime(emotion_row.get('date'), errors='coerce').date()}｜情緒資料來源：既有 sentiment.parquet；情緒層目前為規則化重建資料。")

with tab_kg:
    st.subheader("金融知識圖譜")
    st.caption("把市場資訊整理成「事件 → 市場敘事 → 實際價格反應」，用來累積日後可驗證的研究證據，而不是直接預測或替你下結論。")
    if "updated_at_utc" in metadata:
        st.caption(
            f"KG 更新 UTC：{metadata.get('updated_at_utc', 'n/a')}｜"
            f"平均 fact 品質：{num(metadata.get('avg_fact_quality'), 0)}｜"
            f"平均來源可靠度：{num(metadata.get('avg_source_reliability'), 2)}"
        )

    if kg_payload.facts.empty:
        st.info("目前尚未建立知識圖譜事件。等待下一次資料更新後會自動填入。")
    else:
        facts_view = kg_payload.facts.copy()
        facts_view["timestamp_utc"] = pd.to_datetime(facts_view["timestamp_utc"], errors="coerce", utc=True)
        facts_view = facts_view.sort_values("timestamp_utc", ascending=False)
        narrative_view = kg_payload.narratives.copy()
        reaction_view = kg_payload.reactions.copy()
        verified_reactions = reaction_view[reaction_view.get("reaction_available", pd.Series(False, index=reaction_view.index)).fillna(False)].copy()

        themes = (
            narrative_view.groupby("dominant_theme", dropna=True)
            .agg(事件數=("event_id", "nunique"), 敘事強度=("narrative_strength", "mean"), 情緒分數=("sentiment_score", "mean"))
            .sort_values(["事件數", "敘事強度"], ascending=False)
            if not narrative_view.empty else pd.DataFrame()
        )
        top_theme = str(themes.index[0]) if not themes.empty else "尚未形成明確主題"
        top_theme_count = int(themes.iloc[0]["事件數"]) if not themes.empty else 0
        affected = facts_view.get("affected_tickers", pd.Series(dtype=str)).fillna("").astype(str)
        top_affected = affected[affected.ne("")].iloc[0] if not affected[affected.ne("")].empty else "尚未辨識"
        avg_reaction = pd.to_numeric(verified_reactions.get("return", pd.Series(dtype=float)), errors="coerce").mean()

        kg_metrics = st.columns(4)
        kg_metrics[0].metric("近期主題", top_theme, f"{top_theme_count} 個去重事件")
        kg_metrics[1].metric("最近影響對象", top_affected)
        kg_metrics[2].metric("已驗證市場反應", f"{len(verified_reactions)} 筆", f"平均 {pct(avg_reaction)}")
        kg_metrics[3].metric("研究成熟度", "資料累積中" if len(verified_reactions) < 100 else "可開始回測", f"事件 {len(facts_view)} 筆")

        st.markdown("#### 這一頁現在告訴你什麼")
        st.info(
            "先看『近期主題』理解市場正集中討論什麼；再看下方傳導鏈確認事件影響哪些標的；"
            "最後用『已驗證市場反應』判斷系統是否已有足夠結果可回測。完整事件與數字仍在量化數據中心。"
        )

        st.markdown("#### 最近市場傳導鏈")
        for event in facts_view.head(3).itertuples():
            event_id = getattr(event, "event_id", "")
            narrative_match = narrative_view[narrative_view["event_id"] == event_id] if not narrative_view.empty else pd.DataFrame()
            reaction_match = verified_reactions[verified_reactions["event_id"] == event_id] if not verified_reactions.empty else pd.DataFrame()
            theme = narrative_match.iloc[0].get("dominant_theme", "尚未歸類") if not narrative_match.empty else "尚未歸類"
            if reaction_match.empty:
                reaction_text = "尚未到可驗證的後續交易日"
            else:
                latest_reaction = reaction_match.sort_values("horizon_days", ascending=False).iloc[0]
                reaction_text = f"{latest_reaction.get('affected_ticker', '市場')} 在 {latest_reaction.get('time_horizon', 'n/a')} 的實際反應 {pct(latest_reaction.get('return'))}"
            st.markdown(
                f"**事實：{getattr(event, 'event_title', 'n/a')}**  \\n"
                f"敘事：{theme} ｜影響對象：{getattr(event, 'affected_tickers', '未辨識')} ｜市場反應：{reaction_text}"
            )

        st.markdown("#### KG 多因子趨勢觀察 V2")
        st.caption("實驗性研究紀錄：以事實、敘事、技術趨勢、市場壓力與跨來源確認形成 QQQ 趨勢觀察；保存實際報酬與回撤，不以成功／失敗二分。")
        if kg_prediction_v2_log.empty:
            st.info("下一個成功的資料更新會建立第一組 V2 每日、每週與每月趨勢觀察。V1 歷史紀錄已保留，不會被覆寫。")
        else:
            latest_date = pd.to_datetime(kg_prediction_v2_log["prediction_date"], errors="coerce").max()
            latest_predictions = kg_prediction_v2_log[
                pd.to_datetime(kg_prediction_v2_log["prediction_date"], errors="coerce").eq(latest_date)
            ].copy()
            pred_tabs = st.tabs(["每日（1D）", "每週（5D）", "每月（20D）"])
            for prediction_tab, horizon in zip(pred_tabs, ["每日（1D）", "每週（5D）", "每月（20D）"]):
                with prediction_tab:
                    row = latest_predictions[latest_predictions["horizon"] == horizon]
                    if row.empty:
                        st.info("此觀察期尚未建立紀錄。")
                        continue
                    latest_row = row.iloc[0]
                    prediction_metrics = st.columns(5)
                    prediction_metrics[0].metric("預測趨勢", latest_row.get("prediction_direction", "觀望／分歧"))
                    prediction_metrics[1].metric("信心等級", latest_row.get("confidence", "低"), f"分數 {num(latest_row.get('confidence_score'), 2)}")
                    prediction_metrics[2].metric("趨勢分數", num(latest_row.get("trend_score"), 2))
                    prediction_metrics[3].metric("預測日", pd.to_datetime(latest_row.get("prediction_date"), errors="coerce").strftime("%Y-%m-%d"))
                    actual_return = latest_row.get("actual_return")
                    prediction_metrics[4].metric("實際 QQQ 報酬", pct(actual_return) if pd.notna(actual_return) else "尚待驗證")
                    st.info(str(latest_row.get("reason", "尚無可說明的研究依據。")))
                    factor_rows = pd.DataFrame(
                        [
                            ("事實事件方向", latest_row.get("factor_fact_direction"), latest_row.get("factor_fact_score")),
                            ("敘事情緒", latest_row.get("factor_narrative_direction"), latest_row.get("factor_narrative_score")),
                            ("技術趨勢", latest_row.get("factor_technical_direction"), latest_row.get("factor_technical_score")),
                            ("市場壓力", latest_row.get("factor_pressure_direction"), latest_row.get("factor_pressure_score")),
                            ("跨來源確認", latest_row.get("factor_source_direction"), latest_row.get("factor_source_score")),
                        ],
                        columns=["因子", "方向", "分數"],
                    )
                    st.dataframe(factor_rows, hide_index=True, width="stretch", column_config={"分數": st.column_config.NumberColumn(format="%.2f")})
                    baseline_text = (
                        f"基準：持續偏多／50DMA {latest_row.get('baseline_50dma_direction', 'n/a')}／"
                        f"20日動能 {latest_row.get('baseline_momentum_direction', 'n/a')}。"
                    )
                    st.caption(f"{baseline_text}｜信心校準：{latest_row.get('calibration_state', '樣本不足')}（樣本 {int(latest_row.get('calibration_sample', 0) or 0)}）。")
                    if pd.notna(actual_return):
                        st.caption(f"已產生實際數值：報酬 {pct(actual_return)}｜期間最大回撤 {pct(latest_row.get('max_drawdown'))}。")
                    else:
                        st.caption(f"尚待 {int(latest_row.get('horizon_days', 0))} 個交易日後驗證。")

            kg_prediction_stats = kg_prediction_v2_summary(kg_prediction_v2_log)
            if not kg_prediction_stats.empty:
                st.caption("已完成的數值回測")
                st.dataframe(
                    kg_prediction_stats.rename(columns={"horizon": "觀察期", "prediction_direction": "預測趨勢", "confidence": "信心", "樣本數": "樣本數", "平均實際報酬": "平均實際報酬", "中位數報酬": "中位數報酬", "平均最大回撤": "平均最大回撤"}),
                    hide_index=True,
                    width="stretch",
                    column_config={"平均實際報酬": st.column_config.NumberColumn(format="%.2%"), "中位數報酬": st.column_config.NumberColumn(format="%.2%"), "平均最大回撤": st.column_config.NumberColumn(format="%.2%")},
                )

        st.markdown("#### 歷史回測資料準備度")
        st.caption("固定分層隨機樣本用於檢查不同市場環境，不會混入日常正式預測。完整 KG 回測必須先補齊當時的多來源新聞與 point-in-time 基本面。")
        if historical_market_samples.empty:
            st.info("歷史市場樣本會在下一次資料更新建立。")
        else:
            sample_metrics = st.columns(4)
            sample_metrics[0].metric("分層樣本", f"{len(historical_market_samples)} 個日期")
            sample_metrics[1].metric("市場區間", f"{pd.to_datetime(historical_market_samples['prediction_date']).min():%Y-%m} 至 {pd.to_datetime(historical_market_samples['prediction_date']).max():%Y-%m}")
            sample_metrics[2].metric("完整 KG 可用", f"{int(historical_market_samples.get('eligible_for_full_kg_backtest', pd.Series(dtype=bool)).sum())} 個")
            fundamental_ready = historical_market_samples.get("fundamental_coverage_state", pd.Series(dtype=str)).eq("完整（官方）").sum()
            sample_metrics[3].metric("基本面 PIT 完整", f"{int(fundamental_ready)} 個")
            st.info("技術與市場壓力可完整回測；基本面採 SEC filing date 回補。缺少歷史多來源事件的日期仍會標示覆蓋缺口，絕不以今天的資料倒灌回去。")
            readiness_summary = kg_backtest_readiness_summary(kg_backtest_readiness)
            if not readiness_summary.empty:
                st.markdown("##### KG 回測資格")
                st.dataframe(
                    readiness_summary.rename(columns={"kg_backtest_state": "資格狀態"}),
                    hide_index=True,
                    width="stretch",
                    column_config={"平均20日報酬": st.column_config.NumberColumn(format="%.2%"), "平均20日回撤": st.column_config.NumberColumn(format="%.2%")},
                )
                st.caption("只有同時具備當時多來源事件與官方基本面的日期，才會進入完整 KG 回測。其餘樣本僅作市場技術基線。")

with tab_memory:
    st.subheader("專案記憶 / 討論摘要")
    st.caption("這一頁用來保存長期有效的專案上下文，避免聊天壓縮後遺失已確認的規則、決策與討論結論。")

    status_left, status_right = st.columns([1, 1.3])
    with status_left:
        st.markdown("#### 記憶狀態")
        st.dataframe(project_memory.status_table, hide_index=True, width="stretch")
        if MEMORY_DOCX_FILE.exists():
            st.success("Word 摘要已產生，可直接在資料夾中查看。")
        else:
            st.warning("Word 摘要尚未產生或尚未同步。")
        st.caption(f"記憶資料夾：{project_memory.directory}")
    with status_right:
        st.markdown("#### 最近更新")
        if project_memory.latest_updates.empty:
            st.info("目前尚未載入更新紀錄。")
        else:
            st.dataframe(project_memory.latest_updates, hide_index=True, width="stretch")

    mem_tab0, mem_tab1, mem_tab2, mem_tab3, mem_tab4 = st.tabs(["Word 網頁版", "長期記憶", "重要討論", "當前上下文", "決策與變更"])
    with mem_tab0:
        st.markdown("#### 專案記憶與討論摘要")
        st.caption("此頁是 Word 摘要的網頁閱讀版，內容會隨專案記憶更新而同步，不需要下載檔案。")
        st.markdown("##### 長期記憶")
        st.markdown(project_memory.project_memory or "目前沒有長期記憶內容。")
        st.markdown("##### 重要討論")
        st.markdown(project_memory.conversation_log or "目前沒有討論摘要內容。")
        st.markdown("##### 當前上下文")
        st.markdown(project_memory.active_context or "目前沒有當前上下文內容。")
        st.markdown("##### 已確認決策")
        if project_memory.decision_register.empty:
            st.caption("目前沒有決策登錄資料。")
        else:
            st.dataframe(
                project_memory.decision_register.rename(
                    columns={"date": "日期", "status": "狀態", "category": "類別", "title": "標題", "decision": "決策", "reason": "原因", "impact_scope": "影響範圍"}
                ),
                hide_index=True,
                width="stretch",
            )
    with mem_tab1:
        st.markdown(project_memory.project_memory or "目前沒有長期記憶內容。")
    with mem_tab2:
        st.markdown(project_memory.conversation_log or "目前沒有討論摘要內容。")
    with mem_tab3:
        st.markdown(project_memory.active_context or "目前沒有當前上下文內容。")
    with mem_tab4:
        st.markdown("#### 決策登錄")
        if project_memory.decision_register.empty:
            st.info("目前沒有決策登錄資料。")
        else:
            st.dataframe(
                project_memory.decision_register.rename(
                    columns={
                        "decision_id": "編號",
                        "date": "日期",
                        "status": "狀態",
                        "category": "類別",
                        "title": "標題",
                        "decision": "決策",
                        "reason": "原因",
                        "impact_scope": "影響範圍",
                        "superseded_by": "被取代",
                    }
                ),
                hide_index=True,
                width="stretch",
            )
        st.markdown("#### 記憶更新紀錄")
        if project_memory.memory_changelog.empty:
            st.info("目前沒有記憶更新紀錄。")
        else:
            st.dataframe(
                project_memory.memory_changelog.rename(
                    columns={
                        "date": "日期",
                        "change_summary": "更新內容",
                        "reason": "原因",
                        "source": "來源",
                    }
                ),
                hide_index=True,
                width="stretch",
            )

with tab_charts:
    symbols = st.multiselect("圖表標的", ETF_TICKERS + STOCK_TICKERS + ["SPY", "^VIX", "TLT", "HYG"], default=["QQQ", "SMH", "XLK", "NVDA"])
    chart_data = indicators[indicators["symbol"].isin(symbols)].copy()
    if not chart_data.empty:
        norm = chart_data.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").ffill()
        norm = norm / norm.dropna().iloc[0] - 1
        fig = go.Figure()
        for symbol in norm.columns:
            fig.add_trace(go.Scatter(x=norm.index, y=norm[symbol], mode="lines", name=symbol))
        fig.update_layout(height=460, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")

        selected = st.selectbox("單一標的技術圖", options=symbols, index=0)
        one = indicators[indicators["symbol"] == selected].copy()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=one["date"], y=one["close"], name="Close", mode="lines"))
        for ma in ["ma_50", "ma_200"]:
            if ma in one:
                fig2.add_trace(go.Scatter(x=one["date"], y=one[ma], name=ma.upper(), mode="lines"))
        fig2.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig2, width="stretch")

with tab_portfolio:
    st.subheader("我的持倉")
    portfolio_config = load_portfolio_config(st.secrets)
    if portfolio_config is None or portfolio_config.positions.empty:
        st.info("尚未設定持倉資料。請在 Streamlit Secrets 新增 `[portfolio]` 設定。")
        st.code(
            """
[portfolio]
password = "your-password"
cash_usd = 0
max_position_weight = 0.15
refresh_seconds = 900
positions_csv = '''
ticker,shares,avg_cost,market_value_usd
AMD,20,110,
VOO,,,500
'''
            """.strip(),
            language="toml",
        )
    else:
        portfolio_unlocked = True
        if portfolio_config.password:
            if "portfolio_unlocked" not in st.session_state:
                st.session_state["portfolio_unlocked"] = False
            if not st.session_state["portfolio_unlocked"]:
                password = st.text_input("持倉頁密碼", type="password")
                if st.button("解鎖持倉", width="stretch"):
                    st.session_state["portfolio_unlocked"] = password == portfolio_config.password
                    if not st.session_state["portfolio_unlocked"]:
                        st.error("密碼不正確。")
                portfolio_unlocked = st.session_state["portfolio_unlocked"]

        if not portfolio_unlocked:
            st.info("請輸入密碼後查看私人持倉資料。")
        else:
            if st_autorefresh is not None:
                st_autorefresh(
                    interval=portfolio_config.refresh_seconds * 1000,
                    key="portfolio_refresh",
                )
            else:
                st.caption("目前環境尚未安裝 streamlit-autorefresh，雲端部署後會依 requirements 自動安裝。")

            tickers = tuple(portfolio_config.positions["ticker"].dropna().astype(str).str.upper().drop_duplicates())
            st.caption(f"持倉價格與新聞會在頁面開啟時自動刷新，間隔約 {portfolio_config.refresh_seconds // 60} 分鐘。")

            with st.spinner("更新持倉價格與新聞..."):
                portfolio_history = load_portfolio_prices(tickers)
                portfolio_news = load_portfolio_news(tickers, news_days)
                market_context = {
                    "risk_label": conclusion["label"],
                    "qqq_strong": bool(
                        isinstance(qqq, pd.Series)
                        and qqq.get("dist_ma_50", np.nan) > 0
                        and regime.get("score", 0) >= 55
                    ),
                }
                try:
                    portfolio_view, portfolio_summary, portfolio_alerts = build_portfolio_view(
                        portfolio_config.positions,
                        portfolio_history,
                        portfolio_news,
                        cash_usd=portfolio_config.cash_usd,
                        max_position_weight=portfolio_config.max_position_weight,
                        market_context=market_context,
                    )
                except TypeError as exc:
                    if "market_context" not in str(exc):
                        raise
                    portfolio_view, portfolio_summary, portfolio_alerts = build_portfolio_view(
                        portfolio_config.positions,
                        portfolio_history,
                        portfolio_news,
                        cash_usd=portfolio_config.cash_usd,
                        max_position_weight=portfolio_config.max_position_weight,
                    )

            st.markdown("#### A. 投資組合總覽")
            if not portfolio_alerts.empty:
                st.error("目前有持倉觸發風險警示，請先查看下方風險警示區。")

            overview_cols = st.columns(7)
            overview_cols[0].metric("總市值", f"${num(portfolio_summary['total_market_value'], 0)}")
            overview_cols[1].metric("總投入成本", f"${num(portfolio_summary['total_cost'], 0)}")
            overview_cols[2].metric("未實現損益", f"${num(portfolio_summary['total_pnl'], 0)}")
            overview_cols[3].metric("總報酬率", pct(portfolio_summary["total_return"]))
            overview_cols[4].metric("最大持倉", portfolio_summary["largest_position"])
            overview_cols[5].metric("風險最高", portfolio_summary["riskiest_position"])
            overview_cols[6].metric("今日建議操作", f"{portfolio_summary['action_count']} 檔")
            st.caption(f"現金：${num(portfolio_summary['cash_usd'], 0)}｜總資產：${num(portfolio_summary['total_assets'], 0)}")

            if portfolio_view.empty:
                st.info("目前沒有可顯示的持倉資料。")
            else:
                st.markdown("#### 今日最需要注意的 3 檔")
                attention = attention_positions(portfolio_view, limit=3)
                if attention.empty:
                    st.success("目前沒有特別需要優先處理的持倉。")
                else:
                    attention_display = attention.rename(
                        columns={
                            "ticker": "股票代號",
                            "attention_score": "注意分數",
                            "attention_reason": "注意原因",
                            "position_state": "目前狀態",
                            "suggestion": "操作傾向",
                            "suggestion_intensity": "強度",
                            "market_heat_score": "市場熱度",
                            "risk_score": "風險分數",
                            "position_weight": "持倉占比",
                            "unrealized_return": "未實現報酬率",
                            "action_zone": "操作區間",
                        }
                    )
                    st.dataframe(
                        attention_display,
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "注意分數": st.column_config.NumberColumn(format="%.0f"),
                            "市場熱度": st.column_config.NumberColumn(format="%.0f"),
                            "風險分數": st.column_config.NumberColumn(format="%.0f"),
                            "持倉占比": st.column_config.NumberColumn(format="%.2%"),
                            "未實現報酬率": st.column_config.NumberColumn(format="%.2%"),
                        },
                    )

                detail_columns = [
                    "ticker",
                    "position_bucket",
                    "shares",
                    "avg_cost",
                    "current_price",
                    "market_value",
                    "unrealized_pnl",
                    "unrealized_return",
                    "position_weight",
                    "market_heat_score",
                    "market_heat_label",
                    "risk_score",
                    "position_state",
                    "suggestion_intensity",
                    "suggestion",
                ]
                detail = portfolio_view[[col for col in detail_columns if col in portfolio_view.columns]].rename(
                    columns={
                        "ticker": "股票代號",
                        "position_bucket": "持倉分層",
                        "shares": "持股數量",
                        "avg_cost": "平均成本",
                        "current_price": "現價",
                        "market_value": "市值",
                        "unrealized_pnl": "未實現損益",
                        "unrealized_return": "未實現報酬率",
                        "position_weight": "持倉占總資產比例",
                        "market_heat_score": "市場熱度分數",
                        "market_heat_label": "熱度解讀",
                        "risk_score": "風險分數",
                        "position_state": "目前狀態",
                        "suggestion_intensity": "建議強度",
                        "suggestion": "操作建議",
                    }
                )
                st.markdown("#### C. 操作建議")
                st.markdown("##### 持倉風險分層")
                bucket = bucket_summary(portfolio_view)
                with st.expander("分層規則"):
                    st.dataframe(
                        bucket_guidelines().rename(columns={"position_bucket": "分層", "target_role": "角色", "risk_rule": "風險規則"}),
                        hide_index=True,
                        width="stretch",
                    )

                advice_columns = [
                    "ticker",
                    "position_state",
                    "suggestion",
                    "suggestion_intensity",
                    "suggestion_reason",
                    "market_link",
                    "action_zone",
                    "add_price",
                    "trim_price",
                    "stop_loss_price",
                    "ret_1d",
                    "ret_5d",
                    "ret_20d",
                    "volume_ratio_20d",
                    "news_count",
                    "news_sentiment",
                    "negative_keywords",
                ]
                advice = portfolio_view[[col for col in advice_columns if col in portfolio_view.columns]].rename(
                    columns={
                        "ticker": "股票代號",
                        "position_state": "目前狀態",
                        "suggestion": "建議",
                        "suggestion_intensity": "強度",
                        "suggestion_reason": "理由",
                        "market_link": "持倉與市場風險連動",
                        "action_zone": "操作區間",
                        "add_price": "加倉價",
                        "trim_price": "減碼價",
                        "stop_loss_price": "停損價",
                        "ret_1d": "今日漲跌",
                        "ret_5d": "5日漲跌",
                        "ret_20d": "20日漲跌",
                        "volume_ratio_20d": "量/20日均量",
                        "news_count": "最新新聞數量",
                        "news_sentiment": "新聞情緒分數",
                        "negative_keywords": "負面關鍵字",
                    }
                )
                st.markdown("#### D. 風險警示")
                if portfolio_alerts.empty:
                    st.success("目前沒有觸發主要持倉警示。")
                else:
                    for alert in portfolio_alerts.itertuples():
                        st.error(f"{alert.ticker}：{alert.alerts}")

with tab_quant:
    st.subheader("量化數據中心")
    st.caption("集中查核各模組的原始數字與可回測欄位；文字結論、圖表與研究解讀保留在原本功能頁面。")
    quant_market, quant_analogs, quant_emotion, quant_model, quant_kg, quant_portfolio = st.tabs(
        ["市場數據", "歷史相似", "情緒數據", "預測數據", "知識圖譜數據", "持倉數據"]
    )

    with quant_market:
        st.markdown("#### 市場與技術指標")
        quant_watch = snapshot[snapshot["symbol"].isin(ETF_TICKERS + STOCK_TICKERS)].copy()
        market_columns = [
            "symbol", "name", "group", "close", "ret_1d", "ret_20d", "ret_50d", "dist_ma_50",
            "dist_ma_200", "drawdown_52w", "volume_ratio_20d", "realized_vol_20d",
        ]
        quant_watch = quant_watch[[column for column in market_columns if column in quant_watch]].rename(
            columns={
                "symbol": "代號", "name": "名稱", "group": "類別", "close": "收盤價", "ret_1d": "1日報酬",
                "ret_20d": "20日報酬", "ret_50d": "50日報酬", "dist_ma_50": "距50DMA",
                "dist_ma_200": "距200DMA", "drawdown_52w": "距52週高點", "volume_ratio_20d": "量/20日均量",
                "realized_vol_20d": "20日年化波動",
            }
        )
        st.dataframe(
            quant_watch,
            hide_index=True,
            width="stretch",
            column_config={
                "收盤價": st.column_config.NumberColumn(format="$%.2f"),
                "1日報酬": st.column_config.NumberColumn(format="%.2%"),
                "20日報酬": st.column_config.NumberColumn(format="%.2%"),
                "50日報酬": st.column_config.NumberColumn(format="%.2%"),
                "距50DMA": st.column_config.NumberColumn(format="%.2%"),
                "距200DMA": st.column_config.NumberColumn(format="%.2%"),
                "距52週高點": st.column_config.NumberColumn(format="%.2%"),
                "量/20日均量": st.column_config.NumberColumn(format="%.2fx"),
                "20日年化波動": st.column_config.NumberColumn(format="%.2%"),
            },
        )
        st.markdown("#### 市場廣度數據")
        st.dataframe(
            breadth_table(snapshot).rename(
                columns={"group": "類別", "count": "數量", "above_50dma": "高於50DMA", "above_200dma": "高於200DMA", "avg_1m_return": "平均1M", "avg_drawdown_52w": "平均距52週高點"}
            ),
            hide_index=True,
            width="stretch",
            column_config={"平均1M": st.column_config.NumberColumn(format="%.2%"), "平均距52週高點": st.column_config.NumberColumn(format="%.2%")},
        )

    with quant_analogs:
        st.markdown("#### QQQ 歷史相似樣本")
        quant_sample = st.segmented_control("樣本層級", options=["核心 50 筆", "參考 100 筆"], default="核心 50 筆", key="quant_analog_sample")
        quant_analog_rows = historical_analogs(indicators, target="QQQ", top_n=50 if quant_sample == "核心 50 筆" else 100)
        if quant_analog_rows.empty:
            st.info("資料量不足，暫時無法顯示歷史相似樣本。")
        else:
            quant_stats = analog_stats(quant_analog_rows).rename(
                columns={"horizon": "後續週期", "sample": "樣本數", "avg_return": "平均報酬", "median_return": "中位數報酬", "win_rate": "原始勝率", "win_rate_conservative": "保守勝率", "worst_decile_avg": "最差10%均值", "confidence": "信心"}
            )
            st.dataframe(
                quant_stats,
                hide_index=True,
                width="stretch",
                column_config={
                    "平均報酬": st.column_config.NumberColumn(format="%.2%"), "中位數報酬": st.column_config.NumberColumn(format="%.2%"),
                    "原始勝率": st.column_config.NumberColumn(format="%.2%"), "保守勝率": st.column_config.NumberColumn(format="%.2%"), "最差10%均值": st.column_config.NumberColumn(format="%.2%"),
                },
            )
            quant_analog_rows = quant_analog_rows.copy()
            quant_analog_rows["date"] = pd.to_datetime(quant_analog_rows["date"]).dt.date
            st.dataframe(
                quant_analog_rows.rename(columns={"date": "日期", "similarity": "相似度", "regime_snapshot": "當時狀態", "1M": "後1M", "3M": "後3M", "6M": "後6M", "12M": "後12M"})[["日期", "相似度", "當時狀態", "後1M", "後3M", "後6M", "後12M"]],
                hide_index=True,
                width="stretch",
                column_config={"相似度": st.column_config.NumberColumn(format="%.2f"), "後1M": st.column_config.NumberColumn(format="%.2%"), "後3M": st.column_config.NumberColumn(format="%.2%"), "後6M": st.column_config.NumberColumn(format="%.2%"), "後12M": st.column_config.NumberColumn(format="%.2%")},
            )

    with quant_emotion:
        st.markdown("#### 情緒、恐懼貪婪與市場事件數據")
        fear_greed_components = fear_greed.get("components", pd.DataFrame())
        if not fear_greed_components.empty:
            st.markdown("#### 恐懼貪婪組成")
            st.dataframe(fear_greed_components, hide_index=True, width="stretch", column_config={"分數": st.column_config.NumberColumn(format="%.1f")})
        if not emotion_components_table.empty:
            st.markdown("#### 情緒來源拆解")
            st.dataframe(emotion_components_table, hide_index=True, width="stretch", column_config={"數值": st.column_config.NumberColumn(format="%.1f")})
        if not emotion_divergence_table.empty:
            st.markdown("#### 情緒與價格背離")
            st.dataframe(emotion_divergence_table, hide_index=True, width="stretch", column_config={"20日報酬": st.column_config.NumberColumn(format="%.2%"), "情緒20日變化": st.column_config.NumberColumn(format="%.1f")})
        if not market_event_windows.empty:
            st.markdown("#### 大盤事件窗")
            st.dataframe(market_event_windows.sort_values("end_date", ascending=False), hide_index=True, width="stretch")

    with quant_model:
        st.markdown("#### LSTM 即時推論與回測")
        if not lstm_predictions.empty:
            st.dataframe(lstm_predictions, hide_index=True, width="stretch", column_config={"predicted_prob_up": st.column_config.NumberColumn(format="%.2%")})
        if not lstm_backtest.empty:
            st.markdown("#### LSTM 回測紀錄")
            st.dataframe(lstm_backtest, hide_index=True, width="stretch")
        if not prediction_log.empty:
            st.markdown("#### 規則預測驗證紀錄")
            st.dataframe(prediction_log.sort_values("prediction_date", ascending=False), hide_index=True, width="stretch")

    with quant_kg:
        st.markdown("#### 圖譜資料健康")
        st.dataframe(kg_health, hide_index=True, width="stretch")
        st.markdown("#### KG 多因子趨勢觀察 V2 紀錄")
        if kg_prediction_v2_log.empty:
            st.info("尚無 KG V2 多因子趨勢觀察紀錄。")
        else:
            st.dataframe(
                kg_prediction_v2_log.sort_values(["prediction_date", "horizon_days"], ascending=[False, True]),
                hide_index=True,
                width="stretch",
                column_config={
                    "confidence_score": st.column_config.NumberColumn(format="%.2f"),
                    "trend_score": st.column_config.NumberColumn(format="%.2f"),
                    "actual_return": st.column_config.NumberColumn(format="%.2%"),
                    "max_drawdown": st.column_config.NumberColumn(format="%.2%"),
                },
            )
        st.caption("V1 預測紀錄已保留，僅供與 V2 的長期研究比較；V2 不再以命中／失敗標示結果。")
        st.markdown("#### 分層歷史市場樣本")
        if historical_market_samples.empty:
            st.info("尚無分層歷史市場樣本。")
        else:
            historical_summary = historical_backtest_summary(historical_market_samples)
            if not historical_summary.empty:
                st.dataframe(
                    historical_summary.rename(columns={"regime_bucket": "市場狀態"}),
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "平均1日報酬": st.column_config.NumberColumn(format="%.2%"),
                        "平均5日報酬": st.column_config.NumberColumn(format="%.2%"),
                        "平均20日報酬": st.column_config.NumberColumn(format="%.2%"),
                        "平均20日回撤": st.column_config.NumberColumn(format="%.2%"),
                    },
                )
            st.dataframe(
                historical_market_samples.sort_values("prediction_date", ascending=False),
                hide_index=True,
                width="stretch",
                column_config={
                    "qqq_ret_1d": st.column_config.NumberColumn(format="%.2%"),
                    "qqq_ret_20d": st.column_config.NumberColumn(format="%.2%"),
                    "qqq_dist_ma_50": st.column_config.NumberColumn(format="%.2%"),
                    "qqq_dist_ma_200": st.column_config.NumberColumn(format="%.2%"),
                    "future_return_1d": st.column_config.NumberColumn(format="%.2%"),
                    "future_return_5d": st.column_config.NumberColumn(format="%.2%"),
                    "future_return_20d": st.column_config.NumberColumn(format="%.2%"),
                    "future_max_drawdown_20d": st.column_config.NumberColumn(format="%.2%"),
                },
            )
        st.markdown("#### 因子有效性報告")
        st.caption("目前先驗證市場技術因子與後續 20 日報酬／回撤的關聯；這是資料驅動的描述，不代表因果或交易承諾。")
        factor_summary = factor_effectiveness_summary(factor_effectiveness)
        if factor_summary.empty:
            st.info("因子有效性報告會在下一次成功資料更新後建立。")
        else:
            st.dataframe(
                factor_summary,
                hide_index=True,
                width="stretch",
                column_config={
                    "與20日報酬相關性": st.column_config.NumberColumn(format="%.2f"),
                    "最佳分層平均報酬": st.column_config.NumberColumn(format="%.2%"),
                    "最差分層平均報酬": st.column_config.NumberColumn(format="%.2%"),
                },
            )
            st.dataframe(
                factor_effectiveness.rename(columns={"factor": "因子", "bucket": "分層", "sample_count": "樣本數", "avg_20d_return": "平均20日報酬", "median_20d_return": "中位數20日報酬", "positive_rate": "正報酬比例", "avg_20d_drawdown": "平均20日回撤", "correlation_20d_return": "與20日報酬相關性", "coverage_note": "資料說明"}),
                hide_index=True,
                width="stretch",
                column_config={"平均20日報酬": st.column_config.NumberColumn(format="%.2%"), "中位數20日報酬": st.column_config.NumberColumn(format="%.2%"), "正報酬比例": st.column_config.NumberColumn(format="%.2%"), "平均20日回撤": st.column_config.NumberColumn(format="%.2%"), "與20日報酬相關性": st.column_config.NumberColumn(format="%.2f")},
            )
        if not kg_backtest_readiness.empty:
            st.markdown("#### KG 歷史回測資格明細")
            st.dataframe(kg_backtest_readiness.sort_values("prediction_date", ascending=False), hide_index=True, width="stretch")
        st.markdown("#### SEC Point-in-Time 基本面觀測")
        if fundamental_observations.empty:
            st.info("基本面觀測會在下一次資料更新取得。")
        else:
            st.dataframe(
                fundamental_observations.sort_values(["filed_at", "ticker"], ascending=[False, True]),
                hide_index=True,
                width="stretch",
                column_config={"value": st.column_config.NumberColumn(format="%.2f")},
            )
        st.markdown("#### 事實事件")
        if kg_payload.facts.empty:
            st.info("目前尚未建立可查核的 KG 事實事件。")
        else:
            st.dataframe(kg_payload.facts, hide_index=True, width="stretch")
        st.markdown("#### 事件後市場反應")
        if kg_payload.reactions.empty:
            st.info("反應層資料仍在累積。")
        else:
            st.dataframe(kg_payload.reactions, hide_index=True, width="stretch", column_config={"return": st.column_config.NumberColumn(format="%.2%"), "relative_return": st.column_config.NumberColumn(format="%.2%")})

    with quant_portfolio:
        if "portfolio_view" not in globals() or portfolio_view.empty:
            st.info("持倉為私人資料；請先在「我的持倉」解鎖後，此處會顯示可查核的數值明細。")
        else:
            st.markdown("#### 持倉原始數據")
            st.dataframe(portfolio_view, hide_index=True, width="stretch")
