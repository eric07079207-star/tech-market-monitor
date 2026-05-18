from __future__ import annotations

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
    from src.ai_summary import ai_summary_quality, load_cached_ai_summary
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
from src.data import MACRO_CACHE, PRICE_CACHE, cache_path, load_metadata
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
    from src.health import data_health_report, missing_price_symbols
except ImportError:  # pragma: no cover - protects Streamlit Cloud during partial redeploys
    def data_health_report(*args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame([{"資料項目": "資料同步中", "筆數": 0, "最新日期": "n/a", "說明": "等待雲端部署完成"}])

    def missing_price_symbols(prices: pd.DataFrame, symbols: list[str]) -> list[str]:
        return []
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

from src.news import fetch_news_batch
try:
    from src.news import international_news_selection, portfolio_news_impact
except ImportError:  # pragma: no cover - protects Streamlit Cloud during partial redeploys
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
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    h1 {font-size: 2.35rem; line-height: 1.15;}
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px 14px;
    }
    .small-muted {color: #6b7280; font-size: 0.86rem;}
    .insight-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 14px 16px;
        min-height: 116px;
    }
    .insight-card .label {color: #6b7280; font-size: 0.82rem; margin-bottom: 8px;}
    .insight-card .value {font-size: 1.12rem; font-weight: 700; line-height: 1.25; color: #111827;}
    .insight-card .detail {color: #4b5563; font-size: 0.84rem; margin-top: 8px; line-height: 1.35;}
    .light-green {color: #15803d; font-weight: 700;}
    .light-yellow {color: #a16207; font-weight: 700;}
    .light-red {color: #b91c1c; font-weight: 700;}
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


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def load_discovery_perf() -> pd.DataFrame:
    return load_discovery_performance()


@st.cache_data(show_spinner=False, ttl=60 * 5)
def load_portfolio_prices(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_portfolio_prices(list(tickers), period="1y")


@st.cache_data(show_spinner=False, ttl=60 * 10)
def load_portfolio_news(tickers: tuple[str, ...], days: int) -> pd.DataFrame:
    return fetch_news_batch(symbols=list(tickers), days=days, limit_per_symbol=8)


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.1%}"


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
    news_days = st.slider("新聞回看天數", 3, 30, 10)
    show_health = st.button("顯示資料健康檢查", width="stretch")
    st.caption("資料由 GitHub Actions 每 6 小時自動更新；前台只讀快取，避免人為刷新造成偏差。")
    st.caption("AI 摘要每日 07:00（台灣時間）由 OpenAI 自動生成；前台只讀取摘要快取。")

with st.spinner("讀取市場資料..."):
    prices, macro = load_market(str(default_start_date()))

if prices.empty:
    st.error("目前沒有市場資料。請等待 GitHub Actions 完成下一次資料更新。")
    st.stop()

news = load_news(news_days)
international_news = load_international_news(min(news_days, 7))
discovery_news, discovery_mentions, discovery_candidates, discovery_history = load_discovery()
discovery_performance = load_discovery_perf()
ai_summary = load_cached_ai_summary()
ai_quality = ai_summary_quality(ai_summary) if ai_summary else {}
indicators = add_price_indicators(prices)
snapshot = latest_snapshot(indicators)
anomalies = detect_anomalies(snapshot)
regime = regime_summary(indicators, macro)
conclusion = today_conclusion(regime, snapshot, anomalies)
market_prediction = build_market_prediction(regime, conclusion, snapshot)
prediction_log = load_prediction_log()
metadata = load_metadata()
annual_picks = annual_picks_table(prices)

st.title("科技股量化監控儀表板")
last_date = pd.to_datetime(snapshot["date"]).max().date() if not snapshot.empty else None
updated_at = metadata.get("updated_at_utc", "尚未寫入")
st.markdown(f"<span class='small-muted'>市場資料日期：{last_date}｜快取更新 UTC：{updated_at}</span>", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("最後更新")
    st.caption(f"市場價格：{last_date}")
    st.caption(f"標的新聞：{latest_value(news, 'published')}")
    st.caption(f"國際新聞：{latest_value(international_news, 'published')}")
    st.caption(f"新聞探索：{latest_value(discovery_news, 'published')}")
    st.caption(f"探索歷史：{latest_value(discovery_history, 'date')}")
    st.caption(f"AI 摘要：{ai_summary.get('generated_at_utc', '尚未產生')}")
    st.caption(f"快取寫入 UTC：{updated_at}")

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
    st.dataframe(
        data_health_report(
            prices,
            macro,
            news,
            international_news,
            prediction_log,
            metadata,
            discovery_news,
            discovery_candidates,
            discovery_history,
        ),
        hide_index=True,
        width="stretch",
    )
    missing = missing_price_symbols(prices, ETF_TICKERS + STOCK_TICKERS + ANNUAL_PICK_TICKERS + ["SPY", "QQQ"])
    if missing:
        st.warning("缺少價格資料：" + ", ".join(missing[:20]))
    else:
        st.success("主要追蹤標的價格資料完整。")

tab_overview, tab_anomaly, tab_analog, tab_news, tab_prediction, tab_discovery, tab_charts, tab_portfolio = st.tabs(
    ["總覽", "異常雷達", "歷史相似情境", "新聞與摘要", "預測驗證", "新聞探索", "走勢圖", "我的持倉"]
)

with tab_overview:
    left, right = st.columns([1.45, 1])
    with left:
        watch = snapshot[snapshot["symbol"].isin(ETF_TICKERS + STOCK_TICKERS)].copy()
        display = watch[
            [
                "symbol",
                "name",
                "group",
                "ret_1d",
                "ret_20d",
                "ret_50d",
                "dist_ma_50",
                "dist_ma_200",
                "drawdown_52w",
                "volume_ratio_20d",
                "realized_vol_20d",
            ]
        ].rename(
            columns={
                "symbol": "代號",
                "name": "名稱",
                "group": "類別",
                "ret_1d": "1D",
                "ret_20d": "1M",
                "ret_50d": "約 10W",
                "dist_ma_50": "距 50DMA",
                "dist_ma_200": "距 200DMA",
                "drawdown_52w": "距 52W 高點",
                "volume_ratio_20d": "量/20日均量",
                "realized_vol_20d": "20D 年化波動",
            }
        )
        st.subheader("Watchlist Heatmap")
        st.dataframe(
            display,
            hide_index=True,
            width="stretch",
            column_config={
                "1D": st.column_config.NumberColumn(format="%.2%"),
                "1M": st.column_config.NumberColumn(format="%.2%"),
                "約 10W": st.column_config.NumberColumn(format="%.2%"),
                "距 50DMA": st.column_config.NumberColumn(format="%.2%"),
                "距 200DMA": st.column_config.NumberColumn(format="%.2%"),
                "距 52W 高點": st.column_config.NumberColumn(format="%.2%"),
                "量/20日均量": st.column_config.NumberColumn(format="%.2fx"),
                "20D 年化波動": st.column_config.NumberColumn(format="%.2%"),
            },
        )

    with right:
        st.subheader("市場廣度")
        breadth = breadth_table(snapshot)
        st.dataframe(
            breadth.rename(
                columns={
                    "group": "類別",
                    "count": "數量",
                    "above_50dma": "高於50DMA",
                    "above_200dma": "高於200DMA",
                    "avg_1m_return": "平均1M",
                    "avg_drawdown_52w": "平均距52W高點",
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "平均1M": st.column_config.NumberColumn(format="%.2%"),
                "平均距52W高點": st.column_config.NumberColumn(format="%.2%"),
            },
        )

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

        st.subheader("預測追蹤")
        validation = prediction_validation_summary(prediction_log)
        if validation.empty:
            st.caption("預測紀錄已建立；等 5D / 20D / 60D 週期走完後，這裡會開始顯示成功率。")
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

        analog_display = analogs.copy()
        analog_display["date"] = pd.to_datetime(analog_display["date"]).dt.date
        st.dataframe(
            analog_display.rename(
                columns={
                    "date": "日期",
                    "similarity": "相似度",
                    "regime_snapshot": "當時狀態",
                    "1M": "後1M",
                    "3M": "後3M",
                    "6M": "後6M",
                    "12M": "後12M",
                }
            )[["日期", "相似度", "當時狀態", "後1M", "後3M", "後6M", "後12M"]],
            hide_index=True,
            width="stretch",
            column_config={
                "相似度": st.column_config.NumberColumn(format="%.2f"),
                "後1M": st.column_config.NumberColumn(format="%.2%"),
                "後3M": st.column_config.NumberColumn(format="%.2%"),
                "後6M": st.column_config.NumberColumn(format="%.2%"),
                "後12M": st.column_config.NumberColumn(format="%.2%"),
            },
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
        qcols = st.columns(4)
        qcols[0].metric("摘要品質", ai_quality.get("quality_label", "n/a"))
        qcols[1].metric("完整度", f"{ai_quality.get('section_count', 0)}/{ai_quality.get('required_sections', 6)}")
        qcols[2].metric("字數", f"{ai_quality.get('text_length', 0)}")
        qcols[3].metric("品質分", num(ai_quality.get("quality_score"), 0))
        if ai_quality.get("missing_sections") not in {None, "", "無"}:
            st.caption("缺少章節：" + str(ai_quality.get("missing_sections")))
        st.markdown(ai_summary.get("text", ""))
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

    st.markdown("#### 最近預測紀錄")
    recent_predictions = recent_prediction_table(prediction_log, limit=45)
    if recent_predictions.empty:
        st.caption("尚無預測紀錄。")
    else:
        st.dataframe(
            recent_predictions.rename(
                columns={
                    "prediction_date": "預測日",
                    "horizon": "週期",
                    "prediction_direction": "方向",
                    "confidence": "信心",
                    "regime_score": "Regime",
                    "reason": "理由",
                    "actual_return": "實際報酬",
                    "max_drawdown": "最大回撤",
                    "success": "成功",
                    "validated_at": "驗證日",
                }
            )[["預測日", "週期", "方向", "信心", "Regime", "理由", "實際報酬", "最大回撤", "成功", "驗證日"]],
            hide_index=True,
            width="stretch",
            column_config={
                "Regime": st.column_config.NumberColumn(format="%.0f"),
                "實際報酬": st.column_config.NumberColumn(format="%.2%"),
                "最大回撤": st.column_config.NumberColumn(format="%.2%"),
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

                st.markdown("#### B. 持倉明細表")
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
                st.dataframe(
                    detail,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "持股數量": st.column_config.NumberColumn(format="%.4f"),
                        "平均成本": st.column_config.NumberColumn(format="$%.2f"),
                        "現價": st.column_config.NumberColumn(format="$%.2f"),
                        "市值": st.column_config.NumberColumn(format="$%.0f"),
                        "未實現損益": st.column_config.NumberColumn(format="$%.0f"),
                        "未實現報酬率": st.column_config.NumberColumn(format="%.2%"),
                        "持倉占總資產比例": st.column_config.NumberColumn(format="%.2%"),
                        "市場熱度分數": st.column_config.NumberColumn(format="%.0f"),
                        "風險分數": st.column_config.NumberColumn(format="%.0f"),
                    },
                )

                st.markdown("#### C. 操作建議")
                st.markdown("##### 持倉風險分層")
                bucket = bucket_summary(portfolio_view)
                if not bucket.empty:
                    st.dataframe(
                        bucket.rename(
                            columns={
                                "position_bucket": "分層",
                                "market_value": "市值",
                                "asset_weight": "資產占比",
                                "avg_return": "平均報酬",
                                "avg_risk": "平均風險分",
                                "max_weight": "最大單檔占比",
                                "count": "檔數",
                            }
                        ),
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "市值": st.column_config.NumberColumn(format="$%.0f"),
                            "資產占比": st.column_config.NumberColumn(format="%.2%"),
                            "平均報酬": st.column_config.NumberColumn(format="%.2%"),
                            "平均風險分": st.column_config.NumberColumn(format="%.0f"),
                            "最大單檔占比": st.column_config.NumberColumn(format="%.2%"),
                        },
                    )
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
                st.dataframe(
                    advice,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "加倉價": st.column_config.NumberColumn(format="$%.2f"),
                        "減碼價": st.column_config.NumberColumn(format="$%.2f"),
                        "停損價": st.column_config.NumberColumn(format="$%.2f"),
                        "今日漲跌": st.column_config.NumberColumn(format="%.2%"),
                        "5日漲跌": st.column_config.NumberColumn(format="%.2%"),
                        "20日漲跌": st.column_config.NumberColumn(format="%.2%"),
                        "量/20日均量": st.column_config.NumberColumn(format="%.2fx"),
                        "新聞情緒分數": st.column_config.NumberColumn(format="%.2f"),
                    },
                )

                st.markdown("#### D. 風險警示")
                if portfolio_alerts.empty:
                    st.success("目前沒有觸發主要持倉警示。")
                else:
                    for alert in portfolio_alerts.itertuples():
                        st.error(f"{alert.ticker}：{alert.alerts}")
