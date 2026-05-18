from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:  # pragma: no cover - optional local dependency
    st_autorefresh = None

from src.ai_summary import build_ai_summary
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
from src.data import cache_path, load_cached_market_data, load_metadata, refresh_market_data
try:
    from src.discovery import build_discovery_candidates, fetch_discovery_news
except ImportError:  # pragma: no cover - protects Streamlit Cloud during partial redeploys
    def fetch_discovery_news(days: int = 7, topics_per_day: int = 5, limit_per_topic: int = 7) -> pd.DataFrame:
        return pd.DataFrame(columns=["topic", "symbol", "title", "source", "published", "tags", "link", "tickers"])

    def build_discovery_candidates(news: pd.DataFrame, lookback_days: int = 180, top_n: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
        return pd.DataFrame(), pd.DataFrame()
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
    from src.news import fetch_international_news, international_news_selection, portfolio_news_impact
except ImportError:  # pragma: no cover - protects Streamlit Cloud during partial redeploys
    def fetch_international_news(days: int = 3, limit_per_topic: int = 8) -> pd.DataFrame:
        return pd.DataFrame(columns=["symbol", "title", "source", "published", "tags", "link", "is_major", "priority"])

    def international_news_selection(news: pd.DataFrame, random_count: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
        return news.copy(), news.copy()

    def portfolio_news_impact(news: pd.DataFrame, portfolio_view: pd.DataFrame | None = None, max_items: int = 8) -> pd.DataFrame:
        return pd.DataFrame(columns=["ticker", "impact_level", "impact", "headline_count", "key_tags", "sample_headline"])

from src.portfolio import build_portfolio_view, fetch_portfolio_prices, load_portfolio_config
try:
    from src.portfolio import attention_positions
except ImportError:  # pragma: no cover - protects Streamlit Cloud during partial redeploys
    def attention_positions(view: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
        return pd.DataFrame()
try:
    from src.predictions import build_market_prediction, load_prediction_log, prediction_validation_summary
except ImportError:  # pragma: no cover - protects Streamlit Cloud during partial redeploys
    def build_market_prediction(regime: dict, conclusion: dict, snapshot: pd.DataFrame) -> dict:
        return {"target": "QQQ", "prediction_direction": "資料同步中"}

    def load_prediction_log() -> pd.DataFrame:
        return pd.DataFrame()

    def prediction_validation_summary(log: pd.DataFrame) -> pd.DataFrame:
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
def load_market(force_refresh: bool, start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if force_refresh:
        return refresh_market_data(start=start)
    return load_cached_market_data(start=start, force_refresh=False)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 3)
def load_news(force_refresh: bool, days: int) -> pd.DataFrame:
    news_path = cache_path("news.parquet")
    if not force_refresh and news_path.exists():
        news = pd.read_parquet(news_path)
        news["published"] = pd.to_datetime(news["published"], utc=True, errors="coerce")
        return news
    news = fetch_news_batch(symbols=list(NEWS_QUERIES), days=days, limit_per_symbol=8)
    if not news.empty:
        news.to_parquet(news_path, index=False)
    return news


@st.cache_data(show_spinner=False, ttl=60 * 60 * 3)
def load_international_news(force_refresh: bool, days: int) -> pd.DataFrame:
    news_path = cache_path("international_news.parquet")
    if not force_refresh and news_path.exists():
        news = pd.read_parquet(news_path)
        news["published"] = pd.to_datetime(news["published"], utc=True, errors="coerce")
        return news
    news = fetch_international_news(days=min(days, 7), limit_per_topic=8)
    if not news.empty:
        news.to_parquet(news_path, index=False)
    return news


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def load_discovery(force_refresh: bool, days: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    news_path = cache_path("discovery_news.parquet")
    candidates_path = cache_path("discovery_candidates.parquet")
    mentions_path = cache_path("discovery_mentions.parquet")
    if not force_refresh and news_path.exists() and candidates_path.exists():
        discovery_news = pd.read_parquet(news_path)
        candidates = pd.read_parquet(candidates_path)
        mentions = pd.read_parquet(mentions_path) if mentions_path.exists() else pd.DataFrame()
        for data, column in [(discovery_news, "published"), (mentions, "published")]:
            if not data.empty and column in data:
                data[column] = pd.to_datetime(data[column], utc=True, errors="coerce")
        return discovery_news, mentions, candidates
    discovery_news = fetch_discovery_news(days=min(days, 7), topics_per_day=5, limit_per_topic=7)
    mentions, candidates = build_discovery_candidates(discovery_news, top_n=12)
    if not discovery_news.empty:
        discovery_news.to_parquet(news_path, index=False)
    if not mentions.empty:
        mentions.to_parquet(mentions_path, index=False)
    if not candidates.empty:
        candidates.to_parquet(candidates_path, index=False)
    return discovery_news, mentions, candidates


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


with st.sidebar:
    st.header("設定")
    start_date = st.date_input("歷史起點", value=default_start_date())
    news_days = st.slider("新聞回看天數", 3, 30, 10)
    force_data = st.button("更新市場資料", width="stretch")
    force_news = st.button("更新新聞", width="stretch")
    use_ai = st.toggle("產生 AI 摘要", value=False)
    show_health = st.button("顯示資料健康檢查", width="stretch")
    st.caption("每日收盤後可執行 `scripts/update_data.py` 更新快取。")

with st.spinner("讀取市場資料..."):
    prices, macro = load_market(force_data, str(start_date))

if prices.empty:
    st.error("目前沒有市場資料。請稍後再按一次更新市場資料。")
    st.stop()

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
    with st.spinner("整理資料健康檢查..."):
        health_news = load_news(False, news_days)
        health_international = load_international_news(False, min(news_days, 7))
        health_discovery_news, _, health_discovery_candidates = load_discovery(False, news_days)
    st.subheader("資料健康檢查")
    st.dataframe(
        data_health_report(
            prices,
            macro,
            health_news,
            health_international,
            prediction_log,
            metadata,
            health_discovery_news,
            health_discovery_candidates,
        ),
        hide_index=True,
        width="stretch",
    )
    missing = missing_price_symbols(prices, ETF_TICKERS + STOCK_TICKERS + ANNUAL_PICK_TICKERS + ["SPY", "QQQ"])
    if missing:
        st.warning("缺少價格資料：" + ", ".join(missing[:20]))
    else:
        st.success("主要追蹤標的價格資料完整。")

tab_overview, tab_anomaly, tab_analog, tab_news, tab_discovery, tab_charts, tab_portfolio = st.tabs(
    ["總覽", "異常雷達", "歷史相似情境", "新聞與摘要", "新聞探索", "走勢圖", "我的持倉"]
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
            with st.spinner("讀取消息異常..."):
                news_for_anomaly = load_news(force_news, news_days)
            news_anomaly = news_for_anomaly[
                news_for_anomaly["tags"].astype(str).str.contains("大盤風險|監管/訴訟|財報/財測|國際", na=False)
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
    with st.spinner("讀取新聞..."):
        news = load_news(force_news, news_days)
        international_news = load_international_news(force_news, min(news_days, 7))
    summary = ""
    if use_ai:
        with st.spinner("產生摘要..."):
            summary, used_ai, ai_status = build_ai_summary(snapshot, anomalies, news)
        st.subheader("AI 市場摘要" if used_ai else "規則摘要")
        (st.success if used_ai else st.warning)(ai_status)
        st.markdown(summary)
    else:
        from src.news import rule_based_news_summary

        st.subheader("規則摘要")
        st.caption("AI 摘要目前未開啟；這裡使用規則摘要，不會消耗 AI 額度。")
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

with tab_discovery:
    st.subheader("新聞探索候選股")
    st.caption("每天用日期作為隨機種子抽取市場主題，從新聞中找 ticker，再用量價規則評分；這是觀察清單，不是買入建議。")
    refresh_discovery = st.button("重新整理新聞探索", width="stretch")
    if refresh_discovery:
        load_discovery.clear()
    with st.spinner("讀取新聞探索資料..."):
        discovery_news, discovery_mentions, discovery_candidates = load_discovery(force_news or refresh_discovery, news_days)

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
        st.markdown("#### 候選觀察股 Top 12")
        if discovery_candidates.empty:
            st.info("目前沒有從探索新聞抽到可驗證的候選股。")
        else:
            st.dataframe(
                discovery_candidates.rename(
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
                )[
                    ["股票", "相關主題", "候選分數", "分數解讀", "現價", "5日", "20日", "量/20日均量", "距50DMA", "相對QQQ", "風險標籤", "觀察理由", "代表新聞"]
                ],
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
            refresh_portfolio = st.button("立即刷新持倉資料", width="stretch")
            if refresh_portfolio:
                load_portfolio_prices.clear()
                load_portfolio_news.clear()

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
