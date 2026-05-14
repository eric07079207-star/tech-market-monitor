from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ai_summary import build_ai_summary
from src.config import ETF_TICKERS, NEWS_QUERIES, STOCK_TICKERS, default_start_date
from src.data import cache_path, load_cached_market_data, load_metadata, refresh_market_data
from src.indicators import (
    add_price_indicators,
    analog_stats,
    breadth_table,
    detect_anomalies,
    historical_analogs,
    latest_snapshot,
    regime_summary,
)
from src.news import fetch_news_batch


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


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.1%}"


def num(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"


with st.sidebar:
    st.header("設定")
    start_date = st.date_input("歷史起點", value=default_start_date())
    news_days = st.slider("新聞回看天數", 3, 30, 10)
    force_data = st.button("更新市場資料", width="stretch")
    force_news = st.button("更新新聞", width="stretch")
    use_ai = st.toggle("產生 AI 摘要", value=False)
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
metadata = load_metadata()

st.title("科技股量化監控儀表板")
last_date = pd.to_datetime(snapshot["date"]).max().date() if not snapshot.empty else None
updated_at = metadata.get("updated_at_utc", "尚未寫入")
st.markdown(f"<span class='small-muted'>市場資料日期：{last_date}｜快取更新 UTC：{updated_at}</span>", unsafe_allow_html=True)

top = st.columns([1.2, 1, 1, 1])
top[0].metric("Regime Score", f"{num(regime['score'], 0)}/100")
top[0].caption(regime["label"])

qqq = snapshot[snapshot["symbol"] == "QQQ"].squeeze()
smh = snapshot[snapshot["symbol"] == "SMH"].squeeze()
vix = snapshot[snapshot["symbol"] == "^VIX"].squeeze()
top[1].metric("QQQ 1M", pct(qqq.get("ret_20d") if isinstance(qqq, pd.Series) else np.nan), pct(qqq.get("ret_1d") if isinstance(qqq, pd.Series) else np.nan))
top[2].metric("SMH 1M", pct(smh.get("ret_20d") if isinstance(smh, pd.Series) else np.nan), pct(smh.get("ret_1d") if isinstance(smh, pd.Series) else np.nan))
top[3].metric("VIX", num(vix.get("close") if isinstance(vix, pd.Series) else np.nan, 1), pct(vix.get("ret_1d") if isinstance(vix, pd.Series) else np.nan))

if regime["drivers"]:
    st.caption(" / ".join(regime["drivers"]))

tab_overview, tab_anomaly, tab_analog, tab_news, tab_charts = st.tabs(
    ["總覽", "異常雷達", "歷史相似情境", "新聞與摘要", "走勢圖"]
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

with tab_anomaly:
    st.subheader("今日異常訊號")
    if anomalies.empty:
        st.info("目前 watchlist 沒有觸發主要異常規則。")
    else:
        anomaly_display = anomalies[
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

    st.subheader("下跌前風險線索")
    risk_notes = [
        "QQQ 跌破 200DMA 且 200DMA 斜率轉負",
        "QQQ 相對 SPY 連續 3 個月轉弱",
        "SMH 相對 QQQ 轉弱，半導體不再領先",
        "VIX 高於 30 或 20 日實現波動進入近一年高分位",
        "HYG 走弱或高收益債利差擴大",
        "少數大型股撐盤，但 watchlist 多數個股低於 50DMA",
    ]
    st.write("\n".join(f"- {note}" for note in risk_notes))

with tab_analog:
    st.subheader("QQQ 歷史相似情境")
    analogs = historical_analogs(indicators, target="QQQ", top_n=12)
    if analogs.empty:
        st.info("資料量不足，暫時無法計算歷史相似情境。")
    else:
        stats = analog_stats(analogs)
        st.caption(
            "這裡的上漲比例只代表目前最相似的 12 個歷史樣本中，後續報酬為正的比例；"
            "樣本很小，不是未來勝率或保證。"
        )
        stat_cols = st.columns(len(stats) if len(stats) else 1)
        for i, row in enumerate(stats.itertuples()):
            stat_cols[i].metric(
                row.horizon,
                pct(row.avg_return),
                f"樣本上漲 {pct(row.win_rate)}",
            )
            stat_cols[i].caption(
                f"N={row.sample}｜保守估計 {pct(row.win_rate_conservative)}｜最差 {pct(row.worst_return)}"
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
    summary = ""
    if use_ai:
        with st.spinner("產生摘要..."):
            summary, used_ai = build_ai_summary(snapshot, anomalies, news)
        st.subheader("AI 市場摘要" if used_ai else "規則摘要")
        st.markdown(summary)
    else:
        from src.news import rule_based_news_summary

        st.subheader("規則摘要")
        st.markdown(rule_based_news_summary(news))

    st.subheader("新聞標籤與連結")
    if news.empty:
        st.info("目前沒有抓到近期新聞。")
    else:
        filtered_symbols = st.multiselect("標的", options=list(NEWS_QUERIES), default=list(NEWS_QUERIES)[:6])
        news_view = news[news["symbol"].isin(filtered_symbols)] if filtered_symbols else news
        for row in news_view.head(80).itertuples():
            published = row.published.strftime("%Y-%m-%d %H:%M") if pd.notna(row.published) else ""
            st.markdown(f"**{row.symbol}** · `{row.tags}` · {row.source} · {published}  \n[{row.title}]({row.link})")

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
