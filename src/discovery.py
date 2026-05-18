from __future__ import annotations

import re
from datetime import date

import numpy as np
import pandas as pd

from .config import DISCOVERY_NEWS_TOPICS
from .data import cache_path, fetch_price_history
from .news import fetch_google_news


FALSE_TICKERS = {
    "AI", "CEO", "CFO", "COO", "USA", "SEC", "GDP", "IPO", "ETF", "FED", "FBI", "DOJ", "FDA",
    "EPS", "EBITDA", "NYSE", "NASDAQ", "US", "EU", "UK", "Q", "A", "I", "AM", "PM", "THE",
    "YTD", "SPAC", "SLIM", "EV", "ETF", "CEO", "CAN", "MSN",
}


def fetch_discovery_news(days: int = 7, topics_per_day: int = 5, limit_per_topic: int = 7) -> pd.DataFrame:
    topics = _daily_topics(topics_per_day)
    rows = []
    seen = set()
    for topic, query in topics.items():
        for item in fetch_google_news(topic, query, days=days, limit=limit_per_topic):
            if item.title in seen:
                continue
            seen.add(item.title)
            rows.append(
                {
                    "topic": topic,
                    "symbol": topic,
                    "title": item.title,
                    "source": item.source,
                    "published": item.published,
                    "tags": item.tags,
                    "link": item.link,
                    "tickers": ",".join(extract_tickers(item.title)),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["topic", "symbol", "title", "source", "published", "tags", "link", "tickers"])
    news = pd.DataFrame(rows)
    news["published"] = pd.to_datetime(news["published"], utc=True, errors="coerce")
    return news.sort_values(["published", "topic"], ascending=[False, True]).reset_index(drop=True)


def build_discovery_candidates(news: pd.DataFrame, lookback_days: int = 180, top_n: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    if news.empty or "tickers" not in news:
        return pd.DataFrame(), pd.DataFrame()

    mentions = []
    for row in news.itertuples():
        for ticker in str(row.tickers).split(","):
            ticker = ticker.strip().upper()
            if not ticker:
                continue
            mentions.append(
                {
                    "ticker": ticker,
                    "topic": row.topic,
                    "title": row.title,
                    "source": row.source,
                    "published": row.published,
                    "tags": row.tags,
                    "link": row.link,
                }
            )
    if not mentions:
        return pd.DataFrame(), pd.DataFrame()

    mention_df = pd.DataFrame(mentions)
    tickers = mention_df["ticker"].drop_duplicates().head(40).tolist()
    start = (pd.Timestamp.today().normalize() - pd.DateOffset(days=lookback_days)).date()
    prices = fetch_price_history(tickers=tickers + ["SPY", "QQQ"], start=start)
    if prices.empty:
        return mention_df, pd.DataFrame()

    metrics = _candidate_metrics(prices, mention_df)
    if metrics.empty:
        return mention_df, metrics
    return mention_df, metrics.sort_values("candidate_score", ascending=False).head(top_n).reset_index(drop=True)


def extract_tickers(text: str) -> list[str]:
    candidates = re.findall(r"(?<![A-Za-z])\$?([A-Z]{1,5})(?![A-Za-z])", text or "")
    result = []
    for ticker in candidates:
        if len(ticker) == 1:
            continue
        if ticker in FALSE_TICKERS:
            continue
        if ticker not in result:
            result.append(ticker)
    return result[:8]


def _daily_topics(count: int) -> dict[str, str]:
    seed = int(pd.Timestamp.today(tz="UTC").strftime("%Y%m%d"))
    keys = pd.Series(list(DISCOVERY_NEWS_TOPICS)).sample(
        n=min(count, len(DISCOVERY_NEWS_TOPICS)),
        random_state=seed,
    ).tolist()
    return {key: DISCOVERY_NEWS_TOPICS[key] for key in keys}


def _candidate_metrics(prices: pd.DataFrame, mentions: pd.DataFrame) -> pd.DataFrame:
    wide = prices.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index().ffill()
    rows = []
    for ticker, group in mentions.groupby("ticker", sort=False):
        data = prices[prices["symbol"].eq(ticker)].sort_values("date").copy()
        if len(data) < 60 or ticker not in wide:
            continue
        data["ret_1d"] = data["close"].pct_change()
        data["ret_5d"] = data["close"].pct_change(5)
        data["ret_20d"] = data["close"].pct_change(20)
        data["ma_50"] = data["close"].rolling(50).mean()
        data["ma_200"] = data["close"].rolling(200, min_periods=120).mean()
        data["dist_ma_50"] = data["close"] / data["ma_50"] - 1
        data["dist_ma_200"] = data["close"] / data["ma_200"] - 1
        data["volume_ratio_20d"] = data["volume"] / data["volume"].rolling(20).mean()
        data["high_126d"] = data["close"].rolling(126).max()
        data["drawdown_6m"] = data["close"] / data["high_126d"] - 1
        latest = data.dropna(subset=["close"]).tail(1).iloc[0]
        rel_spy = _relative_return(wide, ticker, "SPY", 20)
        rel_qqq = _relative_return(wide, ticker, "QQQ", 20)
        titles = " ".join(group["title"].fillna("").astype(str).str.lower())
        negative = any(word in titles for word in ["offering", "dilution", "guidance cut", "miss", "lawsuit", "probe"])
        score = 35
        score += min(len(group) * 5, 18)
        score += np.clip(_num(latest.get("ret_20d")) * 120, -15, 18)
        score += 10 if _num(latest.get("dist_ma_50")) > 0 else -8
        score += 8 if _num(latest.get("dist_ma_200")) > 0 else -5
        score += np.clip((_num(latest.get("volume_ratio_20d"), 1) - 1) * 10, -6, 12)
        score += np.clip(_num(rel_spy) * 100, -8, 10)
        score += np.clip(_num(rel_qqq) * 80, -8, 10)
        if _num(latest.get("ret_5d")) > 0.2 or _num(latest.get("volume_ratio_20d"), 1) > 3:
            score -= 8
        if negative:
            score -= 15
        score = float(np.clip(score, 0, 100))
        rows.append(
            {
                "ticker": ticker,
                "topic": "、".join(group["topic"].drop_duplicates().head(3).tolist()),
                "headline_count": int(len(group)),
                "sample_headline": group.sort_values("published", ascending=False).iloc[0]["title"],
                "source": group.sort_values("published", ascending=False).iloc[0]["source"],
                "link": group.sort_values("published", ascending=False).iloc[0]["link"],
                "current_price": latest["close"],
                "ret_5d": latest.get("ret_5d", np.nan),
                "ret_20d": latest.get("ret_20d", np.nan),
                "volume_ratio_20d": latest.get("volume_ratio_20d", np.nan),
                "dist_ma_50": latest.get("dist_ma_50", np.nan),
                "dist_ma_200": latest.get("dist_ma_200", np.nan),
                "drawdown_6m": latest.get("drawdown_6m", np.nan),
                "rel_spy_20d": rel_spy,
                "rel_qqq_20d": rel_qqq,
                "risk_flags": _risk_flags(latest, negative),
                "candidate_score": score,
                "candidate_label": _score_label(score),
                "observation_reason": _observation_reason(score, latest, group, negative),
            }
        )
    return pd.DataFrame(rows)


def _relative_return(wide: pd.DataFrame, ticker: str, benchmark: str, window: int) -> float:
    if ticker not in wide or benchmark not in wide:
        return np.nan
    rel = (wide[ticker] / wide[benchmark]).pct_change(window).dropna()
    return float(rel.iloc[-1]) if not rel.empty else np.nan


def _risk_flags(latest: pd.Series, negative: bool) -> str:
    flags = []
    if _num(latest.get("ret_5d")) > 0.2:
        flags.append("短線漲幅過大")
    if _num(latest.get("volume_ratio_20d"), 1) > 3:
        flags.append("量能過熱")
    if _num(latest.get("dist_ma_50")) < 0:
        flags.append("低於50DMA")
    if negative:
        flags.append("負面關鍵字")
    return "；".join(flags) if flags else "未觸發主要風險"


def _score_label(score: float) -> str:
    if score >= 80:
        return "高優先觀察"
    if score >= 60:
        return "可觀察"
    if score >= 40:
        return "題材有熱度但訊號普通"
    if score >= 20:
        return "暫時不強"
    return "低優先或資料不足"


def _observation_reason(score: float, latest: pd.Series, group: pd.DataFrame, negative: bool) -> str:
    if negative:
        return "新聞有負面關鍵字，僅列入風險觀察。"
    if score >= 70 and _num(latest.get("dist_ma_50")) > 0:
        return f"新聞提及 {len(group)} 次，且價格站上 50DMA，值得加入觀察清單。"
    if _num(latest.get("ret_5d")) > 0.2:
        return "題材熱但短線漲幅偏大，等待回檔或量價整理。"
    return f"新聞提及 {len(group)} 次，量價訊號尚需確認。"


def _num(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default
