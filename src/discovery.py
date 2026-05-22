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
    "YTD", "SPAC", "SLIM", "EV", "ETF", "CEO", "CAN", "MSN", "CNN", "CNBC", "AOL", "INC",
    "LLC", "LTD", "PLC", "CORP", "CO", "ADR", "ADS", "ETF", "ETN", "IPO", "SPAC",
    "TSE", "TSX", "LSE", "HKEX", "OTC", "CBOE", "AMEX", "NIKKEI", "DAX", "CAC",
    "API", "SAAS", "USD", "EUR", "CPI", "PPI", "FOMC", "ISM", "PMI", "OPEC",
    "WHO", "UN", "NATO", "GOP", "IRS", "FTC", "EURO", "AP", "PR", "DJIA", "ISG",
    "SHS", "NV", "NY", "HDFC", "SP",
}

TICKER_CONTEXT_WORDS = {
    "stock", "stocks", "share", "shares", "equity", "equities", "ticker", "ipo", "earnings",
    "revenue", "profit", "sales", "guidance", "outlook", "upgrade", "downgrade", "price target",
    "analyst", "market cap", "trading", "surge", "rally", "plunge", "slump", "breakout",
    "nasdaq", "nyse", "amex", "quarter", "q1", "q2", "q3", "q4",
}

EXPLICIT_TICKER_PATTERNS = [
    re.compile(r"(?<![A-Za-z0-9])\$([A-Z][A-Z0-9.]{1,5})(?![A-Za-z0-9])"),
    re.compile(r"\b(?:NASDAQ|NYSE|AMEX|CBOE)\s*:\s*([A-Z][A-Z0-9.]{1,5})\b"),
    re.compile(r"\(([A-Z][A-Z0-9.]{1,5})\)"),
]

BARE_TICKER_PATTERN = re.compile(r"(?<![A-Za-z])([A-Z]{2,5})(?![A-Za-z])")


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


def build_discovery_candidates(news: pd.DataFrame, lookback_days: int = 180, top_n: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
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
                    "matched_keywords": getattr(row, "matched_keywords", ""),
                    "keyword_group": getattr(row, "keyword_group", ""),
                    "analysis_note": getattr(row, "analysis_note", ""),
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


def update_discovery_history(candidates: pd.DataFrame, path=None, run_date: date | str | None = None) -> pd.DataFrame:
    path = path or cache_path("discovery_history.parquet")
    history = load_discovery_history(path)
    if candidates.empty:
        return history

    current_date = pd.to_datetime(run_date or pd.Timestamp.today(tz="UTC").date()).date().isoformat()
    today = candidates.copy().head(15)
    today["date"] = current_date
    ordered_cols = ["date"] + [col for col in today.columns if col != "date"]
    today = today[ordered_cols]

    if not history.empty and "date" in history:
        history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.date.astype(str)
        history = history[history["date"] != current_date]
    history = pd.concat([history, today], ignore_index=True)
    if "date" in history:
        history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.date.astype(str)
    path.parent.mkdir(parents=True, exist_ok=True)
    history.to_parquet(path, index=False)
    return history


def load_discovery_history(path=None) -> pd.DataFrame:
    path = path or cache_path("discovery_history.parquet")
    if not path.exists():
        return pd.DataFrame()
    history = pd.read_parquet(path)
    if "date" in history:
        history["date"] = pd.to_datetime(history["date"], errors="coerce")
    return history


def summarize_discovery_history(history: pd.DataFrame, days: int, top_n: int = 15) -> pd.DataFrame:
    if history.empty or "date" not in history:
        return pd.DataFrame()
    data = history.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=days - 1)
    data = data[data["date"] >= cutoff]
    if data.empty:
        return pd.DataFrame()

    rows = []
    for ticker, group in data.groupby("ticker", sort=False):
        risk_hits = group["risk_flags"].fillna("").astype(str).ne("未觸發主要風險").sum() if "risk_flags" in group else 0
        topics = sorted(set("、".join(group.get("topic", pd.Series(dtype=str)).dropna().astype(str)).split("、")) - {""})
        avg_score = group["candidate_score"].mean()
        max_score = group["candidate_score"].max()
        appearance_days = group["date"].dt.date.nunique()
        headline_count = group.get("headline_count", pd.Series(0, index=group.index)).sum()
        avg_rel_qqq = group.get("rel_qqq_20d", pd.Series(dtype=float)).mean()
        rank_score = (
            avg_score * 0.38
            + max_score * 0.18
            + min(appearance_days, days) * 4.0
            + min(len(topics), 5) * 3.0
            + min(headline_count, 20) * 1.2
            + _num(avg_rel_qqq) * 80
            - risk_hits * 4.0
        )
        latest = group.sort_values("date").iloc[-1]
        rows.append(
            {
                "ticker": ticker,
                "rank_score": float(np.clip(rank_score, 0, 100)),
                "appearance_days": int(appearance_days),
                "avg_candidate_score": float(avg_score),
                "max_candidate_score": float(max_score),
                "topic_count": int(len(topics)),
                "topics": "、".join(topics[:4]),
                "headline_count": int(headline_count),
                "avg_rel_qqq": float(avg_rel_qqq) if pd.notna(avg_rel_qqq) else np.nan,
                "risk_count": int(risk_hits),
                "latest_reason": latest.get("observation_reason", ""),
                "latest_risk": latest.get("risk_flags", ""),
                "sample_headline": latest.get("sample_headline", ""),
            }
        )
    return pd.DataFrame(rows).sort_values("rank_score", ascending=False).head(top_n).reset_index(drop=True)


def update_discovery_performance(history: pd.DataFrame, path=None) -> pd.DataFrame:
    path = path or cache_path("discovery_performance.parquet")
    if history.empty or "ticker" not in history or "date" not in history:
        return pd.DataFrame()
    entries = history.copy()
    entries["date"] = pd.to_datetime(entries["date"], errors="coerce").dt.normalize()
    tickers = entries["ticker"].dropna().astype(str).str.upper().drop_duplicates().tolist()
    start = entries["date"].min()
    if pd.isna(start) or not tickers:
        return pd.DataFrame()
    prices = fetch_price_history(tickers=tickers + ["QQQ"], start=start.date())
    performance = discovery_performance_table(entries, prices)
    if not performance.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        performance.to_parquet(path, index=False)
    return performance


def load_discovery_performance(path=None) -> pd.DataFrame:
    path = path or cache_path("discovery_performance.parquet")
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_parquet(path)
    for col in ["date", "validated_at"]:
        if col in data:
            data[col] = pd.to_datetime(data[col], errors="coerce")
    return data


def discovery_performance_table(history: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if history.empty or prices.empty:
        return pd.DataFrame()
    wide = prices.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index().ffill()
    rows = []
    for entry in history.to_dict("records"):
        ticker = str(entry.get("ticker", "")).upper()
        entry_date = pd.to_datetime(entry.get("date"), errors="coerce")
        if not ticker or ticker not in wide or pd.isna(entry_date):
            continue
        series = wide[ticker].dropna()
        qqq = wide["QQQ"].dropna() if "QQQ" in wide else pd.Series(dtype=float)
        start_pos = series.index.searchsorted(entry_date)
        if start_pos >= len(series):
            continue
        start_date = series.index[start_pos]
        start_price = float(series.iloc[start_pos])
        qqq_start = qqq.iloc[qqq.index.searchsorted(start_date)] if not qqq.empty and qqq.index.searchsorted(start_date) < len(qqq) else np.nan
        base = {key: entry.get(key) for key in ["date", "ticker", "candidate_score", "candidate_label", "topic", "risk_flags", "observation_reason", "sample_headline"]}
        base["entry_price"] = start_price
        for label, days in [("5D", 5), ("20D", 20), ("60D", 60)]:
            end_pos = start_pos + days
            row = base.copy()
            row["horizon"] = label
            row["horizon_days"] = days
            if end_pos < len(series):
                end_date = series.index[end_pos]
                end_price = float(series.iloc[end_pos])
                actual_return = end_price / start_price - 1
                qqq_return = np.nan
                if not qqq.empty and pd.notna(qqq_start):
                    qqq_end_pos = qqq.index.searchsorted(end_date)
                    if qqq_end_pos < len(qqq):
                        qqq_return = float(qqq.iloc[qqq_end_pos] / qqq_start - 1)
                row.update(
                    {
                        "validated_at": end_date,
                        "actual_return": actual_return,
                        "qqq_return": qqq_return,
                        "relative_qqq_return": actual_return - qqq_return if pd.notna(qqq_return) else np.nan,
                        "success": actual_return > 0,
                    }
                )
            else:
                row.update({"validated_at": pd.NaT, "actual_return": np.nan, "qqq_return": np.nan, "relative_qqq_return": np.nan, "success": np.nan})
            rows.append(row)
    return pd.DataFrame(rows)


def discovery_performance_summary(performance: pd.DataFrame) -> pd.DataFrame:
    if performance.empty or "success" not in performance:
        return pd.DataFrame()
    done = performance.dropna(subset=["success"]).copy()
    if done.empty:
        return pd.DataFrame()
    done["success"] = done["success"].astype(bool)
    return (
        done.groupby("horizon")
        .agg(
            sample=("success", "size"),
            success_rate=("success", "mean"),
            avg_return=("actual_return", "mean"),
            avg_relative_qqq=("relative_qqq_return", "mean"),
        )
        .reset_index()
    )


def extract_tickers(text: str) -> list[str]:
    result = []
    text = text or ""

    for pattern in EXPLICIT_TICKER_PATTERNS:
        for match in pattern.finditer(text):
            _append_ticker(result, match.group(1))

    for match in BARE_TICKER_PATTERN.finditer(text):
        if _has_ticker_context(text, match.start(), match.end()):
            _append_ticker(result, match.group(1))
    return result[:8]


def _append_ticker(result: list[str], ticker: str) -> None:
    ticker = ticker.strip().upper().replace(".", "-")
    if len(ticker) == 1:
        return
    if ticker in FALSE_TICKERS:
        return
    if not re.fullmatch(r"[A-Z][A-Z0-9-]{1,5}", ticker):
        return
    if ticker not in result:
        result.append(ticker)


def _has_ticker_context(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - 70) : min(len(text), end + 70)].lower()
    return any(word in window for word in TICKER_CONTEXT_WORDS)


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
                "matched_keywords": "、".join(sorted(set("、".join(group.get("matched_keywords", pd.Series(dtype=str)).fillna("").astype(str)).split("、")) - {""}))[:160],
                "keyword_group": "、".join(group.get("keyword_group", pd.Series(dtype=str)).fillna("").astype(str).replace("", pd.NA).dropna().drop_duplicates().head(3).tolist()),
                "analysis_note": " ".join(group.get("analysis_note", pd.Series(dtype=str)).fillna("").astype(str).replace("", pd.NA).dropna().drop_duplicates().head(2).tolist()),
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
