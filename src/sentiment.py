from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import CACHE_DIR


SENTIMENT_CACHE = CACHE_DIR / "sentiment.parquet"
EVENT_WINDOWS_CACHE = CACHE_DIR / "market_event_windows.parquet"

FEAR_TERMS = ["fear", "panic", "selloff", "warning", "recession", "risk", "stress", "crash"]
HYPE_TERMS = ["ai", "record", "surge", "breakout", "boom", "upgrade", "beat", "strong demand"]
POLICY_TERMS = ["fed", "rate", "inflation", "cpi", "pce", "yield", "tariff", "trade", "sanction", "powell"]


@dataclass(frozen=True)
class HistoricalWindow:
    key: str
    label: str
    start: str
    end: str
    description: str


HISTORICAL_WINDOWS = [
    HistoricalWindow("gfc_2008_2009", "金融危機", "2008-01-01", "2009-12-31", "信用危機與流動性壓力期"),
    HistoricalWindow("trade_war_2018_2019", "貿易戰/關稅", "2018-01-01", "2019-12-31", "中美貿易戰與科技供應鏈壓力"),
    HistoricalWindow("covid_2020_2021", "疫情/流動性衝擊", "2020-01-01", "2021-12-31", "疫情崩跌、流動性危機與極速反彈"),
    HistoricalWindow("rates_ai_2022_2024", "升息+AI主線", "2022-01-01", "2024-12-31", "升息重估與 ChatGPT 問世後 AI 敘事擴散"),
]


def build_sentiment_features(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    news: pd.DataFrame | None = None,
    international_news: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "vix_level",
                "vix_5d_change",
                "qqq_spy_rel_63d",
                "hyg_tlt_rel_20d",
                "hy_oas_level",
                "hy_oas_20d_change",
                "curve_10y2y",
                "news_count",
                "news_fear_score",
                "news_hype_score",
                "policy_risk_score",
                "vix_percentile_252d",
                "hy_oas_percentile_252d",
                "fear_percentile_252d",
                "hype_percentile_252d",
                "policy_percentile_252d",
                "market_mood_score",
                "market_mood_label",
                "regime_window",
                "data_origin",
            ]
        )

    wide_close = _wide_prices(prices, "close")
    close = pd.DataFrame(index=wide_close.index)
    close["vix_level"] = wide_close.get("^VIX")
    close["vix_5d_change"] = close["vix_level"].pct_change(5)
    close["qqq_spy_rel_63d"] = _relative_return(wide_close, "QQQ", "SPY", 63)
    close["hyg_tlt_rel_20d"] = _relative_return(wide_close, "HYG", "TLT", 20)
    close["qqq_voo_rel_20d"] = _relative_return(wide_close, "QQQ", "VOO", 20)

    macro_wide = _wide_macro(macro)
    close["hy_oas_level"] = macro_wide.get("BAMLH0A0HYM2")
    close["hy_oas_20d_change"] = close["hy_oas_level"].diff(20)
    close["curve_10y2y"] = macro_wide.get("T10Y2Y")
    close["cpi_yoy_proxy"] = _year_over_year(macro_wide.get("CPIAUCSL"))
    pce_series = macro_wide.get("PCEPI")
    if pce_series is not None and not pce_series.dropna().empty:
        close["pce_yoy_proxy"] = pce_series.where(pce_series.abs() < 20, pce_series.pct_change(12))
    else:
        close["pce_yoy_proxy"] = pd.Series(index=close.index, dtype=float)
    close["unemployment_rate"] = macro_wide.get("UNRATE")
    close["nonfarm_payrolls"] = macro_wide.get("PAYEMS")
    close["initial_jobless_claims"] = macro_wide.get("ICSA")

    news_features = _daily_news_sentiment(news, international_news, close.index)
    sentiment = close.join(news_features, how="left").ffill()

    percentile_columns = {
        "vix_level": "vix_percentile_252d",
        "hy_oas_level": "hy_oas_percentile_252d",
        "news_fear_score": "news_fear_percentile_252d",
        "news_hype_score": "news_hype_percentile_252d",
        "policy_risk_score": "policy_risk_percentile_252d",
    }
    for source_col, target_col in percentile_columns.items():
        sentiment[target_col] = _rolling_percentile(sentiment[source_col], 252)

    risk_off = (
        0.30 * sentiment["vix_percentile_252d"].fillna(0.5)
        + 0.25 * sentiment["hy_oas_percentile_252d"].fillna(0.5)
        + 0.20 * sentiment["news_fear_percentile_252d"].fillna(0.5)
        + 0.15 * sentiment["policy_risk_percentile_252d"].fillna(0.5)
        - 0.10 * sentiment["news_hype_percentile_252d"].fillna(0.5)
    )
    rel_boost = np.clip(sentiment["qqq_spy_rel_63d"].fillna(0) * 2.5, -0.2, 0.2)
    credit_boost = np.clip(sentiment["hyg_tlt_rel_20d"].fillna(0) * 2.0, -0.2, 0.2)
    mood_score = 100 * np.clip(0.5 - risk_off + rel_boost + credit_boost, 0, 1)
    sentiment["market_mood_score"] = mood_score.round(2)
    sentiment["market_mood_label"] = sentiment["market_mood_score"].apply(_mood_label)
    sentiment["regime_window"] = sentiment.index.to_series().apply(classify_historical_window)
    sentiment["data_origin"] = "reconstructed"

    sentiment = sentiment.reset_index().rename(columns={"index": "date"})
    sentiment["date"] = pd.to_datetime(sentiment["date"], errors="coerce")
    return sentiment.sort_values("date").reset_index(drop=True)


def build_market_event_windows(prices: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    wide_close = _wide_prices(prices, "close")
    sentiment_daily = sentiment.set_index("date") if not sentiment.empty and "date" in sentiment else pd.DataFrame()

    rows: list[dict] = []
    for symbol in ["VOO", "QQQ"]:
        if symbol not in wide_close.columns:
            continue
        series = wide_close[symbol].dropna()
        if len(series) < 25:
            continue
        last_kept_end: pd.Timestamp | None = None
        for idx in range(20, len(series)):
            start_date = series.index[idx - 20]
            end_date = series.index[idx]
            if last_kept_end is not None and start_date <= last_kept_end:
                continue
            start_price = float(series.iloc[idx - 20])
            end_price = float(series.iloc[idx])
            window_return = end_price / start_price - 1
            if abs(window_return) < 0.10:
                continue
            window_slice = series.iloc[idx - 20 : idx + 1]
            max_drawdown = float((window_slice / window_slice.cummax() - 1).min())
            max_rebound = float((window_slice / window_slice.cummin() - 1).max())
            snap = sentiment_daily.loc[end_date] if end_date in sentiment_daily.index else pd.Series(dtype=float)
            rows.append(
                {
                    "symbol": symbol,
                    "window_type": "up_10_event" if window_return > 0 else "down_10_event",
                    "direction": "up" if window_return > 0 else "down",
                    "start_date": start_date,
                    "end_date": end_date,
                    "trading_days": 20,
                    "window_return": window_return,
                    "max_drawdown": max_drawdown,
                    "max_rebound": max_rebound,
                    "vix_level": _safe_float(snap.get("vix_level")),
                    "hy_oas_level": _safe_float(snap.get("hy_oas_level")),
                    "news_fear_score": _safe_float(snap.get("news_fear_score")),
                    "news_hype_score": _safe_float(snap.get("news_hype_score")),
                    "policy_risk_score": _safe_float(snap.get("policy_risk_score")),
                    "market_mood_score": _safe_float(snap.get("market_mood_score")),
                    "market_mood_label": str(snap.get("market_mood_label", "") or ""),
                    "historical_window": classify_historical_window(end_date),
                    "data_origin": "reconstructed",
                }
            )
            last_kept_end = end_date

    if not rows:
        return pd.DataFrame()
    windows = pd.DataFrame(rows).sort_values(["end_date", "symbol"], ascending=[False, True]).reset_index(drop=True)
    return windows


def classify_historical_window(value: pd.Timestamp | str | None) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return "outside_named_windows"
    for window in HISTORICAL_WINDOWS:
        if pd.Timestamp(window.start) <= ts <= pd.Timestamp(window.end):
            return window.label
    if ts >= pd.Timestamp.today().normalize() - pd.DateOffset(years=1):
        return "近1年基準窗"
    return "outside_named_windows"


def _wide_prices(prices: pd.DataFrame, field: str) -> pd.DataFrame:
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    wide = frame.pivot_table(index="date", columns="symbol", values=field, aggfunc="last").sort_index()
    return wide.ffill()


def _wide_macro(macro: pd.DataFrame) -> pd.DataFrame:
    if macro.empty:
        return pd.DataFrame()
    frame = macro.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.pivot_table(index="date", columns="series", values="value", aggfunc="last").sort_index().ffill()


def _relative_return(wide: pd.DataFrame, lhs: str, rhs: str, window: int) -> pd.Series:
    if lhs not in wide.columns or rhs not in wide.columns:
        return pd.Series(index=wide.index, dtype=float)
    rel = wide[lhs] / wide[rhs]
    return rel.pct_change(window)


def _year_over_year(series: pd.Series | None) -> pd.Series:
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)
    return series.pct_change(12)


def _daily_news_sentiment(
    news: pd.DataFrame | None,
    international_news: pd.DataFrame | None,
    index: Iterable[pd.Timestamp],
) -> pd.DataFrame:
    frames = [frame for frame in [news, international_news] if frame is not None and not frame.empty]
    if not frames:
        empty = pd.DataFrame(index=pd.Index(index, name="date"))
        empty["news_count"] = 0
        empty["news_fear_score"] = 0.0
        empty["news_hype_score"] = 0.0
        empty["policy_risk_score"] = 0.0
        return empty

    data = pd.concat(frames, ignore_index=True)
    data["published"] = pd.to_datetime(data["published"], errors="coerce", utc=True).dt.tz_localize(None)
    data = data.dropna(subset=["published", "title"]).copy()
    if data.empty:
        return _daily_news_sentiment(None, None, index)

    data["date"] = data["published"].dt.normalize()
    data["news_fear_score"] = data["title"].astype(str).str.lower().apply(lambda text: _term_hits(text, FEAR_TERMS))
    data["news_hype_score"] = data["title"].astype(str).str.lower().apply(lambda text: _term_hits(text, HYPE_TERMS))
    data["policy_risk_score"] = data["title"].astype(str).str.lower().apply(lambda text: _term_hits(text, POLICY_TERMS))
    daily = (
        data.groupby("date")
        .agg(
            news_count=("title", "size"),
            news_fear_score=("news_fear_score", "mean"),
            news_hype_score=("news_hype_score", "mean"),
            policy_risk_score=("policy_risk_score", "mean"),
        )
        .sort_index()
    )
    full_index = pd.Index(index, name="date")
    daily = daily.reindex(full_index).fillna({"news_count": 0, "news_fear_score": 0.0, "news_hype_score": 0.0, "policy_risk_score": 0.0})
    return daily


def _term_hits(text: str, terms: list[str]) -> float:
    if not text:
        return 0.0
    hits = sum(1 for term in terms if term in text)
    return float(hits / max(len(terms), 1))


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    def _pct(values: np.ndarray) -> float:
        arr = pd.Series(values).dropna()
        if arr.empty:
            return np.nan
        return float(arr.rank(pct=True).iloc[-1])

    return series.rolling(window=window, min_periods=min(60, window)).apply(_pct, raw=True)


def _mood_label(score: float) -> str:
    if pd.isna(score):
        return "資料不足"
    if score >= 70:
        return "偏多"
    if score >= 55:
        return "觀望偏多"
    if score >= 40:
        return "中性"
    if score >= 25:
        return "風險升溫"
    return "防守"


def _safe_float(value: object) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    return float(numeric) if pd.notna(numeric) else np.nan
