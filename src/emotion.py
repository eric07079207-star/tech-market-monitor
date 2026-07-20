"""Presentation-ready market emotion analytics built from existing cache layers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def latest_emotion(sentiment: pd.DataFrame) -> dict:
    if sentiment is None or sentiment.empty:
        return {}
    data = sentiment.copy()
    data["date"] = pd.to_datetime(data.get("date"), errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date")
    if data.empty:
        return {}
    row = data.iloc[-1].to_dict()
    row["date"] = data.iloc[-1]["date"]
    row["mood_5d_change"] = _change(data, "market_mood_score", 5)
    row["mood_20d_change"] = _change(data, "market_mood_score", 20)
    row["fear_pressure"] = _mean_numeric(row, ["vix_percentile_252d", "hy_oas_percentile_252d", "news_fear_percentile_252d", "policy_risk_percentile_252d"]) * 100
    row["risk_appetite"] = _clip_numeric(row.get("market_mood_score"), 0, 100)
    row["news_confidence"] = _clip_numeric(row.get("news_sentiment_confidence"), 0, 100)
    return row


def emotion_components(row: dict) -> pd.DataFrame:
    if not row:
        return pd.DataFrame(columns=["指標", "數值", "狀態", "解讀"])
    components = [
        ("市場情緒總分", row.get("market_mood_score"), _mood_label(row.get("market_mood_score")), "既有情緒模型綜合分數"),
        ("恐慌壓力", row.get("fear_pressure"), _pressure_label(row.get("fear_pressure")), "VIX、信用利差、新聞恐慌與政策風險"),
        ("VIX壓力", _percentile_value(row.get("vix_percentile_252d")), _pressure_label(_percentile_value(row.get("vix_percentile_252d"))), "VIX 相對過去 252 個交易日的位置"),
        ("信用壓力", _percentile_value(row.get("hy_oas_percentile_252d")), _pressure_label(_percentile_value(row.get("hy_oas_percentile_252d"))), "高收益債信用利差相對位置"),
        ("新聞恐慌", _percentile_value(row.get("news_fear_percentile_252d")), _pressure_label(_percentile_value(row.get("news_fear_percentile_252d"))), "可用新聞中的恐慌關鍵字強度"),
        ("政策風險", _percentile_value(row.get("policy_risk_percentile_252d")), _pressure_label(_percentile_value(row.get("policy_risk_percentile_252d"))), "利率、通膨、關稅與政策關鍵字"),
        ("新聞信心", row.get("news_sentiment_confidence"), _confidence_label(row.get("news_sentiment_confidence")), "來源可靠度、正式新聞比例與新聞覆蓋量"),
    ]
    frame = pd.DataFrame(components, columns=["指標", "數值", "狀態", "解讀"])
    frame["數值"] = pd.to_numeric(frame["數值"], errors="coerce").round(2)
    return frame


def emotion_trend(sentiment: pd.DataFrame, days: int = 120) -> pd.DataFrame:
    if sentiment is None or sentiment.empty:
        return pd.DataFrame()
    columns = ["date", "market_mood_score", "vix_percentile_252d", "news_fear_percentile_252d", "news_hype_percentile_252d"]
    data = sentiment[[column for column in columns if column in sentiment]].copy()
    if "date" not in data or "market_mood_score" not in data:
        return pd.DataFrame()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date").tail(days)
    data["恐慌壓力"] = data[[column for column in ["vix_percentile_252d", "news_fear_percentile_252d"] if column in data]].mean(axis=1) * 100
    data["市場情緒"] = pd.to_numeric(data["market_mood_score"], errors="coerce")
    return data[["date", "市場情緒", "恐慌壓力"]].dropna(how="all", subset=["市場情緒", "恐慌壓力"])


def emotion_divergence(prices: pd.DataFrame, sentiment: pd.DataFrame, symbols: list[str] | None = None) -> pd.DataFrame:
    symbols = symbols or ["QQQ", "VOO", "SMH", "NVDA", "TSLA"]
    if prices is None or prices.empty or sentiment is None or sentiment.empty:
        return pd.DataFrame()
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    wide = frame[frame["symbol"].astype(str).isin(symbols)].pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index().ffill()
    mood = sentiment.copy()
    mood["date"] = pd.to_datetime(mood["date"], errors="coerce")
    mood["market_mood_score"] = pd.to_numeric(mood["market_mood_score"], errors="coerce")
    mood = mood.dropna(subset=["date", "market_mood_score"]).set_index("date").sort_index()
    if wide.empty or mood.empty:
        return pd.DataFrame()
    rows = []
    latest_mood = float(mood["market_mood_score"].iloc[-1])
    prior_mood = float(mood["market_mood_score"].iloc[-21]) if len(mood) > 20 else np.nan
    mood_change = latest_mood - prior_mood if np.isfinite(prior_mood) else np.nan
    for symbol in symbols:
        if symbol not in wide or len(wide[symbol].dropna()) <= 20:
            continue
        series = wide[symbol].dropna()
        price_return = float(series.iloc[-1] / series.iloc[-21] - 1)
        if np.isfinite(mood_change) and price_return > 0.03 and mood_change < -8:
            interpretation = "價格上漲但情緒降溫"
        elif np.isfinite(mood_change) and price_return < -0.03 and mood_change > 8:
            interpretation = "價格下跌但情緒修復"
        elif price_return > 0 and (not np.isfinite(mood_change) or mood_change >= 0):
            interpretation = "價格與情緒同向偏強"
        elif price_return < 0 and (not np.isfinite(mood_change) or mood_change <= 0):
            interpretation = "價格與情緒同向轉弱"
        else:
            interpretation = "暫無明顯背離"
        rows.append({"標的": symbol, "20日報酬": price_return, "情緒20日變化": mood_change, "解讀": interpretation})
    return pd.DataFrame(rows)


def emotion_alerts(row: dict) -> list[dict]:
    if not row:
        return [{"燈號": "🟡", "項目": "情緒資料", "說明": "目前沒有可用的市場情緒資料。"}]
    mood = _numeric(row.get("market_mood_score"))
    change = _numeric(row.get("mood_5d_change"))
    alerts = []
    if mood < 25 or change <= -15:
        alerts.append({"燈號": "🔴", "項目": "市場情緒", "說明": "情緒偏防守或 5 日內快速惡化。"})
    elif mood < 40 or change <= -8:
        alerts.append({"燈號": "🟡", "項目": "市場情緒", "說明": "情緒風險升溫，需觀察是否持續。"})
    else:
        alerts.append({"燈號": "🟢", "項目": "市場情緒", "說明": "目前沒有觸發主要情緒警示。"})
    if _numeric(row.get("news_sentiment_confidence")) < 35:
        alerts.append({"燈號": "🟡", "項目": "新聞信心", "說明": "新聞覆蓋或來源可靠度偏低，情緒解讀需保守。"})
    return alerts


def _change(data: pd.DataFrame, column: str, periods: int) -> float:
    values = pd.to_numeric(data.get(column), errors="coerce").dropna()
    if len(values) <= periods:
        return np.nan
    return float(values.iloc[-1] - values.iloc[-periods - 1])


def _mean_numeric(row: dict, columns: list[str]) -> float:
    values = [float(row[column]) for column in columns if np.isfinite(_numeric(row.get(column)))]
    return float(np.mean(values)) if values else np.nan


def _numeric(value: object) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if pd.notna(number) else np.nan


def _clip_numeric(value: object, low: float, high: float) -> float:
    number = _numeric(value)
    return float(np.clip(number, low, high)) if np.isfinite(number) else np.nan


def _percentile_value(value: object) -> float:
    number = _numeric(value)
    return number * 100 if np.isfinite(number) else np.nan


def _mood_label(value: object) -> str:
    score = _numeric(value)
    if not np.isfinite(score):
        return "資料不足"
    if score >= 70:
        return "偏樂觀"
    if score >= 55:
        return "觀望偏多"
    if score >= 40:
        return "中性"
    if score >= 25:
        return "風險升溫"
    return "防守"


def _pressure_label(value: object) -> str:
    score = _numeric(value)
    if not np.isfinite(score):
        return "資料不足"
    if score >= 75:
        return "高壓力"
    if score >= 50:
        return "中壓力"
    return "低壓力"


def _confidence_label(value: object) -> str:
    score = _numeric(value)
    if not np.isfinite(score) or score < 35:
        return "低信心"
    if score < 60:
        return "中信心"
    return "高信心"
