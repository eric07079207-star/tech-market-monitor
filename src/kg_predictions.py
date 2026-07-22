from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CACHE_DIR


KG_PREDICTION_CACHE = CACHE_DIR / "kg" / "kg_prediction_log.parquet"
KG_PREDICTION_HORIZONS = [
    ("每日（1D）", 1),
    ("每週（5D）", 5),
    ("每月（20D）", 20),
]


def load_kg_prediction_log(path: Path | None = None) -> pd.DataFrame:
    path = path or KG_PREDICTION_CACHE
    if not path.exists():
        return _empty_log()
    try:
        log = pd.read_parquet(path)
    except Exception:
        return _empty_log()
    return _sanitize_log(log)


def update_kg_prediction_log(
    facts: pd.DataFrame,
    narratives: pd.DataFrame,
    prices: pd.DataFrame,
    path: Path | None = None,
    existing_log: pd.DataFrame | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Record one QQQ knowledge-graph observation per trading day and validate mature rows."""
    path = path or KG_PREDICTION_CACHE
    history = _qqq_history(prices)
    log = _sanitize_log(existing_log.copy() if existing_log is not None else load_kg_prediction_log(path))
    if history.empty:
        return log

    latest = history.iloc[-1]
    prediction_date = pd.Timestamp(latest["date"]).normalize()
    if not ((log["prediction_date"] == prediction_date).any() if not log.empty else False):
        observation = build_kg_observation(facts, narratives, prediction_date)
        rows = []
        for horizon, horizon_days in KG_PREDICTION_HORIZONS:
            rows.append(
                {
                    "prediction_date": prediction_date,
                    "target": "QQQ",
                    "horizon": horizon,
                    "horizon_days": horizon_days,
                    "prediction_direction": observation["prediction_direction"],
                    "confidence": observation["confidence"],
                    "confidence_score": observation["confidence_score"],
                    "signal_score": observation["signal_score"],
                    "reason": observation["reason"],
                    "dominant_theme": observation["dominant_theme"],
                    "event_count": observation["event_count"],
                    "source_count": observation["source_count"],
                    "close_at_prediction": float(latest["close"]),
                    "actual_return": np.nan,
                    "max_drawdown": np.nan,
                    "success": pd.NA,
                    "validated_at": pd.NaT,
                    "created_at_utc": pd.Timestamp.now(tz="UTC"),
                }
            )
        additions = pd.DataFrame(rows)
        log = additions if log.empty else pd.concat([log, additions], ignore_index=True)

    log = validate_kg_prediction_log(log, history)
    log = _sanitize_log(log)
    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        log.to_parquet(path, index=False)
    return log


def build_kg_observation(facts: pd.DataFrame, narratives: pd.DataFrame, prediction_date: pd.Timestamp) -> dict:
    """Convert the recent KG state into a deliberately conservative research observation."""
    cutoff = prediction_date.tz_localize("UTC") - pd.Timedelta(days=7)
    recent_facts = _recent_rows(facts, "timestamp_utc", cutoff)
    recent_narratives = _recent_rows(narratives, "timestamp_utc", cutoff)

    fact_weight = pd.to_numeric(recent_facts.get("impact_score"), errors="coerce").fillna(0.5)
    fact_weight *= pd.to_numeric(recent_facts.get("source_reliability_score"), errors="coerce").fillna(0.5)
    direction_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    fact_direction = recent_facts.get("impact_direction", pd.Series(dtype=str)).map(direction_map).fillna(0.0)
    fact_signal = _weighted_mean(fact_direction, fact_weight)

    sentiment = pd.to_numeric(recent_narratives.get("sentiment_score"), errors="coerce")
    sentiment_signal = (sentiment.mean() - 0.5) * 2 if sentiment.notna().any() else 0.0
    risk_columns = [
        "fear_score", "recession_score", "policy_risk_score", "earnings_risk_score",
        "liquidity_risk_score", "geopolitical_risk_score",
    ]
    risk_values = recent_narratives[[column for column in risk_columns if column in recent_narratives]].apply(pd.to_numeric, errors="coerce")
    risk_signal = risk_values.mean(axis=1).mean() if not risk_values.empty else 0.5
    risk_signal = 0.0 if pd.isna(risk_signal) else float(risk_signal)

    signal_score = float(np.clip(0.45 * sentiment_signal + 0.35 * fact_signal - 0.20 * (risk_signal - 0.5) * 2, -1, 1))
    if signal_score >= 0.15:
        direction = "偏多"
    elif signal_score <= -0.15:
        direction = "偏空"
    else:
        direction = "觀望"

    event_count = int(recent_facts.get("canonical_event_id", pd.Series(dtype=str)).nunique())
    source_count = int(recent_facts.get("source_domain", pd.Series(dtype=str)).replace("", np.nan).dropna().nunique())
    confidence_score = float(np.clip(0.25 + min(event_count, 15) * 0.025 + min(source_count, 8) * 0.035, 0.25, 0.75))
    # Repeated reporting from one outlet is evidence of attention, not independent confirmation.
    if source_count < 2:
        confidence_score = min(confidence_score, 0.40)
        confidence = "低"
    elif event_count < 5:
        confidence = "低"
    elif event_count < 12 or source_count < 4:
        confidence = "中"
    else:
        confidence = "中高"

    themes = recent_narratives.get("dominant_theme", pd.Series(dtype=str)).replace("", np.nan).dropna()
    dominant_theme = str(themes.value_counts().index[0]) if not themes.empty else "尚未形成明確主題"
    direction_text = "偏正向" if fact_signal > 0.1 else "偏負向" if fact_signal < -0.1 else "中性"
    risk_text = "偏高" if risk_signal >= 0.65 else "偏低" if risk_signal <= 0.35 else "中等"
    reason = (
        f"近 7 日 {event_count} 個去重事件、{source_count} 個來源；"
        f"主題：{dominant_theme}；事件方向{direction_text}，敘事風險{risk_text}。"
    )
    return {
        "prediction_direction": direction,
        "confidence": confidence,
        "confidence_score": confidence_score,
        "signal_score": signal_score,
        "reason": reason,
        "dominant_theme": dominant_theme,
        "event_count": event_count,
        "source_count": source_count,
    }


def validate_kg_prediction_log(log: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    if log.empty or history.empty:
        return log
    result = log.copy()
    dates = pd.to_datetime(history["date"], errors="coerce").dt.normalize().reset_index(drop=True)
    closes = pd.to_numeric(history["close"], errors="coerce").to_numpy()
    for index, row in result.iterrows():
        if pd.notna(row.get("success")):
            continue
        matches = np.flatnonzero(dates.to_numpy() == pd.Timestamp(row["prediction_date"]).normalize())
        if not len(matches):
            continue
        start = int(matches[-1])
        end = start + int(row["horizon_days"])
        if end >= len(history):
            continue
        future = closes[start + 1 : end + 1]
        actual_return = closes[end] / closes[start] - 1
        max_drawdown = future.min() / closes[start] - 1 if len(future) else np.nan
        direction = str(row.get("prediction_direction", ""))
        success = actual_return > 0 if direction == "偏多" else actual_return < 0 if direction == "偏空" else abs(actual_return) <= 0.02
        result.at[index, "actual_return"] = actual_return
        result.at[index, "max_drawdown"] = max_drawdown
        result.at[index, "success"] = success
        result.at[index, "validated_at"] = dates.iloc[end]
    return result


def kg_prediction_summary(log: pd.DataFrame) -> pd.DataFrame:
    if log.empty or "success" not in log:
        return pd.DataFrame()
    complete = log.dropna(subset=["success"]).copy()
    if complete.empty:
        return pd.DataFrame()
    complete["success"] = complete["success"].astype(bool)
    return (
        complete.groupby("horizon", dropna=False)
        .agg(已驗證筆數=("success", "size"), 命中率=("success", "mean"), 平均實際報酬=("actual_return", "mean"))
        .reset_index()
    )


def _qqq_history(prices: pd.DataFrame) -> pd.DataFrame:
    if prices is None or prices.empty or not {"date", "symbol", "close"}.issubset(prices.columns):
        return pd.DataFrame(columns=["date", "close"])
    history = prices[prices["symbol"].astype(str).str.upper().eq("QQQ")][["date", "close"]].copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history["close"] = pd.to_numeric(history["close"], errors="coerce")
    return history.dropna().sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _recent_rows(frame: pd.DataFrame, column: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    if frame is None or frame.empty or column not in frame:
        return pd.DataFrame()
    result = frame.copy()
    result[column] = pd.to_datetime(result[column], errors="coerce", utc=True)
    return result[result[column] >= cutoff].copy()


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return 0.0
    return float(np.average(values[valid], weights=weights[valid]))


def _sanitize_log(log: pd.DataFrame) -> pd.DataFrame:
    if log is None or log.empty:
        return _empty_log()
    result = log.copy()
    for column in _empty_log().columns:
        if column not in result:
            result[column] = pd.NA
    result["prediction_date"] = pd.to_datetime(result["prediction_date"], errors="coerce").dt.normalize()
    result["horizon_days"] = pd.to_numeric(result["horizon_days"], errors="coerce")
    result = result.dropna(subset=["prediction_date", "target", "horizon", "horizon_days"]).copy()
    result["horizon_days"] = result["horizon_days"].astype(int)
    result["success"] = pd.array(result["success"], dtype="boolean")
    result["validated_at"] = pd.to_datetime(result["validated_at"], errors="coerce")
    return result.drop_duplicates(["prediction_date", "target", "horizon"], keep="last").sort_values(["prediction_date", "horizon_days"]).reset_index(drop=True)


def _empty_log() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "prediction_date", "target", "horizon", "horizon_days", "prediction_direction", "confidence",
        "confidence_score", "signal_score", "reason", "dominant_theme", "event_count", "source_count",
        "close_at_prediction", "actual_return", "max_drawdown", "success", "validated_at", "created_at_utc",
    ])
