from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CACHE_DIR


KG_PREDICTION_V2_CACHE = CACHE_DIR / "kg" / "kg_prediction_v2_log.parquet"
KG_PREDICTION_V2_HORIZONS = [("每日（1D）", 1), ("每週（5D）", 5), ("每月（20D）", 20)]
MODEL_VERSION = "KG_V2_MULTIFACTOR"


def load_kg_prediction_v2_log(path: Path | None = None) -> pd.DataFrame:
    path = path or KG_PREDICTION_V2_CACHE
    if not path.exists():
        return _empty_log()
    try:
        return _sanitize_log(pd.read_parquet(path))
    except Exception:
        return _empty_log()


def update_kg_prediction_v2_log(
    facts: pd.DataFrame,
    narratives: pd.DataFrame,
    prices: pd.DataFrame,
    path: Path | None = None,
    existing_log: pd.DataFrame | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Create one versioned, non-binary KG research observation per QQQ trading day."""
    path = path or KG_PREDICTION_V2_CACHE
    history = _ticker_history(prices, "QQQ")
    log = _sanitize_log(existing_log.copy() if existing_log is not None else load_kg_prediction_v2_log(path))
    if history.empty:
        return log

    prediction_date = pd.Timestamp(history.iloc[-1]["date"]).normalize()
    if not ((log["prediction_date"] == prediction_date).any() if not log.empty else False):
        observation = build_kg_v2_observation(facts, narratives, prices, prediction_date, log)
        rows = []
        for horizon, horizon_days in KG_PREDICTION_V2_HORIZONS:
            rows.append(
                {
                    "model_version": MODEL_VERSION,
                    "prediction_date": prediction_date,
                    "target": "QQQ",
                    "horizon": horizon,
                    "horizon_days": horizon_days,
                    "prediction_direction": observation["prediction_direction"],
                    "confidence": observation["confidence"],
                    "confidence_score": observation["confidence_score"],
                    "trend_score": observation["trend_score"],
                    "data_state": observation["data_state"],
                    "reason": observation["reason"],
                    "dominant_theme": observation["dominant_theme"],
                    "event_count": observation["event_count"],
                    "source_count": observation["source_count"],
                    "factor_fact_score": observation["factor_fact_score"],
                    "factor_fact_direction": observation["factor_fact_direction"],
                    "factor_narrative_score": observation["factor_narrative_score"],
                    "factor_narrative_direction": observation["factor_narrative_direction"],
                    "factor_technical_score": observation["factor_technical_score"],
                    "factor_technical_direction": observation["factor_technical_direction"],
                    "factor_pressure_score": observation["factor_pressure_score"],
                    "factor_pressure_direction": observation["factor_pressure_direction"],
                    "factor_source_score": observation["factor_source_score"],
                    "factor_source_direction": observation["factor_source_direction"],
                    "factor_agreement": observation["factor_agreement"],
                    "baseline_always_long_direction": "偏多",
                    "baseline_50dma_direction": observation["baseline_50dma_direction"],
                    "baseline_momentum_direction": observation["baseline_momentum_direction"],
                    "calibration_state": observation["calibration_state"],
                    "calibration_sample": observation["calibration_sample"],
                    "close_at_prediction": float(history.iloc[-1]["close"]),
                    "actual_return": np.nan,
                    "max_drawdown": np.nan,
                    "relative_to_qqq": 0.0,
                    "validated_at": pd.NaT,
                    "created_at_utc": pd.Timestamp.now(tz="UTC"),
                }
            )
        additions = pd.DataFrame(rows)
        log = additions if log.empty else pd.concat([log, additions], ignore_index=True)

    log = validate_kg_prediction_v2_log(log, history)
    log = _sanitize_log(log)
    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        log.to_parquet(path, index=False)
    return log


def build_kg_v2_observation(
    facts: pd.DataFrame,
    narratives: pd.DataFrame,
    prices: pd.DataFrame,
    prediction_date: pd.Timestamp,
    existing_log: pd.DataFrame,
) -> dict:
    cutoff = prediction_date.tz_localize("UTC") - pd.Timedelta(days=7)
    recent_facts = _recent_rows(facts, "timestamp_utc", cutoff)
    recent_narratives = _recent_rows(narratives, "timestamp_utc", cutoff)

    fact_score = _fact_score(recent_facts)
    narrative_score = _narrative_score(recent_narratives)
    technical_score, technical_baseline, momentum_baseline = _technical_score(prices, prediction_date)
    pressure_score = _pressure_score(prices, prediction_date)
    event_count = int(recent_facts.get("canonical_event_id", pd.Series(dtype=str)).nunique())
    source_count = int(recent_facts.get("source_domain", pd.Series(dtype=str)).replace("", np.nan).dropna().nunique())
    source_score = _source_score(recent_facts, event_count, source_count)

    factor_scores = [fact_score, narrative_score, technical_score, pressure_score]
    factor_directions = [_direction_from_score(score) for score in factor_scores]
    non_neutral = [direction for direction in factor_directions if direction != "中性"]
    agreement = max(non_neutral.count("偏多"), non_neutral.count("偏空")) if non_neutral else 0
    trend_score = float(np.nanmean(factor_scores)) if factor_scores else 0.0
    data_state = "可用"
    if source_count < 2 or event_count < 5:
        data_state = "資料不足"
        direction = "資料不足"
    elif agreement < 3:
        data_state = "因子分歧"
        direction = "觀望／分歧"
    elif trend_score >= 0.15:
        direction = "上漲"
    elif trend_score <= -0.15:
        direction = "下跌"
    else:
        data_state = "因子分歧"
        direction = "觀望／分歧"

    calibration_sample, calibration_state, calibration_adjustment = _calibration(existing_log, direction)
    confidence_score = float(np.clip(0.28 + min(event_count, 15) * 0.018 + min(source_count, 8) * 0.035 + agreement * 0.05 + calibration_adjustment, 0.20, 0.80))
    if data_state == "資料不足" or source_count < 2:
        confidence, confidence_score = "低", min(confidence_score, 0.40)
    elif agreement < 3 or confidence_score < 0.48:
        confidence = "低"
    elif confidence_score < 0.64:
        confidence = "中"
    else:
        confidence = "中高"

    themes = recent_narratives.get("dominant_theme", pd.Series(dtype=str)).replace("", np.nan).dropna()
    theme = str(themes.value_counts().index[0]) if not themes.empty else "尚未形成明確主題"
    factor_text = "、".join([
        f"事實{_direction_from_score(fact_score)}",
        f"敘事{_direction_from_score(narrative_score)}",
        f"技術{_direction_from_score(technical_score)}",
        f"壓力{_direction_from_score(pressure_score)}",
        f"來源{_direction_from_score(source_score)}",
    ])
    reason = f"主題：{theme}；{factor_text}；近 7 日 {event_count} 個去重事件、{source_count} 個來源。"
    return {
        "prediction_direction": direction,
        "confidence": confidence,
        "confidence_score": confidence_score,
        "trend_score": trend_score,
        "data_state": data_state,
        "reason": reason,
        "dominant_theme": theme,
        "event_count": event_count,
        "source_count": source_count,
        "factor_fact_score": fact_score,
        "factor_fact_direction": _direction_from_score(fact_score),
        "factor_narrative_score": narrative_score,
        "factor_narrative_direction": _direction_from_score(narrative_score),
        "factor_technical_score": technical_score,
        "factor_technical_direction": _direction_from_score(technical_score),
        "factor_pressure_score": pressure_score,
        "factor_pressure_direction": _direction_from_score(pressure_score),
        "factor_source_score": source_score,
        "factor_source_direction": _direction_from_score(source_score),
        "factor_agreement": agreement,
        "baseline_50dma_direction": technical_baseline,
        "baseline_momentum_direction": momentum_baseline,
        "calibration_state": calibration_state,
        "calibration_sample": calibration_sample,
    }


def validate_kg_prediction_v2_log(log: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    if log.empty or history.empty:
        return log
    result = log.copy()
    dates = pd.to_datetime(history["date"], errors="coerce").dt.normalize().reset_index(drop=True)
    closes = pd.to_numeric(history["close"], errors="coerce").to_numpy()
    for index, row in result.iterrows():
        if pd.notna(row.get("actual_return")):
            continue
        matches = np.flatnonzero(dates.to_numpy() == pd.Timestamp(row["prediction_date"]).normalize())
        if not len(matches):
            continue
        start, end = int(matches[-1]), int(matches[-1]) + int(row["horizon_days"])
        if end >= len(history):
            continue
        future = closes[start + 1 : end + 1]
        result.at[index, "actual_return"] = closes[end] / closes[start] - 1
        result.at[index, "max_drawdown"] = future.min() / closes[start] - 1 if len(future) else np.nan
        result.at[index, "relative_to_qqq"] = 0.0
        result.at[index, "validated_at"] = dates.iloc[end]
    return result


def kg_prediction_v2_summary(log: pd.DataFrame) -> pd.DataFrame:
    if log.empty:
        return pd.DataFrame()
    complete = log.dropna(subset=["actual_return"]).copy()
    if complete.empty:
        return pd.DataFrame()
    return (
        complete.groupby(["horizon", "prediction_direction", "confidence"], dropna=False)
        .agg(樣本數=("actual_return", "size"), 平均實際報酬=("actual_return", "mean"), 中位數報酬=("actual_return", "median"), 平均最大回撤=("max_drawdown", "mean"))
        .reset_index()
        .sort_values(["horizon", "樣本數"], ascending=[True, False])
    )


def _fact_score(facts: pd.DataFrame) -> float:
    if facts.empty:
        return 0.0
    weights = pd.to_numeric(facts.get("impact_score"), errors="coerce").fillna(0.5)
    weights *= pd.to_numeric(facts.get("source_reliability_score"), errors="coerce").fillna(0.5)
    mapped = facts.get("impact_direction", pd.Series(dtype=str)).map({"positive": 1.0, "negative": -1.0, "neutral": 0.0}).fillna(0.0)
    return _weighted_mean(mapped, weights)


def _narrative_score(narratives: pd.DataFrame) -> float:
    if narratives.empty:
        return 0.0
    sentiment = pd.to_numeric(narratives.get("sentiment_score"), errors="coerce")
    sentiment_score = (sentiment.mean() - 0.5) * 2 if sentiment.notna().any() else 0.0
    risks = ["fear_score", "recession_score", "policy_risk_score", "earnings_risk_score", "liquidity_risk_score", "geopolitical_risk_score"]
    available = [column for column in risks if column in narratives]
    risk = narratives[available].apply(pd.to_numeric, errors="coerce").mean(axis=1).mean() if available else 0.5
    risk = 0.5 if pd.isna(risk) else float(risk)
    return float(np.clip(0.65 * sentiment_score - 0.35 * (risk - 0.5) * 2, -1, 1))


def _technical_score(prices: pd.DataFrame, prediction_date: pd.Timestamp) -> tuple[float, str, str]:
    qqq = _ticker_history(prices, "QQQ")
    qqq = qqq[qqq["date"] <= prediction_date].tail(201)
    if len(qqq) < 51:
        return 0.0, "資料不足", "資料不足"
    close = qqq["close"].astype(float)
    latest = close.iloc[-1]
    ma20, ma50 = close.tail(20).mean(), close.tail(50).mean()
    ret20 = latest / close.iloc[-21] - 1 if len(close) >= 21 else 0.0
    volume = pd.to_numeric(qqq.get("volume"), errors="coerce")
    volume_ratio = volume.iloc[-1] / volume.tail(20).mean() if len(volume) >= 20 and volume.tail(20).mean() > 0 else 1.0
    parts = [np.sign(latest / ma20 - 1), np.sign(latest / ma50 - 1), np.sign(ret20), np.sign(volume_ratio - 1) * np.sign(ret20)]
    score = float(np.clip(np.nanmean(parts), -1, 1))
    return score, "偏多" if latest >= ma50 else "偏空", "偏多" if ret20 >= 0 else "偏空"


def _pressure_score(prices: pd.DataFrame, prediction_date: pd.Timestamp) -> float:
    vix = _ticker_history(prices, "^VIX")
    hyg = _ticker_history(prices, "HYG")
    components = []
    for frame, inverse in [(vix, True), (hyg, False)]:
        frame = frame[frame["date"] <= prediction_date].tail(6)
        if len(frame) < 6:
            continue
        change = float(frame["close"].iloc[-1] / frame["close"].iloc[0] - 1)
        components.append(-np.sign(change) if inverse else np.sign(change))
    return float(np.nanmean(components)) if components else 0.0


def _source_score(facts: pd.DataFrame, event_count: int, source_count: int) -> float:
    if source_count < 2 or event_count < 5:
        return -1.0
    reliability = pd.to_numeric(facts.get("source_reliability_score"), errors="coerce").mean()
    reliability = 0.5 if pd.isna(reliability) else float(reliability)
    return float(np.clip((source_count - 2) / 4 + (reliability - 0.5), -1, 1))


def _calibration(log: pd.DataFrame, direction: str) -> tuple[int, str, float]:
    complete = log[(log.get("prediction_direction", pd.Series(dtype=str)) == direction) & log.get("actual_return", pd.Series(dtype=float)).notna()].copy()
    sample = int(len(complete))
    if sample < 10:
        return sample, "樣本不足", 0.0
    mean_return = float(pd.to_numeric(complete["actual_return"], errors="coerce").mean())
    expected_sign = 1 if direction == "上漲" else -1 if direction == "下跌" else 0
    if expected_sign and mean_return * expected_sign <= 0:
        return sample, "歷史校準偏弱", -0.12
    return sample, "歷史校準可用", 0.05


def _ticker_history(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if prices is None or prices.empty or not {"date", "symbol", "close"}.issubset(prices.columns):
        return pd.DataFrame(columns=["date", "close", "volume"])
    columns = [column for column in ["date", "close", "volume"] if column in prices]
    frame = prices[prices["symbol"].astype(str).str.upper().eq(ticker.upper())][columns].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    return frame.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _recent_rows(frame: pd.DataFrame, column: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    if frame is None or frame.empty or column not in frame:
        return pd.DataFrame()
    result = frame.copy()
    result[column] = pd.to_datetime(result[column], errors="coerce", utc=True)
    return result[result[column] >= cutoff].copy()


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    return float(np.average(values[valid], weights=weights[valid])) if valid.any() else 0.0


def _direction_from_score(score: float) -> str:
    return "偏多" if score >= 0.15 else "偏空" if score <= -0.15 else "中性"


def _sanitize_log(log: pd.DataFrame) -> pd.DataFrame:
    if log is None or log.empty:
        return _empty_log()
    result = log.copy()
    for column in _empty_log().columns:
        if column not in result:
            result[column] = pd.NA
    result["prediction_date"] = pd.to_datetime(result["prediction_date"], errors="coerce").dt.normalize()
    result["validated_at"] = pd.to_datetime(result["validated_at"], errors="coerce")
    result["horizon_days"] = pd.to_numeric(result["horizon_days"], errors="coerce")
    result = result.dropna(subset=["prediction_date", "target", "horizon", "horizon_days"]).copy()
    result["horizon_days"] = result["horizon_days"].astype(int)
    return result.drop_duplicates(["model_version", "prediction_date", "target", "horizon"], keep="last").sort_values(["prediction_date", "horizon_days"]).reset_index(drop=True)


def _empty_log() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "model_version", "prediction_date", "target", "horizon", "horizon_days", "prediction_direction", "confidence", "confidence_score", "trend_score", "data_state", "reason", "dominant_theme", "event_count", "source_count",
        "factor_fact_score", "factor_fact_direction", "factor_narrative_score", "factor_narrative_direction", "factor_technical_score", "factor_technical_direction", "factor_pressure_score", "factor_pressure_direction", "factor_source_score", "factor_source_direction", "factor_agreement",
        "baseline_always_long_direction", "baseline_50dma_direction", "baseline_momentum_direction", "calibration_state", "calibration_sample", "close_at_prediction", "actual_return", "max_drawdown", "relative_to_qqq", "validated_at", "created_at_utc",
    ])
