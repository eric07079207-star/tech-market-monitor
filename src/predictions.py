from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import cache_path


PREDICTION_HORIZONS = [
    ("5D", 5),
    ("20D", 20),
    ("60D", 60),
]


def build_market_prediction(regime: dict, conclusion: dict, snapshot: pd.DataFrame) -> dict:
    qqq = snapshot[snapshot["symbol"] == "QQQ"].squeeze() if not snapshot.empty else pd.Series(dtype=float)
    score = _num(regime.get("score"))
    ret_20d = _num(qqq.get("ret_20d") if isinstance(qqq, pd.Series) else np.nan)
    dist_ma_50 = _num(qqq.get("dist_ma_50") if isinstance(qqq, pd.Series) else np.nan)
    dist_ma_200 = _num(qqq.get("dist_ma_200") if isinstance(qqq, pd.Series) else np.nan)
    vol_pctile = _num(qqq.get("realized_vol_pctile_252d") if isinstance(qqq, pd.Series) else np.nan)

    if score >= 65 and dist_ma_50 > 0:
        direction = "偏多"
    elif score <= 38 or dist_ma_200 < 0:
        direction = "偏空/防守"
    elif abs(ret_20d) <= 0.03:
        direction = "震盪"
    else:
        direction = "觀望"

    confidence = conclusion.get("confidence", "低")
    reasons = []
    if pd.notna(score):
        reasons.append(f"Regime {score:.0f}/100")
    if pd.notna(dist_ma_50):
        reasons.append("QQQ 高於 50DMA" if dist_ma_50 > 0 else "QQQ 低於 50DMA")
    if pd.notna(dist_ma_200):
        reasons.append("QQQ 高於 200DMA" if dist_ma_200 > 0 else "QQQ 低於 200DMA")
    if pd.notna(vol_pctile) and vol_pctile >= 0.8:
        reasons.append("波動位於近一年高檔")

    return {
        "target": "QQQ",
        "conclusion": conclusion.get("label", "資料不足"),
        "prediction_direction": direction,
        "confidence": confidence,
        "regime_score": score,
        "qqq_ret_20d": ret_20d,
        "qqq_dist_ma_50": dist_ma_50,
        "qqq_dist_ma_200": dist_ma_200,
        "qqq_vol_pctile": vol_pctile,
        "reason": "；".join(reasons[:6]),
    }


def load_prediction_log(path: Path | None = None) -> pd.DataFrame:
    path = path or cache_path("prediction_log.csv")
    if not path.exists():
        return _empty_log()
    log = pd.read_csv(path)
    for col in ["prediction_date", "validated_at"]:
        if col in log:
            log[col] = pd.to_datetime(log[col], errors="coerce")
    return log


def update_prediction_log(
    indicators: pd.DataFrame,
    prediction: dict,
    path: Path | None = None,
) -> pd.DataFrame:
    path = path or cache_path("prediction_log.csv")
    log = load_prediction_log(path)
    qqq = indicators[indicators["symbol"] == prediction.get("target", "QQQ")].dropna(subset=["close"]).sort_values("date")
    if qqq.empty:
        return log

    latest = qqq.iloc[-1]
    prediction_date = pd.to_datetime(latest["date"]).normalize()
    base = {
        "prediction_date": prediction_date.date().isoformat(),
        "target": prediction.get("target", "QQQ"),
        "conclusion": prediction.get("conclusion", ""),
        "prediction_direction": prediction.get("prediction_direction", ""),
        "confidence": prediction.get("confidence", "低"),
        "regime_score": prediction.get("regime_score", np.nan),
        "qqq_ret_20d": prediction.get("qqq_ret_20d", np.nan),
        "qqq_dist_ma_50": prediction.get("qqq_dist_ma_50", np.nan),
        "qqq_dist_ma_200": prediction.get("qqq_dist_ma_200", np.nan),
        "qqq_vol_pctile": prediction.get("qqq_vol_pctile", np.nan),
        "reason": prediction.get("reason", ""),
        "close_at_prediction": latest["close"],
    }

    if log.empty or not (
        (pd.to_datetime(log["prediction_date"], errors="coerce").dt.normalize() == prediction_date)
        & (log["target"].astype(str) == base["target"])
    ).any():
        additions = []
        for label, days in PREDICTION_HORIZONS:
            row = base.copy()
            row["horizon"] = label
            row["horizon_days"] = days
            row["actual_return"] = np.nan
            row["max_drawdown"] = np.nan
            row["success"] = np.nan
            row["validated_at"] = ""
            additions.append(row)
        log = pd.concat([log, pd.DataFrame(additions)], ignore_index=True)

    log = validate_prediction_log(log, qqq)
    path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(path, index=False)
    return log


def validate_prediction_log(log: pd.DataFrame, target_history: pd.DataFrame) -> pd.DataFrame:
    if log.empty or target_history.empty:
        return log
    history = target_history.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(history["date"]).dt.normalize()
    closes = history["close"].astype(float).to_numpy()

    result = log.copy()
    for idx, row in result.iterrows():
        if pd.notna(row.get("success")):
            continue
        pred_date = pd.to_datetime(row.get("prediction_date"), errors="coerce")
        if pd.isna(pred_date):
            continue
        start_matches = np.where(dates == pred_date.normalize())[0]
        if len(start_matches) == 0:
            start_pos = int(np.searchsorted(dates.to_numpy(), pred_date.to_datetime64()))
            if start_pos >= len(history):
                continue
        else:
            start_pos = int(start_matches[-1])
        horizon_days = int(row.get("horizon_days", 0))
        end_pos = start_pos + horizon_days
        if end_pos >= len(history):
            continue

        start_close = closes[start_pos]
        future_window = closes[start_pos + 1 : end_pos + 1]
        actual_return = closes[end_pos] / start_close - 1
        max_drawdown = future_window.min() / start_close - 1 if len(future_window) else np.nan
        success = _prediction_success(str(row.get("prediction_direction", "")), actual_return, max_drawdown)
        result.at[idx, "actual_return"] = actual_return
        result.at[idx, "max_drawdown"] = max_drawdown
        result.at[idx, "success"] = success
        result.at[idx, "validated_at"] = dates.iloc[end_pos].date().isoformat()
    return result


def prediction_validation_summary(log: pd.DataFrame) -> pd.DataFrame:
    if log.empty or "success" not in log:
        return pd.DataFrame()
    done = log.dropna(subset=["success"]).copy()
    if done.empty:
        return pd.DataFrame()
    done["success"] = done["success"].astype(bool)
    grouped = done.groupby(["horizon", "prediction_direction"], dropna=False)
    return (
        grouped.agg(
            sample=("success", "size"),
            success_rate=("success", "mean"),
            avg_return=("actual_return", "mean"),
            avg_max_drawdown=("max_drawdown", "mean"),
        )
        .reset_index()
        .sort_values(["horizon", "sample"], ascending=[True, False])
    )


def _prediction_success(direction: str, actual_return: float, max_drawdown: float) -> bool:
    if direction == "偏多":
        return bool(actual_return > 0)
    if direction == "偏空/防守":
        return bool(actual_return < 0 or max_drawdown <= -0.04)
    if direction == "震盪":
        return bool(-0.02 <= actual_return <= 0.02 and max_drawdown > -0.05)
    return bool(actual_return >= -0.02)


def _empty_log() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "prediction_date",
            "target",
            "conclusion",
            "prediction_direction",
            "confidence",
            "horizon",
            "horizon_days",
            "regime_score",
            "qqq_ret_20d",
            "qqq_dist_ma_50",
            "qqq_dist_ma_200",
            "qqq_vol_pctile",
            "reason",
            "close_at_prediction",
            "actual_return",
            "max_drawdown",
            "success",
            "validated_at",
        ]
    )


def _num(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan
