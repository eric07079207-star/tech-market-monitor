"""Low-cost, reproducible evaluation helpers for model and rule predictions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def evaluate_lstm_backtest(backtest: pd.DataFrame) -> dict:
    if backtest is None or backtest.empty:
        return {"rows": 0, "status": "資料不足"}
    data = backtest.copy()
    required = {"date", "label", "predicted_label", "predicted_prob_up", "future_return", "success"}
    missing = sorted(required - set(data.columns))
    if missing:
        return {"rows": int(len(data)), "status": "欄位不足", "missing_columns": missing}
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in ["label", "predicted_label", "predicted_prob_up", "future_return"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["date", "label", "predicted_label", "predicted_prob_up", "future_return"]).sort_values("date")
    if data.empty:
        return {"rows": 0, "status": "資料不足"}
    y_true = data["label"].astype(int).to_numpy()
    y_pred = data["predicted_label"].astype(int).to_numpy()
    positive_rate = float(y_true.mean())
    baseline_label = int(positive_rate >= 0.5)
    baseline_accuracy = float((y_true == baseline_label).mean())
    always_long_accuracy = float((y_true == 1).mean())
    walk_forward = _walk_forward_majority_accuracy(y_true, min_history=20)
    returns = data["future_return"].to_numpy(dtype=float)
    result = {
        "status": "可用",
        "rows": int(len(data)),
        "positive_rate": positive_rate,
        "predicted_positive_rate": float(y_pred.mean()),
        "accuracy": float((y_true == y_pred).mean()),
        "baseline_accuracy": baseline_accuracy,
        "accuracy_edge_vs_baseline": float((y_true == y_pred).mean() - baseline_accuracy),
        "always_long_accuracy": always_long_accuracy,
        "accuracy_edge_vs_always_long": float((y_true == y_pred).mean() - always_long_accuracy),
        "walk_forward_majority_accuracy": walk_forward,
        "balanced_accuracy": _balanced_accuracy(y_true, y_pred),
        "precision_up": _precision(y_true, y_pred, 1),
        "recall_up": _recall(y_true, y_pred, 1),
        "f1_up": _f1(y_true, y_pred, 1),
        "brier_score": float(np.mean((data["predicted_prob_up"].to_numpy() - y_true) ** 2)),
        "avg_future_return": float(np.mean(returns)),
        "median_future_return": float(np.median(returns)),
        "worst_future_return": float(np.min(returns)),
        "best_future_return": float(np.max(returns)),
        "always_long_avg_return": float(np.mean(returns)),
        "model_directional_avg_return": float(np.mean(np.where(y_pred == 1, returns, -returns))),
        "leakage_rows": int((data["date"] >= pd.to_datetime(data.get("target_date"), errors="coerce")).sum()) if "target_date" in data else 0,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    result["directional_return_edge_vs_always_long"] = result["model_directional_avg_return"] - result["always_long_avg_return"]
    result["probability_calibration"] = _probability_calibration(data)
    result["model_beats_majority_baseline"] = bool(result["accuracy_edge_vs_baseline"] > 0)
    result["confidence_warning"] = _confidence_warning(len(data), result["accuracy_edge_vs_baseline"])
    return result


def evaluate_rule_predictions(prediction_log: pd.DataFrame) -> dict:
    if prediction_log is None or prediction_log.empty:
        return {"status": "資料不足", "rows": 0}
    data = prediction_log.copy()
    data["success"] = data.get("success", pd.Series(dtype=object)).map(_to_bool)
    rows = []
    for horizon, group in data.groupby(data.get("horizon", pd.Series("unknown", index=data.index))):
        validated = group["success"].dropna()
        rows.append(
            {
                "horizon": str(horizon),
                "rows": int(len(group)),
                "validated_rows": int(len(validated)),
                "accuracy": float(validated.mean()) if not validated.empty else None,
                "avg_return": _mean_numeric(group.get("actual_return")),
            }
        )
    return {"status": "可用", "rows": int(len(data)), "by_horizon": rows, "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def leakage_audit(features: pd.DataFrame | None, predictions: pd.DataFrame | None) -> dict:
    result = {"status": "可用", "feature_future_label_rows": 0, "prediction_date_after_target_rows": 0, "issues": []}
    if features is not None and not features.empty:
        feature_columns = {"future_return", "label"}
        result["feature_future_label_rows"] = int(features[list(feature_columns & set(features.columns))].notna().any(axis=1).sum()) if feature_columns & set(features.columns) else 0
    if predictions is not None and not predictions.empty and {"prediction_date", "target_date"}.issubset(predictions.columns):
        prediction_dates = pd.to_datetime(predictions["prediction_date"], errors="coerce")
        target_dates = pd.to_datetime(predictions["target_date"], errors="coerce")
        result["prediction_date_after_target_rows"] = int((prediction_dates >= target_dates).fillna(False).sum())
    if result["prediction_date_after_target_rows"]:
        result["issues"].append("即時預測列存在預測日不早於目標日")
    return result


def save_evaluation(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return path


def _walk_forward_majority_accuracy(labels: np.ndarray, min_history: int) -> float | None:
    if len(labels) <= min_history:
        return None
    predictions = []
    actual = []
    for index in range(min_history, len(labels)):
        history = labels[:index]
        predictions.append(int(history.mean() >= 0.5))
        actual.append(int(labels[index]))
    return float(np.mean(np.asarray(predictions) == np.asarray(actual))) if actual else None


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    recalls = [_recall(y_true, y_pred, label) for label in [0, 1]]
    valid = [value for value in recalls if value is not None]
    return float(np.mean(valid)) if len(valid) == 2 else None


def _precision(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> float | None:
    selected = y_pred == label
    denominator = int(selected.sum())
    return float(((y_true == label) & selected).sum() / denominator) if denominator else None


def _recall(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> float | None:
    actual = y_true == label
    denominator = int(actual.sum())
    return float(((y_pred == label) & actual).sum() / denominator) if denominator else None


def _f1(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> float | None:
    precision = _precision(y_true, y_pred, label)
    recall = _recall(y_true, y_pred, label)
    return float(2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else None


def _mean_numeric(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _confidence_warning(rows: int, edge: float) -> str:
    if rows < 100:
        return "樣本少，低信心"
    if edge <= 0:
        return "未超過多數類別基準"
    return "仍需滾動回測確認"


def _probability_calibration(data: pd.DataFrame) -> list[dict]:
    """Return coarse probability buckets so confidence can be inspected, not assumed."""
    report = data[["predicted_prob_up", "label"]].copy()
    try:
        report["bucket"] = pd.cut(
            report["predicted_prob_up"],
            bins=[-0.001, 0.4, 0.5, 0.6, 1.001],
            labels=["0-40%", "40-50%", "50-60%", "60-100%"],
        )
    except ValueError:
        return []
    grouped = report.groupby("bucket", observed=True).agg(樣本數=("label", "size"), 實際上漲率=("label", "mean")).reset_index()
    return [{"bucket": str(row.bucket), "rows": int(row.樣本數), "actual_up_rate": float(row.實際上漲率)} for row in grouped.itertuples(index=False)]


def _to_bool(value: object) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "成功"}


def _json_default(value: object):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)
