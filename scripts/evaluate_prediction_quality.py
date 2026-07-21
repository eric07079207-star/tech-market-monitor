from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import cache_path
from src.evaluation import evaluate_lstm_backtest, evaluate_rule_predictions, leakage_audit, save_evaluation
from src.lstm import LSTM_BACKTEST_CACHE, LSTM_MONITOR_FEATURE_CACHE, LSTM_PREDICTIONS_CACHE


def main() -> None:
    backtest = _read_parquet(LSTM_BACKTEST_CACHE)
    predictions = _read_parquet(LSTM_PREDICTIONS_CACHE)
    monitor_features = _read_parquet(LSTM_MONITOR_FEATURE_CACHE)
    prediction_log = _read_csv(cache_path("prediction_log.csv"))
    payload = {
        "lstm": evaluate_lstm_backtest(backtest),
        "rule_predictions": evaluate_rule_predictions(prediction_log),
        "leakage_audit": leakage_audit(monitor_features, predictions),
    }
    path = save_evaluation(payload, cache_path("lstm/lstm_evaluation.json"))
    print(json.dumps({"path": str(path), "lstm": payload["lstm"], "leakage_audit": payload["leakage_audit"]}, ensure_ascii=False, default=str))


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


if __name__ == "__main__":
    main()
