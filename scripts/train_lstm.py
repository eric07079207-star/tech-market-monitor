from __future__ import annotations

import json
import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lstm import (
    LSTM_BACKTEST_CACHE,
    LSTM_DIR,
    LSTM_TRAIN_FEATURE_CACHE,
    LSTM_MONITOR_FEATURE_CACHE,
    LSTM_MODEL_CACHE,
    LSTM_MODEL_METADATA_CACHE,
    LSTM_PREDICTIONS_CACHE,
    LSTM_SCALER_CACHE,
    LSTM_SPLIT_CACHE,
    LSTM_FEATURE_VERSION,
    LSTM_TARGET_HORIZON_DAYS,
    LSTM_SEQUENCE_LENGTH,
    LSTM_SHARED_SYMBOLS,
    build_lstm_feature_table,
    build_lstm_monitor_feature_table,
    build_lstm_status_from_artifacts,
    feature_columns,
    load_lstm_status,
    save_lstm_status,
)
from src.data import PRICE_CACHE
from src.fundamentals import load_fundamental_observations
from src.kg import load_knowledge_graph


try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover - surfaced in CI/runtime
    raise RuntimeError("torch is required for LSTM training. Install dependencies first.") from exc


MODEL_VERSION = "lstm-regime-v5"
DEFAULT_SYMBOLS = LSTM_SHARED_SYMBOLS
# GitHub-hosted CPU runners have a bounded job window.  Six epochs preserve
# more training passes than the previous five-epoch model, while the larger
# batch keeps the shared eight-stock and TSLA specialist jobs reliable.
EPOCHS = 6
BATCH_SIZE = 256
HIDDEN_SIZE = 48
NUM_LAYERS = 1
LEARNING_RATE = 1e-3
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
ABSTENTION_MARGIN = 0.07
REGIME_THRESHOLD_MIN_ROWS = 60
THRESHOLD_MAX_CHANGE = 0.03
PROMOTION_MIN_EDGE = 0.01
PROMOTION_MAX_INCUMBENT_REGRESSION = 0.002
PRODUCTION_MIN_EDGE = 0.01


@dataclass(frozen=True)
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_valid: np.ndarray
    y_valid: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_regimes: np.ndarray
    valid_regimes: np.ndarray
    test_meta: pd.DataFrame
    split_meta: pd.DataFrame
    feature_mean: pd.Series
    feature_std: pd.Series


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE, num_layers: int = NUM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the LSTM model.")
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS, help="Symbols to use for the training run.")
    parser.add_argument("--start-date", default="2013-01-01", help="Earliest point-in-time training date.")
    parser.add_argument("--lookback-years", type=int, default=None, help="Optional rolling lookback; normally leave unset to preserve full regimes.")
    parser.add_argument("--sample-step", type=int, default=5, help="Sample every N trading days.")
    args = parser.parse_args()
    previous_threshold, previous_regime_thresholds = _load_previous_thresholds()
    incumbent_quality = _load_incumbent_quality()
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if not PRICE_CACHE.exists():
        raise RuntimeError("prices.parquet is missing; run scripts/update_data.py first.")
    prices = pd.read_parquet(PRICE_CACHE)
    if "date" in prices:
        prices["date"] = pd.to_datetime(prices["date"])

    kg = load_knowledge_graph()
    fundamentals = load_fundamental_observations()
    features = build_lstm_feature_table(
        prices=prices,
        target_horizon_days=LSTM_TARGET_HORIZON_DAYS,
        sequence_length=LSTM_SEQUENCE_LENGTH,
        symbols=args.symbols,
        lookback_years=args.lookback_years,
        start_date=args.start_date,
        sample_step=args.sample_step,
        fact_events=kg.facts,
        fundamentals=fundamentals,
    )
    # Specialist runs redirect all artifacts to a nested directory. Ensure the
    # selected artifact directory exists before writing the first feature file.
    LSTM_TRAIN_FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(LSTM_TRAIN_FEATURE_CACHE, index=False)

    if features.empty:
        raise RuntimeError("No LSTM features available for training.")

    split_data = _prepare_split(features)
    if split_data.x_train.size == 0 or split_data.x_valid.size == 0:
        raise RuntimeError("Insufficient data to train LSTM.")

    model = LSTMClassifier(input_size=split_data.x_train.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    positive = max(float(split_data.y_train.sum()), 1.0)
    negative = max(float(len(split_data.y_train) - split_data.y_train.sum()), 1.0)
    # Correct the historical class imbalance without changing the evaluation labels.
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, dtype=torch.float32))

    train_ds = torch.utils.data.TensorDataset(torch.tensor(split_data.x_train, dtype=torch.float32), torch.tensor(split_data.y_train, dtype=torch.float32))
    valid_x = torch.tensor(split_data.x_valid, dtype=torch.float32)
    valid_y = torch.tensor(split_data.y_valid, dtype=torch.float32)
    regime_counts = pd.Series(split_data.train_regimes).value_counts()
    regime_weights = np.asarray(
        [1.0 / max(float(regime_counts.get(regime, 1)), 1.0) for regime in split_data.train_regimes],
        dtype=np.float64,
    )
    regime_weights /= max(regime_weights.mean(), 1e-12)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.tensor(regime_weights, dtype=torch.double),
        num_samples=len(regime_weights),
        replacement=True,
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)

    best_state = None
    best_valid_loss = np.inf
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
        train_loss /= max(len(train_ds), 1)

        model.eval()
        with torch.no_grad():
            valid_logits = model(valid_x)
            valid_loss = criterion(valid_logits, valid_y).item()
            valid_prob = torch.sigmoid(valid_logits).cpu().numpy()
            valid_pred = (valid_prob >= 0.5).astype(int)
            valid_acc = float((valid_pred == split_data.y_valid).mean())
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss, "valid_acc": valid_acc})
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "valid_loss": valid_loss,
                "valid_acc": valid_acc,
            }

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid checkpoint.")

    model.load_state_dict(best_state["model_state"])
    model.eval()
    with torch.no_grad():
        valid_logits = model(valid_x)
        valid_prob = torch.sigmoid(valid_logits).cpu().numpy()
        decision_threshold, threshold_metrics = _calibrate_threshold(
            valid_prob,
            split_data.y_valid,
            split_data.valid_regimes,
            previous_threshold=previous_threshold,
            previous_regime_thresholds=previous_regime_thresholds,
        )
        regime_thresholds = threshold_metrics.get("regime_thresholds", {})
        test_logits = model(torch.tensor(split_data.x_test, dtype=torch.float32))
        test_prob = torch.sigmoid(test_logits).cpu().numpy()
        test_thresholds = _thresholds_for_regimes(
            split_data.test_meta.get("regime_bucket", pd.Series(dtype=str)).astype(str).to_numpy(),
            decision_threshold,
            regime_thresholds,
        )
        test_pred = (test_prob >= test_thresholds).astype(int)
        test_acc = float((test_pred == split_data.y_test).mean()) if len(split_data.y_test) else np.nan

    backtest = _build_backtest_frame(split_data.test_meta, test_prob, test_pred, test_thresholds)
    monitor_features = build_lstm_monitor_feature_table(
        prices=prices,
        symbols=args.symbols,
        target_horizon_days=LSTM_TARGET_HORIZON_DAYS,
        sequence_length=LSTM_SEQUENCE_LENGTH,
        fact_events=kg.facts,
        fundamentals=fundamentals,
    )
    monitor_features.to_parquet(LSTM_MONITOR_FEATURE_CACHE, index=False)
    challenger_quality = _backtest_quality(backtest)
    promotion = _promotion_decision(challenger_quality, incumbent_quality)
    _save_challenger_report(challenger_quality, incumbent_quality, promotion, threshold_metrics)
    if not promotion["promoted"]:
        incumbent_status = load_lstm_status()
        production_eligible = _is_production_eligible(incumbent_quality)
        model_state_text = "正式模型已保留" if production_eligible else "現行模型未超過基準，僅供研究"
        incumbent_status.update(
            {
                "status": f"{model_state_text}；最新挑戰模型未達升級門檻（{promotion['reason']}）",
                "production_eligible": production_eligible,
                "production_accuracy": incumbent_quality.get("accuracy"),
                "production_baseline_accuracy": incumbent_quality.get("baseline_accuracy"),
                "production_edge_vs_baseline": incumbent_quality.get("edge_vs_baseline"),
                "last_challenger_at_utc": promotion["evaluated_at_utc"],
                "last_challenger_version": MODEL_VERSION,
                "last_challenger_accuracy": challenger_quality.get("accuracy"),
                "last_challenger_edge_vs_baseline": challenger_quality.get("edge_vs_baseline"),
                "last_challenger_promoted": False,
            }
        )
        incumbent_predictions = _refresh_incumbent_predictions(monitor_features)
        if not incumbent_predictions.empty:
            incumbent_status.update(
                {
                    "last_predict_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "monitor_updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "prediction_rows": int(len(incumbent_predictions)),
                    "monitor_feature_rows": int(len(monitor_features)),
                    "prediction_date": str(incumbent_predictions.iloc[-1]["prediction_date"]),
                    "prediction_target_date": str(incumbent_predictions.iloc[-1]["target_date"]),
                }
            )
        save_lstm_status(incumbent_status)
        print(
            f"rejected challenger model_version={MODEL_VERSION} reason={promotion['reason']} "
            f"accuracy={challenger_quality.get('accuracy', float('nan')):.4f} "
            f"baseline={challenger_quality.get('baseline_accuracy', float('nan')):.4f}"
        )
        return

    LSTM_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state["model_state"], LSTM_MODEL_CACHE)
    _save_scaler(split_data, LSTM_SCALER_CACHE)
    backtest.to_parquet(LSTM_BACKTEST_CACHE, index=False)
    split_data.split_meta.to_parquet(LSTM_SPLIT_CACHE, index=False)

    latest_predictions = _build_monitor_predictions(
        model,
        monitor_features,
        split_data.feature_mean,
        split_data.feature_std,
        decision_threshold,
        regime_thresholds,
        model_version=MODEL_VERSION,
    )
    latest_predictions.to_parquet(LSTM_PREDICTIONS_CACHE, index=False)

    metadata = {
        "model_version": MODEL_VERSION,
        "feature_version": str(features["feature_version"].iloc[-1]),
        "sequence_length": LSTM_SEQUENCE_LENGTH,
        "horizon_days": LSTM_TARGET_HORIZON_DAYS,
        "epochs": EPOCHS,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "learning_rate": LEARNING_RATE,
        "random_seed": RANDOM_SEED,
        "symbols": args.symbols,
        "training_start_date": args.start_date,
        "sample_step": args.sample_step,
        "class_balance_positive_weight": negative / positive,
        "regime_balance_method": "inverse_frequency_weighted_sampler",
        "train_regime_counts": {str(key): int(value) for key, value in regime_counts.items()},
        "decision_threshold": float(decision_threshold),
        "decision_thresholds_by_regime": regime_thresholds,
        "abstention_margin": ABSTENTION_MARGIN,
        "threshold_calibration": threshold_metrics,
        "threshold_calibration_source": "nested_walk_forward_validation_only",
        "split_method": "time_series_purged_20d",
        "train_ratio": TRAIN_RATIO,
        "valid_ratio": VALID_RATIO,
        "test_ratio": TEST_RATIO,
        "train_rows": int(len(split_data.y_train)),
        "valid_rows": int(len(split_data.y_valid)),
        "test_rows": int(len(split_data.y_test)),
        "feature_rows": int(len(features)),
        "valid_loss": float(best_state["valid_loss"]),
        "valid_acc": float(best_state["valid_acc"]),
        "calibrated_valid_acc": float(threshold_metrics.get("accuracy", best_state["valid_acc"])),
        "calibrated_valid_balanced_acc": float(threshold_metrics.get("balanced_accuracy", np.nan)),
        "test_acc": test_acc,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "history": history,
    }
    LSTM_MODEL_METADATA_CACHE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    status = build_lstm_status_from_artifacts(latest_predictions, backtest, features, monitor_features)
    status.update(
        {
            "enabled": True,
            "mode": "train",
            "model_version": MODEL_VERSION,
            "feature_version": str(features["feature_version"].iloc[-1]),
            "horizon_days": LSTM_TARGET_HORIZON_DAYS,
            "sequence_length": LSTM_SEQUENCE_LENGTH,
            "last_train_at_utc": metadata["trained_at_utc"],
            "status": f"已訓練完成，嵌套驗證門檻 {decision_threshold:.2f}，測試準確率 {test_acc:.2%}",
            "decision_threshold": float(decision_threshold),
            "decision_thresholds_by_regime": regime_thresholds,
            "abstention_margin": ABSTENTION_MARGIN,
            "threshold_calibration_source": "nested_walk_forward_validation_only",
            "train_regime_counts": {str(key): int(value) for key, value in regime_counts.items()},
            "prediction_rows": int(len(latest_predictions)),
            "backtest_rows": int(len(backtest)),
            "feature_rows": int(len(features)),
            "train_feature_rows": int(len(features)),
            "monitor_feature_rows": int(len(monitor_features)),
            "train_rows": int(len(split_data.y_train)),
            "valid_rows": int(len(split_data.y_valid)),
            "test_rows": int(len(split_data.y_test)),
            "backtest_accuracy": test_acc,
            "production_eligible": True,
            "production_accuracy": challenger_quality.get("accuracy"),
            "production_baseline_accuracy": challenger_quality.get("baseline_accuracy"),
            "production_edge_vs_baseline": challenger_quality.get("edge_vs_baseline"),
            "prediction_date": str(latest_predictions.iloc[-1]["prediction_date"]) if not latest_predictions.empty else "",
            "prediction_target_date": str(latest_predictions.iloc[-1]["target_date"]) if not latest_predictions.empty else "",
            "prediction_target_date_type": "estimated",
            "monitor_updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "updated_at_utc": metadata["trained_at_utc"],
            "last_challenger_promoted": True,
        }
    )
    save_lstm_status(status)

    print(
        f"trained lstm model_version={MODEL_VERSION} feature_rows={len(features)} "
        f"train={len(split_data.y_train)} valid={len(split_data.y_valid)} test={len(split_data.y_test)} "
        f"valid_acc={best_state['valid_acc']:.4f} test_acc={test_acc:.4f}"
    )


def _prepare_split(features: pd.DataFrame) -> SplitData:
    data = features.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce", utc=True)
    data["target_date"] = pd.to_datetime(data["target_date"], errors="coerce", utc=True)
    data = data.dropna(subset=["date", "target_date"]).sort_values(["date", "symbol"]).reset_index(drop=True)
    data["regime_bucket"] = data["sequence_json"].map(_market_regime_from_sequence)
    unique_dates = np.sort(data["date"].unique())
    train_cutoff = pd.Timestamp(unique_dates[max(int(len(unique_dates) * TRAIN_RATIO) - 1, 0)])
    valid_cutoff = pd.Timestamp(unique_dates[max(int(len(unique_dates) * (TRAIN_RATIO + VALID_RATIO)) - 1, 0)])
    purge = pd.offsets.BDay(LSTM_TARGET_HORIZON_DAYS)
    valid_start = train_cutoff + purge
    test_start = valid_cutoff + purge
    data["split"] = "purged"
    data.loc[data["date"] <= train_cutoff, "split"] = "train"
    data.loc[(data["date"] >= valid_start) & (data["date"] <= valid_cutoff), "split"] = "valid"
    data.loc[data["date"] >= test_start, "split"] = "test"
    data = data[data["split"].isin(["train", "valid", "test"])].reset_index(drop=True)
    rows = []
    for row in data.itertuples():
        seq = json.loads(row.sequence_json)
        rows.append(seq)
    sequence_array = np.asarray(rows, dtype=np.float32)
    labels = data["label"].astype(int).to_numpy()
    meta_columns = [column for column in ["symbol", "date", "target_date", "future_return", "label", "qqq_future_return", "qqq_label", "excess_return_vs_qqq", "material_up_label", "regime_bucket", "feature_version", "split"] if column in data]
    meta = data[meta_columns].copy()
    train_mask = data["split"].to_numpy() == "train"
    train_rows = sequence_array[train_mask]
    feature_mean = pd.Series(_safe_nanmean(train_rows, axis=(0, 1)), index=feature_columns())
    feature_std = pd.Series(_safe_nanstd(train_rows, axis=(0, 1)), index=feature_columns()).replace(0, 1.0)
    normalized = (sequence_array - feature_mean.to_numpy()) / feature_std.to_numpy()
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    valid_mask = data["split"].to_numpy() == "valid"
    test_mask = data["split"].to_numpy() == "test"

    return SplitData(
        x_train=normalized[train_mask],
        y_train=labels[train_mask],
        x_valid=normalized[valid_mask],
        y_valid=labels[valid_mask],
        x_test=normalized[test_mask],
        y_test=labels[test_mask],
        train_regimes=data.loc[train_mask, "regime_bucket"].astype(str).to_numpy(),
        valid_regimes=data.loc[valid_mask, "regime_bucket"].astype(str).to_numpy(),
        test_meta=meta[test_mask].reset_index(drop=True),
        split_meta=meta.reset_index(drop=True),
        feature_mean=feature_mean,
        feature_std=feature_std,
    )


def _save_scaler(split: SplitData, path: Path) -> None:
    payload = {
        "feature_version": LSTM_FEATURE_VERSION,
        "columns": feature_columns(),
        "mean": split.feature_mean.fillna(0.0).tolist(),
        "std": split.feature_std.fillna(1.0).tolist(),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_backtest_frame(
    meta: pd.DataFrame,
    prob: np.ndarray,
    pred: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    if meta.empty:
        return pd.DataFrame()
    frame = meta.copy()
    frame["predicted_prob_up"] = prob
    frame["predicted_label"] = pred
    frame["decision_threshold"] = thresholds
    frame["abstention_margin"] = ABSTENTION_MARGIN
    frame["signal_distance"] = np.abs(frame["predicted_prob_up"] - frame["decision_threshold"])
    frame["is_actionable"] = frame["signal_distance"].ge(ABSTENTION_MARGIN)
    frame["prediction_direction"] = np.select(
        [
            frame["is_actionable"] & frame["predicted_label"].eq(1),
            frame["is_actionable"] & frame["predicted_label"].eq(0),
        ],
        ["看漲", "看跌"],
        default="低信心／不預測",
    )
    frame["model_version"] = MODEL_VERSION
    frame["created_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    frame["validated_at_utc"] = frame["created_at_utc"]
    success = (frame["predicted_label"].eq(1) & frame["future_return"].gt(0)) | (frame["predicted_label"].eq(0) & frame["future_return"].lt(0))
    frame["success"] = success.where(frame["is_actionable"], pd.NA).astype("boolean")
    return frame


def _build_monitor_predictions(
    model: torch.nn.Module,
    monitor_features: pd.DataFrame,
    mean: pd.Series,
    std: pd.Series,
    decision_threshold: float,
    regime_thresholds: dict[str, float],
    model_version: str = MODEL_VERSION,
) -> pd.DataFrame:
    if monitor_features.empty:
        return pd.DataFrame()

    latest_rows = []
    for symbol, group in monitor_features.groupby("symbol", sort=False):
        latest = group.sort_values("prediction_date").iloc[-1]
        seq = np.asarray(json.loads(latest.sequence_json), dtype=np.float32)
        normalized = (seq - mean.to_numpy()) / std.to_numpy()
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        with torch.no_grad():
            logits = model(torch.tensor(normalized[None, :, :], dtype=torch.float32))
            prob = torch.sigmoid(logits).item()
        regime = _market_regime_from_sequence(latest.sequence_json)
        threshold = float(regime_thresholds.get(regime, decision_threshold))
        signal_distance = abs(float(prob) - threshold)
        actionable = signal_distance >= ABSTENTION_MARGIN
        predicted_label = int(prob >= threshold)
        direction = ("看漲" if predicted_label else "看跌") if actionable else "低信心／不預測"
        latest_rows.append(
            {
                "symbol": symbol,
                "prediction_date": pd.to_datetime(latest.prediction_date, utc=True),
                "target_date": pd.to_datetime(latest.target_date, utc=True),
                "target_date_type": str(latest.get("target_date_type", "estimated")),
                "horizon_days": int(latest.horizon_days),
                "model_version": model_version,
                "feature_version": str(latest.feature_version),
                "predicted_prob_up": float(prob),
                "predicted_label": predicted_label,
                "prediction_direction": direction,
                "decision_threshold": threshold,
                "abstention_margin": ABSTENTION_MARGIN,
                "regime_bucket": regime,
                "signal_distance": signal_distance,
                "is_actionable": actionable,
                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "prediction_type": "real_time_inference",
                "production_eligible": True,
            }
        )
    return pd.DataFrame(latest_rows)


def _calibrate_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    regimes: np.ndarray | None = None,
    previous_threshold: float | None = None,
    previous_regime_thresholds: dict[str, float] | None = None,
) -> tuple[float, dict]:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    regimes = np.asarray(regimes, dtype=str) if regimes is not None else np.full(len(labels), "all", dtype=str)
    if not len(probabilities) or len(np.unique(labels)) < 2:
        return 0.5, {"objective": "robust_validation_balanced_accuracy", "status": "insufficient_validation_classes"}

    candidates = np.linspace(0.25, 0.75, 101)
    temporal_blocks = [block for block in np.array_split(np.arange(len(labels)), 4) if len(block) >= 30 and len(np.unique(labels[block])) == 2]
    regime_blocks = [
        np.flatnonzero(regimes == regime)
        for regime in np.unique(regimes)
        if int((regimes == regime).sum()) >= 30 and len(np.unique(labels[regimes == regime])) == 2
    ]

    def balanced_accuracy(prediction: np.ndarray, indices: np.ndarray) -> float:
        recalls = []
        for label in (0, 1):
            actual = labels[indices] == label
            recalls.append(float((prediction[indices][actual] == label).mean()) if actual.any() else np.nan)
        return float(np.nanmean(recalls))

    rows = []
    all_indices = np.arange(len(labels))
    for threshold in candidates:
        prediction = (probabilities >= threshold).astype(int)
        overall_balanced = balanced_accuracy(prediction, all_indices)
        temporal_scores = [balanced_accuracy(prediction, indices) for indices in temporal_blocks]
        regime_scores = [balanced_accuracy(prediction, indices) for indices in regime_blocks]
        temporal_mean = float(np.mean(temporal_scores)) if temporal_scores else overall_balanced
        regime_mean = float(np.mean(regime_scores)) if regime_scores else overall_balanced
        temporal_std = float(np.std(temporal_scores)) if len(temporal_scores) > 1 else 0.0
        regime_std = float(np.std(regime_scores)) if len(regime_scores) > 1 else 0.0
        predicted_positive_rate = float(prediction.mean())
        minority_prediction_rate = min(predicted_positive_rate, 1.0 - predicted_positive_rate)
        class_collapse_penalty = max(0.0, 0.10 - minority_prediction_rate) * 1.5
        robust_score = (
            0.50 * overall_balanced
            + 0.30 * temporal_mean
            + 0.20 * regime_mean
            - 0.15 * temporal_std
            - 0.10 * regime_std
            - class_collapse_penalty
            - 0.01 * abs(float(threshold) - 0.5)
        )
        accuracy = float((prediction == labels).mean())
        rows.append(
            (
                robust_score,
                overall_balanced,
                accuracy,
                -abs(float(threshold) - 0.5),
                float(threshold),
                temporal_mean,
                temporal_std,
                regime_mean,
                regime_std,
                predicted_positive_rate,
            )
        )
    (
        best_score,
        best_balanced,
        best_accuracy,
        _,
        best_threshold,
        best_temporal_mean,
        best_temporal_std,
        best_regime_mean,
        best_regime_std,
        best_positive_rate,
    ) = max(rows)
    full_validation_threshold = float(best_threshold)
    chronological_blocks = [block for block in np.array_split(np.arange(len(labels)), 5) if len(block) >= 20]
    walk_forward_rows = []
    for fold_index in range(1, len(chronological_blocks)):
        calibration_indices = np.concatenate(chronological_blocks[:fold_index])
        evaluation_indices = chronological_blocks[fold_index]
        if len(calibration_indices) < 30 or len(np.unique(labels[calibration_indices])) < 2 or len(np.unique(labels[evaluation_indices])) < 2:
            continue
        fold_threshold = _best_balanced_threshold(probabilities[calibration_indices], labels[calibration_indices], candidates)
        fold_prediction = (probabilities[evaluation_indices] >= fold_threshold).astype(int)
        fold_balanced = _balanced_accuracy(labels[evaluation_indices], fold_prediction)
        walk_forward_rows.append(
            {"fold": fold_index, "threshold": float(fold_threshold), "rows": int(len(evaluation_indices)), "balanced_accuracy": fold_balanced}
        )

    raw_thresholds = [full_validation_threshold] + [row["threshold"] for row in walk_forward_rows]
    raw_nested_threshold = float(np.median(raw_thresholds))
    stabilized_threshold = _limit_threshold_change(raw_nested_threshold, previous_threshold, THRESHOLD_MAX_CHANGE)

    previous_regime_thresholds = previous_regime_thresholds or {}
    regime_thresholds: dict[str, float] = {}
    regime_details: dict[str, dict] = {}
    for regime in sorted(np.unique(regimes)):
        indices = np.flatnonzero(regimes == regime)
        if len(indices) < REGIME_THRESHOLD_MIN_ROWS or len(np.unique(labels[indices])) < 2:
            continue
        raw_regime_threshold = _best_balanced_threshold(probabilities[indices], labels[indices], candidates)
        raw_regime_threshold = float(np.clip(raw_regime_threshold, stabilized_threshold - 0.08, stabilized_threshold + 0.08))
        regime_threshold = _limit_threshold_change(
            raw_regime_threshold,
            previous_regime_thresholds.get(str(regime), stabilized_threshold),
            THRESHOLD_MAX_CHANGE,
        )
        regime_thresholds[str(regime)] = regime_threshold
        regime_prediction = (probabilities[indices] >= regime_threshold).astype(int)
        regime_details[str(regime)] = {
            "rows": int(len(indices)),
            "raw_threshold": raw_regime_threshold,
            "threshold": regime_threshold,
            "balanced_accuracy": _balanced_accuracy(labels[indices], regime_prediction),
        }

    return stabilized_threshold, {
        "objective": "robust_validation_balanced_accuracy",
        "status": "calibrated",
        "robust_score": best_score,
        "balanced_accuracy": best_balanced,
        "accuracy": best_accuracy,
        "temporal_balanced_accuracy_mean": best_temporal_mean,
        "temporal_balanced_accuracy_std": best_temporal_std,
        "regime_balanced_accuracy_mean": best_regime_mean,
        "regime_balanced_accuracy_std": best_regime_std,
        "predicted_positive_rate": best_positive_rate,
        "temporal_block_count": int(len(temporal_blocks)),
        "regime_block_count": int(len(regime_blocks)),
        "candidate_min": float(candidates.min()),
        "candidate_max": float(candidates.max()),
        "candidate_count": int(len(candidates)),
        "full_validation_threshold": full_validation_threshold,
        "raw_nested_threshold": raw_nested_threshold,
        "stabilized_threshold": stabilized_threshold,
        "previous_threshold": previous_threshold,
        "max_threshold_change": THRESHOLD_MAX_CHANGE,
        "walk_forward_folds": walk_forward_rows,
        "walk_forward_balanced_accuracy_mean": float(np.mean([row["balanced_accuracy"] for row in walk_forward_rows])) if walk_forward_rows else None,
        "regime_threshold_min_rows": REGIME_THRESHOLD_MIN_ROWS,
        "regime_thresholds": regime_thresholds,
        "regime_threshold_details": regime_details,
    }


def _best_balanced_threshold(probabilities: np.ndarray, labels: np.ndarray, candidates: np.ndarray) -> float:
    scored = []
    for threshold in candidates:
        prediction = (probabilities >= threshold).astype(int)
        balanced = _balanced_accuracy(labels, prediction)
        positive_rate = float(prediction.mean())
        collapse_penalty = max(0.0, 0.10 - min(positive_rate, 1.0 - positive_rate)) * 1.5
        scored.append((balanced - collapse_penalty - 0.01 * abs(float(threshold) - 0.5), -abs(float(threshold) - 0.5), float(threshold)))
    return max(scored)[2]


def _balanced_accuracy(labels: np.ndarray, prediction: np.ndarray) -> float:
    recalls = []
    for label in (0, 1):
        actual = labels == label
        if actual.any():
            recalls.append(float((prediction[actual] == label).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def _limit_threshold_change(raw: float, previous: float | None, max_change: float) -> float:
    if previous is None or not np.isfinite(previous):
        return float(raw)
    return float(np.clip(raw, float(previous) - max_change, float(previous) + max_change))


def _thresholds_for_regimes(regimes: np.ndarray, default: float, regime_thresholds: dict[str, float]) -> np.ndarray:
    return np.asarray([float(regime_thresholds.get(str(regime), default)) for regime in regimes], dtype=float)


def _load_previous_thresholds() -> tuple[float | None, dict[str, float]]:
    if not LSTM_MODEL_METADATA_CACHE.exists():
        return None, {}
    try:
        payload = json.loads(LSTM_MODEL_METADATA_CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, {}
    threshold = pd.to_numeric(payload.get("decision_threshold"), errors="coerce")
    regime_thresholds = payload.get("decision_thresholds_by_regime", {})
    if not isinstance(regime_thresholds, dict):
        regime_thresholds = {}
    return (float(threshold) if pd.notna(threshold) else None), {
        str(key): float(value) for key, value in regime_thresholds.items() if pd.notna(pd.to_numeric(value, errors="coerce"))
    }


def _load_incumbent_quality() -> dict:
    if not LSTM_BACKTEST_CACHE.exists():
        return {"status": "missing"}
    try:
        return _backtest_quality(pd.read_parquet(LSTM_BACKTEST_CACHE))
    except (OSError, ValueError):
        return {"status": "unreadable"}


def _refresh_incumbent_predictions(monitor_features: pd.DataFrame) -> pd.DataFrame:
    if monitor_features.empty or not LSTM_MODEL_CACHE.exists() or not LSTM_SCALER_CACHE.exists() or not LSTM_MODEL_METADATA_CACHE.exists():
        return pd.DataFrame()
    try:
        metadata = json.loads(LSTM_MODEL_METADATA_CACHE.read_text(encoding="utf-8"))
        scaler = json.loads(LSTM_SCALER_CACHE.read_text(encoding="utf-8"))
        columns = scaler.get("columns", [])
        if columns != feature_columns():
            return pd.DataFrame()
        mean = pd.Series(scaler["mean"], index=columns, dtype=float)
        std = pd.Series(scaler["std"], index=columns, dtype=float).replace(0, 1.0)
        incumbent_model = LSTMClassifier(input_size=len(columns))
        incumbent_model.load_state_dict(torch.load(LSTM_MODEL_CACHE, map_location="cpu", weights_only=True))
        incumbent_model.eval()
        threshold = float(metadata.get("decision_threshold", 0.5))
        regime_thresholds = metadata.get("decision_thresholds_by_regime", {})
        if not isinstance(regime_thresholds, dict):
            regime_thresholds = {}
        predictions = _build_monitor_predictions(
            incumbent_model,
            monitor_features,
            mean,
            std,
            threshold,
            {str(key): float(value) for key, value in regime_thresholds.items()},
            model_version=str(metadata.get("model_version", "incumbent")),
        )
        quality = _load_incumbent_quality()
        production_eligible = _is_production_eligible(quality)
        predictions["raw_prediction_direction"] = predictions["prediction_direction"]
        predictions["production_eligible"] = production_eligible
        predictions["production_edge_vs_baseline"] = quality.get("edge_vs_baseline")
        if not production_eligible:
            predictions["prediction_direction"] = "研究模型未超過基準／不預測"
            predictions["is_actionable"] = False
        predictions.to_parquet(LSTM_PREDICTIONS_CACHE, index=False)
        return predictions
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError):
        return pd.DataFrame()


def _backtest_quality(backtest: pd.DataFrame) -> dict:
    if backtest.empty or not {"label", "predicted_label"}.issubset(backtest.columns):
        return {"status": "insufficient", "rows": int(len(backtest))}
    labels = pd.to_numeric(backtest["label"], errors="coerce")
    predictions = pd.to_numeric(backtest["predicted_label"], errors="coerce")
    valid = labels.notna() & predictions.notna()
    labels = labels[valid].astype(int).to_numpy()
    predictions = predictions[valid].astype(int).to_numpy()
    if not len(labels):
        return {"status": "insufficient", "rows": 0}
    accuracy = float((labels == predictions).mean())
    baseline_accuracy = float(max(labels.mean(), 1.0 - labels.mean()))
    return {
        "status": "usable",
        "rows": int(len(labels)),
        "accuracy": accuracy,
        "balanced_accuracy": _balanced_accuracy(labels, predictions),
        "baseline_accuracy": baseline_accuracy,
        "edge_vs_baseline": accuracy - baseline_accuracy,
        "predicted_positive_rate": float(predictions.mean()),
    }


def _promotion_decision(challenger: dict, incumbent: dict) -> dict:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if challenger.get("status") != "usable":
        return {"promoted": False, "reason": "挑戰模型測試資料不足", "evaluated_at_utc": now}
    challenger_accuracy = float(challenger["accuracy"])
    challenger_baseline = float(challenger["baseline_accuracy"])
    challenger_balanced = float(challenger["balanced_accuracy"])
    if challenger_accuracy < challenger_baseline + PROMOTION_MIN_EDGE:
        return {"promoted": False, "reason": "未超過多數類別基準 1%", "evaluated_at_utc": now}
    if challenger_balanced < 0.50:
        return {"promoted": False, "reason": "平衡準確率低於 50%", "evaluated_at_utc": now}
    if incumbent.get("status") == "usable" and float(incumbent.get("edge_vs_baseline", -1.0)) > 0:
        incumbent_floor = float(incumbent["accuracy"]) - PROMOTION_MAX_INCUMBENT_REGRESSION
        if challenger_accuracy < incumbent_floor:
            return {"promoted": False, "reason": "明顯差於現行正式模型", "evaluated_at_utc": now}
    return {"promoted": True, "reason": "通過基準與回歸檢查", "evaluated_at_utc": now}


def _is_production_eligible(quality: dict) -> bool:
    return bool(
        quality.get("status") == "usable"
        and float(quality.get("edge_vs_baseline", -1.0)) >= PRODUCTION_MIN_EDGE
        and float(quality.get("balanced_accuracy", 0.0)) >= 0.50
    )


def _save_challenger_report(challenger: dict, incumbent: dict, promotion: dict, threshold_metrics: dict) -> None:
    payload = {
        "model_version": MODEL_VERSION,
        "feature_version": LSTM_FEATURE_VERSION,
        "challenger": challenger,
        "incumbent": incumbent,
        "promotion": promotion,
        "promotion_rules": {
            "minimum_edge_vs_majority_baseline": PROMOTION_MIN_EDGE,
            "minimum_balanced_accuracy": 0.50,
            "maximum_regression_vs_qualified_incumbent": PROMOTION_MAX_INCUMBENT_REGRESSION,
        },
        "threshold_calibration": threshold_metrics,
    }
    path = LSTM_DIR / "lstm_challenger_evaluation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    history_path = LSTM_DIR / "lstm_challenger_history.parquet"
    history_row = pd.DataFrame(
        [
            {
                "evaluated_at_utc": promotion.get("evaluated_at_utc"),
                "model_version": MODEL_VERSION,
                "feature_version": LSTM_FEATURE_VERSION,
                "promoted": bool(promotion.get("promoted", False)),
                "reason": promotion.get("reason", ""),
                "rows": challenger.get("rows"),
                "accuracy": challenger.get("accuracy"),
                "balanced_accuracy": challenger.get("balanced_accuracy"),
                "baseline_accuracy": challenger.get("baseline_accuracy"),
                "edge_vs_baseline": challenger.get("edge_vs_baseline"),
                "predicted_positive_rate": challenger.get("predicted_positive_rate"),
                "incumbent_accuracy": incumbent.get("accuracy"),
                "incumbent_edge_vs_baseline": incumbent.get("edge_vs_baseline"),
            }
        ]
    )
    if history_path.exists():
        try:
            history_row = pd.concat([pd.read_parquet(history_path), history_row], ignore_index=True)
        except (OSError, ValueError):
            pass
    history_row["evaluated_at_utc"] = pd.to_datetime(history_row["evaluated_at_utc"], errors="coerce", utc=True)
    history_row = history_row.dropna(subset=["evaluated_at_utc"]).drop_duplicates(
        subset=["evaluated_at_utc", "model_version"], keep="last"
    ).sort_values("evaluated_at_utc")
    history_row.to_parquet(history_path, index=False)


def _market_regime_from_sequence(sequence_json: str) -> str:
    try:
        sequence = np.asarray(json.loads(sequence_json), dtype=float)
        latest = sequence[-1]
        columns = feature_columns()
        market_offset = columns.index("market_qqq_ret_5d")
        qqq_return = float(latest[market_offset])
        vix_return = float(latest[columns.index("market_vix_ret_5d")])
        hyg_return = float(latest[columns.index("market_hyg_ret_5d")])
        breadth = float(latest[columns.index("market_breadth_5d")])
    except (TypeError, ValueError, IndexError, json.JSONDecodeError):
        return "資料不足"
    if not all(np.isfinite(value) for value in [qqq_return, vix_return, hyg_return, breadth]):
        return "資料不足"

    if qqq_return <= -0.04 or vix_return >= 0.12 or hyg_return <= -0.02:
        return "高波動／壓力"
    if qqq_return <= -0.015 or breadth < 0.35:
        return "明顯下跌趨勢"
    if qqq_return >= 0.04 and (vix_return < 0 or hyg_return > 0):
        return "強勢反彈／轉折"
    if abs(qqq_return) < 0.01 and abs(vix_return) < 0.05:
        return "橫盤／低波動"
    return "正常／溫和上漲"


def _safe_nanmean(values: np.ndarray, axis=None) -> np.ndarray:
    if values.size == 0:
        return np.array([])
    with np.errstate(all="ignore"):
        result = np.nanmean(values, axis=axis)
    return np.nan_to_num(result, nan=0.0)


def _safe_nanstd(values: np.ndarray, axis=None) -> np.ndarray:
    if values.size == 0:
        return np.array([])
    with np.errstate(all="ignore"):
        result = np.nanstd(values, axis=axis)
    return np.nan_to_num(result, nan=1.0)


if __name__ == "__main__":
    main()
