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
    LSTM_FEATURE_CACHE,
    LSTM_MODEL_CACHE,
    LSTM_MODEL_METADATA_CACHE,
    LSTM_PREDICTIONS_CACHE,
    LSTM_SCALER_CACHE,
    LSTM_SPLIT_CACHE,
    LSTM_FEATURE_VERSION,
    LSTM_TARGET_HORIZON_DAYS,
    LSTM_SEQUENCE_LENGTH,
    build_lstm_feature_table,
    build_lstm_status_from_artifacts,
    feature_columns,
    save_lstm_status,
)
from src.data import PRICE_CACHE


try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover - surfaced in CI/runtime
    raise RuntimeError("torch is required for LSTM training. Install dependencies first.") from exc


MODEL_VERSION = "lstm-direction-v1"
DEFAULT_SYMBOLS = ["TSLA"]
EPOCHS = 5
BATCH_SIZE = 64
HIDDEN_SIZE = 48
NUM_LAYERS = 1
LEARNING_RATE = 1e-3
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15


@dataclass(frozen=True)
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_valid: np.ndarray
    y_valid: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    test_meta: pd.DataFrame
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
    parser.add_argument("--lookback-years", type=int, default=5, help="Historical lookback window for feature building.")
    parser.add_argument("--sample-step", type=int, default=5, help="Sample every N trading days.")
    args = parser.parse_args()

    if not PRICE_CACHE.exists():
        raise RuntimeError("prices.parquet is missing; run scripts/update_data.py first.")
    prices = pd.read_parquet(PRICE_CACHE)
    if "date" in prices:
        prices["date"] = pd.to_datetime(prices["date"])

    features = build_lstm_feature_table(
        prices=prices,
        target_horizon_days=LSTM_TARGET_HORIZON_DAYS,
        sequence_length=LSTM_SEQUENCE_LENGTH,
        symbols=args.symbols,
        lookback_years=args.lookback_years,
        sample_step=args.sample_step,
    )
    features.to_parquet(LSTM_FEATURE_CACHE, index=False)

    if features.empty:
        raise RuntimeError("No LSTM features available for training.")

    split_data = _prepare_split(features)
    if split_data.x_train.size == 0 or split_data.x_valid.size == 0:
        raise RuntimeError("Insufficient data to train LSTM.")

    model = LSTMClassifier(input_size=split_data.x_train.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = torch.utils.data.TensorDataset(torch.tensor(split_data.x_train, dtype=torch.float32), torch.tensor(split_data.y_train, dtype=torch.float32))
    valid_x = torch.tensor(split_data.x_valid, dtype=torch.float32)
    valid_y = torch.tensor(split_data.y_valid, dtype=torch.float32)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

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
        test_logits = model(torch.tensor(split_data.x_test, dtype=torch.float32))
        test_prob = torch.sigmoid(test_logits).cpu().numpy()
        test_pred = (test_prob >= 0.5).astype(int)
        test_acc = float((test_pred == split_data.y_test).mean()) if len(split_data.y_test) else np.nan

    LSTM_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state["model_state"], LSTM_MODEL_CACHE)
    _save_scaler(split_data, LSTM_SCALER_CACHE)

    backtest = _build_backtest_frame(split_data.test_meta, test_prob, test_pred)
    backtest.to_parquet(LSTM_BACKTEST_CACHE, index=False)

    latest_predictions = _build_latest_predictions(model, features, split_data.feature_mean, split_data.feature_std, split_data.test_meta)
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
        "train_ratio": TRAIN_RATIO,
        "valid_ratio": VALID_RATIO,
        "test_ratio": TEST_RATIO,
        "train_rows": int(len(split_data.y_train)),
        "valid_rows": int(len(split_data.y_valid)),
        "test_rows": int(len(split_data.y_test)),
        "valid_loss": float(best_state["valid_loss"]),
        "valid_acc": float(best_state["valid_acc"]),
        "test_acc": test_acc,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "history": history,
    }
    LSTM_MODEL_METADATA_CACHE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    status = build_lstm_status_from_artifacts(latest_predictions, backtest, features)
    status.update(
        {
            "enabled": True,
            "mode": "train",
            "model_version": MODEL_VERSION,
            "feature_version": str(features["feature_version"].iloc[-1]),
            "horizon_days": LSTM_TARGET_HORIZON_DAYS,
            "sequence_length": LSTM_SEQUENCE_LENGTH,
            "last_train_at_utc": metadata["trained_at_utc"],
            "status": f"已訓練完成，驗證準確率 {best_state['valid_acc']:.2%}，測試準確率 {test_acc:.2%}",
            "prediction_rows": int(len(latest_predictions)),
            "backtest_rows": int(len(backtest)),
            "feature_rows": int(len(features)),
        }
    )
    save_lstm_status(status)

    print(
        f"trained lstm model_version={MODEL_VERSION} feature_rows={len(features)} "
        f"train={len(split_data.y_train)} valid={len(split_data.y_valid)} test={len(split_data.y_test)} "
        f"valid_acc={best_state['valid_acc']:.4f} test_acc={test_acc:.4f}"
    )


def _prepare_split(features: pd.DataFrame) -> SplitData:
    data = features.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
    n = len(data)
    train_end = max(int(n * TRAIN_RATIO), 1)
    valid_end = max(int(n * (TRAIN_RATIO + VALID_RATIO)), train_end + 1)
    split = np.array(["test"] * n, dtype=object)
    split[:train_end] = "train"
    split[train_end:valid_end] = "valid"
    data["split"] = split
    rows = []
    for row in data.itertuples():
        seq = json.loads(row.sequence_json)
        rows.append(seq)
    sequence_array = np.asarray(rows, dtype=np.float32)
    labels = data["label"].astype(int).to_numpy()
    meta = data[["symbol", "date", "target_date", "future_return", "label", "feature_version", "split"]].copy()
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
        test_meta=meta[test_mask].reset_index(drop=True),
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


def _build_backtest_frame(meta: pd.DataFrame, prob: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    if meta.empty:
        return pd.DataFrame()
    frame = meta.copy()
    frame["predicted_prob_up"] = prob
    frame["predicted_label"] = pred
    frame["prediction_direction"] = np.where(frame["predicted_label"].eq(1), "看漲", "看跌")
    frame["model_version"] = MODEL_VERSION
    frame["created_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    frame["validated_at_utc"] = frame["created_at_utc"]
    frame["success"] = ((frame["predicted_label"].eq(1) & frame["future_return"].gt(0)) | (frame["predicted_label"].eq(0) & frame["future_return"].lt(0))).astype(bool)
    return frame


def _build_latest_predictions(
    model: torch.nn.Module,
    features: pd.DataFrame,
    mean: pd.Series,
    std: pd.Series,
    test_meta: pd.DataFrame,
) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()

    latest_rows = []
    for symbol, group in features.groupby("symbol", sort=False):
        latest = group.sort_values("date").iloc[-1]
        seq = np.asarray(json.loads(latest.sequence_json), dtype=np.float32)
        normalized = (seq - mean.to_numpy()) / std.to_numpy()
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        with torch.no_grad():
            logits = model(torch.tensor(normalized[None, :, :], dtype=torch.float32))
            prob = torch.sigmoid(logits).item()
        latest_rows.append(
            {
                "symbol": symbol,
                "prediction_date": pd.to_datetime(latest.date, utc=True),
                "target_date": pd.to_datetime(latest.target_date, utc=True),
                "horizon_days": int(latest.horizon_days),
                "model_version": MODEL_VERSION,
                "feature_version": str(latest.feature_version),
                "predicted_prob_up": float(prob),
                "predicted_label": int(prob >= 0.5),
                "prediction_direction": "看漲" if prob >= 0.5 else "看跌",
                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
    return pd.DataFrame(latest_rows)


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
