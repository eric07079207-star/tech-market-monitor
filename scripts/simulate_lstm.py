from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import PRICE_CACHE
from src.lstm import (
    LSTM_DIR,
    LSTM_FEATURE_VERSION,
    build_lstm_feature_table,
    feature_columns,
)

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover
    raise RuntimeError("torch is required for LSTM simulation. Install dependencies first.") from exc


MODEL_VERSION = "lstm-sim-v1"
SIM_DIR = LSTM_DIR / "simulation_2020_2021"
SIM_FEATURE_CACHE = SIM_DIR / "lstm_features.parquet"
SIM_SPLIT_CACHE = SIM_DIR / "lstm_split.parquet"
SIM_BACKTEST_CACHE = SIM_DIR / "lstm_backtest.parquet"
SIM_PREDICTIONS_CACHE = SIM_DIR / "lstm_predictions.parquet"
SIM_MODEL_CACHE = SIM_DIR / "lstm_model.pt"
SIM_MODEL_METADATA_CACHE = SIM_DIR / "lstm_model.json"
SIM_SCALER_CACHE = SIM_DIR / "lstm_scaler.json"
SIM_STATUS_CACHE = SIM_DIR / "lstm_status.json"


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 48):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an LSTM market simulation on a fixed historical window.")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2021-12-31")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=["QQQ", "NVDA", "TSLA"],
        help="Optional symbol list. Defaults to the compact demo set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not PRICE_CACHE.exists():
        raise RuntimeError("prices.parquet is missing; run scripts/update_data.py first.")

    prices = pd.read_parquet(PRICE_CACHE)
    if "date" in prices.columns:
        prices["date"] = pd.to_datetime(prices["date"])

    features = build_lstm_feature_table(
        prices=prices,
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        max_samples=args.max_samples,
        sample_step=1,
        lookback_years=10,
    )
    if features.empty:
        raise RuntimeError("No simulation features could be built for the requested window.")

    split = _prepare_split(features)
    model, summary, backtest, predictions = _train_model(split)

    SIM_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(SIM_FEATURE_CACHE, index=False)
    split["frame"].to_parquet(SIM_SPLIT_CACHE, index=False)
    backtest.to_parquet(SIM_BACKTEST_CACHE, index=False)
    predictions.to_parquet(SIM_PREDICTIONS_CACHE, index=False)
    torch.save(model.state_dict(), SIM_MODEL_CACHE)
    _save_scaler(split, SIM_SCALER_CACHE)
    SIM_MODEL_METADATA_CACHE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    SIM_STATUS_CACHE.write_text(json.dumps(_build_status(summary, features, predictions, backtest), ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"simulation complete rows={len(features)} "
        f"train={summary['train_rows']} valid={summary['valid_rows']} test={summary['test_rows']} "
        f"valid_acc={summary['valid_acc']:.4f} test_acc={summary['test_acc']:.4f}"
    )


def _prepare_split(features: pd.DataFrame) -> dict:
    data = features.copy().sort_values(["date", "symbol"]).reset_index(drop=True)
    n = len(data)
    if n < 3:
        raise RuntimeError(f"Not enough rows for simulation split: {n}")
    train_end = max(int(n * 0.7), 1)
    valid_end = max(int(n * 0.85), train_end + 1)
    split = np.array(["test"] * n, dtype=object)
    split[:train_end] = "train"
    split[train_end:valid_end] = "valid"
    data["split"] = split

    rows = np.asarray([json.loads(row) for row in data["sequence_json"]], dtype=np.float32)
    labels = data["label"].astype(int).to_numpy()
    train_mask = split == "train"
    valid_mask = split == "valid"
    test_mask = split == "test"

    train_rows = rows[train_mask]
    feature_mean = pd.Series(_safe_nanmean(train_rows, axis=(0, 1)), index=feature_columns())
    feature_std = pd.Series(_safe_nanstd(train_rows, axis=(0, 1)), index=feature_columns()).replace(0, 1.0)
    normalized = (rows - feature_mean.to_numpy()) / feature_std.to_numpy()
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    meta = data[["symbol", "date", "target_date", "future_return", "label", "feature_version", "split", "horizon_days"]].copy()
    return {
        "frame": data,
        "x_train": normalized[train_mask],
        "y_train": labels[train_mask],
        "x_valid": normalized[valid_mask],
        "y_valid": labels[valid_mask],
        "x_test": normalized[test_mask],
        "y_test": labels[test_mask],
        "test_meta": meta[test_mask].reset_index(drop=True),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }


def _train_model(split: dict):
    model = LSTMClassifier(input_size=split["x_train"].shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(split["x_train"], dtype=torch.float32),
        torch.tensor(split["y_train"], dtype=torch.float32),
    )
    valid_x = torch.tensor(split["x_valid"], dtype=torch.float32)
    valid_y = torch.tensor(split["y_valid"], dtype=torch.float32)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

    best_state = None
    best_valid_loss = np.inf
    best_metrics = {"valid_loss": np.nan, "valid_acc": np.nan, "epoch": 0}
    history = []

    for epoch in range(1, 6):
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
            valid_acc = float((valid_pred == split["y_valid"]).mean())
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss, "valid_acc": valid_acc})
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_metrics = {"valid_loss": valid_loss, "valid_acc": valid_acc, "epoch": epoch}

    if best_state is None:
        raise RuntimeError("simulation training failed")
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_logits = model(torch.tensor(split["x_test"], dtype=torch.float32))
        test_prob = torch.sigmoid(test_logits).cpu().numpy()
        test_pred = (test_prob >= 0.5).astype(int)
        test_acc = float((test_pred == split["y_test"]).mean()) if len(split["y_test"]) else np.nan

    backtest = _build_backtest_frame(split["test_meta"], test_prob, test_pred)
    predictions = _build_predictions(model, split["test_meta"], split["x_test"])
    summary = {
        "model_version": MODEL_VERSION,
        "feature_version": str(split["test_meta"]["feature_version"].iloc[-1]) if not split["test_meta"].empty else LSTM_FEATURE_VERSION,
        "train_rows": int(len(split["y_train"])),
        "valid_rows": int(len(split["y_valid"])),
        "test_rows": int(len(split["y_test"])),
        "valid_loss": float(best_metrics["valid_loss"]),
        "valid_acc": float(best_metrics["valid_acc"]),
        "test_acc": test_acc,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "history": history,
    }
    return model, summary, backtest, predictions


def _build_backtest_frame(meta: pd.DataFrame, prob: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    frame = meta.copy()
    frame["predicted_prob_up"] = prob
    frame["predicted_label"] = pred
    frame["prediction_direction"] = np.where(frame["predicted_label"].eq(1), "看漲", "看跌")
    frame["model_version"] = MODEL_VERSION
    frame["created_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    frame["validated_at_utc"] = frame["created_at_utc"]
    frame["success"] = (
        (frame["predicted_label"].eq(1) & frame["future_return"].gt(0))
        | (frame["predicted_label"].eq(0) & frame["future_return"].lt(0))
    ).astype(bool)
    return frame


def _build_predictions(model, meta: pd.DataFrame, x_test: np.ndarray) -> pd.DataFrame:
    rows = []
    for row, seq in zip(meta.itertuples(index=False), x_test):
        with torch.no_grad():
            logits = model(torch.tensor(seq[None, :, :], dtype=torch.float32))
            prob = torch.sigmoid(logits).item()
        rows.append(
            {
                "symbol": row.symbol,
                "prediction_date": pd.to_datetime(row.date, utc=True),
                "target_date": pd.to_datetime(row.target_date, utc=True),
                "horizon_days": int(getattr(row, "horizon_days", 20)),
                "model_version": MODEL_VERSION,
                "feature_version": str(row.feature_version),
                "predicted_prob_up": float(prob),
                "predicted_label": int(prob >= 0.5),
                "prediction_direction": "看漲" if prob >= 0.5 else "看跌",
                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
    return pd.DataFrame(rows)


def _save_scaler(split: dict, path: Path) -> None:
    payload = {
        "feature_version": LSTM_FEATURE_VERSION,
        "columns": feature_columns(),
        "mean": split["feature_mean"].fillna(0.0).tolist(),
        "std": split["feature_std"].fillna(1.0).tolist(),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _build_status(summary: dict, features: pd.DataFrame, predictions: pd.DataFrame, backtest: pd.DataFrame) -> dict:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "enabled": True,
        "mode": "simulation",
        "status": "模擬版已完成",
        "model_version": summary["model_version"],
        "feature_version": summary["feature_version"],
        "horizon_days": int(features["horizon_days"].iloc[-1]) if not features.empty else 20,
        "sequence_length": int(features["sequence_length"].iloc[-1]) if not features.empty else 60,
        "last_train_at_utc": summary["trained_at_utc"],
        "last_predict_at_utc": _latest_timestamp(predictions, "created_at_utc"),
        "last_backtest_at_utc": _latest_timestamp(backtest, "created_at_utc"),
        "prediction_rows": int(len(predictions)),
        "backtest_rows": int(len(backtest)),
        "feature_rows": int(len(features)),
        "updated_at_utc": now,
    }


def _latest_timestamp(df: pd.DataFrame, column: str) -> str:
    if df.empty or column not in df:
        return ""
    values = pd.to_datetime(df[column], errors="coerce", utc=True).dropna()
    if values.empty:
        return ""
    return values.max().isoformat()


if __name__ == "__main__":
    main()
