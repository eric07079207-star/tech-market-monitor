from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ALL_TICKERS
from .data import PRICE_CACHE, cache_path
from .indicators import add_price_indicators


LSTM_DIR = cache_path("lstm")
LSTM_STATUS_CACHE = LSTM_DIR / "lstm_status.json"
LSTM_FEATURE_CACHE = LSTM_DIR / "lstm_features.parquet"
LSTM_SPLIT_CACHE = LSTM_DIR / "lstm_split.parquet"
LSTM_PREDICTIONS_CACHE = LSTM_DIR / "lstm_predictions.parquet"
LSTM_BACKTEST_CACHE = LSTM_DIR / "lstm_backtest.parquet"
LSTM_MODEL_CACHE = LSTM_DIR / "lstm_model.pt"
LSTM_MODEL_METADATA_CACHE = LSTM_DIR / "lstm_model.json"
LSTM_SCALER_CACHE = LSTM_DIR / "lstm_scaler.json"

LSTM_FEATURE_VERSION = "lstm-demo-v1"
LSTM_TARGET_HORIZON_DAYS = 20
LSTM_SEQUENCE_LENGTH = 60
LSTM_DEFAULT_SYMBOLS = ["QQQ", "NVDA", "TSLA"]

BASE_FEATURE_COLUMNS = [
    "close",
    "volume",
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "dist_ma_20",
    "dist_ma_50",
    "dist_ma_200",
    "ma200_slope_20d",
    "realized_vol_20d",
    "realized_vol_pctile_252d",
    "ret_z_20d",
    "volume_ratio_20d",
    "volume_z_60d",
    "atr_20d_pct",
    "gap_pct",
    "drawdown_52w",
]


@dataclass(frozen=True)
class LSTMDataBundle:
    features: pd.DataFrame
    split: pd.DataFrame


def load_lstm_status(path: Path | None = None) -> dict:
    path = path or LSTM_STATUS_CACHE
    if not path.exists():
        return default_lstm_status()
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return default_lstm_status()
    return _normalize_status(payload)


def save_lstm_status(payload: dict, path: Path | None = None) -> dict:
    path = path or LSTM_STATUS_CACHE
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_status(payload)
    path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2))
    return normalized


def default_lstm_status() -> dict:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "enabled": False,
        "mode": "scaffold",
        "status": "尚未建立 LSTM 模型，先使用穩定化骨架。",
        "model_version": "n/a",
        "feature_version": LSTM_FEATURE_VERSION,
        "horizon_days": LSTM_TARGET_HORIZON_DAYS,
        "sequence_length": LSTM_SEQUENCE_LENGTH,
        "last_train_at_utc": "",
        "last_predict_at_utc": "",
        "last_backtest_at_utc": "",
        "prediction_rows": 0,
        "backtest_rows": 0,
        "feature_rows": 0,
        "updated_at_utc": now,
    }


def load_lstm_artifacts() -> dict[str, pd.DataFrame | dict]:
    return {
        "features": _read_dataframe(LSTM_FEATURE_CACHE),
        "split": _read_dataframe(LSTM_SPLIT_CACHE),
        "predictions": _read_dataframe(LSTM_PREDICTIONS_CACHE),
        "backtest": _read_dataframe(LSTM_BACKTEST_CACHE),
        "status": load_lstm_status(),
    }


def build_lstm_feature_table(
    prices: pd.DataFrame | None = None,
    symbols: list[str] | None = None,
    target_horizon_days: int = LSTM_TARGET_HORIZON_DAYS,
    sequence_length: int = LSTM_SEQUENCE_LENGTH,
    lookback_years: int = 1,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    sample_step: int = 10,
    max_samples: int | None = None,
) -> pd.DataFrame:
    prices = prices.copy() if prices is not None else _load_prices()
    if prices.empty:
        return _empty_feature_table()

    symbols = symbols or LSTM_DEFAULT_SYMBOLS
    price_source = prices[prices["symbol"].isin(symbols)].copy()
    if price_source.empty:
        return _empty_feature_table()
    indicators = add_price_indicators(price_source)
    if indicators.empty:
        return _empty_feature_table()

    indicators = indicators.sort_values(["symbol", "date"]).reset_index(drop=True)
    max_date = pd.to_datetime(indicators["date"]).max()
    sample_cutoff = pd.to_datetime(start_date) if start_date is not None else max_date - pd.DateOffset(years=lookback_years)
    sample_end = pd.to_datetime(end_date) if end_date is not None else max_date
    wide = indicators.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index().ffill()
    wide.columns = [str(col) for col in wide.columns]

    rows = []
    for symbol, group in indicators.groupby("symbol", sort=False):
        data = group.sort_values("date").reset_index(drop=True)
        data["date"] = pd.to_datetime(data["date"], utc=False, errors="coerce")
        for idx in range(sequence_length - 1, len(data) - target_horizon_days, sample_step):
            current = data.iloc[idx]
            if pd.to_datetime(current["date"], errors="coerce") < sample_cutoff:
                continue
            if pd.to_datetime(current["date"], errors="coerce") > sample_end:
                continue
            window = data.iloc[idx - sequence_length + 1 : idx + 1]
            if window[BASE_FEATURE_COLUMNS].dropna().empty:
                continue
            future_price = data.iloc[idx + target_horizon_days]["close"]
            current_close = current.get("close", np.nan)
            if pd.isna(current_close) or pd.isna(future_price):
                continue
            future_return = future_price / current_close - 1
            label = int(future_return > 0)
            seq = _sequence_features(window, wide, symbol)
            if seq is None:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "date": current["date"],
                    "target_date": data.iloc[idx + target_horizon_days]["date"],
                    "horizon_days": target_horizon_days,
                    "sequence_length": sequence_length,
                    "future_return": float(future_return),
                    "label": label,
                    "feature_version": LSTM_FEATURE_VERSION,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "sequence_json": json.dumps(seq, ensure_ascii=False),
                }
            )

    if not rows:
        return _empty_feature_table()

    features = pd.DataFrame(rows)
    features["date"] = pd.to_datetime(features["date"], utc=True, errors="coerce")
    features["target_date"] = pd.to_datetime(features["target_date"], utc=True, errors="coerce")
    features = features.dropna(subset=["date", "target_date", "sequence_json"])
    if max_samples is not None and len(features) > max_samples:
        idx = np.linspace(0, len(features) - 1, max_samples, dtype=int)
        features = features.iloc[idx].reset_index(drop=True)
    return features.sort_values(["symbol", "date"]).reset_index(drop=True)


def save_lstm_feature_table(features: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or LSTM_FEATURE_CACHE
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(path, index=False)
    return path


def build_lstm_train_split(
    features: pd.DataFrame,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    data = features.copy().sort_values("date").reset_index(drop=True)
    cutoff_train = data["date"].quantile(train_ratio)
    cutoff_valid = data["date"].quantile(train_ratio + valid_ratio)
    data["split"] = np.select(
        [data["date"] <= cutoff_train, data["date"] <= cutoff_valid],
        ["train", "valid"],
        default="test",
    )
    return data


def save_lstm_split(split: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or LSTM_SPLIT_CACHE
    path.parent.mkdir(parents=True, exist_ok=True)
    split.to_parquet(path, index=False)
    return path


def build_lstm_status_from_artifacts(
    predictions: pd.DataFrame | None = None,
    backtest: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
) -> dict:
    predictions = predictions if predictions is not None else _read_dataframe(LSTM_PREDICTIONS_CACHE)
    backtest = backtest if backtest is not None else _read_dataframe(LSTM_BACKTEST_CACHE)
    features = features if features is not None else _read_dataframe(LSTM_FEATURE_CACHE)
    status = default_lstm_status()
    status.update(
        {
            "prediction_rows": int(len(predictions)),
            "backtest_rows": int(len(backtest)),
            "feature_rows": int(len(features)),
            "last_predict_at_utc": _latest_timestamp(predictions, "created_at_utc"),
            "last_backtest_at_utc": _latest_timestamp(backtest, "validated_at_utc"),
            "last_train_at_utc": _metadata_timestamp(LSTM_MODEL_METADATA_CACHE, "trained_at_utc"),
            "model_version": _safe_str(predictions["model_version"].iloc[-1]) if not predictions.empty and "model_version" in predictions else "n/a",
            "feature_version": _safe_str(features["feature_version"].iloc[-1]) if not features.empty and "feature_version" in features else LSTM_FEATURE_VERSION,
            "status": "已接上特徵表" if len(features) else "尚未接上特徵表",
            "enabled": bool(len(features)),
        }
    )
    return status


def summarize_lstm_status(status: dict) -> pd.DataFrame:
    payload = _normalize_status(status)
    rows = [
        ("狀態", payload.get("status", "n/a")),
        ("模式", payload.get("mode", "n/a")),
        ("模型版本", payload.get("model_version", "n/a")),
        ("特徵版本", payload.get("feature_version", "n/a")),
        ("特徵數", payload.get("feature_rows", 0)),
        ("預測筆數", payload.get("prediction_rows", 0)),
        ("回測筆數", payload.get("backtest_rows", 0)),
        ("最後預測 UTC", payload.get("last_predict_at_utc", "")),
        ("最後回測 UTC", payload.get("last_backtest_at_utc", "")),
        ("最後訓練 UTC", payload.get("last_train_at_utc", "")),
        ("最後更新 UTC", payload.get("updated_at_utc", "")),
    ]
    return pd.DataFrame(rows, columns=["項目", "值"])


def load_lstm_predictions(path: Path | None = None) -> pd.DataFrame:
    path = path or LSTM_PREDICTIONS_CACHE
    return _read_dataframe(path)


def load_lstm_backtest(path: Path | None = None) -> pd.DataFrame:
    path = path or LSTM_BACKTEST_CACHE
    return _read_dataframe(path)


def _sequence_features(window: pd.DataFrame, wide_close: pd.DataFrame, symbol: str) -> list[list[float]] | None:
    seq_rows = []
    for _, row in window.iterrows():
        current_date = pd.to_datetime(row.get("date"), errors="coerce")
        if pd.isna(current_date):
            return None
        seq_row = []
        for col in BASE_FEATURE_COLUMNS:
            seq_row.append(_numeric(row.get(col)))
        seq_row.append(_relative_feature(wide_close, symbol, current_date, "QQQ"))
        seq_row.append(_relative_feature(wide_close, symbol, current_date, "SPY"))
        seq_row.append(_relative_feature(wide_close, symbol, current_date, "SMH"))
        seq_rows.append(seq_row)
    return seq_rows


def feature_columns() -> list[str]:
    return BASE_FEATURE_COLUMNS + ["rel_qqq_close", "rel_spy_close", "rel_smh_close"]


def _relative_feature(wide_close: pd.DataFrame, symbol: str, date: pd.Timestamp, benchmark: str) -> float:
    if wide_close.empty or symbol not in wide_close or benchmark not in wide_close:
        return np.nan
    history = wide_close[[symbol, benchmark]].dropna()
    if history.empty:
        return np.nan
    if date not in history.index:
        pos = history.index.searchsorted(date)
        if pos >= len(history.index):
            return np.nan
        date = history.index[pos]
    base = history.loc[date]
    if isinstance(base, pd.DataFrame):
        base = base.iloc[0]
    sym_val = base[symbol]
    bench_val = base[benchmark]
    if isinstance(sym_val, pd.Series):
        sym_val = sym_val.iloc[0]
    if isinstance(bench_val, pd.Series):
        bench_val = bench_val.iloc[0]
    if pd.isna(sym_val) or pd.isna(bench_val) or bench_val == 0:
        return np.nan
    return float(sym_val / bench_val - 1)


def _load_prices() -> pd.DataFrame:
    if not PRICE_CACHE.exists():
        return pd.DataFrame()
    prices = pd.read_parquet(PRICE_CACHE)
    if "date" in prices:
        prices["date"] = pd.to_datetime(prices["date"])
    return prices


def _read_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        if path.suffix == ".json":
            return pd.DataFrame([json.loads(path.read_text())])
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _metadata_timestamp(path: Path, key: str) -> str:
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return ""
    value = payload.get(key, "")
    return str(value) if value else ""


def _empty_feature_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "symbol",
            "date",
            "target_date",
            "horizon_days",
            "sequence_length",
            "future_return",
            "label",
            "feature_version",
            "created_at_utc",
            "sequence_json",
        ]
    )


def _latest_timestamp(data: pd.DataFrame, column: str) -> str:
    if data.empty or column not in data:
        return ""
    value = pd.to_datetime(data[column], errors="coerce", utc=True).max()
    if pd.isna(value):
        return ""
    return value.isoformat()


def _normalize_status(payload: dict) -> dict:
    status = default_lstm_status()
    if not payload:
        return status
    status.update(payload)
    return status


def _safe_str(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    return text if text.lower() != "nan" else ""


def _numeric(value: object) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return np.nan
    return num if np.isfinite(num) else np.nan
