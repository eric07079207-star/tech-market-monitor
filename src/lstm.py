from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ALL_TICKERS, STOCK_TICKERS
from .data import PRICE_CACHE, cache_path
from .indicators import add_price_indicators


_ARTIFACT_SUBDIR = os.getenv("LSTM_ARTIFACT_SUBDIR", "").strip("/ ")
LSTM_DIR = cache_path(f"lstm/{_ARTIFACT_SUBDIR}") if _ARTIFACT_SUBDIR else cache_path("lstm")
LSTM_STATUS_CACHE = LSTM_DIR / "lstm_status.json"
LSTM_TRAIN_FEATURE_CACHE = LSTM_DIR / "lstm_train_features.parquet"
LSTM_MONITOR_FEATURE_CACHE = LSTM_DIR / "lstm_monitor_features.parquet"
LSTM_LEGACY_FEATURE_CACHE = LSTM_DIR / "lstm_features.parquet"
# Backward-compatible alias for older local tools. Training now writes the explicit train file.
LSTM_FEATURE_CACHE = LSTM_TRAIN_FEATURE_CACHE
LSTM_SPLIT_CACHE = LSTM_DIR / "lstm_split.parquet"
LSTM_PREDICTIONS_CACHE = LSTM_DIR / "lstm_predictions.parquet"
LSTM_BACKTEST_CACHE = LSTM_DIR / "lstm_backtest.parquet"
LSTM_MODEL_CACHE = LSTM_DIR / "lstm_model.pt"
LSTM_MODEL_METADATA_CACHE = LSTM_DIR / "lstm_model.json"
LSTM_SCALER_CACHE = LSTM_DIR / "lstm_scaler.json"

LSTM_FEATURE_VERSION = "lstm-regime-v2"
LSTM_TARGET_HORIZON_DAYS = 20
LSTM_SEQUENCE_LENGTH = 60
LSTM_SHARED_SYMBOLS = STOCK_TICKERS.copy()
LSTM_SPECIALIST_SYMBOLS = ["TSLA"]
LSTM_DEFAULT_SYMBOLS = LSTM_SHARED_SYMBOLS

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

MARKET_FEATURE_COLUMNS = [
    "market_qqq_ret_5d",
    "market_spy_ret_5d",
    "market_smh_ret_5d",
    "market_vix_ret_5d",
    "market_tlt_ret_5d",
    "market_hyg_ret_5d",
    "market_breadth_5d",
    "market_dispersion_5d",
]

EVENT_FEATURE_COLUMNS = ["event_count_20d", "event_signed_impact_20d", "event_confidence_20d"]
FUNDAMENTAL_FEATURE_COLUMNS = ["fundamental_revenue_growth", "fundamental_ocf_growth", "fundamental_age_days"]
TICKER_FEATURE_COLUMNS = [f"ticker_{ticker.lower()}" for ticker in LSTM_SHARED_SYMBOLS]


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
        "train_feature_rows": 0,
        "monitor_feature_rows": 0,
        "train_rows": 0,
        "valid_rows": 0,
        "test_rows": 0,
        "backtest_accuracy": None,
        "prediction_confidence": None,
        "prediction_confidence_level": "低信心",
        "prediction_date": "",
        "prediction_target_date": "",
        "prediction_target_date_type": "estimated",
        "monitor_updated_at_utc": "",
        "updated_at_utc": now,
    }


def load_lstm_artifacts() -> dict[str, pd.DataFrame | dict]:
    train_features = _load_train_features()
    return {
        "features": train_features,
        "train_features": train_features,
        "monitor_features": _read_dataframe(LSTM_MONITOR_FEATURE_CACHE),
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
    fact_events: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    prices = prices.copy() if prices is not None else _load_prices()
    if prices.empty:
        return _empty_feature_table()

    symbols = symbols or LSTM_DEFAULT_SYMBOLS
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    price_source = prices[prices["symbol"].isin(symbols)].copy()
    if price_source.empty:
        return _empty_feature_table()
    indicators = add_price_indicators(price_source)
    if indicators.empty:
        return _empty_feature_table()

    indicators = indicators.sort_values(["symbol", "date"]).reset_index(drop=True)
    max_date = pd.to_datetime(indicators["date"]).max()
    sample_cutoff = pd.to_datetime(start_date) if start_date is not None else max_date - pd.DateOffset(years=lookback_years or 20)
    sample_end = pd.to_datetime(end_date) if end_date is not None else max_date
    # Target stocks are trained individually, while the complete cached universe supplies
    # point-in-time market context. Keeping those separate avoids accidentally treating
    # a benchmark as a target label.
    market_indicators = add_price_indicators(prices.dropna(subset=["date", "symbol", "close"]))
    wide = market_indicators.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index().ffill()
    wide.columns = [str(col) for col in wide.columns]
    event_context = _prepare_event_context(fact_events)
    fundamental_context = _prepare_fundamental_context(fundamentals)
    # Overlapping training windows reuse almost every daily row.  Cache the
    # point-in-time vector so purged walk-forward samples stay identical while
    # avoiding thousands of repeated pandas lookups.
    row_feature_cache: dict[tuple[str, pd.Timestamp], list[float]] = {}
    market_feature_cache: dict[pd.Timestamp, list[float]] = {}

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
            seq = _sequence_features(
                window,
                wide,
                symbol,
                event_context,
                fundamental_context,
                row_feature_cache=row_feature_cache,
                market_feature_cache=market_feature_cache,
            )
            if seq is None:
                continue
            qqq_future_return = _benchmark_future_return(wide, "QQQ", current["date"], data.iloc[idx + target_horizon_days]["date"])
            rows.append(
                {
                    "symbol": symbol,
                    "date": current["date"],
                    "target_date": data.iloc[idx + target_horizon_days]["date"],
                    "horizon_days": target_horizon_days,
                    "sequence_length": sequence_length,
                    "future_return": float(future_return),
                    "label": label,
                    "qqq_future_return": qqq_future_return,
                    "qqq_label": int(qqq_future_return > 0) if np.isfinite(qqq_future_return) else np.nan,
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


def build_lstm_monitor_feature_table(
    prices: pd.DataFrame | None = None,
    symbols: list[str] | None = None,
    target_horizon_days: int = LSTM_TARGET_HORIZON_DAYS,
    sequence_length: int = LSTM_SEQUENCE_LENGTH,
    fact_events: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build current inference rows without requiring a known future return."""
    prices = prices.copy() if prices is not None else _load_prices()
    if prices.empty:
        return _empty_monitor_feature_table()
    symbols = symbols or LSTM_DEFAULT_SYMBOLS
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    source = prices[prices["symbol"].astype(str).isin(symbols)].copy()
    if source.empty:
        return _empty_monitor_feature_table()
    indicators = add_price_indicators(source)
    if indicators.empty:
        return _empty_monitor_feature_table()
    indicators["date"] = pd.to_datetime(indicators["date"], errors="coerce")
    indicators = indicators.dropna(subset=["date"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    market_indicators = add_price_indicators(prices.dropna(subset=["date", "symbol", "close"]))
    wide = market_indicators.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index().ffill()
    wide.columns = [str(col) for col in wide.columns]
    event_context = _prepare_event_context(fact_events)
    fundamental_context = _prepare_fundamental_context(fundamentals)
    row_feature_cache: dict[tuple[str, pd.Timestamp], list[float]] = {}
    market_feature_cache: dict[pd.Timestamp, list[float]] = {}
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for symbol, group in indicators.groupby("symbol", sort=False):
        data = group.sort_values("date").reset_index(drop=True)
        if len(data) < sequence_length:
            continue
        window = data.iloc[-sequence_length:]
        if window[BASE_FEATURE_COLUMNS].isna().all(axis=None):
            continue
        seq = _sequence_features(
            window,
            wide,
            str(symbol),
            event_context,
            fundamental_context,
            row_feature_cache=row_feature_cache,
            market_feature_cache=market_feature_cache,
        )
        current = data.iloc[-1]
        current_date = pd.to_datetime(current["date"], utc=True)
        if seq is None or pd.isna(current.get("close")):
            continue
        target_date = current_date + pd.offsets.BDay(target_horizon_days)
        rows.append(
            {
                "symbol": str(symbol),
                "prediction_date": current_date,
                "target_date": target_date,
                "target_date_type": "estimated",
                "horizon_days": int(target_horizon_days),
                "sequence_length": int(sequence_length),
                "current_close": float(current["close"]),
                "feature_version": LSTM_FEATURE_VERSION,
                "created_at_utc": created_at,
                "sequence_json": json.dumps(seq, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows) if rows else _empty_monitor_feature_table()


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
    monitor_features: pd.DataFrame | None = None,
) -> dict:
    predictions = predictions if predictions is not None else _read_dataframe(LSTM_PREDICTIONS_CACHE)
    backtest = backtest if backtest is not None else _read_dataframe(LSTM_BACKTEST_CACHE)
    features = features if features is not None else _load_train_features()
    monitor_features = monitor_features if monitor_features is not None else _read_dataframe(LSTM_MONITOR_FEATURE_CACHE)
    status = load_lstm_status()
    metadata = _read_json(LSTM_MODEL_METADATA_CACHE)
    metadata_feature_rows = int(metadata.get("feature_rows", 0) or 0)
    metadata_feature_version = _safe_str(metadata.get("feature_version"))
    test_acc = metadata.get("test_acc")
    latest_prediction = predictions.sort_values("prediction_date").tail(1) if not predictions.empty and "prediction_date" in predictions else pd.DataFrame()
    probability = _numeric(latest_prediction.iloc[0].get("predicted_prob_up")) if not latest_prediction.empty else np.nan
    confidence, confidence_level = prediction_confidence(probability, int(metadata.get("test_rows", len(backtest))))
    status.update(
        {
            "prediction_rows": int(len(predictions)),
            "backtest_rows": int(len(backtest)),
            # Large training features are intentionally not always published to
            # GitHub. Prefer the immutable model metadata over a stale local demo.
            "feature_rows": metadata_feature_rows or int(len(features)),
            "train_feature_rows": metadata_feature_rows or int(len(features)),
            "monitor_feature_rows": int(len(monitor_features)),
            "train_rows": int(metadata.get("train_rows", 0)),
            "valid_rows": int(metadata.get("valid_rows", 0)),
            "test_rows": int(metadata.get("test_rows", 0)),
            "backtest_accuracy": float(test_acc) if test_acc is not None else None,
            "prediction_confidence": confidence,
            "prediction_confidence_level": confidence_level,
            "prediction_date": _safe_str(latest_prediction.iloc[0].get("prediction_date")) if not latest_prediction.empty else "",
            "prediction_target_date": _safe_str(latest_prediction.iloc[0].get("target_date")) if not latest_prediction.empty else "",
            "prediction_target_date_type": _safe_str(latest_prediction.iloc[0].get("target_date_type")) or "estimated" if not latest_prediction.empty else "estimated",
            "monitor_updated_at_utc": _latest_timestamp(monitor_features, "created_at_utc"),
            "last_predict_at_utc": _latest_timestamp(predictions, "created_at_utc"),
            "last_backtest_at_utc": _latest_timestamp(backtest, "validated_at_utc"),
            "last_train_at_utc": _metadata_timestamp(LSTM_MODEL_METADATA_CACHE, "trained_at_utc"),
            "model_version": _safe_str(predictions["model_version"].iloc[-1]) if not predictions.empty and "model_version" in predictions else "n/a",
            "feature_version": metadata_feature_version or (_safe_str(features["feature_version"].iloc[-1]) if not features.empty and "feature_version" in features else LSTM_FEATURE_VERSION),
            "status": status.get("status") or ("已接上訓練特徵與監控特徵" if len(features) or len(monitor_features) else "尚未接上特徵表"),
            "enabled": bool(metadata.get("model_version") or len(features)),
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
        ("訓練特徵數", payload.get("train_feature_rows", 0)),
        ("日常監控特徵數", payload.get("monitor_feature_rows", 0)),
        ("訓練/驗證/測試樣本", f"{payload.get('train_rows', 0)} / {payload.get('valid_rows', 0)} / {payload.get('test_rows', 0)}"),
        ("回測正確率", _format_percent(payload.get("backtest_accuracy"))),
        ("預測信心", f"{payload.get('prediction_confidence_level', '低信心')} ({_format_percent(payload.get('prediction_confidence'))})"),
        ("預測目標日", payload.get("prediction_target_date", "")),
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


def _sequence_features(
    window: pd.DataFrame,
    wide_close: pd.DataFrame,
    symbol: str,
    event_context: dict[str, pd.DataFrame] | None = None,
    fundamental_context: dict[str, pd.DataFrame] | None = None,
    row_feature_cache: dict[tuple[str, pd.Timestamp], list[float]] | None = None,
    market_feature_cache: dict[pd.Timestamp, list[float]] | None = None,
) -> list[list[float]] | None:
    seq_rows = []
    for _, row in window.iterrows():
        current_date = pd.to_datetime(row.get("date"), errors="coerce")
        if pd.isna(current_date):
            return None
        cache_date = pd.Timestamp(current_date).tz_localize(None).normalize()
        cache_key = (str(symbol).upper(), cache_date)
        cached_row = (row_feature_cache or {}).get(cache_key)
        if cached_row is not None:
            seq_rows.append(cached_row)
            continue
        seq_row = []
        for col in BASE_FEATURE_COLUMNS:
            seq_row.append(_numeric(row.get(col)))
        seq_row.append(_relative_feature(wide_close, symbol, current_date, "QQQ"))
        seq_row.append(_relative_feature(wide_close, symbol, current_date, "SPY"))
        seq_row.append(_relative_feature(wide_close, symbol, current_date, "SMH"))
        market_values = (market_feature_cache or {}).get(cache_date)
        if market_values is None:
            market_values = _market_context_features(wide_close, current_date)
            if market_feature_cache is not None:
                market_feature_cache[cache_date] = market_values
        seq_row.extend(market_values)
        seq_row.extend(_event_context_features(event_context, symbol, current_date))
        seq_row.extend(_fundamental_context_features(fundamental_context, symbol, current_date))
        seq_row.extend(_ticker_identity_features(symbol))
        if row_feature_cache is not None:
            row_feature_cache[cache_key] = seq_row
        seq_rows.append(seq_row)
    return seq_rows


def feature_columns() -> list[str]:
    return BASE_FEATURE_COLUMNS + ["rel_qqq_close", "rel_spy_close", "rel_smh_close"] + MARKET_FEATURE_COLUMNS + EVENT_FEATURE_COLUMNS + FUNDAMENTAL_FEATURE_COLUMNS + TICKER_FEATURE_COLUMNS


def _ticker_identity_features(symbol: str) -> list[float]:
    return [1.0 if symbol.upper() == ticker else 0.0 for ticker in LSTM_SHARED_SYMBOLS]


def _market_context_features(wide_close: pd.DataFrame, date: pd.Timestamp) -> list[float]:
    returns = [_return_feature(wide_close, ticker, date, 5) for ticker in ["QQQ", "SPY", "SMH", "^VIX", "TLT", "HYG"]]
    target_symbols = [ticker for ticker in LSTM_SHARED_SYMBOLS if ticker in wide_close]
    cross_returns = [_return_feature(wide_close, ticker, date, 5) for ticker in target_symbols]
    finite = [value for value in cross_returns if np.isfinite(value)]
    breadth = float(np.mean(np.asarray(finite) > 0)) if finite else np.nan
    dispersion = float(np.std(finite)) if finite else np.nan
    return returns + [breadth, dispersion]


def _return_feature(wide_close: pd.DataFrame, ticker: str, date: pd.Timestamp, periods: int) -> float:
    if wide_close.empty or ticker not in wide_close:
        return np.nan
    series = pd.to_numeric(wide_close[ticker], errors="coerce").dropna()
    if series.empty:
        return np.nan
    position = series.index.searchsorted(date, side="right") - 1
    if position < periods:
        return np.nan
    current, previous = series.iloc[position], series.iloc[position - periods]
    return float(current / previous - 1) if pd.notna(current) and pd.notna(previous) and previous else np.nan


def _prepare_event_context(facts: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    if facts is None or facts.empty:
        return {}
    data = facts.copy()
    data["timestamp"] = pd.to_datetime(data.get("timestamp_utc", data.get("event_date")), errors="coerce", utc=True).dt.tz_localize(None)
    data["impact_score"] = pd.to_numeric(data.get("impact_score"), errors="coerce").fillna(0.0)
    data["confidence"] = pd.to_numeric(data.get("confidence"), errors="coerce").fillna(0.0)
    direction = data.get("impact_direction", pd.Series("neutral", index=data.index)).astype(str).str.lower()
    data["signed_impact"] = data["impact_score"] * np.where(direction.eq("positive"), 1.0, np.where(direction.eq("negative"), -1.0, 0.0))
    return {ticker: group.dropna(subset=["timestamp"]).sort_values("timestamp") for ticker, group in _expand_affected_tickers(data).groupby("ticker")}


def _expand_affected_tickers(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in data.itertuples(index=False):
        affected = str(getattr(row, "affected_tickers", "") or getattr(row, "entity", "")).upper().replace(";", ",").split(",")
        for ticker in {item.strip() for item in affected if item.strip()}:
            rows.append({"ticker": ticker, "timestamp": getattr(row, "timestamp"), "signed_impact": getattr(row, "signed_impact"), "confidence": getattr(row, "confidence")})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["ticker", "timestamp", "signed_impact", "confidence"])


def _event_context_features(context: dict[str, pd.DataFrame] | None, symbol: str, date: pd.Timestamp) -> list[float]:
    data = (context or {}).get(symbol.upper())
    if data is None or data.empty:
        return [0.0, 0.0, 0.0]
    end = pd.Timestamp(date).tz_localize(None) if getattr(pd.Timestamp(date), "tzinfo", None) else pd.Timestamp(date)
    recent = data[(data["timestamp"] <= end) & (data["timestamp"] > end - pd.Timedelta(days=20))]
    if recent.empty:
        return [0.0, 0.0, 0.0]
    return [float(len(recent)), float(recent["signed_impact"].sum()), float(recent["confidence"].mean())]


def _prepare_fundamental_context(observations: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    if observations is None or observations.empty:
        return {}
    data = observations.copy()
    data["filed_at"] = pd.to_datetime(data.get("filed_at"), errors="coerce")
    data["value"] = pd.to_numeric(data.get("value"), errors="coerce")
    data = data.dropna(subset=["ticker", "metric", "filed_at", "value"])
    return {ticker: group.sort_values("filed_at") for ticker, group in data.groupby(data["ticker"].astype(str).str.upper())}


def _fundamental_context_features(context: dict[str, pd.DataFrame] | None, symbol: str, date: pd.Timestamp) -> list[float]:
    data = (context or {}).get(symbol.upper())
    if data is None or data.empty:
        return [0.0, 0.0, 730.0]
    asof = pd.Timestamp(date).tz_localize(None) if getattr(pd.Timestamp(date), "tzinfo", None) else pd.Timestamp(date)
    available = data[data["filed_at"] <= asof]
    if available.empty:
        return [0.0, 0.0, 730.0]
    growth = []
    for metric in ["revenue", "operating_cash_flow"]:
        values = available[available["metric"].eq(metric)].sort_values("filed_at")["value"].tail(2).to_numpy(dtype=float)
        growth.append(float(np.clip(values[-1] / values[-2] - 1, -2.0, 2.0)) if len(values) == 2 and values[-2] else 0.0)
    age = float((asof - available["filed_at"].max()).days)
    return growth + [float(np.clip(age, 0, 730))]


def _benchmark_future_return(wide_close: pd.DataFrame, ticker: str, start: object, end: object) -> float:
    if wide_close.empty or ticker not in wide_close:
        return np.nan
    series = pd.to_numeric(wide_close[ticker], errors="coerce").dropna()
    if series.empty:
        return np.nan
    start_value = _asof_value(series, start)
    end_value = _asof_value(series, end)
    return float(end_value / start_value - 1) if np.isfinite(start_value) and np.isfinite(end_value) and start_value else np.nan


def _asof_value(series: pd.Series, date: object) -> float:
    index = series.index.searchsorted(pd.Timestamp(date), side="right") - 1
    return float(series.iloc[index]) if index >= 0 else np.nan


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


def _load_train_features() -> pd.DataFrame:
    data = _read_dataframe(LSTM_TRAIN_FEATURE_CACHE)
    if not data.empty:
        return data
    return _read_dataframe(LSTM_LEGACY_FEATURE_CACHE)


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
            "qqq_future_return",
            "qqq_label",
            "feature_version",
            "created_at_utc",
            "sequence_json",
        ]
    )


def _empty_monitor_feature_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "symbol", "prediction_date", "target_date", "target_date_type", "horizon_days",
            "sequence_length", "current_close", "feature_version", "created_at_utc", "sequence_json",
        ]
    )


def prediction_confidence(probability: float, sample_count: int) -> tuple[float | None, str]:
    if not np.isfinite(probability):
        return None, "低信心"
    distance = abs(float(probability) - 0.5) * 2
    if sample_count < 50 or distance < 0.16:
        level = "低信心"
    elif sample_count < 100 or distance < 0.30:
        level = "中信心"
    else:
        level = "高信心"
    return float(distance), level


def _format_percent(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(number) else f"{number:.1%}"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


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
