from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CACHE_DIR


HISTORICAL_BACKTEST_CACHE = CACHE_DIR / "backtest" / "stratified_market_samples.parquet"
HISTORICAL_BACKTEST_SEED = 20260724
HISTORICAL_BACKTEST_TARGET_SAMPLES = 500
REGIME_TARGETS = {
    "正常／溫和上漲": 0.30,
    "明顯下跌趨勢": 0.25,
    "高波動／壓力": 0.20,
    "強勢反彈／轉折": 0.15,
    "橫盤／低波動": 0.10,
}


def build_stratified_market_samples(
    prices: pd.DataFrame,
    target_samples: int = HISTORICAL_BACKTEST_TARGET_SAMPLES,
    seed: int = HISTORICAL_BACKTEST_SEED,
) -> pd.DataFrame:
    """Build reproducible point-in-time technical/market samples without inventing KG news history."""
    qqq = _history(prices, "QQQ")
    vix = _history(prices, "^VIX")
    hyg = _history(prices, "HYG")
    if len(qqq) < 260:
        return _empty_samples()

    features = qqq.set_index("date")[["close", "volume"]].rename(columns={"close": "qqq_close", "volume": "qqq_volume"})
    features["qqq_ret_1d"] = features["qqq_close"].pct_change()
    features["qqq_ret_20d"] = features["qqq_close"].pct_change(20)
    features["qqq_ma_50"] = features["qqq_close"].rolling(50).mean()
    features["qqq_ma_200"] = features["qqq_close"].rolling(200).mean()
    features["qqq_dist_ma_50"] = features["qqq_close"] / features["qqq_ma_50"] - 1
    features["qqq_dist_ma_200"] = features["qqq_close"] / features["qqq_ma_200"] - 1
    features["qqq_volume_ratio_20"] = features["qqq_volume"] / features["qqq_volume"].rolling(20).mean()
    features["qqq_realized_vol_20"] = features["qqq_ret_1d"].rolling(20).std() * np.sqrt(252)
    features["qqq_drawdown_60"] = features["qqq_close"] / features["qqq_close"].rolling(60).max() - 1
    features["future_return_1d"] = features["qqq_close"].shift(-1) / features["qqq_close"] - 1
    features["future_return_5d"] = features["qqq_close"].shift(-5) / features["qqq_close"] - 1
    features["future_return_20d"] = features["qqq_close"].shift(-20) / features["qqq_close"] - 1
    features["future_max_drawdown_1d"] = _future_drawdown(features["qqq_close"], 1)
    features["future_max_drawdown_5d"] = _future_drawdown(features["qqq_close"], 5)
    features["future_max_drawdown_20d"] = _future_drawdown(features["qqq_close"], 20)

    features = features.join(_market_changes(vix, "vix", 5), how="left")
    features = features.join(_market_changes(hyg, "hyg", 5), how="left")
    features["technical_trend_score"] = _technical_score(features)
    features["market_pressure_score"] = _pressure_score(features)
    features["regime_bucket"] = _regime_bucket(features)
    required = ["qqq_dist_ma_200", "future_return_20d", "future_max_drawdown_20d"]
    candidates = features.dropna(subset=required).reset_index().rename(columns={"date": "prediction_date", "index": "prediction_date"})
    sampled = _stratified_sample(candidates, target_samples=target_samples, seed=seed)
    if sampled.empty:
        return _empty_samples()

    sampled["sample_id"] = [f"HIST-{stamp:%Y%m%d}" for stamp in pd.to_datetime(sampled["prediction_date"])]
    sampled["sample_seed"] = seed
    sampled["model_scope"] = "市場技術與壓力基準"
    sampled["kg_coverage_state"] = "需要歷史多來源事件回補"
    sampled["fundamental_coverage_state"] = "需要 point-in-time 基本面回補"
    sampled["independent_news_source_count"] = 0
    sampled["eligible_for_full_kg_backtest"] = False
    sampled["created_at_utc"] = pd.Timestamp.now(tz="UTC")
    columns = [
        "sample_id", "prediction_date", "regime_bucket", "sample_seed", "model_scope",
        "qqq_close", "qqq_ret_1d", "qqq_ret_20d", "qqq_dist_ma_50", "qqq_dist_ma_200", "qqq_volume_ratio_20", "qqq_realized_vol_20", "qqq_drawdown_60",
        "vix_close", "vix_change_5d", "hyg_close", "hyg_change_5d", "technical_trend_score", "market_pressure_score",
        "future_return_1d", "future_return_5d", "future_return_20d", "future_max_drawdown_1d", "future_max_drawdown_5d", "future_max_drawdown_20d",
        "kg_coverage_state", "fundamental_coverage_state", "independent_news_source_count", "eligible_for_full_kg_backtest", "created_at_utc",
    ]
    return sampled[[column for column in columns if column in sampled]].sort_values("prediction_date").reset_index(drop=True)


def load_stratified_market_samples(path: Path | None = None) -> pd.DataFrame:
    path = path or HISTORICAL_BACKTEST_CACHE
    if not path.exists():
        return _empty_samples()
    try:
        return pd.read_parquet(path)
    except Exception:
        return _empty_samples()


def historical_backtest_summary(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame()
    return (
        samples.groupby("regime_bucket", dropna=False)
        .agg(
            樣本數=("sample_id", "size"),
            平均1日報酬=("future_return_1d", "mean"),
            平均5日報酬=("future_return_5d", "mean"),
            平均20日報酬=("future_return_20d", "mean"),
            平均20日回撤=("future_max_drawdown_20d", "mean"),
        )
        .reset_index()
        .sort_values("樣本數", ascending=False)
    )


def _history(prices: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame(columns=["date", "close", "volume"])
    columns = [column for column in ["date", "close", "volume"] if column in prices]
    frame = prices[prices["symbol"].astype(str).str.upper().eq(symbol.upper())][columns].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    if "volume" not in frame:
        frame["volume"] = np.nan
    return frame.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _market_changes(history: pd.DataFrame, prefix: str, days: int) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=[f"{prefix}_close", f"{prefix}_change_{days}d"])
    result = history.set_index("date")[["close"]].rename(columns={"close": f"{prefix}_close"})
    result[f"{prefix}_change_{days}d"] = result[f"{prefix}_close"].pct_change(days)
    return result


def _future_drawdown(close: pd.Series, horizon: int) -> pd.Series:
    values = close.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for index in range(len(values) - horizon):
        future = values[index + 1 : index + horizon + 1]
        out[index] = future.min() / values[index] - 1
    return pd.Series(out, index=close.index)


def _technical_score(features: pd.DataFrame) -> pd.Series:
    parts = pd.concat(
        [
            np.sign(features["qqq_dist_ma_50"]),
            np.sign(features["qqq_dist_ma_200"]),
            np.sign(features["qqq_ret_20d"]),
        ],
        axis=1,
    )
    return parts.mean(axis=1).clip(-1, 1)


def _pressure_score(features: pd.DataFrame) -> pd.Series:
    parts = pd.concat([-np.sign(features.get("vix_change_5d")), np.sign(features.get("hyg_change_5d"))], axis=1)
    return parts.mean(axis=1).fillna(0).clip(-1, 1)


def _regime_bucket(features: pd.DataFrame) -> pd.Series:
    labels = pd.Series("正常／溫和上漲", index=features.index, dtype="object")
    labels.loc[(features["qqq_ret_20d"] < -0.05) | (features["qqq_drawdown_60"] < -0.08)] = "明顯下跌趨勢"
    labels.loc[(features["qqq_realized_vol_20"] > features["qqq_realized_vol_20"].rolling(252, min_periods=60).quantile(0.80)) | (features["vix_change_5d"] > 0.15)] = "高波動／壓力"
    rebound = (features["qqq_ret_20d"] > 0.05) & (features["qqq_drawdown_60"] < -0.05)
    labels.loc[rebound] = "強勢反彈／轉折"
    flat = (features["qqq_ret_20d"].abs() < 0.02) & (features["qqq_realized_vol_20"] < features["qqq_realized_vol_20"].rolling(252, min_periods=60).median())
    labels.loc[flat] = "橫盤／低波動"
    return labels


def _stratified_sample(candidates: pd.DataFrame, target_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = []
    selected_ids: set[int] = set()
    for label, share in REGIME_TARGETS.items():
        pool = candidates[candidates["regime_bucket"] == label]
        count = min(len(pool), max(1, round(target_samples * share)))
        if count:
            picked = pool.iloc[rng.choice(len(pool), size=count, replace=False)]
            groups.append(picked)
            selected_ids.update(picked.index.tolist())
    result = pd.concat(groups, ignore_index=False) if groups else pd.DataFrame(columns=candidates.columns)
    remaining = max(0, target_samples - len(result))
    if remaining:
        pool = candidates.loc[~candidates.index.isin(selected_ids)]
        count = min(len(pool), remaining)
        if count:
            result = pd.concat([result, pool.iloc[rng.choice(len(pool), size=count, replace=False)]], ignore_index=False)
    return result.reset_index(drop=True)


def _empty_samples() -> pd.DataFrame:
    return pd.DataFrame(columns=["sample_id", "prediction_date", "regime_bucket"])
