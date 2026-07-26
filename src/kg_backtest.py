"""Coverage-aware KG backtest readiness and outcome summaries.

This module deliberately separates *market samples* from samples that have
enough point-in-time event evidence for a full knowledge-graph backtest.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CACHE_DIR


KG_BACKTEST_READINESS_CACHE = CACHE_DIR / "kg" / "historical_backtest_readiness.parquet"


def build_kg_backtest_readiness(samples: pd.DataFrame) -> pd.DataFrame:
    if samples is None or samples.empty:
        return _empty_readiness()

    data = samples.copy()
    source_count = pd.to_numeric(data.get("independent_news_source_count"), errors="coerce").fillna(0)
    fundamental_ok = data.get("fundamental_coverage_state", pd.Series("資料不足", index=data.index)).eq("完整（官方）")
    event_ok = source_count.ge(2)
    data["event_evidence_ready"] = event_ok
    data["fundamental_evidence_ready"] = fundamental_ok
    data["eligible_for_full_kg_backtest"] = event_ok & fundamental_ok
    data["kg_backtest_state"] = "需要歷史多來源事件回補"
    data.loc[event_ok & ~fundamental_ok, "kg_backtest_state"] = "需要 point-in-time 基本面回補"
    data.loc[~event_ok & fundamental_ok, "kg_backtest_state"] = "需要歷史多來源事件回補"
    data.loc[data["eligible_for_full_kg_backtest"], "kg_backtest_state"] = "可納入完整 KG 回測"
    data["research_scope"] = data.get("model_scope", "市場技術與壓力基準")
    columns = [
        "sample_id", "prediction_date", "regime_bucket", "research_scope",
        "independent_news_source_count", "fundamental_ticker_coverage",
        "fundamental_coverage_state", "event_evidence_ready", "fundamental_evidence_ready",
        "eligible_for_full_kg_backtest", "kg_backtest_state",
        "future_return_1d", "future_return_5d", "future_return_20d", "future_max_drawdown_20d",
    ]
    return data[[column for column in columns if column in data]].sort_values("prediction_date").reset_index(drop=True)


def kg_backtest_readiness_summary(readiness: pd.DataFrame) -> pd.DataFrame:
    if readiness is None or readiness.empty:
        return pd.DataFrame()
    return (
        readiness.groupby("kg_backtest_state", dropna=False)
        .agg(
            樣本數=("sample_id", "size"),
            平均20日報酬=("future_return_20d", "mean"),
            平均20日回撤=("future_max_drawdown_20d", "mean"),
        )
        .reset_index()
        .sort_values("樣本數", ascending=False)
    )


def load_kg_backtest_readiness(path: Path | None = None) -> pd.DataFrame:
    path = path or KG_BACKTEST_READINESS_CACHE
    if not path.exists():
        return _empty_readiness()
    try:
        return pd.read_parquet(path)
    except Exception:
        return _empty_readiness()


def _empty_readiness() -> pd.DataFrame:
    return pd.DataFrame(columns=["sample_id", "prediction_date", "kg_backtest_state"])
