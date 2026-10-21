"""Coverage-aware KG backtest readiness and outcome summaries.

This module deliberately separates *market samples* from samples that have
enough point-in-time event evidence for a full knowledge-graph backtest.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CACHE_DIR


KG_BACKTEST_READINESS_CACHE = CACHE_DIR / "kg" / "historical_backtest_readiness.parquet"
HISTORICAL_KG_BACKTEST_CACHE = CACHE_DIR / "backtest" / "historical_kg_backtest.parquet"


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
    unavailable = data.get("kg_coverage_state", pd.Series("", index=data.index)).eq("歷史新聞來源不可得")
    data.loc[unavailable, "kg_backtest_state"] = "歷史新聞來源不可得"
    data.loc[event_ok & ~fundamental_ok, "kg_backtest_state"] = "需要 point-in-time 基本面回補"
    data.loc[~event_ok & fundamental_ok, "kg_backtest_state"] = "需要歷史多來源事件回補"
    data.loc[data["eligible_for_full_kg_backtest"], "kg_backtest_state"] = "可納入完整 KG 回測"
    data["research_scope"] = data.get("model_scope", "市場技術與壓力基準")
    columns = [
        "sample_id", "prediction_date", "regime_bucket", "research_scope",
        "independent_news_source_count", "historical_event_count", "confirmed_event_cluster_count",
        "historical_event_types", "fundamental_ticker_coverage",
        "fundamental_coverage_state", "event_evidence_ready", "fundamental_evidence_ready",
        "eligible_for_full_kg_backtest", "kg_backtest_state",
        "future_return_1d", "future_return_5d", "future_return_20d", "future_max_drawdown_20d",
    ]
    return data[[column for column in columns if column in data]].sort_values("prediction_date").reset_index(drop=True)


def build_historical_kg_backtest(samples: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    """Evaluate only samples with point-in-time multi-source event evidence."""
    if samples is None or samples.empty or evidence is None or evidence.empty:
        return _empty_backtest()
    readiness = build_kg_backtest_readiness(samples)
    eligible = readiness[readiness["eligible_for_full_kg_backtest"]].copy()
    if eligible.empty:
        return _empty_backtest()

    event = evidence[evidence["sample_id"].isin(eligible["sample_id"])].copy()
    direction_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    event["event_direction_score"] = event["impact_direction"].map(direction_map).fillna(0.0)
    aggregate = (
        event.groupby("sample_id")
        .agg(
            event_direction_score=("event_direction_score", "mean"),
            event_article_count=("title_hash", "nunique"),
            event_source_count=("source_domain", "nunique"),
            event_type_count=("event_type_primary", "nunique"),
            mean_source_reliability=("source_reliability_score", "mean"),
        )
        .reset_index()
    )
    result = eligible.merge(aggregate, on="sample_id", how="left")
    result["technical_vote"] = pd.to_numeric(samples.set_index("sample_id").reindex(result["sample_id"])["technical_trend_score"], errors="coerce").to_numpy()
    result["pressure_vote"] = pd.to_numeric(samples.set_index("sample_id").reindex(result["sample_id"])["market_pressure_score"], errors="coerce").to_numpy()
    result["kg_vote_score"] = result[["event_direction_score", "technical_vote", "pressure_vote"]].mean(axis=1)
    result["predicted_direction"] = "中性"
    result.loc[result["kg_vote_score"].gt(0.20), "predicted_direction"] = "偏多"
    result.loc[result["kg_vote_score"].lt(-0.20), "predicted_direction"] = "偏空"
    actual = pd.to_numeric(result["future_return_20d"], errors="coerce")
    result["actual_direction"] = "中性"
    result.loc[actual.gt(0.005), "actual_direction"] = "上漲"
    result.loc[actual.lt(-0.005), "actual_direction"] = "下跌"
    result["direction_correct"] = (
        (result["predicted_direction"].eq("偏多") & result["actual_direction"].eq("上漲"))
        | (result["predicted_direction"].eq("偏空") & result["actual_direction"].eq("下跌"))
        | (result["predicted_direction"].eq("中性") & result["actual_direction"].eq("中性"))
    )
    result["backtest_version"] = "historical-kg-v1"
    result["generated_at_utc"] = pd.Timestamp.now(tz="UTC")
    return result.sort_values("prediction_date").reset_index(drop=True)


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


def load_historical_kg_backtest(path: Path | None = None) -> pd.DataFrame:
    path = path or HISTORICAL_KG_BACKTEST_CACHE
    if not path.exists():
        return _empty_backtest()
    try:
        return pd.read_parquet(path)
    except Exception:
        return _empty_backtest()


def _empty_readiness() -> pd.DataFrame:
    return pd.DataFrame(columns=["sample_id", "prediction_date", "kg_backtest_state"])


def _empty_backtest() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sample_id", "prediction_date", "regime_bucket", "predicted_direction", "actual_direction",
            "future_return_20d", "future_max_drawdown_20d", "direction_correct", "backtest_version",
        ]
    )
