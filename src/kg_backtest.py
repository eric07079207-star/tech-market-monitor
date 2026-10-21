"""Coverage-aware KG backtest readiness and outcome summaries.

This module deliberately separates *market samples* from samples that have
enough point-in-time event evidence for a full knowledge-graph backtest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import CACHE_DIR


KG_BACKTEST_READINESS_CACHE = CACHE_DIR / "kg" / "historical_backtest_readiness.parquet"
HISTORICAL_KG_BACKTEST_CACHE = CACHE_DIR / "backtest" / "historical_kg_backtest.parquet"
HISTORICAL_KG_QUALITY_CACHE = CACHE_DIR / "backtest" / "historical_kg_quality_report.json"
REGIME_ORDER = ["高波動／壓力", "明顯下跌趨勢", "強勢反彈／轉折", "橫盤／低波動", "正常／溫和上漲"]
OFFICIAL_DOMAIN_SUFFIXES = (
    "sec.gov", "federalreserve.gov", "bls.gov", "bea.gov", "whitehouse.gov", "commerce.gov",
    "apple.com", "microsoft.com", "nvidia.com", "amd.com", "tesla.com", "amazon.com", "abc.xyz", "meta.com",
)


def build_kg_backtest_readiness(samples: pd.DataFrame) -> pd.DataFrame:
    if samples is None or samples.empty:
        return _empty_readiness()

    data = samples.copy()
    source_count = pd.to_numeric(data.get("independent_news_source_count"), errors="coerce").fillna(0)
    fundamental_ok = data.get("fundamental_coverage_state", pd.Series("資料不足", index=data.index)).eq("完整（官方）")
    official_count = pd.to_numeric(data.get("official_event_count"), errors="coerce").fillna(0)
    event_ok = source_count.ge(2) | official_count.ge(1)
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
        "historical_event_types", "official_event_count", "fundamental_ticker_coverage",
        "fundamental_coverage_state", "fundamental_missing_reason", "event_evidence_ready", "fundamental_evidence_ready",
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


def build_historical_kg_quality_report(readiness: pd.DataFrame, evidence: pd.DataFrame) -> dict:
    """Build auditable coverage, source-quality and leakage diagnostics."""
    ready = readiness.copy() if readiness is not None else pd.DataFrame()
    news = evidence.copy() if evidence is not None else pd.DataFrame()
    eligible = ready.get("eligible_for_full_kg_backtest", pd.Series(False, index=ready.index)).fillna(False).astype(bool)

    regime_rows = []
    for regime in REGIME_ORDER:
        group = ready[ready.get("regime_bucket", pd.Series("", index=ready.index)).eq(regime)]
        qualified = int(group.get("eligible_for_full_kg_backtest", pd.Series(False, index=group.index)).fillna(False).sum())
        regime_rows.append(
            {
                "market_regime": regime,
                "candidate_samples": int(len(group)),
                "qualified_samples": qualified,
                "diagnostic_target": 30,
                "weight_target": 50,
                "diagnostic_completion_pct": round(min(qualified / 30, 1.0) * 100, 1),
                "weight_completion_pct": round(min(qualified / 50, 1.0) * 100, 1),
                "readiness_state": "可評估權重" if qualified >= 50 else "可開始因子診斷" if qualified >= 30 else "資料累積中",
            }
        )

    source_count = int(news.get("source_domain", pd.Series(dtype=str)).replace("", pd.NA).nunique(dropna=True))
    domains = news.get("source_domain", pd.Series("", index=news.index)).fillna("").astype(str).str.lower()
    official = domains.map(lambda value: any(value == suffix or value.endswith(f".{suffix}") for suffix in OFFICIAL_DOMAIN_SUFFIXES))
    official_event_evidence_count = int(news.get("provider", pd.Series("", index=news.index)).eq("Official event layer").sum())
    official_ratio = float(official.mean()) if len(news) else 0.0
    cluster = pd.DataFrame()
    if not news.empty and {"sample_id", "event_cluster_key", "source_domain", "title_hash"}.issubset(news.columns):
        cluster = news.groupby(["sample_id", "event_cluster_key"], dropna=False).agg(
            source_count=("source_domain", "nunique"), story_count=("title_hash", "nunique")
        ).reset_index()
    confirmed = int((cluster["source_count"].ge(2) & cluster["story_count"].ge(2)).sum()) if not cluster.empty else 0
    confirmation_ratio = confirmed / len(cluster) if len(cluster) else 0.0

    duplicate_rows = 0
    if not news.empty and {"sample_id", "title_hash", "source_domain"}.issubset(news.columns):
        duplicate_rows = int(news.duplicated(["sample_id", "title_hash", "source_domain"], keep="first").sum())
    future_leakage_rows = 0
    if not news.empty and {"published_at", "prediction_date"}.issubset(news.columns):
        published = pd.to_datetime(news["published_at"], errors="coerce", utc=True)
        cutoff = pd.to_datetime(news["prediction_date"], errors="coerce", utc=True) + pd.Timedelta(days=1)
        future_leakage_rows = int((published >= cutoff).fillna(False).sum())
    top_source = domains.value_counts().head(1)
    top_source_domain = str(top_source.index[0]) if not top_source.empty else ""
    top_source_ratio = float(top_source.iloc[0] / len(news)) if len(news) and not top_source.empty else 0.0

    event_ready = ready.get("event_evidence_ready", pd.Series(False, index=ready.index)).fillna(False).astype(bool)
    fundamental_ready = ready.get("fundamental_evidence_ready", pd.Series(False, index=ready.index)).fillna(False).astype(bool)
    reaction_cols = [column for column in ("future_return_1d", "future_return_5d", "future_return_20d") if column in ready]
    reaction_ready = ready[reaction_cols].notna().all(axis=1) if reaction_cols else pd.Series(False, index=ready.index)
    missing_reasons = [
        {"reason": "缺少跨來源新聞或官方事件", "sample_count": int((~event_ready).sum())},
        {"reason": "缺少後續市場反應", "sample_count": int((~reaction_ready).sum())},
    ]
    if "fundamental_missing_reason" in ready:
        reason_counts = ready.loc[~fundamental_ready, "fundamental_missing_reason"].fillna("未分類基本面缺口").replace("", "未分類基本面缺口").value_counts()
        missing_reasons.extend({"reason": f"基本面：{reason}", "sample_count": int(count)} for reason, count in reason_counts.items())
    else:
        missing_reasons.append({"reason": "缺少 point-in-time 基本面", "sample_count": int((~fundamental_ready).sum())})
    checks = [
        {"check": "未來資料洩漏", "status": "正常" if future_leakage_rows == 0 else "異常", "count": future_leakage_rows},
        {"check": "重複事件列", "status": "正常" if duplicate_rows == 0 else "注意", "count": duplicate_rows},
        {"check": "單一媒體偏重", "status": "正常" if top_source_ratio <= 0.35 else "注意", "count": round(top_source_ratio * 100, 1)},
    ]
    diagnostic_ready = all(row["qualified_samples"] >= 30 for row in regime_rows) and int(eligible.sum()) >= 150
    weight_ready = all(row["qualified_samples"] >= 50 for row in regime_rows) and int(eligible.sum()) >= 250
    return {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "candidate_samples": int(len(ready)),
        "qualified_samples": int(eligible.sum()),
        "regimes": regime_rows,
        "sources": {
            "unique_source_count": source_count,
            "official_article_ratio": round(official_ratio, 4),
            "official_event_evidence_count": official_event_evidence_count,
            "confirmed_cluster_count": confirmed,
            "cross_source_confirmation_ratio": round(confirmation_ratio, 4),
            "top_source_domain": top_source_domain,
            "top_source_ratio": round(top_source_ratio, 4),
        },
        "missing_reasons": missing_reasons,
        "quality_checks": checks,
        "thresholds": {
            "factor_diagnostic_ready": diagnostic_ready,
            "weight_evaluation_ready": weight_ready,
            "current_state": "可評估權重" if weight_ready else "可開始因子診斷" if diagnostic_ready else "資料累積中",
        },
    }


def save_historical_kg_quality_report(report: dict, path: Path | None = None) -> None:
    path = path or HISTORICAL_KG_QUALITY_CACHE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def load_historical_kg_quality_report(path: Path | None = None) -> dict:
    path = path or HISTORICAL_KG_QUALITY_CACHE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


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
