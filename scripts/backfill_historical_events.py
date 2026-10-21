from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.historical_backtest import HISTORICAL_BACKTEST_CACHE
from src.fundamentals import annotate_sample_fundamental_coverage, load_fundamental_observations
from src.historical_events import (
    annotate_historical_event_coverage,
    backfill_historical_event_evidence,
    load_historical_event_evidence,
    save_historical_event_artifacts,
)
from src.kg_backtest import (
    HISTORICAL_KG_BACKTEST_CACHE,
    KG_BACKTEST_READINESS_CACHE,
    build_historical_kg_backtest,
    build_historical_kg_quality_report,
    build_kg_backtest_readiness,
    save_historical_kg_quality_report,
)
from src.official_events import load_official_events, official_events_as_historical_evidence, refresh_official_events_if_due


def main() -> None:
    parser = argparse.ArgumentParser(description="分批回補 point-in-time 歷史事件並重建 KG 回測")
    parser.add_argument("--max-samples", type=int, default=12, help="本次最多處理的歷史樣本視窗")
    parser.add_argument("--pause-seconds", type=float, default=5.2, help="GDELT 請求間隔")
    args = parser.parse_args()

    if not HISTORICAL_BACKTEST_CACHE.exists():
        raise SystemExit("missing stratified historical market samples")
    samples = pd.read_parquet(HISTORICAL_BACKTEST_CACHE)
    samples = annotate_sample_fundamental_coverage(samples, load_fundamental_observations())
    evidence, status = backfill_historical_event_evidence(
        samples,
        load_historical_event_evidence(),
        max_samples=args.max_samples,
        pause_seconds=args.pause_seconds,
    )
    official_events = refresh_official_events_if_due()
    official_evidence = official_events_as_historical_evidence(samples, official_events)
    if not official_evidence.empty:
        evidence = pd.concat([evidence, official_evidence], ignore_index=True)
        evidence = evidence.drop_duplicates(["sample_id", "article_url"], keep="last")
        evidence = evidence.drop_duplicates(["sample_id", "title_hash", "source_domain"], keep="last")
        evidence = evidence.sort_values(["prediction_date", "published_at", "source_domain"]).reset_index(drop=True)
        status["article_rows"] = int(len(evidence))
        status["official_event_rows"] = int(evidence.get("provider", pd.Series(dtype=str)).eq("Official event layer").sum())
    annotated = annotate_historical_event_coverage(samples, evidence)
    readiness = build_kg_backtest_readiness(annotated)
    backtest = build_historical_kg_backtest(annotated, evidence)
    quality_report = build_historical_kg_quality_report(readiness, evidence)

    save_historical_event_artifacts(evidence, status)
    annotated.to_parquet(HISTORICAL_BACKTEST_CACHE, index=False)
    readiness.to_parquet(KG_BACKTEST_READINESS_CACHE, index=False)
    backtest.to_parquet(HISTORICAL_KG_BACKTEST_CACHE, index=False)
    save_historical_kg_quality_report(quality_report)

    ready = int(readiness.get("eligible_for_full_kg_backtest", pd.Series(dtype=bool)).sum())
    print(
        f"historical backfill attempted={status['attempted_this_run']} "
        f"added={status['articles_added_this_run']} evidence_rows={len(evidence)} "
        f"full_kg_ready={ready} backtest_rows={len(backtest)} failures={len(status['failures_this_run'])}"
    )


if __name__ == "__main__":
    main()
