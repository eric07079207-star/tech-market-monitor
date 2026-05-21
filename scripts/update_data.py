from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import NEWS_QUERIES, default_start_date
from src.data import cache_path, refresh_market_data
from src.discovery import build_discovery_candidates, fetch_discovery_news, update_discovery_history, update_discovery_performance
from src.governance import annotate_governance, governance_summary
from src.kg import build_knowledge_graph, save_knowledge_graph
from src.indicators import add_price_indicators, detect_anomalies, latest_snapshot, regime_summary, today_conclusion
from src.lstm import build_lstm_feature_table, build_lstm_status_from_artifacts, save_lstm_feature_table, save_lstm_status
from src.news import DEFAULT_TSLA_KEYWORDS, fetch_international_news, fetch_news_batch, fetch_symbol_keyword_news
from src.predictions import build_market_prediction, update_prediction_log


def _stamp_fetch_time(data, fetched_at_utc: str):
    if data is not None and not data.empty:
        data = data.copy()
        data["fetched_at_utc"] = fetched_at_utc
    return data


def _write_parquet(data, filename: str) -> None:
    path = cache_path(filename)
    frame = data.copy() if data is not None else pd.DataFrame()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _load_keywords(path: Path | None = None) -> list[str]:
    path = path or cache_path("news_keywords.txt")
    if not path.exists():
        return DEFAULT_TSLA_KEYWORDS.copy()
    keywords = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [term for term in keywords if term]


def _backfill_prediction_log(indicators: pd.DataFrame, macro: pd.DataFrame, prediction_log: pd.DataFrame) -> pd.DataFrame:
    if indicators.empty or prediction_log.empty or "prediction_date" not in prediction_log:
        return prediction_log
    existing_dates = pd.to_datetime(prediction_log["prediction_date"], errors="coerce").dropna()
    if existing_dates.empty:
        return prediction_log

    qqq_dates = (
        indicators[indicators["symbol"] == "QQQ"]["date"]
        .pipe(pd.to_datetime, errors="coerce")
        .dropna()
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
    )
    if qqq_dates.empty:
        return prediction_log
    start_date = existing_dates.min().normalize()
    target_dates = qqq_dates[(qqq_dates >= start_date) & (qqq_dates <= qqq_dates.max())].tail(10)

    log = prediction_log
    for target_date in target_dates:
        indicator_slice = indicators[pd.to_datetime(indicators["date"], errors="coerce").dt.normalize() <= target_date].copy()
        macro_slice = macro.copy()
        if not macro_slice.empty and "date" in macro_slice:
            macro_slice = macro_slice[pd.to_datetime(macro_slice["date"], errors="coerce").dt.normalize() <= target_date]
        snapshot_slice = latest_snapshot(indicator_slice)
        anomalies_slice = detect_anomalies(snapshot_slice)
        regime_slice = regime_summary(indicator_slice, macro_slice)
        conclusion_slice = today_conclusion(regime_slice, snapshot_slice, anomalies_slice)
        prediction_slice = build_market_prediction(regime_slice, conclusion_slice, snapshot_slice)
        log = update_prediction_log(indicator_slice, prediction_slice)
    return log


def main() -> None:
    fetched_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    keywords = _load_keywords()
    prices, macro = refresh_market_data(start=default_start_date())
    news = _stamp_fetch_time(fetch_news_batch(symbols=list(NEWS_QUERIES), days=10, limit_per_symbol=8), fetched_at_utc)
    news = annotate_governance(news, "watchlist_news")
    _write_parquet(news, "news.parquet")
    international_news = _stamp_fetch_time(fetch_international_news(days=7, limit_per_topic=8), fetched_at_utc)
    international_news = annotate_governance(international_news, "international_news")
    _write_parquet(international_news, "international_news.parquet")
    discovery_news = _stamp_fetch_time(fetch_discovery_news(days=7, topics_per_day=5, limit_per_topic=7), fetched_at_utc)
    discovery_news = annotate_governance(discovery_news, "discovery_news")
    tsla_keyword_news = _stamp_fetch_time(
        fetch_symbol_keyword_news("TSLA", keywords or DEFAULT_TSLA_KEYWORDS, base_query="Tesla OR TSLA", days=7, limit_per_keyword=3),
        fetched_at_utc,
    )
    tsla_keyword_news = annotate_governance(tsla_keyword_news, "tsla_keyword_news")
    discovery_mentions, discovery_candidates = build_discovery_candidates(discovery_news, top_n=15)
    discovery_mentions = _stamp_fetch_time(discovery_mentions, fetched_at_utc)
    discovery_candidates = _stamp_fetch_time(discovery_candidates, fetched_at_utc)
    discovery_history = update_discovery_history(discovery_candidates)
    discovery_performance = update_discovery_performance(discovery_history)
    _write_parquet(discovery_news, "discovery_news.parquet")
    _write_parquet(tsla_keyword_news, "tsla_keyword_news.parquet")
    _write_parquet(discovery_mentions, "discovery_mentions.parquet")
    _write_parquet(discovery_candidates, "discovery_candidates.parquet")
    _write_parquet(
        governance_summary(
            {
                "watchlist_news": news,
                "international_news": international_news,
                "discovery_news": discovery_news,
                "tsla_keyword_news": tsla_keyword_news,
            }
        ),
        "governance_summary.parquet",
    )

    indicators = add_price_indicators(prices)
    snapshot = latest_snapshot(indicators)
    anomalies = detect_anomalies(snapshot)
    regime = regime_summary(indicators, macro)
    conclusion = today_conclusion(regime, snapshot, anomalies)
    kg = build_knowledge_graph(news, international_news, prices, macro, regime_context=regime, run_date=fetched_at_utc[:10])
    save_knowledge_graph(kg)
    prediction = build_market_prediction(regime, conclusion, snapshot)
    prediction_log = update_prediction_log(indicators, prediction)
    prediction_log = _backfill_prediction_log(indicators, macro, prediction_log)
    lstm_features = build_lstm_feature_table(prices=prices)
    if not lstm_features.empty:
        save_lstm_feature_table(lstm_features)
    lstm_status = save_lstm_status(build_lstm_status_from_artifacts(features=lstm_features))
    print(
        f"updated prices={len(prices)} macro={len(macro)} news={len(news)} "
        f"international_news={len(international_news)} discovery_candidates={len(discovery_candidates)} "
        f"discovery_history={len(discovery_history)} discovery_performance={len(discovery_performance)} "
        f"kg_facts={len(kg.facts)} kg_narratives={len(kg.narratives)} kg_reactions={len(kg.reactions)} "
        f"lstm_features={len(lstm_features)} lstm_status={lstm_status.get('status', 'n/a')} "
        f"predictions={len(prediction_log)}"
    )


if __name__ == "__main__":
    main()
