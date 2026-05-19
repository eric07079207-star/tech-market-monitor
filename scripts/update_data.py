from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import NEWS_QUERIES, default_start_date
from src.data import cache_path, refresh_market_data
from src.discovery import build_discovery_candidates, fetch_discovery_news, update_discovery_history, update_discovery_performance
from src.kg import build_knowledge_graph, save_knowledge_graph
from src.indicators import add_price_indicators, detect_anomalies, latest_snapshot, regime_summary, today_conclusion
from src.news import fetch_international_news, fetch_news_batch
from src.predictions import build_market_prediction, update_prediction_log


def _stamp_fetch_time(data, fetched_at_utc: str):
    if data is not None and not data.empty:
        data = data.copy()
        data["fetched_at_utc"] = fetched_at_utc
    return data


def main() -> None:
    fetched_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    prices, macro = refresh_market_data(start=default_start_date())
    news = _stamp_fetch_time(fetch_news_batch(symbols=list(NEWS_QUERIES), days=10, limit_per_symbol=8), fetched_at_utc)
    if not news.empty:
        news.to_parquet(cache_path("news.parquet"), index=False)
    international_news = _stamp_fetch_time(fetch_international_news(days=7, limit_per_topic=8), fetched_at_utc)
    if not international_news.empty:
        international_news.to_parquet(cache_path("international_news.parquet"), index=False)
    discovery_news = _stamp_fetch_time(fetch_discovery_news(days=7, topics_per_day=5, limit_per_topic=7), fetched_at_utc)
    discovery_mentions, discovery_candidates = build_discovery_candidates(discovery_news, top_n=15)
    discovery_mentions = _stamp_fetch_time(discovery_mentions, fetched_at_utc)
    discovery_candidates = _stamp_fetch_time(discovery_candidates, fetched_at_utc)
    discovery_history = update_discovery_history(discovery_candidates)
    discovery_performance = update_discovery_performance(discovery_history)
    if not discovery_news.empty:
        discovery_news.to_parquet(cache_path("discovery_news.parquet"), index=False)
    if not discovery_mentions.empty:
        discovery_mentions.to_parquet(cache_path("discovery_mentions.parquet"), index=False)
    if not discovery_candidates.empty:
        discovery_candidates.to_parquet(cache_path("discovery_candidates.parquet"), index=False)

    kg = build_knowledge_graph(news, international_news, prices, macro, run_date=fetched_at_utc[:10])
    save_knowledge_graph(kg)

    indicators = add_price_indicators(prices)
    snapshot = latest_snapshot(indicators)
    anomalies = detect_anomalies(snapshot)
    regime = regime_summary(indicators, macro)
    conclusion = today_conclusion(regime, snapshot, anomalies)
    prediction = build_market_prediction(regime, conclusion, snapshot)
    prediction_log = update_prediction_log(indicators, prediction)
    print(
        f"updated prices={len(prices)} macro={len(macro)} news={len(news)} "
        f"international_news={len(international_news)} discovery_candidates={len(discovery_candidates)} "
        f"discovery_history={len(discovery_history)} discovery_performance={len(discovery_performance)} "
        f"kg_facts={len(kg.facts)} kg_narratives={len(kg.narratives)} kg_reactions={len(kg.reactions)} "
        f"predictions={len(prediction_log)}"
    )


if __name__ == "__main__":
    main()
