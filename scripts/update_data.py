from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import NEWS_QUERIES, default_start_date
from src.data import cache_path, refresh_market_data
from src.discovery import build_discovery_candidates, fetch_discovery_news
from src.indicators import add_price_indicators, detect_anomalies, latest_snapshot, regime_summary, today_conclusion
from src.news import fetch_international_news, fetch_news_batch
from src.predictions import build_market_prediction, update_prediction_log


def main() -> None:
    prices, macro = refresh_market_data(start=default_start_date())
    news = fetch_news_batch(symbols=list(NEWS_QUERIES), days=10, limit_per_symbol=8)
    if not news.empty:
        news.to_parquet(cache_path("news.parquet"), index=False)
    international_news = fetch_international_news(days=7, limit_per_topic=8)
    if not international_news.empty:
        international_news.to_parquet(cache_path("international_news.parquet"), index=False)
    discovery_news = fetch_discovery_news(days=7, topics_per_day=5, limit_per_topic=7)
    discovery_mentions, discovery_candidates = build_discovery_candidates(discovery_news, top_n=12)
    if not discovery_news.empty:
        discovery_news.to_parquet(cache_path("discovery_news.parquet"), index=False)
    if not discovery_mentions.empty:
        discovery_mentions.to_parquet(cache_path("discovery_mentions.parquet"), index=False)
    if not discovery_candidates.empty:
        discovery_candidates.to_parquet(cache_path("discovery_candidates.parquet"), index=False)

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
        f"predictions={len(prediction_log)}"
    )


if __name__ == "__main__":
    main()
