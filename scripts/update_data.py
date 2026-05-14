from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import NEWS_QUERIES, default_start_date
from src.data import cache_path, refresh_market_data
from src.news import fetch_news_batch


def main() -> None:
    prices, macro = refresh_market_data(start=default_start_date())
    news = fetch_news_batch(symbols=list(NEWS_QUERIES), days=10, limit_per_symbol=8)
    if not news.empty:
        news.to_parquet(cache_path("news.parquet"), index=False)
    print(f"updated prices={len(prices)} macro={len(macro)} news={len(news)}")


if __name__ == "__main__":
    main()
