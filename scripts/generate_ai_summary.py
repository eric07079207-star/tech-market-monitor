from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai_summary import build_openai_summary, load_cached_ai_summary, openai_configuration_status, save_ai_summary
from src.data import cache_path
from src.indicators import add_price_indicators, detect_anomalies, latest_snapshot
from src.news import portfolio_news_impact
from src.portfolio import load_portfolio_config


def _read_parquet(name: str) -> pd.DataFrame:
    path = cache_path(name)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def main() -> None:
    config = openai_configuration_status()
    existing = load_cached_ai_summary()
    if not config.get("configured") and existing:
        print(
            "skip ai_summary generation "
            f"reason=no_openai_key existing_provider={existing.get('provider', 'n/a')} "
            f"existing_generated_at={existing.get('generated_at_utc', 'n/a')}"
        )
        return

    prices = _read_parquet("prices.parquet")
    news = _read_parquet("news.parquet")
    international_news = _read_parquet("international_news.parquet")
    discovery_candidates = _read_parquet("discovery_candidates.parquet")

    if prices.empty:
        raise RuntimeError("prices.parquet is missing or empty")

    indicators = add_price_indicators(prices)
    snapshot = latest_snapshot(indicators)
    anomalies = detect_anomalies(snapshot)

    portfolio_impact = pd.DataFrame()
    portfolio_config = load_portfolio_config({})
    if portfolio_config is not None and not portfolio_config.positions.empty and not news.empty:
        portfolio_impact = portfolio_news_impact(news, portfolio_config.positions, max_items=8)

    payload = build_openai_summary(
        snapshot=snapshot,
        anomalies=anomalies,
        news=news,
        international_news=international_news,
        discovery_candidates=discovery_candidates,
        portfolio_impact=portfolio_impact,
    )
    before = existing or load_cached_ai_summary()
    save_ai_summary(payload)
    after = load_cached_ai_summary()
    preserved_existing = before == after and before != payload
    print(
        "generated ai_summary "
        f"provider={payload.get('provider')} "
        f"model={payload.get('model')} "
        f"used_ai={payload.get('used_ai')} "
        f"status={payload.get('status')} "
        f"preserved_existing={preserved_existing}"
    )


if __name__ == "__main__":
    main()
