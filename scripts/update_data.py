from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import NEWS_QUERIES, default_start_date
from src.data import fetch_macro_series, fetch_price_history, load_cached_market_data
from src.discovery import (
    build_discovery_candidates,
    fetch_discovery_news,
    load_discovery_history,
    update_discovery_history,
    update_discovery_performance,
)
from src.governance import annotate_governance, governance_summary
from src.indicators import add_price_indicators, detect_anomalies, latest_snapshot, regime_summary, today_conclusion
from src.kg import (
    FACT_CACHE,
    LINK_CACHE,
    METADATA_CACHE as KG_METADATA_CACHE,
    NARRATIVE_CACHE,
    REACTION_CACHE,
    build_knowledge_graph,
)
from src.lstm import (
    LSTM_FEATURE_CACHE,
    LSTM_STATUS_CACHE,
    build_lstm_feature_table,
    build_lstm_status_from_artifacts,
    save_lstm_status,
)
from src.news import DEFAULT_TSLA_KEYWORDS, fetch_international_news, fetch_news_batch, fetch_symbol_keyword_news
from src.predictions import build_market_prediction, load_prediction_log, update_prediction_log
from src.sentiment import build_market_event_windows, build_sentiment_features
from src.update_pipeline import (
    FrameValidation,
    append_update_logs,
    latest_value,
    load_cached_frame,
    now_utc,
    safe_write_frame,
    safe_write_json,
)


def _stamp_fetch_time(data: pd.DataFrame | None, fetched_at_utc: str) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame() if data is None else data
    stamped = data.copy()
    stamped["fetched_at_utc"] = fetched_at_utc
    return stamped


def _load_keywords(path: Path | None = None) -> list[str]:
    from src.data import cache_path

    path = path or cache_path("news_keywords.txt")
    if not path.exists():
        return DEFAULT_TSLA_KEYWORDS.copy()
    keywords = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [term for term in keywords if term]


def _current_market() -> tuple[pd.DataFrame, pd.DataFrame]:
    prices, macro = load_cached_market_data(start=default_start_date(), force_refresh=False)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    if not macro.empty:
        macro["date"] = pd.to_datetime(macro["date"], errors="coerce")
    return prices, macro


def _normalize_dates(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    data = frame.copy()
    for column in columns:
        if column in data.columns:
            data[column] = pd.to_datetime(data[column], errors="coerce", utc="utc" in column.lower())
    return data


def _latest_dates(prices: pd.DataFrame, macro: pd.DataFrame) -> tuple[str, str]:
    price_latest = latest_value(prices, ["date"])
    macro_latest = latest_value(macro, ["date"])
    return price_latest, macro_latest


def _record(module_records: list[dict], name: str, category: str, status: str, message: str, critical: bool, rows: int = 0, latest: str = "") -> None:
    module_records.append(
        {
            "run_id": RUN_ID,
            "module_name": name,
            "category": category,
            "status": status,
            "critical": bool(critical),
            "rows": int(rows),
            "latest_value": latest or "",
            "message": message,
            "finished_at_utc": now_utc(),
        }
    )


def _write_frame_module(
    module_records: list[dict],
    *,
    module_name: str,
    category: str,
    filename: str,
    frame: pd.DataFrame,
    validation: FrameValidation,
    critical: bool,
    latest_columns: list[str],
) -> pd.DataFrame:
    previous = load_cached_frame(filename)
    try:
        ok, issues, current = safe_write_frame(filename, frame, validation)
    except Exception as exc:
        status = "fallback" if not previous.empty else "failed"
        _record(module_records, module_name, category, status, f"exception: {exc}", critical, len(previous), latest_value(previous, latest_columns))
        return previous

    if ok:
        _record(module_records, module_name, category, "success", f"updated {filename}", critical, len(current), latest_value(current, latest_columns))
        return current

    status = "fallback" if not previous.empty else "failed"
    _record(module_records, module_name, category, status, "; ".join(issues), critical, len(previous), latest_value(previous, latest_columns))
    return previous


def _write_json_module(module_records: list[dict], *, module_name: str, category: str, filename: str, payload: dict, critical: bool, rows: int = 1, latest: str = "") -> None:
    try:
        safe_write_json(filename, payload)
        _record(module_records, module_name, category, "success", f"updated {filename}", critical, rows, latest)
    except Exception as exc:
        _record(module_records, module_name, category, "failed", f"exception: {exc}", critical, 0, latest)


def _build_metadata(prices: pd.DataFrame, macro: pd.DataFrame, module_records: list[dict], started_at_utc: str, fetched_at_utc: str) -> dict:
    success_count = sum(row["status"] == "success" for row in module_records)
    fallback_count = sum(row["status"] == "fallback" for row in module_records)
    failed = [row["module_name"] for row in module_records if row["status"] == "failed"]
    status = "success"
    if failed:
        status = "failed"
    elif fallback_count:
        status = "partial"
    return {
        "updated_at_utc": fetched_at_utc,
        "start": str(default_start_date()),
        "price_rows": int(len(prices)),
        "macro_rows": int(len(macro)),
        "pipeline_run_id": RUN_ID,
        "pipeline_started_at_utc": started_at_utc,
        "pipeline_finished_at_utc": fetched_at_utc,
        "pipeline_status": status,
        "pipeline_success_count": success_count,
        "pipeline_fallback_count": fallback_count,
        "pipeline_failure_count": len(failed),
        "pipeline_failed_modules": failed,
        "pipeline_module_summary": {row["module_name"]: row["status"] for row in module_records},
    }


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
        log = update_prediction_log(indicator_slice, prediction_slice, existing_log=log, save=False)
    return log


RUN_ID = now_utc().replace(":", "").replace("-", "")


def main() -> None:
    started_at_utc = now_utc()
    fetched_at_utc = started_at_utc
    module_records: list[dict] = []
    keywords = _load_keywords()

    # Market data
    try:
        fresh_prices = fetch_price_history(start=default_start_date())
        previous_macro = load_cached_frame("macro.parquet")
        fresh_macro = fetch_macro_series(start=default_start_date(), previous=previous_macro)
    except Exception as exc:
        fresh_prices = pd.DataFrame()
        fresh_macro = pd.DataFrame()
        _record(module_records, "market_data", "主資料", "failed", f"exception: {exc}", True)
    else:
        _write_frame_module(
            module_records,
            module_name="market_prices",
            category="主資料",
            filename="prices.parquet",
            frame=fresh_prices,
            validation=FrameValidation(required_columns=("date", "symbol", "close"), latest_column="date", min_rows=5000, min_fraction_of_previous=0.2),
            critical=True,
            latest_columns=["date"],
        )
        _write_frame_module(
            module_records,
            module_name="market_macro",
            category="主資料",
            filename="macro.parquet",
            frame=fresh_macro,
            validation=FrameValidation(required_columns=("date", "series", "value"), latest_column="date", min_rows=10, min_fraction_of_previous=0.2),
            critical=True,
            latest_columns=["date"],
        )

    prices, macro = _current_market()

    # News modules
    try:
        news = _stamp_fetch_time(fetch_news_batch(symbols=list(NEWS_QUERIES), days=10, limit_per_symbol=8), fetched_at_utc)
        news = annotate_governance(news, "watchlist_news")
    except Exception as exc:
        _record(module_records, "watchlist_news", "新聞", "fallback" if not load_cached_frame("news.parquet").empty else "failed", f"exception: {exc}", True)
        news = load_cached_frame("news.parquet")
    else:
        news = _write_frame_module(
            module_records,
            module_name="watchlist_news",
            category="新聞",
            filename="news.parquet",
            frame=news,
            validation=FrameValidation(required_columns=("symbol", "title", "source", "published"), latest_column="published", min_rows=10, min_fraction_of_previous=0.15),
            critical=True,
            latest_columns=["published", "fetched_at_utc"],
        )

    try:
        international_news = _stamp_fetch_time(fetch_international_news(days=7, limit_per_topic=8), fetched_at_utc)
        international_news = annotate_governance(international_news, "international_news")
    except Exception as exc:
        _record(module_records, "international_news", "新聞", "fallback" if not load_cached_frame("international_news.parquet").empty else "failed", f"exception: {exc}", False)
        international_news = load_cached_frame("international_news.parquet")
    else:
        international_news = _write_frame_module(
            module_records,
            module_name="international_news",
            category="新聞",
            filename="international_news.parquet",
            frame=international_news,
            validation=FrameValidation(required_columns=("title", "source", "published"), latest_column="published", min_rows=3, min_fraction_of_previous=0.1),
            critical=False,
            latest_columns=["published", "fetched_at_utc"],
        )

    try:
        discovery_news = _stamp_fetch_time(fetch_discovery_news(days=7, topics_per_day=5, limit_per_topic=7), fetched_at_utc)
        discovery_news = annotate_governance(discovery_news, "discovery_news")
    except Exception as exc:
        _record(module_records, "discovery_news", "探索", "fallback" if not load_cached_frame("discovery_news.parquet").empty else "failed", f"exception: {exc}", False)
        discovery_news = load_cached_frame("discovery_news.parquet")
    else:
        discovery_news = _write_frame_module(
            module_records,
            module_name="discovery_news",
            category="探索",
            filename="discovery_news.parquet",
            frame=discovery_news,
            validation=FrameValidation(required_columns=("topic", "title", "published"), latest_column="published", min_rows=5, min_fraction_of_previous=0.1),
            critical=False,
            latest_columns=["published", "fetched_at_utc"],
        )

    try:
        tsla_keyword_news = _stamp_fetch_time(
            fetch_symbol_keyword_news("TSLA", keywords or DEFAULT_TSLA_KEYWORDS, base_query="Tesla OR TSLA", days=7, limit_per_keyword=3),
            fetched_at_utc,
        )
        tsla_keyword_news = annotate_governance(tsla_keyword_news, "tsla_keyword_news")
    except Exception as exc:
        _record(module_records, "tsla_focus_news", "專題", "fallback" if not load_cached_frame("tsla_keyword_news.parquet").empty else "failed", f"exception: {exc}", False)
        tsla_keyword_news = load_cached_frame("tsla_keyword_news.parquet")
    else:
        tsla_keyword_news = _write_frame_module(
            module_records,
            module_name="tsla_focus_news",
            category="專題",
            filename="tsla_keyword_news.parquet",
            frame=tsla_keyword_news,
            validation=FrameValidation(required_columns=("symbol", "title", "published"), latest_column="published", min_rows=3, min_fraction_of_previous=0.1),
            critical=False,
            latest_columns=["published", "fetched_at_utc"],
        )

    # Discovery derived data
    try:
        discovery_mentions, discovery_candidates = build_discovery_candidates(discovery_news, top_n=15)
        discovery_mentions = _stamp_fetch_time(discovery_mentions, fetched_at_utc)
        discovery_candidates = _stamp_fetch_time(discovery_candidates, fetched_at_utc)
        history_before = load_discovery_history()
        discovery_history = update_discovery_history(discovery_candidates, history=history_before, save=False)
        discovery_performance = update_discovery_performance(discovery_history, save=False)
    except Exception as exc:
        _record(module_records, "discovery_candidates", "探索", "fallback", f"exception: {exc}", False)
        discovery_mentions = load_cached_frame("discovery_mentions.parquet")
        discovery_candidates = load_cached_frame("discovery_candidates.parquet")
        discovery_history = load_cached_frame("discovery_history.parquet")
        discovery_performance = load_cached_frame("discovery_performance.parquet")
    else:
        discovery_mentions = _write_frame_module(
            module_records,
            module_name="discovery_mentions",
            category="探索",
            filename="discovery_mentions.parquet",
            frame=discovery_mentions,
            validation=FrameValidation(required_columns=("ticker", "title"), allow_empty=True),
            critical=False,
            latest_columns=["published", "fetched_at_utc"],
        )
        discovery_candidates = _write_frame_module(
            module_records,
            module_name="discovery_candidates",
            category="探索",
            filename="discovery_candidates.parquet",
            frame=discovery_candidates,
            validation=FrameValidation(required_columns=("ticker", "candidate_score"), allow_empty=False, min_fraction_of_previous=0.1),
            critical=False,
            latest_columns=["fetched_at_utc"],
        )
        discovery_history = _write_frame_module(
            module_records,
            module_name="discovery_history",
            category="探索",
            filename="discovery_history.parquet",
            frame=discovery_history,
            validation=FrameValidation(required_columns=("date", "ticker"), allow_empty=True),
            critical=False,
            latest_columns=["date"],
        )
        discovery_performance = _write_frame_module(
            module_records,
            module_name="discovery_performance",
            category="探索",
            filename="discovery_performance.parquet",
            frame=discovery_performance,
            validation=FrameValidation(required_columns=("date", "ticker", "horizon"), allow_empty=True),
            critical=False,
            latest_columns=["validated_at"],
        )

    governance = governance_summary(
        {
            "watchlist_news": news,
            "international_news": international_news,
            "discovery_news": discovery_news,
            "tsla_keyword_news": tsla_keyword_news,
        }
    )
    _write_frame_module(
        module_records,
        module_name="governance_summary",
        category="治理",
        filename="governance_summary.parquet",
        frame=governance,
        validation=FrameValidation(required_columns=("dataset", "official"), allow_empty=False),
        critical=False,
        latest_columns=[],
    )

    indicators = add_price_indicators(prices)
    snapshot = latest_snapshot(indicators)
    anomalies = detect_anomalies(snapshot)
    regime = regime_summary(indicators, macro)
    conclusion = today_conclusion(regime, snapshot, anomalies)

    # Sentiment layer
    try:
        sentiment = build_sentiment_features(prices, macro, news, international_news)
        market_event_windows = build_market_event_windows(prices, sentiment)
    except Exception as exc:
        _record(module_records, "sentiment_layer", "情緒", "fallback" if not load_cached_frame("sentiment.parquet").empty else "failed", f"exception: {exc}", False)
        sentiment = load_cached_frame("sentiment.parquet")
        market_event_windows = load_cached_frame("market_event_windows.parquet")
    else:
        sentiment = _write_frame_module(
            module_records,
            module_name="sentiment_layer",
            category="情緒",
            filename="sentiment.parquet",
            frame=sentiment,
            validation=FrameValidation(required_columns=("date", "market_mood_score", "market_mood_label"), allow_empty=False, min_rows=252),
            critical=False,
            latest_columns=["date"],
        )
        market_event_windows = _write_frame_module(
            module_records,
            module_name="market_event_windows",
            category="情緒",
            filename="market_event_windows.parquet",
            frame=market_event_windows,
            validation=FrameValidation(required_columns=("symbol", "start_date", "end_date", "window_return"), allow_empty=True),
            critical=False,
            latest_columns=["end_date"],
        )

    # Knowledge graph
    try:
        kg = build_knowledge_graph(news, international_news, prices, macro, regime_context=regime, run_date=fetched_at_utc[:10])
    except Exception as exc:
        _record(module_records, "knowledge_graph", "知識圖譜", "fallback", f"exception: {exc}", False)
    else:
        _write_frame_module(
            module_records,
            module_name="kg_fact_events",
            category="知識圖譜",
            filename=str(FACT_CACHE.relative_to(FACT_CACHE.parents[1])),
            frame=kg.facts,
            validation=FrameValidation(required_columns=("event_id", "timestamp_utc", "event_type_primary"), allow_empty=True),
            critical=False,
            latest_columns=["timestamp_utc"],
        )
        _write_frame_module(
            module_records,
            module_name="kg_narratives",
            category="知識圖譜",
            filename=str(NARRATIVE_CACHE.relative_to(NARRATIVE_CACHE.parents[1])),
            frame=kg.narratives,
            validation=FrameValidation(required_columns=("event_id", "timestamp_utc", "dominant_theme"), allow_empty=True),
            critical=False,
            latest_columns=["timestamp_utc"],
        )
        _write_frame_module(
            module_records,
            module_name="kg_reactions",
            category="知識圖譜",
            filename=str(REACTION_CACHE.relative_to(REACTION_CACHE.parents[1])),
            frame=kg.reactions,
            validation=FrameValidation(required_columns=("event_id", "affected_ticker", "time_horizon"), allow_empty=True),
            critical=False,
            latest_columns=["validated_at_utc"],
        )
        _write_frame_module(
            module_records,
            module_name="kg_links",
            category="知識圖譜",
            filename=str(LINK_CACHE.relative_to(LINK_CACHE.parents[1])),
            frame=kg.links,
            validation=FrameValidation(required_columns=("source_event_id", "target_entity"), allow_empty=True),
            critical=False,
            latest_columns=["created_at_utc", "timestamp_utc"],
        )
        kg_metadata = {
            "updated_at_utc": fetched_at_utc,
            "fact_rows": int(len(kg.facts)),
            "narrative_rows": int(len(kg.narratives)),
            "reaction_rows": int(len(kg.reactions)),
            "link_rows": int(len(kg.links)),
        }
        temp_name = str(KG_METADATA_CACHE.relative_to(KG_METADATA_CACHE.parents[1]))
        _write_json_module(module_records, module_name="kg_metadata", category="知識圖譜", filename=temp_name, payload=kg_metadata, critical=False, latest=fetched_at_utc)

    # Prediction log
    try:
        prediction = build_market_prediction(regime, conclusion, snapshot)
        existing_log = load_prediction_log()
        prediction_log = update_prediction_log(indicators, prediction, existing_log=existing_log, save=False)
        prediction_log = _backfill_prediction_log(indicators, macro, prediction_log)
    except Exception as exc:
        _record(module_records, "prediction_log", "模型", "fallback" if not load_cached_frame("prediction_log.csv").empty else "failed", f"exception: {exc}", False)
    else:
        _write_frame_module(
            module_records,
            module_name="prediction_log",
            category="模型",
            filename="prediction_log.csv",
            frame=prediction_log,
            validation=FrameValidation(required_columns=("prediction_date", "target", "horizon"), allow_empty=True),
            critical=False,
            latest_columns=["prediction_date", "validated_at"],
        )

    # LSTM lightweight artifacts
    try:
        lstm_features = build_lstm_feature_table(prices=prices)
        if lstm_features.empty:
            lstm_status = build_lstm_status_from_artifacts(features=lstm_features)
        else:
            lstm_status = build_lstm_status_from_artifacts(features=lstm_features)
    except Exception as exc:
        _record(module_records, "lstm_features", "模型", "fallback" if not load_cached_frame("lstm/lstm_features.parquet").empty else "failed", f"exception: {exc}", False)
    else:
        if not lstm_features.empty:
            _write_frame_module(
                module_records,
                module_name="lstm_features",
                category="模型",
                filename=str(LSTM_FEATURE_CACHE.relative_to(LSTM_FEATURE_CACHE.parents[1])),
                frame=lstm_features,
                validation=FrameValidation(required_columns=("symbol", "date", "target_date", "label"), allow_empty=True),
                critical=False,
                latest_columns=["date", "created_at_utc"],
            )
        status_name = str(LSTM_STATUS_CACHE.relative_to(LSTM_STATUS_CACHE.parents[1]))
        try:
            save_lstm_status(lstm_status, LSTM_STATUS_CACHE)
            _record(module_records, "lstm_status", "模型", "success", "updated lstm status", False, int(lstm_status.get("feature_rows", 0)), lstm_status.get("updated_at_utc", ""))
        except Exception as exc:
            _record(module_records, "lstm_status", "模型", "failed", f"exception: {exc}", False)

    metadata = _build_metadata(prices, macro, module_records, started_at_utc, now_utc())
    safe_write_json("metadata.json", metadata)
    append_update_logs(
        {
            "run_id": RUN_ID,
            "started_at_utc": started_at_utc,
            "finished_at_utc": metadata["pipeline_finished_at_utc"],
            "status": metadata["pipeline_status"],
            "success_count": metadata["pipeline_success_count"],
            "fallback_count": metadata["pipeline_fallback_count"],
            "failure_count": metadata["pipeline_failure_count"],
            "failed_modules": "；".join(metadata["pipeline_failed_modules"]),
        },
        module_records,
    )

    print(
        f"update run {RUN_ID} status={metadata['pipeline_status']} "
        f"success={metadata['pipeline_success_count']} fallback={metadata['pipeline_fallback_count']} failed={metadata['pipeline_failure_count']}"
    )
    price_latest, macro_latest = _latest_dates(prices, macro)
    print(
        f"market price_rows={len(prices)} price_latest={price_latest} "
        f"macro_rows={len(macro)} macro_latest={macro_latest}"
    )
    print(
        f"news_rows={len(news)} international_rows={len(international_news)} "
        f"discovery_news_rows={len(discovery_news)} discovery_candidates_rows={len(discovery_candidates)} "
        f"tsla_focus_rows={len(tsla_keyword_news)}"
    )

    critical_failures = [row["module_name"] for row in module_records if row["critical"] and row["status"] == "failed"]
    if critical_failures:
        raise SystemExit(f"critical update modules failed: {', '.join(critical_failures)}")


if __name__ == "__main__":
    main()
