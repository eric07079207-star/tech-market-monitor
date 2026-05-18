from __future__ import annotations

import pandas as pd


def data_health_report(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    news: pd.DataFrame,
    international_news: pd.DataFrame,
    prediction_log: pd.DataFrame,
    metadata: dict,
    discovery_news: pd.DataFrame | None = None,
    discovery_candidates: pd.DataFrame | None = None,
    discovery_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = [
        _row("市場價格", len(prices), _max_date(prices, "date"), "價格與量能快取"),
        _row("總經資料", len(macro), _max_date(macro, "date"), "FRED 與市場壓力資料"),
        _row("標的新聞", len(news), _max_date(news, "published"), "watchlist 新聞"),
        _row("國際新聞", len(international_news), _max_date(international_news, "published"), "國際重大與隨機新聞"),
        _row("預測紀錄", len(prediction_log), _max_date(prediction_log, "prediction_date"), "5D/20D/60D 驗證資料"),
        _row("新聞探索", len(discovery_news) if discovery_news is not None else 0, _max_date(discovery_news, "published") if discovery_news is not None else "n/a", "隨機主題新聞"),
        _row("候選觀察股", len(discovery_candidates) if discovery_candidates is not None else 0, "n/a", "新聞探索量化候選"),
        _row("候選歷史紀錄", len(discovery_history) if discovery_history is not None else 0, _max_date(discovery_history, "date") if discovery_history is not None else "n/a", "每日 Top 15 候選追蹤"),
    ]
    report = pd.DataFrame(rows)
    report["快取更新 UTC"] = metadata.get("updated_at_utc", "尚未寫入")
    return report


def missing_price_symbols(prices: pd.DataFrame, symbols: list[str]) -> list[str]:
    if prices.empty:
        return symbols
    available = set(prices["symbol"].dropna().astype(str).unique())
    return [symbol for symbol in symbols if symbol not in available]


def _row(name: str, count: int, latest: object, note: str) -> dict:
    return {"資料項目": name, "筆數": int(count), "最新日期": str(latest), "說明": note}


def _max_date(data: pd.DataFrame | None, column: str) -> str:
    if data is None or data.empty or column not in data:
        return "n/a"
    value = pd.to_datetime(data[column], errors="coerce").max()
    if pd.isna(value):
        return "n/a"
    return str(value.date())
