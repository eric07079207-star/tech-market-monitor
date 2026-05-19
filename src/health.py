from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import cache_path


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
    cache_updated = metadata.get("updated_at_utc", "尚未寫入")
    rows = [
        _row("市場價格", "prices.parquet", prices, "date", "價格與量能快取", cache_updated, 36),
        _row("總經資料", "macro.parquet", macro, "date", "FRED 與市場壓力資料", cache_updated, 72),
        _row("標的新聞", "news.parquet", news, "published", "watchlist 新聞", cache_updated, 12),
        _row("國際新聞", "international_news.parquet", international_news, "published", "國際重大與隨機新聞", cache_updated, 12),
        _row("預測紀錄", "prediction_log.csv", prediction_log, "prediction_date", "5D/20D/60D 驗證資料", cache_updated, 72),
        _row("新聞探索", "discovery_news.parquet", discovery_news, "published", "隨機主題新聞", cache_updated, 12),
        _row("候選觀察股", "discovery_candidates.parquet", discovery_candidates, "", "新聞探索量化候選", cache_updated, 12),
        _row("候選歷史紀錄", "discovery_history.parquet", discovery_history, "date", "每日 Top 15 候選追蹤", cache_updated, 36),
    ]
    return pd.DataFrame(rows)


def missing_price_symbols(prices: pd.DataFrame, symbols: list[str]) -> list[str]:
    if prices.empty:
        return symbols
    available = set(prices["symbol"].dropna().astype(str).unique())
    return [symbol for symbol in symbols if symbol not in available]


def _row(
    name: str,
    filename: str,
    data: pd.DataFrame | None,
    date_column: str,
    note: str,
    cache_updated: str,
    stale_after_hours: int,
) -> dict:
    count = 0 if data is None else len(data)
    latest = _max_date(data, date_column) if date_column else "n/a"
    fetched_at = _max_datetime(data, "fetched_at_utc") or cache_updated
    file_size = _file_size(cache_path(filename))
    status, detail = _health_status(count, fetched_at, stale_after_hours)
    return {
        "狀態": status,
        "資料項目": name,
        "筆數": int(count),
        "最新資料日期": str(latest),
        "最近抓取 UTC": fetched_at or "n/a",
        "檔案大小": file_size,
        "自動更新": "是",
        "Streamlit使用": "是",
        "說明": f"{note}；{detail}",
    }


def _max_date(data: pd.DataFrame | None, column: str) -> str:
    if data is None or data.empty or column not in data:
        return "n/a"
    value = pd.to_datetime(data[column], errors="coerce").max()
    if pd.isna(value):
        return "n/a"
    return str(value.date())


def _max_datetime(data: pd.DataFrame | None, column: str) -> str:
    if data is None or data.empty or column not in data:
        return ""
    value = pd.to_datetime(data[column], errors="coerce", utc=True).max()
    if pd.isna(value):
        return ""
    return value.isoformat()


def _health_status(count: int, fetched_at: str, stale_after_hours: int) -> tuple[str, str]:
    if count <= 0:
        return "🔴 無資料", "目前沒有資料"
    if not fetched_at or fetched_at == "尚未寫入":
        return "🟡 未知", "尚未記錄抓取時間"
    value = pd.to_datetime(fetched_at, errors="coerce", utc=True)
    if pd.isna(value):
        return "🟡 未知", "抓取時間格式無法判讀"
    age_hours = (datetime.now(timezone.utc) - value.to_pydatetime()).total_seconds() / 3600
    if age_hours <= stale_after_hours:
        return "🟢 正常", f"約 {age_hours:.1f} 小時前更新"
    if age_hours <= stale_after_hours * 2:
        return "🟡 注意", f"約 {age_hours:.1f} 小時未更新"
    return "🔴 過期", f"約 {age_hours:.1f} 小時未更新"


def _file_size(path: Path) -> str:
    if not path.exists():
        return "n/a"
    size = path.stat().st_size
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / 1024 / 1024:.1f} MB"
