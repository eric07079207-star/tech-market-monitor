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
    ai_summary_history: pd.DataFrame | None = None,
    lstm_status: dict | None = None,
    discovery_news: pd.DataFrame | None = None,
    discovery_candidates: pd.DataFrame | None = None,
    discovery_history: pd.DataFrame | None = None,
    focus_news: pd.DataFrame | None = None,
    governance: pd.DataFrame | None = None,
    kg_fact_events: pd.DataFrame | None = None,
    kg_narratives: pd.DataFrame | None = None,
    kg_reactions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cache_updated = metadata.get("updated_at_utc", "尚未寫入")
    rows = [
        _row("市場價格", "主資料", "prices.parquet", prices, "date", "價格與量能快取", cache_updated, 36),
        _row("總經資料", "主資料", "macro.parquet", macro, "date", "FRED 與市場壓力資料", cache_updated, 72),
        _row("標的新聞", "主資料", "news.parquet", news, "published", "watchlist 新聞", cache_updated, 12),
        _row("國際新聞", "主資料", "international_news.parquet", international_news, "published", "國際重大與隨機新聞", cache_updated, 12),
        _row("AI摘要歷史", "摘要", "ai_summary_history.parquet", ai_summary_history, "generated_at_utc", "每日摘要版本與品質紀錄", cache_updated, 72),
        _status_row("LSTM流程", "模型", "lstm/lstm_status.json", lstm_status, cache_updated),
        _prediction_row(prediction_log, prices, cache_updated),
        _row("新聞探索", "探索", "discovery_news.parquet", discovery_news, "published", "隨機主題新聞", cache_updated, 12),
        _row("候選觀察股", "探索", "discovery_candidates.parquet", discovery_candidates, "", "新聞探索量化候選", cache_updated, 12),
        _row("候選歷史紀錄", "探索", "discovery_history.parquet", discovery_history, "date", "每日 Top 15 候選追蹤", cache_updated, 36),
        _row("TSLA專題新聞", "專題", "tsla_keyword_news.parquet", focus_news, "published", "特定個股關鍵字專題追蹤", cache_updated, 12),
        _governance_row(governance, cache_updated),
        _row("知識圖譜事實層", "知識圖譜", "kg/fact_events.parquet", kg_fact_events, "timestamp_utc", "客觀事件與來源", cache_updated, 72),
        _row("知識圖譜敘事層", "知識圖譜", "kg/narrative_features.parquet", kg_narratives, "timestamp_utc", "量化敘事與情緒", cache_updated, 72),
        _row("知識圖譜反應層", "知識圖譜", "kg/market_reactions.parquet", kg_reactions, "validated_at_utc", "事件後市場反應", cache_updated, 72),
    ]
    return pd.DataFrame(rows)


def missing_price_symbols(prices: pd.DataFrame, symbols: list[str]) -> list[str]:
    if prices.empty:
        return symbols
    available = set(prices["symbol"].dropna().astype(str).unique())
    return [symbol for symbol in symbols if symbol not in available]


def _row(
    name: str,
    category: str,
    filename: str,
    data: pd.DataFrame | None,
    date_column: str,
    note: str,
    cache_updated: str,
    stale_after_hours: int,
) -> dict:
    count = 0 if data is None else len(data)
    latest = _max_date(data, date_column) if date_column else _max_date(data, "fetched_at_utc")
    fetched_at = _max_datetime(data, "fetched_at_utc") or cache_updated
    file_size = _file_size(cache_path(filename))
    status, detail = _health_status(count, fetched_at, stale_after_hours)
    return {
        "狀態": status,
        "資料分類": category,
        "資料項目": name,
        "筆數": int(count),
        "最新資料日期": _format_dateish(latest),
        "最近抓取 UTC": _format_datetimeish(fetched_at),
        "檔案大小": file_size,
        "自動更新": "是",
        "Streamlit使用": "是",
        "說明": f"{note}；{detail}",
    }


def _status_row(name: str, category: str, filename: str, status: dict | None, cache_updated: str) -> dict:
    payload = status or {}
    file_size = _file_size(cache_path(filename))
    enabled = bool(payload.get("enabled"))
    updated_at = str(payload.get("updated_at_utc") or cache_updated or "n/a")
    summary = str(payload.get("status") or "尚未建立")
    return {
        "狀態": "🟢 正常" if enabled else "🟡 未啟用",
        "資料分類": category,
        "資料項目": name,
        "筆數": int(payload.get("prediction_rows", 0)) + int(payload.get("backtest_rows", 0)),
        "最新資料日期": _format_dateish(payload.get("last_predict_at_utc") or payload.get("last_backtest_at_utc") or "n/a"),
        "最近抓取 UTC": _format_datetimeish(updated_at),
        "檔案大小": file_size,
        "自動更新": "是",
        "Streamlit使用": "是",
        "說明": summary,
    }


def _governance_row(governance: pd.DataFrame | None, cache_updated: str) -> dict:
    count = 0 if governance is None else len(governance)
    file_size = _file_size(cache_path("governance_summary.parquet"))
    pending = 0
    rejected = 0
    if governance is not None and not governance.empty:
        for column in ["pending_short", "pending_medium", "pending_long"]:
            if column in governance:
                pending += int(pd.to_numeric(governance[column], errors="coerce").fillna(0).sum())
        if "rejected" in governance:
            rejected = int(pd.to_numeric(governance["rejected"], errors="coerce").fillna(0).sum())
    status = "🟢 正常" if count > 0 and rejected == 0 else ("🟡 注意" if count > 0 else "🔴 無資料")
    detail = f"待確認 {pending} 筆；拒收 {rejected} 筆"
    return {
        "狀態": status,
        "資料分類": "治理",
        "資料項目": "資料治理摘要",
        "筆數": int(count),
        "最新資料日期": _format_dateish(cache_updated),
        "最近抓取 UTC": _format_datetimeish(cache_updated),
        "檔案大小": file_size,
        "自動更新": "是",
        "Streamlit使用": "是",
        "說明": f"official / pending / rejected 分層統計；{detail}",
    }


def _prediction_row(prediction_log: pd.DataFrame | None, prices: pd.DataFrame | None, cache_updated: str) -> dict:
    count = 0 if prediction_log is None else len(prediction_log)
    fetched_at = cache_updated
    file_size = _file_size(cache_path("prediction_log.csv"))
    latest = _max_date(prediction_log, "prediction_date")
    status, freshness = _health_status(count, fetched_at, 72)
    detail = _prediction_integrity_detail(prediction_log, prices)
    if count > 0 and detail != "交易日與 horizon 完整":
        status = "🟡 注意"
    return {
        "狀態": status,
        "資料分類": "模型",
        "資料項目": "預測紀錄",
        "筆數": int(count),
        "最新資料日期": _format_dateish(latest),
        "最近抓取 UTC": _format_datetimeish(fetched_at),
        "檔案大小": file_size,
        "自動更新": "是",
        "Streamlit使用": "是",
        "說明": f"5D/20D/60D 驗證資料；{freshness}；{detail}",
    }


def _prediction_integrity_detail(prediction_log: pd.DataFrame | None, prices: pd.DataFrame | None) -> str:
    if prediction_log is None or prediction_log.empty:
        return "目前沒有預測紀錄"
    if "prediction_date" not in prediction_log:
        return "缺少 prediction_date 欄位"
    data = prediction_log.copy()
    data["prediction_date"] = pd.to_datetime(data["prediction_date"], errors="coerce").dt.normalize()
    blank_dates = int(data["prediction_date"].isna().sum())
    if blank_dates:
        return f"有 {blank_dates} 筆 prediction_date 空白"
    required_horizons = {"5D", "20D", "60D"}
    duplicate_count = int(data.duplicated(["prediction_date", "target", "horizon"]).sum()) if {"target", "horizon"}.issubset(data.columns) else 0
    if duplicate_count:
        return f"有 {duplicate_count} 筆重複日期/標的/horizon"
    if prices is None or prices.empty or "symbol" not in prices or "date" not in prices:
        return "無法比對 QQQ 交易日"
    qqq_dates = (
        prices[prices["symbol"].astype(str).eq("QQQ")]["date"]
        .pipe(pd.to_datetime, errors="coerce")
        .dropna()
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
    )
    if qqq_dates.empty:
        return "缺少 QQQ 交易日基準"
    pred_dates = set(data["prediction_date"].dropna())
    start = min(pred_dates)
    end = min(max(pred_dates), qqq_dates.max())
    expected_dates = qqq_dates[(qqq_dates >= start) & (qqq_dates <= end)]
    missing_dates = [date.date().isoformat() for date in expected_dates if date not in pred_dates]
    if missing_dates:
        return "缺少交易日：" + "、".join(missing_dates[:5])
    missing_horizons = []
    for date, group in data.groupby("prediction_date"):
        horizons = set(group.get("horizon", pd.Series(dtype=str)).dropna().astype(str))
        missing = sorted(required_horizons - horizons)
        if missing:
            missing_horizons.append(f"{date.date().isoformat()} 缺 {','.join(missing)}")
    if missing_horizons:
        return "；".join(missing_horizons[:3])
    return "交易日與 horizon 完整"


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


def _format_dateish(value) -> str:
    if value in {None, "", "n/a", "尚未寫入"}:
        return "n/a"
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return str(value)
    return str(dt.date())


def _format_datetimeish(value) -> str:
    if value in {None, "", "n/a", "尚未寫入"}:
        return "n/a"
    dt = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(dt):
        return str(value)
    return dt.strftime("%Y-%m-%d %H:%M UTC")
