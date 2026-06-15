from __future__ import annotations

import re
from datetime import timedelta

import numpy as np
import pandas as pd


RETENTION_DAYS = {
    "official": 3650,
    "pending_short": 7,
    "pending_medium": 14,
    "pending_long": 30,
    "rejected": 30,
    "archived": 3650,
}

LONG_PENDING_TAGS = ("國際", "戰爭", "貿易", "Fed", "利率", "監管")
MEDIUM_PENDING_TAGS = ("財報", "財測", "產品", "需求", "分析師", "訴訟")
SPAM_TERMS = (
    "subscribe",
    "newsletter",
    "sponsored",
    "advertisement",
    "promo",
    "coupon",
    "click here",
    "read more",
    "sign up",
    "watch now",
)
PLACEHOLDER_TERMS = ("untitled", "no title", "n/a", "null", "none")
MAX_SYMBOL_RATIO = 0.35


def annotate_governance(data: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if data is None or data.empty:
        return _with_empty_columns(pd.DataFrame() if data is None else data.copy(), dataset)

    frame = data.copy()
    if "title" in frame:
        frame["title"] = frame["title"].astype(str).apply(_clean_text)
    frame["normalized_title"] = frame.get("title", pd.Series("", index=frame.index)).astype(str).apply(_normalize_title)
    duplicate_counts = frame["normalized_title"].map(frame["normalized_title"].value_counts(dropna=False))
    frame["duplicate_key"] = frame["normalized_title"]
    frame["duplicate_count"] = pd.to_numeric(duplicate_counts, errors="coerce").fillna(0).astype(int)
    frame["title_noise_score"] = frame.get("title", pd.Series("", index=frame.index)).astype(str).apply(_noise_score)

    reasons = []
    statuses = []

    for _, row in frame.iterrows():
        status, reason = _classify_row(row)
        statuses.append(status)
        reasons.append(reason)

    frame["governance_dataset"] = dataset
    frame["governance_status"] = statuses
    frame["governance_reason"] = reasons
    frame["retention_days"] = [RETENTION_DAYS.get(status, 30) for status in statuses]
    frame["first_seen_at_utc"] = _first_available_timestamp(frame)
    frame["expires_at_utc"] = _expires_at(frame["first_seen_at_utc"], frame["retention_days"])
    frame["official_ready"] = frame["governance_status"].eq("official")
    frame["usable_for_sentiment"] = frame["governance_status"].isin(["official", "pending_medium", "pending_long"])
    return frame


def governance_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset, data in frames.items():
        if data is None or data.empty:
            rows.append(_summary_row(dataset, 0, pd.Series(dtype=object), pd.Series(dtype=object)))
            continue
        if "governance_status" not in data:
            data = annotate_governance(data, dataset)
        rows.append(_summary_row(dataset, len(data), data["governance_status"], data.get("governance_reason", pd.Series(dtype=object))))
    return pd.DataFrame(rows)


def _classify_row(row: pd.Series) -> tuple[str, str]:
    title = str(row.get("title", "") or "").strip()
    normalized_title = str(row.get("normalized_title", "") or "").strip()
    published = pd.to_datetime(row.get("published"), errors="coerce", utc=True)
    quality = pd.to_numeric(row.get("quality_score", np.nan), errors="coerce")
    reliability = pd.to_numeric(row.get("source_reliability_score", np.nan), errors="coerce")
    noise = pd.to_numeric(row.get("title_noise_score", np.nan), errors="coerce")
    duplicate_value = pd.to_numeric(row.get("duplicate_count", 0), errors="coerce")
    duplicate_count = int(duplicate_value) if pd.notna(duplicate_value) else 0
    tags = str(row.get("tags", "") or "")

    if not title:
        return "rejected", "missing_title"
    if len(title) < 8:
        return "rejected", "title_too_short"
    if normalized_title in PLACEHOLDER_TERMS:
        return "rejected", "placeholder_title"
    if _looks_like_url_only(title):
        return "rejected", "url_only_title"
    if any(term in title.lower() for term in SPAM_TERMS):
        return "rejected", "promotional_or_spam_text"
    if pd.notna(noise) and noise >= 0.75:
        return "rejected", "high_noise_title"
    if duplicate_count >= 4:
        return "pending_short", "duplicate_cluster"
    if pd.isna(published):
        return "pending_short", "missing_or_invalid_published"
    if published > pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1):
        return "pending_short", "future_timestamp"
    if pd.notna(quality) and quality < 45:
        return _pending_status(tags), "low_quality_score"
    if pd.notna(reliability) and reliability < 0.55:
        return _pending_status(tags), "low_source_reliability"
    return "official", "passed_basic_checks"


def _pending_status(tags: str) -> str:
    if any(tag in tags for tag in LONG_PENDING_TAGS):
        return "pending_long"
    if any(tag in tags for tag in MEDIUM_PENDING_TAGS):
        return "pending_medium"
    return "pending_short"


def _first_available_timestamp(frame: pd.DataFrame) -> pd.Series:
    if "fetched_at_utc" in frame:
        values = pd.to_datetime(frame["fetched_at_utc"], errors="coerce", utc=True)
    else:
        values = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    if "published" in frame:
        published = pd.to_datetime(frame["published"], errors="coerce", utc=True)
        values = values.fillna(published)
    values = values.fillna(pd.Timestamp.now(tz="UTC"))
    return values.dt.strftime("%Y-%m-%dT%H:%M:%S%z")


def _expires_at(first_seen: pd.Series, retention_days: pd.Series) -> pd.Series:
    starts = pd.to_datetime(first_seen, errors="coerce", utc=True)
    days = pd.to_numeric(retention_days, errors="coerce").fillna(30)
    expires = [start + timedelta(days=int(day)) if pd.notna(start) else pd.NaT for start, day in zip(starts, days)]
    return pd.Series(expires).dt.strftime("%Y-%m-%dT%H:%M:%S%z")


def _with_empty_columns(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    frame["governance_dataset"] = dataset
    frame["governance_status"] = pd.Series(dtype=str)
    frame["governance_reason"] = pd.Series(dtype=str)
    frame["retention_days"] = pd.Series(dtype=int)
    frame["first_seen_at_utc"] = pd.Series(dtype=str)
    frame["expires_at_utc"] = pd.Series(dtype=str)
    frame["official_ready"] = pd.Series(dtype=bool)
    frame["usable_for_sentiment"] = pd.Series(dtype=bool)
    frame["normalized_title"] = pd.Series(dtype=str)
    frame["duplicate_key"] = pd.Series(dtype=str)
    frame["duplicate_count"] = pd.Series(dtype=int)
    frame["title_noise_score"] = pd.Series(dtype=float)
    return frame


def _summary_row(dataset: str, count: int, statuses: pd.Series, reasons: pd.Series) -> dict:
    status_counts = statuses.value_counts(dropna=False).to_dict()
    reason_counts = reasons.value_counts(dropna=True).head(3)
    duplicate_rows = int((reasons == "duplicate_cluster").sum()) if not reasons.empty else 0
    garbage_rows = int(reasons.isin(["missing_title", "title_too_short", "placeholder_title", "url_only_title", "promotional_or_spam_text", "high_noise_title"]).sum()) if not reasons.empty else 0
    return {
        "dataset": dataset,
        "rows": int(count),
        "official": int(status_counts.get("official", 0)),
        "pending_short": int(status_counts.get("pending_short", 0)),
        "pending_medium": int(status_counts.get("pending_medium", 0)),
        "pending_long": int(status_counts.get("pending_long", 0)),
        "rejected": int(status_counts.get("rejected", 0)),
        "garbage_rows": garbage_rows,
        "duplicate_rows": duplicate_rows,
        "top_reasons": "；".join(f"{reason}:{int(total)}" for reason, total in reason_counts.items()),
    }


def _clean_text(value: object) -> str:
    text = str(value or "").replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def _normalize_title(value: object) -> str:
    text = _clean_text(value).lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _looks_like_url_only(value: object) -> bool:
    text = _clean_text(value).lower()
    if not text:
        return False
    without_url = re.sub(r"https?://\S+", "", text).strip()
    return text.startswith(("http://", "https://")) and len(without_url) <= 5


def _noise_score(value: object) -> float:
    text = _clean_text(value)
    if not text:
        return 1.0
    symbols = sum(1 for char in text if not char.isalnum() and not char.isspace())
    symbol_ratio = symbols / max(len(text), 1)
    repeated = 1.0 if re.search(r"(.)\1{5,}", text) else 0.0
    word_count = len(text.split())
    short_penalty = 0.25 if word_count <= 2 else 0.0
    spam_penalty = 0.35 if any(term in text.lower() for term in SPAM_TERMS) else 0.0
    return float(np.clip(symbol_ratio / MAX_SYMBOL_RATIO * 0.45 + repeated * 0.35 + short_penalty + spam_penalty, 0, 1))
