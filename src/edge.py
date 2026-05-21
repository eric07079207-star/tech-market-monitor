from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

import numpy as np
import pandas as pd


RELIABLE_DOMAINS = {
    "reuters.com": 0.95,
    "sec.gov": 0.98,
    "federalreserve.gov": 0.98,
    "fomc": 0.98,
    "stlouisfed.org": 0.96,
    "whitehouse.gov": 0.95,
    "commerce.gov": 0.95,
    "bloomberg.com": 0.94,
    "wsj.com": 0.92,
    "marketwatch.com": 0.82,
    "cnbc.com": 0.80,
    "investors.com": 0.80,
    "finance.yahoo.com": 0.72,
}


@dataclass(frozen=True)
class EdgeQuality:
    source_domain: str
    source_reliability_score: float
    text_density_score: float
    structure_score: float
    dedup_key: str
    quality_score: float


def source_domain(source: str | None) -> str:
    if not source:
        return ""
    value = str(source).strip()
    if not value:
        return ""
    if "://" not in value:
        value = f"https://{value}"
    parsed = urlparse(value)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def source_reliability(source: str | None) -> float:
    domain = source_domain(source)
    if not domain:
        return 0.5
    for known, score in RELIABLE_DOMAINS.items():
        if known in domain:
            return score
    return 0.62


def dedup_key(*parts: object) -> str:
    clean = " | ".join(str(part).strip().lower() for part in parts if str(part).strip())
    return clean.replace("\n", " ")


def quality_score(
    title: str,
    source: str | None = None,
    tags: str | None = None,
    published: pd.Timestamp | None = None,
    raw_text: str | None = None,
) -> EdgeQuality:
    domain = source_domain(source)
    source_score = source_reliability(source)
    text = str(raw_text or title or "")
    word_count = len([token for token in text.split() if token])
    tag_count = len([tag for tag in str(tags or "").split("；") if tag and tag != "未分類"])
    density = float(np.clip(word_count / 20, 0, 1))
    structure = float(np.clip((tag_count / 4) + (0.25 if domain else 0), 0, 1))
    freshness = 0.0
    if published is not None and pd.notna(published):
        age_days = max((pd.Timestamp.now(tz="UTC") - pd.to_datetime(published, utc=True)).days, 0)
        freshness = float(np.clip(1 - age_days / 30, 0, 1))
    score = float(np.clip(source_score * 55 + density * 20 + structure * 15 + freshness * 10, 0, 100))
    return EdgeQuality(
        source_domain=domain,
        source_reliability_score=source_score,
        text_density_score=density,
        structure_score=structure,
        dedup_key=dedup_key(title, domain, tags),
        quality_score=score,
    )


def summarize_quality_frame(data: pd.DataFrame, score_column: str = "quality_score") -> dict:
    if data.empty or score_column not in data:
        return {
            "rows": 0,
            "avg_quality": np.nan,
            "median_quality": np.nan,
            "low_quality_rows": 0,
            "unique_sources": 0,
            "dup_ratio": np.nan,
        }
    score = pd.to_numeric(data[score_column], errors="coerce")
    unique_sources = data["source_domain"].nunique(dropna=True) if "source_domain" in data else 0
    dedup_ratio = np.nan
    if "dedup_key" in data:
        dedup_ratio = 1 - (data["dedup_key"].nunique(dropna=True) / max(len(data), 1))
    return {
        "rows": int(len(data)),
        "avg_quality": float(score.mean()),
        "median_quality": float(score.median()),
        "low_quality_rows": int((score < 55).sum()),
        "unique_sources": int(unique_sources),
        "dup_ratio": float(dedup_ratio) if pd.notna(dedup_ratio) else np.nan,
    }
