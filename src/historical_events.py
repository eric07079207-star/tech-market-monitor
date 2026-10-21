"""Rate-limited, point-in-time historical news evidence for KG backtests."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import quote, urlparse
from xml.etree import ElementTree

import pandas as pd
import requests

from .config import CACHE_DIR
from .edge import source_reliability
from .kg import EVENT_TAXONOMY, NEGATIVE_TERMS, POSITIVE_TERMS


HISTORICAL_EVENT_DIR = CACHE_DIR / "backtest"
HISTORICAL_EVENT_EVIDENCE_CACHE = HISTORICAL_EVENT_DIR / "historical_event_evidence.parquet"
HISTORICAL_EVENT_STATUS_CACHE = HISTORICAL_EVENT_DIR / "historical_event_backfill_status.json"
GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
HISTORICAL_SOURCE_START = pd.Timestamp("2006-01-01", tz="UTC")
QUERY = '(NASDAQ OR "stock market" OR "Federal Reserve" OR inflation OR earnings OR tariff OR semiconductor)'
OFFICIAL_QUERY = (
    '(site:sec.gov OR site:federalreserve.gov OR site:bls.gov OR site:bea.gov '
    'OR site:apple.com OR site:microsoft.com OR site:nvidia.com OR site:amd.com '
    'OR site:tesla.com OR site:amazon.com OR site:abc.xyz OR site:meta.com) '
    '(earnings OR filing OR guidance OR inflation OR employment OR rate OR policy OR investor)'
)
DIRECT_MARKET_TERMS = {
    "nasdaq", "stock market", "wall street", "fed", "federal reserve", "earnings", "guidance",
    "semiconductor", "chip", "technology stocks",
    "apple", "microsoft", "nvidia", "amd", "tesla", "amazon", "alphabet", "meta",
}
CONTEXTUAL_MACRO_TERMS = {"inflation", "cpi", "pce", "tariff", "trade", "interest rate", "recession"}
US_MARKET_CONTEXT_TERMS = {"u.s.", " us ", "united states", "america", "market", "technology", "fed", "wall street"}
OFFICIAL_DOMAIN_SUFFIXES = (
    "sec.gov", "federalreserve.gov", "bls.gov", "bea.gov", "whitehouse.gov", "commerce.gov",
    "apple.com", "microsoft.com", "nvidia.com", "amd.com", "tesla.com", "amazon.com", "abc.xyz", "meta.com",
)


def load_historical_event_evidence(path: Path | None = None) -> pd.DataFrame:
    path = path or HISTORICAL_EVENT_EVIDENCE_CACHE
    if not path.exists():
        return _empty_evidence()
    try:
        evidence = pd.read_parquet(path)
    except Exception:
        return _empty_evidence()
    if evidence.empty:
        return evidence
    if {"sample_id", "article_url"}.issubset(evidence.columns):
        evidence = evidence.drop_duplicates(["sample_id", "article_url"], keep="last")
    if {"sample_id", "title_hash", "source_domain"}.issubset(evidence.columns):
        evidence = evidence.drop_duplicates(["sample_id", "title_hash", "source_domain"], keep="last")
    sort_columns = [column for column in ("prediction_date", "published_at", "source_domain") if column in evidence]
    if sort_columns:
        evidence = evidence.sort_values(sort_columns)
    return evidence.reset_index(drop=True)


def load_historical_event_status(path: Path | None = None) -> dict:
    path = path or HISTORICAL_EVENT_STATUS_CACHE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def backfill_historical_event_evidence(
    samples: pd.DataFrame,
    existing: pd.DataFrame | None = None,
    *,
    max_samples: int = 12,
    pause_seconds: float = 5.2,
    request_timeout: int = 45,
) -> tuple[pd.DataFrame, dict]:
    """Fetch a bounded number of unprocessed sample windows without future leakage."""
    existing = existing.copy() if existing is not None else load_historical_event_evidence()
    if not existing.empty and "title" in existing:
        existing["event_type_primary"] = existing["title"].map(_event_type)
        existing["canonical_event"] = [
            _canonical_event(title, event_type)
            for title, event_type in zip(existing["title"], existing["event_type_primary"])
        ]
        existing["event_cluster_key"] = existing["canonical_event"]
        existing["impact_direction"] = existing["title"].map(_impact_direction)
    previous_status = load_historical_event_status()
    # Official events are mapped across all matching sample dates, but that does
    # not mean the broader historical news window has been searched. Keep these
    # states separate so official coverage cannot prematurely stop backfilling.
    providers = existing.get("provider", pd.Series("", index=existing.index)).fillna("").astype(str)
    searched_evidence = existing[providers.ne("Official event layer")]
    processed = set(searched_evidence.get("sample_id", pd.Series(dtype=str)).dropna().astype(str))
    processed.update(map(str, previous_status.get("searched_sample_ids", [])))
    regime_priority = _regime_priority(samples)
    queue = samples.copy()
    queue["prediction_date"] = pd.to_datetime(queue["prediction_date"], errors="coerce", utc=True)
    queue = queue[queue["prediction_date"].ge(HISTORICAL_SOURCE_START)]
    queue = queue[~queue["sample_id"].astype(str).isin(processed)]
    queue = _balanced_queue(queue, max_samples=max_samples, regime_priority=regime_priority)

    fetched_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    completed_this_run: list[str] = []
    for position, row in enumerate(queue.itertuples(index=False)):
        if position:
            time.sleep(max(0.0, pause_seconds))
        try:
            articles = _fetch_historical_window(pd.Timestamp(row.prediction_date), timeout=request_timeout)
        except Exception as exc:
            failures.append({"sample_id": str(row.sample_id), "error": str(exc)[:300]})
            continue
        fetched_rows.extend(_normalize_articles(str(row.sample_id), pd.Timestamp(row.prediction_date), articles))
        completed_this_run.append(str(row.sample_id))

    additions = pd.DataFrame(fetched_rows)
    if existing.empty:
        combined = additions.copy()
    elif additions.empty:
        combined = existing
    else:
        combined = pd.concat([existing, additions], ignore_index=True)
    if not combined.empty:
        combined = combined.drop_duplicates(["sample_id", "article_url"], keep="last")
        combined = combined.drop_duplicates(["sample_id", "title_hash", "source_domain"], keep="last")
        combined = combined.sort_values(["prediction_date", "published_at", "source_domain"]).reset_index(drop=True)

    status = {
        "updated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "provider": "Google News RSS; GDELT DOC 2.0 fallback",
        "provider_coverage_start": HISTORICAL_SOURCE_START.isoformat(),
        "processed_sample_ids": sorted(processed | set(completed_this_run)),
        "searched_sample_ids": sorted(processed | set(completed_this_run)),
        "processed_sample_windows": int(len(processed | set(completed_this_run))),
        "eligible_sample_windows": int(samples["prediction_date"].pipe(pd.to_datetime, errors="coerce", utc=True).ge(HISTORICAL_SOURCE_START).sum()),
        "article_rows": int(len(combined)),
        "attempted_this_run": int(len(queue)),
        "articles_added_this_run": int(len(additions)),
        "failures_this_run": failures,
        "complete": False,
        "note": "Google 歷史索引不保證完整；僅保留當日以前、可追溯來源且通過跨出版者確認的事件。",
    }
    status["complete"] = status["processed_sample_windows"] >= status["eligible_sample_windows"]
    return combined if not combined.empty else _empty_evidence(), status


def annotate_historical_event_coverage(samples: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    result = samples.copy()
    if result.empty:
        return result
    result["prediction_date"] = pd.to_datetime(result["prediction_date"], errors="coerce")
    metrics = _evidence_metrics(evidence)
    drop_columns = [
        "independent_news_source_count", "historical_event_count", "confirmed_event_cluster_count",
        "historical_event_types", "official_event_count", "kg_coverage_state",
    ]
    result = result.drop(columns=[column for column in drop_columns if column in result], errors="ignore")
    result = result.merge(metrics, on="sample_id", how="left")
    for column in ["independent_news_source_count", "historical_event_count", "confirmed_event_cluster_count", "official_event_count"]:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0).astype(int)
    result["historical_event_types"] = result["historical_event_types"].fillna("")
    result["kg_coverage_state"] = "需要歷史多來源事件回補"
    unavailable = pd.to_datetime(result["prediction_date"], utc=True).lt(HISTORICAL_SOURCE_START)
    result.loc[unavailable, "kg_coverage_state"] = "歷史新聞來源不可得"
    ready = (result["confirmed_event_cluster_count"].ge(1) & result["independent_news_source_count"].ge(2)) | result["official_event_count"].ge(1)
    result.loc[ready, "kg_coverage_state"] = "歷史多來源事件可用"
    return result


def save_historical_event_artifacts(evidence: pd.DataFrame, status: dict) -> None:
    HISTORICAL_EVENT_DIR.mkdir(parents=True, exist_ok=True)
    evidence.to_parquet(HISTORICAL_EVENT_EVIDENCE_CACHE, index=False)
    HISTORICAL_EVENT_STATUS_CACHE.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def _fetch_historical_window(prediction_date: pd.Timestamp, timeout: int) -> list[dict]:
    # Search official releases first, then supplement them with broad independent
    # reporting. The publication cutoff is still enforced during normalization.
    official = _fetch_google_news_window(prediction_date, timeout=timeout, query_text=OFFICIAL_QUERY)
    broad = _fetch_google_news_window(prediction_date, timeout=timeout, query_text=QUERY)
    articles = official + broad
    if articles:
        unique: dict[str, dict] = {}
        for article in articles:
            key = str(article.get("url") or article.get("title") or "")
            if key:
                unique[key] = article
        return list(unique.values())
    return _fetch_gdelt_window(prediction_date, timeout=timeout)


def _fetch_google_news_window(prediction_date: pd.Timestamp, timeout: int, query_text: str = QUERY) -> list[dict]:
    end_date = prediction_date.tz_convert("UTC").normalize()
    start_date = end_date - pd.Timedelta(days=3)
    before_date = end_date + pd.Timedelta(days=1)
    query = f'{query_text} after:{start_date:%Y-%m-%d} before:{before_date:%Y-%m-%d}'
    url = f"{GOOGLE_NEWS_RSS_URL}?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 tech-market-monitor-research"})
    response.raise_for_status()
    root = ElementTree.fromstring(response.content)
    articles: list[dict] = []
    for item in root.findall(".//item"):
        source_node = item.find("source")
        source_name = source_node.text.strip() if source_node is not None and source_node.text else ""
        source_url = source_node.attrib.get("url", "") if source_node is not None else ""
        articles.append(
            {
                "url": item.findtext("link", default=""),
                "title": item.findtext("title", default=""),
                "seendate": item.findtext("pubDate", default=""),
                "domain": _domain(source_url) or source_name,
                "sourcecountry": "",
                "language": "English",
                "provider": "Google News RSS",
            }
        )
    return articles


def _fetch_gdelt_window(prediction_date: pd.Timestamp, timeout: int) -> list[dict]:
    end = prediction_date.tz_convert("UTC").normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    start = end.normalize() - pd.Timedelta(days=3)
    params = {
        "query": QUERY,
        "mode": "artlist",
        "maxrecords": 75,
        "format": "json",
        "sort": "datedesc",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }
    last_error = ""
    for attempt in range(3):
        response = requests.get(GDELT_DOC_URL, params=params, timeout=timeout, headers={"User-Agent": "tech-market-monitor-research/1.0"})
        if response.status_code != 429:
            response.raise_for_status()
            return response.json().get("articles", [])
        last_error = response.text.strip()[:200]
        if attempt < 2:
            time.sleep(6.0 * (attempt + 1))
    raise RuntimeError(f"GDELT rate limited the request after retries: {last_error}")


def _normalize_articles(sample_id: str, prediction_date: pd.Timestamp, articles: list[dict]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cutoff = prediction_date.tz_convert("UTC").normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    for article in articles:
        title = str(article.get("title") or "").strip()
        url = str(article.get("url") or "").strip()
        domain = str(article.get("domain") or _domain(url)).lower().removeprefix("www.")
        published = pd.to_datetime(article.get("seendate"), errors="coerce", utc=True)
        if not title or not url or pd.isna(published) or published > cutoff:
            continue
        if not _market_relevant(title) and not _is_official_domain(domain):
            continue
        event_type = _event_type(title)
        rows.append(
            {
                "sample_id": sample_id,
                "prediction_date": cutoff.normalize().tz_localize(None),
                "published_at": published,
                "event_type_primary": event_type,
                "canonical_event": _canonical_event(title, event_type),
                "event_cluster_key": _canonical_event(title, event_type),
                "title": title,
                "article_url": url,
                "source_domain": domain,
                "source_country": str(article.get("sourcecountry") or ""),
                "language": str(article.get("language") or ""),
                "source_reliability_score": source_reliability(domain),
                "title_hash": hashlib.sha256(_normalize_title(title).encode("utf-8")).hexdigest()[:20],
                "impact_direction": _impact_direction(title),
                "provider": str(article.get("provider") or "GDELT DOC 2.0"),
                "fetched_at_utc": pd.Timestamp.now(tz="UTC"),
            }
        )
    return rows


def _evidence_metrics(evidence: pd.DataFrame) -> pd.DataFrame:
    columns = ["sample_id", "independent_news_source_count", "historical_event_count", "confirmed_event_cluster_count", "historical_event_types", "official_event_count"]
    if evidence is None or evidence.empty:
        return pd.DataFrame(columns=columns)
    data = evidence.copy()
    data = data[data["source_domain"].astype(str).ne("")]
    if data.empty:
        return pd.DataFrame(columns=columns)
    event_groups = (
        data.groupby(["sample_id", "event_cluster_key"], dropna=False)
        .agg(source_count=("source_domain", "nunique"), story_count=("title_hash", "nunique"))
        .reset_index()
    )
    official_mask = data.get("provider", pd.Series("", index=data.index)).eq("Official event layer")
    official_counts = data[official_mask].groupby("sample_id")["event_cluster_key"].nunique().rename("official_event_count").reset_index()
    confirmed = event_groups[event_groups["source_count"].ge(2) & event_groups["story_count"].ge(2)]
    confirmed_evidence = data.merge(confirmed[["sample_id", "event_cluster_key"]], on=["sample_id", "event_cluster_key"], how="inner")
    base = (
        confirmed_evidence.groupby("sample_id")
        .agg(
            independent_news_source_count=("source_domain", "nunique"),
            historical_event_count=("title_hash", "nunique"),
            historical_event_types=("event_type_primary", lambda values: "；".join(sorted(set(map(str, values))))),
        )
        .reset_index()
    )
    counts = confirmed.groupby("sample_id").size().rename("confirmed_event_cluster_count").reset_index()
    result = base.merge(counts, on="sample_id", how="left").merge(official_counts, on="sample_id", how="outer")
    return result.fillna({"independent_news_source_count": 0, "historical_event_count": 0, "confirmed_event_cluster_count": 0, "historical_event_types": "", "official_event_count": 0})


def _balanced_queue(queue: pd.DataFrame, max_samples: int, regime_priority: list[str] | None = None) -> pd.DataFrame:
    """Round-robin regimes so early results are not all bull or stress samples."""
    limit = max(0, int(max_samples))
    if limit == 0 or queue.empty:
        return queue.head(0)
    regimes = regime_priority or ["高波動／壓力", "明顯下跌趨勢", "強勢反彈／轉折", "橫盤／低波動", "正常／溫和上漲"]
    pools = {
        regime: group.sort_values("prediction_date", ascending=False).reset_index(drop=True)
        for regime, group in queue.groupby("regime_bucket", dropna=False)
    }
    selected: list[pd.Series] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for regime in regimes:
            pool = pools.get(regime)
            if pool is not None and offset < len(pool):
                selected.append(pool.iloc[offset])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        offset += 1
    return pd.DataFrame(selected).reset_index(drop=True) if selected else queue.head(0)


def _regime_priority(samples: pd.DataFrame) -> list[str]:
    default = ["高波動／壓力", "明顯下跌趨勢", "強勢反彈／轉折", "橫盤／低波動", "正常／溫和上漲"]
    if samples is None or samples.empty or "regime_bucket" not in samples:
        return default
    source_count = pd.to_numeric(samples.get("independent_news_source_count"), errors="coerce").fillna(0)
    official_count = pd.to_numeric(samples.get("official_event_count"), errors="coerce").fillna(0)
    event_ready = source_count.ge(2) | official_count.ge(1)
    fundamental_ready = samples.get("fundamental_coverage_state", pd.Series("", index=samples.index)).eq("完整（官方）")
    eligible = event_ready & fundamental_ready
    counts = samples.assign(_eligible=eligible).groupby("regime_bucket")["_eligible"].sum().to_dict()
    return sorted(default, key=lambda regime: (int(counts.get(regime, 0)), default.index(regime)))


def _event_type(title: str) -> str:
    lower = title.lower()
    scores = {event_type: sum(_term_in_text(term, lower) for term in terms) for event_type, terms in EVENT_TAXONOMY.items()}
    event_type, score = max(scores.items(), key=lambda item: item[1])
    return event_type if score else "market_structure"


def _impact_direction(title: str) -> str:
    lower = title.lower()
    positive = sum(term in lower for term in POSITIVE_TERMS)
    negative = sum(term in lower for term in NEGATIVE_TERMS)
    return "positive" if positive > negative else "negative" if negative > positive else "neutral"


def _canonical_event(title: str, event_type: str) -> str:
    lower = title.lower()
    phrases = {
        "macro_policy": ["rate cut", "rate hike", "fomc", "federal reserve", "fed"],
        "inflation_data": ["core pce", "pce", "cpi", "ppi", "inflation"],
        "earnings": ["nvidia", "apple", "microsoft", "tesla", "amazon", "alphabet", "meta", "amd", "earnings"],
        "guidance": ["guidance cut", "cuts guidance", "raises guidance", "outlook", "guidance"],
        "geopolitical": ["trade war", "tariff", "sanction", "war", "conflict", "trade"],
        "regulation": ["antitrust", "export control", "doj", "ftc", "regulation", "lawsuit"],
        "product_launch": ["artificial intelligence", "ai", "semiconductor", "chip", "launch", "product"],
    }
    for phrase in phrases.get(event_type, EVENT_TAXONOMY.get(event_type, [])):
        if _term_in_text(phrase, lower):
            return f"{event_type}:{phrase.replace(' ', '_')}"
    return f"{event_type}:general"


def _market_relevant(title: str) -> bool:
    lower = f" {title.lower()} "
    if any(term in lower for term in DIRECT_MARKET_TERMS):
        return True
    return any(term in lower for term in CONTEXTUAL_MACRO_TERMS) and any(term in lower for term in US_MARKET_CONTEXT_TERMS)


def _term_in_text(term: str, text: str) -> bool:
    pattern = rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])"
    return re.search(pattern, text.lower()) is not None


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _domain(url: str) -> str:
    return urlparse(url).netloc


def _is_official_domain(domain: str) -> bool:
    value = str(domain).lower().removeprefix("www.")
    return any(value == suffix or value.endswith(f".{suffix}") for suffix in OFFICIAL_DOMAIN_SUFFIXES)


def _empty_evidence() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sample_id", "prediction_date", "published_at", "event_type_primary", "canonical_event",
            "event_cluster_key", "title", "article_url", "source_domain", "source_country", "language",
            "source_reliability_score", "title_hash", "impact_direction", "provider", "fetched_at_utc",
        ]
    )
