"""Structured official event layer for point-in-time KG research."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

from .config import CACHE_DIR, STOCK_TICKERS


OFFICIAL_EVENT_CACHE = CACHE_DIR / "official" / "official_events.parquet"
SEC_HEADERS = {
    "User-Agent": os.getenv("SEC_USER_AGENT", "tech-market-monitor research dashboard contact@tech-market-monitor.local"),
    "Accept-Encoding": "gzip, deflate",
}
FED_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
SEC_FORMS = {"10-K", "10-Q", "8-K", "6-K", "20-F", "DEF 14A", "S-1", "S-3", "424B5"}


def load_official_events(path: Path | None = None) -> pd.DataFrame:
    path = path or OFFICIAL_EVENT_CACHE
    if not path.exists():
        return _empty_events()
    try:
        return _sanitize(pd.read_parquet(path))
    except Exception:
        return _empty_events()


def refresh_official_events_if_due(max_age_hours: int = 24, path: Path | None = None) -> pd.DataFrame:
    path = path or OFFICIAL_EVENT_CACHE
    if path.exists():
        age = pd.Timestamp.now(tz="UTC").timestamp() - path.stat().st_mtime
        if age < max_age_hours * 3600:
            return load_official_events(path)
    frames = [fetch_sec_filing_events(STOCK_TICKERS), fetch_fed_statement_events(), build_official_macro_observations()]
    fresh = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in frames) else pd.DataFrame()
    previous = load_official_events(path)
    combined = pd.concat([previous, fresh], ignore_index=True) if not previous.empty else fresh
    combined = _sanitize(combined)
    if not combined.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(path, index=False)
    return combined


def fetch_sec_filing_events(tickers: list[str]) -> pd.DataFrame:
    try:
        response = requests.get("https://www.sec.gov/files/company_tickers.json", headers=SEC_HEADERS, timeout=30)
        response.raise_for_status()
        ticker_payload = response.json()
    except Exception:
        return _empty_events()
    ticker_to_cik = {str(row.get("ticker", "")).upper(): int(row.get("cik_str", 0)) for row in ticker_payload.values()}
    rows: list[dict[str, object]] = []
    for ticker in tickers:
        cik = ticker_to_cik.get(ticker.upper())
        if not cik:
            continue
        try:
            response = requests.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", headers=SEC_HEADERS, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            continue
        filings = payload.get("filings", {}).get("recent", {})
        rows.extend(_sec_rows(ticker.upper(), cik, filings))
        for archive in payload.get("filings", {}).get("files", []):
            name = str(archive.get("name") or "")
            if not name:
                continue
            try:
                archived = requests.get(f"https://data.sec.gov/submissions/{name}", headers=SEC_HEADERS, timeout=30)
                archived.raise_for_status()
                rows.extend(_sec_rows(ticker.upper(), cik, archived.json()))
            except Exception:
                continue
    return _sanitize(pd.DataFrame(rows))


def _sec_rows(ticker: str, cik: int, filings: dict) -> list[dict[str, object]]:
    rows = []
    forms = filings.get("form", [])
    for index, form in enumerate(forms):
        if str(form) not in SEC_FORMS:
            continue
        filed = _at(filings, "filingDate", index)
        accession = _at(filings, "accessionNumber", index)
        primary = _at(filings, "primaryDocument", index)
        if not filed or not accession:
            continue
        accession_flat = accession.replace("-", "")
        source_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_flat}/{primary}" if primary else f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_flat}/"
        rows.append(
            _event_row(
                event_id=f"SEC-{ticker}-{accession}",
                published_at=filed,
                agency="SEC",
                event_type="earnings" if form in {"10-K", "10-Q", "20-F"} else "offering_dilution" if form in {"S-1", "S-3", "424B5"} else "regulation",
                title=f"{ticker} filed {form}",
                source_url=source_url,
                affected_tickers=ticker,
                raw_value=form,
                revision_state="immutable_filing",
                timestamp_basis="official_filing_date",
                point_in_time_eligible=True,
            )
        )
    return rows


def fetch_fed_statement_events() -> pd.DataFrame:
    try:
        response = requests.get(FED_CALENDAR_URL, timeout=30, headers={"User-Agent": SEC_HEADERS["User-Agent"]})
        response.raise_for_status()
    except Exception:
        return _empty_events()
    links = set(re.findall(r'href=["\']([^"\']*fomcstatement(\d{8})[^"\']*)["\']', response.text, flags=re.I))
    links.update(re.findall(r'href=["\']([^"\']*monetarypolicy/files/monetary(\d{8})[^"\']*)["\']', response.text, flags=re.I))
    links = sorted(links)
    rows = []
    for href, date_text in links:
        rows.append(
            _event_row(
                event_id=f"FED-FOMC-{date_text}",
                published_at=pd.to_datetime(date_text, format="%Y%m%d", errors="coerce"),
                agency="Federal Reserve",
                event_type="macro_policy",
                title=f"FOMC statement {date_text}",
                source_url=urljoin(FED_CALENDAR_URL, href),
                affected_tickers="QQQ;SPY;VOO",
                raw_value="FOMC statement",
                revision_state="official_archive",
                timestamp_basis="official_release_date",
                point_in_time_eligible=True,
            )
        )
    return _sanitize(pd.DataFrame(rows))


def build_official_macro_observations() -> pd.DataFrame:
    """Preserve BLS/BEA official values without inventing historical release times.

    The existing macro pipeline reads these providers directly. Their period dates
    are not the same as publication timestamps, so they remain in the official
    layer but are excluded from point-in-time event matching until an exact release
    timestamp is available.
    """
    macro_path = CACHE_DIR / "macro.parquet"
    if not macro_path.exists():
        return _empty_events()
    try:
        macro = pd.read_parquet(macro_path)
    except Exception:
        return _empty_events()
    macro = macro[macro.get("source", pd.Series("", index=macro.index)).isin(["BLS", "BEA"])].copy()
    rows = []
    source_urls = {
        "BLS": "https://api.bls.gov/publicAPI/v2/timeseries/data/",
        "BEA": "https://www.bea.gov/data/personal-consumption-expenditures-price-index",
    }
    for row in macro.itertuples(index=False):
        period = pd.to_datetime(getattr(row, "date", None), errors="coerce")
        agency = str(getattr(row, "source", ""))
        series = str(getattr(row, "series", ""))
        value = getattr(row, "value", None)
        if pd.isna(period) or not agency or not series:
            continue
        rows.append(
            _event_row(
                event_id=f"{agency}-{series}-{period:%Y%m%d}", published_at=pd.NaT, period_date=period,
                agency=agency, event_type="inflation_data" if series in {"CPIAUCSL", "PCEPI"} else "macro_policy",
                title=f"{agency} {series} official observation {period:%Y-%m}", source_url=source_urls[agency],
                affected_tickers="QQQ;SPY;VOO", raw_value=str(value), numeric_value=value, revision_state="current_official_vintage_only",
                timestamp_basis="observation_period_not_release_time", point_in_time_eligible=False,
            )
        )
    return _sanitize(pd.DataFrame(rows))


def official_events_as_historical_evidence(samples: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if samples is None or samples.empty or events is None or events.empty:
        return pd.DataFrame()
    eligible_events = events[events["point_in_time_eligible"].fillna(False)].copy()
    rows: list[dict[str, object]] = []
    for sample in samples[["sample_id", "prediction_date"]].itertuples(index=False):
        prediction_date = pd.to_datetime(sample.prediction_date, errors="coerce", utc=True)
        if pd.isna(prediction_date):
            continue
        start = prediction_date.normalize() - pd.Timedelta(days=3)
        cutoff = prediction_date.normalize() + pd.Timedelta(days=1)
        window = eligible_events[(eligible_events["published_at"] >= start) & (eligible_events["published_at"] < cutoff)]
        for event in window.itertuples(index=False):
            title_hash = hashlib.sha256(str(event.title).lower().encode("utf-8")).hexdigest()[:20]
            rows.append(
                {
                    "sample_id": str(sample.sample_id), "prediction_date": prediction_date.tz_localize(None),
                    "published_at": event.published_at, "event_type_primary": event.event_type,
                    "canonical_event": f"{event.event_type}:{event.agency.lower().replace(' ', '_')}",
                    "event_cluster_key": f"official:{event.event_id}", "title": event.title,
                    "article_url": event.source_url, "source_domain": _domain_for_agency(event.agency),
                    "source_country": "US", "language": "English", "source_reliability_score": 0.99,
                    "title_hash": title_hash, "impact_direction": "neutral", "provider": "Official event layer",
                    "fetched_at_utc": pd.Timestamp.now(tz="UTC"), "official_event_id": event.event_id,
                }
            )
    return pd.DataFrame(rows)


def _event_row(**kwargs: object) -> dict[str, object]:
    return {**kwargs, "fetched_at_utc": pd.Timestamp.now(tz="UTC")}


def _at(data: dict, key: str, index: int) -> str:
    values = data.get(key, [])
    return str(values[index]) if index < len(values) and values[index] is not None else ""


def _domain_for_agency(agency: str) -> str:
    return "sec.gov" if agency == "SEC" else "federalreserve.gov"


def _sanitize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return _empty_events()
    result = frame.copy()
    result["published_at"] = pd.to_datetime(result["published_at"], errors="coerce", utc=True)
    result["fetched_at_utc"] = pd.to_datetime(result["fetched_at_utc"], errors="coerce", utc=True)
    if "period_date" not in result:
        result["period_date"] = pd.NaT
    result["period_date"] = pd.to_datetime(result["period_date"], errors="coerce")
    if "numeric_value" not in result:
        result["numeric_value"] = pd.NA
    result["numeric_value"] = pd.to_numeric(result["numeric_value"], errors="coerce")
    result["raw_value"] = result["raw_value"].fillna("").astype(str)
    result = result.dropna(subset=["event_id", "agency", "source_url"])
    eligible = result["point_in_time_eligible"].fillna(False).astype(bool)
    result = result[~eligible | result["published_at"].notna()]
    return result.drop_duplicates("event_id", keep="last").sort_values("published_at").reset_index(drop=True)


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "event_id", "published_at", "agency", "event_type", "title", "source_url", "affected_tickers",
        "raw_value", "numeric_value", "revision_state", "timestamp_basis", "point_in_time_eligible", "period_date", "fetched_at_utc",
    ])
