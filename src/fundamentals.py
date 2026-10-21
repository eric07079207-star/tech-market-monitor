from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

from .config import CACHE_DIR, STOCK_TICKERS


FUNDAMENTAL_CACHE = CACHE_DIR / "fundamentals" / "sec_fundamental_observations.parquet"
EXTRACT_VERSION = "all-concepts-v2"
FUNDAMENTAL_TICKERS = [ticker for ticker in STOCK_TICKERS if ticker not in {"TSM"}]
SEC_HEADERS = {
    "User-Agent": os.getenv("SEC_USER_AGENT", "tech-market-monitor research dashboard contact@tech-market-monitor.local"),
    "Accept-Encoding": "gzip, deflate",
}
METRIC_CONCEPTS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "Revenues"],
    "net_income": ["NetIncomeLoss"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "eps_diluted": ["EarningsPerShareDiluted"],
}


def load_fundamental_observations(path: Path | None = None) -> pd.DataFrame:
    path = path or FUNDAMENTAL_CACHE
    if not path.exists():
        return _empty_observations()
    try:
        return _sanitize_observations(pd.read_parquet(path))
    except Exception:
        return _empty_observations()


def refresh_sec_fundamentals_if_due(max_age_hours: int = 24, path: Path | None = None) -> pd.DataFrame:
    path = path or FUNDAMENTAL_CACHE
    if path.exists():
        age_seconds = pd.Timestamp.now(tz="UTC").timestamp() - path.stat().st_mtime
        cached = load_fundamental_observations(path)
        version_ok = "extract_version" in cached and cached["extract_version"].eq(EXTRACT_VERSION).all()
        if age_seconds < max_age_hours * 3600 and version_ok:
            return cached
    observations = fetch_sec_fundamental_observations(FUNDAMENTAL_TICKERS)
    if not observations.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        observations.to_parquet(path, index=False)
    return observations if not observations.empty else load_fundamental_observations(path)


def fetch_sec_fundamental_observations(tickers: list[str]) -> pd.DataFrame:
    try:
        ticker_payload = requests.get("https://www.sec.gov/files/company_tickers.json", headers=SEC_HEADERS, timeout=25).json()
    except Exception:
        return _empty_observations()
    ticker_to_cik = {str(row.get("ticker", "")).upper(): int(row.get("cik_str", 0)) for row in ticker_payload.values()}
    rows: list[dict[str, object]] = []
    for ticker in tickers:
        cik = ticker_to_cik.get(ticker.upper())
        if not cik:
            continue
        try:
            facts = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", headers=SEC_HEADERS, timeout=30).json()
        except Exception:
            continue
        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        for metric, concepts in METRIC_CONCEPTS.items():
            # Companies changed XBRL concepts over time. Keeping only the first
            # modern concept silently erased otherwise valid older filings.
            for concept in (name for name in concepts if name in us_gaap):
                units = us_gaap[concept].get("units", {})
                unit, values = _preferred_unit(units)
                for item in values:
                    if item.get("form") not in {"10-Q", "10-K"} or not item.get("filed") or not item.get("end"):
                        continue
                    rows.append(
                        {
                            "ticker": ticker.upper(),
                            "cik": str(cik),
                            "metric": metric,
                            "concept": concept,
                            "unit": unit,
                            "value": item.get("val"),
                            "period_start": item.get("start", ""),
                            "period_end": item.get("end", ""),
                            "filed_at": item.get("filed", ""),
                            "form": item.get("form", ""),
                            "fy": item.get("fy", ""),
                            "fp": item.get("fp", ""),
                            "frame": item.get("frame", ""),
                            "source": "SEC Company Facts",
                            "source_url": f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
                            "extract_version": EXTRACT_VERSION,
                        }
                    )
    return _sanitize_observations(pd.DataFrame(rows))


def annotate_sample_fundamental_coverage(samples: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return samples.copy()
    result = samples.copy()
    if observations.empty:
        result["fundamental_ticker_coverage"] = 0
        result["fundamental_coverage_state"] = "資料不足"
        result["fundamental_source"] = "SEC 資料尚未取得"
        result["fundamental_missing_reason"] = "SEC 資料取得失敗，等待重試"
        return result

    data = observations.copy()
    data["filed_at"] = pd.to_datetime(data["filed_at"], errors="coerce")
    data["period_end"] = pd.to_datetime(data["period_end"], errors="coerce")
    data = data.dropna(subset=["filed_at", "period_end"])
    coverage, state, reasons = [], [], []
    required_metrics = {"revenue", "operating_cash_flow"}
    first_complete_dates: list[pd.Timestamp] = []
    for ticker in FUNDAMENTAL_TICKERS:
        ticker_rows = data[data["ticker"] == ticker]
        metric_first = ticker_rows[ticker_rows["metric"].isin(required_metrics)].groupby("metric")["filed_at"].min()
        if required_metrics.issubset(set(metric_first.index)):
            first_complete_dates.append(metric_first.max())
    official_breadth_start = sorted(first_complete_dates)[5] if len(first_complete_dates) >= 6 else pd.Timestamp("2010-01-01")
    for sample_date in pd.to_datetime(result["prediction_date"], errors="coerce"):
        asof = data[data["filed_at"] <= sample_date]
        available = 0
        for ticker in FUNDAMENTAL_TICKERS:
            ticker_rows = asof[asof["ticker"] == ticker]
            metric_rows = ticker_rows[ticker_rows["metric"].isin(required_metrics)]
            latest = metric_rows.sort_values("filed_at").groupby("metric").tail(1)
            if required_metrics.issubset(set(latest["metric"])) and (sample_date - latest["filed_at"].min()).days <= 450:
                available += 1
        coverage.append(available)
        if available >= 6:
            state.append("完整（官方）")
            reasons.append("")
        elif sample_date < official_breadth_start:
            state.append("不適用（官方歷史範圍外）")
            reasons.append(f"早於至少六家公司可同時比較的 SEC XBRL 起點 {official_breadth_start:%Y-%m-%d}，不再等待")
        elif available >= 3:
            state.append("部分可用")
            reasons.append(f"僅 {available}/{len(FUNDAMENTAL_TICKERS)} 家具備營收與營運現金流")
        else:
            state.append("關鍵指標不足（不再等待）")
            reasons.append(f"僅 {available}/{len(FUNDAMENTAL_TICKERS)} 家具備必要欄位；官方歷史資料不足")
    result["fundamental_ticker_coverage"] = coverage
    result["fundamental_coverage_state"] = state
    result["fundamental_missing_reason"] = reasons
    result["fundamental_source"] = "SEC Company Facts（point-in-time filed_at）"
    return result


def _preferred_unit(units: dict) -> tuple[str, list[dict]]:
    if "USD" in units:
        return "USD", units["USD"]
    if "USD/shares" in units:
        return "USD/shares", units["USD/shares"]
    if "shares" in units:
        return "shares", units["shares"]
    if not units:
        return "", []
    unit = next(iter(units))
    return unit, units[unit]


def _sanitize_observations(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return _empty_observations()
    result = frame.copy()
    for column in ["value"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in ["period_start", "period_end", "filed_at"]:
        result[column] = pd.to_datetime(result[column], errors="coerce")
    result = result.dropna(subset=["ticker", "metric", "value", "filed_at", "period_end"])
    return result.drop_duplicates(["ticker", "metric", "period_start", "period_end", "filed_at", "form", "value"]).sort_values(["ticker", "metric", "filed_at"]).reset_index(drop=True)


def _empty_observations() -> pd.DataFrame:
    return pd.DataFrame(columns=["ticker", "cik", "metric", "concept", "unit", "value", "period_start", "period_end", "filed_at", "form", "fy", "fp", "frame", "source", "source_url", "extract_version"])
