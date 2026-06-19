from __future__ import annotations

import json
import csv
import io
import re
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import ALL_TICKERS, CACHE_DIR, FRED_SERIES, default_start_date


PRICE_CACHE = CACHE_DIR / "prices.parquet"
MACRO_CACHE = CACHE_DIR / "macro.parquet"
METADATA_CACHE = CACHE_DIR / "metadata.json"

BLS_PUBLIC_SERIES = {
    "CPIAUCSL": {"provider_series": "CUUR0000SA0", "label": "US CPI"},
    "UNRATE": {"provider_series": "LNS14000000", "label": "US Unemployment Rate"},
    "PAYEMS": {"provider_series": "CES0000000001", "label": "US Nonfarm Payrolls"},
}

BEA_PAGE_SERIES = {
    "PCEPI": {"label": "US PCE YoY"},
}

DOL_SERIES = {
    "ICSA": {"label": "US Initial Jobless Claims"},
}

HTTP_TIMEOUT_SHORT = 8
HTTP_TIMEOUT_MEDIUM = 20
HTTP_TIMEOUT_LONG = 30
YFINANCE_TIMEOUT = 20
YFINANCE_BATCH_SIZE = 8


def _build_http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    return session


HTTP = _build_http_session()


def _normalize_yfinance(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    frames = []
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        level1 = set(raw.columns.get_level_values(1))
        price_fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

        for ticker in tickers:
            if ticker in level0:
                sub = raw[ticker].copy()
            elif ticker in level1:
                sub = raw.xs(ticker, axis=1, level=1, drop_level=True).copy()
            else:
                continue

            if not set(sub.columns).intersection(price_fields):
                continue
            sub = sub.reset_index()
            sub = _ensure_date_column(sub, raw.index)
            sub["symbol"] = ticker
            frames.append(sub)
    else:
        sub = raw.copy().reset_index()
        sub = _ensure_date_column(sub, raw.index)
        sub["symbol"] = tickers[0]
        frames.append(sub)

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, ignore_index=True)
    prices.columns = [str(col).lower().replace(" ", "_") for col in prices.columns]
    if "date" not in prices.columns and "datetime" in prices.columns:
        prices = prices.rename(columns={"datetime": "date"})

    keep = [col for col in ["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"] if col in prices]
    prices = prices[keep].copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in prices:
            prices[col] = pd.to_numeric(prices[col], errors="coerce")
    prices = prices.dropna(subset=["date", "symbol", "close"]).sort_values(["symbol", "date"])
    return prices.reset_index(drop=True)


def _ensure_date_column(frame: pd.DataFrame, source_index: pd.Index) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(col) for col in frame.columns]
    lowered = {str(col).lower(): str(col) for col in frame.columns}
    if "date" in lowered:
        return frame.rename(columns={lowered["date"]: "date"})
    if "datetime" in lowered:
        return frame.rename(columns={lowered["datetime"]: "date"})

    first_col = str(frame.columns[0]) if len(frame.columns) else ""
    if first_col and first_col.lower().startswith("unnamed"):
        frame = frame.rename(columns={first_col: "date"})
        return frame

    if len(frame) == len(source_index):
        frame.insert(0, "date", pd.Index(source_index))
        return frame
    raise KeyError("date")


def fetch_price_history(
    tickers: Iterable[str] = ALL_TICKERS,
    start: date | str | None = None,
    end: date | str | None = None,
) -> pd.DataFrame:
    ticker_list = list(dict.fromkeys(tickers))
    start = start or default_start_date()
    frames: list[pd.DataFrame] = []
    for batch in _chunked(ticker_list, YFINANCE_BATCH_SIZE):
        raw = yf.download(
            batch,
            start=str(start),
            end=str(end) if end else None,
            auto_adjust=True,
            group_by="ticker",
            threads=False,
            progress=False,
            timeout=YFINANCE_TIMEOUT,
        )
        normalized = _normalize_yfinance(raw, batch)
        if not normalized.empty:
            frames.append(normalized)
    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"])
    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")
    return prices.reset_index(drop=True)


def _chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(items), max(size, 1)):
        yield items[index : index + max(size, 1)]


def fetch_fred_series(
    start: date | str | None = None,
    series_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    start_ts = pd.to_datetime(start or default_start_date())
    series_map = series_map or FRED_SERIES
    frames = []
    for series_id, label in series_map.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            response = HTTP.get(url, timeout=HTTP_TIMEOUT_SHORT)
            response.raise_for_status()
            data = pd.read_csv(StringIO(response.text))
        except Exception:
            continue

        if data.empty:
            continue
        data = data.rename(columns={"observation_date": "date", series_id: "value"})
        data["date"] = pd.to_datetime(data["date"])
        data["value"] = pd.to_numeric(data["value"].replace(".", pd.NA), errors="coerce")
        data = data[data["date"] >= start_ts].dropna(subset=["value"])
        data["series"] = series_id
        data["label"] = label
        data["source"] = "FRED"
        data["provider_series"] = series_id
        frames.append(data[["date", "series", "label", "value", "source", "provider_series"]])

    if not frames:
        return pd.DataFrame(columns=["date", "series", "label", "value", "source", "provider_series"])
    return pd.concat(frames, ignore_index=True).sort_values(["series", "date"])


def fetch_bls_series(start: date | str | None = None) -> pd.DataFrame:
    start_ts = pd.to_datetime(start or default_start_date())
    start_year = int(start_ts.year)
    end_year = int(pd.Timestamp.today().year)
    frames = []
    provider_to_canonical = {meta["provider_series"]: canonical for canonical, meta in BLS_PUBLIC_SERIES.items()}
    series_ids = list(provider_to_canonical)

    for year_start in range(start_year, end_year + 1, 10):
        year_end = min(year_start + 9, end_year)
        payload = {
            "seriesid": series_ids,
            "startyear": str(year_start),
            "endyear": str(year_end),
        }
        try:
            response = HTTP.post(
                "https://api.bls.gov/publicAPI/v2/timeseries/data/",
                json=payload,
                timeout=HTTP_TIMEOUT_MEDIUM,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            payload_json = response.json()
        except Exception:
            continue

        for series_blob in payload_json.get("Results", {}).get("series", []):
            provider_series = str(series_blob.get("seriesID", "") or "")
            canonical = provider_to_canonical.get(provider_series)
            if not canonical:
                continue
            label = BLS_PUBLIC_SERIES[canonical]["label"]
            for point in series_blob.get("data", []):
                period = str(point.get("period", "") or "")
                if not period.startswith("M") or period == "M13":
                    continue
                try:
                    year = int(point["year"])
                    month = int(period[1:])
                    value = float(str(point["value"]).replace(",", ""))
                except Exception:
                    continue
                frames.append(
                    {
                        "date": pd.Timestamp(year=year, month=month, day=1),
                        "series": canonical,
                        "label": label,
                        "value": value,
                        "source": "BLS",
                        "provider_series": provider_series,
                    }
                )

    if not frames:
        return pd.DataFrame(columns=["date", "series", "label", "value", "source", "provider_series"])
    data = pd.DataFrame(frames).drop_duplicates(subset=["series", "date"]).sort_values(["series", "date"])
    data = data[data["date"] >= start_ts]
    return data.reset_index(drop=True)


def fetch_bea_pce_series(start: date | str | None = None) -> pd.DataFrame:
    start_ts = pd.to_datetime(start or default_start_date())
    try:
        response = HTTP.get(
            "https://www.bea.gov/data/personal-consumption-expenditures-price-index",
            timeout=HTTP_TIMEOUT_MEDIUM,
        )
        response.raise_for_status()
        html = response.text
    except Exception:
        return pd.DataFrame(columns=["date", "series", "label", "value", "source", "provider_series"])

    rows = []
    for month_name, pct_text in re.findall(r"<tr class=\"item-fact-row\"><td>([^<]+)</td><td>\+?([0-9.]+)%</td></tr>", html):
        ts = pd.to_datetime(month_name, format="%B %Y", errors="coerce")
        if pd.isna(ts) or ts < start_ts:
            continue
        rows.append(
            {
                "date": ts.normalize(),
                "series": "PCEPI",
                "label": BEA_PAGE_SERIES["PCEPI"]["label"],
                "value": float(pct_text),
                "source": "BEA",
                "provider_series": "bea_pce_yoy_page",
            }
        )
    if not rows:
        return pd.DataFrame(columns=["date", "series", "label", "value", "source", "provider_series"])
    return pd.DataFrame(rows).drop_duplicates(subset=["series", "date"]).sort_values("date").reset_index(drop=True)


def fetch_dol_initial_claims_series(start: date | str | None = None) -> pd.DataFrame:
    start_ts = pd.to_datetime(start or default_start_date())
    try:
        response = HTTP.get(
            "https://oui.doleta.gov/unemploy/csv/ar539.csv",
            timeout=HTTP_TIMEOUT_LONG,
        )
        response.raise_for_status()
    except Exception:
        return pd.DataFrame(columns=["date", "series", "label", "value", "source", "provider_series"])

    try:
        reader = csv.DictReader(io.StringIO(response.text))
        rows = list(reader)
    except Exception:
        return pd.DataFrame(columns=["date", "series", "label", "value", "source", "provider_series"])

    if not rows:
        return pd.DataFrame(columns=["date", "series", "label", "value", "source", "provider_series"])

    data = pd.DataFrame(rows)
    if "c2" not in data or "c3" not in data:
        return pd.DataFrame(columns=["date", "series", "label", "value", "source", "provider_series"])

    data["date"] = pd.to_datetime(data["c2"], errors="coerce")
    data["value"] = pd.to_numeric(data["c3"], errors="coerce")
    data = data.dropna(subset=["date", "value"])
    data = data[data["date"] >= start_ts]
    weekly = data.groupby("date", as_index=False)["value"].sum().sort_values("date")
    weekly["series"] = "ICSA"
    weekly["label"] = DOL_SERIES["ICSA"]["label"]
    weekly["source"] = "DOL"
    weekly["provider_series"] = "eta539_c3_aggregate"
    return weekly[["date", "series", "label", "value", "source", "provider_series"]]


def fetch_macro_series(start: date | str | None = None, previous: pd.DataFrame | None = None) -> pd.DataFrame:
    previous = previous.copy() if previous is not None else pd.DataFrame()
    official_canonicals = set(BLS_PUBLIC_SERIES)
    fred_only_map = {series_id: label for series_id, label in FRED_SERIES.items() if series_id not in official_canonicals}

    frames = [
        fetch_fred_series(start=start, series_map=fred_only_map),
        fetch_bls_series(start=start),
        fetch_bea_pce_series(start=start),
        fetch_dol_initial_claims_series(start=start),
    ]

    if previous.empty and all(frame.empty for frame in frames):
        return pd.DataFrame(columns=["date", "series", "label", "value", "source", "provider_series"])
    return _merge_macro_frames(previous, frames)


def _merge_macro_frames(previous: pd.DataFrame, frames: list[pd.DataFrame]) -> pd.DataFrame:
    pieces = []
    candidate_series = set(FRED_SERIES) | set(BLS_PUBLIC_SERIES) | set(BEA_PAGE_SERIES) | set(DOL_SERIES)
    previous = previous.copy()
    if not previous.empty and "date" in previous:
        previous["date"] = pd.to_datetime(previous["date"], errors="coerce")

    combined_new = pd.concat([frame for frame in frames if frame is not None and not frame.empty], ignore_index=True) if any(frame is not None and not frame.empty for frame in frames) else pd.DataFrame()
    if not combined_new.empty and "date" in combined_new:
        combined_new["date"] = pd.to_datetime(combined_new["date"], errors="coerce")

    for series_id in sorted(candidate_series):
        new_slice = combined_new[combined_new["series"] == series_id].copy() if not combined_new.empty else pd.DataFrame()
        old_slice = previous[previous["series"] == series_id].copy() if not previous.empty and "series" in previous else pd.DataFrame()
        merged_slice = _merge_macro_series_slice(new_slice, old_slice)
        if not merged_slice.empty:
            pieces.append(merged_slice)

    if not pieces:
        return pd.DataFrame(columns=["date", "series", "label", "value", "source", "provider_series"])
    merged = pd.concat(pieces, ignore_index=True)
    if "source" not in merged:
        merged["source"] = "unknown"
    if "provider_series" not in merged:
        merged["provider_series"] = merged["series"]
    merged = merged.drop_duplicates(subset=["series", "date"], keep="last").sort_values(["series", "date"]).reset_index(drop=True)
    return merged


def _merge_macro_series_slice(new_slice: pd.DataFrame, old_slice: pd.DataFrame) -> pd.DataFrame:
    if new_slice.empty:
        return old_slice
    if old_slice.empty:
        return new_slice

    new_slice = new_slice.copy()
    old_slice = old_slice.copy()
    if "date" in new_slice:
        new_slice["date"] = pd.to_datetime(new_slice["date"], errors="coerce")
    if "date" in old_slice:
        old_slice["date"] = pd.to_datetime(old_slice["date"], errors="coerce")

    merged = pd.concat([old_slice, new_slice], ignore_index=True)
    merged = merged.drop_duplicates(subset=["series", "date"], keep="last")
    return merged.sort_values(["series", "date"]).reset_index(drop=True)


def refresh_market_data(start: date | str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    previous_macro = pd.read_parquet(MACRO_CACHE) if MACRO_CACHE.exists() else pd.DataFrame()
    prices = fetch_price_history(start=start)
    macro = fetch_macro_series(start=start, previous=previous_macro)
    if not prices.empty:
        prices.to_parquet(PRICE_CACHE, index=False)
    if not macro.empty:
        macro.to_parquet(MACRO_CACHE, index=False)
    METADATA_CACHE.write_text(
        json.dumps(
            {
                "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "start": str(start or default_start_date()),
                "price_rows": int(len(prices)),
                "macro_rows": int(len(macro)),
            },
            indent=2,
        )
    )
    return prices, macro


def load_cached_market_data(start: date | str | None = None, force_refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if force_refresh or not PRICE_CACHE.exists():
        return refresh_market_data(start=start)

    prices = pd.read_parquet(PRICE_CACHE)
    prices["date"] = pd.to_datetime(prices["date"])
    if MACRO_CACHE.exists():
        macro = pd.read_parquet(MACRO_CACHE)
        macro["date"] = pd.to_datetime(macro["date"])
    else:
        macro = pd.DataFrame(columns=["date", "series", "label", "value"])
    return prices, macro


def load_metadata() -> dict:
    if not METADATA_CACHE.exists():
        return {}
    try:
        return json.loads(METADATA_CACHE.read_text())
    except json.JSONDecodeError:
        return {}


def cache_path(path: str | Path) -> Path:
    return CACHE_DIR / path
