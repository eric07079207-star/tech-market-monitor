from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from .config import ALL_TICKERS, CACHE_DIR, FRED_SERIES, default_start_date


PRICE_CACHE = CACHE_DIR / "prices.parquet"
MACRO_CACHE = CACHE_DIR / "macro.parquet"
METADATA_CACHE = CACHE_DIR / "metadata.json"


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
    raw = yf.download(
        ticker_list,
        start=str(start),
        end=str(end) if end else None,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    return _normalize_yfinance(raw, ticker_list)


def fetch_fred_series(start: date | str | None = None) -> pd.DataFrame:
    start_ts = pd.to_datetime(start or default_start_date())
    frames = []
    for series_id, label in FRED_SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            data = pd.read_csv(url)
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
        frames.append(data[["date", "series", "label", "value"]])

    if not frames:
        return pd.DataFrame(columns=["date", "series", "label", "value"])
    return pd.concat(frames, ignore_index=True).sort_values(["series", "date"])


def refresh_market_data(start: date | str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = fetch_price_history(start=start)
    macro = fetch_fred_series(start=start)
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
