from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ETF_TICKERS = ["QQQ", "XLK", "SMH", "SOXX", "IGV", "IYW", "VGT"]
STOCK_TICKERS = ["AAPL", "MSFT", "NVDA", "AMD", "META", "GOOGL", "AMZN", "TSLA"]
MARKET_TICKERS = ["SPY", "IWM", "^VIX", "TLT", "HYG", "DX-Y.NYB"]
ALL_TICKERS = ETF_TICKERS + STOCK_TICKERS + MARKET_TICKERS

DISPLAY_NAMES = {
    "QQQ": "Nasdaq 100",
    "XLK": "S&P Tech",
    "SMH": "Semiconductor",
    "SOXX": "Semiconductor",
    "IGV": "Software",
    "IYW": "US Tech",
    "VGT": "Vanguard Tech",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "AMD": "AMD",
    "META": "Meta",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "SPY": "S&P 500",
    "IWM": "Russell 2000",
    "^VIX": "VIX",
    "TLT": "Long Treasury",
    "HYG": "High Yield Bond",
    "DX-Y.NYB": "US Dollar Index",
}

ASSET_GROUPS = {
    **{ticker: "ETF" for ticker in ETF_TICKERS},
    **{ticker: "個股" for ticker in STOCK_TICKERS},
    **{ticker: "市場壓力" for ticker in MARKET_TICKERS},
}

FRED_SERIES = {
    "DGS10": "US 10Y Yield",
    "DGS2": "US 2Y Yield",
    "T10Y2Y": "10Y-2Y Curve",
    "BAMLH0A0HYM2": "High Yield OAS",
    "BAMLC0A0CM": "Investment Grade OAS",
    "NFCI": "Chicago Fed NFCI",
}

NEWS_QUERIES = {
    "QQQ": "Nasdaq 100 ETF QQQ technology stocks",
    "XLK": "XLK technology sector ETF",
    "SMH": "SMH semiconductor ETF",
    "SOXX": "SOXX semiconductor ETF",
    "IGV": "IGV software ETF",
    "IYW": "IYW technology ETF",
    "VGT": "VGT technology ETF",
    "AAPL": "Apple AAPL stock",
    "MSFT": "Microsoft MSFT stock",
    "NVDA": "Nvidia NVDA stock AI chips",
    "AMD": "AMD stock AI chips",
    "META": "Meta Platforms META stock",
    "GOOGL": "Alphabet Google GOOGL stock",
    "AMZN": "Amazon AMZN AWS stock",
    "TSLA": "Tesla TSLA stock",
}

INTERNATIONAL_NEWS_QUERIES = {
    "GLOBAL": "global markets economy geopolitics trade",
    "WAR": "war conflict geopolitical risk global markets",
    "TRADE": "international trade negotiations tariffs export controls",
    "CENTRAL_BANKS": "central banks interest rates inflation global markets",
    "ENERGY": "oil energy supply shock global markets",
}


@dataclass(frozen=True)
class Horizon:
    label: str
    days: int


HORIZONS = [
    Horizon("1M", 21),
    Horizon("3M", 63),
    Horizon("6M", 126),
    Horizon("12M", 252),
]


def default_start_date() -> date:
    return (pd.Timestamp.today().normalize() - pd.DateOffset(years=20)).date()
