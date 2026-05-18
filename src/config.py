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
ANNUAL_PICK_TICKERS = ["TSLA", "PLTR", "CRWD", "VST", "RKLB", "IONQ", "OKLO", "SOFI", "HOOD", "TMDX"]
ALL_TICKERS = ETF_TICKERS + STOCK_TICKERS + MARKET_TICKERS + ANNUAL_PICK_TICKERS

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
    "PLTR": "Palantir",
    "CRWD": "CrowdStrike",
    "VST": "Vistra",
    "RKLB": "Rocket Lab",
    "IONQ": "IonQ",
    "OKLO": "Oklo",
    "SOFI": "SoFi",
    "HOOD": "Robinhood",
    "TMDX": "TransMedics",
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
    **{ticker: "年度十大" for ticker in ANNUAL_PICK_TICKERS},
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

DISCOVERY_NEWS_TOPICS = {
    "AI/資料中心": "AI data center stocks high growth",
    "半導體設備": "semiconductor equipment stocks breakout",
    "網路安全": "cybersecurity stocks earnings growth",
    "核能/電力": "nuclear energy power grid stocks data centers",
    "國防/航太": "defense aerospace drone stocks contracts",
    "太空": "space stocks satellite launch contracts",
    "量子運算": "quantum computing stocks commercial breakthrough",
    "金融科技": "fintech stocks digital banking payments",
    "加密/交易平台": "crypto exchange brokerage stocks regulation",
    "醫療科技": "medical device biotech growth stocks",
    "機器人/自動化": "robotics automation stocks AI",
    "能源/材料": "energy materials stocks supply shortage",
    "消費復甦": "consumer discretionary stocks earnings recovery",
    "小型成長股": "small cap growth stocks breakout",
    "國際貿易": "tariff trade negotiation stocks winners",
}

ANNUAL_PICKS_2026 = [
    {
        "ticker": "TSLA",
        "theme": "AI / 自動駕駛 / 機器人 / 能源",
        "risk_level": "高",
        "reason": "大型權值股中仍具高選擇權特性，市場重新定價自動駕駛、機器人與能源平台的可能性。",
        "risk": "估值高、資本支出大、執行與需求波動。",
    },
    {
        "ticker": "PLTR",
        "theme": "AI 軟體 / 政府與企業資料平台",
        "risk_level": "高",
        "reason": "AI 軟體商業化與政府/企業資料平台需求仍具延伸空間。",
        "risk": "估值壓力高，成長若放緩容易大幅修正。",
    },
    {
        "ticker": "CRWD",
        "theme": "網路安全 / AI 安全營運",
        "risk_level": "中高",
        "reason": "網安需求結構性存在，商業模式與現金流品質相對成熟。",
        "risk": "估值高，企業 IT 支出與競爭可能壓縮倍數。",
    },
    {
        "ticker": "VST",
        "theme": "AI 資料中心電力需求",
        "risk_level": "中",
        "reason": "AI 資料中心推升電力需求，且相對具現金流支撐。",
        "risk": "電力價格、監管與能源週期波動。",
    },
    {
        "ticker": "RKLB",
        "theme": "太空發射 / 衛星基礎建設",
        "risk_level": "高",
        "reason": "太空基礎建設與發射服務具高成長選擇權。",
        "risk": "燒錢、發射執行、合約時程與融資風險。",
    },
    {
        "ticker": "IONQ",
        "theme": "量子運算",
        "risk_level": "極高",
        "reason": "量子運算若進入商業化早期，股價彈性大。",
        "risk": "商業化時間高度不確定，估值與題材波動大。",
    },
    {
        "ticker": "OKLO",
        "theme": "核能 / 小型模組反應爐",
        "risk_level": "極高",
        "reason": "AI 電力需求帶動核能題材，SMR 具長線想像空間。",
        "risk": "監管、技術落地、營收能見度與融資風險。",
    },
    {
        "ticker": "SOFI",
        "theme": "數位金融 / 消費金融平台",
        "risk_level": "高",
        "reason": "金融科技平台化與獲利改善具中期重評價機會。",
        "risk": "利率、信用週期、金融監管與消費信貸風險。",
    },
    {
        "ticker": "HOOD",
        "theme": "散戶交易 / 加密 / 金融平台",
        "risk_level": "高",
        "reason": "交易活躍度、加密題材與平台產品擴張可能帶來彈性。",
        "risk": "交易量週期、監管、加密市場波動。",
    },
    {
        "ticker": "TMDX",
        "theme": "醫療科技 / 器官移植物流",
        "risk_level": "高",
        "reason": "醫療設備與器官移植物流具差異化成長路徑。",
        "risk": "執行、估值、醫療採用速度與營運擴張風險。",
    },
]


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
