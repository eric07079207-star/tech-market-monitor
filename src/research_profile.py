"""Personal research priorities used by the scheduled AI market brief.

This profile intentionally stores themes and tickers only. Position quantities,
cost basis, and any portfolio credentials remain in Streamlit Secrets.
"""

from __future__ import annotations


RESEARCH_PROFILE_VERSION = "2026-08-personal-v1"

PERSONAL_RESEARCH_PROFILE = {
    "portfolio_focus": ["TSLA", "NVDA", "GOOGL", "META", "MSFT", "TSM", "GRAB", "ONDS", "SOFI", "VOO"],
    "market_focus": ["QQQ", "SMH", "SOXX", "XLK", "^VIX", "HYG"],
    "themes": [
        "AI 資本支出與資料中心",
        "半導體與供應鏈",
        "利率、通膨與信用壓力",
        "關稅、出口管制與國際貿易",
        "電動車、自動駕駛與機器人",
        "高成長候選股與年度觀察股",
    ],
    "must_watch": ["Fed", "CPI", "PCE", "就業", "財報", "指引", "增發", "關稅", "出口管制", "戰爭"],
}
