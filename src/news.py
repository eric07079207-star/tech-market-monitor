from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from .config import NEWS_QUERIES


TAG_RULES = {
    "AI/晶片": ["ai", "artificial intelligence", "chip", "semiconductor", "gpu", "datacenter", "data center"],
    "財報/財測": ["earnings", "revenue", "profit", "guidance", "forecast", "quarter", "results"],
    "Fed/利率": ["fed", "federal reserve", "rate cut", "rate hike", "treasury yield", "inflation", "cpi", "pce"],
    "監管/訴訟": ["regulator", "antitrust", "lawsuit", "court", "doj", "ftc", "eu fine", "probe"],
    "產品/需求": ["iphone", "cloud", "aws", "azure", "advertising", "ev", "deliveries", "model y"],
    "分析師": ["analyst", "upgrade", "downgrade", "price target", "rating"],
    "併購/合作": ["acquire", "merger", "partnership", "deal", "investment"],
    "大盤風險": ["selloff", "recession", "risk", "tariff", "geopolitical", "war"],
}


@dataclass(frozen=True)
class NewsItem:
    symbol: str
    title: str
    link: str
    source: str
    published: datetime | None
    tags: str


def fetch_google_news(symbol: str, query: str, days: int = 7, limit: int = 8) -> list[NewsItem]:
    q = quote_plus(f"{query} when:{days}d")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    try:
        response = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
    except Exception:
        return []

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError:
        return []

    items: list[NewsItem] = []
    for item in root.findall("./channel/item")[:limit]:
        title = _node_text(item, "title")
        link = _node_text(item, "link")
        source_node = item.find("source")
        source = source_node.text if source_node is not None and source_node.text else ""
        published = _parse_pub_date(_node_text(item, "pubDate"))
        tags = classify_tags(title)
        items.append(NewsItem(symbol=symbol, title=title, link=link, source=source, published=published, tags=tags))
    return items


def fetch_news_batch(symbols: list[str] | None = None, days: int = 7, limit_per_symbol: int = 6) -> pd.DataFrame:
    symbols = symbols or list(NEWS_QUERIES)
    rows = []
    seen = set()
    for symbol in symbols:
        query = NEWS_QUERIES.get(symbol, symbol)
        for item in fetch_google_news(symbol, query, days=days, limit=limit_per_symbol):
            key = (item.symbol, item.title)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "symbol": item.symbol,
                    "title": item.title,
                    "source": item.source,
                    "published": item.published,
                    "tags": item.tags,
                    "link": item.link,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["symbol", "title", "source", "published", "tags", "link"])
    news = pd.DataFrame(rows)
    news["published"] = pd.to_datetime(news["published"], utc=True, errors="coerce")
    return news.sort_values(["published", "symbol"], ascending=[False, True]).reset_index(drop=True)


def classify_tags(text: str) -> str:
    lower = text.lower()
    tags = [label for label, words in TAG_RULES.items() if any(word in lower for word in words)]
    return "；".join(tags) if tags else "未分類"


def rule_based_news_summary(news: pd.DataFrame, max_items: int = 12) -> str:
    if news.empty:
        return "目前沒有抓到近期新聞。"

    tag_counts = (
        news.assign(tag=news["tags"].str.split("；"))
        .explode("tag")
        .query("tag != '未分類'")
        .groupby("tag")
        .size()
        .sort_values(ascending=False)
    )
    leading_tags = "、".join(tag_counts.head(4).index.tolist()) if not tag_counts.empty else "未分類消息"
    symbols = ", ".join(news["symbol"].drop_duplicates().head(8).tolist())
    headlines = news.head(max_items)["title"].tolist()
    bullets = "\n".join(f"- {title}" for title in headlines[:5])
    return f"近期新聞集中在 {leading_tags}，主要覆蓋 {symbols}。\n{bullets}"


def _node_text(item: ET.Element, name: str) -> str:
    node = item.find(name)
    return node.text if node is not None and node.text else ""


def _parse_pub_date(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return None
