from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .config import CACHE_DIR


KG_DIR = CACHE_DIR / "kg"
FACT_CACHE = KG_DIR / "fact_events.parquet"
NARRATIVE_CACHE = KG_DIR / "narrative_features.parquet"
REACTION_CACHE = KG_DIR / "market_reactions.parquet"
LINK_CACHE = KG_DIR / "event_links.parquet"
METADATA_CACHE = KG_DIR / "kg_metadata.json"

EVENT_TAXONOMY = {
    "macro_policy": ["fed", "rate hike", "rate cut", "fomc", "powell", "central bank"],
    "inflation_data": ["cpi", "pce", "inflation", "core pce", "ppi"],
    "earnings": ["earnings", "eps", "revenue", "quarterly results", "profit"],
    "guidance": ["guidance", "outlook", "forecast", "revised guidance", "raises guidance", "cuts guidance"],
    "offering_dilution": ["offering", "dilution", "share issuance", "equity raise", "secondary offering"],
    "contract": ["contract", "deal", "award", "order", "partnership"],
    "regulation": ["regulation", "antitrust", "doj", "ftc", "probe", "lawsuit", "export control"],
    "geopolitical": ["war", "conflict", "tariff", "trade talks", "trade negotiation", "sanction"],
    "fund_flow": ["fund flow", "inflow", "outflow", "etf flow", "fund flow"],
    "analyst_rating": ["upgrade", "downgrade", "rating", "price target", "analyst"],
    "insider_activity": ["insider", "sold shares", "bought shares", "sec filing"],
    "product_launch": ["launch", "unveil", "release", "product", "service"],
    "supply_chain": ["supply chain", "shipment", "shortage", "factory", "inventory"],
}

NARRATIVE_TERMS = {
    "fear_score": ["selloff", "fear", "panic", "recession", "warning", "risk"],
    "bubble_score": ["bubble", "overheated", "valuation", "froth", "mania", "ai bubble"],
    "recession_score": ["recession", "slowdown", "contraction", "jobless", "weak demand"],
    "policy_risk_score": ["fed", "rate", "inflation", "yield", "central bank"],
    "earnings_risk_score": ["miss", "guidance cut", "lowered outlook", "profit warning", "weak quarter"],
    "liquidity_risk_score": ["spread", "credit", "liquidity", "fund flow", "outflow"],
    "ai_theme_score": ["ai", "gpu", "semiconductor", "chip", "data center"],
    "geopolitical_risk_score": ["war", "conflict", "tariff", "sanction", "trade"],
}

NEGATIVE_TERMS = ["miss", "guidance cut", "dilution", "offering", "lawsuit", "probe", "selloff", "weak"]
POSITIVE_TERMS = ["beat", "raise", "upgrade", "contract", "order", "record", "strong", "gain"]
RELIABLE_SOURCES = {
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
class KGOutput:
    facts: pd.DataFrame
    narratives: pd.DataFrame
    reactions: pd.DataFrame
    links: pd.DataFrame


def build_knowledge_graph(
    news: pd.DataFrame,
    international_news: pd.DataFrame,
    prices: pd.DataFrame,
    macro: pd.DataFrame | None = None,
    regime_context: dict | None = None,
    run_date: date | str | None = None,
) -> KGOutput:
    KG_DIR.mkdir(parents=True, exist_ok=True)
    facts = _build_fact_events(news, international_news, regime_context=regime_context, run_date=run_date)
    facts = _finalize_fact_events(facts)
    narratives = _build_narrative_features(facts)
    reactions = _build_market_reactions(facts, prices)
    reactions = _finalize_reactions(reactions)
    links = _build_event_links(facts, reactions)
    return KGOutput(facts=facts, narratives=narratives, reactions=reactions, links=links)


def save_knowledge_graph(payload: KGOutput) -> None:
    payload.facts.to_parquet(FACT_CACHE, index=False)
    payload.narratives.to_parquet(NARRATIVE_CACHE, index=False)
    payload.reactions.to_parquet(REACTION_CACHE, index=False)
    payload.links.to_parquet(LINK_CACHE, index=False)
    METADATA_CACHE.write_text(
        json.dumps(
            {
                "updated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                "fact_rows": int(len(payload.facts)),
                "narrative_rows": int(len(payload.narratives)),
                "reaction_rows": int(len(payload.reactions)),
                "link_rows": int(len(payload.links)),
                "max_dedup_group_size": int(payload.facts["dedup_group_size"].max()) if "dedup_group_size" in payload.facts and not payload.facts.empty else 0,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_knowledge_graph() -> KGOutput:
    facts = pd.read_parquet(FACT_CACHE) if FACT_CACHE.exists() else pd.DataFrame()
    narratives = pd.read_parquet(NARRATIVE_CACHE) if NARRATIVE_CACHE.exists() else pd.DataFrame()
    reactions = pd.read_parquet(REACTION_CACHE) if REACTION_CACHE.exists() else pd.DataFrame()
    links = pd.read_parquet(LINK_CACHE) if LINK_CACHE.exists() else pd.DataFrame()
    return KGOutput(facts=facts, narratives=narratives, reactions=reactions, links=links)


def kg_summary(payload: KGOutput) -> pd.DataFrame:
    rows = [
        ("事實層", len(payload.facts), _latest_ts(payload.facts), "客觀事件與來源"),
        ("敘事層", len(payload.narratives), _latest_ts(payload.narratives), "量化敘事與情緒特徵"),
        ("反應層", len(payload.reactions), _latest_ts(payload.reactions), "事件後市場反應"),
        ("連結層", len(payload.links), _latest_ts(payload.links), "事件與標的關聯"),
    ]
    return pd.DataFrame(rows, columns=["層級", "筆數", "最新時間", "說明"])


def _build_fact_events(
    news: pd.DataFrame,
    international_news: pd.DataFrame,
    regime_context: dict | None = None,
    run_date: date | str | None = None,
) -> pd.DataFrame:
    rows = []
    for frame, source_layer in [(news, "news"), (international_news, "international_news")]:
        if frame is None or frame.empty:
            continue
        for row in frame.itertuples():
            timestamp = pd.to_datetime(getattr(row, "published", pd.NaT), utc=True, errors="coerce")
            title = str(getattr(row, "title", "") or "")
            event_type_primary, event_type_secondary = _classify_event(title)
            affected = _affected_entities(getattr(row, "symbol", ""), title)
            canonical_event = _canonical_event_name(title, event_type_primary)
            source = str(getattr(row, "source", "") or "")
            rows.append(
                {
                    "event_id": _event_id(source_layer, title, timestamp, affected),
                    "canonical_event_id": _canonical_event_id(canonical_event, event_type_primary, affected, timestamp),
                    "event_hash": _event_hash(canonical_event, event_type_primary, affected, source_layer),
                    "event_date": timestamp.date().isoformat() if pd.notna(timestamp) else str(run_date or ""),
                    "timestamp_utc": timestamp.isoformat() if pd.notna(timestamp) else "",
                    "entity": str(getattr(row, "symbol", "") or "").upper(),
                    "affected_tickers": ",".join(affected),
                    "event_type_primary": event_type_primary,
                    "event_type_secondary": "；".join(event_type_secondary),
                    "canonical_event": canonical_event,
                    "event_title": title,
                    "event_value": np.nan,
                    "event_unit": "",
                    "impact_direction": _impact_direction(title),
                    "impact_score": _impact_score(title),
                    "confidence": _confidence_from_text(title),
                    "source": source,
                    "source_domain": _source_domain(getattr(row, "link", "") or source),
                    "source_reliability_score": _source_reliability_score(getattr(row, "link", "") or source),
                    "regime_context": _regime_label(regime_context),
                    "source_url": str(getattr(row, "link", "") or ""),
                    "raw_title": title,
                    "raw_tags": str(getattr(row, "tags", "") or ""),
                    "source_layer": source_layer,
                    "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                }
            )
    facts = pd.DataFrame(rows)
    if facts.empty:
        return pd.DataFrame(
            columns=[
                "event_id", "canonical_event_id", "event_hash", "event_date", "timestamp_utc", "entity",
                "affected_tickers", "event_type_primary", "event_type_secondary", "canonical_event",
                "event_title", "event_value", "event_unit", "impact_direction", "impact_score", "confidence",
                "source", "source_domain", "source_reliability_score", "regime_context", "source_url",
                "raw_title", "raw_tags", "source_layer", "created_at_utc",
            ]
        )
    return facts.drop_duplicates(subset=["event_hash", "source_url", "event_title"]).sort_values("timestamp_utc", ascending=False).reset_index(drop=True)


def _build_narrative_features(facts: pd.DataFrame) -> pd.DataFrame:
    if facts.empty:
        return pd.DataFrame()
    rows = []
    for row in facts.itertuples():
        title = str(row.event_title or "").lower()
        base_confidence = float(np.clip(row.confidence, 0, 1))
        source_conf = float(np.clip(getattr(row, "source_reliability_score", base_confidence), 0, 1))
        rows.append(
            {
                "event_id": row.event_id,
                "canonical_event_id": getattr(row, "canonical_event_id", row.event_id),
                "timestamp_utc": row.timestamp_utc,
                "entity": row.entity,
                "canonical_event": getattr(row, "canonical_event", row.event_type_primary),
                "dominant_theme": _dominant_theme(title, row.event_type_primary),
                **{name: _term_score(title, terms) for name, terms in NARRATIVE_TERMS.items()},
                "sentiment_score": _sentiment_score(title),
                "certainty_score": float(np.clip((base_confidence + source_conf) / 2, 0, 1)),
                "surprise_score": _surprise_score(title),
                "narrative_strength": _narrative_strength(title, row.event_type_primary),
                "confidence": float(np.clip((base_confidence + source_conf) / 2, 0, 1)),
                "source_reliability_score": source_conf,
                "narrative_decay_score": _narrative_decay_score(title, row.event_type_primary),
                "model": "rules",
                "prompt_version": "kg-v2",
                "created_at_utc": row.created_at_utc,
            }
        )
    return pd.DataFrame(rows)


def _build_market_reactions(facts: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if facts.empty or prices.empty:
        return pd.DataFrame()
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    wide = prices.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index().ffill()
    rows = []
    for row in facts.itertuples():
        event_date = pd.to_datetime(row.event_date, errors="coerce")
        if pd.isna(event_date):
            continue
        symbols = [s for s in str(row.affected_tickers).split(",") if s]
        if not symbols:
            symbols = [row.entity]
        for symbol in symbols:
            if symbol not in wide:
                continue
            series = wide[symbol].dropna()
            if series.empty:
                continue
            start_pos = series.index.searchsorted(event_date)
            if start_pos >= len(series):
                continue
            start_date = series.index[start_pos]
            start_price = float(series.iloc[start_pos])
            qqq_start = _bench_start(wide, "QQQ", start_date)
            for horizon in [1, 5, 20, 60]:
                end_pos = start_pos + horizon
                payload = {
                    "event_id": row.event_id,
                    "canonical_event_id": getattr(row, "canonical_event_id", row.event_id),
                    "entity": row.entity,
                    "affected_ticker": symbol,
                    "benchmark": "QQQ",
                    "event_date": row.event_date,
                    "horizon_days": horizon,
                    "time_horizon": f"{horizon}D",
                    "start_price": start_price,
                    "start_date": start_date.date().isoformat(),
                    "reaction_available": False,
                    "validated_at_utc": "",
                }
                if end_pos < len(series):
                    end_date = series.index[end_pos]
                    end_price = float(series.iloc[end_pos])
                    payload.update(
                        {
                            "validated_at_utc": end_date.isoformat(),
                            "reaction_available": True,
                            "return": end_price / start_price - 1,
                            "volume_ratio": _volume_ratio(prices, symbol, end_date),
                            "benchmark_return": _benchmark_return(wide, "QQQ", start_date, end_date, qqq_start),
                            "relative_return": np.nan,
                            "qqq_return": np.nan,
                        }
                    )
                    if pd.notna(payload["benchmark_return"]):
                        payload["qqq_return"] = payload["benchmark_return"]
                        payload["relative_return"] = payload["return"] - payload["benchmark_return"]
                else:
                    payload.update({"return": np.nan, "volume_ratio": np.nan, "benchmark_return": np.nan, "relative_return": np.nan, "qqq_return": np.nan})
                rows.append(payload)
    reactions = pd.DataFrame(rows)
    if reactions.empty:
        return reactions
    reactions["market_impact_rank"] = reactions.apply(_market_impact_rank, axis=1)
    return reactions.sort_values(["event_date", "event_id", "horizon_days"]).reset_index(drop=True)


def _build_event_links(facts: pd.DataFrame, reactions: pd.DataFrame) -> pd.DataFrame:
    if facts.empty:
        return pd.DataFrame()
    rows = []
    for fact in facts.itertuples():
        targets = [s for s in str(fact.affected_tickers).split(",") if s]
        if not targets:
            targets = [fact.entity]
        for target in targets:
            rows.append(
                {
                    "source_event_id": fact.event_id,
                    "canonical_event_id": getattr(fact, "canonical_event_id", fact.event_id),
                    "target_entity": target,
                    "relationship": f"{fact.event_type_primary}_affects",
                    "confidence": float(np.clip(fact.confidence, 0, 1)),
                    "causal_confidence_score": _causal_confidence_score(fact, reactions),
                    "evidence_count": int((reactions["event_id"] == fact.event_id).sum()) if not reactions.empty else 0,
                    "impact_direction": fact.impact_direction,
                    "impact_score": fact.impact_score,
                    "time_horizon": "20D",
                    "created_at_utc": fact.created_at_utc,
                }
            )
    return pd.DataFrame(rows)


def _classify_event(title: str) -> tuple[str, list[str]]:
    lower = title.lower()
    primary = "product_launch"
    secondary = []
    for event_type, keywords in EVENT_TAXONOMY.items():
        if any(keyword in lower for keyword in keywords):
            if primary == "product_launch":
                primary = event_type
            elif event_type != primary and event_type not in secondary:
                secondary.append(event_type)
    if primary == "product_launch" and any(term in lower for term in POSITIVE_TERMS + NEGATIVE_TERMS):
        primary = "earnings"
    return primary, secondary[:3]


def _canonical_event_name(title: str, event_type_primary: str) -> str:
    lower = title.lower()
    if event_type_primary == "macro_policy":
        if "hawk" in lower:
            return "hawkish_policy"
        if "dove" in lower:
            return "dovish_policy"
        return "macro_policy"
    if event_type_primary == "inflation_data":
        return "inflation_print"
    if event_type_primary == "earnings":
        return "earnings_release"
    if event_type_primary == "guidance":
        return "guidance_revision"
    if event_type_primary == "offering_dilution":
        return "equity_dilution"
    if event_type_primary == "contract":
        return "commercial_contract"
    if event_type_primary == "regulation":
        return "regulatory_action"
    if event_type_primary == "geopolitical":
        return "geopolitical_shock"
    if event_type_primary == "fund_flow":
        return "fund_flow_shift"
    if event_type_primary == "analyst_rating":
        return "analyst_action"
    if event_type_primary == "insider_activity":
        return "insider_activity"
    if event_type_primary == "product_launch":
        return "product_launch"
    if event_type_primary == "supply_chain":
        return "supply_chain_disruption"
    return event_type_primary


def _dominant_theme(title: str, event_type_primary: str) -> str:
    lower = title.lower()
    if "ai" in lower or "chip" in lower or "gpu" in lower:
        return "AI / 晶片"
    if any(term in lower for term in ["fed", "rate", "inflation"]):
        return "利率 / 宏觀"
    if any(term in lower for term in ["tariff", "trade", "export control"]):
        return "貿易 / 地緣"
    if any(term in lower for term in ["earnings", "guidance", "revenue", "eps"]):
        return "財報 / 財測"
    return event_type_primary.replace("_", " ")


def _term_score(text: str, terms: list[str]) -> float:
    hits = sum(1 for term in terms if term in text)
    return float(np.clip(hits / max(len(terms), 1), 0, 1))


def _sentiment_score(text: str) -> float:
    pos = sum(term in text for term in POSITIVE_TERMS)
    neg = sum(term in text for term in NEGATIVE_TERMS)
    return float(np.clip((pos - neg + 3) / 6, 0, 1))


def _surprise_score(text: str) -> float:
    cues = ["beat", "miss", "raise", "cut", "unexpected", "surprise"]
    return float(np.clip(sum(term in text for term in cues) / len(cues), 0, 1))


def _narrative_strength(text: str, event_type_primary: str) -> float:
    base = 0.35 + 0.1 * (event_type_primary in {"earnings", "guidance", "macro_policy"})
    return float(np.clip(base + 0.3 * _surprise_score(text), 0, 1))


def _event_id(source_layer: str, title: str, timestamp: pd.Timestamp, affected: list[str]) -> str:
    stamp = timestamp.isoformat() if pd.notna(timestamp) else "unknown"
    text = "|".join([source_layer, stamp, title[:120], ",".join(affected[:4])])
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _canonical_event_id(canonical_event: str, event_type_primary: str, affected: list[str], timestamp: pd.Timestamp) -> str:
    stamp = timestamp.strftime("%Y-%m-%d") if pd.notna(timestamp) else "unknown"
    text = "|".join([canonical_event, event_type_primary, ",".join(sorted(set(affected[:6]))), stamp])
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _event_hash(canonical_event: str, event_type_primary: str, affected: list[str], source_layer: str) -> str:
    text = "|".join([canonical_event, event_type_primary, ",".join(sorted(set(affected[:6]))), source_layer])
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _affected_entities(symbol: str, title: str) -> list[str]:
    entities = []
    symbol = str(symbol or "").upper()
    if symbol:
        entities.append(symbol)
    for candidate in re.findall(r"(?<![A-Za-z])\$?([A-Z]{2,5})(?![A-Za-z])", title or ""):
        candidate = candidate.upper()
        if len(candidate) >= 2 and candidate not in entities:
            entities.append(candidate)
    return entities[:6]


def _impact_direction(title: str) -> str:
    lower = title.lower()
    if any(term in lower for term in POSITIVE_TERMS):
        return "positive"
    if any(term in lower for term in NEGATIVE_TERMS):
        return "negative"
    return "neutral"


def _impact_score(title: str) -> float:
    lower = title.lower()
    score = 0.5
    score += 0.15 * sum(term in lower for term in POSITIVE_TERMS)
    score -= 0.15 * sum(term in lower for term in NEGATIVE_TERMS)
    return float(np.clip(score, 0, 1))


def _source_domain(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    if "://" not in text:
        text = f"https://{text}"
    parsed = urlparse(text)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host or "unknown"


def _source_reliability_score(value: str) -> float:
    domain = _source_domain(value)
    for key, score in RELIABLE_SOURCES.items():
        if key in domain:
            return float(score)
    if domain == "unknown":
        return 0.55
    return 0.68


def _regime_label(regime_context: dict | None) -> str:
    if not regime_context:
        return "unknown"
    return str(regime_context.get("label") or regime_context.get("regime_label") or regime_context.get("name") or "unknown")


def _narrative_decay_score(title: str, event_type_primary: str) -> float:
    lower = title.lower()
    if event_type_primary in {"macro_policy", "inflation_data", "earnings", "guidance"}:
        return 0.35
    if any(term in lower for term in ["breaking", "urgent", "latest"]):
        return 0.15
    if any(term in lower for term in ["partnership", "contract", "product", "launch"]):
        return 0.5
    return 0.6


def _market_impact_rank(row: pd.Series) -> float:
    rank = 0.25
    rank += 0.25 * float(np.clip(row.get("impact_score", 0.5), 0, 1))
    rank += 0.2 * float(np.clip(row.get("source_reliability_score", 0.6), 0, 1))
    rank += 0.15 if pd.notna(row.get("return")) else 0.0
    rank += 0.1 if pd.notna(row.get("relative_return")) and abs(float(row.get("relative_return"))) > 0.02 else 0.0
    rank += 0.05 if pd.notna(row.get("volume_ratio")) and float(row.get("volume_ratio", 0)) > 1.5 else 0.0
    return float(np.clip(rank, 0, 1))


def _causal_confidence_score(fact: pd.Series, reactions: pd.DataFrame) -> float:
    base = float(np.clip(getattr(fact, "confidence", 0.5), 0, 1))
    source = float(np.clip(getattr(fact, "source_reliability_score", 0.6), 0, 1))
    evidence = 0.0
    if not reactions.empty:
        evidence = min(int((reactions["event_id"] == fact.event_id).sum()) / 4.0, 1.0)
    return float(np.clip(base * 0.45 + source * 0.35 + evidence * 0.2, 0, 1))


def _finalize_fact_events(facts: pd.DataFrame) -> pd.DataFrame:
    if facts.empty:
        return facts
    facts = facts.copy()
    if "canonical_event_id" not in facts:
        facts["canonical_event_id"] = facts["event_id"]
    if "event_hash" not in facts:
        facts["event_hash"] = facts["event_id"]
    facts["dedup_group_size"] = facts.groupby("canonical_event_id")["canonical_event_id"].transform("size")
    facts["event_strength"] = (
        0.4 * facts["impact_score"].fillna(0)
        + 0.3 * facts["source_reliability_score"].fillna(0.5)
        + 0.2 * facts["confidence"].fillna(0.5)
        + 0.1 * facts["dedup_group_size"].clip(1, 10) / 10.0
    ).clip(0, 1)
    facts["market_impact_rank"] = facts["event_strength"]
    facts["confidence_method"] = "source_reliability+event_signal"
    return facts


def _finalize_reactions(reactions: pd.DataFrame) -> pd.DataFrame:
    if reactions.empty:
        return reactions
    reactions = reactions.copy()
    for column in ["return", "relative_return", "qqq_return", "benchmark_return", "volume_ratio"]:
        if column in reactions:
            reactions[column] = pd.to_numeric(reactions[column], errors="coerce")
    reactions["event_strength"] = (
        0.45 * reactions["market_impact_rank"].fillna(0.25)
        + 0.35 * reactions["relative_return"].abs().fillna(0).clip(upper=0.2)
        + 0.2 * reactions["volume_ratio"].fillna(1).clip(lower=0, upper=5).div(5)
    ).clip(0, 1)
    return reactions


def _confidence_from_text(title: str) -> float:
    lower = title.lower()
    score = 0.45
    if any(term in lower for term in ["earnings", "guidance", "fed", "cpi", "contract", "tariff"]):
        score += 0.35
    if any(term in lower for term in ["rumor", "may", "could", "reportedly"]):
        score -= 0.2
    return float(np.clip(score, 0.15, 0.98))


def _bench_start(wide: pd.DataFrame, symbol: str, start_date: pd.Timestamp) -> float:
    if symbol not in wide:
        return np.nan
    series = wide[symbol].dropna()
    if series.empty:
        return np.nan
    pos = series.index.searchsorted(start_date)
    if pos >= len(series):
        return np.nan
    return float(series.iloc[pos])


def _benchmark_return(wide: pd.DataFrame, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp, start_price: float) -> float:
    if symbol not in wide or pd.isna(start_price):
        return np.nan
    series = wide[symbol].dropna()
    if series.empty:
        return np.nan
    start_pos = series.index.searchsorted(start_date)
    end_pos = series.index.searchsorted(end_date)
    if start_pos >= len(series) or end_pos >= len(series):
        return np.nan
    return float(series.iloc[end_pos] / series.iloc[start_pos] - 1)


def _volume_ratio(prices: pd.DataFrame, symbol: str, end_date: pd.Timestamp) -> float:
    data = prices[prices["symbol"].astype(str).str.upper() == symbol].copy()
    if data.empty:
        return np.nan
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.sort_values("date")
    row = data[data["date"] == end_date]
    if row.empty or "volume" not in row.columns:
        return np.nan
    idx = row.index[0]
    loc = data.index.get_loc(idx)
    if loc < 20:
        return np.nan
    current = float(row.iloc[0]["volume"])
    avg = pd.to_numeric(data.iloc[max(0, loc - 20): loc]["volume"], errors="coerce").mean()
    return float(current / avg) if avg and pd.notna(avg) else np.nan


def _latest_ts(df: pd.DataFrame) -> str:
    if df.empty:
        return "n/a"
    for column in ["timestamp_utc", "generated_at_utc", "validated_at_utc", "created_at_utc", "event_date"]:
        if column in df.columns:
            value = pd.to_datetime(df[column], errors="coerce", utc=True).max()
            if pd.notna(value):
                return value.isoformat()
    return "n/a"
