from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


NEGATIVE_EVENT_KEYWORDS = [
    "dilution",
    "offering",
    "share issuance",
    "secondary offering",
    "guidance cut",
    "earnings miss",
    "misses estimates",
    "downgrade",
    "insider selling",
    "sec investigation",
    "lawsuit",
]

MAJOR_EVENT_KEYWORDS = [
    "earnings",
    "contract",
    "guidance",
    "offering",
    "share issuance",
    "partnership",
    "deal",
    "approval",
    "investigation",
]

POSITIVE_WORDS = [
    "beat",
    "beats",
    "upgrade",
    "raises",
    "raised",
    "contract",
    "partnership",
    "deal",
    "growth",
    "profit",
    "buyback",
    "ai",
]

NEGATIVE_WORDS = [
    "miss",
    "misses",
    "cut",
    "cuts",
    "downgrade",
    "offering",
    "dilution",
    "lawsuit",
    "probe",
    "investigation",
    "loss",
    "falls",
    "slumps",
]


@dataclass(frozen=True)
class PortfolioConfig:
    positions: pd.DataFrame
    cash_usd: float
    max_position_weight: float
    refresh_seconds: int
    password: str


def load_portfolio_config(secrets: Any) -> PortfolioConfig | None:
    portfolio = _section(secrets, "portfolio")
    if not portfolio:
        return None

    csv_text = str(portfolio.get("positions_csv", "") or "").strip()
    positions = parse_positions_csv(csv_text)
    cash_usd = _float(portfolio.get("cash_usd"), 0.0)
    max_weight = _float(portfolio.get("max_position_weight"), 0.15)
    refresh_seconds = int(_float(portfolio.get("refresh_seconds"), 900))
    password = str(portfolio.get("password", "") or "")
    return PortfolioConfig(
        positions=positions,
        cash_usd=max(cash_usd, 0.0),
        max_position_weight=float(np.clip(max_weight, 0.01, 1.0)),
        refresh_seconds=int(np.clip(refresh_seconds, 300, 3600)),
        password=password,
    )


def parse_positions_csv(csv_text: str) -> pd.DataFrame:
    columns = ["ticker", "shares", "avg_cost", "market_value_usd"]
    if not csv_text:
        return pd.DataFrame(columns=columns)

    data = pd.read_csv(StringIO(csv_text), comment="#")
    data.columns = [str(col).strip().lower() for col in data.columns]
    if "ticker" not in data:
        return pd.DataFrame(columns=columns)

    data["ticker"] = data["ticker"].astype(str).str.strip().str.upper()
    for col in ["shares", "avg_cost", "market_value_usd"]:
        if col not in data:
            data[col] = np.nan
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data[data["ticker"].ne("")]
    return data[columns].reset_index(drop=True)


def fetch_portfolio_prices(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    ticker_list = list(dict.fromkeys(ticker for ticker in tickers if ticker))
    if not ticker_list:
        return pd.DataFrame()

    raw = yf.download(
        ticker_list,
        period=period,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    return _normalize_yfinance(raw, ticker_list)


def build_portfolio_view(
    positions: pd.DataFrame,
    history: pd.DataFrame,
    news: pd.DataFrame,
    cash_usd: float = 0.0,
    max_position_weight: float = 0.15,
    market_context: dict | None = None,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    if positions.empty:
        return pd.DataFrame(), _empty_summary(cash_usd), pd.DataFrame()

    latest = _latest_metrics(history)
    news_metrics = _news_metrics(news)
    rows = []
    for position in positions.itertuples(index=False):
        ticker = position.ticker
        metrics = latest.get(ticker, {})
        price = _clean_number(metrics.get("current_price"))
        shares = _clean_number(position.shares)
        market_value_hint = _clean_number(position.market_value_usd)
        if pd.isna(shares) and pd.notna(market_value_hint) and pd.notna(price) and price > 0:
            shares = market_value_hint / price

        avg_cost = _clean_number(position.avg_cost)
        market_value = shares * price if pd.notna(shares) and pd.notna(price) else market_value_hint
        invested_cost = shares * avg_cost if pd.notna(shares) and pd.notna(avg_cost) else np.nan
        if pd.isna(invested_cost) and pd.notna(market_value_hint):
            invested_cost = market_value_hint
        unrealized_pnl = market_value - invested_cost if pd.notna(market_value) and pd.notna(invested_cost) else np.nan
        unrealized_return = (price - avg_cost) / avg_cost if pd.notna(price) and pd.notna(avg_cost) and avg_cost > 0 else np.nan

        item_news = news_metrics.get(ticker, {})
        heat_score = market_heat_score(metrics, item_news)
        risk_score = position_risk_score(
            metrics=metrics,
            news_metrics=item_news,
            heat_score=heat_score,
            unrealized_return=unrealized_return,
            position_weight=np.nan,
            max_position_weight=max_position_weight,
        )
        rows.append(
            {
                "ticker": ticker,
                "shares": shares,
                "avg_cost": avg_cost,
                "current_price": price,
                "market_value": market_value,
                "invested_cost": invested_cost,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_return": unrealized_return,
                "position_weight": np.nan,
                "market_heat_score": heat_score,
                "market_heat_label": heat_label(heat_score),
                "risk_score": risk_score,
                "ret_1d": metrics.get("ret_1d", np.nan),
                "ret_5d": metrics.get("ret_5d", np.nan),
                "ret_20d": metrics.get("ret_20d", np.nan),
                "volume_ratio_20d": metrics.get("volume_ratio_20d", np.nan),
                "dist_ma_20": metrics.get("dist_ma_20", np.nan),
                "dist_ma_50": metrics.get("dist_ma_50", np.nan),
                "news_count": item_news.get("news_count", 0),
                "news_sentiment": item_news.get("news_sentiment", 0.0),
                "major_event": item_news.get("major_event", False),
                "negative_event": item_news.get("negative_event", False),
                "negative_keywords": item_news.get("negative_keywords", ""),
            }
        )

    view = pd.DataFrame(rows)
    total_market_value = float(view["market_value"].dropna().sum())
    total_assets = total_market_value + cash_usd
    if total_assets > 0:
        view["position_weight"] = view["market_value"] / total_assets
    else:
        view["position_weight"] = np.nan

    view["risk_score"] = view.apply(
        lambda row: position_risk_score(
            metrics=row.to_dict(),
            news_metrics=row.to_dict(),
            heat_score=row["market_heat_score"],
            unrealized_return=row["unrealized_return"],
            position_weight=row["position_weight"],
            max_position_weight=max_position_weight,
        ),
        axis=1,
    )
    market_context = market_context or {}
    suggestions = view.apply(
        lambda row: recommendation(row.to_dict(), max_position_weight, market_context),
        axis=1,
        result_type="expand",
    )
    view = pd.concat([view, suggestions], axis=1)
    alerts = risk_alerts(view, max_position_weight)
    return view.sort_values("market_value", ascending=False), summary_metrics(view, cash_usd), alerts


def market_heat_score(metrics: dict, news_metrics: dict) -> float:
    ret_1d = _safe(metrics.get("ret_1d"))
    ret_5d = _safe(metrics.get("ret_5d"))
    ret_20d = _safe(metrics.get("ret_20d"))
    volume_ratio = _safe(metrics.get("volume_ratio_20d"), default=1.0)
    news_count = _safe(news_metrics.get("news_count"), default=0.0)
    sentiment = _safe(news_metrics.get("news_sentiment"), default=0.0)

    score = 50
    score += np.clip(ret_1d * 260, -18, 18)
    score += np.clip(ret_5d * 130, -16, 16)
    score += np.clip(ret_20d * 75, -18, 18)
    score += np.clip((volume_ratio - 1) * 12, -8, 14)
    score += np.clip(news_count * 2.0, 0, 10)
    score += np.clip(sentiment * 10, -10, 10)
    if news_metrics.get("major_event"):
        score += 5
    if news_metrics.get("negative_event"):
        score -= 10
    return float(np.clip(score, 0, 100))


def heat_label(score: float) -> str:
    if score >= 80:
        return "過熱"
    if score >= 60:
        return "偏熱"
    if score >= 40:
        return "正常"
    if score >= 20:
        return "偏冷"
    return "弱勢"


def position_risk_score(
    metrics: dict,
    news_metrics: dict,
    heat_score: float,
    unrealized_return: float,
    position_weight: float,
    max_position_weight: float,
) -> float:
    score = 20
    if pd.notna(position_weight):
        score += np.clip((position_weight - max_position_weight) * 220, 0, 25)
    if pd.notna(unrealized_return) and unrealized_return <= -0.1:
        score += min(abs(unrealized_return) * 150, 25)
    if heat_score >= 85:
        score += 10
    if _safe(metrics.get("ret_1d")) <= -0.08:
        score += 18
    if _safe(metrics.get("ret_5d")) <= -0.12:
        score += 10
    if _safe(metrics.get("dist_ma_20")) < 0:
        score += 5
    if _safe(metrics.get("dist_ma_50")) < 0:
        score += 8
    if _safe(metrics.get("volume_ratio_20d"), default=1.0) >= 1.8 and _safe(metrics.get("ret_1d")) < 0:
        score += 10
    if news_metrics.get("negative_event"):
        score += 18
    return float(np.clip(score, 0, 100))


def recommendation(row: dict, max_position_weight: float, market_context: dict | None = None) -> dict:
    market_context = market_context or {}
    price = _clean_number(row.get("current_price"))
    avg_cost = _clean_number(row.get("avg_cost"))
    heat = _safe(row.get("market_heat_score"), default=50)
    weight = _safe(row.get("position_weight"), default=0)
    ret_5d = _safe(row.get("ret_5d"))
    ret_20d = _safe(row.get("ret_20d"))
    dist_ma_20 = _safe(row.get("dist_ma_20"))
    dist_ma_50 = _safe(row.get("dist_ma_50"))
    volume_ratio = _safe(row.get("volume_ratio_20d"), default=1)
    unrealized_return = _clean_number(row.get("unrealized_return"))
    negative_event = bool(row.get("negative_event"))
    qqq_strong = bool(market_context.get("qqq_strong", False))
    market_risk = str(market_context.get("risk_label", "觀望"))

    stop_loss = price * 0.9 if pd.isna(avg_cost) else min(avg_cost * 0.88, price * 0.93)
    add_price = price * 0.98 if pd.notna(price) else np.nan
    trim_price = price * 1.08 if pd.notna(price) else np.nan

    severe_breakdown = (
        pd.notna(unrealized_return)
        and unrealized_return <= -0.12
        and dist_ma_20 < 0
        and dist_ma_50 < 0
        and (volume_ratio >= 1.5 or negative_event)
    )
    risk_control = (
        negative_event
        or (dist_ma_20 < 0 and dist_ma_50 < 0 and volume_ratio >= 1.5 and _safe(row.get("ret_1d")) < 0)
        or (pd.notna(unrealized_return) and unrealized_return <= -0.15)
    )
    overheated = heat > 85 or ret_5d > 0.18
    overweight = weight > max_position_weight
    uptrend = ret_20d > 0 and dist_ma_20 > 0 and dist_ma_50 > -0.02

    if severe_breakdown:
        state = "風險升溫"
        action = "停損觀察"
        intensity = "高"
        reason = "虧損較深且同時跌破短中期均線，若放量或負面消息延續，需要優先控管風險。"
    elif risk_control or (overweight and heat >= 75):
        state = "偏防守"
        action = "風險控管"
        intensity = "高" if negative_event or overweight else "中"
        reason = "風險事件、放量轉弱或持倉占比偏高，適合先控制倉位，不急著加碼。"
    elif overheated:
        state = "偏熱但趨勢強" if uptrend or qqq_strong else "短線過熱"
        action = "續抱觀察"
        intensity = "中"
        reason = "短線熱度偏高，若已有獲利可觀察分批鎖利，但大盤未轉弱前不必視為停損訊號。"
    elif uptrend and 45 <= heat <= 78 and not negative_event and weight < max_position_weight:
        state = "偏強"
        action = "可分批加倉"
        intensity = "低" if market_risk in {"風險升溫", "防守"} else "中"
        reason = "趨勢向上且未明顯過熱，持倉比例仍低於上限，適合用分批方式等待合理價格。"
    elif pd.notna(avg_cost) and pd.notna(price) and abs(price / avg_cost - 1) <= 0.06 and 35 <= heat <= 75:
        state = "正常"
        action = "中性觀望"
        intensity = "低"
        reason = "股價接近成本、熱度正常，先等突破或跌破訊號確認。"
    elif pd.notna(unrealized_return) and unrealized_return > 0.25 and qqq_strong:
        state = "偏強"
        action = "穩健持有"
        intensity = "低"
        reason = "已有明顯獲利且 QQQ 仍偏強，重點是避免追高並持續追蹤熱度。"
    else:
        state = "偏弱" if ret_20d < 0 or dist_ma_50 < 0 else "正常"
        action = "續抱觀察" if state == "正常" else "等訊號確認"
        intensity = "中" if state == "偏弱" else "低"
        reason = "目前訊號未達明確調整條件，維持追蹤，等待趨勢、量能或消息面提供更清楚方向。"

    market_link = _market_link_text(row, market_context)

    return {
        "position_state": state,
        "suggestion": action,
        "suggestion_intensity": intensity,
        "suggestion_reason": reason,
        "market_link": market_link,
        "add_price": add_price,
        "trim_price": trim_price,
        "stop_loss_price": stop_loss,
    }


def summary_metrics(view: pd.DataFrame, cash_usd: float) -> dict:
    if view.empty:
        return _empty_summary(cash_usd)
    total_market_value = float(view["market_value"].dropna().sum())
    total_cost = float(view["invested_cost"].dropna().sum()) if view["invested_cost"].notna().any() else np.nan
    total_pnl = total_market_value - total_cost if pd.notna(total_cost) else np.nan
    total_return = total_pnl / total_cost if pd.notna(total_pnl) and total_cost > 0 else np.nan
    largest = view.sort_values("market_value", ascending=False).iloc[0]
    riskiest = view.sort_values("risk_score", ascending=False).iloc[0]
    action_count = (
        int(view["suggestion_intensity"].isin(["中", "高"]).sum())
        if "suggestion_intensity" in view
        else 0
    )
    return {
        "total_market_value": total_market_value,
        "cash_usd": cash_usd,
        "total_assets": total_market_value + cash_usd,
        "total_cost": total_cost,
        "total_pnl": total_pnl,
        "total_return": total_return,
        "largest_position": largest["ticker"],
        "riskiest_position": riskiest["ticker"],
        "action_count": action_count,
    }


def risk_alerts(view: pd.DataFrame, max_position_weight: float) -> pd.DataFrame:
    rows = []
    for row in view.to_dict("records"):
        reasons = []
        if pd.notna(row.get("unrealized_return")) and row["unrealized_return"] <= -0.1:
            reasons.append("單檔虧損超過 10%")
        if pd.notna(row.get("position_weight")) and row["position_weight"] > max_position_weight:
            reasons.append(f"持倉超過總資產 {max_position_weight:.0%}")
        if row.get("market_heat_score", 0) > 85:
            reasons.append("市場熱度超過 85")
        if row.get("ret_1d", 0) <= -0.08:
            reasons.append("今日跌幅超過 8%")
        if row.get("negative_event"):
            reasons.append(f"負面關鍵字：{row.get('negative_keywords', '')}")
        if reasons:
            rows.append({"ticker": row["ticker"], "alerts": "；".join(reasons), "risk_score": row.get("risk_score", np.nan)})
    return pd.DataFrame(rows)


def _latest_metrics(history: pd.DataFrame) -> dict[str, dict]:
    rows = {}
    if history.empty:
        return rows
    for ticker, data in history.groupby("ticker", sort=False):
        df = data.sort_values("date").copy()
        df["ret_1d"] = df["close"].pct_change()
        df["ret_5d"] = df["close"].pct_change(5)
        df["ret_20d"] = df["close"].pct_change(20)
        for window in [20, 50]:
            df[f"ma_{window}"] = df["close"].rolling(window).mean()
            df[f"dist_ma_{window}"] = df["close"] / df[f"ma_{window}"] - 1
        df["volume_ratio_20d"] = df["volume"] / df["volume"].rolling(20).mean()
        latest = df.dropna(subset=["close"]).tail(1)
        if latest.empty:
            continue
        item = latest.iloc[0].to_dict()
        item["current_price"] = item.get("close")
        rows[ticker] = item
    return rows


def _news_metrics(news: pd.DataFrame) -> dict[str, dict]:
    metrics = {}
    if news.empty:
        return metrics
    for ticker, data in news.groupby("symbol", sort=False):
        titles = " ".join(data["title"].dropna().astype(str).tolist()).lower()
        pos = sum(titles.count(word) for word in POSITIVE_WORDS)
        neg = sum(titles.count(word) for word in NEGATIVE_WORDS)
        negative_hits = [word for word in NEGATIVE_EVENT_KEYWORDS if word in titles]
        metrics[ticker] = {
            "news_count": int(len(data)),
            "news_sentiment": float(np.clip((pos - neg) / max(pos + neg, 1), -1, 1)),
            "major_event": any(word in titles for word in MAJOR_EVENT_KEYWORDS),
            "negative_event": bool(negative_hits),
            "negative_keywords": ", ".join(negative_hits[:5]),
        }
    return metrics


def _normalize_yfinance(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])

    frames = []
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        level1 = set(raw.columns.get_level_values(1))
        for ticker in tickers:
            if ticker in level0:
                sub = raw[ticker].copy()
            elif ticker in level1:
                sub = raw.xs(ticker, axis=1, level=1, drop_level=True).copy()
            else:
                continue
            sub["ticker"] = ticker
            frames.append(sub.reset_index())
    else:
        sub = raw.copy()
        sub["ticker"] = tickers[0]
        frames.append(sub.reset_index())

    data = pd.concat(frames, ignore_index=True)
    data.columns = [str(col).lower().replace(" ", "_") for col in data.columns]
    if "datetime" in data and "date" not in data:
        data = data.rename(columns={"datetime": "date"})
    keep = [col for col in ["date", "ticker", "open", "high", "low", "close", "volume"] if col in data]
    data = data[keep].copy()
    data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in data:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.dropna(subset=["date", "ticker", "close"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def _section(secrets: Any, name: str) -> dict:
    try:
        section = secrets.get(name, {})
    except Exception:
        return {}
    return dict(section) if section else {}


def _float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe(value: Any, default: float = 0.0) -> float:
    value = _clean_number(value)
    return float(value) if pd.notna(value) else default


def _clean_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _empty_summary(cash_usd: float) -> dict:
    return {
        "total_market_value": 0.0,
        "cash_usd": cash_usd,
        "total_assets": cash_usd,
        "total_cost": np.nan,
        "total_pnl": np.nan,
        "total_return": np.nan,
        "largest_position": "n/a",
        "riskiest_position": "n/a",
        "action_count": 0,
    }


def _market_link_text(row: dict, market_context: dict) -> str:
    label = str(market_context.get("risk_label", "觀望"))
    qqq_strong = bool(market_context.get("qqq_strong", False))
    heat = _safe(row.get("market_heat_score"), default=50)
    ticker = row.get("ticker", "")
    if heat >= 80 and qqq_strong:
        return f"{ticker} 偏熱，但 QQQ 仍強，偏向續抱觀察而非立即防守。"
    if heat >= 80 and not qqq_strong:
        return f"{ticker} 偏熱且大盤不強，追高風險較高。"
    if label in {"風險升溫", "防守"}:
        return "大盤風險偏高，個股加倉訊號需更保守解讀。"
    return "大盤環境未明顯拖累，主要依個股趨勢與持倉比例判斷。"
