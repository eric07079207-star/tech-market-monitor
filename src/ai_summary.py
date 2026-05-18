from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd
import requests

from .data import cache_path
from .news import rule_based_news_summary


AI_SUMMARY_CACHE = cache_path("ai_summary.json")


def build_gemini_summary(
    snapshot: pd.DataFrame,
    anomalies: pd.DataFrame,
    news: pd.DataFrame,
    international_news: pd.DataFrame | None = None,
    discovery_candidates: pd.DataFrame | None = None,
    portfolio_impact: pd.DataFrame | None = None,
) -> dict:
    api_key = _get_secret("GEMINI_API_KEY")
    model = _get_secret("GEMINI_MODEL") or "gemini-2.0-flash"
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    fallback_text = _fallback(snapshot, anomalies, news)
    if not api_key:
        return _summary_payload(
            text=fallback_text,
            provider="rules",
            model="rule_based",
            generated_at=generated_at,
            used_ai=False,
            status="尚未設定 GEMINI_API_KEY，已使用規則摘要。",
        )

    try:
        prompt = _build_gemini_prompt(snapshot, anomalies, news, international_news, discovery_candidates, portfolio_impact)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        response = requests.post(
            url,
            params={"key": api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.25, "maxOutputTokens": 1800},
            },
            timeout=45,
        )
        response.raise_for_status()
        data = response.json()
        text = _extract_gemini_text(data)
        if _is_complete_summary(text):
            return _summary_payload(
                text=text,
                provider="gemini",
                model=model,
                generated_at=generated_at,
                used_ai=True,
                status="Gemini AI 摘要已成功產生。",
            )
        return _summary_payload(
            text=fallback_text,
            provider="rules",
            model="rule_based",
            generated_at=generated_at,
            used_ai=False,
            status="Gemini 回覆過短或章節不完整，已改用規則摘要。",
        )
    except Exception as exc:
        return _summary_payload(
            text=fallback_text,
            provider="rules",
            model="rule_based",
            generated_at=generated_at,
            used_ai=False,
            status=f"Gemini 摘要失敗，已改用規則摘要：{_safe_error(exc)}",
        )


def save_ai_summary(payload: dict, path=None) -> None:
    path = path or AI_SUMMARY_CACHE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def load_cached_ai_summary(path=None) -> dict:
    path = path or AI_SUMMARY_CACHE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _get_secret(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        return None
    return None


def _summary_payload(text: str, provider: str, model: str, generated_at: str, used_ai: bool, status: str) -> dict:
    return {
        "text": text,
        "provider": provider,
        "model": model,
        "generated_at_utc": generated_at,
        "used_ai": used_ai,
        "status": status,
    }


def _fallback(snapshot: pd.DataFrame, anomalies: pd.DataFrame, news: pd.DataFrame) -> str:
    lines = []
    if not anomalies.empty:
        top = anomalies[["symbol", "flags"]].head(8)
        lines.append("今日主要異常：")
        lines.extend(f"- {row.symbol}: {row.flags}" for row in top.itertuples())
    else:
        lines.append("今日未偵測到重大量價異常。")
    lines.append("")
    lines.append(rule_based_news_summary(news))
    return "\n".join(lines)


def _build_gemini_prompt(
    snapshot: pd.DataFrame,
    anomalies: pd.DataFrame,
    news: pd.DataFrame,
    international_news: pd.DataFrame | None,
    discovery_candidates: pd.DataFrame | None,
    portfolio_impact: pd.DataFrame | None,
) -> str:
    market_cols = ["symbol", "ret_1d", "ret_20d", "dist_ma_50", "dist_ma_200", "volume_ratio_20d", "drawdown_52w"]
    market = snapshot[[col for col in market_cols if col in snapshot]].dropna(how="all").to_dict("records")
    anomaly_rows = (
        anomalies[["symbol", "flags", "ret_1d", "volume_ratio_20d"]].head(18).to_dict("records")
        if not anomalies.empty else []
    )
    news_rows = news[["symbol", "title", "source", "tags"]].head(24).to_dict("records") if not news.empty else []
    intl_rows = (
        international_news[["symbol", "title", "source", "tags", "is_major"]].head(12).to_dict("records")
        if international_news is not None and not international_news.empty else []
    )
    discovery_rows = (
        discovery_candidates[["ticker", "candidate_score", "candidate_label", "observation_reason", "risk_flags", "sample_headline"]]
        .head(5)
        .to_dict("records")
        if discovery_candidates is not None and not discovery_candidates.empty else []
    )
    impact_rows = (
        portfolio_impact[["ticker", "impact_level", "impact", "headline_count", "key_tags", "sample_headline"]]
        .head(8)
        .to_dict("records")
        if portfolio_impact is not None and not portfolio_impact.empty else []
    )
    return f"""
你是一位謹慎的市場研究助理。請用繁體中文輸出每日科技股監控摘要，避免保證式預測，避免直接叫使用者買賣。

輸出格式請固定為：
## 今日市場結論
用 2-4 句話說現在偏多、觀望、風險升溫或防守，並說明信心等級。

## 量化訊號
列出 3-5 點，連結 QQQ、SMH、VIX、均線、成交量或異常雷達。

## 新聞與國際風險
整理消息面線索，特別留意戰爭、貿易談判、出口管制、利率、財報、增發與指引下修。

## 對持倉影響
根據命中的持倉新聞與市場背景，用中性語氣說需要注意哪些持倉。

## 候選觀察股
只解讀 Top 5 候選觀察股，說明為什麼值得觀察與主要風險。

## 明日觀察重點
列出 3 點下一個交易日要看的指標或事件。

市場快照：
{market}

異常訊號：
{anomaly_rows}

標的新聞：
{news_rows}

國際新聞：
{intl_rows}

持倉新聞影響：
{impact_rows}

候選觀察股 Top 5：
{discovery_rows}
""".strip()


def _safe_error(exc: Exception) -> str:
    name = exc.__class__.__name__
    message = str(exc).split("\n")[0][:160]
    return f"{name}: {message}"


def _extract_gemini_text(data: dict) -> str:
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = "\n".join(str(part.get("text", "")).strip() for part in parts if part.get("text"))
    return text.strip()


def _is_complete_summary(text: str) -> bool:
    if len(text.strip()) < 350:
        return False
    required = ["今日市場結論", "量化訊號", "新聞與國際風險", "明日觀察重點"]
    return all(section in text for section in required)
