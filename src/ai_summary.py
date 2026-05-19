from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from .data import cache_path
from .news import rule_based_news_summary


AI_SUMMARY_CACHE = cache_path("ai_summary.json")
AI_SUMMARY_HISTORY_CACHE = cache_path("ai_summary_history.parquet")
AI_SUMMARY_PROMPT_VERSION = "2026-05-19-v1"


def build_openai_summary(
    snapshot: pd.DataFrame,
    anomalies: pd.DataFrame,
    news: pd.DataFrame,
    international_news: pd.DataFrame | None = None,
    discovery_candidates: pd.DataFrame | None = None,
    portfolio_impact: pd.DataFrame | None = None,
) -> dict:
    api_key = _get_secret("OPENAI_API_KEY")
    model = _get_secret("OPENAI_MODEL") or "gpt-4.1-mini"
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    fallback_text = _fallback(snapshot, anomalies, news)
    if not api_key:
        return _summary_payload(
            text=fallback_text,
            provider="rules",
            model="rule_based",
            generated_at=generated_at,
            used_ai=False,
            status="尚未設定 OPENAI_API_KEY，已使用規則摘要。",
        )

    try:
        prompt = _build_summary_prompt(snapshot, anomalies, news, international_news, discovery_candidates, portfolio_impact)
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "input": prompt,
                "temperature": 0.25,
                "max_output_tokens": 1200,
            },
            timeout=45,
        )
        response.raise_for_status()
        text = _extract_openai_text(response.json())
        if _is_complete_summary(text):
            return _summary_payload(
                text=text,
                provider="openai",
                model=model,
                generated_at=generated_at,
                used_ai=True,
                status="OpenAI AI 摘要已成功產生。",
            )
        return _summary_payload(
            text=fallback_text,
            provider="rules",
            model="rule_based",
            generated_at=generated_at,
            used_ai=False,
            status="OpenAI 回覆過短或章節不完整，已改用規則摘要。",
        )
    except Exception as exc:
        return _summary_payload(
            text=fallback_text,
            provider="rules",
            model="rule_based",
            generated_at=generated_at,
            used_ai=False,
            status=f"OpenAI 摘要失敗，已改用規則摘要：{_safe_error(exc)}",
        )


def save_ai_summary(payload: dict, path=None) -> None:
    path = path or AI_SUMMARY_CACHE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    if path == AI_SUMMARY_CACHE:
        append_ai_summary_history(payload)


def append_ai_summary_history(payload: dict, path=None) -> pd.DataFrame:
    path = path or AI_SUMMARY_HISTORY_CACHE
    path.parent.mkdir(parents=True, exist_ok=True)
    history = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    row = _summary_history_row(payload)
    if not history.empty and "summary_date" in history:
        history = history[history["summary_date"].astype(str) != row["summary_date"]]
    history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    history = history.sort_values("generated_at_utc").reset_index(drop=True)
    history.to_parquet(path, index=False)
    return history


def load_ai_summary_history(path=None) -> pd.DataFrame:
    path = path or AI_SUMMARY_HISTORY_CACHE
    if not path.exists():
        return pd.DataFrame()
    history = pd.read_parquet(path)
    if "generated_at_utc" in history:
        history["generated_at_utc"] = pd.to_datetime(history["generated_at_utc"], errors="coerce", utc=True)
    return history


def load_cached_ai_summary(path=None) -> dict:
    path = path or AI_SUMMARY_CACHE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def ai_summary_quality(payload: dict) -> dict:
    text = str(payload.get("text", "") or "")
    required_sections = ["今日市場結論", "量化訊號", "新聞與國際風險", "對持倉影響", "候選觀察股", "明日觀察重點"]
    present = [section for section in required_sections if section in text]
    word_count = len(text)
    used_ai = bool(payload.get("used_ai"))
    generated = pd.to_datetime(payload.get("generated_at_utc"), errors="coerce", utc=True)
    age_hours = np.nan
    if pd.notna(generated):
        age_hours = (pd.Timestamp.now(tz="UTC") - generated).total_seconds() / 3600
    score = 0
    score += min(word_count / 900, 1) * 30
    score += len(present) / len(required_sections) * 45
    score += 15 if used_ai else 6
    score += 10 if pd.notna(age_hours) and age_hours <= 30 else 0
    if word_count < 350:
        label = "不完整"
    elif len(present) < 4:
        label = "需檢查"
    elif not used_ai:
        label = "規則備援"
    else:
        label = "完整"
    return {
        "quality_score": float(np.clip(score, 0, 100)),
        "quality_label": label,
        "text_length": int(word_count),
        "section_count": int(len(present)),
        "required_sections": len(required_sections),
        "missing_sections": "、".join([section for section in required_sections if section not in present]) or "無",
        "age_hours": age_hours,
    }


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
        "prompt_version": AI_SUMMARY_PROMPT_VERSION,
    }


def _summary_history_row(payload: dict) -> dict:
    generated = pd.to_datetime(payload.get("generated_at_utc"), errors="coerce", utc=True)
    if pd.isna(generated):
        generated = pd.Timestamp.now(tz="UTC")
    quality = ai_summary_quality(payload)
    text = str(payload.get("text", "") or "")
    return {
        "summary_date": generated.date().isoformat(),
        "generated_at_utc": generated.isoformat(),
        "provider": str(payload.get("provider", "")),
        "model": str(payload.get("model", "")),
        "used_ai": bool(payload.get("used_ai")),
        "status": str(payload.get("status", "")),
        "quality_score": quality.get("quality_score"),
        "quality_label": quality.get("quality_label"),
        "text_length": quality.get("text_length"),
        "section_count": quality.get("section_count"),
        "missing_sections": quality.get("missing_sections"),
        "prompt_version": str(payload.get("prompt_version", AI_SUMMARY_PROMPT_VERSION)),
        "text": text,
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


def _build_summary_prompt(
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
    message = re.sub(r"key=([^&\\s]+)", "key=[REDACTED]", message)
    message = re.sub(r"AIza[0-9A-Za-z_\\-]{20,}", "[REDACTED_API_KEY]", message)
    message = re.sub(r"sk-[0-9A-Za-z_\\-]{20,}", "[REDACTED_API_KEY]", message)
    return f"{name}: {message}"


def _extract_openai_text(data: dict) -> str:
    if data.get("output_text"):
        return str(data["output_text"]).strip()
    chunks = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                chunks.append(str(content["text"]))
    return "\n".join(chunks).strip()


def _is_complete_summary(text: str) -> bool:
    if len(text.strip()) < 350:
        return False
    required = ["今日市場結論", "量化訊號", "新聞與國際風險", "明日觀察重點"]
    return all(section in text for section in required)
