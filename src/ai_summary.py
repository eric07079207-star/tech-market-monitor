from __future__ import annotations

import os

import pandas as pd

from .news import rule_based_news_summary


def build_ai_summary(snapshot: pd.DataFrame, anomalies: pd.DataFrame, news: pd.DataFrame) -> tuple[str, bool]:
    api_key = _get_secret("OPENAI_API_KEY")
    if not api_key:
        return _fallback(snapshot, anomalies, news), False

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        model = _get_secret("OPENAI_MODEL") or "gpt-4.1-mini"
        prompt = _build_prompt(snapshot, anomalies, news)
        response = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=700,
        )
        text = getattr(response, "output_text", "") or ""
        if text.strip():
            return text.strip(), True
    except Exception as exc:
        return f"{_fallback(snapshot, anomalies, news)}\n\nAI 摘要暫時無法產生：{exc}", False

    return _fallback(snapshot, anomalies, news), False


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


def _build_prompt(snapshot: pd.DataFrame, anomalies: pd.DataFrame, news: pd.DataFrame) -> str:
    market = snapshot[
        ["symbol", "ret_1d", "ret_20d", "dist_ma_50", "dist_ma_200", "volume_ratio_20d", "drawdown_52w"]
    ].dropna(how="all").to_dict("records")
    anomaly_rows = anomalies[["symbol", "flags", "ret_1d", "volume_ratio_20d"]].head(20).to_dict("records") if not anomalies.empty else []
    news_rows = news[["symbol", "title", "source", "tags"]].head(30).to_dict("records") if not news.empty else []
    return f"""
你是一位謹慎的市場研究助理。請用繁體中文輸出科技股監控摘要，避免下投資建議或保證式預測。

請分成三段：
1. 市場狀態：根據量價與趨勢資料，用 3-5 句話說明。
2. 異常訊號：列出最值得追蹤的 3-6 個訊號。
3. 消息面線索：把新聞標籤和標題歸納成可能原因，明確說這是「線索」不是因果定論。

市場快照：
{market}

異常：
{anomaly_rows}

新聞：
{news_rows}
""".strip()
