from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from .data import cache_path
from .edge import dedup_key, quality_score
from .news import rule_based_news_summary


AI_SUMMARY_CACHE = cache_path("ai_summary.json")
AI_SUMMARY_HISTORY_CACHE = cache_path("ai_summary_history.parquet")
AI_SUMMARY_PROMPT_VERSION = "2026-06-19-v3"
TAIPEI = ZoneInfo("Asia/Taipei")
PRESERVE_AI_SUMMARY_HOURS = 72


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
    fallback_text = _fallback(
        snapshot,
        anomalies,
        news,
        international_news=international_news,
        discovery_candidates=discovery_candidates,
        portfolio_impact=portfolio_impact,
    )
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
    existing = load_cached_ai_summary(path)
    if _should_preserve_existing_summary(existing, payload):
        return
    payload = _attach_edge_metadata(payload)
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


def latest_ai_history_entry(history: pd.DataFrame | None = None) -> dict:
    history = load_ai_summary_history() if history is None else history
    if history is None or history.empty:
        return {}
    frame = history.copy()
    if "generated_at_utc" in frame:
        frame["generated_at_utc"] = pd.to_datetime(frame["generated_at_utc"], errors="coerce", utc=True)
        frame = frame.sort_values("generated_at_utc")
    ai_only = frame[frame.get("used_ai", False).fillna(False).astype(bool)] if "used_ai" in frame else pd.DataFrame()
    target = ai_only if not ai_only.empty else frame
    if target.empty:
        return {}
    row = target.iloc[-1].to_dict()
    value = row.get("generated_at_utc")
    if isinstance(value, pd.Timestamp):
        row["generated_at_utc"] = value.isoformat()
    return row


def load_cached_ai_summary(path=None) -> dict:
    path = path or AI_SUMMARY_CACHE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def openai_configuration_status() -> dict:
    api_key = _get_secret("OPENAI_API_KEY")
    model = _get_secret("OPENAI_MODEL") or "gpt-4.1-mini"
    if api_key:
        masked = f"{api_key[:3]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
        return {
            "configured": True,
            "status": "ready",
            "model": model,
            "api_key_preview": masked,
        }
    return {
        "configured": False,
        "status": "missing_key",
        "model": model,
        "api_key_preview": "n/a",
    }


def ai_summary_quality(payload: dict) -> dict:
    text = str(payload.get("text", "") or "")
    required_sections = [
        "今日市場結論",
        "與昨日相比",
        "量化訊號",
        "新聞與國際風險",
        "對持倉影響",
        "候選觀察股",
        "明日觀察重點",
        "一句話行動意義",
    ]
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


def ai_summary_edge_quality(payload: dict) -> dict:
    text = str(payload.get("text", "") or "")
    source = str(payload.get("source", "") or payload.get("provider", "") or payload.get("model", "") or "")
    tags = "；".join(
        [part for part in [payload.get("provider", ""), payload.get("model", ""), payload.get("status", "")] if part]
    )
    published = pd.to_datetime(payload.get("generated_at_utc"), errors="coerce", utc=True)
    edge = quality_score(text[:200] or payload.get("status", ""), source=source, tags=tags, published=published, raw_text=text)
    return {
        "source_domain": edge.source_domain,
        "source_reliability_score": edge.source_reliability_score,
        "text_density_score": edge.text_density_score,
        "structure_score": edge.structure_score,
        "dedup_key": edge.dedup_key,
        "quality_score_edge": edge.quality_score,
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


def _should_preserve_existing_summary(existing: dict, incoming: dict) -> bool:
    if not existing:
        return False
    if bool(incoming.get("used_ai")):
        return False
    if not bool(existing.get("used_ai")):
        return False
    existing_ts = pd.to_datetime(existing.get("generated_at_utc"), errors="coerce", utc=True)
    incoming_ts = pd.to_datetime(incoming.get("generated_at_utc"), errors="coerce", utc=True)
    if pd.isna(existing_ts) or pd.isna(incoming_ts):
        return False
    existing_local = existing_ts.tz_convert(TAIPEI).date()
    incoming_local = incoming_ts.tz_convert(TAIPEI).date()
    if existing_local >= incoming_local:
        return True
    age_hours = (incoming_ts - existing_ts).total_seconds() / 3600
    return age_hours <= PRESERVE_AI_SUMMARY_HOURS


def _attach_edge_metadata(payload: dict) -> dict:
    payload = dict(payload)
    edge_quality = ai_summary_edge_quality(payload)
    payload.update(edge_quality)
    summary_quality = ai_summary_quality(payload)
    payload["quality_score"] = summary_quality.get("quality_score")
    payload["quality_label"] = summary_quality.get("quality_label")
    payload["history_dedup_key"] = dedup_key(payload.get("generated_at_utc", ""), payload.get("model", ""), payload.get("provider", ""))
    return payload


def _summary_history_row(payload: dict) -> dict:
    generated = pd.to_datetime(payload.get("generated_at_utc"), errors="coerce", utc=True)
    if pd.isna(generated):
        generated = pd.Timestamp.now(tz="UTC")
    quality = ai_summary_quality(payload)
    edge_quality = ai_summary_edge_quality(payload)
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
        "source_domain": edge_quality.get("source_domain"),
        "source_reliability_score": edge_quality.get("source_reliability_score"),
        "text_density_score": edge_quality.get("text_density_score"),
        "structure_score": edge_quality.get("structure_score"),
        "dedup_key": edge_quality.get("dedup_key"),
        "quality_score_edge": edge_quality.get("quality_score_edge"),
        "text": text,
    }


def _fallback(
    snapshot: pd.DataFrame,
    anomalies: pd.DataFrame,
    news: pd.DataFrame,
    international_news: pd.DataFrame | None = None,
    discovery_candidates: pd.DataFrame | None = None,
    portfolio_impact: pd.DataFrame | None = None,
) -> str:
    lines: list[str] = []
    market_view, confidence = _fallback_market_view(snapshot, anomalies)
    qqq_note = _fallback_symbol_note(snapshot, "QQQ")
    smh_note = _fallback_symbol_note(snapshot, "SMH")
    vix_note = _fallback_symbol_note(snapshot, "^VIX")

    lines.append("## 今日市場結論")
    lines.append(f"目前市場判讀偏向{market_view}，信心{confidence}。")
    notes = [note for note in [qqq_note, smh_note, vix_note] if note]
    if notes:
        lines.append("；".join(notes[:3]) + "。")
    else:
        lines.append("今天以量價與均線結構作為主要判讀依據。")
    lines.append("")

    lines.append("## 與昨日相比")
    for bullet in _fallback_change_lines(snapshot, anomalies, international_news):
        lines.append(f"- {bullet}")
    lines.append("")

    lines.append("## 量化訊號")
    if not anomalies.empty:
        top = anomalies[["symbol", "flags"]].head(5)
        for row in top.itertuples():
            lines.append(f"- {row.symbol}：{row.flags}")
    else:
        lines.append("- 今日未偵測到重大量價異常。")
    lines.append("")

    lines.append("## 新聞與國際風險")
    news_summary = rule_based_news_summary(news)
    lines.append(news_summary if news_summary else "近期沒有足夠新聞可供整理。")
    intl_lines = _fallback_international_lines(international_news)
    if intl_lines:
        lines.extend(intl_lines)
    lines.append("")

    lines.append("## 對持倉影響")
    impact_lines = _fallback_portfolio_impact_lines(portfolio_impact)
    lines.extend(impact_lines)
    lines.append("")

    lines.append("## 候選觀察股")
    candidate_lines = _fallback_candidate_lines(discovery_candidates)
    lines.extend(candidate_lines)
    lines.append("")

    lines.append("## 明日觀察重點")
    for bullet in _fallback_watchlist(snapshot, anomalies, international_news):
        lines.append(f"- {bullet}")
    lines.append("")
    lines.append("## 一句話行動意義")
    lines.append(_fallback_action_line(snapshot, anomalies, international_news))
    return "\n".join(lines)


def _fallback_market_view(snapshot: pd.DataFrame, anomalies: pd.DataFrame) -> tuple[str, str]:
    if snapshot.empty:
        return "觀望", "低"
    score = 0
    qqq = _snapshot_row(snapshot, "QQQ")
    smh = _snapshot_row(snapshot, "SMH")
    vix = _snapshot_row(snapshot, "^VIX")
    for row in [qqq, smh]:
        if row is not None:
            if _num(row.get("ret_20d")) > 0:
                score += 1
            if _num(row.get("dist_ma_50")) > 0:
                score += 1
            if _num(row.get("dist_ma_200")) > 0:
                score += 1
    if vix is not None and _num(vix.get("ret_1d")) > 0:
        score -= 1
    if not anomalies.empty and len(anomalies) >= 6:
        score -= 1
    if score >= 4:
        return "偏多", "中"
    if score >= 2:
        return "觀望偏多", "中低"
    if score >= 0:
        return "觀望", "中低"
    return "風險升溫", "中"


def _fallback_symbol_note(snapshot: pd.DataFrame, symbol: str) -> str:
    row = _snapshot_row(snapshot, symbol)
    if row is None:
        return ""
    ret_1d = _num(row.get("ret_1d")) * 100
    ret_20d = _num(row.get("ret_20d")) * 100
    dist_50 = _num(row.get("dist_ma_50")) * 100
    if symbol == "^VIX":
        return f"VIX 單日變動 {ret_1d:.1f}%，20 日位置對風險偏好仍有影響"
    return f"{symbol} 近 20 日報酬 {ret_20d:.1f}%，相對 50DMA {dist_50:.1f}%"


def _fallback_international_lines(international_news: pd.DataFrame | None) -> list[str]:
    if international_news is None or international_news.empty:
        return ["- 目前沒有新增的國際重大風險新聞。"]
    data = international_news.copy()
    if "is_major" in data.columns:
        major = data[data["is_major"].fillna(False)]
    else:
        major = pd.DataFrame()
    if major.empty:
        major = data.head(2)
    lines = []
    for row in major.head(2).itertuples():
        source = f"（{row.source}）" if getattr(row, "source", "") else ""
        lines.append(f"- {row.title}{source}")
    return lines


def _fallback_portfolio_impact_lines(portfolio_impact: pd.DataFrame | None) -> list[str]:
    if portfolio_impact is None or portfolio_impact.empty:
        return ["- 目前沒有命中持倉的新增新聞，先以市場整體風險與趨勢判讀為主。"]
    lines = []
    for row in portfolio_impact.head(4).itertuples():
        impact = getattr(row, "impact", "") or getattr(row, "impact_level", "中性")
        headline = getattr(row, "sample_headline", "")
        lines.append(f"- {row.ticker}：{impact}；{headline}")
    return lines


def _fallback_candidate_lines(discovery_candidates: pd.DataFrame | None) -> list[str]:
    if discovery_candidates is None or discovery_candidates.empty:
        return ["- 今日沒有新的高信心候選觀察股，先保留名單並觀察是否有新催化。"]
    label_col = "candidate_label" if "candidate_label" in discovery_candidates.columns else None
    reason_col = "observation_reason" if "observation_reason" in discovery_candidates.columns else None
    lines = []
    for row in discovery_candidates.head(5).itertuples():
        label = getattr(row, label_col, "") if label_col else ""
        reason = getattr(row, reason_col, "") if reason_col else ""
        prefix = f"{row.ticker}"
        if label:
            prefix += f"（{label}）"
        lines.append(f"- {prefix}：{reason or '具備持續觀察價值，但仍需等進一步驗證。'}")
    return lines


def _fallback_watchlist(
    snapshot: pd.DataFrame,
    anomalies: pd.DataFrame,
    international_news: pd.DataFrame | None,
) -> list[str]:
    bullets: list[str] = []
    qqq = _snapshot_row(snapshot, "QQQ")
    if qqq is not None:
        bullets.append(
            f"觀察 QQQ 是否能守住 50DMA 附近，近 20 日報酬目前為 {_num(qqq.get('ret_20d')) * 100:.1f}%"
        )
    if not anomalies.empty:
        top_symbols = "、".join(anomalies["symbol"].astype(str).head(3).tolist())
        bullets.append(f"留意 {top_symbols} 的異常是否延續到下一個交易日")
    if international_news is not None and not international_news.empty:
        bullets.append("追蹤國際新聞是否從單一事件擴散到利率、貿易或風險資產")
    if len(bullets) < 3:
        bullets.append("確認半導體與大型科技是否維持相對強弱結構")
    return bullets[:3]


def _fallback_change_lines(
    snapshot: pd.DataFrame,
    anomalies: pd.DataFrame,
    international_news: pd.DataFrame | None,
) -> list[str]:
    bullets: list[str] = []
    qqq = _snapshot_row(snapshot, "QQQ")
    smh = _snapshot_row(snapshot, "SMH")
    vix = _snapshot_row(snapshot, "^VIX")
    if qqq is not None:
        bullets.append(f"QQQ 單日變動 {_num(qqq.get('ret_1d')) * 100:.1f}%，20 日報酬 {_num(qqq.get('ret_20d')) * 100:.1f}%")
    if smh is not None:
        bullets.append(f"SMH 相對 50DMA {_num(smh.get('dist_ma_50')) * 100:.1f}%，反映半導體相對強弱變化")
    if vix is not None:
        bullets.append(f"VIX 單日變動 {_num(vix.get('ret_1d')) * 100:.1f}%，可用來觀察風險偏好是否升降")
    if not anomalies.empty:
        bullets.append(f"今日異常主要集中在 {'、'.join(anomalies['symbol'].astype(str).head(3).tolist())}")
    if international_news is not None and not international_news.empty:
        bullets.append("國際新聞仍在影響市場風險評價，需確認是否擴散到利率或供應鏈")
    return bullets[:3] or ["今日與昨日相比沒有明顯結構翻轉，先聚焦量價與波動變化。"]


def _fallback_action_line(
    snapshot: pd.DataFrame,
    anomalies: pd.DataFrame,
    international_news: pd.DataFrame | None,
) -> str:
    qqq = _snapshot_row(snapshot, "QQQ")
    qqq_ret = _num(qqq.get("ret_1d")) * 100 if qqq is not None else 0.0
    anomaly_count = len(anomalies) if anomalies is not None else 0
    if qqq_ret > 1.5 and anomaly_count <= 3:
        return "短線仍偏多，但真正要追的是量價延續，不是只看單日上漲。"
    if qqq_ret < -1.5 or anomaly_count >= 6:
        return "今天最重要的是風險在擴散，先確認賣壓是否從個股蔓延到整體科技板塊。"
    if international_news is not None and not international_news.empty:
        return "今天不必急著改變方向，但要特別確認國際事件是否開始影響資金風險偏好。"
    return "今天的重點不是預測轉折，而是確認強弱分化是否持續。"


def _snapshot_row(snapshot: pd.DataFrame, symbol: str) -> pd.Series | None:
    if snapshot.empty or "symbol" not in snapshot.columns:
        return None
    matched = snapshot[snapshot["symbol"].astype(str).eq(symbol)]
    if matched.empty:
        return None
    return matched.iloc[0]


def _num(value) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return 0.0
    return float(numeric)


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
    previous_context = _previous_summary_context()
    market_change_rows = _market_change_rows(snapshot)
    anomaly_change_rows = _anomaly_change_rows(anomalies)
    news_focus_rows = _news_focus_rows(news, international_news, portfolio_impact)
    return f"""
你是一位謹慎、務實、資訊密度高的市場研究助理。請用繁體中文輸出每日科技股監控摘要，避免保證式預測，避免直接叫使用者買賣。

你的目標不是把新聞重寫一遍，而是回答：
1. 今天和昨天相比，真正變了什麼？
2. 哪些變化最值得投資人注意？
3. 這些變化對持倉、候選股、明日觀察有什麼意義？

請嚴格遵守以下規則：
- 不要使用空泛句子，例如「留意利率風險」「市場保持觀望」；除非後面立刻接具體證據。
- 每個段落至少要出現具體標的、數值、事件或新聞主題。
- 如果今天和昨天差異不大，要明說「差異有限」，但仍要指出最值得注意的1到2個新變化。
- 不要把長期背景重複當成今日重點，除非今天有新的證據。
- 對持倉影響請優先排序最值得注意的1到3檔，不要平均分配篇幅。

輸出格式請固定為：
## 今日市場結論
先用 2-3 句話直接說明今日市場最重要的變化、目前偏多/觀望/風險升溫/防守，以及信心等級。

## 與昨日相比
列出 3 點「今天和昨天真正不同」的地方。若差異有限，也要點出最不一樣的1到2項，不要留白。

## 量化訊號
列出 3-5 點，只保留最重要的量化證據。要連結 QQQ、SMH、VIX、均線、成交量或異常雷達，並說明這代表什麼。

## 新聞與國際風險
整理真正有增量的消息面線索，特別留意戰爭、貿易談判、出口管制、利率、財報、增發與指引下修。不要把舊主題重講成新主題。

## 對持倉影響
只聚焦最值得注意的 1-3 檔持倉。要說明：為什麼是它、是價格/量能/新聞/情緒哪一類風險、目前偏短期噪音還是結構變化。

## 候選觀察股
只解讀 Top 5 候選觀察股，優先指出今天新冒出來或排序明顯前進的標的，說明值得觀察的理由與主要風險。

## 明日觀察重點
列出 3 點下一個交易日最該追蹤的指標、價位、事件或新聞驗證點。

## 一句話行動意義
用 1 句話總結：今天最值得記住的是什麼。不要空泛。

前一版摘要參考：
{previous_context}

今日市場變化重點：
{market_change_rows}

今日異常變化重點：
{anomaly_change_rows}

今日新聞焦點：
{news_focus_rows}

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
    required = ["今日市場結論", "與昨日相比", "量化訊號", "新聞與國際風險", "明日觀察重點", "一句話行動意義"]
    return all(section in text for section in required)


def _previous_summary_context() -> str:
    history = load_ai_summary_history()
    if history.empty or "text" not in history.columns:
        return "無前一日摘要可供對照。"
    frame = history.copy()
    if "generated_at_utc" in frame.columns:
        frame["generated_at_utc"] = pd.to_datetime(frame["generated_at_utc"], errors="coerce", utc=True)
        frame = frame.sort_values("generated_at_utc")
    if len(frame) < 1:
        return "無前一日摘要可供對照。"
    previous = frame.iloc[-1]
    summary_date = str(previous.get("summary_date", "n/a"))
    text = re.sub(r"\s+", " ", str(previous.get("text", "") or "")).strip()
    snippet = text[:420] + ("..." if len(text) > 420 else "")
    return f"{summary_date}：{snippet}" if snippet else f"{summary_date}：無內容"


def _market_change_rows(snapshot: pd.DataFrame) -> list[dict]:
    focus = ["QQQ", "SMH", "SOXX", "IGV", "IYW", "XLK", "^VIX", "NVDA", "AMD", "MSFT", "META", "TSLA"]
    rows: list[dict] = []
    if snapshot.empty or "symbol" not in snapshot.columns:
        return rows
    for symbol in focus:
        matched = snapshot[snapshot["symbol"].astype(str).eq(symbol)]
        if matched.empty:
            continue
        row = matched.iloc[0]
        rows.append(
            {
                "symbol": symbol,
                "ret_1d": round(_num(row.get("ret_1d")) * 100, 2),
                "ret_20d": round(_num(row.get("ret_20d")) * 100, 2),
                "dist_ma_50": round(_num(row.get("dist_ma_50")) * 100, 2),
                "dist_ma_200": round(_num(row.get("dist_ma_200")) * 100, 2),
                "volume_ratio_20d": round(_num(row.get("volume_ratio_20d")), 2),
            }
        )
    return rows


def _anomaly_change_rows(anomalies: pd.DataFrame) -> list[dict]:
    if anomalies.empty:
        return []
    cols = [col for col in ["symbol", "flags", "ret_1d", "volume_ratio_20d", "dist_ma_50", "dist_ma_200"] if col in anomalies.columns]
    return anomalies[cols].head(10).to_dict("records")


def _news_focus_rows(
    news: pd.DataFrame,
    international_news: pd.DataFrame | None,
    portfolio_impact: pd.DataFrame | None,
) -> dict:
    result: dict[str, list[dict]] = {"watchlist": [], "international": [], "portfolio": []}
    if news is not None and not news.empty:
        cols = [col for col in ["symbol", "title", "source", "tags", "published"] if col in news.columns]
        result["watchlist"] = news[cols].head(8).to_dict("records")
    if international_news is not None and not international_news.empty:
        cols = [col for col in ["title", "source", "tags", "is_major", "published"] if col in international_news.columns]
        major = international_news.copy()
        if "is_major" in major.columns:
            major = major[major["is_major"].fillna(False)]
        result["international"] = (major if not major.empty else international_news)[cols].head(5).to_dict("records")
    if portfolio_impact is not None and not portfolio_impact.empty:
        cols = [col for col in ["ticker", "impact_level", "impact", "headline_count", "sample_headline"] if col in portfolio_impact.columns]
        result["portfolio"] = portfolio_impact[cols].head(5).to_dict("records")
    return result
