from __future__ import annotations

import json
import os
import re
from hashlib import sha256
from datetime import datetime, timezone
from pathlib import Path
import tomllib
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from .data import cache_path
from .edge import dedup_key, quality_score
from .news import rule_based_news_summary
from .research_profile import PERSONAL_RESEARCH_PROFILE, RESEARCH_PROFILE_VERSION


AI_SUMMARY_CACHE = cache_path("ai_summary.json")
AI_SUMMARY_HISTORY_CACHE = cache_path("ai_summary_history.parquet")
AI_SUMMARY_PROMPT_VERSION = "2026-08-02-personal-v4"
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
    research_plan = _build_personal_research_plan(snapshot, anomalies, news, international_news, discovery_candidates)
    fallback_text = _fallback(
        snapshot,
        anomalies,
        news,
        international_news=international_news,
        discovery_candidates=discovery_candidates,
        portfolio_impact=portfolio_impact,
        research_plan=research_plan,
    )
    if not api_key:
        return _summary_payload(
            text=fallback_text,
            provider="rules",
            model="rule_based",
            generated_at=generated_at,
            used_ai=False,
            status="尚未設定 OPENAI_API_KEY，已使用規則摘要。",
            research_plan=research_plan,
        )

    try:
        prompt = _build_summary_prompt(snapshot, anomalies, news, international_news, discovery_candidates, portfolio_impact, research_plan)
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
                research_plan=research_plan,
            )
        return _summary_payload(
            text=fallback_text,
            provider="rules",
            model="rule_based",
            generated_at=generated_at,
            used_ai=False,
            status="OpenAI 回覆過短或章節不完整，已改用規則摘要。",
            research_plan=research_plan,
        )
    except Exception as exc:
        return _summary_payload(
            text=fallback_text,
            provider="rules",
            model="rule_based",
            generated_at=generated_at,
            used_ai=False,
            status=f"OpenAI 摘要失敗，已改用規則摘要：{_safe_error(exc)}",
            research_plan=research_plan,
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
        "今日研究結論",
        "今日優先焦點",
        "可能被忽略的訊號",
        "下一步觀察",
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
        pass
    secrets_path = Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            data = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
            if name in data and data[name] not in {None, ""}:
                return str(data[name])
        except Exception:
            return None
    return None


def _summary_payload(
    text: str,
    provider: str,
    model: str,
    generated_at: str,
    used_ai: bool,
    status: str,
    research_plan: dict | None = None,
) -> dict:
    plan = research_plan or {}
    return {
        "text": text,
        "provider": provider,
        "model": model,
        "generated_at_utc": generated_at,
        "used_ai": used_ai,
        "status": status,
        "prompt_version": AI_SUMMARY_PROMPT_VERSION,
        "research_profile_version": RESEARCH_PROFILE_VERSION,
        "research_focus": "；".join(plan.get("topics", [])),
        "change_state": plan.get("change_state", "資料同步中"),
        "input_fingerprint": plan.get("input_fingerprint", ""),
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
        "research_profile_version": str(payload.get("research_profile_version", "")),
        "research_focus": str(payload.get("research_focus", "")),
        "change_state": str(payload.get("change_state", "")),
        "input_fingerprint": str(payload.get("input_fingerprint", "")),
        "text": text,
    }


def _fallback(
    snapshot: pd.DataFrame,
    anomalies: pd.DataFrame,
    news: pd.DataFrame,
    international_news: pd.DataFrame | None = None,
    discovery_candidates: pd.DataFrame | None = None,
    portfolio_impact: pd.DataFrame | None = None,
    research_plan: dict | None = None,
) -> str:
    lines: list[str] = []
    market_view, confidence = _fallback_market_view(snapshot, anomalies)
    qqq_note = _fallback_symbol_note(snapshot, "QQQ")
    smh_note = _fallback_symbol_note(snapshot, "SMH")
    vix_note = _fallback_symbol_note(snapshot, "^VIX")

    lines.append("## 今日研究結論")
    lines.append(f"目前市場判讀偏向{market_view}，信心{confidence}。")
    notes = [note for note in [qqq_note, smh_note, vix_note] if note]
    if notes:
        lines.append("；".join(notes[:3]) + "。")
    else:
        lines.append("今天以量價與均線結構作為主要判讀依據。")
    lines.append("")

    lines.append("## 今日優先焦點")
    plan = research_plan or {}
    for topic in plan.get("topics", [])[:4]:
        lines.append(f"### {topic}")
    if plan.get("topics"):
        lines.append("以上焦點依持倉關聯、價格異常、新聞與市場結構自動排序。")
    if not anomalies.empty:
        top = anomalies[["symbol", "flags"]].head(5)
        for row in top.itertuples():
            lines.append(f"- {row.symbol}：{row.flags}")
    else:
        lines.append("- 今日未偵測到重大量價異常。")
    lines.append("")

    lines.append("## 可能被忽略的訊號")
    news_summary = rule_based_news_summary(news)
    lines.append(news_summary if news_summary else "近期沒有足夠新聞可供整理。")
    intl_lines = _fallback_international_lines(international_news)
    if intl_lines:
        lines.extend(intl_lines)
    lines.append("")

    lines.append("## 下一步觀察")
    impact_lines = _fallback_portfolio_impact_lines(portfolio_impact)
    lines.extend(impact_lines)
    lines.append("")

    candidate_lines = _fallback_candidate_lines(discovery_candidates)
    lines.extend(candidate_lines[:2])
    for bullet in _fallback_watchlist(snapshot, anomalies, international_news):
        lines.append(f"- {bullet}")
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
    research_plan: dict,
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

你的目標不是把新聞重寫一遍，而是替這位長期研究科技股、同時持有科技與成長股的投資人做每日研究排序。
優先關注標的：{PERSONAL_RESEARCH_PROFILE['portfolio_focus']}
長期研究主題：{PERSONAL_RESEARCH_PROFILE['themes']}

你要回答：今天在使用者真正關心的範圍中，最值得花時間的是什麼；沒有新增證據的主題不要硬寫。

請嚴格遵守以下規則：
- 不要使用空泛句子，例如「留意利率風險」「市場保持觀望」；除非後面立刻接具體證據。
- 每個段落至少要出現具體標的、數值、事件或新聞主題。
- 如果今天和昨天差異不大，要明說「差異有限」，但仍要指出最值得注意的1到2個新變化。
- 不要把長期背景重複當成今日重點，除非今天有新的證據。
- 對持倉影響請優先排序最值得注意的1到3檔，不要平均分配篇幅。
- 只能根據提供資料說明，不知道就直接寫資料不足，不可補想像中的事件。
- 不要為了每一天都有相同版型而提及所有持倉、所有市場指標或所有候選股。

輸出格式採「動態研究備忘錄」，必須有以下四個區塊，但第二區的子標題須依今天的焦點清單命名：
## 今日研究結論
用 2-3 句說明最重要變化、偏多/觀望/風險升溫/防守與信心來源。

## 今日優先焦點
只挑以下焦點清單的 3-4 項，按重要性排序。每項用 `### 焦點名稱` 作標題，說明新事實、量化證據、與使用者的關聯。不可為未入選主題補一段。

## 可能被忽略的訊號
只寫 1-2 項真正不同類型的訊號，例如宏觀、國際、信用壓力或候選股；若沒有，明說沒有足夠新增證據。

## 下一步觀察
列出 2-3 個可驗證的下一步：數字、事件、標的或新聞驗證點。

前一版摘要參考：
{previous_context}

今日個人研究焦點清單：
{research_plan}

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
    required = ["今日研究結論", "今日優先焦點", "可能被忽略的訊號", "下一步觀察"]
    return all(section in text for section in required)


def _build_personal_research_plan(
    snapshot: pd.DataFrame,
    anomalies: pd.DataFrame,
    news: pd.DataFrame,
    international_news: pd.DataFrame | None,
    discovery_candidates: pd.DataFrame | None,
) -> dict:
    """Rank a small, changing agenda from the user's actual research interests."""
    previous_text = _previous_summary_text().lower()
    holding_rows = _rank_focus_holdings(snapshot, anomalies, news, previous_text)
    agenda: list[dict[str, object]] = []
    if holding_rows:
        symbols = [row["symbol"] for row in holding_rows[:3]]
        agenda.append({"title": f"持倉焦點：{' / '.join(symbols)}", "why": "持倉相關的價格、異常或新增新聞優先。", "evidence": holding_rows[:3]})

    market_rows = _market_focus_evidence(snapshot, previous_text)
    if market_rows:
        agenda.append({"title": "科技市場結構：QQQ / 半導體 / 波動率", "why": "用市場結構確認個股訊號是否有大盤支持。", "evidence": market_rows})

    international_rows = _major_international_evidence(international_news, previous_text)
    if international_rows:
        agenda.append({"title": "國際與政策風險", "why": "關稅、出口管制、戰爭與利率會改變科技股估值或供應鏈條件。", "evidence": international_rows})

    candidate_rows = _candidate_evidence(discovery_candidates, previous_text)
    if candidate_rows:
        agenda.append({"title": "候選股的新線索", "why": "只保留今日候選排序或新聞理由具變化的標的。", "evidence": candidate_rows})

    # A compact agenda is deliberate: unrelated themes should not appear merely to fill a template.
    agenda = agenda[:4]
    topic_names = [str(item["title"]) for item in agenda]
    evidence_blob = json.dumps(agenda, ensure_ascii=False, default=str, sort_keys=True)
    fingerprint = sha256(evidence_blob.encode("utf-8")).hexdigest()[:16]
    prior_fingerprint = _latest_history_value("input_fingerprint")
    return {
        "profile_version": RESEARCH_PROFILE_VERSION,
        "topics": topic_names,
        "agenda": agenda,
        "change_state": "低變化日" if prior_fingerprint and prior_fingerprint == fingerprint else "有新增或重新排序的研究焦點",
        "input_fingerprint": fingerprint,
    }


def _rank_focus_holdings(snapshot: pd.DataFrame, anomalies: pd.DataFrame, news: pd.DataFrame, previous_text: str) -> list[dict[str, object]]:
    if snapshot is None or snapshot.empty or "symbol" not in snapshot:
        return []
    anomaly_symbols = set(anomalies.get("symbol", pd.Series(dtype=str)).astype(str).str.upper()) if anomalies is not None else set()
    result: list[dict[str, object]] = []
    for rank, symbol in enumerate(PERSONAL_RESEARCH_PROFILE["portfolio_focus"]):
        row = snapshot[snapshot["symbol"].astype(str).str.upper().eq(symbol)]
        if row.empty:
            continue
        item = row.iloc[0]
        news_count = int(news[news.get("symbol", pd.Series(dtype=str)).astype(str).str.upper().eq(symbol)].shape[0]) if news is not None and not news.empty else 0
        ret_1d = abs(_num(item.get("ret_1d")))
        ret_20d = abs(_num(item.get("ret_20d")))
        novel = symbol.lower() not in previous_text
        score = 3.0 + max(0, 10 - rank) * 0.08 + min(ret_1d * 100, 5) + min(ret_20d * 30, 2) + news_count * 0.45
        if symbol in anomaly_symbols:
            score += 2.0
        if novel:
            score += 0.8
        result.append({"symbol": symbol, "score": round(score, 2), "ret_1d_pct": round(_num(item.get("ret_1d")) * 100, 2), "ret_20d_pct": round(_num(item.get("ret_20d")) * 100, 2), "news_count": news_count, "anomaly": symbol in anomaly_symbols})
    return sorted(result, key=lambda item: float(item["score"]), reverse=True)


def _market_focus_evidence(snapshot: pd.DataFrame, previous_text: str) -> list[dict[str, object]]:
    if snapshot is None or snapshot.empty:
        return []
    rows: list[dict[str, object]] = []
    for symbol in PERSONAL_RESEARCH_PROFILE["market_focus"]:
        matched = snapshot[snapshot.get("symbol", pd.Series(dtype=str)).astype(str).eq(symbol)]
        if matched.empty:
            continue
        item = matched.iloc[0]
        magnitude = abs(_num(item.get("ret_1d"))) + abs(_num(item.get("ret_20d"))) * 0.2
        if magnitude >= 0.012 or symbol.lower() not in previous_text:
            rows.append({"symbol": symbol, "ret_1d_pct": round(_num(item.get("ret_1d")) * 100, 2), "ret_20d_pct": round(_num(item.get("ret_20d")) * 100, 2), "dist_ma_50_pct": round(_num(item.get("dist_ma_50")) * 100, 2)})
    return rows[:4]


def _major_international_evidence(international_news: pd.DataFrame | None, previous_text: str) -> list[dict[str, object]]:
    if international_news is None or international_news.empty:
        return []
    frame = international_news.copy()
    if "is_major" in frame:
        major = frame[frame["is_major"].fillna(False)]
        if not major.empty:
            frame = major
    rows = []
    for row in frame.head(8).itertuples():
        title = str(getattr(row, "title", ""))
        if title and (title.lower() not in previous_text or bool(getattr(row, "is_major", False))):
            rows.append({"title": title, "source": str(getattr(row, "source", "")), "tags": str(getattr(row, "tags", ""))})
    return rows[:3]


def _candidate_evidence(candidates: pd.DataFrame | None, previous_text: str) -> list[dict[str, object]]:
    if candidates is None or candidates.empty:
        return []
    rows = []
    for row in candidates.head(8).itertuples():
        ticker = str(getattr(row, "ticker", ""))
        if not ticker:
            continue
        if ticker.lower() not in previous_text or len(rows) < 2:
            rows.append({"ticker": ticker, "score": round(_num(getattr(row, "candidate_score", np.nan)), 1), "reason": str(getattr(row, "observation_reason", ""))[:180], "risk": str(getattr(row, "risk_flags", ""))[:120]})
    return rows[:3]


def _latest_history_value(column: str) -> str:
    history = load_ai_summary_history()
    if history.empty or column not in history:
        return ""
    value = history.sort_values("generated_at_utc").iloc[-1].get(column, "")
    return str(value or "")


def _previous_summary_text() -> str:
    history = load_ai_summary_history()
    if history.empty or "text" not in history:
        return ""
    return str(history.sort_values("generated_at_utc").iloc[-1].get("text", "") or "")


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
