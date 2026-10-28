from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pandas as pd

from .data import cache_path


ROOT = Path(__file__).resolve().parents[1]
API_BALANCE_CONFIG = ROOT / "data" / "api_balance_config.json"
API_USAGE_LEDGER = cache_path("api_usage_ledger.parquet")
API_BALANCE_CACHE = cache_path("api_balance.json")


def load_api_balance_config(path: Path | None = None) -> dict:
    path = path or API_BALANCE_CONFIG
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_api_usage_ledger(path: Path | None = None) -> pd.DataFrame:
    path = path or API_USAGE_LEDGER
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if "generated_at_utc" in frame:
        frame["generated_at_utc"] = pd.to_datetime(frame["generated_at_utc"], errors="coerce", utc=True)
    return frame


def append_openai_usage(response_payload: dict, model: str, generated_at_utc: str) -> dict:
    usage = response_payload.get("usage") or {}
    input_details = usage.get("input_tokens_details") or {}
    input_tokens = _integer(usage.get("input_tokens"))
    cached_input_tokens = min(_integer(input_details.get("cached_tokens")), input_tokens)
    output_tokens = _integer(usage.get("output_tokens"))
    total_tokens = _integer(usage.get("total_tokens")) or input_tokens + output_tokens
    pricing = _pricing(model)
    uncached_input_tokens = max(input_tokens - cached_input_tokens, 0)
    estimated_cost = (
        uncached_input_tokens * pricing["input_usd_per_million"]
        + cached_input_tokens * pricing["cached_input_usd_per_million"]
        + output_tokens * pricing["output_usd_per_million"]
    ) / 1_000_000
    request_id = str(response_payload.get("id") or "").strip()
    if not request_id:
        fingerprint = f"{generated_at_utc}|{model}|{input_tokens}|{output_tokens}|{total_tokens}"
        request_id = f"local-{sha256(fingerprint.encode('utf-8')).hexdigest()[:20]}"
    row = {
        "request_id": request_id,
        "generated_at_utc": generated_at_utc,
        "provider": "openai",
        "model": model,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_usd_per_million": pricing["input_usd_per_million"],
        "cached_input_usd_per_million": pricing["cached_input_usd_per_million"],
        "output_usd_per_million": pricing["output_usd_per_million"],
        "estimated_cost_usd": round(float(estimated_cost), 8),
        "usage_status": "recorded" if total_tokens > 0 else "usage_missing",
    }
    API_USAGE_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    ledger = load_api_usage_ledger()
    if ledger.empty or "request_id" not in ledger or request_id not in set(ledger["request_id"].astype(str)):
        ledger = pd.concat([ledger, pd.DataFrame([row])], ignore_index=True)
        ledger = ledger.sort_values("generated_at_utc").reset_index(drop=True)
        ledger.to_parquet(API_USAGE_LEDGER, index=False)
    snapshot = api_balance_snapshot(ledger=ledger)
    API_BALANCE_CACHE.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def api_balance_snapshot(ledger: pd.DataFrame | None = None, now_utc: pd.Timestamp | None = None) -> dict:
    config = load_api_balance_config()
    baseline_balance = _number(config.get("baseline_balance_usd"), 0.0)
    baseline_at = pd.to_datetime(config.get("baseline_at_utc"), errors="coerce", utc=True)
    now_utc = now_utc if now_utc is not None else pd.Timestamp.now(tz="UTC")
    ledger = load_api_usage_ledger() if ledger is None else ledger.copy()
    eligible = pd.DataFrame()
    if not ledger.empty:
        ledger["generated_at_utc"] = pd.to_datetime(ledger.get("generated_at_utc"), errors="coerce", utc=True)
        eligible = ledger.dropna(subset=["generated_at_utc"]).copy()
        if pd.notna(baseline_at):
            eligible = eligible[eligible["generated_at_utc"] >= baseline_at]
    costs = pd.to_numeric(eligible.get("estimated_cost_usd", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    cumulative_cost = float(costs.sum())
    if eligible.empty:
        seven_day_cost = 0.0
        last_usage_at = ""
    else:
        seven_day = eligible[eligible["generated_at_utc"] >= now_utc - pd.Timedelta(days=7)]
        seven_day_cost = float(pd.to_numeric(seven_day.get("estimated_cost_usd"), errors="coerce").fillna(0.0).sum())
        last_usage_at = eligible["generated_at_utc"].max().isoformat()
    remaining = baseline_balance - cumulative_cost
    daily_average = seven_day_cost / 7
    estimated_days_remaining = remaining / daily_average if daily_average > 0 and remaining > 0 else None
    return {
        "currency": str(config.get("currency", "USD")),
        "baseline_balance_usd": round(baseline_balance, 6),
        "baseline_at_utc": baseline_at.isoformat() if pd.notna(baseline_at) else "",
        "estimated_spend_since_baseline_usd": round(cumulative_cost, 8),
        "estimated_remaining_usd": round(remaining, 6),
        "estimated_spend_7d_usd": round(seven_day_cost, 8),
        "estimated_daily_average_7d_usd": round(daily_average, 8),
        "estimated_days_remaining": round(float(estimated_days_remaining), 1) if estimated_days_remaining is not None else None,
        "usage_requests_since_baseline": int(len(eligible)),
        "last_usage_at_utc": last_usage_at,
        "calculated_at_utc": now_utc.isoformat(),
        "calculation_method": "baseline_minus_recorded_token_cost",
        "is_estimate": True,
    }


def _pricing(model: str) -> dict[str, float]:
    config = load_api_balance_config()
    model_pricing = config.get("model_pricing") or {}
    selected = model_pricing.get(model) or config.get("default_pricing") or {}
    return {
        "input_usd_per_million": _number(os.getenv("OPENAI_INPUT_USD_PER_1M"), _number(selected.get("input_usd_per_million"), 0.75)),
        "cached_input_usd_per_million": _number(os.getenv("OPENAI_CACHED_INPUT_USD_PER_1M"), _number(selected.get("cached_input_usd_per_million"), 0.075)),
        "output_usd_per_million": _number(os.getenv("OPENAI_OUTPUT_USD_PER_1M"), _number(selected.get("output_usd_per_million"), 4.5)),
    }


def _integer(value: object) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _number(value: object, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)
