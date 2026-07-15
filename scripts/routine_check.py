from __future__ import annotations

import json
import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_observation_data import main as export_observation_data_main
from src.data import cache_path
from src.health import data_health_report

REMOTE_CORE_CACHE_FILES = [
    "data/cache/metadata.json",
    "data/cache/prices.parquet",
    "data/cache/macro.parquet",
    "data/cache/news.parquet",
    "data/cache/international_news.parquet",
    "data/cache/ai_summary.json",
    "data/cache/ai_summary_history.parquet",
    "data/cache/prediction_log.csv",
    "data/cache/discovery_news.parquet",
    "data/cache/discovery_candidates.parquet",
    "data/cache/discovery_history.parquet",
    "data/cache/tsla_keyword_news.parquet",
    "data/cache/governance_summary.parquet",
    "data/cache/sentiment.parquet",
    "data/cache/market_event_windows.parquet",
    "data/cache/update_runs.csv",
    "data/cache/update_modules.csv",
    "data/cache/kg/fact_events.parquet",
    "data/cache/kg/narrative_features.parquet",
    "data/cache/kg/market_reactions.parquet",
    "data/cache/kg/kg_metadata.json",
    "data/cache/lstm/lstm_status.json",
]


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _git_status() -> str:
    result = subprocess.run(
        ["git", "status", "-sb"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _git_ahead_behind() -> str:
    try:
        subprocess.run(
            ["git", "fetch", "origin"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        result = subprocess.run(
            ["git", "rev-list", "--left-right", "--count", "HEAD...origin/main"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "timeout"
    except Exception:
        return "n/a"


def _status_counts(report: pd.DataFrame) -> str:
    if report.empty or "狀態" not in report:
        return "n/a"
    counts = report["狀態"].value_counts()
    return " / ".join(f"{label}:{int(count)}" for label, count in counts.items())


def _parse_timestamp(value: str | None):
    if not value:
        return pd.NaT
    return pd.to_datetime(value, errors="coerce", utc=True)


def _remote_metadata() -> dict:
    url = "https://raw.githubusercontent.com/eric07079207-star/tech-market-monitor/main/data/cache/metadata.json"
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _download_remote_file(repo_path: str) -> None:
    url = f"https://raw.githubusercontent.com/eric07079207-star/tech-market-monitor/main/{repo_path}"
    request = urllib.request.Request(url, headers={"User-Agent": "codex-routine-check"})
    destination = ROOT / repo_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())


def _download_and_extract_remote_data(ref: str = "main") -> None:
    if ref != "main":
        raise ValueError("Only main ref is supported for raw cache sync")
    for repo_path in REMOTE_CORE_CACHE_FILES:
        _download_remote_file(repo_path)
    export_observation_data_main()


def _sync_remote_cache_if_newer(local_metadata: dict) -> tuple[bool, str, dict]:
    try:
        remote_metadata = _remote_metadata()
    except Exception as exc:
        return False, f"remote metadata unavailable: {exc}", {}

    local_time = _parse_timestamp(local_metadata.get("updated_at_utc"))
    remote_time = _parse_timestamp(remote_metadata.get("updated_at_utc"))
    if pd.isna(remote_time):
        return False, "remote metadata missing updated_at_utc", remote_metadata
    if not pd.isna(local_time) and local_time >= remote_time:
        return False, "local cache already up to date", remote_metadata

    try:
        _download_and_extract_remote_data("main")
    except Exception as exc:
        return False, f"remote cache sync failed: {exc}", remote_metadata
    return True, "synced local cache from remote main", remote_metadata


def _repair_and_refresh() -> list[str]:
    """Apply only deterministic, reversible maintenance actions."""
    from scripts.build_project_memory_doc import build_doc as build_project_memory_doc

    actions: list[str] = []
    export_observation_data_main()
    actions.append("重新匯出觀察版資料與中文檢查檔")
    build_project_memory_doc()
    actions.append("刷新專案記憶 Word 摘要")
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description="執行資料健康檢查與安全修復")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只檢查，不重新匯出觀察版資料或重建 Word 摘要",
    )
    args = parser.parse_args()

    cache_dir = cache_path("")
    metadata = _read_json(cache_dir / "metadata.json", {})
    sync_applied, sync_message, remote_metadata = _sync_remote_cache_if_newer(metadata)
    if sync_applied:
        metadata = _read_json(cache_dir / "metadata.json", {})
    ai_summary = _read_json(cache_dir / "ai_summary.json", {})
    kg_metadata = _read_json(cache_dir / "kg" / "kg_metadata.json", {})

    report = data_health_report(
        _read_parquet(cache_dir / "prices.parquet"),
        _read_parquet(cache_dir / "macro.parquet"),
        _read_parquet(cache_dir / "news.parquet"),
        _read_parquet(cache_dir / "international_news.parquet"),
        _read_csv(cache_dir / "prediction_log.csv"),
        metadata,
        ai_summary_history=_read_parquet(cache_dir / "ai_summary_history.parquet"),
        lstm_status=_read_json(cache_dir / "lstm" / "lstm_status.json", {}),
        discovery_news=_read_parquet(cache_dir / "discovery_news.parquet"),
        discovery_candidates=_read_parquet(cache_dir / "discovery_candidates.parquet"),
        discovery_history=_read_parquet(cache_dir / "discovery_history.parquet"),
        focus_news=_read_parquet(cache_dir / "tsla_keyword_news.parquet"),
        governance=_read_parquet(cache_dir / "governance_summary.parquet"),
        sentiment=_read_parquet(cache_dir / "sentiment.parquet"),
        market_event_windows=_read_parquet(cache_dir / "market_event_windows.parquet"),
        kg_fact_events=_read_parquet(cache_dir / "kg" / "fact_events.parquet"),
        kg_narratives=_read_parquet(cache_dir / "kg" / "narrative_features.parquet"),
        kg_reactions=_read_parquet(cache_dir / "kg" / "market_reactions.parquet"),
    )

    repair_actions: list[str] = []
    if not args.check_only:
        try:
            repair_actions = _repair_and_refresh()
        except Exception as exc:
            repair_actions.append(f"修復流程失敗：{exc}")

    stale_items = []
    if not report.empty:
        stale_rows = report[report["狀態"].astype(str).str.contains("🟡|🔴", regex=True)]
        stale_items = stale_rows["資料項目"].astype(str).tolist()

    print("例行檢查摘要")
    print(f"- git: {_git_status()}")
    print(f"- sync(HEAD...origin/main): {_git_ahead_behind()}")
    print(f"- remote_sync: {sync_message}")
    if remote_metadata:
        print(f"- remote_cache_updated_at_utc: {remote_metadata.get('updated_at_utc', 'n/a')}")
    print(f"- cache_updated_at_utc: {metadata.get('updated_at_utc', 'n/a')}")
    print(f"- pipeline_status: {metadata.get('pipeline_status', 'n/a')}")
    print(f"- ai_summary_at_utc: {ai_summary.get('generated_at_utc', 'n/a')}")
    print(f"- ai_summary_model: {ai_summary.get('model', 'n/a')}")
    print(f"- kg_updated_at_utc: {kg_metadata.get('updated_at_utc', 'n/a')}")
    print(f"- health_counts: {_status_counts(report)}")
    print(f"- attention_items: {', '.join(stale_items) if stale_items else 'none'}")
    print(f"- repair_mode: {'check-only' if args.check_only else 'auto-repair'}")
    print(f"- repair_actions: {'；'.join(repair_actions) if repair_actions else 'none'}")


if __name__ == "__main__":
    main()
