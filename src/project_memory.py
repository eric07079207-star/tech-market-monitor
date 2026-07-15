from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DATA_DIR


MEMORY_DIR = DATA_DIR / "觀察版資料" / "13_專案記憶"
MEMORY_ARCHIVE_DIR = MEMORY_DIR / "archive"
PROJECT_MEMORY_FILE = MEMORY_DIR / "project_memory.md"
CONVERSATION_LOG_FILE = MEMORY_DIR / "conversation_log.md"
ACTIVE_CONTEXT_FILE = MEMORY_DIR / "active_context.md"
DECISION_REGISTER_FILE = MEMORY_DIR / "decision_register.csv"
MEMORY_CHANGELOG_FILE = MEMORY_DIR / "memory_changelog.csv"
MEMORY_DOCX_FILE = MEMORY_DIR / "專案記憶與討論摘要.docx"

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"sk-proj-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
    re.compile(r"(?i)(api[_ -]?key|token|secret|password)\s*[:=]\s*[^\s]+"),
]


@dataclass(frozen=True)
class MemoryBundle:
    project_memory: str
    conversation_log: str
    active_context: str
    decision_register: pd.DataFrame
    memory_changelog: pd.DataFrame
    status_table: pd.DataFrame
    latest_updates: pd.DataFrame
    directory: Path


def ensure_memory_dirs() -> None:
    MEMORY_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).fillna("")


def scrub_secrets(text: str) -> str:
    cleaned = text or ""
    for pattern in SECRET_PATTERNS:
        cleaned = pattern.sub("[REDACTED]", cleaned)
    return cleaned


def load_memory_bundle() -> MemoryBundle:
    ensure_memory_dirs()
    project_memory = scrub_secrets(_read_text(PROJECT_MEMORY_FILE))
    conversation_log = scrub_secrets(_read_text(CONVERSATION_LOG_FILE))
    active_context = scrub_secrets(_read_text(ACTIVE_CONTEXT_FILE))
    decision_register = _read_csv(DECISION_REGISTER_FILE)
    memory_changelog = _read_csv(MEMORY_CHANGELOG_FILE)
    status_table = build_memory_status_table(decision_register, memory_changelog)
    latest_updates = build_latest_updates(decision_register, memory_changelog)
    return MemoryBundle(
        project_memory=project_memory,
        conversation_log=conversation_log,
        active_context=active_context,
        decision_register=decision_register,
        memory_changelog=memory_changelog,
        status_table=status_table,
        latest_updates=latest_updates,
        directory=MEMORY_DIR,
    )


def build_memory_status_table(decision_register: pd.DataFrame, memory_changelog: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    decisions = decision_register.copy() if not decision_register.empty else pd.DataFrame(columns=["status", "date"])
    changelog = memory_changelog.copy() if not memory_changelog.empty else pd.DataFrame(columns=["date"])
    if "date" in decisions:
        decisions["date"] = pd.to_datetime(decisions["date"], errors="coerce")
    if "date" in changelog:
        changelog["date"] = pd.to_datetime(changelog["date"], errors="coerce")

    rows.append(
        {
            "項目": "已記錄決策",
            "數值": str(int(len(decisions))),
            "說明": "已確認並寫入決策登錄的項目數。",
        }
    )
    rows.append(
        {
            "項目": "已執行決策",
            "數值": str(int((decisions.get("status", pd.Series(dtype=str)) == "已執行").sum()) if not decisions.empty else 0),
            "說明": "已確認且已完成落地的決策數。",
        }
    )
    rows.append(
        {
            "項目": "待處理決策",
            "數值": str(int((decisions.get("status", pd.Series(dtype=str)).isin(["待執行", "討論中"])).sum()) if not decisions.empty else 0),
            "說明": "仍在討論或後續要落地的工作數。",
        }
    )
    rows.append(
        {
            "項目": "最近記憶更新",
            "數值": str(changelog["date"].max().date()) if not changelog.empty and changelog["date"].notna().any() else "n/a",
            "說明": "最後一次寫入專案記憶的日期。",
        }
    )
    return pd.DataFrame(rows)


def build_latest_updates(decision_register: pd.DataFrame, memory_changelog: pd.DataFrame) -> pd.DataFrame:
    updates: list[dict[str, object]] = []
    if not decision_register.empty:
        decision_view = decision_register.copy()
        if "date" in decision_view:
            decision_view["date"] = pd.to_datetime(decision_view["date"], errors="coerce")
        decision_view = decision_view.sort_values("date", ascending=False).head(5)
        for row in decision_view.itertuples():
            updates.append(
                {
                    "來源": "決策",
                    "日期": str(row.date.date()) if pd.notna(getattr(row, "date", pd.NaT)) else "n/a",
                    "標題": getattr(row, "title", ""),
                    "狀態": getattr(row, "status", ""),
                }
            )
    if not memory_changelog.empty:
        changelog_view = memory_changelog.copy()
        if "date" in changelog_view:
            changelog_view["date"] = pd.to_datetime(changelog_view["date"], errors="coerce")
        changelog_view = changelog_view.sort_values("date", ascending=False).head(5)
        for row in changelog_view.itertuples():
            updates.append(
                {
                    "來源": "記憶更新",
                    "日期": str(row.date.date()) if pd.notna(getattr(row, "date", pd.NaT)) else "n/a",
                    "標題": getattr(row, "change_summary", ""),
                    "狀態": getattr(row, "source", ""),
                }
            )
    if not updates:
        return pd.DataFrame(columns=["來源", "日期", "標題", "狀態"])
    return pd.DataFrame(updates).head(10)
