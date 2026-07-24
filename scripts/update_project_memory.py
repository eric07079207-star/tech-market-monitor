from __future__ import annotations

"""Append a confirmed project decision to the durable memory sources.

This script deliberately accepts only structured, explicit inputs. GitHub Actions can
rebuild the Word view, but it cannot infer decisions from a private Codex conversation.
Run this after a user confirms a decision or an implementation is completed.
"""

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.project_memory import (  # noqa: E402
    ACTIVE_CONTEXT_FILE,
    CONVERSATION_LOG_FILE,
    DECISION_REGISTER_FILE,
    MEMORY_CHANGELOG_FILE,
    PROJECT_MEMORY_FILE,
    ensure_memory_dirs,
    scrub_secrets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write one confirmed decision into project memory.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--status", choices=["已執行", "待執行", "討論中"], default="已執行")
    parser.add_argument("--discussion", default="")
    parser.add_argument("--long-term", dest="long_term", default="")
    parser.add_argument("--context", default="")
    parser.add_argument("--date", dest="record_date", default=date.today().isoformat())
    return parser.parse_args()


def append_markdown(path: Path, heading: str, body: str) -> None:
    if not body.strip():
        return
    cleaned = scrub_secrets(body).strip()
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    entry = f"\n\n## {heading}\n\n- {cleaned}\n"
    if cleaned in existing:
        return
    path.write_text(existing.rstrip() + entry, encoding="utf-8")


def next_decision_id(path: Path) -> str:
    if not path.exists():
        return "D-001"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    numbers = []
    for row in rows:
        value = str(row.get("decision_id", ""))
        if value.startswith("D-"):
            try:
                numbers.append(int(value[2:]))
            except ValueError:
                pass
    return f"D-{max(numbers, default=0) + 1:03d}"


def append_csv(path: Path, fields: list[str], row: dict[str, str]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({field: scrub_secrets(str(row.get(field, ""))) for field in fields})


def main() -> None:
    args = parse_args()
    ensure_memory_dirs()
    decision_id = next_decision_id(DECISION_REGISTER_FILE)
    title = scrub_secrets(args.title)
    decision = scrub_secrets(args.decision)
    reason = scrub_secrets(args.reason)
    scope = scrub_secrets(args.scope)

    append_markdown(PROJECT_MEMORY_FILE, f"{args.record_date}｜{title}", args.long_term)
    append_markdown(CONVERSATION_LOG_FILE, f"{args.record_date}｜{title}", args.discussion or f"決策：{decision}。")
    append_markdown(ACTIVE_CONTEXT_FILE, f"{args.record_date}｜{title}", args.context or f"{args.status}：{decision}。")
    append_csv(
        DECISION_REGISTER_FILE,
        ["decision_id", "date", "status", "category", "title", "decision", "reason", "impact_scope", "superseded_by"],
        {
            "decision_id": decision_id,
            "date": args.record_date,
            "status": args.status,
            "category": args.category,
            "title": title,
            "decision": decision,
            "reason": reason,
            "impact_scope": scope,
            "superseded_by": "",
        },
    )
    append_csv(
        MEMORY_CHANGELOG_FILE,
        ["date", "change_summary", "reason", "source"],
        {"date": args.record_date, "change_summary": title, "reason": reason, "source": "Codex 已確認決策"},
    )
    print(f"updated project memory: {decision_id} {title}")


if __name__ == "__main__":
    main()
