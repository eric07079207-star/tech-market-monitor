from __future__ import annotations

import argparse
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_ai_summary import main as generate_summary
from src.ai_summary import load_ai_summary_history


TAIPEI = ZoneInfo("Asia/Taipei")


def _has_summary_for_local_date(target_date) -> bool:
    history = load_ai_summary_history()
    if history.empty or "generated_at_utc" not in history:
        return False
    generated = pd.to_datetime(history["generated_at_utc"], errors="coerce", utc=True).dropna()
    if generated.empty:
        return False
    local_dates = generated.dt.tz_convert(TAIPEI).dt.date
    return target_date in set(local_dates)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one AI summary per Asia/Taipei day after 07:00.")
    parser.add_argument("--force", action="store_true", help="Generate a summary even if today's summary already exists.")
    args = parser.parse_args()

    now_local = pd.Timestamp.now(tz=TAIPEI)
    target_date = now_local.date()
    if not args.force and now_local.hour < 7:
        print(f"skip ai summary: local time {now_local.isoformat()} is before 07:00")
        return
    if not args.force and _has_summary_for_local_date(target_date):
        print(f"skip ai summary: summary already exists for Asia/Taipei date {target_date}")
        return
    generate_summary()


if __name__ == "__main__":
    main()
