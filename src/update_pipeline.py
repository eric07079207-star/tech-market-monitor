from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .data import cache_path


UPDATE_RUNS_CACHE = cache_path("update_runs.csv")
UPDATE_MODULES_CACHE = cache_path("update_modules.csv")


@dataclass(frozen=True)
class FrameValidation:
    required_columns: tuple[str, ...] = ()
    latest_column: str | None = None
    min_rows: int = 0
    min_fraction_of_previous: float | None = None
    allow_empty: bool = False
    max_missing_ratio: float = 0.5


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_cached_frame(filename: str) -> pd.DataFrame:
    path = cache_path(filename)
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.DataFrame()


def latest_value(frame: pd.DataFrame | None, preferred_columns: list[str] | tuple[str, ...]) -> str:
    if frame is None or frame.empty:
        return ""
    for column in preferred_columns:
        if column not in frame.columns:
            continue
        value = pd.to_datetime(frame[column], errors="coerce", utc=True).max()
        if pd.notna(value):
            return value.isoformat()
    return ""


def validate_frame(frame: pd.DataFrame | None, previous: pd.DataFrame | None, rules: FrameValidation) -> list[str]:
    data = frame if frame is not None else pd.DataFrame()
    prior = previous if previous is not None else pd.DataFrame()
    issues: list[str] = []

    if data.empty:
        if rules.allow_empty or prior.empty:
            return issues
        issues.append("new frame is empty while previous cache has data")
        return issues

    missing_columns = [column for column in rules.required_columns if column not in data.columns]
    if missing_columns:
        issues.append(f"missing required columns: {', '.join(missing_columns)}")
        return issues

    if len(data) < rules.min_rows:
        issues.append(f"row count {len(data)} below minimum {rules.min_rows}")

    if rules.min_fraction_of_previous is not None and not prior.empty:
        threshold = max(rules.min_rows, int(len(prior) * rules.min_fraction_of_previous))
        if len(data) < threshold:
            issues.append(f"row count {len(data)} below {rules.min_fraction_of_previous:.0%} of previous cache ({len(prior)})")

    for column in rules.required_columns:
        missing_ratio = float(data[column].isna().mean())
        if missing_ratio > rules.max_missing_ratio:
            issues.append(f"column {column} missing ratio {missing_ratio:.0%} exceeds {rules.max_missing_ratio:.0%}")

    if rules.latest_column and rules.latest_column in data.columns and rules.latest_column in prior.columns and not prior.empty:
        new_latest = pd.to_datetime(data[rules.latest_column], errors="coerce", utc=True).max()
        old_latest = pd.to_datetime(prior[rules.latest_column], errors="coerce", utc=True).max()
        if pd.notna(new_latest) and pd.notna(old_latest) and new_latest < old_latest:
            issues.append(f"latest {rules.latest_column} regressed from {old_latest.isoformat()} to {new_latest.isoformat()}")

    return issues


def safe_write_frame(filename: str, frame: pd.DataFrame, rules: FrameValidation) -> tuple[bool, list[str], pd.DataFrame]:
    path = cache_path(filename)
    previous = load_cached_frame(filename)
    issues = validate_frame(frame, previous, rules)
    if issues:
        return False, issues, previous

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temp_path(path)
    if path.suffix == ".parquet":
        frame.to_parquet(temp_path, index=False)
    elif path.suffix == ".csv":
        frame.to_csv(temp_path, index=False)
    else:
        raise ValueError(f"unsupported file type for safe write: {path.suffix}")
    os.replace(temp_path, path)
    return True, [], frame


def safe_write_json(filename: str, payload: dict[str, Any]) -> None:
    path = cache_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temp_path(path)
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, path)


def append_update_logs(run_record: dict[str, Any], module_records: list[dict[str, Any]]) -> None:
    _append_csv(UPDATE_RUNS_CACHE, [run_record])
    _append_csv(UPDATE_MODULES_CACHE, module_records)


def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    new_rows = pd.DataFrame(rows)
    if path.exists():
        current = pd.read_csv(path)
        new_rows = pd.concat([current, new_rows], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temp_path(path)
    new_rows.to_csv(temp_path, index=False)
    os.replace(temp_path, path)


def _temp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")
