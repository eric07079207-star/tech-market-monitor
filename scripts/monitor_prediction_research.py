"""Frozen shadow inference. No fitting, tuning, paid API, or baseline replacement."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from src.prediction_research import (SYMBOLS, MARKETS, FEATURES, ResearchLSTM, SequenceClassifier,
    completed_prices, build_panel, with_sequences, probability, calibrated)
from scripts.run_prediction_research import write_json, digest

CACHE = ROOT / "data/cache/prediction_research"


def append_locked(existing, incoming):
    if incoming.empty:
        return existing
    return pd.concat([existing, incoming], ignore_index=True).drop_duplicates("key", keep="first")


def forward_window(schedule, signal_date, horizon, now):
    sessions = pd.DatetimeIndex(schedule.index).tz_localize(None)
    idx = sessions.get_loc(signal_date)
    entry, target = sessions[idx+1], sessions[idx+1+horizon]
    return (entry, target) if schedule.loc[entry, "open"] > now else None


def publish_bundle(run_dir):
    report = json.loads((run_dir / "report.json").read_text())
    bundle = CACHE / "models" / report["run_id"]
    bundle.mkdir(parents=True, exist_ok=True)
    entries = []
    snapshot = pd.read_parquet(run_dir / "price_snapshot.parquet")
    for horizon, result in report["horizons"].items():
        name = result["development_selected_model"]
        filenames = [f"{horizon}d_{name}.joblib"]
        if name == "lstm":
            filenames.append(f"{horizon}d_{name}.pt")
        files = []
        for filename in filenames:
            src = run_dir / filename
            dest = bundle / filename
            if dest.exists() and digest(src) != digest(dest):
                raise RuntimeError("Immutable model version conflict")
            shutil.copy2(src, dest)
            files.append({"path": str(dest.relative_to(CACHE)), "sha256": digest(dest)})
        reference_panel = with_sequences(build_panel(snapshot, int(horizon)))
        reference_panel = reference_panel[reference_panel.groupby("date").symbol.transform("nunique").eq(len(SYMBOLS))]
        boundary = result["partition"]["calibration_start"]
        reference = reference_panel[reference_panel.target_date.lt(boundary)]
        bins = {}
        for col in FEATURES:
            edges = np.unique(np.quantile(reference[col], np.linspace(0, 1, 11)))
            edges[0], edges[-1] = -np.inf, np.inf
            fractions = np.histogram(reference[col], edges)[0]/len(reference)
            bins[col] = {"interior_edges": edges[1:-1].tolist(), "fractions": fractions.tolist()}
        entries.append({"horizon": int(horizon), "model": name, "files": files, "reference_bins": bins})
    write_json(CACHE / "bundle.json", {"run_id": report["run_id"], "entries": entries,
                                      "mode": "shadow_only", "features": FEATURES})


def load_model(entry):
    import joblib
    paths = []
    for item in entry["files"]:
        path = (CACHE / item["path"]).resolve()
        if not path.is_relative_to((CACHE / "models").resolve()) or digest(path) != item["sha256"]:
            raise ValueError("Model path/hash validation failed")
        paths.append(path)
    # Only load our own checksum-verified repository artifacts, never user uploads.
    payload = joblib.load(paths[0])
    if entry["model"] == "lstm":
        import torch
        model = SequenceClassifier(payload["columns"])
        model.indices = [FEATURES.index(c) for c in model.columns]
        model.mean, model.std = payload["mean"], payload["std"]
        model.model = ResearchLSTM(len(model.columns))
        model.model.load_state_dict(torch.load(paths[1], map_location="cpu", weights_only=True))
    else:
        model = payload["model"]
    return model, payload["calibrator"], payload["threshold"]


def main():
    import exchange_calendars as xcals
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish-run", type=Path)
    args = parser.parse_args()
    if args.publish_run:
        publish_bundle(args.publish_run)
    bundle = json.loads((CACHE / "bundle.json").read_text())
    now = pd.Timestamp.now(tz="UTC")
    raw = pd.read_parquet(ROOT / "data/cache/prices.parquet")
    prices, integrity = completed_prices(raw[raw.symbol.isin(SYMBOLS + MARKETS)], now)
    if prices.date.max() < pd.Timestamp(integrity["last_completed_session"]):
        raise RuntimeError("Price cache has not reached the latest completed session; keep last good shadow cache")
    log_path = CACHE / "shadow_predictions.parquet"
    log = pd.read_parquet(log_path) if log_path.exists() else pd.DataFrame()
    calendar = xcals.get_calendar("XNYS", start=prices.date.min(), end=(now + pd.Timedelta(days=120)).date())
    schedule = calendar.schedule
    new_rows, waiting, drift = [], [], []
    for entry in bundle["entries"]:
        h = entry["horizon"]
        panel = with_sequences(build_panel(prices, h))
        latest = panel[panel.date.eq(panel.date.max())]
        signal_date = latest.date.max()
        recent = panel[panel.date.ge(signal_date-pd.Timedelta(days=90))]
        for col, reference in entry.get("reference_bins", {}).items():
            edges = [-np.inf, *reference["interior_edges"], np.inf]
            a = np.maximum(reference["fractions"], .0001)
            b = np.maximum(np.histogram(recent[col], edges)[0]/len(recent), .0001)
            psi = float(((b-a)*np.log(b/a)).sum())
            drift.append({"horizon": h, "feature": col, "psi": psi, "warning": psi >= .25})
        if latest.symbol.nunique() != len(SYMBOLS):
            raise RuntimeError("Incomplete live research universe")
        model, calibrator, threshold = load_model(entry)
        p = calibrated(calibrator, probability(model, latest))
        window = forward_window(schedule, signal_date, h, now)
        if window is None:
            waiting.append(f"{h}D: cached signal {signal_date.date()} is too old for a new next-open prediction")
            continue
        entry_date, target_date = window
        for (_, row), value in zip(latest.iterrows(), p):
            key = f"{bundle['run_id']}:{h}:{row.symbol}:{signal_date.date()}"
            new_rows.append({"key": key, "run_id": bundle["run_id"], "model": entry["model"],
                             "symbol": row.symbol, "horizon": h, "signal_date": signal_date,
                             "generated_at_utc": now.isoformat(), "entry_date": entry_date, "target_date": target_date,
                             "probability": float(value), "threshold": threshold,
                             "actual_return": np.nan, "settled_at_utc": "", "mode": "shadow_only"})
    if new_rows:
        log = append_locked(log, pd.DataFrame(new_rows))
    if not log.empty:
        opens = prices.pivot(index="date", columns="symbol", values="open")
        for idx, row in log[log.actual_return.isna()].iterrows():
            start, end = pd.Timestamp(row.entry_date), pd.Timestamp(row.target_date)
            if start in opens.index and end in opens.index:
                a, b = opens.loc[start, row.symbol], opens.loc[end, row.symbol]
                if np.isfinite(a) and np.isfinite(b) and a > 0:
                    log.loc[idx, "actual_return"] = b/a-1
                    log.loc[idx, "settled_at_utc"] = now.isoformat()
        tmp = log_path.with_suffix(".tmp")
        log.to_parquet(tmp, index=False)
        tmp.replace(log_path)
    report = json.loads((CACHE / "report.json").read_text())
    drift_warnings = sum(v.get("drift_warning_count", 0) for v in report["horizons"].values())
    state = {"scanned_at_utc": now.isoformat(), "source_last_date": str(prices.date.max().date()),
             "rows": len(log), "settled": int(log.actual_return.notna().sum()) if not log.empty else 0,
             "waiting": waiting, "status": "waiting_for_next_close" if waiting else "shadow_updated",
             "model_status": "unproven_no_production_promotion", "integrity": integrity,
             "last_research_drift_warning_count": drift_warnings, "run_id": bundle["run_id"]}
    state["models_verified"] = len(bundle["entries"])
    state["current_drift"] = drift
    state["current_drift_warning_count"] = sum(row["warning"] for row in drift)
    write_json(CACHE / "monitor.json", state)
    print(json.dumps({k: v for k, v in state.items() if k != "current_drift"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
