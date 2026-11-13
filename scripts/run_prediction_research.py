from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.prediction_research import (
    FEATURES, BASE_FEATURES, SYMBOLS, MARKETS, VERSION, SEED, SequenceClassifier,
    completed_prices, build_panel, with_sequences, purged_before, temporal_folds,
    fit_model, probability, tune, calibrate, calibrated, select_threshold,
    classification, portfolio, block_interval, drift_report,
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def preserve_baselines(output):
    paths = []
    for subdir in ("", "tsla_specialist"):
        for name in ("lstm_model.pt", "lstm_model.json", "lstm_scaler.json", "lstm_backtest.parquet"):
            path = ROOT / "data/cache/lstm" / subdir / name
            if path.exists():
                destination = output / "baseline" / subdir / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                paths.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path)})
    return paths


def baseline_scores():
    result = []
    for subdir in ("", "tsla_specialist"):
        path = ROOT / "data/cache/lstm" / subdir / "lstm_backtest.parquet"
        if path.exists():
            f = pd.read_parquet(path)
            result.append({"name": "legacy_" + (subdir or "shared"), "rows": len(f),
                           "accuracy": float(f.label.eq(f.predicted_label).mean()),
                           "always_long_accuracy": float(f.label.mean()),
                           "start": str(pd.to_datetime(f.date).min().date()),
                           "end": str(pd.to_datetime(f.date).max().date()),
                           "comparable": False, "reason": "Different 20D close-to-close target, dates and sampling; historical baseline only."})
    return result


def save_model(model, path, calibrator, threshold):
    import joblib
    if isinstance(model, SequenceClassifier):
        import torch
        torch.save(model.model.state_dict(), path.with_suffix(".pt"))
        joblib.dump({"mean": model.mean, "std": model.std, "columns": model.columns,
                     "calibrator": calibrator, "threshold": threshold}, path)
    else:
        joblib.dump({"model": model, "calibrator": calibrator, "threshold": threshold}, path)


def research(args):
    from sklearn.metrics import log_loss
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "data/research_runs" / run_id
    output.mkdir(parents=True)
    price_path = ROOT / "data/cache/prices.parquet"
    baseline = preserve_baselines(output)
    raw_prices = pd.read_parquet(price_path)
    prices, integrity = completed_prices(raw_prices[raw_prices.symbol.isin(SYMBOLS + MARKETS)])
    integrity["universe_filter_rows_excluded"] = int((~raw_prices.symbol.isin(SYMBOLS + MARKETS)).sum())
    prices.to_parquet(output / "price_snapshot.parquet", index=False)
    report = {
        "version": VERSION, "run_id": run_id, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "source_sha256": digest(price_path), "snapshot_sha256": digest(output / "price_snapshot.parquet"),
        "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                        (Path(__file__), ROOT / "src/prediction_research.py")},
        "source_last_date": str(prices.date.max().date()), "integrity": integrity,
        "test_start": args.test_start, "test_end": args.test_end,
        "mode": "retrospective_shadow_only", "promotion": "not_permitted",
        "test_warning": "Historical dates overlap previous experiments; not a globally untouched test. Future locked shadow records are needed.",
        "target": "after-close signal; next open to open h exchange sessions later",
        "universe": SYMBOLS, "sequence_length": 20, "seed": SEED,
        "features": FEATURES, "optuna_trials_per_tree": args.trials,
        "costs": {"fee_bps": 5, "slippage_bps": 5, "one_way_total_bps": 10, "cash_yield": 0, "shorting": False},
        "baseline_files": baseline, "legacy_baselines": baseline_scores(),
        "dependencies": {p: importlib.metadata.version(p) for p in
                         ("numpy", "pandas", "scikit-learn", "optuna", "torch", "exchange-calendars", "xgboost", "lightgbm")},
        "limitations": [
            "Current eight surviving stocks; not a survivorship-free market universe.",
            "Current adjusted Yahoo prices are not point-in-time price vintages; raw prices/split histories need independent reconciliation.",
            "News, macro revisions and fundamentals excluded until publication time and comparable accounting periods are verified.",
            "No guarantee of profit. Bootstrap intervals are diagnostic, not multiple-testing-adjusted promotion evidence.",
            "Chronos skipped: financial pretraining overlap unverified. TFT/Transformer deferred pending a superior simple baseline.",
            "No new API calls. Retrospective LSTM is a small fixed CPU candidate, not exhaustive architecture search.",
        ], "horizons": {},
    }
    # Lock and archive protocol before any test labels are evaluated.
    write_json(output / "protocol.json", report)
    all_predictions, latest_signals = [], []
    for h in args.horizons:
        print(f"horizon={h}: preparing causal panel", flush=True)
        full = with_sequences(build_panel(prices, h))
        full = full[full.groupby("date").symbol.transform("nunique").eq(len(SYMBOLS))].copy()
        mature = full.dropna(subset=["label", "target_date"])
        test = mature[mature.date.between(args.test_start, args.test_end)].copy()
        if test.empty or test.date.nunique() < 100:
            raise ValueError("Insufficient matured held-out sessions")
        pretest = purged_before(mature, args.test_start)
        dates = np.sort(pretest.date.unique())
        cal_start, decision_start = dates[-252], dates[-126]
        development = purged_before(pretest, cal_start)
        cal = purged_before(pretest[pretest.date.ge(cal_start)], decision_start)
        decision = pretest[pretest.date.ge(decision_start)]
        if len(development) < 3000 or cal.empty or decision.empty:
            raise ValueError("Insufficient training/calibration/selection data")
        partition = {"development": len(development), "calibration": len(cal),
                     "threshold_selection": len(decision), "test": len(test),
                     "development_label_end": str(development.target_date.max().date()),
                     "calibration_start": str(cal.date.min().date()),
                     "calibration_label_end": str(cal.target_date.max().date()),
                     "threshold_start": str(decision.date.min().date()),
                     "threshold_label_end": str(decision.target_date.max().date()),
                     "test_start": str(test.date.min().date()), "test_end": str(test.date.max().date()),
                     "test_label_end": str(test.target_date.max().date())}
        models, cv, params_by_name = {}, {}, {}
        for name in args.models:
            print(f"horizon={h} model={name}: walk-forward tuning", flush=True)
            params, scores = tune(name, development, args.trials)
            params_by_name[name], cv[name] = params, scores
            print(f"horizon={h} model={name}: cv_log_loss={scores['cv_log_loss']:.5f}", flush=True)
        chosen = min(cv, key=lambda name: cv[name]["cv_log_loss"])
        # Ablation is measured only on development folds, never selected on test results.
        base_losses = []
        for train, valid in temporal_folds(development):
            m = fit_model(chosen, train, params_by_name[chosen], BASE_FEATURES)
            base_losses.append(log_loss(valid.label, probability(m, valid, BASE_FEATURES), labels=[0, 1]))
        ablation = {"model": chosen, "base_only_cv_log_loss": float(np.mean(base_losses)),
                    "with_market_cv_log_loss": cv[chosen]["cv_log_loss"],
                    "market_log_loss_improvement": float(np.mean(base_losses)-cv[chosen]["cv_log_loss"])}
        results, prediction_map, daily_map = {}, {}, {}
        train_up_return = float(development.loc[development.label.eq(1), "future_return"].mean())
        train_down_return = float(development.loc[development.label.eq(0), "future_return"].mean())
        for name in args.models:
            print(f"horizon={h} model={name}: final fit/calibration/test", flush=True)
            model = fit_model(name, development, params_by_name[name])
            calibrator = calibrate(probability(model, cal), cal.label)
            dprob = calibrated(calibrator, probability(model, decision))
            threshold = select_threshold(dprob, decision.label)
            raw = probability(model, test)
            p = calibrated(calibrator, raw)
            result = classification(test, p, threshold)
            result["raw_brier"] = float(np.mean((raw-test.label.to_numpy())**2))
            result["threshold"] = threshold
            estimate = p*train_up_return+(1-p)*train_down_return
            result["mean_predicted_return"] = float(estimate.mean())
            result["return_mae"] = float(np.mean(abs(estimate-test.future_return.to_numpy())))
            result["mean_actual_return"] = float(test.future_return.mean())
            result["portfolio"], daily_map[name] = portfolio(test, prices, p, threshold, h)
            result["cost_stress_20bps"], _ = portfolio(test, prices, p, threshold, h, one_way_bps=20)
            result["by_symbol"] = []
            for symbol in SYMBOLS:
                mask = test.symbol.eq(symbol).to_numpy()
                stats = classification(test.loc[mask], p[mask], threshold)
                stats.update(symbol=symbol, mean_future_return=float(test.loc[mask, "future_return"].mean()))
                stats["portfolio"], _ = portfolio(test.loc[mask], prices, p[mask], threshold, h, universe=[symbol])
                result["by_symbol"].append(stats)
            result["by_regime"] = [dict(regime=r, **classification(test.loc[g.index], p[test.index.isin(g.index)], threshold))
                                   for r, g in test.groupby("regime")]
            results[name], prediction_map[name] = result, p >= threshold
            pred = test[["date", "symbol", "entry_date", "target_date", "label", "future_return", "regime"]].copy()
            pred["probability"], pred["raw_probability"], pred["threshold"] = p, raw, threshold
            pred["predicted_return"] = estimate
            pred["model"], pred["horizon"] = name, h
            all_predictions.append(pred)
            artifact = output / f"{h}d_{name}.joblib"
            save_model(model, artifact, calibrator, threshold)
            if name == chosen:
                recent = full[full.date.eq(full.date.max())].copy()
                live_p = calibrated(calibrator, probability(model, recent))
                for (idx, row), lp in zip(recent.iterrows(), live_p):
                    latest_signals.append({"date": str(row.date.date()), "symbol": row.symbol, "horizon": h,
                                           "model": name, "probability": float(lp), "threshold": threshold,
                                           "direction": "資料漂移待驗證" if abs(lp-threshold) < .05 else ("偏多" if lp >= threshold else "偏空"),
                                           "mode": "shadow_only", "generated_at_utc": report["generated_at_utc"]})
        benchmarks = {
            "buy_hold": np.ones(len(test)),
            "momentum20": (test.ret20 > 0).astype(float).to_numpy(),
            "qqq_momentum20": (test.QQQ_ret20 > 0).astype(float).to_numpy(),
            "random_walk": np.full(len(test), .5),
            "train_majority": np.full(len(test), float(development.label.mean())),
        }
        for name, p in benchmarks.items():
            results[name] = classification(test, p)
            results[name]["portfolio"], daily_map[name] = portfolio(test, prices, p, .5, h, buy_hold=name == "buy_hold")
            prediction_map[name] = p >= .5
        for name in args.models:
            comparisons = {}
            for benchmark in ("buy_hold", "momentum20", "qqq_momentum20"):
                difference = (prediction_map[name] == test.label.to_numpy()).astype(float) - (prediction_map[benchmark] == test.label.to_numpy()).astype(float)
                by_date = pd.Series(difference, index=test.date).groupby(level=0).mean()
                ci = block_interval(by_date, block=max(20, h))
                aligned = pd.concat([daily_map[name], daily_map[benchmark]], axis=1).dropna()
                daily_ci = block_interval(aligned.iloc[:, 0]-aligned.iloc[:, 1], block=max(20, h))
                comparisons[benchmark] = {"accuracy_difference": float(by_date.mean()), "accuracy_95ci": ci,
                                           "mean_daily_net_difference_95ci": daily_ci}
            results[name]["paired_comparisons"] = comparisons
        recent = full[full.date.ge(full.date.max()-pd.Timedelta(days=90))]
        drift = drift_report(development, recent)
        report["horizons"][str(h)] = {"partition": partition, "cv": cv, "params": params_by_name,
                                         "development_selected_model": chosen, "ablation": ablation,
                                         "results": results, "drift": drift,
                                         "drift_warning_count": sum(r["warning"] for r in drift)}
        write_json(output / "progress.json", report)
    for item in baseline:
        if digest(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError("Production baseline changed during research")
    report["baseline_unchanged"] = True
    pd.concat(all_predictions).to_parquet(output / "test_predictions.parquet", index=False)
    report["shadow_signals"] = latest_signals
    report["artifact_files"] = [{"file": str(p.relative_to(output)), "sha256": digest(p)}
                                for p in output.iterdir() if p.is_file() and p.name not in ("progress.json",)]
    write_json(output / "report.json", report)
    write_json(ROOT / "data/cache/prediction_research/report.json", report)
    print(json.dumps({"run_id": run_id, "output": str(output), "baseline_unchanged": True}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=6)
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10, 20])
    p.add_argument("--models", nargs="+", default=["logistic", "xgboost", "lightgbm", "lstm"])
    p.add_argument("--test-start", default="2025-10-01")
    p.add_argument("--test-end", default="2026-09-30")
    args = p.parse_args()
    if any(h not in (1, 5, 10, 20) for h in args.horizons):
        p.error("Supported horizons: 1, 5, 10, 20")
    research(args)


if __name__ == "__main__":
    main()
