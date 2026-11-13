"""Isolated, retrospective research. Never writes production LSTM artifacts."""
from __future__ import annotations

from copy import deepcopy
import random

import numpy as np
import pandas as pd

SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMD", "META", "GOOGL", "AMZN", "TSLA"]
MARKETS = ["QQQ", "SPY", "SMH", "TLT", "HYG", "^VIX"]
BASE_FEATURES = ["ret1", "ret5", "ret20", "ret60", "dist20", "dist50", "dist200",
                 "vol20", "vol60", "volume_ratio", "range", "gap", "drawdown252"]
MARKET_FEATURES = [f"{s}_{f}" for s in MARKETS for f in ("ret5", "ret20", "vol20")]
MARKET_FEATURES += ["qqq_dist200", "relative20", "relative60"]
FEATURES = BASE_FEATURES + MARKET_FEATURES
SEED = 42
VERSION = "oos-research-v1"


def completed_prices(prices: pd.DataFrame, now=None) -> tuple[pd.DataFrame, dict]:
    """Fail closed on invalid OHLCV; exclude unfinished exchange sessions."""
    import exchange_calendars as xcals
    data = prices.copy()
    data["date"] = pd.to_datetime(data["date"], utc=True).dt.tz_localize(None).dt.normalize()
    now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC"))
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    calendar = xcals.get_calendar("XNYS", start=data.date.min(), end=now.date())
    schedule = calendar.schedule
    closed = schedule.index[schedule["close"] + pd.Timedelta(minutes=30) <= now]
    closed = pd.DatetimeIndex(closed).tz_localize(None)
    numeric = ["open", "high", "low", "close", "volume"]
    for col in numeric:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    tolerance = data.close.abs() * 1e-8
    invalid = (~np.isfinite(data[numeric]).all(axis=1) | data[numeric[:4]].le(0).any(axis=1)
               | data.volume.lt(0) | (data.high + tolerance).lt(data[["open", "close", "low"]].max(axis=1))
               | (data.low - tolerance).gt(data[["open", "close"]].min(axis=1)))
    duplicates = data.duplicated(["date", "symbol"], keep=False)
    if invalid.any() or duplicates.any():
        raise ValueError(f"Price integrity failure: invalid={invalid.sum()}, duplicates={duplicates.sum()}")
    rejected = int((~data.date.isin(closed)).sum())
    return data[data.date.isin(closed)].sort_values(["symbol", "date"]), {
        "unfinished_or_non_session_rows_excluded": rejected,
        "last_completed_session": str(closed[-1].date()),
        "missing_sessions_policy": "no forward-fill of prices or targets",
    }


def build_panel(prices: pd.DataFrame, horizon: int, start="2013-01-01") -> pd.DataFrame:
    """After-close features; target is NEXT open to open h sessions later."""
    qqq = prices.loc[prices.symbol.eq("QQQ"), "date"].sort_values().unique()
    calendar = pd.DatetimeIndex(qqq)
    wide = {s: g.set_index("date").reindex(calendar) for s, g in prices.groupby("symbol")}
    market = pd.DataFrame(index=calendar)
    for symbol in MARKETS:
        close = wide[symbol]["close"]
        for period in (5, 20):
            market[f"{symbol}_ret{period}"] = close.pct_change(period, fill_method=None)
        market[f"{symbol}_vol20"] = close.pct_change(fill_method=None).rolling(20).std()
    qclose = wide["QQQ"]["close"]
    market["qqq_dist200"] = qclose / qclose.rolling(200).mean() - 1
    frames = []
    for symbol in SYMBOLS:
        g = wide[symbol]
        close, volume = g.close, g.volume
        f = pd.DataFrame(index=calendar)
        for period in (1, 5, 20, 60):
            f[f"ret{period}"] = close.pct_change(period, fill_method=None)
        for period in (20, 50, 200):
            f[f"dist{period}"] = close / close.rolling(period).mean() - 1
        for period in (20, 60):
            f[f"vol{period}"] = f.ret1.rolling(period).std()
        f["volume_ratio"] = volume / volume.rolling(20).mean().replace(0, np.nan)
        f["range"] = (g.high - g.low) / close
        f["gap"] = g.open / close.shift() - 1
        f["drawdown252"] = close / close.rolling(252).max() - 1
        f = f.join(market)
        for period in (20, 60):
            f[f"relative{period}"] = f[f"ret{period}"] - qclose.pct_change(period, fill_method=None)
        f["symbol"] = symbol
        f["date"] = calendar
        f["session_id"] = np.arange(len(calendar))
        f["entry_date"] = pd.Series(calendar, index=calendar).shift(-1)
        f["target_date"] = pd.Series(calendar, index=calendar).shift(-(horizon + 1))
        f["future_return"] = g.open.shift(-(horizon + 1)) / g.open.shift(-1) - 1
        f["regime"] = np.select(
            [f["QQQ_vol20"] > .025, f.qqq_dist200 < 0], ["high_volatility", "below_200dma"],
            default="above_200dma")
        # Keep immature rows for live inference, but never manufacture their labels.
        f["label"] = f.future_return.gt(0).astype(float).where(f.future_return.notna())
        frames.append(f)
    data = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    return data[data.date.ge(start)].sort_values(["date", "symbol"]).reset_index(drop=True)


def with_sequences(panel: pd.DataFrame, length=20) -> pd.DataFrame:
    groups = []
    for _, g in panel.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        values = g[FEATURES].to_numpy(dtype=np.float32)
        sequences = [None] * len(g)
        for i in range(length - 1, len(g)):
            seq = values[i - length + 1:i + 1]
            if np.isfinite(seq).all() and g.session_id.iloc[i] - g.session_id.iloc[i-length+1] == length - 1:
                sequences[i] = seq.copy()
        g["sequence"] = sequences
        groups.append(g[g.sequence.notna()])
    return pd.concat(groups).sort_values(["date", "symbol"]).reset_index(drop=True)


def purged_before(data: pd.DataFrame, boundary) -> pd.DataFrame:
    boundary = pd.Timestamp(boundary)
    return data[data.date.lt(boundary) & data.target_date.lt(boundary)].copy()


def temporal_folds(data: pd.DataFrame, n_splits=3):
    from sklearn.model_selection import TimeSeriesSplit
    dates = np.sort(data.date.unique())
    for train_idx, valid_idx in TimeSeriesSplit(n_splits=n_splits).split(dates):
        valid = data[data.date.isin(dates[valid_idx])].copy()
        train = purged_before(data[data.date.isin(dates[train_idx])], valid.date.min())
        if train.empty or train.target_date.max() >= valid.date.min():
            raise ValueError("Non-causal temporal fold")
        yield train, valid


class SequenceClassifier:
    """Small CPU candidate, with train-only scaling and purged early stopping."""
    def __init__(self, columns=FEATURES, epochs=4):
        self.columns = list(columns)
        self.epochs = epochs

    def fit(self, frame):
        import torch
        from torch import nn
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        dates = np.sort(frame.date.unique())
        boundary = dates[int(len(dates) * .85)]
        train = purged_before(frame, boundary)
        valid = frame[frame.date.ge(boundary)]
        self.indices = [FEATURES.index(c) for c in self.columns]
        x = np.stack(train.sequence)[:, :, self.indices]
        self.mean = x.mean(axis=(0, 1))
        self.std = np.maximum(x.std(axis=(0, 1)), 1e-6)
        x = (x - self.mean) / self.std
        vx = self.transform(valid)
        self.model = ResearchLSTM(len(self.columns))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001, weight_decay=.01)
        loss_fn = nn.BCEWithLogitsLoss()
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            torch.tensor(x), torch.tensor(train.label.to_numpy(), dtype=torch.float32)),
            batch_size=256, shuffle=True)
        best, best_loss = None, np.inf
        for _ in range(self.epochs):
            self.model.train()
            for bx, by in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model(bx), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                loss = loss_fn(self.model(torch.tensor(vx)), torch.tensor(valid.label.to_numpy(), dtype=torch.float32)).item()
            if loss < best_loss:
                best, best_loss = deepcopy(self.model.state_dict()), loss
        self.model.load_state_dict(best)
        self.fit_end = str(train.date.max().date())
        return self

    def transform(self, frame):
        return ((np.stack(frame.sequence)[:, :, self.indices] - self.mean) / self.std).astype(np.float32)

    def predict_proba(self, frame):
        import torch
        self.model.eval()
        chunks = []
        x = self.transform(frame)
        with torch.no_grad():
            for start in range(0, len(x), 512):
                chunks.append(torch.sigmoid(self.model(torch.tensor(x[start:start+512]))).numpy())
        p = np.concatenate(chunks)
        return np.c_[1-p, p]


def ResearchLSTM(input_size):
    # Defined lazily so Streamlit can read research results without torch installed.
    import torch
    from torch import nn

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size, 24, batch_first=True)
            self.head = nn.Linear(24, 1)

        def forward(self, x):
            sequence, _ = self.lstm(x)
            return self.head(sequence[:, -1]).squeeze(-1)
    return Network()


def make_model(name, params=None, columns=FEATURES):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    params = params or {}
    if name == "lstm":
        return SequenceClassifier(columns)
    if name == "logistic":
        return make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=.1, max_iter=500))
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=120, n_jobs=2, random_state=SEED,
                             tree_method="hist", eval_metric="logloss", **params)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=120, n_jobs=2, random_state=SEED,
                              verbosity=-1, deterministic=True, force_col_wise=True, **params)
    raise ValueError(name)


def fit_model(name, frame, params=None, columns=FEATURES):
    model = make_model(name, params, columns)
    if name == "lstm":
        model.fit(frame)
    else:
        model.fit(frame[list(columns)], frame.label.astype(int))
    return model


def probability(model, frame, columns=FEATURES):
    return model.predict_proba(frame if isinstance(model, SequenceClassifier) else frame[list(columns)])[:, 1]


def tune(name, development, trials=6):
    import optuna
    from sklearn.metrics import log_loss
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = list(temporal_folds(development))
    def objective(trial):
        params = {}
        if name == "xgboost":
            params = {"max_depth": trial.suggest_int("max_depth", 2, 4),
                      "learning_rate": trial.suggest_float("learning_rate", .015, .08, log=True),
                      "min_child_weight": trial.suggest_int("min_child_weight", 20, 100),
                      "reg_lambda": trial.suggest_float("reg_lambda", 5, 50, log=True)}
        elif name == "lightgbm":
            params = {"num_leaves": trial.suggest_int("num_leaves", 4, 16),
                      "learning_rate": trial.suggest_float("learning_rate", .015, .08, log=True),
                      "min_child_samples": trial.suggest_int("min_child_samples", 100, 400),
                      "reg_lambda": trial.suggest_float("reg_lambda", 5, 50, log=True)}
        scores, metadata = [], []
        for train, valid in folds:
            m = fit_model(name, train, params)
            p = probability(m, valid)
            score = log_loss(valid.label, p, labels=[0, 1])
            scores.append(score)
            metadata.append({"train_rows": len(train), "valid_rows": len(valid),
                             "train_label_end": str(train.target_date.max().date()),
                             "valid_start": str(valid.date.min().date()), "log_loss": score,
                             "accuracy": classification(valid, p)["accuracy"],
                             "balanced_accuracy": classification(valid, p)["balanced_accuracy"],
                             "brier": classification(valid, p)["brier"]})
        trial.set_user_attr("folds", metadata)
        return float(np.mean(scores))
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials if name in ("xgboost", "lightgbm") else 1)
    return study.best_params, {"cv_log_loss": study.best_value, "folds": study.best_trial.user_attrs["folds"],
                               "trials": [{"number": t.number, "value": t.value, "params": t.params} for t in study.trials]}


def logit(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p / (1-p)).reshape(-1, 1)


def calibrate(raw, labels):
    from sklearn.linear_model import LogisticRegression
    if len(np.unique(labels)) < 2:
        raise ValueError("Calibration requires both classes")
    return LogisticRegression(C=1, max_iter=500).fit(logit(raw), labels.astype(int))


def calibrated(calibrator, p):
    return calibrator.predict_proba(logit(p))[:, 1]


def select_threshold(p, labels):
    from sklearn.metrics import balanced_accuracy_score
    grid = sorted(np.arange(.4, .601, .02), key=lambda t: abs(t-.5))
    return float(max(grid, key=lambda t: balanced_accuracy_score(labels, p >= t)))


def classification(frame, p, threshold=.5):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                                 recall_score, f1_score, roc_auc_score, brier_score_loss, log_loss)
    y = frame.label.astype(int).to_numpy()
    pred = p >= threshold
    bins = []
    for low in np.arange(0, 1, .1):
        mask = (p >= low) & (p < low + .1 if low < .9 else p <= 1)
        if mask.any():
            bins.append({"lower": float(low), "count": int(mask.sum()),
                         "predicted": float(p[mask].mean()), "observed": float(y[mask].mean())})
    actionable = np.abs(p-threshold) >= .05
    return {"rows": len(y), "dates": int(frame.date.nunique()),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "up_rate": float(y.mean()), "predicted_up_rate": float(pred.mean()),
            "coverage": float(actionable.mean()),
            "actionable_accuracy": float((pred[actionable] == y[actionable]).mean()) if actionable.any() else None,
            "ece": float(sum(b["count"] * abs(b["predicted"]-b["observed"]) for b in bins) / len(y)),
            "calibration": bins}


def portfolio(frame, prices, p, threshold, horizon, one_way_bps=10., buy_hold=False, universe=None):
    """Long/cash, next-open execution, drifting weights and daily NAV. No shorts."""
    signals = frame[["date", "symbol", "entry_date", "target_date"]].copy()
    universe = SYMBOLS if universe is None else universe
    signals["weight"] = (p >= threshold + .05).astype(float) / len(universe)
    open_prices = prices.pivot(index="date", columns="symbol", values="open")[universe].sort_index()
    calendar = pd.DatetimeIndex(prices.loc[prices.symbol.eq("QQQ"), "date"].sort_values().unique())
    open_prices = open_prices.reindex(calendar)
    signal_dates = pd.DatetimeIndex(sorted(frame.date.unique()))
    rebalance = signal_dates[::horizon]
    end = signals.loc[signals.date.isin(rebalance), "target_date"].max()
    if buy_hold:
        rebalance = signal_dates[:1]
    selected = signals[signals.date.isin(rebalance)]
    start = selected.entry_date.min()
    grid = calendar[(calendar >= start) & (calendar <= end)]
    daily_prices = open_prices.reindex(grid)
    if daily_prices.isna().any().any() or (daily_prices <= 0).any().any():
        raise ValueError("Execution prices missing; do not silently forward-fill")
    weights, nav = np.zeros(len(universe)), 1.
    daily, turnovers = [], []
    cost = one_way_bps / 10000
    for i, date in enumerate(grid[:-1]):
        old_nav = nav
        trade = selected[selected.entry_date.eq(date)]
        turnover = 0.
        if not trade.empty:
            desired = trade.set_index("symbol").weight.reindex(universe, fill_value=0).to_numpy()
            if buy_hold:
                desired[:] = 1 / len(universe)
            turnover = float(abs(desired-weights).sum())
            nav *= 1 - turnover * cost
            weights = desired
        returns = (daily_prices.iloc[i+1] / daily_prices.iloc[i] - 1).to_numpy()
        gross = float(weights @ returns)
        nav *= 1 + gross
        weights = weights * (1+returns) / (1+gross)
        if i == len(grid)-2:
            liquidation = float(weights.sum())
            nav *= 1 - liquidation * cost
            turnover += liquidation
        daily.append(nav/old_nav-1)
        turnovers.append(turnover)
    r = pd.Series(daily, index=grid[1:])
    equity = np.r_[1., np.cumprod(1+r)]
    dd = equity / np.maximum.accumulate(equity) - 1
    return {"net_return": float(equity[-1]-1), "max_drawdown": float(dd.min()),
            "sharpe": float(r.mean()/r.std(ddof=1)*np.sqrt(252)) if r.std(ddof=1) > 0 else None,
            "annualized_return": float(equity[-1]**(252/len(r))-1),
            "daily_marks": len(r), "turnover": float(sum(turnovers)),
            "one_way_cost_bps": one_way_bps, "risk_free_rate": 0.}, r


def block_interval(differences, block=20, repetitions=500):
    """Paired date blocks retain cross-stock and short serial dependence."""
    x = np.asarray(differences, dtype=float)
    if len(x) < block*2:
        return {"low": None, "high": None, "reason": "insufficient_dates"}
    rng = np.random.default_rng(SEED)
    means = []
    for _ in range(repetitions):
        starts = rng.integers(0, len(x), size=int(np.ceil(len(x)/block)))
        indices = np.concatenate([(s+np.arange(block)) % len(x) for s in starts])[:len(x)]
        means.append(x[indices].mean())
    low, high = np.quantile(means, [.025, .975])
    return {"low": float(low), "high": float(high), "block_dates": block, "replicates": repetitions}


def drift_report(train, recent):
    rows = []
    for col in FEATURES:
        base = train[col].dropna().to_numpy()
        current = recent[col].dropna().to_numpy()
        edges = np.unique(np.quantile(base, np.linspace(0, 1, 11)))
        if len(edges) < 3 or len(current) == 0:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        a = np.maximum(np.histogram(base, edges)[0]/len(base), .0001)
        b = np.maximum(np.histogram(current, edges)[0]/len(current), .0001)
        psi = float(((b-a)*np.log(b/a)).sum())
        rows.append({"feature": col, "psi": psi, "warning": psi >= .25})
    return sorted(rows, key=lambda r: r["psi"], reverse=True)
