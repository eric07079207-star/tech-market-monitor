from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ASSET_GROUPS, DISPLAY_NAMES, HORIZONS


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)


def add_price_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return prices.copy()

    frames = []
    for symbol, data in prices.sort_values(["symbol", "date"]).groupby("symbol", sort=False):
        df = data.copy()
        df["ret_1d"] = df["close"].pct_change()
        for window in [5, 20, 50, 100, 200]:
            df[f"ret_{window}d"] = df["close"].pct_change(window)
            df[f"ma_{window}"] = df["close"].rolling(window).mean()
            df[f"dist_ma_{window}"] = df["close"] / df[f"ma_{window}"] - 1

        df["ma200_slope_20d"] = df["ma_200"].pct_change(20)
        df["realized_vol_20d"] = df["ret_1d"].rolling(20).std() * np.sqrt(252)
        df["realized_vol_pctile_252d"] = df["realized_vol_20d"].rolling(252).rank(pct=True)
        df["ret_z_20d"] = _zscore(df["ret_1d"], 20)
        df["volume_ratio_20d"] = df["volume"] / df["volume"].rolling(20).mean()
        df["volume_z_60d"] = _zscore(np.log1p(df["volume"]), 60)
        df["high_252d"] = df["close"].rolling(252).max()
        df["low_252d"] = df["close"].rolling(252).min()
        df["drawdown_52w"] = df["close"] / df["high_252d"] - 1
        df["range_pct"] = (df["high"] - df["low"]) / df["close"]
        prev_close = df["close"].shift(1)
        true_range = pd.concat(
            [
                (df["high"] - df["low"]),
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr_20d_pct"] = true_range.rolling(20).mean() / df["close"]
        df["gap_pct"] = df["open"] / prev_close - 1

        for horizon in HORIZONS:
            df[f"fwd_ret_{horizon.days}d"] = df["close"].shift(-horizon.days) / df["close"] - 1
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def latest_snapshot(indicators: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for symbol, data in indicators.groupby("symbol", sort=False):
        latest = data.dropna(subset=["close"]).tail(1)
        if latest.empty:
            continue
        row = latest.iloc[0].to_dict()
        row["name"] = DISPLAY_NAMES.get(symbol, symbol)
        row["group"] = ASSET_GROUPS.get(symbol, "其他")
        rows.append(row)
    snapshot = pd.DataFrame(rows)
    if snapshot.empty:
        return snapshot

    columns = [
        "symbol",
        "name",
        "group",
        "date",
        "close",
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "ret_50d",
        "dist_ma_50",
        "dist_ma_200",
        "ma200_slope_20d",
        "drawdown_52w",
        "realized_vol_20d",
        "realized_vol_pctile_252d",
        "ret_z_20d",
        "volume_ratio_20d",
        "volume_z_60d",
        "atr_20d_pct",
        "gap_pct",
    ]
    return snapshot[[col for col in columns if col in snapshot]].sort_values(["group", "symbol"])


def build_macro_wide(macro: pd.DataFrame) -> pd.DataFrame:
    if macro.empty:
        return pd.DataFrame()
    wide = macro.pivot_table(index="date", columns="series", values="value", aggfunc="last").sort_index()
    return wide.ffill()


def regime_summary(indicators: pd.DataFrame, macro: pd.DataFrame) -> dict:
    wide = _wide_close(indicators)
    if wide.empty or "QQQ" not in wide:
        return {"label": "資料不足", "score": np.nan, "drivers": []}

    qqq = indicators[indicators["symbol"] == "QQQ"].dropna(subset=["close"]).tail(1)
    if qqq.empty:
        return {"label": "資料不足", "score": np.nan, "drivers": []}
    q = qqq.iloc[0]

    score = 50
    drivers: list[str] = []
    if q.get("dist_ma_50", np.nan) > 0:
        score += 10
        drivers.append("QQQ above 50DMA")
    else:
        score -= 10
        drivers.append("QQQ below 50DMA")

    if q.get("dist_ma_200", np.nan) > 0:
        score += 15
        drivers.append("QQQ above 200DMA")
    else:
        score -= 15
        drivers.append("QQQ below 200DMA")

    if q.get("ma200_slope_20d", np.nan) > 0:
        score += 10
        drivers.append("200DMA rising")
    else:
        score -= 10
        drivers.append("200DMA falling")

    if q.get("realized_vol_pctile_252d", np.nan) > 0.8:
        score -= 12
        drivers.append("QQQ realized volatility high")
    elif q.get("realized_vol_pctile_252d", np.nan) < 0.35:
        score += 6
        drivers.append("QQQ volatility contained")

    if {"QQQ", "SPY"}.issubset(wide.columns):
        rel = (wide["QQQ"] / wide["SPY"]).pct_change(63).dropna()
        if not rel.empty and rel.iloc[-1] > 0:
            score += 8
            drivers.append("QQQ beating SPY over 3M")
        elif not rel.empty:
            score -= 8
            drivers.append("QQQ lagging SPY over 3M")

    if {"SMH", "QQQ"}.issubset(wide.columns):
        rel = (wide["SMH"] / wide["QQQ"]).pct_change(63).dropna()
        if not rel.empty and rel.iloc[-1] > 0:
            score += 5
            drivers.append("Semis leading QQQ")
        elif not rel.empty:
            score -= 5
            drivers.append("Semis lagging QQQ")

    macro_wide = build_macro_wide(macro)
    if not macro_wide.empty and "BAMLH0A0HYM2" in macro_wide:
        hy = macro_wide["BAMLH0A0HYM2"].dropna()
        if len(hy) > 65:
            change = hy.iloc[-1] - hy.iloc[-64]
            if change > 0.75:
                score -= 10
                drivers.append("HY spreads widening")
            elif change < -0.5:
                score += 6
                drivers.append("HY spreads easing")

    vix = indicators[indicators["symbol"] == "^VIX"].dropna(subset=["close"]).tail(1)
    if not vix.empty:
        vix_level = vix.iloc[0]["close"]
        if vix_level >= 30:
            score -= 15
            drivers.append("VIX stress above 30")
        elif vix_level <= 18:
            score += 5
            drivers.append("VIX calm below 18")

    score = float(np.clip(score, 0, 100))
    if score >= 72:
        label = "長期多頭 / 風險偏低"
    elif score >= 58:
        label = "多頭但需觀察"
    elif score >= 42:
        label = "震盪 / 方向未明"
    elif score >= 28:
        label = "修正壓力升高"
    else:
        label = "熊市或高壓狀態"

    return {"label": label, "score": score, "drivers": drivers[:8]}


def detect_anomalies(snapshot: pd.DataFrame) -> pd.DataFrame:
    if snapshot.empty:
        return snapshot

    rows = []
    for _, row in snapshot.iterrows():
        flags = []
        if abs(row.get("ret_1d", 0)) >= 0.05:
            flags.append("單日漲跌超過 5%")
        if abs(row.get("ret_z_20d", 0)) >= 2:
            flags.append("日報酬 z-score 異常")
        if row.get("volume_ratio_20d", 0) >= 2:
            flags.append("成交量超過 20 日均量 2 倍")
        if row.get("volume_z_60d", 0) >= 2:
            flags.append("成交量 z-score 異常")
        if abs(row.get("gap_pct", 0)) >= 0.03:
            flags.append("跳空超過 3%")
        if row.get("realized_vol_pctile_252d", 0) >= 0.85:
            flags.append("20 日波動位於近一年高檔")
        if row.get("dist_ma_200", 0) < 0 and row.get("dist_ma_50", 0) < 0:
            flags.append("同時低於 50/200 日均線")
        if flags:
            item = row.to_dict()
            item["flags"] = "；".join(flags)
            rows.append(item)

    if not rows:
        return pd.DataFrame(columns=list(snapshot.columns) + ["flags"])
    return pd.DataFrame(rows)


def breadth_table(snapshot: pd.DataFrame) -> pd.DataFrame:
    if snapshot.empty:
        return pd.DataFrame()
    rows = []
    for group, data in snapshot[snapshot["group"].isin(["ETF", "個股"])].groupby("group"):
        rows.append(
            {
                "group": group,
                "count": len(data),
                "above_50dma": int((data["dist_ma_50"] > 0).sum()),
                "above_200dma": int((data["dist_ma_200"] > 0).sum()),
                "avg_1m_return": data["ret_20d"].mean(),
                "avg_drawdown_52w": data["drawdown_52w"].mean(),
            }
        )
    return pd.DataFrame(rows)


def historical_analogs(indicators: pd.DataFrame, target: str = "QQQ", top_n: int = 12) -> pd.DataFrame:
    wide = _wide_close(indicators)
    if wide.empty or target not in wide:
        return pd.DataFrame()

    target_df = indicators[indicators["symbol"] == target].set_index("date").sort_index()
    features = pd.DataFrame(index=target_df.index)
    for col in ["dist_ma_50", "dist_ma_200", "ma200_slope_20d", "ret_20d", "ret_50d", "ret_100d", "drawdown_52w", "realized_vol_20d", "volume_z_60d"]:
        if col in target_df:
            features[col] = target_df[col]

    if {"QQQ", "SPY"}.issubset(wide.columns):
        features["qqq_spy_rel_63d"] = (wide["QQQ"] / wide["SPY"]).pct_change(63)
    if {"SMH", "QQQ"}.issubset(wide.columns):
        features["smh_qqq_rel_63d"] = (wide["SMH"] / wide["QQQ"]).pct_change(63)
    if "^VIX" in wide:
        features["vix_level"] = wide["^VIX"]
    if "HYG" in wide:
        features["hyg_ret_63d"] = wide["HYG"].pct_change(63)
    if "TLT" in wide:
        features["tlt_ret_63d"] = wide["TLT"].pct_change(63)

    features = features.dropna()
    if len(features) < 600:
        return pd.DataFrame()

    latest_date = features.index.max()
    latest = features.loc[latest_date]
    history = features[features.index <= latest_date - pd.Timedelta(days=365)].copy()
    if history.empty:
        return pd.DataFrame()

    means = history.mean()
    stds = history.std().replace(0, np.nan)
    z_history = (history - means) / stds
    z_latest = (latest - means) / stds
    distances = np.sqrt(((z_history - z_latest) ** 2).mean(axis=1)).sort_values()
    chosen = distances.head(top_n)

    rows = []
    for dt, distance in chosen.items():
        row = {
            "date": dt,
            "similarity": float(1 / (1 + distance)),
            "distance": float(distance),
        }
        matched = target_df.loc[dt]
        row["regime_snapshot"] = _compact_regime_text(matched)
        for horizon in HORIZONS:
            row[horizon.label] = matched.get(f"fwd_ret_{horizon.days}d", np.nan)
        rows.append(row)

    return pd.DataFrame(rows)


def analog_stats(analogs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        if horizon.label not in analogs:
            continue
        values = analogs[horizon.label].dropna()
        if values.empty:
            continue
        wins = int((values > 0).sum())
        sample = int(values.count())
        win_rate = float(wins / sample)
        rows.append(
            {
                "horizon": horizon.label,
                "sample": sample,
                "win_rate": win_rate,
                "win_rate_conservative": _wilson_lower_bound(wins, sample),
                "avg_return": float(values.mean()),
                "median_return": float(values.median()),
                "worst_return": float(values.min()),
                "best_return": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def _wilson_lower_bound(wins: int, sample: int, z: float = 1.96) -> float:
    if sample == 0:
        return np.nan
    p = wins / sample
    denominator = 1 + z**2 / sample
    centre = p + z**2 / (2 * sample)
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * sample)) / sample)
    return float((centre - margin) / denominator)


def _wide_close(indicators: pd.DataFrame) -> pd.DataFrame:
    if indicators.empty:
        return pd.DataFrame()
    return indicators.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index().ffill()


def _compact_regime_text(row: pd.Series) -> str:
    parts = []
    if row.get("dist_ma_200", np.nan) > 0:
        parts.append("above 200DMA")
    else:
        parts.append("below 200DMA")
    if row.get("drawdown_52w", np.nan) < -0.15:
        parts.append("deep drawdown")
    if row.get("realized_vol_pctile_252d", np.nan) > 0.8:
        parts.append("high vol")
    if row.get("volume_z_60d", np.nan) > 2:
        parts.append("volume spike")
    return ", ".join(parts)
