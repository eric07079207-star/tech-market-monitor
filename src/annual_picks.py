from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ANNUAL_PICKS_2026


START_DATE = pd.Timestamp("2026-05-18")


def annual_picks_table(prices: pd.DataFrame) -> pd.DataFrame:
    picks = pd.DataFrame(ANNUAL_PICKS_2026)
    if prices.empty:
        return picks
    rows = []
    wide = prices.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index().ffill()
    for pick in ANNUAL_PICKS_2026:
        ticker = pick["ticker"]
        if ticker not in wide:
            row = pick.copy()
            row.update(_empty_metrics())
            rows.append(row)
            continue
        series = wide[ticker].dropna()
        if series.empty:
            row = pick.copy()
            row.update(_empty_metrics())
            rows.append(row)
            continue
        start_slice = series[series.index >= START_DATE]
        if start_slice.empty:
            start_price = float(series.iloc[-1])
            current = start_price
            period = series.tail(1)
        else:
            start_price = float(start_slice.iloc[0])
            current = float(start_slice.iloc[-1])
            period = start_slice
        total_return = current / start_price - 1 if start_price else np.nan
        max_drawdown = (period / period.cummax() - 1).min() if len(period) else np.nan
        rel_spy = _relative_return(wide, ticker, "SPY", START_DATE)
        rel_qqq = _relative_return(wide, ticker, "QQQ", START_DATE)
        row = pick.copy()
        row.update(
            {
                "selected_date": START_DATE.date().isoformat(),
                "selected_price": start_price,
                "current_price": current,
                "return_since_selected": total_return,
                "relative_spy": rel_spy,
                "relative_qqq": rel_qqq,
                "max_drawdown": float(max_drawdown) if pd.notna(max_drawdown) else np.nan,
                "status": _status(total_return, max_drawdown),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def annual_picks_summary(table: pd.DataFrame) -> dict:
    if table.empty or "return_since_selected" not in table:
        return {"avg_return": np.nan, "win_rate": np.nan, "best": "n/a", "worst": "n/a", "avg_rel_qqq": np.nan}
    returns = table["return_since_selected"].dropna()
    if returns.empty:
        return {"avg_return": np.nan, "win_rate": np.nan, "best": "n/a", "worst": "n/a", "avg_rel_qqq": np.nan}
    best = table.sort_values("return_since_selected", ascending=False).iloc[0]
    worst = table.sort_values("return_since_selected", ascending=True).iloc[0]
    return {
        "avg_return": float(returns.mean()),
        "win_rate": float((returns > 0).mean()),
        "best": best["ticker"],
        "worst": worst["ticker"],
        "avg_rel_qqq": float(table["relative_qqq"].dropna().mean()) if table["relative_qqq"].notna().any() else np.nan,
    }


def _relative_return(wide: pd.DataFrame, ticker: str, benchmark: str, start: pd.Timestamp) -> float:
    if ticker not in wide or benchmark not in wide:
        return np.nan
    pair = wide[[ticker, benchmark]].dropna()
    pair = pair[pair.index >= start]
    if pair.empty:
        return np.nan
    stock_ret = pair[ticker].iloc[-1] / pair[ticker].iloc[0] - 1
    bench_ret = pair[benchmark].iloc[-1] / pair[benchmark].iloc[0] - 1
    return float(stock_ret - bench_ret)


def _status(total_return: float, max_drawdown: float) -> str:
    if pd.notna(max_drawdown) and max_drawdown <= -0.25:
        return "高風險觀察"
    if pd.notna(total_return) and total_return >= 0.15:
        return "續列觀察"
    if pd.notna(total_return) and total_return <= -0.15:
        return "降低優先"
    return "觀察中"


def _empty_metrics() -> dict:
    return {
        "selected_date": START_DATE.date().isoformat(),
        "selected_price": np.nan,
        "current_price": np.nan,
        "return_since_selected": np.nan,
        "relative_spy": np.nan,
        "relative_qqq": np.nan,
        "max_drawdown": np.nan,
        "status": "資料不足",
    }
