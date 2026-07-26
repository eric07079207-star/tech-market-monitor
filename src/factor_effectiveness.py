"""Simple, transparent effectiveness reports for existing research factors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CACHE_DIR


FACTOR_EFFECTIVENESS_CACHE = CACHE_DIR / "backtest" / "factor_effectiveness.parquet"
FACTORS = {
    "技術趨勢": "technical_trend_score",
    "市場壓力": "market_pressure_score",
    "20日動能": "qqq_ret_20d",
    "距50日均線": "qqq_dist_ma_50",
    "量能相對值": "qqq_volume_ratio_20",
}


def build_factor_effectiveness(samples: pd.DataFrame) -> pd.DataFrame:
    if samples is None or samples.empty:
        return _empty_report()
    rows: list[dict[str, object]] = []
    for factor_name, column in FACTORS.items():
        if column not in samples:
            continue
        values = pd.to_numeric(samples[column], errors="coerce")
        outcomes = pd.to_numeric(samples.get("future_return_20d"), errors="coerce")
        drawdowns = pd.to_numeric(samples.get("future_max_drawdown_20d"), errors="coerce")
        valid = pd.DataFrame({"value": values, "outcome": outcomes, "drawdown": drawdowns}).dropna(subset=["value", "outcome"])
        if len(valid) < 30:
            continue
        # Terciles are robust to unit differences and are readable in the dashboard.
        try:
            valid["bucket"] = pd.qcut(valid["value"], q=3, labels=["偏低", "中性", "偏高"], duplicates="drop")
        except ValueError:
            continue
        for bucket, group in valid.groupby("bucket", observed=True):
            rows.append(
                {
                    "factor": factor_name,
                    "factor_column": column,
                    "bucket": str(bucket),
                    "sample_count": int(len(group)),
                    "avg_20d_return": float(group["outcome"].mean()),
                    "median_20d_return": float(group["outcome"].median()),
                    "positive_rate": float((group["outcome"] > 0).mean()),
                    "avg_20d_drawdown": float(group["drawdown"].mean()),
                    "correlation_20d_return": float(valid["value"].corr(valid["outcome"])),
                    "coverage_note": "僅市場技術基線；不等同新聞或因果效果",
                }
            )
    return pd.DataFrame(rows) if rows else _empty_report()


def factor_effectiveness_summary(report: pd.DataFrame) -> pd.DataFrame:
    if report is None or report.empty:
        return pd.DataFrame()
    return (
        report.groupby(["factor", "factor_column"], dropna=False)
        .agg(
            分層數=("bucket", "size"),
            總樣本數=("sample_count", "sum"),
            與20日報酬相關性=("correlation_20d_return", "first"),
            最佳分層平均報酬=("avg_20d_return", "max"),
            最差分層平均報酬=("avg_20d_return", "min"),
        )
        .reset_index()
        .sort_values("與20日報酬相關性", key=lambda s: s.abs(), ascending=False)
    )


def load_factor_effectiveness(path: Path | None = None) -> pd.DataFrame:
    path = path or FACTOR_EFFECTIVENESS_CACHE
    if not path.exists():
        return _empty_report()
    try:
        return pd.read_parquet(path)
    except Exception:
        return _empty_report()


def _empty_report() -> pd.DataFrame:
    return pd.DataFrame(columns=["factor", "factor_column", "bucket", "sample_count"])
