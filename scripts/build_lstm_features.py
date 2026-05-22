from __future__ import annotations

import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lstm import LSTM_FEATURE_CACHE, LSTM_SPLIT_CACHE, build_lstm_feature_table, build_lstm_train_split, save_lstm_feature_table, save_lstm_split
from src.data import PRICE_CACHE


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LSTM features.")
    parser.add_argument("--symbols", nargs="*", default=["TSLA"], help="Symbols to build features for.")
    parser.add_argument("--lookback-years", type=int, default=5, help="Historical lookback window.")
    parser.add_argument("--sample-step", type=int, default=5, help="Sample every N trading days.")
    args = parser.parse_args()

    if not PRICE_CACHE.exists():
        raise RuntimeError("prices.parquet is missing; run scripts/update_data.py first.")

    import pandas as pd

    prices = pd.read_parquet(PRICE_CACHE)
    if "date" in prices:
        prices["date"] = pd.to_datetime(prices["date"])

    features = build_lstm_feature_table(prices=prices, symbols=args.symbols, lookback_years=args.lookback_years, sample_step=args.sample_step)
    if features.empty:
        raise RuntimeError("No LSTM features could be built from cached prices.")

    save_lstm_feature_table(features, LSTM_FEATURE_CACHE)
    split = build_lstm_train_split(features)
    save_lstm_split(split, LSTM_SPLIT_CACHE)

    print(
        f"built lstm features rows={len(features)} split_rows={len(split)} "
        f"feature_version={features['feature_version'].iloc[-1] if 'feature_version' in features else 'n/a'} "
        f"updated_at_utc={datetime.now(timezone.utc).isoformat(timespec='seconds')}"
    )


if __name__ == "__main__":
    main()
