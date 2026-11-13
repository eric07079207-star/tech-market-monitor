import unittest
import numpy as np
import pandas as pd

from src.prediction_research import (SYMBOLS, MARKETS, FEATURES, completed_prices, build_panel,
    with_sequences, purged_before, temporal_folds, portfolio, classification, block_interval)
from src.lstm import _relative_feature
from src.evaluation import _walk_forward_majority_accuracy, leakage_audit, _by_symbol_metrics, _by_regime_metrics
from scripts.monitor_prediction_research import append_locked, forward_window


def sample_prices(n=700):
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2013-01-01", periods=n)
    frames = []
    for symbol in dict.fromkeys(SYMBOLS + MARKETS):
        close = 100*np.exp(np.cumsum(rng.normal(.0001, .01, n)))
        frames.append(pd.DataFrame({"date": dates, "symbol": symbol, "open": close*.999,
                                    "high": close*1.02, "low": close*.98, "close": close, "volume": 10000.}))
    return pd.concat(frames, ignore_index=True)


class ResearchTests(unittest.TestCase):
    def test_legacy_signal_coverage_consistent_across_groups(self):
        f = pd.DataFrame({"symbol": ["TSLA"]*2, "regime_bucket": ["risk_on"]*2,
                          "label": [1, 0], "predicted_label": [1, 0],
                          "predicted_prob_up": [.51, .1], "future_return": [.01, -.01]})
        self.assertEqual(_by_symbol_metrics(f)[0]["actionable_rows"], 1)
        self.assertEqual(_by_regime_metrics(f)[0]["actionable_rows"], 1)
    def test_forward_predictions_cannot_be_backdated(self):
        import exchange_calendars as xcals
        cal = xcals.get_calendar("XNYS", start="2026-11-01", end="2027-01-31")
        signal = pd.Timestamp("2026-11-12")
        self.assertIsNone(forward_window(cal.schedule, signal, 20, pd.Timestamp("2026-11-13T17:00:00Z")))
        entry, target = forward_window(cal.schedule, signal, 20, pd.Timestamp("2026-11-13T01:00:00Z"))
        self.assertEqual(entry, pd.Timestamp("2026-11-13"))
        self.assertGreater(target, entry)

    def test_first_shadow_prediction_is_immutable(self):
        old = pd.DataFrame({"key": ["one"], "probability": [.55]})
        new = pd.DataFrame({"key": ["one", "two"], "probability": [.99, .44]})
        result = append_locked(old, new)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0].probability, .55)

    def test_legacy_majority_only_uses_matured_labels(self):
        f = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                          "target_date": pd.to_datetime(["2020-02-01"]*3), "label": [1, 1, 0]})
        self.assertIsNone(_walk_forward_majority_accuracy(f, min_history=1))
        f.loc[0, "target_date"] = pd.Timestamp("2020-01-02")
        self.assertEqual(_walk_forward_majority_accuracy(f, min_history=1), 0.)

    def test_live_labels_are_a_health_warning(self):
        result = leakage_audit(pd.DataFrame({"label": [1], "future_return": [.1]}), None)
        self.assertEqual(result["status"], "需檢查")

    def test_missing_relative_date_never_uses_future(self):
        wide = pd.DataFrame({"A": [10, 999], "B": [2, 2]}, index=pd.to_datetime(["2020-01-02", "2020-01-06"]))
        self.assertEqual(_relative_feature(wide, "A", pd.Timestamp("2020-01-03"), "B"), 4)
        self.assertTrue(np.isnan(_relative_feature(wide, "A", pd.Timestamp("2020-01-01"), "B")))

    def test_features_invariant_to_appended_future(self):
        prices = sample_prices()
        cutoff = prices.date.unique()[500]
        before = build_panel(prices[prices.date <= cutoff], 20)
        after = build_panel(prices, 20)
        pd.testing.assert_frame_equal(before[FEATURES], after[after.date <= cutoff][FEATURES].reset_index(drop=True))
        self.assertTrue(before.loc[before.date.eq(cutoff), "label"].isna().all())

    def test_target_is_next_open_not_signal_close(self):
        prices = sample_prices()
        panel = build_panel(prices, 5)
        row = panel[panel.symbol.eq("TSLA")].iloc[300]
        g = prices[prices.symbol.eq("TSLA")].set_index("date")
        self.assertAlmostEqual(row.future_return, g.loc[row.target_date, "open"]/g.loc[row.entry_date, "open"]-1)
        self.assertGreater(row.entry_date, row.date)

    def test_purging_uses_maturity_and_keeps_date_groups(self):
        panel = with_sequences(build_panel(sample_prices(1200), 20)).dropna(subset=["label"])
        for train, valid in temporal_folds(panel):
            self.assertLess(train.target_date.max(), valid.date.min())
            self.assertFalse(set(train.date).intersection(valid.date))
            self.assertTrue(valid.groupby("date").size().eq(8).all())
        cutoff = pd.Timestamp("2016-01-01")
        train = purged_before(panel, cutoff)
        self.assertTrue(train.target_date.lt(cutoff).all())

    def test_finite_sequence_features_only(self):
        p = with_sequences(build_panel(sample_prices(), 1))
        self.assertTrue(all(x.shape == (20, len(FEATURES)) for x in p.sequence))
        self.assertFalse(any("future" in c or "label" in c for c in FEATURES))

    def test_real_nav_drawdown_and_cost(self):
        dates = pd.bdate_range("2020-01-01", periods=5)
        prices = pd.DataFrame([{"date": d, "symbol": s, "open": p} for s in SYMBOLS+["QQQ"]
                               for d, p in zip(dates, [100, 100, 90, 110, 110])])
        frame = pd.DataFrame({"date": [dates[0]], "symbol": ["TSLA"],
                              "entry_date": [dates[1]], "target_date": [dates[3]]})
        stats, returns = portfolio(frame, prices, np.array([1.]), .5, 2, one_way_bps=0, universe=["TSLA"])
        self.assertAlmostEqual(stats["net_return"], .1)
        self.assertAlmostEqual(stats["max_drawdown"], -.1)
        paid, _ = portfolio(frame, prices, np.array([1.]), .5, 2, one_way_bps=10, universe=["TSLA"])
        self.assertAlmostEqual(paid["net_return"], 1.1*.999**2-1)
        self.assertEqual(len(returns), 2)

    def test_neutral_random_walk_stays_cash(self):
        prices = sample_prices()
        p = build_panel(prices, 5).dropna(subset=["label"]).tail(800)
        stats, _ = portfolio(p, prices, np.full(len(p), .5), .5, 5)
        self.assertEqual(stats["net_return"], 0)
        self.assertEqual(stats["turnover"], 0)

    def test_incomplete_session_and_invalid_input(self):
        p = sample_prices(2)
        p["date"] = p["date"].map({pd.Timestamp("2013-01-01"): pd.Timestamp("2026-11-12"),
                                    pd.Timestamp("2013-01-02"): pd.Timestamp("2026-11-13")})
        data, _ = completed_prices(p, now="2026-11-13T17:00:00Z")
        self.assertEqual(data.date.max(), pd.Timestamp("2026-11-12"))
        with self.assertRaises(ValueError):
            completed_prices(pd.concat([p, p.iloc[:1]]), now="2026-11-14T10:00:00Z")

    def test_metric_bounds_and_bootstrap(self):
        f = pd.DataFrame({"label": [0, 1]*30, "date": pd.date_range("2020-01-01", periods=60)})
        metrics = classification(f, np.array([.1, .9]*30))
        self.assertEqual(metrics["accuracy"], 1)
        self.assertAlmostEqual(metrics["brier"], .01)
        interval = block_interval(np.zeros(100))
        self.assertEqual(interval["low"], 0)
        self.assertEqual(interval["high"], 0)


if __name__ == "__main__":
    unittest.main()
