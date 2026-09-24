"""Causal calendar folds and conservative four-class target contract."""
from __future__ import annotations

import importlib.util
from types import SimpleNamespace
import unittest


ML_AVAILABLE = importlib.util.find_spec("numpy") is not None and importlib.util.find_spec("torch") is not None
if ML_AVAILABLE:
    import numpy as np
    from dockdack.mark1_deep_data import CLASS_NAMES, FOLDS, class_targets, make_splits, split_manifest


@unittest.skipUnless(ML_AVAILABLE, "requires optional ml dependencies")
class Mark1DeepDataTests(unittest.TestCase):
    def setUp(self):
        days = np.arange(np.datetime64("2010-01-01"), np.datetime64("2027-01-01"))
        self.sessions = days[np.is_busday(days)].astype(np.int64)

    def dataset(self, pairs):
        return SimpleNamespace(target_dates=np.asarray([np.datetime64(day, "D").astype(np.int64) for symbol, day in pairs]),
                               symbol_ids=np.asarray([symbol for symbol, day in pairs], dtype=np.int32))

    def first_sessions(self, year):
        return self.sessions[(self.sessions >= np.datetime64(f"{year}-01-01").astype(np.int64))
                             & (self.sessions <= np.datetime64(f"{year}-12-31").astype(np.int64))]

    def day_string(self, number):
        return str(np.datetime64(int(number), "D"))

    def test_declared_folds_and_purge_exactly_thirty_sessions(self):
        self.assertEqual(set(FOLDS), {"walk_2022", "walk_2024"})
        for fold_name, fold in FOLDS.items():
            pairs = [(0, fold["train_end"])]
            for key in ("tune_year", "calibration_year", "selection_year"):
                year_days = self.first_sessions(fold[key])
                pairs.extend((0, self.day_string(day)) for day in year_days[:32])
            data = self.dataset(pairs)
            splits = make_splits(data, self.sessions, fold_name)
            np.testing.assert_array_equal(splits["train"], [0])
            for block, split in enumerate(("tune", "calibration", "selection")):
                np.testing.assert_array_equal(splits[split], [1 + block * 32 + 30, 1 + block * 32 + 31])
                admitted = data.target_dates[splits[split]]
                input_days = self.sessions[np.searchsorted(self.sessions, admitted) - 30]
                self.assertTrue(np.all(input_days > np.datetime64(f"{fold[split + '_year'] - 1}-12-31").astype(np.int64)))

    def test_training_universe_before_caps_and_future_exclusion(self):
        pairs = [(0, "2019-12-30"), (1, "2019-12-31"), (2, "2020-12-30"),
                 (0, "2020-03-02"), (1, "2020-03-02"), (2, "2020-03-02"),
                 (0, "2021-03-01"), (1, "2021-03-01"), (2, "2021-03-01"),
                 (0, "2022-03-01"), (1, "2022-03-01"), (2, "2022-03-01"),
                 (0, "2025-03-03"), (3, "2026-03-02")]
        data = self.dataset(pairs)
        splits = make_splits(data, self.sessions, "walk_2022", max_train=1, max_tune=0)
        self.assertEqual(len(splits["train"]), 1)
        for split in ("tune", "calibration", "selection"):
            self.assertEqual(set(data.symbol_ids[splits[split]]), {0, 1})
        self.assertFalse(any(12 in indices or 13 in indices for indices in splits.values()))

    def test_uniform_caps_deterministic_sorted_and_eval_uncapped(self):
        pairs = []
        for year in (2019, 2020, 2021, 2022):
            pairs.extend((0, self.day_string(day)) for day in self.first_sessions(year)[40:140])
        data = self.dataset(pairs)
        first = make_splits(data, self.sessions, "walk_2022", max_train=13, max_tune=17, seed=42)
        second = make_splits(data, self.sessions, "walk_2022", max_train=13, max_tune=17, seed=42)
        changed = make_splits(data, self.sessions, "walk_2022", max_train=13, max_tune=17, seed=43)
        self.assertEqual({name: len(v) for name, v in first.items()}, {"train": 13, "tune": 17, "calibration": 100, "selection": 100})
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
            self.assertTrue(np.all(np.diff(first[name]) > 0))
            self.assertFalse(first[name].flags.writeable)
        self.assertFalse(np.array_equal(first["train"], changed["train"]))
        np.testing.assert_array_equal(first["selection"], changed["selection"])

    def test_no_label_access_or_caller_mutation(self):
        data = self.dataset([(0, "2019-12-31"), (0, "2020-03-02"), (0, "2021-03-01"), (0, "2022-03-01")])
        before_dates, before_symbols, before_sessions = data.target_dates.copy(), data.symbol_ids.copy(), self.sessions.copy()
        make_splits(data, self.sessions, "walk_2022")  # No OHLC attribute is present.
        np.testing.assert_array_equal(data.target_dates, before_dates)
        np.testing.assert_array_equal(data.symbol_ids, before_symbols)
        np.testing.assert_array_equal(self.sessions, before_sessions)
        self.assertTrue(data.target_dates.flags.writeable)
        self.assertTrue(data.symbol_ids.flags.writeable)
        self.assertTrue(self.sessions.flags.writeable)

    def test_future_rows_do_not_change_development_membership(self):
        pairs = [(0, "2019-12-31"), (0, "2020-03-02"), (0, "2021-03-01"), (0, "2022-03-01")]
        initial = make_splits(self.dataset(pairs), self.sessions, "walk_2022", max_train=1, max_tune=1)
        future = make_splits(self.dataset(pairs + [(999, "2025-03-03"), (0, "2026-03-02")]), self.sessions, "walk_2022", max_train=1, max_tune=1)
        for name in initial:
            np.testing.assert_array_equal(initial[name], future[name])

    def test_rejects_duplicate_keys_but_accepts_shared_dates(self):
        make_splits(self.dataset([(0, "2019-12-31"), (1, "2019-12-31")]), self.sessions, "walk_2022")
        for pairs in ([(0, "2019-12-31"), (0, "2019-12-31")],
                      [(0, "2019-12-31"), (1, "2019-12-30"), (0, "2019-12-31")]):
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                make_splits(self.dataset(pairs), self.sessions, "walk_2022")

    def test_rejects_bad_calendar_missing_target_and_missing_history(self):
        data = self.dataset([(0, "2019-12-31")])
        for calendar in (self.sessions[::-1], np.repeat(self.sessions, 2), np.array([], dtype=int), self.sessions.astype(float)):
            with self.assertRaises(ValueError):
                make_splits(data, calendar, "walk_2022")
        with self.assertRaisesRegex(ValueError, "absent"):
            make_splits(self.dataset([(0, "2019-12-29")]), self.sessions, "walk_2022")
        with self.assertRaisesRegex(ValueError, "historical"):
            make_splits(data, self.sessions[self.sessions >= data.target_dates[0]], "walk_2022")

    def test_rejects_bad_fold_caps_dates_and_no_training_samples(self):
        data = self.dataset([(0, "2019-12-31")])
        for name in ("test", "walk_2025", None, []):
            with self.assertRaises(ValueError):
                make_splits(data, self.sessions, name)
        for kwargs in ({"max_train": -1}, {"max_tune": True}, {"seed": -1}, {"seed": 1.5}):
            with self.assertRaises(ValueError):
                make_splits(data, self.sessions, "walk_2022", **kwargs)
        data.target_dates = data.target_dates.astype(float)
        with self.assertRaises(ValueError):
            make_splits(data, self.sessions, "walk_2022")
        with self.assertRaisesRegex(ValueError, "No approved training"):
            make_splits(self.dataset([(0, "2022-03-01")]), self.sessions, "walk_2022")

    def test_class_order_exact_boundaries_and_conservative_rounding(self):
        self.assertEqual(CLASS_NAMES, ("take_only", "stop_only", "both", "neither"))
        ohlc = np.array([[100, 101, 99.2, 100], [100, 100.5, 99.1, 100],
                         [100, 101, 99.1, 100], [100, 100.5, 99.5, 100],
                         [100, 101 - 1e-12, 99.2, 100], [100, 101, 99.1 + 1e-12, 100]])
        result = class_targets(ohlc)
        self.assertEqual(result.shape, (6, 1))
        self.assertEqual(result.dtype, np.int64)
        np.testing.assert_array_equal(result[:, 0], [0, 1, 2, 3, 0, 2])

    def test_price_factor_broadcast_and_structural_extreme_queries(self):
        ohlc = np.array([[100., 101., 99.2, 100.], [100., 100.1, 99.9, 100.]])
        before = ohlc.copy()
        factors = np.array([.99, 1., 1.01])
        result = class_targets(ohlc, factors)
        np.testing.assert_array_equal(result, [[0, 0, 1], [0, 3, 1]])
        np.testing.assert_array_equal(ohlc, before)
        np.testing.assert_array_equal(factors, [.99, 1., 1.01])
        self.assertTrue(ohlc.flags.writeable)
        self.assertTrue(factors.flags.writeable)

    def test_targets_reject_invalid_prices_shapes_factors_and_allow_empty_rows(self):
        for prices in ([100, 101, 99, 100], [[100, 99, 98, 98.5]], [[0, 101, 99, 100]],
                       [[100, 101, 99, 102]], [[100, np.inf, 99, 100]], [[True] * 4]):
            with self.assertRaises(ValueError):
                class_targets(prices)
        for factors in ([], 1., [0], [-1], [np.nan], [np.inf], [[1]], [True]):
            with self.assertRaises(ValueError):
                class_targets([[100, 101, 99.2, 100]], factors)
        self.assertEqual(class_targets(np.empty((0, 4)), [1, 1.005]).shape, (0, 2))

    def test_split_manifest_date_ranges_and_validation(self):
        data = self.dataset([(0, "2019-12-30"), (1, "2019-12-31"), (0, "2020-03-02")])
        splits = make_splits(data, self.sessions, "walk_2022", max_train=0, max_tune=0)
        report = split_manifest(data, splits)
        self.assertEqual(report["train"], {"count": 2, "symbols": 2, "first": "2019-12-30", "last": "2019-12-31"})
        self.assertEqual(report["selection"], {"count": 0, "symbols": 0, "first": None, "last": None})
        for indices in ([1, 0], [0, 0], [-1], [3]):
            with self.assertRaises(ValueError):
                split_manifest(data, {"train": np.asarray(indices)})


if __name__ == "__main__":
    unittest.main()
