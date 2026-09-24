from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

try:
    import numpy as np
    import torch
except ImportError:
    ML_AVAILABLE = False
else:
    ML_AVAILABLE = True
    from dockdack.mark1_data import (
        FEATURE_NAMES, TARGET, barrier_outcomes, features_from_history, load_dataset,
    )


@unittest.skipUnless(ML_AVAILABLE, "Install the ml extra")
class Mark1DataTests(unittest.TestCase):
    def database(self, path, *, future_symbol=False, duplicates=False):
        from dockdack.clean_daily_dataset import create_schema, write_metadata
        days = np.arange(np.datetime64("2021-10-01"), np.datetime64("2026-04-01"))
        day_strings = [str(value) for value in days]
        targets = ["2021-12-15", "2021-12-16", "2022-01-15", "2022-02-15",
                   "2022-02-16", "2023-01-15", "2023-02-15", "2023-02-16",
                   "2024-01-15", "2024-02-15", "2024-02-16", "2025-01-15",
                   "2025-02-15", "2025-02-16", "2026-02-15"]
        with closing(sqlite3.connect(path)) as connection, connection:
            create_schema(connection)
            write_metadata(connection, {"schema_version": "clean-daily-v1", "build_status": "complete",
                                       "requires_training_samples": True, "market": "domestic",
                                       "policy": {"lookback": 30}, "as_of_exclusive": "2026-04-01",
                                       "source_fingerprints": {"raw": {"sha256": "fixture"}}})
            connection.executemany("INSERT INTO sessions VALUES(?,?)", list(zip(day_strings, range(len(days)))))
            for symbol in (("AAA", "FUTURE") if future_symbol else ("AAA",)):
                connection.execute("INSERT INTO instruments VALUES(?,?,?,?,?,?,?,?,?)",
                                   (symbol, "KRX", symbol, symbol, "KRX", "0", 0, "{}", "2026-04-01"))
                connection.executemany("INSERT INTO daily_bars VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [(symbol, "KRX", day, "100", "102.0000001", "99.2", "100.0000001", 100000,
                      "100000", None, None, None, None, "KRW", "2026-04-01", 1, index + 1, "[]")
                     for index, day in enumerate(day_strings)])
                selected = targets if symbol == "AAA" else ["2025-02-15"]
                samples = []
                for target in selected:
                    index = day_strings.index(target)
                    samples.append((symbol, "KRX", day_strings[index - 30], day_strings[index - 1], target, 0, 1))
                connection.executemany("INSERT INTO training_samples VALUES(?,?,?,?,?,?,?)", samples)
        return targets

    def load(self, path, **overrides):
        arguments = {"start": "2021-10-01", "max_train_samples": 0, "max_eval_samples": 0}
        arguments.update(overrides)
        return load_dataset(path, "domestic", **arguments)

    def test_shapes_masks_and_causal_query(self):
        history = torch.tensor([100., 102., 99.2, 100., 100000.]).repeat(2, 30, 1)
        features = features_from_history(history, torch.tensor([100., 101.]), validate=True)
        self.assertEqual(features.shape, (2, 31, 9))
        self.assertEqual(len(FEATURE_NAMES), 9)
        self.assertTrue(torch.equal(features[0, :30], features[1, :30]))
        self.assertTrue(torch.equal(features[:, :30, 7], torch.ones(2, 30)))
        self.assertTrue(torch.equal(features[:, :30, 8], torch.zeros(2, 30)))
        self.assertTrue(torch.equal(features[:, 30, 1:8], torch.zeros(2, 7)))
        self.assertTrue(torch.equal(features[:, 30, 8], torch.ones(2)))
        self.assertAlmostEqual(float(features[1, 30, 0]), float(np.log(1.01)), places=6)
        self.assertTrue(torch.equal(features[:, 0, 5], torch.zeros(2)))

    def test_log_features_are_scale_invariant_and_volume_centered(self):
        history = torch.tensor([100., 102., 99.2, 100., 100000.], dtype=torch.float64).repeat(1, 30, 1)
        history[0, :, 4] = torch.arange(30, dtype=torch.float64)
        scaled = history.clone()
        scaled[..., :4] *= 10
        original = features_from_history(history, torch.tensor([101.], dtype=torch.float64), validate=True)
        changed = features_from_history(scaled, torch.tensor([1010.], dtype=torch.float64), validate=True)
        torch.testing.assert_close(original, changed)
        self.assertAlmostEqual(float(original[0, :30, 4].mean()), 0, places=12)

    def test_feature_validation_rejects_bad_shapes_prices_and_ohlc(self):
        history = torch.tensor([100., 102., 99.2, 100., 100000.]).repeat(1, 30, 1)
        for entries in (torch.tensor([0.]), torch.tensor([float("nan")])):
            with self.assertRaises(ValueError):
                features_from_history(history, entries, validate=True)
        with self.assertRaises(ValueError):
            features_from_history(history[:, :29], torch.tensor([100.]))
        history[0, 0, 1] = 1
        with self.assertRaises(ValueError):
            features_from_history(history, torch.tensor([100.]), validate=True)

    def test_exact_barriers_and_stop_first_proxy(self):
        result = barrier_outcomes(
            [101, 101, 100.5, 100.5], [99.2, 99.1, 99.1, 99.5], [100, 100, 100, 100.2], 100)
        np.testing.assert_array_equal(result["success"], [True, False, False, False])
        np.testing.assert_array_equal(result["both_touch"], [False, True, False, False])
        np.testing.assert_array_equal(result["take_hit"], [True, True, False, False])
        np.testing.assert_array_equal(result["stop_hit"], [False, True, True, False])
        np.testing.assert_allclose(result["gross_return"], [.01, -.009, -.009, .002])

    def test_barrier_tolerance_is_conservative_and_vectorized(self):
        result = barrier_outcomes(np.array([101 - 1e-12, 101]), [99.2, 99.1 + 1e-12], [100, 100], [100, 100])
        np.testing.assert_array_equal(result["success"], [True, False])
        np.testing.assert_array_equal(result["stop_hit"], [False, True])
        with self.assertRaises(ValueError):
            barrier_outcomes(99, 100, 100, 100)
        with self.assertRaises(ValueError):
            barrier_outcomes(101, 99, 100, 0)

    def test_approved_only_readonly_precise_labels_and_purged_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clean.sqlite3"
            self.database(path, future_symbol=True)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            data = self.load(path)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
            self.assertEqual(data.manifest["selected_symbols"], 1)
            self.assertEqual(data.manifest["excluded_symbols"][0]["symbol"], "FUTURE")
            self.assertEqual(data.manifest["uncapped_split_counts"],
                             {"train": 2, "tune": 2, "calibration": 2, "selection": 2, "test": 3})
            self.assertEqual(data.manifest["purged_split_counts"],
                             {"train": 0, "tune": 1, "calibration": 1, "selection": 1, "test": 1})
            self.assertEqual(len(data.starts), 11)
            self.assertEqual(data.bars.dtype, np.float32)
            self.assertEqual(data.target_ohlc.dtype, np.float64)
            self.assertEqual(data.target_dates.dtype, np.int32)
            self.assertAlmostEqual(data.target_ohlc[0, 1], 102.0000001, places=10)
            self.assertEqual(data.manifest["target"], TARGET)
            self.assertEqual(data.manifest["source_fingerprints"]["raw"]["sha256"], "fixture")
            outcomes = barrier_outcomes(data.target_ohlc[:, 1], data.target_ohlc[:, 2],
                                        data.target_ohlc[:, 3], data.target_ohlc[:, 0])
            self.assertTrue(outcomes["success"].all())  # Original target_up was zero.
            for split, boundary in (("tune", "2021-12-31"), ("calibration", "2022-12-31"),
                                    ("selection", "2023-12-31"), ("test", "2024-12-31")):
                # The fixture has one session daily: target minus 30 = input start.
                input_starts = data.target_dates[data.splits[split]] - 30
                self.assertTrue(np.all(input_starts > np.datetime64(boundary).astype(np.int64)))

    def test_caps_deterministic_and_do_not_change_previous_split_with_future_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clean.sqlite3"
            self.database(path)
            first = self.load(path, max_train_samples=1, max_eval_samples=1)
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("DELETE FROM training_samples WHERE target_date='2026-02-15'")
            second = self.load(path, max_train_samples=1, max_eval_samples=1)
            for split in ("train", "tune", "calibration", "selection"):
                np.testing.assert_array_equal(first.target_dates[first.splits[split]],
                                              second.target_dates[second.splits[split]])
                self.assertEqual(len(first.splits[split]), 1)

    def test_target_values_never_enter_features_and_old_labels_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clean.sqlite3"
            self.database(path)
            data = self.load(path)
            index = data.starts[data.splits["train"][0]]
            historical = torch.tensor(data.bars[index:index + 30]).unsqueeze(0)
            before = features_from_history(historical, torch.tensor([100.]), validate=True)
            data.bars[index + 30] = [100, 1000, .1, 999, 1e9]
            data.target_ohlc[0] = [100, 1000, .1, 999]
            after = features_from_history(torch.tensor(data.bars[index:index + 30]).unsqueeze(0),
                                          torch.tensor([100.]), validate=True)
            torch.testing.assert_close(before, after)
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("UPDATE training_samples SET target_up=1")
            relabeled = self.load(path)
            np.testing.assert_array_equal(data.starts, relabeled.starts)

    def test_rejects_missing_row_in_approved_window_instead_of_bridging(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clean.sqlite3"
            self.database(path)
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("DELETE FROM daily_bars WHERE trade_date='2021-12-01'")
            with self.assertRaises(ValueError):
                self.load(path)

    def test_rejects_invalid_sample_and_incomplete_or_raw_db(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clean.sqlite3"
            self.database(path)
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("UPDATE training_samples SET input_end_date='2021-12-10' WHERE target_date='2021-12-15'")
            with self.assertRaises(ValueError):
                self.load(path)
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("UPDATE metadata SET value='\"building\"' WHERE key='build_status'")
            with self.assertRaises(ValueError):
                self.load(path)
            raw = Path(directory) / "raw.sqlite3"
            with closing(sqlite3.connect(raw)) as connection:
                connection.execute("CREATE TABLE daily_bars(trade_date TEXT)")
            with self.assertRaises(ValueError):
                self.load(raw)

    def test_rejects_unsupported_purge_and_no_training_universe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clean.sqlite3"
            self.database(path)
            with self.assertRaises(ValueError):
                self.load(path, purge_sessions=0)
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("DELETE FROM training_samples WHERE target_date<='2021-12-31'")
            with self.assertRaises(ValueError):
                self.load(path)


if __name__ == "__main__":
    unittest.main()
