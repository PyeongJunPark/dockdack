from __future__ import annotations

import sqlite3
import tempfile
import unittest
import io
from contextlib import closing, redirect_stdout
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
    import torch
except ImportError:
    ML_AVAILABLE = False
else:
    ML_AVAILABLE = True
    from examples.train_lstm30 import (
        LOOKBACK, BatchSource, binary_metrics, clean_rows, configure_cudnn, date_number,
        load_dataset, make_sample_indices, parse_args, resolve_device, symbol_rank, train,
    )


@unittest.skipUnless(ML_AVAILABLE, "Install the ml extra")
class Training30Tests(unittest.TestCase):
    def bars(self, start, count, base=100.0):
        dates = np.arange(np.datetime64(start, "D"), np.datetime64(start, "D") + np.timedelta64(count, "D"))
        return [(str(day), base, base * 1.02, base * .98, base, 1000) for day in dates]

    def database(self, path, symbols=("AAA", "BBB")):
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("CREATE TABLE daily_bars (symbol TEXT, exchange TEXT, trade_date TEXT, "
                               "open TEXT, high TEXT, low TEXT, close TEXT, volume INTEGER)")
            for number, symbol in enumerate(symbols):
                connection.executemany("INSERT INTO daily_bars VALUES (?,?,?,?,?,?,?,?)",
                                       [(symbol, "KRX", *bar) for bar in self.bars("2022-10-01", 180, 100 + number * 100)])

    def test_exact_thirty_raw_bars_exclude_target(self):
        dates = np.arange(date_number("2022-01-01"), date_number("2022-01-01") + 31)
        closes = np.r_[np.full(30, 100.), 101.]
        splits, labels = make_sample_indices(dates, closes, "2022-12-31", "2024-12-31")
        np.testing.assert_array_equal(splits["train"], [0])
        self.assertEqual(labels[LOOKBACK], 1)
        self.assertEqual(LOOKBACK, 30)
        closes[-1] = 100.999
        self.assertEqual(make_sample_indices(dates, closes, "2022-12-31", "2024-12-31")[1][-1], 0)

    def test_thirty_bars_alone_cannot_make_a_labeled_training_sample(self):
        dates = np.arange(date_number("2022-01-01"), date_number("2022-01-01") + 30)
        splits, _ = make_sample_indices(dates, np.full(30, 100.), "2022-12-31", "2024-12-31")
        self.assertEqual(sum(map(len, splits.values())), 0)

    def test_small_decimal_exact_one_percent_is_positive(self):
        dates = np.arange(date_number("2022-01-01"), date_number("2022-01-01") + 31)
        closes = np.r_[np.full(30, 0.07), 0.0707]
        _, labels = make_sample_indices(dates, closes, "2022-12-31", "2024-12-31")
        self.assertEqual(labels[-1], 1)
        closes[-1] = 0.0706999999
        _, labels = make_sample_indices(dates, closes, "2022-12-31", "2024-12-31")
        self.assertEqual(labels[-1], 0)

    def test_all_symbols_share_target_calendar_boundaries(self):
        for start, count in (("2022-11-01", 130), ("2022-12-01", 100)):
            dates = np.arange(date_number(start), date_number(start) + count)
            splits, _ = make_sample_indices(dates, np.ones(count), "2022-12-31", "2023-01-31")
            self.assertTrue((dates[splits["train"] + 30] <= date_number("2022-12-31")).all())
            self.assertTrue((dates[splits["validation"] + 30] > date_number("2022-12-31")).all())
            self.assertTrue((dates[splits["validation"] + 30] <= date_number("2023-01-31")).all())
            self.assertTrue((dates[splits["test"] + 30] > date_number("2023-01-31")).all())

    def test_bad_rows_removed_without_interpolation(self):
        rows = self.bars("2022-01-01", 2)
        rows.extend([("bad", 1, 1, 1, 1, 1), ("2022-01-03", 0, 1, 0, 1, 1),
                     ("2022-01-04", 100, 90, 80, 100, 1), ("2022-01-05", 100, 101, 99, 100, -1),
                     ("2022-01-06", "oops", 101, 99, 100, 1),
                     ("2022-01-07", 100, float("nan"), 99, 100, 1), rows[0]])
        dates, bars, _, dropped = clean_rows(rows)
        self.assertEqual(len(dates), 2)
        self.assertEqual(bars.dtype, np.float32)
        self.assertEqual(dropped, {"invalid_date": 1, "duplicate_date": 1, "invalid_ohlcv": 5})

    def test_packed_windows_never_cross_symbol_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "daily.sqlite3"
            self.database(path)
            data = load_dataset(path, "domestic", start="2022-01-01", train_end="2022-12-31",
                                validation_end="2023-01-31", max_symbols=2, min_train_bars=31)
            for starts in data.splits.values():
                np.testing.assert_array_equal(data.symbol_ids[starts], data.symbol_ids[starts + 30])
            source = BatchSource(data, torch.device("cpu"), .25)
            inputs, labels = source.get(data.splits["train"][:3])
            self.assertEqual(tuple(inputs.shape), (3, 30, 5))
            self.assertEqual(tuple(labels.shape), (3,))
            np.testing.assert_array_equal(inputs[0].numpy(), data.bars[:30])

    def test_future_prices_do_not_change_selection_or_training(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "daily.sqlite3"
            self.database(path, ("AAA", "BBB", "CCC"))
            arguments = dict(start="2022-01-01", train_end="2022-12-31",
                             validation_end="2023-01-31", max_symbols=2, min_train_bars=31, seed=4)
            before = load_dataset(path, "domestic", **arguments)
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("UPDATE daily_bars SET open=2000, high=2001, low=1999, close=2000 "
                                   "WHERE trade_date > '2022-12-31'")
            after = load_dataset(path, "domestic", **arguments)
            selected = lambda dataset: [entry["symbol"] for entry in dataset.manifest["symbols"] if entry["selected"]]
            self.assertEqual(selected(before), selected(after))
            np.testing.assert_array_equal(before.splits["train"], after.splits["train"])
            train_starts = before.splits["train"]
            np.testing.assert_array_equal(before.labels[train_starts + 30], after.labels[train_starts + 30])
            np.testing.assert_array_equal(before.bars[train_starts[:, None] + np.arange(30)],
                                          after.bars[train_starts[:, None] + np.arange(30)])

    def test_training_cap_reproducible_and_manifest_retains_full_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "daily.sqlite3"
            self.database(path)
            arguments = dict(start="2022-01-01", train_end="2022-12-31", validation_end="2023-01-31",
                             max_symbols=2, min_train_bars=31, seed=4, max_train_samples=17)
            data = load_dataset(path, "domestic", **arguments)
            again = load_dataset(path, "domestic", **arguments)
            self.assertEqual(len(data.splits["train"]), 17)
            self.assertGreater(data.manifest["uncapped_training_samples"], 17)
            np.testing.assert_array_equal(data.splits["train"], again.splits["train"])
            self.assertEqual(sum(entry["training_samples_used"] for entry in data.manifest["symbols"]), 17)

    def test_domestic_catalog_excludes_etf_and_elw(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "daily.sqlite3"
            self.database(path, ("STOCK", "ETF", "ELW"))
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("CREATE TABLE instruments(symbol TEXT, exchange TEXT, catalog_market_code TEXT)")
                connection.executemany("INSERT INTO instruments VALUES (?, 'KRX', ?)",
                                       [("STOCK", "0"), ("ETF", "8"), ("ELW", "3")])
            data = load_dataset(path, "domestic", start="2022-01-01", train_end="2022-12-31",
                                validation_end="2023-01-31", max_symbols=3, min_train_bars=31)
            self.assertEqual([entry["symbol"] for entry in data.manifest["symbols"]], ["STOCK"])

    def test_us_catalog_excludes_etf(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "daily.sqlite3"
            self.database(path, ("STOCK", "ETF"))
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("UPDATE daily_bars SET exchange = 'ND'")
                connection.execute("CREATE TABLE instruments(symbol TEXT, exchange TEXT, is_etf INTEGER)")
                connection.executemany("INSERT INTO instruments VALUES (?, 'ND', ?)",
                                       [("STOCK", 0), ("ETF", 1)])
            data = load_dataset(path, "us", start="2022-01-01", train_end="2022-12-31",
                                validation_end="2023-01-31", max_symbols=3, min_train_bars=31)
            self.assertEqual([entry["symbol"] for entry in data.manifest["symbols"]], ["STOCK"])

    def test_tiny_cpu_training_saves_a_loadable_checkpoint_and_reports(self):
        from dockdack.ml30 import Predictor
        with tempfile.TemporaryDirectory() as folder:
            path, output = Path(folder) / "daily.sqlite3", Path(folder) / "output"
            self.database(path)
            arguments = parse_args(["--market", "domestic", "--db", str(path),
                                    "--output-dir", str(output), "--device", "cpu",
                                    "--start", "2022-01-01", "--train-end", "2022-12-31",
                                    "--validation-end", "2023-01-31", "--max-symbols", "2",
                                    "--min-train-bars", "31", "--epochs", "1", "--batch-size", "128",
                                    "--hidden-size", "8", "--num-layers", "1", "--threads", "1"])
            with redirect_stdout(io.StringIO()):
                result = train(arguments)
            self.assertEqual(result["completed_epochs"], 1)
            self.assertEqual(result["data"]["selected_symbols"], 2)
            for name in ("best_model.pt", "last_model.pt", "metrics.json", "manifest.json",
                         "history.json", "run_config.json", "test_predictions.npz"):
                self.assertTrue((output / name).is_file(), name)
            restored = Predictor(output / "best_model.pt", device="cpu")
            self.assertEqual(restored.metadata["lookback"], 30)

    def test_missing_database_read_cannot_create_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing.sqlite3"
            with self.assertRaisesRegex(ValueError, "Database not found"):
                load_dataset(path, "domestic")
            self.assertFalse(path.exists())

    def test_auc_ties_and_fixed_thresholds(self):
        metrics = binary_metrics(np.array([.5, .5, .5, .5]), np.array([0, 1, 0, 1]), .5)
        self.assertEqual(metrics["roc_auc"], .5)
        self.assertEqual(metrics["brier_score"], .25)
        self.assertEqual(set(metrics["fixed_thresholds"]), {"0.5", "0.55", "0.6"})
        self.assertIsNone(metrics["fixed_thresholds"]["0.6"]["precision"])

    def test_auc_perfect_and_reversed(self):
        labels = np.array([0, 0, 1, 1])
        self.assertEqual(binary_metrics(np.array([.1, .2, .8, .9]), labels, .5)["roc_auc"], 1.)
        self.assertEqual(binary_metrics(np.array([.9, .8, .2, .1]), labels, .5)["roc_auc"], 0.)

    def test_dates_and_symbol_hash_are_deterministic(self):
        self.assertEqual(symbol_rank(42, "AAA", "KRX"), symbol_rank(42, "AAA", "KRX"))
        self.assertNotEqual(symbol_rank(42, "AAA", "KRX"), symbol_rank(43, "AAA", "KRX"))
        with self.assertRaises(ValueError):
            date_number("20220101")

    def test_cuda_device_has_explicit_index_for_allocator_controls(self):
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.current_device", return_value=0):
            self.assertEqual(resolve_device("cuda"), torch.device("cuda:0"))
            self.assertEqual(resolve_device("auto"), torch.device("cuda:0"))
            self.assertEqual(resolve_device("cpu"), torch.device("cpu"))

    def test_explicit_cuda_never_silently_falls_back_to_cpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CPU fallback is disabled"):
                resolve_device("cuda")
            self.assertEqual(resolve_device("auto"), torch.device("cpu"))

    def test_cudnn_windows_cuda_auto_is_safe_without_changing_model(self):
        original = torch.backends.cudnn.enabled
        try:
            with patch("examples.train_lstm30.platform.system", return_value="Windows"):
                enabled, note = configure_cudnn("auto", torch.device("cuda:0"))
                self.assertFalse(enabled)
                self.assertFalse(torch.backends.cudnn.enabled)
                self.assertIn("dropout remain enabled", note)
                self.assertTrue(configure_cudnn("on", torch.device("cuda:0"))[0])
                self.assertTrue(torch.backends.cudnn.enabled)
                self.assertFalse(configure_cudnn("off", torch.device("cuda:0"))[0])
                self.assertFalse(torch.backends.cudnn.enabled)
        finally:
            torch.backends.cudnn.enabled = original

    def test_cudnn_auto_retains_linux_and_cpu_default(self):
        original = torch.backends.cudnn.enabled
        try:
            with patch("examples.train_lstm30.platform.system", return_value="Linux"):
                self.assertTrue(configure_cudnn("auto", torch.device("cuda:0"))[0])
            with patch("examples.train_lstm30.platform.system", return_value="Windows"):
                self.assertTrue(configure_cudnn("auto", torch.device("cpu"))[0])
            self.assertEqual(parse_args(["--market", "domestic"]).cudnn, "auto")
        finally:
            torch.backends.cudnn.enabled = original


if __name__ == "__main__":
    unittest.main()
