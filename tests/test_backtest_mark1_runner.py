from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import torch
except ImportError:
    ML_AVAILABLE = False
else:
    ML_AVAILABLE = True
    from dockdack.mark1_data import Mark1Dataset, barrier_outcomes
    from examples import backtest_mark1 as runner


@unittest.skipUnless(ML_AVAILABLE, "Install the ml extra")
class BacktestMark1RunnerTests(unittest.TestCase):
    def dataset(self):
        bars = np.tile(np.array([100., 102., 98., 100., 100_000.], np.float32), (124, 1))
        dates = np.array(["2024-12-31", "2025-02-10", "2025-02-11", "2025-02-12"],
                         dtype="datetime64[D]").astype(np.int32)
        return Mark1Dataset(
            bars=bars, starts=np.array([0, 31, 62, 93], dtype=np.int64),
            target_dates=dates, symbol_ids=np.array([0, 0, 0, 0], dtype=np.int32),
            target_ohlc=np.array([[100., 102., 98., 100.], [100., 102., 100., 101.],
                                  [100., 100.5, 99.8, 100.2], [100., 100.5, 98., 99.]]),
            splits={"train": np.array([0]), "test": np.array([3])},
            manifest={"symbols": [{"symbol_id": 0, "symbol": "EXAMPLE", "exchange": "KRX"}]})

    def test_full_test_expands_cap_and_keeps_original_indices(self):
        dataset = self.dataset()
        before = dataset.splits["test"].copy()
        np.testing.assert_array_equal(runner.full_test_indices(dataset), [1, 2, 3])
        np.testing.assert_array_equal(dataset.splits["test"], before)
        self.assertEqual(int(dataset.target_dates[0]), int(np.datetime64("2024-12-31", "D").astype(int)))

    def test_full_test_rejects_absent_or_inconsistent_original_holdout(self):
        dataset = self.dataset()
        dataset.splits["test"] = np.array([0])
        with self.assertRaisesRegex(ValueError, "held-out"):
            runner.full_test_indices(dataset)
        dataset.target_dates[:] = dataset.target_dates[0]
        with self.assertRaisesRegex(ValueError, "held-out"):
            runner.full_test_indices(dataset)

    def test_strict_threshold_and_historical_twenty_day_liquidity(self):
        dataset = self.dataset()
        start = int(dataset.starts[1])
        dataset.bars[start:start + 10, 4] = 9_000_000
        dataset.bars[start + 10:start + 30, 4] = np.arange(1, 21) * 1000
        dataset.bars[start + 30, 4] = 88_000_000  # Future target volume is unavailable at OPEN.
        candidates = runner.make_candidates(dataset, np.array([1, 2, 3]), [.50000001, .5, .49999])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["liquidity_shares"], 10_500.)
        self.assertEqual(candidates[0]["symbol_id"], 0)
        self.assertEqual(candidates[0]["date"], int(dataset.target_dates[1]))
        dataset.bars[start + 30, 4] = 0
        dataset.target_ohlc[1] = [1., 1000., .01, 100.]
        self.assertEqual(candidates, runner.make_candidates(dataset, np.array([1, 2, 3]),
                                                          [.50000001, .5, .49999]))

    def test_candidates_use_index_mapping_not_prediction_position(self):
        dataset = self.dataset()
        dataset.symbol_ids[3] = 99
        candidates = runner.make_candidates(dataset, np.array([3, 1]), [.7, .6])
        self.assertEqual([row["symbol_id"] for row in candidates], [99, 0])
        self.assertEqual([row["date"] for row in candidates],
                         [int(dataset.target_dates[3]), int(dataset.target_dates[1])])

    def test_candidates_reject_nonfinite_out_of_range_or_wrong_shape(self):
        dataset = self.dataset()
        for probabilities in ([.6], [[.6, .7]], [.6, float("nan")],
                              [.6, float("inf")], [-.1, .6], [.6, 1.1]):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                runner.make_candidates(dataset, np.array([1, 2]), probabilities)

    def test_names_and_iso_dates_are_added_without_replacing_epoch_dates(self):
        day = int(np.datetime64("2025-02-10", "D").astype(int))
        simulation = {"trades": [{"symbol_id": 4, "entry_date": day, "exit_date": day + 1}]}
        result = runner.attach_names(simulation, [{"symbol_id": 4, "symbol": "TEST", "exchange": "NY"}])
        self.assertIs(result, simulation)
        self.assertEqual(result["trades"][0], {"symbol_id": 4, "symbol": "TEST", "exchange": "NY",
                                               "entry_date": day, "exit_date": day + 1,
                                               "entry_date_iso": "2025-02-10", "exit_date_iso": "2025-02-11"})

    def test_frozen_runner_expands_test_and_exports_auditable_named_ledgers(self):
        dataset = self.dataset()
        probabilities = np.array([.8, .5, .7])
        observed_price_sets = []
        actual_simulate = runner.simulate_portfolio
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training = root / "training" / "domestic"
            training.mkdir(parents=True)
            output = root / "backtest"
            output.mkdir()
            source_dir = root / "db"
            source_dir.mkdir()
            source = source_dir / "domestic_daily_clean.sqlite3"
            source.write_bytes(b"read-only source fixture")
            contract = {"database_sha256": runner.file_hash(source)}
            locked = {"winner": "original_winner", "models": [{"variant": "original_winner"}]}
            (training / "selection_locked.json").write_text(json.dumps(locked), encoding="utf-8")
            (training / "cache_contract.json").write_text(json.dumps(contract), encoding="utf-8")
            variant_dir = training / "original_winner"
            variant_dir.mkdir()
            (variant_dir / "model.pt").write_bytes(b"unchanged checkpoint fixture")
            np.savez(variant_dir / "test_predictions.npz", probabilities=probabilities[-1:])
            prices = {(0, int(dataset.target_dates[index])): tuple(dataset.target_ohlc[index])
                      for index in (1, 2, 3)}
            # A retained but zero-volume non-signal row must never reach the execution engine.
            zero_key = (0, int(dataset.target_dates[1]) + 10)
            prices[zero_key] = (100., 200., 1., 100.)
            sessions = [int(value) for value in dataset.target_dates[1:]]

            class FakeBank:
                def __init__(self, expanded_dataset, device):
                    self.indices = expanded_dataset.splits["test"]
                    np.testing.assert_array_equal(self.indices, [1, 2, 3])
                    ohlc = expanded_dataset.target_ohlc[self.indices]
                    self.parts = {"test": {"outcomes": barrier_outcomes(ohlc[:, 1], ohlc[:, 2],
                                                                         ohlc[:, 3], ohlc[:, 0]),
                                           "dates": expanded_dataset.target_dates[self.indices]}}

                def logits(self, model, split, batch_size):
                    return probabilities.copy()

            def predictor(checkpoint, device):
                return SimpleNamespace(market="domestic", model=object(),
                                       metadata={"dataset": contract, "calibration": {}})

            def checked_simulate(candidates, supplied_prices, supplied_sessions, **kwargs):
                observed_price_sets.append(set(supplied_prices))
                self.assertNotIn(zero_key, supplied_prices)
                return actual_simulate(candidates, supplied_prices, supplied_sessions, **kwargs)

            args = SimpleNamespace(output_dir=output, training_run=training.parent, db_dir=source_dir,
                                   cache_dir=root / "cache", device="cpu", initial_krw=10_000_000.,
                                   initial_usd=10_000., batch_size=8, max_positions=20,
                                   position_fraction=.05, cost_bps=20, volume_fraction=.001)
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(runner, "dataset_cache", return_value=(dataset, contract)))
                stack.enter_context(patch.object(runner, "load_price_panel", return_value=(
                    prices, sessions, {"zero_volume_keys": [list(zero_key)]})))
                stack.enter_context(patch.object(runner, "BatchBank", FakeBank))
                stack.enter_context(patch.object(runner, "Predictor", side_effect=predictor))
                stack.enter_context(patch.object(runner, "calibrated_probability", side_effect=lambda values, _: values))
                stack.enter_context(patch.object(runner, "simulate_portfolio", side_effect=checked_simulate))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                summary = runner.run_market("domestic", args)
            self.assertTrue(summary["completed"])
            self.assertEqual(summary["winner_frozen"], "original_winner")
            self.assertEqual(summary["test_samples"], 3)
            self.assertEqual(summary["previous_test_sample_count"], 1)
            self.assertEqual(summary["results"][0]["original_prediction_max_difference"], 0.)
            self.assertEqual(summary["results"][0]["raw_signals"], 2)
            self.assertEqual(len(summary["cost_sensitivity"]), 8)
            self.assertEqual(len(observed_price_sets), 8)
            ledger = json.loads((output / "domestic/original_winner-carry.json").read_text(encoding="utf-8"))
            self.assertEqual(len(ledger["trades"]), 2)
            self.assertTrue(all(trade["symbol"] == "EXAMPLE" and trade["entry_date_iso"] for trade in ledger["trades"]))
            self.assertEqual(ledger["summary"]["trade_count"], 2)
            self.assertEqual(runner.file_hash(source), contract["database_sha256"])


if __name__ == "__main__":
    unittest.main()
