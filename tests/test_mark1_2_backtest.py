"""Synthetic MK1.2 evaluation safeguards; no production DB, fitting or orders."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dockdack.research_artifacts import sha256_file
from examples import backtest_mark1_2 as bt


def dataset():
    bars = np.tile(np.array([100., 100.5, 99.5, 100., 100000.]), (34, 1))
    return SimpleNamespace(bars=bars, starts=np.arange(4),
        target_dates=np.arange(20000, 20004), symbol_ids=np.zeros(4, dtype=np.int64),
        target_ohlc=np.array([[100., 101., 99.5, 100.5], [100., 101., 99.1, 100.],
                              [100., 100.5, 99.5, 100.2], [100., 100.5, 99., 99.5]]),
        manifest={"symbols": [{"symbol_id": 0, "symbol": "TEST", "exchange": "KRX"}]})


class RecordingPredictor:
    def __init__(self, probability=.8):
        self.calls = []
        self.probability = probability

    def predict_proba(self, history, entries):
        self.calls.append((history.copy(), entries.copy()))
        return np.full(len(entries), self.probability)


def fake_trained():
    return {market: {"completed": True, "pilot": False, "market": market,
                    "deployment_allowed": False, "intraday_path_verified": False}
            for market in bt.MARKETS}


class Mark12BacktestTests(unittest.TestCase):
    def test_predictor_gets_only_past_30_bars_and_candidate_price(self):
        data, predictor = dataset(), RecordingPredictor()
        result = bt.predict_dataset(data, np.array([3, 1]), predictor, batch_size=1, factor=.995)
        np.testing.assert_array_equal(result, [.8, .8])
        np.testing.assert_array_equal(predictor.calls[0][0][0], data.bars[3:33])
        np.testing.assert_array_equal(predictor.calls[1][1], [99.5])
        old = copy.deepcopy(predictor.calls)
        data.target_ohlc[:, 1:] *= 2
        again = RecordingPredictor()
        bt.predict_dataset(data, np.array([3, 1]), again, batch_size=1, factor=.995)
        for first, second in zip(old, again.calls):
            for left, right in zip(first, second):
                np.testing.assert_array_equal(left, right)

    def test_invalid_event_query_and_probabilities_are_rejected(self):
        data = dataset()
        for indices in (np.array([-1]), np.array([4]), np.array([1.]), np.array([[1]])):
            with self.assertRaises(ValueError):
                bt.predict_dataset(data, indices, RecordingPredictor())
        for value in (np.nan, np.inf, -.1, 1.1):
            with self.assertRaises(ValueError):
                bt.predict_dataset(data, np.array([1]), RecordingPredictor(value))
        for factor in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                bt.predict_dataset(data, np.array([1]), RecordingPredictor(), factor=factor)

    def test_sensitivity_is_repeatable_and_not_independent_extra_trades(self):
        indices = np.arange(100)
        picked = bt.sensitivity_indices(indices, 12)
        np.testing.assert_array_equal(picked, bt.sensitivity_indices(indices, 12))
        self.assertEqual(len(np.unique(picked)), 12)
        self.assertTrue(np.all(np.diff(picked) > 0))
        result = bt.price_sensitivity(dataset(), np.arange(4), RecordingPredictor(), batch_size=2)
        self.assertTrue(result["counterfactual_not_real_trades"])
        self.assertFalse(result["probability_and_threshold_refit"])
        self.assertEqual(result["independent_observations"], 4)
        self.assertEqual(result["counterfactual_queries"], 20)
        actual = next(row for row in result["rows"] if row["factor"] == 1)
        self.assertEqual(actual["signals"], 4)
        self.assertEqual(actual["precision"], .25)
        self.assertEqual(actual["both_touch_rate"], .25)
        no_signal = bt.price_sensitivity(dataset(), np.arange(4), RecordingPredictor(.5))
        self.assertTrue(all(row["signals"] == 0 and row["precision"] is None for row in no_signal["rows"]))

    def test_price_panel_checks_exact_source_then_disallows_zero_volume(self):
        data = dataset()
        prices = {(0, int(day)): tuple(ohlc) for day, ohlc in zip(data.target_dates, data.target_ohlc)}
        with patch.object(bt, "load_price_panel", return_value=(prices.copy(), list(data.target_dates),
                                                                {"zero_volume_keys": [[0, 20000]]})):
            retained, _, panel = bt.prepare_prices(data, np.arange(4), Path("unused"), "domestic")
        self.assertNotIn((0, 20000), retained)
        self.assertIn("Unfillable", panel["zero_volume_execution_policy"])
        prices[(0, 20001)] = (100., 101., 99., 100.)
        with patch.object(bt, "load_price_panel", return_value=(prices, [], {"zero_volume_keys": []})):
            with self.assertRaises(ValueError):
                bt.prepare_prices(data, np.arange(4), Path("unused"), "domestic")

    def test_output_cannot_replace_inputs_or_existing_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "outputs/mark1/train"
            model = root / "models/model"
            valid = root / "outputs/mark1/evaluation"
            self.assertEqual(bt.output_path(valid, root, train, model), valid)
            for target in (train, train / "nested", root / "elsewhere", root / "outputs/mark1/cache"):
                with self.assertRaises(ValueError):
                    bt.output_path(target, root, train, model)
            valid.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                bt.output_path(valid, root, train, model)

    def test_incomplete_training_rejected_before_bundle_or_data_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(bt, "_inspect_training", return_value={"summary": {"domestic": {}}}), \
                    patch.object(bt, "_tree_hashes") as read_bundle, patch.object(bt, "load_dataset") as read_data:
                with self.assertRaises(ValueError):
                    bt.verify_inputs(root / "run", root / "bundle", root)
                read_bundle.assert_not_called()
                read_data.assert_not_called()

    def test_input_mutation_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "input"
            source.write_bytes(b"original")
            hashes = {str(source): sha256_file(source)}
            bt.recheck_hashes(hashes)
            source.write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                bt.recheck_hashes(hashes)

    def test_screening_comparison_checks_contract_code_and_checkpoint_seals(self):
        from dockdack.mark1_2_models import MODEL_NAMES
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = {"architectures": list(MODEL_NAMES)}
            frozen = {"protocol": protocol, "code_sha256": {"example.py": "a" * 64}}
            (root / "protocol.json").write_text(json.dumps(frozen), encoding="utf-8")
            summaries = {}
            for market in bt.MARKETS:
                summaries[market] = {"protocol": protocol, "source": {"fixture": True},
                    "data_receipt": {"checked": True}, "selected": "rnn", "research_qualified": False}
                for fold in ("walk_2022", "walk_2024"):
                    for name in MODEL_NAMES:
                        folder = root / market / fold / f"{name}-42"
                        folder.mkdir(parents=True)
                        context = {"market": market, "fold": fold, "source": {"fixture": True},
                                   "data_receipt": {"checked": True}, "code_sha256": frozen["code_sha256"]}
                        contract = dict(context=context, architecture=name, seed=42, pilot=False, protocol=protocol)
                        artifacts = {}
                        for filename in ("model.pt", "predictions.npz", "history.json"):
                            artifact = folder / filename
                            artifact.write_bytes(b"synthetic blob checked for integrity, never deserialized")
                            artifacts[filename] = sha256_file(artifact)
                        values = {"contract.json": contract, "model.sha256.json": {"sha256": artifacts["model.pt"]},
                                  "result.json": dict(completed=True, architecture=name, seed=42,
                                      contract=contract, artifact_sha256=artifacts, parameters=10,
                                      epochs=2, best_epoch=1, selection=dict(signal_count=0, precision=None,
                                      net_mean_return=None, overall={"brier": .2}), qualification={"qualified": False})}
                        for filename, value in values.items():
                            (folder / filename).write_text(json.dumps(value), encoding="utf-8")
            rows, hashes = bt.training_comparison(root, summaries)
            self.assertEqual(sum(map(len, rows.values())), 28)
            self.assertEqual(sum(row["selected_architecture"] for group in rows.values() for row in group), 4)
            bt.recheck_hashes(hashes)
            wrong_contract = root / "domestic/walk_2022/mlp-42/contract.json"
            wrong_contract.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                bt.training_comparison(root, summaries)

    def test_actual_export_contract_can_be_used_for_evaluation(self):
        from test_mark1_2_inference import make_run
        from examples.export_mark1_2 import export_bundle
        import torch
        previous = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with tempfile.TemporaryDirectory(prefix="mark12-export-test-", dir=bt.ROOT / "outputs") as temporary:
                root = Path(temporary)
                run, bundle = root / "run", root / "bundle"
                make_run(run)
                export_bundle(run, bundle)
                trained, predictors, hashes = bt.verify_inputs(run, bundle, bt.ROOT)
                self.assertEqual(set(trained), set(bt.MARKETS))
                self.assertEqual(set(predictors), set(bt.MARKETS))
                self.assertTrue(hashes)
                bt.recheck_hashes(hashes)
                manifest_path = bundle / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["source_run"]["summary_sha256"] = "f" * 64
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(ValueError):
                    bt.verify_inputs(run, bundle, bt.ROOT)
        finally:
            torch.set_num_threads(previous)

    def _run_market(self, root, market="domestic"):
        data = dataset()
        database = root / "fake.db"
        database.write_bytes(b"synthetic immutable source, not SQLite")
        source = {"database_sha256": sha256_file(database)}
        receipt = {"source": {"physical_path": str(database)}, "experiment": {"quarantine": []},
                   "cache_sha256": "fixture"}
        trained = {"data_receipt": receipt, "source": source, "selected": "mlp"}
        prices = {(0, int(day)): tuple(ohlc) for day, ohlc in zip(data.target_dates, data.target_ohlc)}
        with ExitStack() as stack:
            stack.enter_context(patch.object(bt, "load_dataset", return_value=(data, source, receipt)))
            stack.enter_context(patch.object(bt, "read_sessions", return_value=data.target_dates))
            stack.enter_context(patch.object(bt, "common_test_indices", return_value=np.arange(4)))
            stack.enter_context(patch.object(bt, "prepare_prices", return_value=(prices, list(map(int, data.target_dates)), {})))
            stack.enter_context(patch.object(bt, "assert_no_wal"))
            summary = bt.run_market(market, SimpleNamespace(workspace=root, output_dir=root, batch_size=2),
                                    trained, RecordingPredictor(), {})
        return summary

    def test_actual_simulator_and_result_artifacts_end_to_end_synthetic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = self._run_market(root)
            self.assertTrue(summary["completed"])
            self.assertFalse(summary["deployment_allowed"])
            self.assertEqual(summary["classification"]["precision"], .25)
            self.assertEqual(summary["raw_signals"], 4)
            self.assertEqual(len(summary["cost_sensitivity"]), 8)
            self.assertLess(summary["portfolios"]["eod"]["total_return"], 0)
            with np.load(root / "domestic/predictions.npz", allow_pickle=False) as predictions:
                np.testing.assert_array_equal(predictions["labels"], [True, False, False, False])
                self.assertTrue(predictions["both_touch"][1])
            self.assertEqual(len(list((root / "domestic").glob("*-cost*.json"))), 8)
            summaries = {market: summary for market in bt.MARKETS}
            report = bt.report_text(summaries, root)
            self.assertIn("실거래 아님", report)
            self.assertIn("새 테스트", report)
            self.assertIn(root.as_posix(), report)
            self.assertIn("1천만 원", report)

    def test_main_writes_completion_seal_only_after_both_markets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "outputs/mark1/evaluation"
            one = self._run_market(root)
            with ExitStack() as stack:
                stack.enter_context(patch.object(bt, "verify_inputs", return_value=(fake_trained(),
                                    dict.fromkeys(bt.MARKETS, RecordingPredictor()), {})))
                stack.enter_context(patch.object(bt, "training_comparison", return_value=(dict.fromkeys(bt.MARKETS, []), {})))
                stack.enter_context(patch.object(bt, "CODE_FILES", ()))
                stack.enter_context(patch.object(bt, "run_market", side_effect=lambda *args: copy.deepcopy(one)))
                self.assertEqual(bt.main(["--workspace", str(root), "--training-run", "outputs/mark1/train",
                                          "--bundle", "models/model", "--output-dir", str(output)]), 0)
            receipt = json.loads((output / "completed.json").read_text(encoding="utf-8"))
            self.assertTrue(receipt["completed"])
            self.assertEqual(receipt["markets"], list(bt.MARKETS))
            self.assertFalse(receipt["orders_started"])
            self.assertEqual(receipt["output_sha256"]["REPORT.md"], sha256_file(output / "REPORT.md"))


if __name__ == "__main__":
    unittest.main()
