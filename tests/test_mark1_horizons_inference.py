"""Frozen E4 H3/H5 model seal, timing and GPU-source-score parity."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from dockdack.mark1_horizons_inference import MarkHorizonPredictor
from dockdack.runtime_paths import model_bundle
from dockdack.signals.preopen_series import PREOPEN_MODELS, load_preopen_predictor
from dockdack.signals.prototype_external import MODEL_IDS, PrototypeProcessClient, PrototypeWorker
from dockdack.signal_bridge import prototype_family


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "models" / "mark1_horizons"


class MarkHorizonInferenceTests(unittest.TestCase):
    def test_cuda_reference_scores_reproduce_and_horizons_remain_distinct(self):
        for market in ("domestic", "us"):
            with np.load(BUNDLE / f"{market}-reference.npz", allow_pickle=False) as reference:
                windows = reference["windows"]
                identities = list(zip(reference["symbols"].tolist(),
                                      reference["exchanges"].tolist()))
                expected = reference["seed43_cuda_scores"]
            scores = {}
            for variant, horizon in (("mark1.11", 3), ("mark1.12", 5)):
                with self.subTest(market=market, variant=variant):
                    predictor = MarkHorizonPredictor(BUNDLE, market, variant)
                    results = predictor.score_many(windows, identities)
                    actual = np.asarray([row["score"] for row in results])
                    np.testing.assert_allclose(
                        actual, expected, rtol=0,
                        atol=5e-6 if market == "domestic" else 2e-6,
                    )
                    scores[variant] = actual
                    self.assertEqual(predictor.metadata["horizon_sessions"], horizon)
                    self.assertEqual(predictor.metadata["backtest_exit_ordinal_offset_from_entry"],
                                     horizon - 1)
                    self.assertEqual(predictor.metadata["backtest_entry_stride_sessions"], horizon)
                    self.assertEqual(results[0]["strategy_id"],
                                     "mark1-11-prototype" if horizon == 3 else "mark1-12-prototype")
                    self.assertIs(results[0]["runtime_not_backtest_equivalent"], True)
                    self.assertIs(results[0]["research_qualified"], False)
                    self.assertIs(results[0]["deployment_allowed"], False)
                    self.assertTrue(all(not row["above_frozen_threshold"]
                                        for row in results if row["uncertain_numeric_boundary"]))
            np.testing.assert_array_equal(scores["mark1.11"], scores["mark1.12"])

    def test_invalid_bars_or_identity_fail_closed(self):
        with np.load(BUNDLE / "domestic-reference.npz", allow_pickle=False) as reference:
            window = reference["windows"][0]
            symbol = str(reference["symbols"][0])
            exchange = str(reference["exchanges"][0])
        predictor = MarkHorizonPredictor(BUNDLE, "domestic", "mark1.11")
        one = predictor.score_one(window, symbol, exchange)
        self.assertEqual(one["symbol"], symbol)
        with self.assertRaisesRegex(ValueError, "repeated"):
            predictor.score_many([window, window], [(symbol, exchange), (symbol, exchange)])
        bad = window.copy()
        bad[0, 0] = -1
        with self.assertRaisesRegex(ValueError, "valid, completed"):
            predictor.score_one(bad, symbol, exchange)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            MarkHorizonPredictor(BUNDLE, "domestic", "mark1.13")

    def test_bundle_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "bundle"
            shutil.copytree(BUNDLE, copied)
            with (copied / "domestic-seed43-champion.json").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "provenance"):
                MarkHorizonPredictor(copied, "domestic", "mark1.11")
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "bundle"
            shutil.copytree(BUNDLE, copied)
            with (copied / "manifest.json").open("ab") as stream:
                stream.write(b" ")
            with self.assertRaisesRegex(ValueError, "checksum"):
                MarkHorizonPredictor(copied, "domestic", "mark1.12")

    def test_registered_demo_worker_loads_each_horizon_independently(self):
        self.assertEqual(model_bundle("mark1_horizons").name, "mark1_horizons")
        for model_id, variant in (("mark1-11-prototype", "mark1.11"),
                                  ("mark1-12-prototype", "mark1.12")):
            with self.subTest(model_id=model_id):
                self.assertIn(model_id, MODEL_IDS)
                self.assertEqual(PREOPEN_MODELS[model_id].bundle_directory,
                                 "models/mark1_horizons")
                self.assertEqual(prototype_family(model_id).id, model_id)
                predictor = load_preopen_predictor(model_id, BUNDLE, "domestic")
                self.assertEqual(predictor.metadata["version"], variant)
                worker = PrototypeWorker(model_id, bundle_root=BUNDLE)
                result = worker.dispatch({"schema_version": 1, "model_id": model_id,
                                          "operation": "metadata", "market": "domestic"})
                self.assertEqual(result["version"], variant)

    def test_isolated_child_can_load_both_horizon_bundles_without_orders(self):
        for model_id, variant in (("mark1-11-prototype", "mark1.11"),
                                  ("mark1-12-prototype", "mark1.12")):
            with self.subTest(model_id=model_id):
                client = PrototypeProcessClient(model_id, bundle_root=BUNDLE, timeout=15)
                try:
                    info = client.request("metadata", market="us")
                    self.assertEqual(info["version"], variant)
                    self.assertEqual(info["market"], "us")
                    self.assertTrue(info["bundle_manifest_sha256"])
                finally:
                    client.close()


if __name__ == "__main__":
    unittest.main()
