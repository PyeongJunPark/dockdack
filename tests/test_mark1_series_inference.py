"""Frozen CUDA-to-CPU score parity and fail-closed bundle contracts."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import torch

from dockdack.mark1_series_inference import Mark1SeriesPredictor, MarkSeriesPredictor


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "models" / "mark1_series"
VERSIONS = ("mark1.5", "mark1.6", "mark1.7")


class Mark1SeriesInferenceTests(unittest.TestCase):
    def test_saved_gpu_scores_reproduce_on_complete_2022_cross_sections(self):
        self.assertIs(MarkSeriesPredictor, Mark1SeriesPredictor)
        for market in ("domestic", "us"):
            with np.load(BUNDLE / f"{market}-reference.npz", allow_pickle=False) as reference:
                windows = reference["windows"]
                identities = list(zip(reference["symbols"].tolist(),
                                      reference["exchanges"].tolist()))
                self.assertGreaterEqual(len(identities), 2)
                for version in VERSIONS:
                    with self.subTest(market=market, version=version):
                        predictor = Mark1SeriesPredictor(BUNDLE, market, version)
                        results = predictor.score_many(windows, identities)
                        actual = np.asarray([row["score"] for row in results], dtype=np.float32)
                        expected = reference[f"{version}_scores"]
                        # cuDNN and CPU LSTM kernels differ slightly despite
                        # identical weights. CNN/attention remain close to the
                        # GPU artifact at the tighter bound.
                        tolerance = 1e-3 if version == "mark1.5" else 3e-5
                        np.testing.assert_allclose(actual, expected,
                                                   rtol=tolerance, atol=tolerance)
                        self.assertEqual(
                            [row["raw_above_frozen_threshold"] for row in results],
                            (expected > predictor.metadata["frozen_numeric_score_threshold"]).tolist(),
                        )
                        self.assertTrue(all(not row["above_frozen_threshold"]
                                            for row in results if row["uncertain_numeric_boundary"]))
                        self.assertTrue(all(row["signal_decision"] == "HOLD_NUMERIC_BOUNDARY"
                                            for row in results if row["uncertain_numeric_boundary"]))
                        self.assertEqual(results[0]["strategy_id"],
                                         f"mark1-{version.split('.')[1]}-prototype")
                        self.assertEqual(results[0]["score_metric"], results[0]["score_unit"])
                        self.assertEqual(results[0]["above_frozen_threshold"],
                                         bool(results[0]["raw_above_frozen_threshold"] and
                                              not results[0]["uncertain_numeric_boundary"]))
                        self.assertIs(results[0]["research_qualified"], False)
                        self.assertIs(results[0]["deployment_allowed"], False)
                        self.assertEqual(predictor.metadata["exit_timing"], "preclose")
                        self.assertEqual(predictor.metadata["exit_after_sessions"], 0)

    def test_ranker_cannot_score_one_and_invalid_inputs_fail_closed(self):
        with np.load(BUNDLE / "domestic-reference.npz", allow_pickle=False) as reference:
            windows = reference["windows"]
            identities = list(zip(reference["symbols"].tolist(),
                                  reference["exchanges"].tolist()))
        ranker = Mark1SeriesPredictor(BUNDLE, "domestic", "mark1.7")
        with self.assertRaisesRegex(ValueError, "single-symbol"):
            ranker.score_one(windows[0], *identities[0])
        with self.assertRaisesRegex(ValueError, "cross-section"):
            ranker.score_many(windows[:1], identities[:1])
        with self.assertRaisesRegex(ValueError, "repeated"):
            ranker.score_many(windows[:2], [identities[0], identities[0]])
        invalid = windows.copy()
        invalid[0, 0, 1] = invalid[0, 0, 2] / 2
        with self.assertRaisesRegex(ValueError, "valid, completed"):
            ranker.score_many(invalid, identities)

    def test_numeric_boundary_is_hold_even_if_raw_score_exceeds_threshold(self):
        with np.load(BUNDLE / "domestic-reference.npz", allow_pickle=False) as reference:
            window = reference["windows"][0]
            identity = (str(reference["symbols"][0]), str(reference["exchanges"][0]))
        predictor = Mark1SeriesPredictor(BUNDLE, "domestic", "mark1.5")
        target = predictor.metadata["frozen_numeric_score_threshold"] + (
            predictor.metadata["cpu_numeric_guard_margin"] / 2)

        class ConstantScore(torch.nn.Module):
            def forward(self, values):
                return torch.full((len(values),), target, dtype=torch.float32)

        predictor._model = ConstantScore()
        score = predictor.score_one(window, *identity)
        self.assertTrue(score["raw_above_frozen_threshold"])
        self.assertTrue(score["uncertain_numeric_boundary"])
        self.assertFalse(score["above_frozen_threshold"])
        self.assertEqual(score["signal_decision"], "HOLD_NUMERIC_BOUNDARY")

    def test_modified_weight_or_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "bundle"
            shutil.copytree(BUNDLE, copied)
            selected_weight = copied / "domestic-mark1.5-seed42-weights.npz"
            with selected_weight.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum|provenance"):
                Mark1SeriesPredictor(copied, "domestic", "mark1.5")
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "bundle"
            shutil.copytree(BUNDLE, copied)
            with (copied / "manifest.json").open("ab") as stream:
                stream.write(b" ")
            with self.assertRaisesRegex(ValueError, "checksum"):
                Mark1SeriesPredictor(copied, "domestic", "mark1.5")


if __name__ == "__main__":
    unittest.main()
