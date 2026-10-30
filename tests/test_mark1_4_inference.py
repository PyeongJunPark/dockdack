"""Frozen Mark1.4 bundle, score semantics, and fail-closed loading."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from dockdack.mark1_4_inference import Mark14Predictor


BUNDLE = Path(__file__).resolve().parents[1] / "models" / "mark1_4"


def completed_window() -> np.ndarray:
    return np.asarray(
        [[100 + day, 102 + day, 99 + day, 101 + day, 1_000_000 + day * 1_000]
         for day in range(30)], dtype=np.float32)


class Mark14InferenceTests(unittest.TestCase):
    def test_frozen_market_models_and_score_meanings(self):
        cases = (
            ("domestic", "005930", "KRX", "e5", 43,
             "predicted_net_return_percent", 0.451765888929366, -0.40535250306129456),
            ("us", "AAPL", "ND", "e1", 41,
             "unscaled_pairwise_ranking_score", 0.6831178557872781, 0.3824901878833771),
        )
        for market, symbol, exchange, experiment, seed, metric, threshold, expected_score in cases:
            with self.subTest(market=market):
                predictor = Mark14Predictor(BUNDLE, market)
                scored = predictor.score_one(completed_window(), symbol, exchange)
                self.assertEqual((scored["experiment"], scored["seed"]), (experiment, seed))
                self.assertEqual(scored["score_metric"], metric)
                self.assertEqual(scored["frozen_numeric_score_threshold"], threshold)
                self.assertAlmostEqual(scored["score"], expected_score, places=6)
                self.assertEqual(scored["above_frozen_threshold"], scored["score"] > threshold)
                self.assertNotIn("probability_success", scored)
                self.assertTrue(scored["research_only"])
                self.assertFalse(scored["deployment_allowed"])
                self.assertEqual(scored["selected_universe_size"], 100)
                self.assertTrue(scored["in_training_universe"])
                self.assertFalse(scored["out_of_training_universe"])
                self.assertEqual(predictor.metadata["score_metric"], metric)
                self.assertEqual(json.loads(json.dumps(scored))["score"], scored["score"])

    def test_quote_independent_alias_and_current_universe_drift(self):
        predictor = Mark14Predictor(BUNDLE, "domestic")
        window = completed_window()
        first = predictor.predict(window, symbol="005930", exchange="KRX",
                                  current_price=100)
        later_quote = predictor.predict(window, symbol="005930", exchange="KRX",
                                        current_price=200)
        self.assertEqual(first, later_quote)
        new_member = predictor.score_one(window, "999999", "KRX")
        self.assertTrue(new_member["out_of_training_universe"])
        self.assertFalse(new_member["in_training_universe"])
        self.assertEqual(new_member["score_metric"], first["score_metric"])
        with self.assertRaisesRegex(ValueError, "identity"):
            predictor.score_one(window, "AAPL", "ND")
        with self.assertRaisesRegex(ValueError, "identity"):
            predictor.score_one(window, "005930", "WRONG")
        with self.assertRaisesRegex(ValueError, "identity"):
            predictor.score_many([window, window], [("005930", "KRX"), ("005930", "KRX")])

    def test_invalid_or_incomplete_windows_fail_closed(self):
        predictor = Mark14Predictor(BUNDLE, "us")
        window = completed_window()
        for invalid in (window[:-1], np.full((30, 5), np.nan),
                        np.column_stack((np.zeros(30), window[:, 1:])),
                        np.column_stack((window[:, :4], -np.ones(30)))):
            with self.subTest(shape=np.shape(invalid)):
                with self.assertRaises(ValueError):
                    predictor.score_one(invalid, "AAPL", "ND")

    def test_missing_corrupt_and_resealed_artifacts_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "mark1_4"
            shutil.copytree(BUNDLE, target)
            (target / "domestic-seed43-model.json").unlink()
            with self.assertRaises(ValueError):
                Mark14Predictor(target, "us")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "mark1_4"
            shutil.copytree(BUNDLE, target)
            with (target / "us-seed41-model.json").open("ab") as stream:
                stream.write(b" ")
            with self.assertRaisesRegex(ValueError, "provenance"):
                Mark14Predictor(target, "us")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "mark1_4"
            shutil.copytree(BUNDLE, target)
            manifest = target / "manifest.json"
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["markets"]["domestic"]["selected_symbols"][0]["symbol"] = "999999"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            import hashlib
            (target / "manifest.sha256").write_text(
                hashlib.sha256(manifest.read_bytes()).hexdigest() + "\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "manifest checksum"):
                Mark14Predictor(target, "domestic")


if __name__ == "__main__":
    unittest.main()
