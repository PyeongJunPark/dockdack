"""Offline contract checks for six sealed experimental daily-horizon models."""
from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from dockdack.mark1_target_horizon_inference import (
    MODEL_IDS, MarkTargetHorizonPredictor,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "models" / "mark1_target_horizon_v1"


def completed_bars() -> np.ndarray:
    close = 100.0 + np.linspace(0, 8, 30)
    opening = close - .2
    high = close + .5
    low = opening - .5
    volume = np.linspace(100_000, 140_000, 30)
    return np.column_stack((opening, high, low, close, volume))


class TargetHorizonInferenceTest(unittest.TestCase):
    def test_each_market_and_model_loads_and_scores_without_side_effects(self):
        bars = completed_bars()
        for market in ("domestic", "us"):
            for model_id in MODEL_IDS:
                with self.subTest(market=market, model_id=model_id):
                    predictor = MarkTargetHorizonPredictor(BUNDLE, market, model_id)
                    output = predictor.predict(bars, current_price=110.)
                    self.assertEqual(output["strategy_id"], model_id)
                    self.assertEqual(output["market"], market)
                    self.assertGreaterEqual(output["probability_success"], 0)
                    self.assertLessEqual(output["probability_success"], 1)
                    self.assertTrue(np.isfinite(output["expected_net_return"]))
                    self.assertIsNone(output["stop_loss_pct"])
                    self.assertTrue(output["research_only"])
                    self.assertFalse(output["deployment_allowed"])
                    self.assertFalse(output["intraday_path_verified"])
                    self.assertEqual(output["candidate_entry_price"], 110.)
                    self.assertEqual(
                        output["candidate_take_price"],
                        110. * (1 + output["take_profit_pct"] / 100),
                    )

    def test_current_price_is_separate_from_completed_history(self):
        predictor = MarkTargetHorizonPredictor(BUNDLE, "domestic", MODEL_IDS[0])
        first = predictor.predict(completed_bars(), current_price=108.)
        second = predictor.predict(completed_bars(), current_price=112.)
        self.assertNotEqual(first["candidate_take_price"], second["candidate_take_price"])
        self.assertTrue(
            first["probability_success"] != second["probability_success"]
            or first["expected_net_return"] != second["expected_net_return"]
        )

    def test_unfinished_or_missing_bar_and_bad_price_fail_closed(self):
        predictor = MarkTargetHorizonPredictor(BUNDLE, "us", MODEL_IDS[0])
        bars = completed_bars()
        for invalid in (bars[:-1], np.vstack((bars, bars[-1])), bars[:, :4]):
            with self.assertRaises(ValueError):
                predictor.predict(invalid, current_price=110.)
        for price in (None, 0, -1, float("nan")):
            with self.assertRaises(ValueError):
                predictor.predict(bars, current_price=price)

    def test_modified_bundle_manifest_or_checkpoint_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            copy = Path(folder) / "bundle"
            shutil.copytree(BUNDLE, copy)
            (copy / "manifest.sha256").write_text("0" * 64 + "\n", encoding="ascii")
            with self.assertRaises(ValueError):
                MarkTargetHorizonPredictor(copy, "domestic", MODEL_IDS[0])
        with tempfile.TemporaryDirectory() as folder:
            copy = Path(folder) / "bundle"
            shutil.copytree(BUNDLE, copy)
            weight = copy / f"domestic-{MODEL_IDS[0]}.pt"
            with weight.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaises(ValueError):
                MarkTargetHorizonPredictor(copy, "domestic", MODEL_IDS[0])


if __name__ == "__main__":
    unittest.main()
