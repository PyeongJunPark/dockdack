"""Small synthetic, broker-free contracts for the Mark1.5--1.7 pipeline."""

from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from dockdack.mark1_4_evolution import EvolutionSamples
from dockdack.mark1_series_models import SCORE_UNITS, fit_series_variant


def _samples() -> EvolutionSamples:
    windows, dates, ordinals, symbols, opened, closed = [], [], [], [], [], []
    for day in range(14):
        for symbol in range(5):
            step = np.arange(30, dtype=np.float32)
            price = 40 + symbol * 5 + day * .1 + step * (.05 + symbol * .004)
            daily_open = price
            daily_close = price * (1 + .001 * ((day + symbol) % 3 - 1))
            window = np.stack((daily_open,
                               np.maximum(daily_open, daily_close) * 1.005,
                               np.minimum(daily_open, daily_close) * .995,
                               daily_close,
                               100_000 + step * 50 + symbol * 3_000 + day * 100),
                              axis=1)
            windows.append(window)
            dates.append(f"2020-01-{day + 1:02d}")
            ordinals.append(day)
            symbols.append(symbol)
            entry = float(price[-1] + .25)
            opened.append(entry)
            closed.append(entry * (1 + .008 * (((day * 3 + symbol) % 5) - 2)))
    return EvolutionSamples(
        windows=np.asarray(windows, dtype=np.float32),
        target_dates=np.asarray(dates, dtype="U10"),
        target_ordinals=np.asarray(ordinals, dtype=np.int32),
        symbol_ids=np.asarray(symbols, dtype=np.int32),
        entry_open=np.asarray(opened, dtype=np.float64),
        exit_close=np.asarray(closed, dtype=np.float64),
        source={"market": "domestic", "research_only": True},
    )


class Mark1SeriesTests(unittest.TestCase):
    def test_variant_scores_are_finite_and_only_train_targets_affect_fit(self):
        samples = _samples()
        train = np.flatnonzero(samples.target_ordinals < 9)
        for variant in ("mark1.5", "mark1.6", "mark1.7"):
            with self.subTest(variant=variant):
                first = fit_series_variant(samples, train, variant, seed=7,
                                           device="cpu", epochs=1, batch_size=24,
                                           day_batch_size=3, hidden=8,
                                           test_only_allow_cpu=True)
                # Change every later price outcome without changing any
                # completed t bar. It must not affect scaler or scores.
                later_open = samples.entry_open.copy()
                later_close = samples.exit_close.copy()
                later_open[45:] = 999.0
                later_close[45:] = 1.0
                changed = replace(samples, entry_open=later_open,
                                  exit_close=later_close)
                second = fit_series_variant(changed, train, variant, seed=7,
                                            device="cpu", epochs=1, batch_size=24,
                                            day_batch_size=3, hidden=8,
                                            test_only_allow_cpu=True)
                self.assertEqual(first.scores.shape, (70,))
                self.assertTrue(np.isfinite(first.scores).all())
                np.testing.assert_array_equal(first.scores, second.scores)
                self.assertEqual(first.artifact["score_unit"], SCORE_UNITS[variant])
                self.assertIs(first.artifact["research_only"], True)
                self.assertIs(first.artifact["deployment_allowed"], False)
                self.assertIs(first.artifact["test_only_cpu_override"], True)
                for key in first.state:
                    np.testing.assert_array_equal(first.state[key], second.state[key])
                if variant == "mark1.6":
                    self.assertTrue(np.all((first.scores >= 0) & (first.scores <= 1)))

    def test_invalid_inputs_fail_closed(self):
        samples = _samples()
        train = np.arange(45)
        with self.assertRaisesRegex(ValueError, "variant"):
            fit_series_variant(samples, train, "mark1.8", device="cpu")
        with self.assertRaisesRegex(ValueError, "unique"):
            fit_series_variant(samples, np.array([0, 0, 1]), "mark1.5", device="cpu")
        with self.assertRaisesRegex(ValueError, "multiple"):
            fit_series_variant(samples, train, "mark1.7", hidden=9, device="cpu")
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            fit_series_variant(samples, train, "mark1.5", device="cpu")
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            fit_series_variant(samples, train, "mark1.5", device="auto")
        with patch("dockdack.mark1_series_models.torch.cuda.is_available",
                   return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA requested but unavailable"):
                fit_series_variant(samples, train, "mark1.5")


if __name__ == "__main__":
    unittest.main()
