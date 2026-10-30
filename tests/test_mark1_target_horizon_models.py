"""Synthetic input-contract tests; not a trading or profitability test."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from dockdack.mark1_target_horizon_models import (
    LOOKBACKS, MODEL_FAMILIES, build_model, sequence_features,
    tabular_features, validate_completed_bars,
)


def _bars(lookback: int, batch: int = 2) -> np.ndarray:
    index = np.arange(lookback, dtype=np.float64)
    opening = 100.0 + index[None, :] + np.arange(batch)[:, None] * 10.0
    high = opening + 3.0
    low = opening - 2.0
    close = opening + 1.0
    volume = np.full_like(opening, 1000.0)
    return np.stack((opening, high, low, close, volume), axis=-1)


class TargetHorizonModelsTest(unittest.TestCase):
    def test_every_lookback_and_family_has_common_output_and_gradients(self):
        torch.manual_seed(7)
        for lookback in LOOKBACKS:
            bars = _bars(lookback)
            sequence = sequence_features(bars, lookback, bars[:, -1, 3] * 1.01)
            self.assertEqual(sequence.shape, (2, lookback, 6))
            self.assertEqual(sequence.dtype, np.float32)
            np.testing.assert_array_equal(tabular_features(bars, lookback,
                                                            bars[:, -1, 3] * 1.01),
                                          sequence.reshape(2, lookback * 6))
            for family in MODEL_FAMILIES:
                with self.subTest(lookback=lookback, family=family):
                    model = build_model(family, lookback)
                    output = model(torch.from_numpy(sequence))
                    self.assertEqual(tuple(output.shape), (2, 2))
                    self.assertTrue(bool(torch.isfinite(output).all()))
                    output.sum().backward()
                    self.assertTrue(any(parameter.grad is not None
                                        for parameter in model.parameters()))

    def test_rejects_malformed_raw_bars_and_appended_future_bar(self):
        base = _bars(20)
        invalid = []
        invalid.append(base[:, :-1])
        invalid.append(np.concatenate((base, base[:, -1:, :]), axis=1))
        invalid.append(base[0])
        for col, new_value in ((0, 0.0), (1, 90.0), (2, 200.0),
                               (3, -1.0), (4, -1.0), (4, np.nan), (0, np.inf)):
            changed = base.copy()
            changed[0, 0, col] = new_value
            invalid.append(changed)
        for bars in invalid:
            with self.subTest(shape=bars.shape, first=bars.reshape(-1)[0]):
                with self.assertRaises(ValueError):
                    validate_completed_bars(bars, 20)
                with self.assertRaises(ValueError):
                    sequence_features(bars, 20, 123.0)
        for lookback in (0, 11, 40, True):
            with self.assertRaises(ValueError):
                sequence_features(base, lookback, 123.0)
        for query in (0.0, float("nan"), [100.0]):
            with self.assertRaises(ValueError):
                sequence_features(base, 20, query)

    def test_earlier_features_do_not_read_later_bars(self):
        base = _bars(20)
        changed = base.copy()
        changed[:, 15:, 0:4] *= 2.0
        changed[:, 15:, 4] *= 100.0
        original = sequence_features(base, 20, 130.0)
        later = sequence_features(changed, 20, 130.0)
        np.testing.assert_array_equal(original[:, :15], later[:, :15])
        self.assertFalse(np.array_equal(original[:, 15:], later[:, 15:]))

    def test_query_price_is_the_only_new_session_information(self):
        bars = _bars(20)
        cheaper = sequence_features(bars, 20, np.asarray([120.0, 130.0]))
        dearer = sequence_features(bars, 20, np.asarray([130.0, 140.0]))
        np.testing.assert_array_equal(cheaper[..., :5], dearer[..., :5])
        np.testing.assert_array_equal(cheaper[:, :-1, 5], dearer[:, :-1, 5])
        self.assertFalse(np.array_equal(cheaper[:, -1, 5], dearer[:, -1, 5]))
        model = build_model("linear", 20)
        with torch.no_grad():
            model.network.weight.zero_()
            model.network.bias.zero_()
            model.network.weight[0, -1] = 1.0
        low_score = model(torch.from_numpy(cheaper))
        high_score = model(torch.from_numpy(dearer))
        self.assertFalse(torch.equal(low_score, high_score))

    def test_network_rejects_wrong_shape_or_nonfinite_features(self):
        model = build_model("cnn", 20)
        for features in (torch.zeros(20, 6), torch.zeros(2, 21, 6),
                         torch.zeros(0, 20, 6), torch.zeros(2, 20, 6, dtype=torch.int32),
                         torch.full((2, 20, 6), float("nan"))):
            with self.subTest(shape=tuple(features.shape), dtype=features.dtype):
                with self.assertRaises(ValueError):
                    model(features)
        with self.assertRaises(ValueError):
            build_model("transformer", 20)
