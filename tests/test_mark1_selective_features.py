from __future__ import annotations

import unittest

import numpy as np

from dockdack.mark1_selective_features import FEATURE_NAMES, features_from_history


def history(count=2):
    rng = np.random.default_rng(17)
    open_ = np.exp(4.5 + np.cumsum(rng.normal(0, .006, (count, 30)), axis=1))
    close = open_ * np.exp(rng.normal(0, .004, (count, 30)))
    high = np.maximum(open_, close) * 1.015
    low = np.minimum(open_, close) * .994
    volume = rng.integers(0, 100000, (count, 30))
    return np.stack((open_, high, low, close, volume), axis=2)


def flat(count=1, price=100., volume=0.):
    bars = np.full((count, 30, 5), price, dtype=np.float64)
    bars[..., 4] = volume
    return bars


class SelectiveFeaturesTests(unittest.TestCase):
    def test_schema_dtype_and_finite(self):
        bars = history()
        result = features_from_history(bars, bars[:, -1, 3])
        self.assertEqual(result.shape, (2, 184))
        self.assertEqual(len(FEATURE_NAMES), 184)
        self.assertEqual(len(set(FEATURE_NAMES)), len(FEATURE_NAMES))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())

    def test_empty(self):
        result = features_from_history(np.empty((0, 30, 5)), np.empty(0))
        self.assertEqual(result.shape, (0, len(FEATURE_NAMES)))

    def test_immutable_readonly_inputs_and_flags(self):
        bars, entries = history(), np.array([91., 92.])
        original, original_entries = bars.copy(), entries.copy()
        bars.setflags(write=False)
        entries.setflags(write=False)
        features_from_history(bars, entries)
        np.testing.assert_array_equal(bars, original)
        np.testing.assert_array_equal(entries, original_entries)
        self.assertFalse(bars.flags.writeable)
        self.assertFalse(entries.flags.writeable)

    def test_batch_independence_and_order(self):
        bars, entries = history(3), np.array([89., 92., 99.])
        result = features_from_history(bars, entries)
        single = np.concatenate([features_from_history(bars[i:i+1], entries[i:i+1]) for i in range(3)])
        np.testing.assert_array_equal(result, single)
        np.testing.assert_array_equal(result[::-1], features_from_history(bars[::-1], entries[::-1]))

    def test_query_does_not_change_historical_features(self):
        bars = history(1)
        before = features_from_history(bars, np.array([90.]))
        after = features_from_history(bars, np.array([110.]))
        historical = [i for i, name in enumerate(FEATURE_NAMES) if not name.startswith('query_')]
        np.testing.assert_array_equal(before[:, historical], after[:, historical])
        self.assertNotEqual(before[0, FEATURE_NAMES.index('query_log_gap')],
                            after[0, FEATURE_NAMES.index('query_log_gap')])

    def test_flat_and_zero_volume(self):
        result = features_from_history(flat(), [100.])[0]
        for window in (5, 10, 20, 30):
            self.assertEqual(result[FEATURE_NAMES.index(f'w{window}_past_neither_rate')], 1.)
            self.assertEqual(result[FEATURE_NAMES.index(f'w{window}_zero_volume_fraction')], 1.)
            self.assertEqual(result[FEATURE_NAMES.index(f'w{window}_body_fraction_mean')], 0.)
            self.assertEqual(result[FEATURE_NAMES.index(f'w{window}_close_range_position_mean')], .5)
            self.assertEqual(result[FEATURE_NAMES.index(f'w{window}_trend_efficiency')], 0.)
        self.assertTrue(np.isfinite(result).all())

    def test_historical_barrier_exact_touches_and_both_failure(self):
        bars = flat(4)
        bars[0, :, 1] = 101.  # Exact take, no stop.
        bars[1, :, 2] = 99.1  # Exact stop, no take.
        bars[2, :, 1], bars[2, :, 2] = 101., 99.1
        result = features_from_history(bars, [100.] * 4)
        for window in (5, 10, 20, 30):
            columns = [FEATURE_NAMES.index(f'w{window}_past_{key}_rate')
                       for key in ('take_only', 'stop_only', 'both_touch', 'neither')]
            np.testing.assert_array_equal(result[:, columns], np.eye(4, dtype=np.float32))

    def test_historical_barrier_tolerance(self):
        bars = flat(4)
        bars[0, :, 1] = 101. * (1 - .5e-12)
        bars[1, :, 1] = 101. * (1 - 5e-12)
        bars[2, :, 2] = 99.1 * (1 + .5e-12)
        bars[3, :, 2] = 99.1 * (1 + 5e-12)
        result = features_from_history(bars, [100.] * 4)
        take = FEATURE_NAMES.index('w30_past_take_only_rate')
        stop = FEATURE_NAMES.index('w30_past_stop_only_rate')
        np.testing.assert_array_equal(result[:, take], [1, 0, 0, 0])
        np.testing.assert_array_equal(result[:, stop], [0, 0, 1, 0])

    def test_window_barrier_rates_use_exact_last_bars(self):
        bars = flat()
        bars[:, -5:, 1] = 101.
        result = features_from_history(bars, [100.])[0]
        for window in (5, 10, 20, 30):
            self.assertAlmostEqual(result[FEATURE_NAMES.index(f'w{window}_past_take_only_rate')],
                                   5 / window, places=7)

    def test_price_scaling_invariant_except_declared_levels(self):
        bars = history()
        entries = bars[:, -1, 3] * 1.001
        original = features_from_history(bars, entries)
        scaled_bars = bars.copy()
        scaled_bars[..., :4] *= 123.4
        scaled = features_from_history(scaled_bars, entries * 123.4)
        invariant = [i for i, name in enumerate(FEATURE_NAMES)
                     if name != 'query_log_price_div10' and 'price_volume_proxy' not in name]
        np.testing.assert_allclose(original[:, invariant], scaled[:, invariant], rtol=2e-6, atol=2e-6)

    def test_extreme_positive_prices_and_volumes(self):
        bars = flat(2)
        bars[0, :, :4], bars[1, :, :4] = 1e-300, 1e300
        bars[:, :, 4] = 1e300
        result = features_from_history(bars, [1e300, 1e-300])
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue((np.abs(result) <= 20).all())

    def test_float32_inputs_preserved(self):
        bars = history().astype(np.float32)
        entries = bars[:, -1, 3].copy()
        original = bars.copy()
        result = features_from_history(bars, entries)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(bars, original)

    def test_return_lags(self):
        bars = flat()
        prices = np.exp(np.arange(30) * .01 + 4.)
        bars[0, :, :4] = prices[:, None]
        result = features_from_history(bars, [prices[-1]])[0]
        for lag in (1, 2, 3, 5, 10, 20, 29):
            self.assertAlmostEqual(result[FEATURE_NAMES.index(f'close_log_return_lag{lag}')],
                                   lag * .01, places=7)

    def test_invalid_shapes_and_types(self):
        for bars, entries in ((np.ones((1, 31, 5)), [1]), (flat(), [1, 2]),
                              (flat().astype(str), [1]), (flat(), ['100']),
                              (np.ones((30, 5)), [1]), (flat(), [[100]])):
            with self.subTest(shape=np.shape(bars), entries=entries), self.assertRaises(ValueError):
                features_from_history(bars, entries)
        with self.assertRaises(ValueError):
            features_from_history(flat(), [100], validate='yes')

    def test_nonfinite_and_nonpositive_rejected_even_without_validation(self):
        for bad in (np.nan, np.inf, -np.inf, 0., -1.):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                features_from_history(flat(), [bad], validate=False)
        for field, bad in ((0, np.nan), (0, np.inf), (0, 0.), (4, -1.), (4, np.inf)):
            bars = flat()
            bars[0, 0, field] = bad
            with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                features_from_history(bars, [100], validate=False)

    def test_ohlc_ordering_rejected(self):
        for field, bad in ((0, 102.), (3, 98.), (1, 99.), (2, 101.)):
            bars = flat()
            bars[0, 0, field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                features_from_history(bars, [100])


if __name__ == '__main__':
    unittest.main()
