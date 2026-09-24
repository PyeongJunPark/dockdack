from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from dockdack.mark1_0504_data import (
    BARRIER_TOLERANCE, FEATURE_NAMES, TARGET, barrier_outcomes,
    class_targets, event_data, features_from_history,
)
from dockdack.mark1_selective_features import (
    FEATURE_NAMES as OLD_FEATURE_NAMES,
    features_from_history as old_features,
)


def flat(count=1):
    result = np.full((count, 30, 5), 100., dtype=np.float64)
    result[..., 4] = 1000.
    return result


class HalfBarrierOutcomeTests(unittest.TestCase):
    def test_exact_touch_all_four_classes_and_stop_first(self):
        ohlc = np.array([[100, 100.5, 99.7, 100], [100, 100.4, 99.6, 100],
                         [100, 100.5, 99.6, 100], [100, 100.4, 99.7, 100.2]])
        outcome = barrier_outcomes(ohlc[:, 1], ohlc[:, 2], ohlc[:, 3], ohlc[:, 0])
        np.testing.assert_array_equal(outcome['success'], [1, 0, 0, 0])
        np.testing.assert_array_equal(outcome['both_touch'], [0, 0, 1, 0])
        np.testing.assert_allclose(outcome['gross_return'], [.005, -.004, -.004, .002])
        np.testing.assert_array_equal(class_targets(ohlc), [[0], [1], [2], [3]])

    def test_exact_touch_across_currency_price_scales(self):
        entries = np.array([.001, 1., 236.48, 84700., 1e8])
        take = barrier_outcomes(entries * 1.005, entries, entries, entries)
        stop = barrier_outcomes(entries, entries * .996, entries, entries)
        self.assertTrue(take['success'].all())
        self.assertTrue(stop['stop_hit'].all())
        self.assertFalse(stop['success'].any())

    def test_numerical_tolerance_declared_and_near_misses_stay_misses(self):
        high = 100.5 * np.array([1 - .5e-12, 1 - 5e-12, 1 - 1e-8])
        result = barrier_outcomes(high, 100., 100., 100.)
        np.testing.assert_array_equal(result['take_hit'], [1, 0, 0])
        low = 99.6 * np.array([1 + .5e-12, 1 + 5e-12, 1 + 1e-8])
        result = barrier_outcomes(100., low, 100., 100.)
        np.testing.assert_array_equal(result['stop_hit'], [1, 0, 0])
        self.assertEqual(BARRIER_TOLERANCE, 1e-12)

    def test_old_and_new_success_events_are_not_nested(self):
        # First wins only at the new target; second wins only at the old target.
        highs, lows = np.array([100.6, 101.2]), np.array([99.8, 99.5])
        new = barrier_outcomes(highs, lows, [100, 100], 100)['success']
        old = (highs >= 101) & (lows > 99.1)
        np.testing.assert_array_equal(new, [1, 0])
        np.testing.assert_array_equal(old, [0, 1])

    def test_candidate_entry_relabels_same_target_prices(self):
        ohlc = np.array([[100., 100.6, 99.8, 100.]])
        factors = np.array([.999, 1., 1.003])
        before, before_factors = ohlc.copy(), factors.copy()
        result = class_targets(ohlc, factors)
        np.testing.assert_array_equal(result, [[0, 0, 1]])
        self.assertEqual(result.dtype, np.int64)
        np.testing.assert_array_equal(ohlc, before)
        np.testing.assert_array_equal(factors, before_factors)

    def test_empty_and_broadcast(self):
        self.assertEqual(class_targets(np.empty((0, 4)), [1, 1.01]).shape, (0, 2))
        result = barrier_outcomes(np.array([[100.5], [101.]]), 99.9, 100, [100, 100.1])
        self.assertEqual(result['success'].shape, (2, 2))
        self.assertEqual(barrier_outcomes(100.5, 100, 100, 100)['success'].shape, ())

    def test_invalid_outcomes_rejected(self):
        for values in ((99, 100, 100, 100), (100, 99, 101, 100),
                       (101, 100, 99, 100), (101, 99, 100, 0),
                       (np.nan, 99, 100, 100), (101, -1, 100, 100)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                barrier_outcomes(*values)

    def test_invalid_class_inputs_rejected(self):
        for ohlc, factors in (([100, 101, 99, 100], [1]),
                              ([[102, 101, 99, 100]], [1]),
                              ([[100, 101, 99, 100]], []),
                              ([[100, 101, 99, 100]], [0]),
                              ([[100, 101, 99, 100]], [True]),
                              ([[100, 101, 99, 100]], [np.inf])):
            with self.subTest(ohlc=ohlc, factors=factors), self.assertRaises(ValueError):
                class_targets(ohlc, factors)


class HalfBarrierFeatureTests(unittest.TestCase):
    def test_schema_renames_all_and_only_sixteen_target_fields(self):
        self.assertIn('0_5pct', TARGET)
        self.assertIn('0_4pct', TARGET)
        self.assertEqual(len(FEATURE_NAMES), 184)
        changed = [i for i, pair in enumerate(zip(FEATURE_NAMES, OLD_FEATURE_NAMES))
                   if pair[0] != pair[1]]
        self.assertEqual(len(changed), 16)
        self.assertEqual(changed, [i for i, name in enumerate(OLD_FEATURE_NAMES) if '_past_' in name])
        self.assertTrue(all(FEATURE_NAMES[i].endswith('_tp005_sl004') for i in changed))

    def test_actual_new_barrier_features_have_four_classes(self):
        bars = flat(4)
        bars[0, :, 1] = 100.5
        bars[1, :, 2] = 99.6
        bars[2, :, 1], bars[2, :, 2] = 100.5, 99.6
        result = features_from_history(bars, [100.] * 4)
        for window in (5, 10, 20, 30):
            columns = [FEATURE_NAMES.index(f'w{window}_past_{kind}_rate_tp005_sl004')
                       for kind in ('take_only', 'stop_only', 'both_touch', 'neither')]
            np.testing.assert_array_equal(result[:, columns], np.eye(4, dtype=np.float32))
        old = old_features(bars, [100.] * 4)
        old_neither = OLD_FEATURE_NAMES.index('w30_past_neither_rate')
        np.testing.assert_array_equal(old[:, old_neither], [1, 1, 1, 1])

    def test_features_use_each_historical_open_and_correct_window(self):
        bars = flat()
        bars[:, -5:, :4] *= 2
        bars[:, -5:, 1] = 201.
        result = features_from_history(bars, [100.])[0]
        for window in (5, 10, 20, 30):
            column = FEATURE_NAMES.index(f'w{window}_past_take_only_rate_tp005_sl004')
            self.assertAlmostEqual(float(result[column]), 5 / window, places=7)

    def test_target_independent_columns_unchanged_and_query_sensitive(self):
        bars = flat(2)
        bars[:, :, 1], bars[:, :, 2] = 100.5, 99.6
        before = bars.copy()
        entries = np.array([100., 100.3])
        bars.setflags(write=False)
        entries.setflags(write=False)
        result, old = features_from_history(bars, entries), old_features(bars, entries)
        unaffected = [i for i, name in enumerate(OLD_FEATURE_NAMES) if '_past_' not in name]
        np.testing.assert_array_equal(result[:, unaffected], old[:, unaffected])
        historical = [i for i, name in enumerate(FEATURE_NAMES) if not name.startswith('query_')]
        np.testing.assert_array_equal(result[0, historical], result[1, historical])
        query = FEATURE_NAMES.index('query_log_gap')
        self.assertNotEqual(result[0, query], result[1, query])
        np.testing.assert_array_equal(bars, before)
        self.assertFalse(bars.flags.writeable)
        self.assertFalse(entries.flags.writeable)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())

    def test_feature_and_label_boundary_comparisons_match(self):
        bars = flat(4)
        bars[0, :, 1] = 100.5 * (1 - .5e-12)
        bars[1, :, 1] = 100.5 * (1 - 5e-12)
        bars[2, :, 2] = 99.6 * (1 + .5e-12)
        bars[3, :, 2] = 99.6 * (1 + 5e-12)
        result = features_from_history(bars, [100.] * 4)
        classes = class_targets(bars[:, 0, :4])[:, 0]
        for code, kind in enumerate(('take_only', 'stop_only', 'both_touch', 'neither')):
            column = FEATURE_NAMES.index(f'w30_past_{kind}_rate_tp005_sl004')
            np.testing.assert_array_equal(result[:, column], classes == code)

    def test_empty_and_input_validation_preserved(self):
        self.assertEqual(features_from_history(np.empty((0, 30, 5)), []).shape, (0, 184))
        bars = flat()
        with self.assertRaises(ValueError):
            features_from_history(bars[:, :29], [100])
        bars[0, 0, 1] = 99
        with self.assertRaises(ValueError):
            features_from_history(bars, [100])


class HalfBarrierEventTests(unittest.TestCase):
    def test_recomputes_new_labels_and_preserves_selection(self):
        dataset = SimpleNamespace(
            target_ohlc=np.array([[100, 100.6, 99.8, 100], [100, 101.2, 99.5, 100]]),
            target_dates=np.array([1000, 1001]), symbol_ids=np.array([5, 9]),
            labels=np.array([False, True]),  # Deliberately obsolete old labels.
        )
        result = event_data(dataset, np.array([1, 0]))
        self.assertEqual(set(result), {'labels', 'gross', 'classes', 'dates', 'symbols'})
        np.testing.assert_array_equal(result['labels'], [False, True])
        np.testing.assert_array_equal(result['classes'], [2, 0])
        np.testing.assert_allclose(result['gross'], [-.004, .005])
        np.testing.assert_array_equal(result['dates'], [1001, 1000])
        np.testing.assert_array_equal(result['symbols'], [9, 5])
        np.testing.assert_array_equal(dataset.labels, [False, True])
        for indices in ([-1], [2], [0.0], [[0]], [True]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                event_data(dataset, indices)


if __name__ == '__main__':
    unittest.main()
