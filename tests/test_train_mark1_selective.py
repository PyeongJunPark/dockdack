"""Synthetic CPU orchestration tests; no databases or model training."""
from __future__ import annotations

import copy
import importlib.util
from types import SimpleNamespace
import unittest


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in
                ("numpy", "torch", "dockdack.mark1_selective_models"))
if AVAILABLE:
    import numpy as np
    from examples import train_mark1_selective as runner


@unittest.skipUnless(AVAILABLE, "requires NumPy, Torch and selective model module")
class SelectiveTrainerTests(unittest.TestCase):
    @staticmethod
    def calendar():
        days = np.arange(np.datetime64("2010-01-01"), np.datetime64("2027-01-01"))
        return days[np.is_busday(days)].astype(np.int64)

    def split_dataset(self, fold_name, *, future=True):
        sessions = self.calendar()
        fold = runner.FOLDS[fold_name]
        dates = sessions[(sessions >= np.datetime64("2018-01-01").astype(int))
                         & (sessions <= np.datetime64("2024-12-31").astype(int))]
        pairs = [(0, int(day)) for day in dates]
        pairs.append((1, int(np.datetime64(fold["train_end"]).astype(int))))
        for year in (fold["tune_year"], fold["calibration_year"], fold["selection_year"]):
            for month in (3, 9):
                day = np.datetime64(f"{year}-{month:02d}-15").astype(int)
                day = int(sessions[np.searchsorted(sessions, day)])
                pairs.extend(((1, day), (2, day)))
        if future:
            pairs.extend(((0, int(np.datetime64("2025-06-02").astype(int))),
                          (999, int(np.datetime64("2026-06-01").astype(int)))))
        dataset = SimpleNamespace(target_dates=np.asarray([day for _, day in pairs], dtype=np.int64),
                                  symbol_ids=np.asarray([symbol for symbol, _ in pairs], dtype=np.int32))
        return dataset, sessions

    @staticmethod
    def feature_dataset(count=7):
        prices = np.arange(count, dtype=np.float64)[:, None] + np.linspace(90., 100., 30)[None, :]
        histories = np.stack((prices, prices * 1.02, prices * .99, prices * 1.001,
                              np.full_like(prices, 10000)), axis=2)
        opens = np.arange(count, dtype=np.float64) + 101.
        return SimpleNamespace(bars=histories.reshape(-1, 5), starts=np.arange(count) * 30,
                               target_ohlc=np.stack((opens, opens * 1.03, opens * .98, opens), axis=1))

    @staticmethod
    def predictions(offset=0., *, joint=False):
        result = {}
        for index, part in enumerate(("probability_calibration", "policy_calibration", "audit")):
            values = np.arange(index + 3, dtype=np.float64) + offset
            result[part] = {"success_logits": values,
                            "stop_logits": -values if joint else None}
        return result

    @staticmethod
    def ranking_results():
        return {fold: {name: {"audit": {
            "block_bootstrap": {"precision_lower": .6, "net_mean_lower": .001},
            "signal_count": 100, "overall": {"brier": .2}}}
            for name in runner.MODEL_NAMES} for fold in runner.FOLDS}

    def test_halfyear_split_and_exact_thirty_session_purge(self):
        for fold_name, fold in runner.FOLDS.items():
            with self.subTest(fold=fold_name):
                data, sessions = self.split_dataset(fold_name)
                parts = runner.selective_splits(data, sessions, fold_name, max_train=0, max_tune=0)
                boundary = np.datetime64(f"{fold['calibration_year']}-06-30").astype(int)
                h1 = data.target_dates[parts["probability_calibration"]]
                h2 = data.target_dates[parts["policy_calibration"]]
                self.assertTrue(np.all(h1 <= boundary))
                self.assertTrue(np.all(h2 > boundary))
                self.assertTrue(np.all(sessions[np.searchsorted(sessions, h2) - 30] > boundary))
                expected_first = sessions[np.searchsorted(sessions, boundary, side="right") + 30]
                self.assertEqual(h2.min(), expected_first)
                audit = data.target_dates[parts["audit"]]
                self.assertTrue(np.all(audit >= np.datetime64(f"{fold['selection_year']}-01-01").astype(int)))
                self.assertTrue(np.all(audit <= np.datetime64(f"{fold['selection_year']}-12-31").astype(int)))
                self.assertLess(h1.max(), h2.min())
                self.assertLess(h2.max(), audit.min())
                self.assertTrue(all(not indices.flags.writeable for indices in parts.values()))

    def test_train_universe_precedes_cap_and_no_future_targets(self):
        for fold_name in runner.FOLDS:
            data, sessions = self.split_dataset(fold_name)
            parts = runner.selective_splits(data, sessions, fold_name, max_train=1, max_tune=0)
            self.assertEqual(len(parts["train"]), 1)
            for name, indices in parts.items():
                self.assertTrue(np.all(data.target_dates[indices] < np.datetime64("2025-01-01").astype(int)))
                if name != 'train':
                    self.assertEqual(set(data.symbol_ids[indices]), {0, 1})

    def test_splitting_needs_no_labels_and_preserves_caller_arrays_and_flags(self):
        data, sessions = self.split_dataset("walk_2024")
        before = (data.target_dates.copy(), data.symbol_ids.copy(), sessions.copy())
        runner.selective_splits(data, sessions, "walk_2024")
        for actual, original in zip((data.target_dates, data.symbol_ids, sessions), before):
            np.testing.assert_array_equal(actual, original)
            self.assertTrue(actual.flags.writeable)

    def test_future_append_does_not_change_split_membership(self):
        for fold in runner.FOLDS:
            original, sessions = self.split_dataset(fold, future=False)
            expanded, _ = self.split_dataset(fold, future=True)
            before = runner.selective_splits(original, sessions, fold, max_train=17, max_tune=19)
            after = runner.selective_splits(expanded, sessions, fold, max_train=17, max_tune=19)
            for part in before:
                np.testing.assert_array_equal(before[part], after[part])

    def test_empty_required_split_rejected(self):
        data, sessions = self.split_dataset('walk_2024')
        boundary = np.datetime64('2023-06-30').astype(int)
        keep = ~((data.target_dates > boundary) & (data.target_dates < np.datetime64('2024-01-01').astype(int)))
        data.target_dates, data.symbol_ids = data.target_dates[keep], data.symbol_ids[keep]
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            runner.selective_splits(data, sessions, 'walk_2024')

    def test_feature_batches_equal_unbatched_and_preserve_input(self):
        data = self.feature_dataset()
        before = copy.deepcopy(data)
        indices = np.array([6, 0, 4, 1, 3])
        one = runner.feature_array(data, indices, batch_size=1)
        two = runner.feature_array(data, indices, batch_size=2)
        whole = runner.feature_array(data, indices, batch_size=99)
        np.testing.assert_array_equal(one, two)
        np.testing.assert_array_equal(two, whole)
        for key in ('bars', 'starts', 'target_ohlc'):
            np.testing.assert_array_equal(getattr(data, key), getattr(before, key))
            self.assertTrue(getattr(data, key).flags.writeable)
        self.assertTrue(indices.flags.writeable)

    def test_feature_array_never_uses_target_high_low_close(self):
        data = self.feature_dataset()
        before = runner.feature_array(data, np.arange(7))
        data.target_ohlc[:, 1:] = np.nan
        after = runner.feature_array(data, np.arange(7))
        np.testing.assert_array_equal(before, after)
        data.target_ohlc[:, 0] *= 1.01
        changed = runner.feature_array(data, np.arange(7))
        self.assertFalse(np.array_equal(before, changed))

    def test_feature_array_empty_invalid_indices_and_batch_size(self):
        data = self.feature_dataset()
        self.assertEqual(runner.feature_array(data, np.array([], dtype=int)).shape,
                         (0, len(runner.FEATURE_NAMES)))
        for indices in (np.array([-1]), np.array([7]), np.array([.5]), np.array([[1]]), np.array([True])):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                runner.feature_array(data, indices)
        for batch in (0, -1, True, 1.5):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                runner.feature_array(data, np.arange(7), batch_size=batch)

    def test_ranking_uses_worst_fold_precision_lower_before_brier_or_mean(self):
        results = self.ranking_results()
        first, second = runner.MODEL_NAMES[:2]
        for fold, lower_first, lower_second in zip(runner.FOLDS, (.66, .67), (.61, .95)):
            results[fold][first]['audit']['block_bootstrap']['precision_lower'] = lower_first
            results[fold][first]['audit']['overall']['brier'] = .3
            results[fold][second]['audit']['block_bootstrap']['precision_lower'] = lower_second
            results[fold][second]['audit']['overall']['brier'] = .01
        self.assertEqual(runner.architecture_ranking(results)[0]['architecture'], first)

    def test_ranking_requires_evidence_in_both_folds(self):
        results = self.ranking_results()
        first = runner.MODEL_NAMES[0]
        for fold in runner.FOLDS:
            results[fold][first]['audit']['overall']['brier'] = 0.
            results[fold][first]['audit']['block_bootstrap']['precision_lower'] = .99
        last = list(runner.FOLDS)[-1]
        results[last][first]['audit']['block_bootstrap']['precision_lower'] = None
        results[last][first]['audit']['block_bootstrap']['net_mean_lower'] = None
        ranked = runner.architecture_ranking(results)
        self.assertNotEqual(ranked[0]['architecture'], first)
        self.assertFalse(next(row for row in ranked if row['architecture'] == first)['both_audits_have_evidence'])

    def test_ranking_net_lower_and_signals_are_tiebreakers(self):
        results = self.ranking_results()
        winner = runner.MODEL_NAMES[-1]
        for fold in runner.FOLDS:
            results[fold][winner]['audit']['block_bootstrap']['net_mean_lower'] = .002
        self.assertEqual(runner.architecture_ranking(results)[0]['architecture'], winner)
        for fold in runner.FOLDS:
            results[fold][winner]['audit']['block_bootstrap']['net_mean_lower'] = .001
            results[fold][winner]['audit']['signal_count'] = 200
        self.assertEqual(runner.architecture_ranking(results)[0]['architecture'], winner)

    def test_no_evidence_fallback_is_explicit_diagnostic_brier(self):
        results = self.ranking_results()
        winner = runner.MODEL_NAMES[-1]
        for fold in runner.FOLDS:
            for name in runner.MODEL_NAMES:
                audit = results[fold][name]['audit']
                audit['block_bootstrap']['precision_lower'] = None
                audit['block_bootstrap']['net_mean_lower'] = None
            results[fold][winner]['audit']['overall']['brier'] = .1
        ranked = runner.architecture_ranking(results)
        self.assertEqual(ranked[0]['architecture'], winner)
        self.assertTrue(all(not row['both_audits_have_evidence'] for row in ranked))

    def test_average_logits_binary_and_joint_without_mutation(self):
        for joint in (False, True):
            members = [self.predictions(value, joint=joint) for value in (0., 1., 5.)]
            before = copy.deepcopy(members)
            result = runner.average_predictions(members)
            for part in result:
                np.testing.assert_array_equal(result[part]['success_logits'],
                                              members[0][part]['success_logits'] + 2.)
                if joint:
                    np.testing.assert_array_equal(result[part]['stop_logits'],
                                                  members[0][part]['stop_logits'] - 2.)
                else:
                    self.assertIsNone(result[part]['stop_logits'])
            for actual, original in zip(members, before):
                for part in actual:
                    np.testing.assert_array_equal(actual[part]['success_logits'], original[part]['success_logits'])

    def test_average_rejects_mixed_binary_joint(self):
        with self.assertRaises(ValueError):
            runner.average_predictions([self.predictions(joint=False), self.predictions(joint=True)])

    def test_average_rejects_empty_members(self):
        with self.assertRaises(ValueError):
            runner.average_predictions([])

    def test_average_rejects_nonfinite_or_nonnumeric_or_nonvector(self):
        for bad in (np.array([np.nan, 1., 2.]), np.array([np.inf, 1., 2.]),
                    np.array([['a', 'b', 'c']]), np.ones((3, 1)), np.empty(0),
                    np.array([True, False, True])):
            member = self.predictions()
            member['probability_calibration']['success_logits'] = bad
            with self.subTest(shape=bad.shape, dtype=bad.dtype), self.assertRaises(ValueError):
                runner.average_predictions([member])

    def test_average_rejects_member_length_or_stop_length_mismatch(self):
        members = [self.predictions(), self.predictions()]
        members[1]['audit']['success_logits'] = np.arange(6.)
        with self.assertRaises(ValueError):
            runner.average_predictions(members)
        member = self.predictions(joint=True)
        member['audit']['stop_logits'] = np.arange(6.)
        with self.assertRaises(ValueError):
            runner.average_predictions([member])

    def test_average_rejects_partition_schema_inconsistency(self):
        member = self.predictions(joint=True)
        member['audit']['stop_logits'] = None
        with self.assertRaises(ValueError):
            runner.average_predictions([member])


if __name__ == '__main__':
    unittest.main()
