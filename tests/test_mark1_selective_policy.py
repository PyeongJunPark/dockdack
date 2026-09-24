"""Synthetic CPU-only checks for selective signals and policy holdout gates."""

from copy import deepcopy
from datetime import date, datetime
import importlib.util
import json
import unittest


HAS_NUMPY = importlib.util.find_spec("numpy") is not None
if HAS_NUMPY:
    import numpy as np
    from dockdack.mark1_metrics import binary_metrics
    from dockdack.mark1_selective_policy import (
        apply_policy, evaluate_signals, qualification, select_policy,
    )


@unittest.skipUnless(HAS_NUMPY, "requires optional numpy dependency")
class Mark1SelectivePolicyTests(unittest.TestCase):
    @staticmethod
    def fixture(days=40, per_day=3, symbols=12):
        count = days * per_day
        return [np.ones(count), np.full(count, .8), np.full(count, .01),
                np.repeat(np.arange(days), per_day), np.arange(count) % symbols,
                np.ones(count, dtype=bool)]

    def test_perfect_fixture_exact_interval_and_gate(self):
        result = evaluate_signals(*self.fixture())
        self.assertEqual(result["signal_count"], 120)
        self.assertEqual(result["signal_days"], 40)
        self.assertEqual(result["symbol_count"], 12)
        self.assertEqual(result["coverage"], 1)
        self.assertEqual(result["precision"], 1)
        self.assertAlmostEqual(result["net_mean_return"], .008)
        block = result["block_bootstrap"]
        self.assertEqual(block["valid_replicates"], 2000)
        self.assertEqual(block["precision_lower"], 1)
        self.assertEqual(block["precision_upper"], 1)
        self.assertAlmostEqual(block["net_mean_lower"], .008)
        self.assertAlmostEqual(block["net_mean_upper"], .008)
        self.assertEqual(qualification(result), {"qualified": True, "reasons": []})
        json.dumps(result, allow_nan=False)

    def test_overall_probability_metrics_are_not_remapped(self):
        args = self.fixture()
        args[0][::2] = 0
        args[1][::2] = .55
        args[2][::2] = -.009
        args[5][::2] = False
        result = evaluate_signals(*args)
        expected = binary_metrics(args[0], args[1], gross_returns=args[2], cost_bps=20)
        self.assertEqual(result["overall"], expected)
        self.assertEqual(result["overall"]["signal_count"], 120)
        self.assertEqual(result["signal_count"], 60)
        self.assertEqual(result["overall"]["precision"], .5)
        self.assertEqual(result["precision"], 1)
        self.assertNotEqual(result["overall"]["brier"],
                            binary_metrics(args[0], np.where(args[5], args[1], 0))["brier"])

    def test_empty_and_no_signal_keep_undefined_statistics(self):
        for args in ([[], [], [], [], [], []],
                     [[0, 1], [.1, .5], [-.009, .01], [1, 2], ["A", "B"], [False, False]]):
            result = evaluate_signals(*args)
            self.assertEqual(result["signal_count"], 0)
            self.assertEqual(result["signal_days"], 0)
            self.assertEqual(result["symbol_count"], 0)
            self.assertIsNone(result["precision"])
            self.assertIsNone(result["net_mean_return"])
            self.assertIsNone(result["block_bootstrap"]["precision_lower"])
            self.assertFalse(qualification(result)["qualified"])
            json.dumps(result, allow_nan=False)

    def test_strict_probability_gate(self):
        with self.assertRaises(ValueError):
            evaluate_signals([1], [.5], [.01], [1], ["A"], [True])
        with self.assertRaises(ValueError):
            evaluate_signals([1], [.49], [.01], [1], ["A"], [True])
        result = evaluate_signals([1], [np.nextafter(.5, 1)], [.01], [1], ["A"], [True])
        self.assertEqual(result["signal_count"], 1)

    def test_policy_strict_confidence_and_inclusive_stop_boundary(self):
        probabilities = np.array([.5, .6, np.nextafter(.6, 1), .7, .8])
        stops = np.array([.1, .1, .25, np.nextafter(.25, 1), .2])
        selected = apply_policy(probabilities, {"threshold": .6, "stop_probability_cap": .25}, stops)
        np.testing.assert_array_equal(selected, [False, False, True, False, True])
        np.testing.assert_array_equal(
            apply_policy(probabilities, {"threshold": .5, "stop_probability_cap": 1}),
            [False, True, True, True, True])
        self.assertFalse(apply_policy([1], {"threshold": 1, "stop_probability_cap": 1})[0])

    def test_three_ci_floor_requirements_and_exact_minimum(self):
        for args in (self.fixture(49, 1), self.fixture(19, 4), self.fixture(30, 2, symbols=9)):
            with self.subTest(shape=len(args[0])):
                result = evaluate_signals(*args)
                self.assertIsNone(result["block_bootstrap"]["precision_lower"])
                self.assertFalse(qualification(result)["qualified"])
        args = self.fixture(20, 3, symbols=10)
        args[5][::6] = False  # Exactly 50 events, 20 days, and 10 symbols.
        result = evaluate_signals(*args)
        self.assertEqual((result["signal_count"], result["signal_days"], result["symbol_count"]),
                         (50, 20, 10))
        self.assertEqual(result["block_bootstrap"]["precision_lower"], 1)

    def test_short_date_population_never_attempts_invalid_block_draw(self):
        result = evaluate_signals(*self.fixture(days=3, per_day=20))
        self.assertEqual(result["block_bootstrap"]["count_days"], 3)
        self.assertEqual(result["block_bootstrap"]["valid_replicates"], 0)

    def test_no_signal_calendar_dates_remain_in_resampling(self):
        args = self.fixture(days=250, per_day=3)
        args[5][:] = args[3] < 20
        args[1][~args[5]] = .2
        block = evaluate_signals(*args)["block_bootstrap"]
        self.assertEqual(block["count_days"], 250)
        self.assertEqual(block["signal_days"], 20)
        self.assertGreater(block["skipped_zero_signal_replicates"], 0)
        self.assertEqual(block["valid_replicates"] + block["skipped_zero_signal_replicates"], 2000)
        self.assertEqual(block["precision_lower"], 1)

    def test_serial_clusters_widen_ci_and_are_deterministic(self):
        args = self.fixture(days=100)
        args[0][args[3] < 50] = 0
        args[2][args[3] < 50] = -.009
        first = evaluate_signals(*args)
        second = evaluate_signals(*args)
        self.assertEqual(first, second)
        block = first["block_bootstrap"]
        self.assertLess(block["precision_lower"], .35)
        self.assertGreater(block["precision_upper"], .65)
        self.assertLess(block["net_mean_lower"], 0)

    def test_replication_does_not_shrink_cluster_interval(self):
        args = self.fixture(days=40)
        args[0][:] = args[3] % 2
        args[2][:] = np.where(args[0], .01, -.009)
        first = evaluate_signals(*args)["block_bootstrap"]
        second = evaluate_signals(*(np.repeat(value, 3) for value in args))["block_bootstrap"]
        for key in ("precision_lower", "precision_upper", "net_mean_lower", "net_mean_upper"):
            self.assertAlmostEqual(first[key], second[key])

    def test_unsorted_rows_and_calendar_types_preserve_intervals(self):
        args = self.fixture()
        args[0][args[3] < 15] = 0
        args[2][args[3] < 15] = -.009
        expected = evaluate_signals(*args)["block_bootstrap"]
        order = np.random.default_rng(4).permutation(len(args[0]))
        calendar = np.datetime64("2021-01-01") + args[3].astype("timedelta64[D]")
        for dates in (args[3].astype(object), calendar, calendar.astype(str),
                      np.array([date.fromisoformat(str(value)) for value in calendar], dtype=object),
                      np.array([datetime.fromisoformat(str(value)) for value in calendar], dtype=object)):
            altered = [value[order] for value in args]
            altered[3] = dates[order]
            self.assertEqual(evaluate_signals(*altered)["block_bootstrap"], expected)

    def test_inputs_writeability_and_global_rng_are_unchanged(self):
        args = self.fixture()
        copies = [value.copy() for value in args]
        for value in args:
            value.flags.writeable = False
        np.random.seed(123)
        expected = np.random.random(4)
        np.random.seed(123)
        evaluate_signals(*args)
        select_policy(*args[:5], stop_probabilities=np.full(len(args[0]), .2))
        np.testing.assert_array_equal(expected, np.random.random(4))
        for value, original in zip(args, copies):
            self.assertFalse(value.flags.writeable)
            np.testing.assert_array_equal(value, original)

    def test_string_and_object_integer_symbol_ids(self):
        args = self.fixture()
        expected = evaluate_signals(*args)
        for symbols in (args[4].astype(str), args[4].astype(object),
                        np.array(["S" + str(value) for value in args[4]], dtype=object)):
            args[4] = symbols
            self.assertEqual(evaluate_signals(*args), expected)

    def test_invalid_vectors_dates_masks_and_symbols(self):
        base = [[1], [.8], [.01], [1], ["A"], [True]]
        invalid = [(0, [2]), (0, [[1]]), (1, [float("nan")]), (1, [1.1]),
                   (1, [-.1]), (1, [".8"]), (2, None), (2, []), (2, [-1.1]),
                   (2, [float("inf")]), (3, []), (3, [None]), (3, [True]),
                   (3, [1.5]), (3, ["2024-99-02"]), (3, [""]),
                   (3, np.array(["NaT"], dtype="datetime64[D]")),
                   (4, []), (4, [None]), (4, [True]), (4, [float("nan")]),
                   (4, [""]), (4, [" "]), (4, [["A"]]), (5, []),
                   (5, [1]), (5, [0]), (5, ["True"]), (5, [[True]])]
        for index, value in invalid:
            args = deepcopy(base)
            args[index] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                evaluate_signals(*args)
        for cost in (True, "20", -1, float("nan"), float("inf"), 10001):
            with self.subTest(cost=cost), self.assertRaises(ValueError):
                evaluate_signals(*base, cost_bps=cost)

    def test_invalid_policies_and_stop_probabilities(self):
        for policy in (None, {}, {"threshold": .49, "stop_probability_cap": 1},
                       {"threshold": True, "stop_probability_cap": 1},
                       {"threshold": .5, "stop_probability_cap": float("nan")},
                       {"threshold": .5, "stop_probability_cap": 1.1},
                       {"threshold": .5, "stop_probability_cap": -.1},
                       {"threshold": .5, "stop_probability_cap": .25}):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                apply_policy([.8], policy)
        for stops in ([float("nan")], [1.1], [-.1], [], [[.2]], [".2"]):
            with self.subTest(stops=stops), self.assertRaises(ValueError):
                apply_policy([.8], {"threshold": .5, "stop_probability_cap": 1}, stops)

    def test_large_finite_returns_have_finite_json_safe_intervals(self):
        args = self.fixture()
        args[2][:] = 1e308
        result = evaluate_signals(*args)
        self.assertTrue(np.isfinite(result["net_mean_return"]))
        self.assertTrue(np.isfinite(result["block_bootstrap"]["net_mean_upper"]))
        json.dumps(result, allow_nan=False)

    def test_grid_sizes_tie_break_and_strict_thresholds(self):
        args = self.fixture()
        result = select_policy(*args[:5])
        self.assertEqual(len(result["grid"]), 10)
        self.assertEqual(result["chosen_policy"], {"threshold": .5, "stop_probability_cap": 1.0})
        self.assertTrue(result["calibration_qualified"])
        self.assertEqual(result["grid"][6]["policy"]["threshold"], .8)
        self.assertEqual(result["grid"][6]["metrics"]["signal_count"], 0)
        joint = select_policy(*args[:5], stop_probabilities=np.full(len(args[0]), .2))
        self.assertEqual(len(joint["grid"]), 20)
        self.assertEqual(joint["chosen_policy"], result["chosen_policy"])
        self.assertFalse(joint["deployment_allowed"])
        json.dumps(joint, allow_nan=False)

    def test_selects_rare_high_precision_over_many_low_precision(self):
        args = self.fixture(days=40, per_day=12)
        args[0][:] = (np.arange(len(args[0])) % 2 == 0)
        args[1][:] = np.where(args[0], .8, .55)
        args[2][:] = np.where(args[0], .01, -.009)
        # Alternating high-probability symbol IDs would otherwise only cover 6 symbols.
        args[4] = np.arange(len(args[0])) // 2 % 12
        result = select_policy(*args[:5])
        self.assertEqual(result["chosen_policy"], {"threshold": .55, "stop_probability_cap": 1.0})
        self.assertEqual(result["chosen_metrics"]["precision"], 1)
        self.assertEqual(result["chosen_metrics"]["coverage"], .5)
        self.assertTrue(result["calibration_qualified"])

    def test_joint_stop_cap_can_select_better_supported_precision(self):
        args = self.fixture(days=40, per_day=12)
        args[0][:] = (np.arange(len(args[0])) % 2 == 0)
        args[2][:] = np.where(args[0], .01, -.009)
        args[4] = np.arange(len(args[0])) // 2 % 12
        stops = np.where(args[0], .25, .26)
        result = select_policy(*args[:5], stop_probabilities=stops)
        self.assertEqual(result["chosen_policy"], {"threshold": .5, "stop_probability_cap": .25})
        self.assertEqual(result["chosen_metrics"]["precision"], 1)

    def test_no_eligible_policy_uses_unqualified_diagnostic(self):
        args = self.fixture(days=10)
        result = select_policy(*args[:5])
        self.assertFalse(result["calibration_qualified"])
        self.assertEqual(result["selection_reason"], "no_eligible_candidate_diagnostic_only")
        self.assertEqual(result["chosen_policy"], {"threshold": .5, "stop_probability_cap": 1.0})
        self.assertEqual(result["chosen_metrics"]["precision"], 1)
        empty = select_policy([], [], [], [], [])
        self.assertIsNone(empty["chosen_metrics"]["precision"])
        self.assertFalse(empty["calibration_qualified"])

    def test_gate_strict_bounds_and_fixed_cost(self):
        result = evaluate_signals(*self.fixture())
        for key, value in (("precision_lower", .579), ("precision_lower", float("nan")),
                           ("precision_upper", .4), ("net_mean_lower", 0),
                           ("net_mean_upper", -.001)):
            altered = deepcopy(result)
            altered["block_bootstrap"][key] = value
            self.assertFalse(qualification(altered)["qualified"])
        result["precision"] = np.nextafter(.65, 0)
        self.assertFalse(qualification(result)["qualified"])
        result["precision"] = .65
        self.assertTrue(qualification(result)["qualified"])
        self.assertFalse(qualification(evaluate_signals(*self.fixture(), cost_bps=0))["qualified"])

    def test_gate_malformed_protocol_fails_closed(self):
        self.assertFalse(qualification({})["qualified"])
        with self.assertRaises(ValueError):
            qualification(None)
        result = evaluate_signals(*self.fixture())
        invalid = [("block_length", 1), ("n_boot", 500), ("seed", 3), ("method", "iid"),
                   ("reason", "suppressed"), ("valid_replicates", 0),
                   ("skipped_zero_signal_replicates", 1), ("count_days", 2),
                   ("signal_count", 999), ("minimum_symbols", 1)]
        for key, value in invalid:
            altered = deepcopy(result)
            altered["block_bootstrap"][key] = value
            with self.subTest(key=key):
                self.assertFalse(qualification(altered)["qualified"])
        for key, value in (("count", None), ("count_days", True), ("symbol_count", 1000),
                           ("signal_count", 3), ("signal_days", 2)):
            altered = deepcopy(result)
            altered[key] = value
            self.assertFalse(qualification(altered)["qualified"])


if __name__ == "__main__":
    unittest.main()
