"""Synthetic, no-GPU tests of mark_1 research uncertainty and gating."""

from datetime import date, datetime
import importlib.util
import json
import unittest


HAS_NUMPY = importlib.util.find_spec("numpy") is not None
if HAS_NUMPY:
    import numpy as np
    from dockdack.mark1_deep_validation import evaluate, qualification


@unittest.skipUnless(HAS_NUMPY, "requires optional numpy dependency")
class Mark1DeepValidationTests(unittest.TestCase):
    @staticmethod
    def fixture(days=60, per_day=5):
        count = days * per_day
        return (np.ones(count), np.full(count, .8), np.full(count, .01),
                np.repeat(np.arange(days), per_day))

    def test_perfect_fixture_exact_intervals_and_qualifies(self):
        metrics = evaluate(*self.fixture())
        block = metrics["block_bootstrap"]
        self.assertEqual(block["signal_count"], 300)
        self.assertEqual(block["signal_days"], 60)
        self.assertEqual(block["valid_replicates"], 500)
        self.assertEqual(block["precision_lower"], 1)
        self.assertEqual(block["precision_upper"], 1)
        self.assertAlmostEqual(block["net_mean_lower"], .008)
        self.assertAlmostEqual(block["net_mean_upper"], .008)
        self.assertEqual(qualification(metrics), {"qualified": True, "reasons": []})
        json.dumps(metrics, allow_nan=False)

    def test_strict_probability_boundary(self):
        result = evaluate([1, 1, 0], [.5, np.nextafter(.5, 1), .2],
                          [.01, .01, -.009], [1, 2, 3])
        self.assertEqual(result["signal_count"], 1)
        self.assertEqual(result["block_bootstrap"]["signal_days"], 1)
        self.assertIsNone(result["block_bootstrap"]["precision_lower"])

    def test_no_signal_and_empty_are_suppressed_and_json_safe(self):
        for result in (evaluate([], [], [], []),
                       evaluate([0, 1], [.2, .5], [-.009, .01], [1, 2])):
            block = result["block_bootstrap"]
            self.assertEqual(block["signal_days"], 0)
            self.assertEqual(block["valid_replicates"], 0)
            self.assertIsNotNone(block["reason"])
            self.assertFalse(qualification(result)["qualified"])
            for key in ("precision_lower", "precision_upper", "net_mean_lower", "net_mean_upper"):
                self.assertIsNone(block[key])
            json.dumps(result, allow_nan=False)

    def test_both_minimum_interval_requirements(self):
        for args in (self.fixture(days=99, per_day=1), self.fixture(days=29, per_day=5)):
            block = evaluate(*args)["block_bootstrap"]
            self.assertIsNone(block["precision_lower"])
        block = evaluate(*self.fixture(days=30, per_day=4))["block_bootstrap"]
        self.assertEqual(block["precision_lower"], 1)

    def test_contiguous_blocks_widen_serial_cluster_uncertainty(self):
        dates = np.repeat(np.arange(100), 5)
        labels = np.repeat(np.r_[np.ones(50), np.zeros(50)], 5)
        probabilities = np.full(len(labels), .8)
        gross_returns = np.where(labels, .01, -.009)
        result = evaluate(labels, probabilities, gross_returns, dates)
        block = result["block_bootstrap"]
        self.assertLess(block["precision_lower"], .35)
        self.assertGreater(block["precision_upper"], .65)
        self.assertLess(block["net_mean_lower"], 0)
        self.assertGreater(block["net_mean_upper"], 0)
        self.assertEqual(result, evaluate(labels, probabilities, gross_returns, dates))

    def test_duplicate_symbols_do_not_shrink_date_uncertainty(self):
        dates = np.repeat(np.arange(60), 5)
        labels = np.repeat(np.arange(60) % 2, 5)
        args = [labels, np.full(len(labels), .8), np.where(labels, .01, -.009), dates]
        first = evaluate(*args)["block_bootstrap"]
        second = evaluate(*(np.repeat(value, 2) for value in args))["block_bootstrap"]
        for key in ("precision_lower", "precision_upper", "net_mean_lower", "net_mean_upper"):
            self.assertAlmostEqual(first[key], second[key])

    def test_no_signal_dates_included_and_zero_signal_draws_skipped(self):
        dates = np.arange(200)
        probabilities = np.where(dates < 30, .8, .2)
        labels = np.ones(200)
        args = [np.repeat(value, 4) for value in
                (labels, probabilities, np.full(200, .01), dates)]
        block = evaluate(*args)["block_bootstrap"]
        self.assertEqual(block["count_days"], 200)
        self.assertEqual(block["signal_days"], 30)
        self.assertGreater(block["skipped_zero_signal_replicates"], 0)
        self.assertEqual(block["valid_replicates"] + block["skipped_zero_signal_replicates"], 500)
        self.assertEqual(block["precision_lower"], 1)

    def test_unsorted_rows_and_calendar_types_preserve_intervals(self):
        labels, probabilities, returns, dates = self.fixture()
        labels[dates < 20] = 0
        returns[dates < 20] = -.009
        expected = evaluate(labels, probabilities, returns, dates)["block_bootstrap"]
        order = np.random.default_rng(5).permutation(len(dates))
        calendar = np.datetime64("2022-01-01") + dates.astype("timedelta64[D]")
        for keys in (dates.astype(object), calendar, calendar.astype(str),
                     np.array([date.fromisoformat(str(day)) for day in calendar], dtype=object),
                     np.array([datetime.fromisoformat(str(day)) for day in calendar], dtype=object)):
            block = evaluate(labels[order], probabilities[order], returns[order], keys[order])["block_bootstrap"]
            self.assertEqual(block, expected)

    def test_invalid_inputs(self):
        invalid = [([2], [.8], [.01], [1]), ([1], [float("nan")], [.01], [1]),
                   ([1], [1.1], [.01], [1]), ([1], [.8], [-1.1], [1]),
                   ([1], [.8], [float("inf")], [1]), ([1], [.8], None, [1]),
                   ([1], [.8], [], [1]), ([1], [.8], [.01], []),
                   ([1], [.8], [.01], [None]), ([1], [.8], [.01], [True]),
                   ([1], [.8], [.01], [1.5]), ([1], [.8], [.01], ["2022-99-02"]),
                   ([1], [.8], [.01], [""]),
                   ([1], [.8], [.01], np.array(["NaT"], dtype="datetime64[D]"))]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                evaluate(*args)
        for cost in (float("nan"), -1, True, "20"):
            with self.subTest(cost=cost), self.assertRaises(ValueError):
                evaluate(*self.fixture(), cost_bps=cost)

    def test_large_finite_returns_do_not_overflow(self):
        labels, probabilities, returns, dates = self.fixture()
        returns[:] = 1e308
        result = evaluate(labels, probabilities, returns, dates)
        self.assertTrue(np.isfinite(result["block_bootstrap"]["net_mean_upper"]))
        json.dumps(result, allow_nan=False)

    def test_gate_count_and_day_limits_independently(self):
        for fixture, expected in ((self.fixture(60, 2), "requires_at_least_200_signals"),
                                  (self.fixture(40, 6), "requires_at_least_50_signal_days")):
            gate = qualification(evaluate(*fixture))
            self.assertFalse(gate["qualified"])
            self.assertIn(expected, gate["reasons"])

    def test_gate_strict_lower_bounds_and_fixed_cost(self):
        for key, value, expected in (("precision_lower", .5, "precision_lower_must_exceed_0_5"),
                                     ("net_mean_lower", 0, "net_mean_lower_must_be_positive_after_cost"),
                                     ("net_mean_lower", float("nan"), "net_mean_lower_must_be_positive_after_cost")):
            metrics = evaluate(*self.fixture())
            metrics["block_bootstrap"][key] = value
            self.assertIn(expected, qualification(metrics)["reasons"])
        self.assertIn("requires_20_bps_round_trip_cost",
                      qualification(evaluate(*self.fixture(), cost_bps=0))["reasons"])

    def test_gate_missing_or_wrong_interval_protocol_fails_closed(self):
        self.assertFalse(qualification({})["qualified"])
        with self.assertRaises(ValueError):
            qualification(None)
        for key, value in (("block_length", 1), ("method", "iid"),
                           ("seed", 1), ("n_boot", 100), ("reason", "suppressed"),
                           ("precision_upper", .4), ("net_mean_upper", -.001),
                           ("valid_replicates", 0), ("skipped_zero_signal_replicates", 1),
                           ("count_days", 20)):
            metrics = evaluate(*self.fixture())
            metrics["block_bootstrap"][key] = value
            self.assertFalse(qualification(metrics)["qualified"])


if __name__ == "__main__":
    unittest.main()
