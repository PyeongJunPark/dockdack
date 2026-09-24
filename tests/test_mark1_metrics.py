"""Known-value metric, calibration and date-cluster interval tests."""

import importlib.util
import json
import math
import unittest


HAS_NUMPY = importlib.util.find_spec("numpy") is not None
if HAS_NUMPY:
    import numpy as np
    from dockdack.mark1_metrics import (
        binary_metrics, calibrated_probability, day_bootstrap_precision,
        fit_calibration, reliability_bins,
    )


@unittest.skipUnless(HAS_NUMPY, "requires optional ml dependencies")
class Mark1MetricsTests(unittest.TestCase):
    def test_perfect_predictions_and_hypothetical_cost(self):
        result = binary_metrics([0, 1, 0, 1], [0, 1, 0, 1],
                                gross_returns=[-0.009, .01, -.009, .02], cost_bps=20)
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["brier"], 0)
        self.assertLess(result["log_loss"], 1e-12)
        for key in ("roc_auc", "pr_auc", "accuracy", "precision", "winrate"):
            self.assertEqual(result[key], 1)
        self.assertEqual(result["signal_count"], 2)
        self.assertEqual(result["coverage"], .5)
        self.assertAlmostEqual(result["gross_mean_return"], .015)
        self.assertAlmostEqual(result["net_mean_return"], .013)
        self.assertEqual(result["ece10"], 0)
        json.dumps(result, allow_nan=False)

    def test_constant_scores_tie_correct_auc_ap_and_strict_threshold(self):
        result = binary_metrics([0, 1, 1, 0], [.5, .5, .5, .5])
        self.assertEqual(result["roc_auc"], .5)
        self.assertEqual(result["pr_auc"], .5)
        self.assertEqual(result["brier"], .25)
        self.assertAlmostEqual(result["log_loss"], math.log(2))
        self.assertEqual(result["signal_count"], 0)
        self.assertEqual(result["accuracy"], .5)
        self.assertEqual(result["ece10"], 0)
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["winrate"])
        self.assertIsNone(result["net_mean_return"])

    def test_nontrivial_ties_ap_is_average_precision_not_trapezoid(self):
        result = binary_metrics([1, 0, 1, 0], [.8, .8, .3, .1])
        self.assertAlmostEqual(result["roc_auc"], .625)
        self.assertAlmostEqual(result["pr_auc"], (0.5 + 2 / 3) / 2)
        permuted = binary_metrics([0, 1, 0, 1], [.8, .8, .1, .3])
        self.assertEqual(result["roc_auc"], permuted["roc_auc"])
        self.assertEqual(result["pr_auc"], permuted["pr_auc"])

    def test_known_brier_logloss_and_bin_boundaries(self):
        result = binary_metrics([0, 1], [.25, .75])
        self.assertEqual(result["brier"], .0625)
        self.assertAlmostEqual(result["log_loss"], -math.log(.75))
        self.assertAlmostEqual(result["ece10"], .25)
        bins = reliability_bins([0, 1, 1], [0, .1, 1])
        self.assertEqual(len(bins), 10)
        self.assertEqual(bins[0]["count"], 1)
        self.assertEqual(bins[1]["count"], 1)
        self.assertEqual(bins[9]["count"], 1)
        self.assertIsNone(bins[2]["observed_rate"])

    def test_empty_single_class_and_no_signals_are_json_safe(self):
        empty = binary_metrics([], [], dates=[])
        self.assertEqual(empty["count"], 0)
        self.assertEqual(empty["signal_count"], 0)
        self.assertIsNone(empty["coverage"])
        self.assertIsNone(empty["precision_date_bootstrap"])
        positive = binary_metrics([1, 1], [.7, .6])
        self.assertIsNone(positive["roc_auc"])
        self.assertEqual(positive["pr_auc"], 1)
        negative = binary_metrics([0, 0], [.1, .2])
        self.assertIsNone(negative["pr_auc"])
        for result in (empty, positive, negative):
            json.dumps(result, allow_nan=False)

    def test_invalid_inputs_rejected(self):
        for labels, probabilities in (([1, 2], [.1, .2]), ([0], [.1, .2]),
                                      ([0], [-.1]), ([0], [1.1]),
                                      ([0], [float("nan")]), ([float("inf")], [.2]),
                                      ([[0]], [[.2]]), (["0"], [.2]), ([0], [.2j])):
            with self.subTest(labels=labels, probabilities=probabilities), self.assertRaises(ValueError):
                binary_metrics(labels, probabilities)
        for kwargs in ({"cost_bps": -1}, {"cost_bps": float("inf")}, {"cost_bps": True},
                       {"gross_returns": [-1.1]}, {"gross_returns": [float("nan")]},
                       {"gross_returns": [0, 0]}, {"dates": []}, {"dates": [None]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                binary_metrics([0], [.2], **kwargs)

    def test_date_bootstrap_resamples_clusters_reproducibly(self):
        labels = [1, 1, 0, 0]
        probabilities = [.8] * 4
        dates = ["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"]
        first = day_bootstrap_precision(labels, probabilities, dates, n_boot=400, seed=42)
        self.assertEqual(first, day_bootstrap_precision(labels, probabilities, dates, n_boot=400, seed=42))
        self.assertEqual(first["count_days"], 2)
        self.assertEqual(first["signal_days"], 2)
        self.assertEqual(first["point"], .5)
        self.assertEqual(first["lower"], 0)
        self.assertEqual(first["upper"], 1)
        self.assertEqual(first["valid_replicates"], 400)
        # Duplicating every symbol on a day does not create independent dates.
        doubled = day_bootstrap_precision(np.repeat(labels, 2), np.repeat(probabilities, 2),
                                           np.repeat(dates, 2), n_boot=400, seed=42)
        self.assertEqual(first, doubled)

    def test_date_bootstrap_empty_days_missing_signals_and_datetime_values(self):
        dates = np.array(["2025-01-02", "2025-01-03"], dtype="datetime64[D]")
        result = day_bootstrap_precision([1, 0], [.7, .2], dates, n_boot=1000)
        self.assertEqual(result["count_days"], 2)
        self.assertEqual(result["signal_days"], 1)
        self.assertEqual(result["point"], 1)
        self.assertLess(result["valid_replicates"], 1000)
        self.assertIsNone(day_bootstrap_precision([1, 0], [.5, .2], dates))
        for kwargs in ({"n_boot": 0}, {"n_boot": True}, {"seed": -1}, {"seed": True}):
            with self.assertRaises(ValueError):
                day_bootstrap_precision([1, 0], [.7, .2], dates, **kwargs)
        with self.assertRaises(ValueError):
            day_bootstrap_precision([1], [.7], np.array(["NaT"], dtype="datetime64[D]"))

    def test_monotone_calibration_improves_independent_fixture(self):
        generator = np.random.default_rng(128)
        train_x = generator.normal(size=6000)
        train_prob = 1 / (1 + np.exp(-(train_x * .7 - .6)))
        train_y = generator.binomial(1, train_prob)
        raw_train_logits = train_x * 3 + 1.2
        calibration = fit_calibration(raw_train_logits, train_y)
        self.assertEqual(calibration["method"], "platt_monotone")
        self.assertGreater(calibration["slope"], 0)
        self.assertEqual(calibration["fit_samples"], 6000)
        self.assertFalse(calibration["weighted"])
        test_x = generator.normal(size=8000)
        test_prob = 1 / (1 + np.exp(-(test_x * .7 - .6)))
        test_y = generator.binomial(1, test_prob)
        raw_test_logits = test_x * 3 + 1.2
        before = binary_metrics(test_y, 1 / (1 + np.exp(-raw_test_logits)))
        calibrated = calibrated_probability(raw_test_logits, calibration)
        after = binary_metrics(test_y, calibrated)
        self.assertLess(after["brier"], before["brier"])
        self.assertLess(after["log_loss"], before["log_loss"])
        self.assertAlmostEqual(before["roc_auc"], after["roc_auc"])
        self.assertTrue(np.all(np.diff(calibrated_probability(np.arange(-10., 10.), calibration)) > 0))
        json.dumps(calibration, allow_nan=False)

    def test_calibration_constant_logits_and_reverse_relationship(self):
        constant = fit_calibration([0] * 100, [0] * 75 + [1] * 25)
        self.assertGreater(constant["slope"], 0)
        self.assertAlmostEqual(calibrated_probability([0], constant)[0], .25, places=4)
        reversed_fit = fit_calibration([-2, -1, 1, 2] * 20, [1, 1, 0, 0] * 20)
        self.assertGreater(reversed_fit["slope"], 0)
        values = calibrated_probability([-10, 0, 10], reversed_fit)
        self.assertTrue(np.all(np.diff(values) >= 0))

    def test_calibration_rejects_empty_singleclass_and_bad_metadata(self):
        for logits, labels in (([], []), ([1], [1]), ([0, 1], [0, 0]),
                                ([0, 1], [0]), ([float("nan"), 0], [0, 1])):
            with self.assertRaises(ValueError):
                fit_calibration(logits, labels)
        for calibration in ({}, {"method": "isotonic"},
                            {"method": "platt_monotone", "slope": 0, "bias": 0},
                            {"method": "platt_monotone", "slope": 1, "bias": float("inf")},
                            {"method": "platt_monotone", "slope": True, "bias": 0}):
            with self.assertRaises(ValueError):
                calibrated_probability([0, 1], calibration)

    def test_probability_transform_saturates_without_overflow(self):
        calibration = {"method": "platt_monotone", "slope": 10., "bias": 0.}
        actual = calibrated_probability([-1e308, -1000, 0, 1000, 1e308], calibration)
        np.testing.assert_array_equal(actual, [0, 0, .5, 1, 1])

    def test_strict_threshold_and_large_finite_returns_are_json_safe(self):
        result = binary_metrics([0, 1, 1], [.5, np.nextafter(.5, 1), .9],
                                gross_returns=[0, 1e308, 1e308])
        self.assertEqual(result["signal_count"], 2)
        self.assertEqual(result["precision"], 1)
        self.assertTrue(math.isfinite(result["gross_mean_return"]))
        json.dumps(result, allow_nan=False)

    def test_reverse_unbalanced_calibration_uses_base_rate_at_positive_boundary(self):
        logits = np.linspace(-3, 3, 1000)
        labels = (logits < -2).astype(int)
        calibration = fit_calibration(logits, labels)
        self.assertGreater(calibration["slope"], 0)
        self.assertAlmostEqual(calibrated_probability(logits, calibration).mean(), labels.mean(), places=4)


if __name__ == "__main__":
    unittest.main()
