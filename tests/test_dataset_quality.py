"""Pure quality/point-in-time checks; no broker or production database access."""
from dataclasses import FrozenInstanceError, replace
from datetime import date, timedelta
from decimal import Decimal
import copy
import random
import statistics
import unittest

from dockdack.dataset_quality import QualityPolicy, assess_series


def sessions(count=140):
    result, day = [], date(2024, 1, 1)
    while len(result) < count:
        if day.weekday() < 5:
            result.append(day.isoformat())
        day += timedelta(days=1)
    return result


def rows_for(calendar, count=None, *, currency="KRW"):
    return [dict(date=day, open=100.0, high=110.0, low=90.0, close=100.0,
                 volume=10000, currency=currency, trade_value=123,
                 liquidity_turnover=1_000_000_000.0)
            for day in calendar[:count]]


class DatasetQualityTests(unittest.TestCase):
    def setUp(self):
        self.calendar = sessions()
        self.as_of = date.fromisoformat(self.calendar[-1]) + timedelta(days=1)
        self.policy = QualityPolicy(min_median_turnover=1_000_000_000)

    def assess(self, rows, **kwargs):
        arguments = dict(market="domestic", as_of=self.as_of,
                         session_dates=self.calendar, policy=self.policy)
        arguments.update(kwargs)
        return assess_series(rows, **arguments)

    def test_sixty_session_warmup_and_thirty_inputs_plus_one_target(self):
        result = self.assess(rows_for(self.calendar, 61))
        self.assertFalse(result.bars[58].input_eligible)
        self.assertTrue(result.bars[59].input_eligible)
        self.assertEqual(len(result.samples), 1)
        sample, = result.samples
        self.assertEqual((sample.input_start_index, sample.input_end_index, sample.target_index), (30, 59, 60))
        self.assertEqual((sample.input_start_date, sample.input_end_date, sample.target_date),
                         (self.calendar[30], self.calendar[59], self.calendar[60]))
        self.assertFalse(sample.target_up)
        self.assertEqual(self.assess(rows_for(self.calendar, 60)).samples, ())

    def test_policy_and_results_are_frozen_and_rows_unchanged(self):
        rows = rows_for(self.calendar, 61)
        original = copy.deepcopy(rows)
        result = self.assess(rows)
        self.assertEqual(rows, original)
        with self.assertRaises(FrozenInstanceError):
            result.bars[0].volume = 9
        with self.assertRaises(FrozenInstanceError):
            self.policy.lookback = 9

    def test_future_values_cannot_change_prior_input_eligibility(self):
        rows = rows_for(self.calendar, 90)
        before = self.assess(rows)
        for row in rows[60:]:
            row.update(open=1000, high=2000, low=900, close=1500,
                       volume=1, liquidity_turnover=None)
        after = self.assess(rows)
        self.assertEqual(before.bars[:60], after.bars[:60])
        self.assertEqual(before.samples[0].input_end_index, after.samples[0].input_end_index)
        self.assertTrue(after.samples[0].target_up)

    def test_target_low_liquidity_and_extreme_return_do_not_remove_sample(self):
        rows = rows_for(self.calendar, 61)
        rows[-1].update(open=400, high=800, low=100, close=500,
                        volume=1, liquidity_turnover=0)
        result = self.assess(rows)
        self.assertEqual(len(result.samples), 1)
        self.assertTrue(result.samples[0].target_up)
        self.assertTrue(result.bars[-1].valid)
        self.assertIn("HIGH_LOW_RATIO_GE_2", result.bars[-1].flags)
        self.assertIn("ABS_RETURN_GE_50PCT", result.bars[-1].flags)

    def test_target_missing_turnover_does_not_select_sample(self):
        rows = rows_for(self.calendar, 61)
        rows[-1]["liquidity_turnover"] = None
        result = self.assess(rows)
        self.assertEqual(len(result.samples), 1)
        self.assertFalse(result.bars[-1].input_eligible)

    def test_target_label_is_at_least_one_percent_not_merely_positive(self):
        for close, expected in ((100, False), (100.5, False), (100.9999, False), (101, True), (101.0001, True)):
            with self.subTest(close=close):
                rows = rows_for(self.calendar, 61)
                rows[-1]["close"] = close
                self.assertIs(self.assess(rows).samples[0].target_up, expected)

    def test_gap_is_not_bridged_and_restarts_full_liquidity_warmup(self):
        rows = rows_for(self.calendar, 125)
        del rows[60]
        result = self.assess(rows)
        by_date = {bar.date: bar for bar in result.bars}
        self.assertFalse(by_date[self.calendar[61]].input_eligible)
        self.assertFalse(by_date[self.calendar[119]].input_eligible)
        self.assertTrue(by_date[self.calendar[120]].input_eligible)
        self.assertEqual(result.samples[0].input_end_date, self.calendar[120])
        self.assertEqual(result.samples[0].target_date, self.calendar[121])

    def test_bad_bar_is_not_deleted_then_bridged(self):
        rows = rows_for(self.calendar, 125)
        rows[60]["close"] = float("nan")
        result = self.assess(rows)
        self.assertEqual(len(result.bars), 125)
        self.assertFalse(result.bars[60].valid)
        self.assertFalse(result.bars[119].input_eligible)
        self.assertTrue(result.bars[120].input_eligible)
        self.assertEqual(result.samples[0].input_end_index, 120)

    def test_duplicate_dates_are_all_quarantined_not_chosen_arbitrarily(self):
        rows = rows_for(self.calendar, 65)
        rows.append(dict(rows[59], close=105))
        result = self.assess(rows)
        self.assertIn("DUPLICATE_DATE", result.bars[59].hard_errors)
        self.assertIn("DUPLICATE_DATE", result.bars[-1].hard_errors)
        self.assertEqual(result.samples, ())

    def test_unsorted_rows_keep_source_indexes_and_produce_chronological_samples(self):
        rows = list(reversed(rows_for(self.calendar, 65)))
        result = self.assess(rows)
        self.assertEqual([bar.date for bar in result.bars], [row["date"] for row in rows])
        self.assertEqual([sample.input_end_date for sample in result.samples], self.calendar[59:64])
        self.assertEqual((result.samples[0].input_start_index, result.samples[0].input_end_index,
                          result.samples[0].target_index), (34, 5, 4))

    def test_zero_volume_is_valid_flagged_and_can_be_an_input(self):
        rows = rows_for(self.calendar, 61)
        rows[40].update(volume=0, liquidity_turnover=0)
        result = self.assess(rows)
        self.assertTrue(result.bars[40].valid)
        self.assertIn("ZERO_VOLUME", result.bars[40].flags)
        self.assertFalse(result.bars[40].input_eligible)
        self.assertEqual(len(result.samples), 1)
        self.assertEqual(result.bars[59].max_zero_run, 1)

    def test_zero_volume_target_is_not_used_but_remains_valid(self):
        rows = rows_for(self.calendar, 61)
        rows[60].update(volume=0, liquidity_turnover=0)
        result = self.assess(rows)
        self.assertTrue(result.bars[60].valid)
        self.assertEqual(result.samples, ())

    def test_two_zero_run_allowed_three_blocked_at_same_active_fraction_boundary(self):
        rows = rows_for(self.calendar, 61)
        for index in (30, 31, 40):
            rows[index].update(volume=0, liquidity_turnover=0)
        result = self.assess(rows)
        self.assertEqual(result.bars[59].active_fraction, 0.95)
        self.assertEqual(result.bars[59].max_zero_run, 2)
        self.assertTrue(result.bars[59].input_eligible)
        rows[40].update(volume=10000, liquidity_turnover=1e9)
        rows[32].update(volume=0, liquidity_turnover=0)
        result = self.assess(rows)
        self.assertFalse(result.bars[59].input_eligible)
        self.assertIn("EXCESS_ZERO_RUN", result.bars[59].flags)

    def test_active_fraction_uses_sixty_sessions_and_zero_runs_expire(self):
        rows = rows_for(self.calendar, 121)
        for index in (0, 1, 2, 10):
            rows[index].update(volume=0, liquidity_turnover=0)
        result = self.assess(rows)
        self.assertFalse(result.bars[59].input_eligible)
        self.assertIn("LOW_ACTIVE_FRACTION", result.bars[59].flags)
        self.assertEqual(result.bars[60].active_fraction, 0.95)
        self.assertEqual(result.bars[60].max_zero_run, 2)
        self.assertTrue(result.bars[60].input_eligible)
        self.assertEqual(result.bars[70].max_zero_run, 0)
        self.assertEqual(result.bars[70].active_fraction, 1)

    def test_median_window_is_twenty_and_includes_current_endpoint(self):
        rows = rows_for(self.calendar, 62)
        for index in range(40, 50):
            rows[index].update(volume=1, liquidity_turnover=1)
        result = self.assess(rows)
        self.assertEqual(result.bars[59].median_volume, 5000.5)
        self.assertFalse(result.bars[59].input_eligible)
        self.assertEqual(result.bars[60].median_volume, 10000)
        self.assertTrue(result.bars[60].input_eligible)

    def test_raw_trade_value_is_preserved_and_never_used_as_money(self):
        rows = rows_for(self.calendar, 61)
        for row in rows:
            row["trade_value"] = "raw-source-units"
        result = self.assess(rows)
        self.assertEqual(result.bars[59].trade_value, "raw-source-units")
        self.assertTrue(result.bars[59].input_eligible)
        for row in rows:
            row.pop("liquidity_turnover")
            row["trade_value"] = 1e30
        result = self.assess(rows)
        self.assertTrue(result.bars[59].valid)
        self.assertFalse(result.bars[59].input_eligible)
        self.assertIn("LIQUIDITY_TURNOVER_UNAVAILABLE", result.bars[59].flags)

    def test_invalid_turnover_is_soft_and_expires_from_median_window(self):
        for value in (None, -1, float("nan"), float("inf"), "bad", True):
            with self.subTest(value=value):
                rows = rows_for(self.calendar, 81)
                rows[59]["liquidity_turnover"] = value
                result = self.assess(rows)
                self.assertTrue(result.bars[59].valid)
                self.assertFalse(result.bars[59].input_eligible)
                self.assertFalse(result.bars[78].input_eligible)
                self.assertTrue(result.bars[79].input_eligible)

    def test_valid_price_extremes_are_retained_and_flagged(self):
        rows = rows_for(self.calendar, 61)
        rows[40].update(open=100, low=50, high=200, close=150)
        result = self.assess(rows)
        self.assertTrue(result.bars[40].valid)
        self.assertIn("HIGH_LOW_RATIO_GE_2", result.bars[40].flags)
        self.assertIn("ABS_RETURN_GE_50PCT", result.bars[40].flags)
        self.assertEqual(len(result.samples), 1)

    def test_prices_must_be_positive_finite_and_float32_representable(self):
        for value in (None, 0, -1, float("nan"), float("inf"), float("-inf"), 1e40, 1e-50, "bad", True):
            for field in ("open", "high", "low", "close"):
                with self.subTest(value=value, field=field):
                    row = rows_for(self.calendar, 1)[0]
                    row[field] = value
                    self.assertIn("INVALID_" + field.upper(), self.assess([row]).bars[0].hard_errors)
        row = rows_for(self.calendar, 1)[0]
        row.update(open=1e-40, high=1e-40, low=1e-40, close=1e-40)
        self.assertTrue(self.assess([row]).bars[0].valid)

    def test_volume_must_be_finite_nonnegative_integer(self):
        for value in (None, -1, 1.5, float("nan"), float("inf"), "1.00000000000000000001", "bad", True):
            with self.subTest(value=value):
                row = rows_for(self.calendar, 1)[0]
                row["volume"] = value
                self.assertIn("INVALID_VOLUME", self.assess([row]).bars[0].hard_errors)
        for value in (0, 10000.0, "10000", Decimal("10000.000")):
            row = rows_for(self.calendar, 1)[0]
            row["volume"] = value
            self.assertTrue(self.assess([row]).bars[0].valid)

    def test_ohlc_bounds_and_currency_are_hard_failures(self):
        for changes in (dict(high=95), dict(low=105), dict(high=80, low=90), dict(currency="USD")):
            row = rows_for(self.calendar, 1)[0]
            row.update(changes)
            self.assertFalse(self.assess([row]).bars[0].valid)
        self.assertTrue(self.assess(rows_for(self.calendar, 1, currency="USD"), market="us").bars[0].valid)

    def test_dates_are_exact_iso_completed_and_calendar_covered(self):
        for value, expected in (("20240101", "INVALID_DATE"), ("2024-1-1", "INVALID_DATE"),
                                ("2024-01-01 ", "INVALID_DATE"), ("2024-02-30", "INVALID_DATE"),
                                (date(2024, 1, 1), "INVALID_DATE"), ("2024-01-06", "NON_SESSION_DATE"),
                                ("2023-12-29", "OUTSIDE_CALENDAR_COVERAGE")):
            row = rows_for(self.calendar, 1)[0]
            row["date"] = value
            self.assertIn(expected, self.assess([row]).bars[0].hard_errors)
        result = self.assess(rows_for(self.calendar, 62), as_of=date.fromisoformat(self.calendar[60]))
        self.assertTrue(result.bars[59].valid)
        self.assertIn("INCOMPLETE_SESSION", result.bars[60].hard_errors)
        self.assertIn("INCOMPLETE_SESSION", result.bars[61].hard_errors)
        self.assertEqual(result.samples, ())

    def test_invalid_calendars_and_policies_fail_explicitly(self):
        for calendar in ([], ["20240101"], ["2024-01-02", "2024-01-01"], ["2024-01-01"] * 2):
            with self.assertRaises(ValueError):
                self.assess([], session_dates=calendar)
        for changes in (dict(lookback=0), dict(lookback=True), dict(median_window=61),
                        dict(max_zero_run=-1), dict(max_zero_run=61), dict(min_active_fraction=1.1),
                        dict(min_active_fraction=float("nan")), dict(min_median_turnover=-1),
                        dict(min_median_volume=float("inf"))):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.policy, **changes)
        with self.assertRaises(ValueError):
            self.assess([], market="unknown")
        with self.assertRaises(ValueError):
            self.assess([], as_of="2024-01-01")

    def test_empty_series_is_empty_with_valid_calendar(self):
        result = self.assess([])
        self.assertEqual(result.bars, ())
        self.assertEqual(result.samples, ())

    def test_rolling_eligibility_matches_independent_small_window_oracle(self):
        rng = random.Random(1729)
        rows = rows_for(self.calendar, 120)
        for row in rows:
            row["volume"] = rng.choice((0, 0, 1000, 10000, 20000))
            row["liquidity_turnover"] = rng.choice((None, 0, 1e8, 1e9, 2e9))
        rows[21]["high"] = 80  # Hard-invalid point must interrupt a window.
        del rows[53]  # A missing scheduled session must also interrupt it.
        policy = QualityPolicy(min_median_turnover=1e8, min_median_volume=1000,
                               lookback=3, liquidity_window=7, median_window=3,
                               min_active_fraction=0.5, max_zero_run=2)
        result = self.assess(rows, policy=policy)
        valid_by_date = {bar.date: bar for bar in result.bars if bar.valid}
        expected_sample_dates = []
        for day_index, day in enumerate(self.calendar[:120]):
            if day not in valid_by_date:
                continue
            bar = valid_by_date[day]
            eligible = False
            history_dates = self.calendar[max(0, day_index - 6):day_index + 1]
            if len(history_dates) == 7 and all(key in valid_by_date for key in history_dates):
                history = [valid_by_date[key] for key in history_dates]
                median_bars = history[-3:]
                zero_run = maximum_run = 0
                for item in history:
                    zero_run = zero_run + 1 if item.volume == 0 else 0
                    maximum_run = max(maximum_run, zero_run)
                turnover_known = all(item.liquidity_turnover is not None for item in median_bars)
                eligible = (bar.volume > 0 and sum(item.volume > 0 for item in history) / 7 >= 0.5
                            and maximum_run <= 2
                            and statistics.median(item.volume for item in median_bars) >= 1000
                            and turnover_known
                            and statistics.median(item.liquidity_turnover for item in median_bars) >= 1e8)
            self.assertEqual(bar.input_eligible, eligible, day)
            target = valid_by_date.get(self.calendar[day_index + 1])
            if eligible and target is not None and target.volume > 0:
                expected_sample_dates.append((day, target.date))
        self.assertEqual([(sample.input_end_date, sample.target_date) for sample in result.samples],
                         expected_sample_dates)


if __name__ == "__main__":
    unittest.main()
