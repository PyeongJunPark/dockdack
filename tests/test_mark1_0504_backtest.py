"""Offline isolated small-barrier accounting and old-target parity checks."""
import copy
import json
import math
import random
import unittest

from dockdack.mark1_backtest import simulate_portfolio as legacy_portfolio
from dockdack.mark1_0504_backtest import simulate_portfolio


def signal(symbol=1, day=10, probability=.6, liquidity=1_000_000):
    return dict(symbol_id=symbol, date=day, probability=probability,
                liquidity_shares=liquidity)


QUIET = (100., 100.2, 99.8, 100.)


class SmallBarrierPortfolioTests(unittest.TestCase):
    def run_case(self, rows=None, prices=None, sessions=None, **kwargs):
        return simulate_portfolio(
            [signal()] if rows is None else rows,
            {(1, 10): QUIET} if prices is None else prices,
            [10] if sessions is None else sessions,
            initial_cash=kwargs.pop("initial_cash", 10_000),
            cost_bps=kwargs.pop("cost_bps", 0),
            position_fraction=kwargs.pop("position_fraction", .5), **kwargs)

    def test_exact_half_abstains_and_next_float_buys(self):
        self.assertEqual(self.run_case([signal(probability=.5)])["summary"]["trade_count"], 0)
        self.assertEqual(self.run_case([signal(probability=math.nextafter(.5, 1))])["summary"]["trade_count"], 1)

    def test_new_take_and_stop_are_exact_and_stop_first(self):
        for high, low, reason, price, both in (
            (100.5, 99.7, "TAKE", 100.5, False),
            (100.4, 99.6, "STOP", 99.6, False),
            (100.5, 99.6, "STOP", 99.6, True),
        ):
            with self.subTest(high=high, low=low):
                result = self.run_case(prices={(1, 10): (100, high, low, 100)})
                trade = result["trades"][0]
                self.assertEqual(trade["exit_reason"], reason)
                self.assertAlmostEqual(trade["exit_price"], price)
                self.assertEqual(trade["both_touch"], both)
                self.assertEqual(result["summary"]["take_profit"], .005)
                self.assertEqual(result["summary"]["stop_loss"], .004)

    def test_twenty_basis_point_costs_reduce_half_percent_take(self):
        result = self.run_case(prices={(1, 10): (100, 100.5, 99.7, 100)}, cost_bps=20)
        trade = result["trades"][0]
        self.assertEqual(trade["quantity"], 50)
        self.assertAlmostEqual(trade["gross_pnl"], 25)
        self.assertAlmostEqual(trade["fees"], 10.025)
        self.assertAlmostEqual(trade["net_pnl"], 14.975)
        self.assertAlmostEqual(result["summary"]["final_equity"], 10_014.975)

    def test_gap_stop_executes_worse_open(self):
        result = self.run_case(prices={(1, 10): QUIET, (1, 11): (90, 91, 89, 90)}, sessions=[10, 11])
        self.assertEqual(result["trades"][0]["exit_reason"], "GAP_STOP")
        self.assertEqual(result["trades"][0]["exit_price"], 90)

    def test_gap_take_executes_actual_open(self):
        result = self.run_case(prices={(1, 10): QUIET, (1, 11): (102, 103, 101, 102)}, sessions=[10, 11])
        self.assertEqual(result["trades"][0]["exit_reason"], "GAP_TAKE")
        self.assertEqual(result["trades"][0]["exit_price"], 102)

    def test_intraday_exit_does_not_reuse_opening_slot(self):
        result = self.run_case([signal(1, probability=.8), signal(2)],
            {(1, 10): (100, 100.5, 99.8, 100), (2, 10): QUIET}, max_positions=1)
        self.assertEqual(len(result["trades"]), 1)
        self.assertEqual(result["rejections"]["counts"]["POSITION_CAP"], 1)

    def test_eod_vs_carry_without_first_day_hit(self):
        prices = {(1, 10): QUIET, (1, 11): (100, 100.5, 99.8, 100)}
        carry = self.run_case(prices=prices, sessions=[10, 11])
        eod = self.run_case(prices=prices, sessions=[10, 11], exit_mode="eod")
        self.assertEqual(carry["trades"][0]["exit_reason"], "TAKE")
        self.assertEqual(carry["trades"][0]["holding_days"], 2)
        self.assertEqual(eod["trades"][0]["exit_reason"], "EOD_CLOSE")

    def test_missing_path_and_stale_terminal_remain_explicit(self):
        result = self.run_case(prices={(1, 10): QUIET}, sessions=[10, 11])
        self.assertFalse(result["summary"]["fully_observed"])
        self.assertEqual(result["trades"][0]["exit_reason"], "END_STALE_MARK")
        self.assertTrue(result["trades"][0]["path_uncertain"])

    def test_missing_path_resumption_is_not_assumed_barrier(self):
        result = self.run_case(prices={(1, 10): QUIET, (1, 12): (80, 81, 79, 80)}, sessions=[10, 11, 12])
        self.assertEqual(result["trades"][0]["exit_reason"], "DATA_GAP_RESUMPTION")
        self.assertEqual(result["trades"][0]["exit_price"], 80)

    def test_parameter_calls_do_not_mutate_defaults_or_legacy(self):
        bar = {(1, 10): (100, 100.5, 99.6, 100)}
        old = self.run_case(prices=bar, take_profit=.01, stop_loss=.009)
        new = self.run_case(prices=bar)
        legacy = legacy_portfolio([signal()], bar, [10], initial_cash=10_000, cost_bps=0)
        self.assertEqual(old["trades"][0]["exit_reason"], "END_OF_DATA")
        self.assertEqual(legacy["trades"][0]["exit_reason"], "END_OF_DATA")
        self.assertEqual(new["trades"][0]["exit_reason"], "STOP")

    def test_invalid_barriers_fail_closed(self):
        for name, values in (("take_profit", (0, -1, math.nan, math.inf, True, "0.005")),
                             ("stop_loss", (0, -1, 1, 2, math.nan, math.inf, True, "0.004"))):
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.run_case(**{name: value})

    def test_inputs_are_unmodified(self):
        rows, prices, sessions = [signal()], {(1, 10): QUIET}, [10]
        prior = copy.deepcopy((rows, prices, sessions))
        result = self.run_case(rows, prices, sessions)
        self.assertEqual((rows, prices, sessions), prior)
        json.dumps(result, allow_nan=False)

    def test_old_barrier_randomized_equivalence_and_new_invariants(self):
        for seed in range(12):
            rng, rows, prices = random.Random(seed), [], {}
            days = list(range(100, 140))
            for symbol in range(7):
                previous = 50 + symbol * 10
                for day in days:
                    opened = previous * (1 + rng.uniform(-.025, .025))
                    close = opened * (1 + rng.uniform(-.008, .008))
                    high = max(opened, close) * (1 + rng.random() * .025)
                    low = min(opened, close) * (1 - rng.random() * .025)
                    previous = close
                    if rng.random() > .04:
                        prices[(symbol, day)] = (opened, high, low, close)
                        if rng.random() < .5:
                            rows.append(signal(symbol, day, rng.random(), 123_456))
            for mode in ("carry", "eod"):
                with self.subTest(seed=seed, mode=mode):
                    options = dict(initial_cash=10_000, cost_bps=20, exit_mode=mode,
                                   max_positions=4, position_fraction=.27)
                    expected = legacy_portfolio(rows, prices, days, **options)
                    old = simulate_portfolio(rows, prices, days, take_profit=.01, stop_loss=.009, **options)
                    old["summary"].pop("take_profit")
                    old["summary"].pop("stop_loss")
                    self.assertEqual(old, expected)
                    new = simulate_portfolio(rows, prices, days, **options)
                    self.assertTrue(all(0 <= row["cash"] <= row["equity"] for row in new["equity"]))
                    self.assertTrue(all(row["position_count"] <= 4 for row in new["equity"]))
                    self.assertEqual(new["equity"][-1]["position_count"], 0)
                    self.assertAlmostEqual(10_000 + sum(t["net_pnl"] for t in new["trades"]),
                                           new["summary"]["final_equity"], places=7)


if __name__ == "__main__":
    unittest.main()
