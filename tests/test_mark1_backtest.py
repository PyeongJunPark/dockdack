"""Deterministic accounting/chronology regression tests; no broker or DB access."""
import copy
import json
import math
import random
import unittest

from dockdack.mark1_backtest import simulate_portfolio


def signal(symbol=1, day=10, probability=.6, liquidity=1_000_000):
    return dict(symbol_id=symbol, date=day, probability=probability,
                liquidity_shares=liquidity)


QUIET = (100., 100.5, 99.5, 100.)


class Mark1BacktestTests(unittest.TestCase):
    def run_case(self, candidates=None, prices=None, sessions=None, **kwargs):
        return simulate_portfolio(
            [signal()] if candidates is None else candidates,
            {(1, 10): QUIET} if prices is None else prices,
            [10] if sessions is None else sessions,
            initial_cash=kwargs.pop("initial_cash", 10_000),
            cost_bps=kwargs.pop("cost_bps", 0),
            position_fraction=kwargs.pop("position_fraction", .5), **kwargs)

    def test_exact_half_does_not_buy(self):
        result = self.run_case([signal(probability=.5)])
        self.assertEqual(result["summary"]["trade_count"], 0)
        self.assertEqual(result["rejections"]["counts"], {"BELOW_THRESHOLD": 1})
        self.assertEqual(result["summary"]["final_equity"], 10_000)
        self.assertIsNone(result["summary"]["win_rate"])

    def test_next_float_above_half_buys(self):
        result = self.run_case([signal(probability=math.nextafter(.5, 1.))])
        self.assertEqual(result["summary"]["trade_count"], 1)

    def test_both_hits_stop_first(self):
        result = self.run_case(prices={(1, 10): (100, 102, 98, 100)})
        trade = result["trades"][0]
        self.assertEqual(trade["exit_reason"], "STOP")
        self.assertTrue(trade["both_touch"])
        self.assertAlmostEqual(trade["exit_price"], 99.1)
        self.assertAlmostEqual(trade["gross_return"], -.009)
        self.assertEqual(result["summary"]["both_touch_stop_count"], 1)

    def test_exact_barriers_include_float_tolerance(self):
        result = self.run_case(prices={(1, 10): (100, 101, 99.1, 100)})
        self.assertEqual(result["trades"][0]["exit_reason"], "STOP")
        result = self.run_case(prices={(1, 10): (100, 101, 99.2, 100)})
        self.assertEqual(result["trades"][0]["exit_reason"], "TAKE")

    def test_costs_cash_and_pnl_reconcile(self):
        result = self.run_case(prices={(1, 10): (100, 101, 99.5, 100)}, cost_bps=20)
        trade = result["trades"][0]
        self.assertEqual(trade["quantity"], 50)
        self.assertAlmostEqual(trade["entry_fee"], 5)
        self.assertAlmostEqual(trade["exit_fee"], 5.05)
        self.assertAlmostEqual(trade["net_pnl"], 39.95)
        self.assertAlmostEqual(result["summary"]["final_equity"], 10_039.95)
        self.assertAlmostEqual(result["summary"]["fees"], 10.05)

    def test_cash_budget_includes_fee_and_no_leverage(self):
        rows = [signal(s) for s in range(5)]
        prices = {(s, 10): QUIET for s in range(5)}
        result = self.run_case(rows, prices, position_fraction=1, cost_bps=20)
        self.assertEqual(len(result["trades"]), 1)
        self.assertEqual(result["trades"][0]["quantity"], 99)
        self.assertLessEqual(result["summary"]["max_open_exposure"], 1)
        self.assertTrue(all(row["cash"] >= 0 for row in result["equity"]))
        self.assertEqual(result["rejections"]["counts"]["INSUFFICIENT_CASH"], 4)

    def test_probability_order_then_symbol_order(self):
        rows = [signal(5, probability=.8), signal(3, probability=.8), signal(1, probability=.6)]
        result = self.run_case(rows, {(s, 10): QUIET for s in (5, 3, 1)}, max_positions=1)
        self.assertEqual(result["trades"][0]["symbol_id"], 3)
        self.assertEqual(result["rejections"]["counts"]["POSITION_CAP"], 2)

    def test_new_intraday_exit_does_not_free_slot_for_another_open(self):
        result = self.run_case([signal(1, probability=.8), signal(2)],
                               {(1, 10): (100, 102, 99.5, 101), (2, 10): QUIET}, max_positions=1)
        self.assertEqual(len(result["trades"]), 1)
        self.assertEqual(result["rejections"]["counts"]["POSITION_CAP"], 1)

    def test_prior_intraday_exit_does_not_free_slot_for_today_open(self):
        prices = {(1, 10): QUIET, (1, 11): (100, 102, 99.5, 101), (2, 11): QUIET}
        result = self.run_case([signal(1, 10), signal(2, 11)], prices, [10, 11], max_positions=1)
        self.assertEqual(len(result["trades"]), 1)
        self.assertEqual(result["trades"][0]["exit_date"], 11)
        self.assertEqual(result["rejections"]["counts"]["POSITION_CAP"], 1)

    def test_prior_gap_exit_frees_slot_and_cash_before_open_entries(self):
        prices = {(1, 10): QUIET, (1, 11): (103, 104, 102, 103), (2, 11): QUIET}
        result = self.run_case([signal(1, 10), signal(2, 11)], prices, [10, 11], max_positions=1)
        self.assertEqual(len(result["trades"]), 2)
        self.assertEqual(result["trades"][0]["exit_reason"], "GAP_TAKE")
        self.assertEqual(result["trades"][0]["exit_price"], 103)

    def test_gap_stop_fills_worse_open_not_stop_price(self):
        prices = {(1, 10): QUIET, (1, 11): (90, 92, 89, 91)}
        result = self.run_case(prices=prices, sessions=[10, 11])
        self.assertEqual(result["trades"][0]["exit_reason"], "GAP_STOP")
        self.assertEqual(result["trades"][0]["exit_price"], 90)
        self.assertAlmostEqual(result["summary"]["total_return"], -.05)

    def test_cannot_reenter_symbol_after_same_open_gap_exit(self):
        prices = {(1, 10): QUIET, (1, 11): (103, 104, 102, 103)}
        result = self.run_case([signal(1, 10), signal(1, 11)], prices, [10, 11])
        self.assertEqual(len(result["trades"]), 1)
        self.assertEqual(result["rejections"]["counts"]["ALREADY_EXITED_TODAY"], 1)

    def test_already_held_not_averaged_up(self):
        result = self.run_case([signal(1, 10), signal(1, 11)],
                               {(1, 10): QUIET, (1, 11): QUIET}, [10, 11])
        self.assertEqual(result["rejections"]["counts"]["ALREADY_HELD"], 1)
        self.assertEqual(result["trades"][0]["holding_days"], 2)

    def test_eod_and_carry_differ_without_hit(self):
        prices = {(1, 10): (100, 100.5, 99.5, 100.2), (1, 11): (100.2, 102, 100, 101)}
        carry = self.run_case(prices=prices, sessions=[10, 11])
        eod = self.run_case(prices=prices, sessions=[10, 11], exit_mode="eod")
        self.assertEqual(carry["trades"][0]["exit_reason"], "TAKE")
        self.assertEqual(eod["trades"][0]["exit_reason"], "EOD_CLOSE")
        self.assertEqual(eod["trades"][0]["exit_date"], 10)

    def test_volume_fraction_integer_floor(self):
        result = self.run_case([signal(liquidity=12_999)])
        self.assertEqual(result["trades"][0]["quantity"], 12)
        result = self.run_case([signal(liquidity=999)])
        self.assertEqual(result["rejections"]["counts"], {"LIQUIDITY_CAP": 1})

    def test_data_gap_locks_cash_and_resumes_at_observed_open(self):
        prices = {(1, 10): (100, 100.5, 99.5, 100.2),
                  (1, 12): (80, 81, 79, 80), (2, 11): QUIET}
        result = self.run_case([signal(1, 10), signal(2, 11)], prices, [10, 11, 12], max_positions=1)
        trade = result["trades"][0]
        self.assertEqual(trade["exit_reason"], "DATA_GAP_RESUMPTION")
        self.assertEqual(trade["exit_date"], 12)
        self.assertEqual(trade["exit_price"], 80)
        self.assertTrue(trade["path_uncertain"])
        self.assertEqual(result["equity"][2]["equity"], result["equity"][1]["equity"])
        self.assertEqual(result["rejections"]["counts"]["POSITION_CAP"], 1)
        self.assertFalse(result["summary"]["fully_observed"])
        self.assertEqual(result["summary"]["gap_valuation_days"], 1)
        self.assertEqual(result["summary"]["uncertain_trades"], 1)

    def test_terminal_stale_mark_is_explicit_uncertain_accounting(self):
        result = self.run_case(prices={(1, 10): QUIET}, sessions=[10, 11, 12])
        trade = result["trades"][0]
        self.assertEqual(trade["exit_reason"], "END_STALE_MARK")
        self.assertEqual(trade["exit_price_date"], 10)
        self.assertEqual(trade["exit_date"], 12)
        self.assertTrue(trade["path_uncertain"])
        self.assertEqual(result["summary"]["gap_valuation_days"], 2)
        self.assertEqual(result["summary"]["gap_calendar_days"], 2)

    def test_missing_entry_price_is_rejected_not_forward_filled(self):
        result = self.run_case(prices={})
        self.assertEqual(result["rejections"]["counts"], {"MISSING_PRICE": 1})
        self.assertTrue(result["summary"]["fully_observed"])

    def test_final_close_liquidation_and_initial_anchor_drawdown(self):
        result = self.run_case(prices={(1, 10): (100, 100.5, 99.5, 99.5)})
        self.assertTrue(result["equity"][0]["is_initial"])
        self.assertEqual(result["trades"][0]["exit_reason"], "END_OF_DATA")
        self.assertAlmostEqual(result["summary"]["max_drawdown"], -.0025)
        self.assertAlmostEqual(result["summary"]["net_pnl"], -25)
        self.assertEqual(result["equity"][-1]["position_count"], 0)

    def test_no_mutation_and_json_finite(self):
        rows, prices, sessions = [signal()], {(1, 10): QUIET}, [10]
        original = copy.deepcopy((rows, prices, sessions))
        result = self.run_case(rows, prices, sessions)
        self.assertEqual((rows, prices, sessions), original)
        json.dumps(result, allow_nan=False)

    def test_empty_signal_calendar_produces_flat_curve(self):
        result = self.run_case([], {}, [10, 11, 12])
        self.assertEqual(len(result["equity"]), 4)
        self.assertEqual(result["summary"]["total_return"], 0)
        self.assertEqual(result["summary"]["mean_exposure"], 0)
        self.assertIsNone(result["summary"]["sharpe"])

    def test_invalid_parameters_rejected(self):
        for kwargs in (dict(initial_cash=0), dict(initial_cash=math.inf),
                       dict(max_positions=0), dict(max_positions=True),
                       dict(position_fraction=0), dict(position_fraction=1.1),
                       dict(volume_fraction=0), dict(volume_fraction=1.1),
                       dict(cost_bps=-1), dict(cost_bps=20_000), dict(exit_mode="unknown")):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_case(**kwargs)

    def test_invalid_candidates_rejected(self):
        for row in (signal(probability=math.nan), signal(probability=1.1),
                    signal(probability=-.1), signal(liquidity=-1),
                    signal(symbol=-1), signal(symbol=True), signal(day=11)):
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.run_case([row])
        with self.assertRaises(ValueError):
            self.run_case([signal(), signal()])

    def test_invalid_prices_and_calendars_rejected(self):
        for bar in ((0, 1, 0, 1), (100, 90, 80, 100), (100, 110, 105, 110),
                    (100, math.inf, 99, 100), (100, 110, 99), (True, 110, 99, 100)):
            with self.subTest(bar=bar), self.assertRaises(ValueError):
                self.run_case(prices={(1, 10): bar})
        for sessions in ([], [10, 10], [11, 10], [True]):
            with self.subTest(sessions=sessions), self.assertRaises(ValueError):
                self.run_case(sessions=sessions)

    def test_full_sessions_include_idle_days_in_returns(self):
        result = self.run_case(prices={(1, 10): (100, 101, 99.5, 100)},
                               sessions=[10, 11, 12, 13])
        self.assertEqual([row["daily_return"] for row in result["equity"]][2:], [0, 0, 0])
        self.assertEqual(result["summary"]["sessions"], 4)
        self.assertAlmostEqual(result["summary"]["mean_exposure"], .5 / 4)
        self.assertGreater(result["summary"]["sharpe"], 0)

    def test_eod_loss_and_idle_days_drawdown_stay_negative(self):
        result = self.run_case(prices={(1, 10): (100, 100.1, 99.5, 99.5)},
                               sessions=[10, 11, 12], exit_mode="eod")
        self.assertAlmostEqual(result["summary"]["max_drawdown"], -.0025)
        self.assertEqual(result["summary"]["profit_factor"], 0)
        self.assertEqual(result["summary"]["win_rate"], 0)

    def test_randomized_accounting_and_exposure_invariants(self):
        for seed in range(20):
            rng = random.Random(seed)
            prices, rows = {}, []
            days = list(range(100, 160))
            for symbol in range(12):
                previous = 50. + symbol * 10
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
                    result = self.run_case(rows, prices, days, exit_mode=mode,
                                           cost_bps=20, max_positions=4, position_fraction=.27)
                    self.assertTrue(all(0 <= row["cash"] <= row["equity"] for row in result["equity"]))
                    self.assertTrue(all(row["position_count"] <= 4 for row in result["equity"]))
                    self.assertLessEqual(result["summary"]["max_open_exposure"], 1)
                    self.assertEqual(result["equity"][-1]["position_count"], 0)
                    self.assertAlmostEqual(10_000 + sum(t["net_pnl"] for t in result["trades"]),
                                           result["summary"]["final_equity"], places=7)
                    self.assertEqual(len({(t["symbol_id"], t["entry_date"]) for t in result["trades"]}),
                                     len(result["trades"]))
                    json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
