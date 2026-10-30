"""Pure paper hedge research: no broker, order, or account fixtures."""

import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from dockdack.minute_hedge_research import (
    HedgeLeg, HedgePolicy, MinuteBar, MinuteSession, PaperCosts, Variant,
    simulate_session, walk_forward,
)


TZ = ZoneInfo("America/New_York")
COSTS = PaperCosts(commission_bps_per_side=1, slippage_bps_per_side=2,
                   index_short_borrow_bps_annual=500, trading_minutes_per_year=252 * 390)
STOCK = (100, 101, 100, 101, 90, 92, 92, 90, 95, 96, 96, 97,
         97, 97, 97, 97, 97, 97, 97, 97)
INDEX = (100, 100.5, 100.2, 100.7, 100.4, 101, 100.6, 100, 100, 100,
         100, 100.2, 100.2, 100.2, 100.2, 100.2, 100.2, 100.2, 100.2, 100.2)


def bars(prices, *, day=1, volume=1_000, turnover=True, interval=1):
    start = datetime(2026, 9, day, 9, 30, tzinfo=TZ)
    return tuple(MinuteBar(start + timedelta(minutes=i * interval), p, p + 0.2, p - 0.2, p,
                           volume, p * volume if turnover else None)
                 for i, p in enumerate(prices))


def session(*, day=1, stock=STOCK, index=INDEX, volume=1_000, etf=True, interval=1):
    inverse = (50, 49.9, 50, 49.8, 50, 50, 49.9, 50, 50, 50, 50,
               49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5)
    return MinuteSession("XYZ", "SP500", "USD", interval,
                         bars(stock, day=day, volume=volume, interval=interval),
                         bars(index, day=day, interval=interval),
                         source="test fixture, not historical returns",
                         sector_id="TECH", sector=bars((100,) * len(stock), day=day, interval=interval),
                         inverse_etf_id="INV" if etf else None,
                         inverse_etf_currency="USD" if etf else None,
                         inverse_etf_index_multiple=-1.0 if etf else None,
                         inverse_etf=bars(inverse, day=day, interval=interval) if etf else None,
                         previous_close_day=date(2026, 8, 31),
                         previous_stock_close=110, previous_index_close=100)


def policy(variant, threshold):
    return HedgePolicy(variant=variant, lookback=4, entry_threshold=threshold,
                       take_profit=0.03, stop_loss=0.3, max_hold_minutes=3,
                       min_prior_volume=3_000, min_prior_turnover=100_000)


class MinuteHedgeResearchTests(unittest.TestCase):
    def test_all_five_distinct_policies_use_next_open_and_close_before_session_end(self):
        thresholds = {Variant.RESIDUAL_Z: 2.0, Variant.SECTOR_RELATIVE: 0.05,
                      Variant.TURNOVER_VWAP: 0.05, Variant.ATR_DROP: 2.0,
                      Variant.GAP_RELATIVE: 0.1}
        data = session()
        for variant, threshold in thresholds.items():
            with self.subTest(variant=variant):
                result = simulate_session(data, policy(variant, threshold), COSTS,
                                          notional_per_leg=1_000,
                                          hedge_leg=HedgeLeg.SYNTHETIC_INDEX_SHORT)
                self.assertGreaterEqual(len(result.trades), 1)
                first = result.trades[0]
                self.assertEqual(first.entry_time, data.stock[7].time)
                self.assertEqual(first.stock_entry, data.stock[7].open)
                self.assertEqual(first.exit_time, data.stock[11].time)
                self.assertEqual(first.exit_reason, "take_profit")
                self.assertEqual(first.hedge_leg, HedgeLeg.SYNTHETIC_INDEX_SHORT)
                self.assertGreater(first.index_short_borrow_cost, 0)
                self.assertEqual(first.inverse_etf_pnl, 0)

    def test_inverse_etf_uses_observed_etf_price_not_perfect_index_short(self):
        data = session()
        strategy = policy(Variant.TURNOVER_VWAP, 0.05)
        synthetic = simulate_session(data, strategy, COSTS, notional_per_leg=1_000,
                                     hedge_leg=HedgeLeg.SYNTHETIC_INDEX_SHORT).trades[0]
        etf = simulate_session(data, strategy, COSTS, notional_per_leg=1_000,
                               hedge_leg=HedgeLeg.LONG_INVERSE_ETF).trades[0]
        self.assertEqual(etf.synthetic_index_short_pnl, 0)
        self.assertAlmostEqual(etf.inverse_etf_pnl, 1_000 * (49.5 / 50 - 1))
        self.assertAlmostEqual(etf.inverse_etf_vs_benchmark_return_gap,
                               49.5 / 50 - 1 + 100.2 / 100 - 1)
        self.assertEqual(etf.index_short_borrow_cost, 0)
        self.assertNotEqual(etf.net_pnl, synthetic.net_pnl)
        self.assertAlmostEqual(etf.net_pnl, etf.long_stock_pnl + etf.inverse_etf_pnl - etf.round_trip_cost)

    def test_five_minute_bars_preserve_next_bar_execution_and_borrow_duration(self):
        data = session(interval=5)
        strategy = replace(policy(Variant.TURNOVER_VWAP, 0.05), max_hold_minutes=15)
        trade = simulate_session(data, strategy, COSTS, notional_per_leg=1_000,
                                 hedge_leg=HedgeLeg.SYNTHETIC_INDEX_SHORT).trades[0]
        self.assertEqual(trade.entry_time, data.stock[7].time)
        self.assertEqual(trade.exit_time, data.stock[11].time)
        expected_borrow = 1_000 * 500 / 10_000 * 20 / (252 * 390)
        self.assertAlmostEqual(trade.index_short_borrow_cost, expected_borrow)
        with self.assertRaisesRegex(ValueError, "multiple"):
            simulate_session(data, policy(Variant.TURNOVER_VWAP, 0.05), COSTS,
                             notional_per_leg=1_000, hedge_leg=HedgeLeg.SYNTHETIC_INDEX_SHORT)

    def test_volume_shock_and_range_recovery_use_only_completed_signal_bar(self):
        data = session()
        stock = list(data.stock)
        stock[4] = replace(stock[4], volume=3_000, turnover=270_000,
                           low=88, high=90.2)
        data = replace(data, stock=tuple(stock))
        for variant in (Variant.VOLUME_SHOCK_REVERSAL, Variant.RANGE_RECOVERY):
            with self.subTest(variant=variant):
                trade = simulate_session(data, policy(variant, 0.05), COSTS,
                                         notional_per_leg=1_000,
                                         hedge_leg=HedgeLeg.LONG_INVERSE_ETF).trades[0]
                self.assertEqual(trade.entry_time, data.stock[7].time)
                self.assertEqual(trade.exit_time, data.stock[11].time)
        # The volume shock is absent when the current completed bar does not
        # exceed the previous four-bar mean. Future volume cannot rescue it.
        calm = replace(data, stock=tuple(replace(bar, volume=1_000,
                                                 turnover=bar.close * 1_000)
                                          if i == 4 else bar
                                          for i, bar in enumerate(data.stock)))
        result = simulate_session(calm, policy(Variant.VOLUME_SHOCK_REVERSAL, 0.05), COSTS,
                                  notional_per_leg=1_000, hedge_leg=HedgeLeg.LONG_INVERSE_ETF)
        self.assertEqual(result.trades, ())

    def test_inverse_etf_absence_or_misalignment_fails_closed(self):
        data = session(etf=False)
        with self.assertRaisesRegex(ValueError, "requires aligned ETF"):
            simulate_session(data, policy(Variant.TURNOVER_VWAP, 0.05), COSTS,
                             notional_per_leg=1_000, hedge_leg=HedgeLeg.LONG_INVERSE_ETF)
        with self.assertRaisesRegex(ValueError, "times must match"):
            replace(session(), inverse_etf=bars((50,) * len(STOCK), day=2))
        with self.assertRaisesRegex(ValueError, "share stock currency"):
            replace(session(), inverse_etf_currency="KRW")
        with self.assertRaisesRegex(ValueError, "-1x"):
            replace(session(), inverse_etf_index_multiple=-2.0)

    def test_prior_liquidity_excludes_current_signal_bar(self):
        data = session(volume=0)
        stock = list(data.stock)
        stock[4] = replace(stock[4], volume=1_000_000, turnover=90_000_000)
        data = replace(data, stock=tuple(stock))
        result = simulate_session(data, policy(Variant.TURNOVER_VWAP, 0.05), COSTS,
                                  notional_per_leg=1_000, hedge_leg=HedgeLeg.LONG_INVERSE_ETF)
        self.assertEqual(result.trades, ())

    def test_no_actual_turnover_no_vwap_trade(self):
        data = session()
        data = replace(data, stock=bars(STOCK, turnover=False))
        result = simulate_session(data, policy(Variant.TURNOVER_VWAP, 0.05), COSTS,
                                  notional_per_leg=1_000, hedge_leg=HedgeLeg.LONG_INVERSE_ETF)
        self.assertEqual(result.trades, ())

    def test_invalid_or_noncontiguous_bars_are_rejected(self):
        with self.assertRaises(ValueError):
            MinuteBar(datetime(2026, 9, 1, 9, 30), 100, 101, 99, 100, 1)
        with self.assertRaises(ValueError):
            MinuteBar(datetime(2026, 9, 1, 9, 30, tzinfo=TZ), 100, 99, 98, 100, 1)
        data = session()
        broken = list(data.stock)
        broken[2] = replace(broken[2], time=broken[2].time + timedelta(minutes=2))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            replace(data, stock=tuple(broken))
        with self.assertRaisesRegex(ValueError, "previous close day"):
            replace(data, previous_close_day=data.day)

    def test_walk_forward_selects_only_from_prior_sessions(self):
        data = (session(day=1), session(day=2), session(day=3))
        permissive = policy(Variant.TURNOVER_VWAP, 0.05)
        impossible = policy(Variant.TURNOVER_VWAP, 0.9)
        folds = walk_forward(data, (impossible, permissive), COSTS,
                             notional_per_leg=1_000, min_train_sessions=2,
                             min_train_trades=2, hedge_leg=HedgeLeg.LONG_INVERSE_ETF)
        self.assertEqual(len(folds), 1)
        self.assertEqual(folds[0].selected_policy, permissive)
        self.assertEqual(folds[0].training_days, (data[0].day, data[1].day))
        self.assertEqual(folds[0].test_days, (data[2].day,))
        changed = (data[0], data[1], session(day=3, stock=(100,) * len(STOCK)))
        later = walk_forward(changed, (impossible, permissive), COSTS,
                             notional_per_leg=1_000, min_train_sessions=2,
                             min_train_trades=2, hedge_leg=HedgeLeg.LONG_INVERSE_ETF)
        self.assertEqual(later[0].selected_policy, permissive)
        self.assertEqual(later[0].training_net_pnl, folds[0].training_net_pnl)

    def test_no_positive_training_evidence_abstains(self):
        data = (session(day=1, stock=(100,) * len(STOCK)),
                session(day=2, stock=(100,) * len(STOCK)))
        folds = walk_forward(data, (policy(Variant.TURNOVER_VWAP, 0.05),), COSTS,
                             notional_per_leg=1_000, min_train_sessions=1,
                             hedge_leg=HedgeLeg.LONG_INVERSE_ETF)
        self.assertIsNone(folds[0].selected_policy)
        self.assertEqual(folds[0].test_results, ())


if __name__ == "__main__":
    unittest.main()
