"""Offline checks for a model lot with an upper target and a timed exit."""

import tempfile
import unittest
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import PropertyMock, patch

from dockdack.autotrade import AutoTrader
from dockdack.gui_service import Instrument
from dockdack.history import DailyHistory
from dockdack.market_schedule import session_on
from dockdack.models import Market, OrderSide, Quote
from dockdack.trading.model_exit_schedule import (
    MODEL_EXIT_SCHEDULES, ModelExitSchedule, planned_model_exit, timed_exit_due,
)
from dockdack.watchlist import MarketSnapshot, TriggerKind, TriggerRule, WatchItem, WatchStore
from test_autotrade import FakeTradingService, position


STRATEGY = "synthetic-take-only-h10"


class TakeOnlyHorizonExitTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.store = WatchStore(Path(folder.name) / "offline.sqlite3")
        self.service = FakeTradingService()
        self.service.positions = (position(),)
        self.item = WatchItem(Instrument(Market.DOMESTIC, "005930", "KRX"))
        self.store.save_item(self.item)
        self.fill = session_on(Market.DOMESTIC, date(2026, 9, 14)).opened + timedelta(minutes=20)
        self.now = self.fill + timedelta(minutes=1)
        self.engine = AutoTrader(self.service, self.store, clock=lambda: self.now)
        self.engine.enable_holdings_exits = True
        self.engine.prototype_lots_enabled = True
        self.lot = {
            "lot_id": "synthetic-confirmed-buy", "strategy_id": STRATEGY,
            "model_title": "synthetic take-only", "average_price": Decimal("100"),
            "take_profit_price": Decimal("103"), "stop_loss_price": None,
            "sellable_quantity": Decimal(1),
            "buy_fill_observed_at": self.fill.isoformat(),
        }

    def snapshot(self, price):
        inst = self.item.instrument
        quote = Quote(inst.market, inst.symbol, "synthetic", inst.exchange, Decimal(price), inst.currency)
        history = DailyHistory(inst.market, inst.symbol, inst.exchange, inst.currency, 0, ())
        return MarketSnapshot(quote, history, self.now)

    def rule(self, kind):
        return TriggerRule("holding-exit-synthetic", self.item.id, kind, OrderSide.SELL,
                           1, Decimal("100000"), Decimal("103") if kind is TriggerKind.PRICE_GE else None)

    def test_upper_only_target_rechecks_fresh_quote_without_a_stop(self):
        rule = self.rule(TriggerKind.PRICE_GE)
        with (patch.object(self.engine, "_validate_lot_inventory", return_value=self.lot),
              patch.object(self.engine, "_prototype_source_for", return_value=STRATEGY)):
            self.service.prices = [Decimal("103")]
            request, _, _ = self.engine._preflight(self.item, rule, self.snapshot("103"))
            self.assertEqual((request.side, request.price), (OrderSide.SELL, Decimal("103")))
            self.service.prices = [Decimal("102")]
            with self.assertRaises(ValueError):
                self.engine._preflight(self.item, rule, self.snapshot("103"))
        self.assertEqual(self.service.submitted, [])

    def test_missing_both_targets_still_blocks_price_exit(self):
        rule = self.rule(TriggerKind.PRICE_GE)
        with (patch.object(self.engine, "_validate_lot_inventory",
                           return_value={**self.lot, "take_profit_price": None}),
              patch.object(self.engine, "_prototype_source_for", return_value=STRATEGY)):
            self.service.prices = [Decimal("103")]
            with self.assertRaisesRegex(ValueError, "목표가격"):
                self.engine._preflight(self.item, rule, self.snapshot("103"))

    def test_tenth_session_preclose_is_independent_of_price_and_rechecked(self):
        with patch.dict(MODEL_EXIT_SCHEDULES, {STRATEGY: ModelExitSchedule(9, "preclose")}):
            planned = planned_model_exit(self.lot, Market.DOMESTIC)
            due_session = session_on(Market.DOMESTIC, planned.day)
            self.assertEqual(planned.timing, "preclose")
            # The Korean exchange is closed for Chuseok on September 24–25.
            self.assertEqual(planned.day, date(2026, 9, 29))
            self.now = due_session.closed - timedelta(minutes=6)
            self.assertFalse(timed_exit_due(self.lot, Market.DOMESTIC, self.now))
            rule = self.rule(TriggerKind.TIME_EXIT)
            with patch.object(self.engine, "_validate_lot_inventory", return_value=self.lot):
                self.service.prices = [Decimal("80")]
                with self.assertRaisesRegex(ValueError, "보유기간"):
                    self.engine._preflight(self.item, rule, self.snapshot("80"))
                self.now = due_session.closed - timedelta(minutes=4)
                self.assertTrue(timed_exit_due(self.lot, Market.DOMESTIC, self.now))
                request, _, _ = self.engine._preflight(self.item, rule, self.snapshot("80"))
            self.assertEqual((request.side, request.price), (OrderSide.SELL, Decimal("80")))
            self.assertEqual(self.service.submitted, [])

    def test_tenth_session_uses_us_calendar_and_local_close(self):
        market = Market.US
        fill = session_on(market, date(2026, 9, 4)).opened + timedelta(minutes=20)
        lot = {**self.lot, "buy_fill_observed_at": fill.isoformat()}
        with patch.dict(MODEL_EXIT_SCHEDULES, {STRATEGY: ModelExitSchedule(9, "preclose")}):
            planned = planned_model_exit(lot, market)
            # Labor Day on September 7 is not counted as a holding session.
            self.assertEqual(planned.day, date(2026, 9, 18))
            due_session = session_on(market, planned.day)
            self.assertFalse(timed_exit_due(lot, market, due_session.closed - timedelta(minutes=6)))
            self.assertTrue(timed_exit_due(lot, market, due_session.closed - timedelta(minutes=4)))

    def test_lot_pass_prefers_target_before_deadline_then_time_exit(self):
        targets = {"reconciled": True, "lots": (self.lot,)}
        with (patch.dict(MODEL_EXIT_SCHEDULES, {STRATEGY: ModelExitSchedule(9, "preclose")}),
              patch.object(AutoTrader, "orders_enabled", new_callable=PropertyMock, return_value=True),
              patch.object(self.store, "save_holding_rule"),
              patch.object(self.store, "reserve_prototype_sell"),
              patch.object(self.store, "pause_rule"),
              patch.object(self.engine, "_message"),
              patch.object(self.engine, "_execute", return_value=True) as execute):
            self.engine._lot_holdings_exits(self.item, self.service.positions[0],
                                            self.snapshot("99"), targets, set())
            execute.assert_not_called()
            self.engine._lot_holdings_exits(self.item, self.service.positions[0],
                                            self.snapshot("103"), targets, set())
            self.assertIs(execute.call_args.args[1].kind, TriggerKind.PRICE_GE)
            execute.reset_mock()
            planned = planned_model_exit(self.lot, Market.DOMESTIC)
            self.now = session_on(Market.DOMESTIC, planned.day).closed - timedelta(minutes=4)
            self.engine._lot_holdings_exits(self.item, self.service.positions[0],
                                            self.snapshot("80"), targets, set())
            self.assertIs(execute.call_args.args[1].kind, TriggerKind.TIME_EXIT)
        self.assertEqual(self.service.submitted, [])


if __name__ == "__main__":
    unittest.main()
