"""Offline contracts for independent pre-open model feeds and model-owned exits."""
from __future__ import annotations

import json
import unittest
from datetime import timedelta
from decimal import Decimal
from unittest.mock import Mock

from dockdack.autotrade import AutoTrader
from dockdack.market_schedule import session_on
from dockdack.models import Market, OrderSide
from dockdack.signal_bridge import ExternalPolicy, prototype_family
from dockdack.signals.preopen_series import PREOPEN_MODELS
from dockdack.trading.model_exit_schedule import timed_exit_due
from dockdack.watchlist import MarketSnapshot, TriggerKind, TriggerRule
from test_autotrade import FakeTradingService, position
import test_strategy_lots as lots_tests
import test_v00_execution as v00_tests


LOT_NOW = lots_tests.NOW


MARK18_SOURCE = "mark1-8-prototype-demo-trigger"


class PreopenIdentityTests(unittest.TestCase):
    def test_each_method_has_distinct_frozen_bundle_source_and_owned_title(self):
        self.assertEqual(set(PREOPEN_MODELS), {
            "mark1-3-prototype", *(f"mark1-{number}-prototype" for number in range(5, 13))
        })
        self.assertEqual(len({spec.source_id for spec in PREOPEN_MODELS.values()}), 9)
        self.assertEqual(len({spec.model_id for spec in PREOPEN_MODELS.values()}), 9)
        for model_id, spec in PREOPEN_MODELS.items():
            with self.subTest(model_id=model_id):
                self.assertEqual(spec.source_id, model_id + "-demo-trigger")
                self.assertIsNotNone(prototype_family(spec.source_id, {
                    "signal_id": model_id + ":offline", "strategy_id": model_id,
                }))
                self.assertIn("완료 30일봉", spec.strategy_notice)
                self.assertIn("모의 전용", spec.risk_notice)


class Mark18AllocationTests(unittest.TestCase):
    """A fake DEMO account exercises the real ingestion and preflight paths."""

    setUp = v00_tests.V00ExecutionTests.setUp
    arm = v00_tests.V00ExecutionTests.arm
    external = v00_tests.V00ExecutionTests.external

    def connect(self, fraction="0.025"):
        self.engine.equity_buy_percent = Decimal("10")
        self.engine.prototype_lots_enabled = True
        fields = {
            "signal_id": "mark1-8-prototype:offline-allocation",
            "strategy_id": "mark1-8-prototype", "model_title": "mark1.8 prototype",
            "model_version": "test-fixture", "model_manifest_sha256": "a" * 64,
        }
        if fraction is not None:
            fields["target_equity_fraction"] = fraction
        self.external(source=MARK18_SOURCE, **fields)
        self.engine.configure_source_validators({MARK18_SOURCE: lambda *args, **kwargs: None})

    def test_learned_fraction_sizes_buy_below_global_ten_percent(self):
        self.connect("0.025")
        self.arm()
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(self.service.submitted[0].side, OrderSide.BUY)
        # Fake same-market equity is 40,000 cash + 60,000 holdings evaluation.
        # A 2.5% model output at 100/share is 25 shares, not the GUI's 10%.
        self.assertEqual(self.service.submitted[0].quantity, 25)

    def test_missing_learned_fraction_cannot_fall_back_to_global_sizing(self):
        self.connect(None)
        self.arm()
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.attempts(), ())

    def test_mutated_or_invalid_fraction_fails_closed(self):
        for value in ("not-a-number", "0.25", "0", "NaN"):
            with self.subTest(value=value):
                self.connect("0.025")
                with self.store.connection() as db:
                    row = db.execute("SELECT source_id, signal_id, payload FROM external_signals").fetchone()
                    payload = json.loads(row[2])
                    payload["target_equity_fraction"] = value
                    db.execute("UPDATE external_signals SET payload=? WHERE source_id=? AND signal_id=?",
                               (json.dumps(payload), row[0], row[1]))
                self.arm()
                self.engine.poll()
                self.assertEqual(self.service.submitted, [])
                self.engine.disarm()
                with self.store.connection() as db:
                    db.execute("DELETE FROM external_signals")
                    db.execute("DELETE FROM rules WHERE kind='external'")


class HorizonExitTests(unittest.TestCase):
    """No future OHLC, price target, or unrelated account lot authorizes a timed exit."""

    setUp = lots_tests.StrategyLotTests.setUp
    tearDown = lots_tests.StrategyLotTests.tearDown
    rule = lots_tests.StrategyLotTests.rule
    accept = lots_tests.StrategyLotTests.accept
    fill = lots_tests.StrategyLotTests.fill
    buy = lots_tests.StrategyLotTests.buy

    @staticmethod
    def session_after(market, first, count):
        result = first
        found = 0
        while found < count:
            result += timedelta(days=1)
            if session_on(market, result) is not None:
                found += 1
        return session_on(market, result)

    def test_three_and_five_session_horizons_ignore_price_once_due(self):
        # The fixed temporary-ledger fill is observed at LOT_NOW + one minute.
        first = LOT_NOW.date()
        while session_on(Market.DOMESTIC, first) is None:
            first += timedelta(days=1)
        for model_id, later_sessions in (("mark1-11-prototype", 2),
                                         ("mark1-12-prototype", 4)):
            with self.subTest(model=model_id):
                lot = {"strategy_id": model_id,
                       "buy_fill_observed_at": (LOT_NOW + timedelta(minutes=1)).isoformat()}
                previous = self.session_after(Market.DOMESTIC, first, later_sessions - 1)
                due = self.session_after(Market.DOMESTIC, first, later_sessions)
                self.assertFalse(timed_exit_due(lot, Market.DOMESTIC,
                                               previous.opened + timedelta(minutes=1)))
                self.assertTrue(timed_exit_due(lot, Market.DOMESTIC,
                                              due.opened + timedelta(minutes=1)))
                self.assertFalse(timed_exit_due({**lot, "buy_fill_observed_at": None},
                                                Market.DOMESTIC, due.opened + timedelta(minutes=1)))
                self.assertFalse(timed_exit_due({**lot, "strategy_id": "manual"},
                                                Market.DOMESTIC, due.opened + timedelta(minutes=1)))

    def test_repeated_cumulative_fill_does_not_restart_holding_period(self):
        buy = self.rule("mark1-11-prototype", quantity=3)
        self.accept(buy)
        t0, t1, t2, t3, t4 = (LOT_NOW + timedelta(minutes=minute)
                             for minute in (1, 2, 3, 4, 5))

        def record(filled, remaining, price, observed):
            self.store.record_execution(
                buy.id, filled_quantity=Decimal(filled), remaining_quantity=Decimal(remaining),
                fill_price=Decimal(price) if price is not None else None, observed_at=observed,
            )
            return self.store.prototype_lots(self.item.id)[0]["buy_fill_observed_at"]

        self.assertIsNone(record(0, 3, None, t0))
        self.assertEqual(record(1, 2, "100", t1), t1.isoformat())
        self.assertEqual(record(1, 2, "100", t2), t1.isoformat())
        self.assertEqual(record(2, 1, "100", t3), t3.isoformat())
        self.assertEqual(record(2, 1, None, t4), t3.isoformat())

    def test_time_exit_null_threshold_survives_sqlite_round_trip_and_claim(self):
        model_buy = self.buy("mark1-11-prototype", average="100")
        rule = TriggerRule("holding-exit-time-offline", self.item.id, TriggerKind.TIME_EXIT,
                           OrderSide.SELL, 1, Decimal("1000"), None)
        self.store.save_holding_rule(self.item, rule)
        self.store.reserve_prototype_sell(rule.id, model_buy.id, 1)
        persisted = next(row for row in self.store.rules(include_inactive=True)
                         if row.id == rule.id)
        self.assertIsNone(persisted.threshold)
        self.assertEqual(persisted, rule)
        self.assertTrue(self.store.claim(persisted, Decimal("100"),
                                         LOT_NOW + timedelta(minutes=2), prototype_lots=True))

    def test_due_horizon_generates_sell_for_only_confirmed_model_lot(self):
        first = LOT_NOW.date()
        while session_on(Market.DOMESTIC, first) is None:
            first += timedelta(days=1)
        due = self.session_after(Market.DOMESTIC, first, 2)
        self.now = due.opened + timedelta(minutes=1)
        model_buy = self.buy("mark1-11-prototype", average="100")
        other_buy = self.buy("mark1-prototype", average="100")
        service = FakeTradingService()
        service.positions = (position(2, 2),)
        engine = AutoTrader(service, self.store, clock=lambda: self.now)
        engine.prototype_lots_enabled = True
        engine.enable_holdings_exits = True
        engine.holding_caps[Market.DOMESTIC] = Decimal("100000")
        engine.enable_orders("DEMO_AUTOTRADE")
        targets = engine.holding_exit_targets(service.positions[0])
        self.assertTrue(targets["reconciled"])
        self.assertEqual({row["lot_id"] for row in targets["lots"]}, {model_buy.id, other_buy.id})
        quote = service.quote(self.item.instrument)
        snapshot = MarketSnapshot(quote, service.history(self.item.instrument, 31), self.now)
        observed = []

        def capture(item, rule, current):
            persisted = next(row for row in self.store.rules(include_inactive=True)
                             if row.id == rule.id)
            self.assertIsNone(persisted.threshold)
            self.assertTrue(self.store.claim(persisted, current.quote.price,
                                             self.now, prototype_lots=True))
            observed.append(rule)
            return True

        engine._execute = Mock(side_effect=capture)
        engine._lot_holdings_exits(self.item, service.positions[0], snapshot, targets, set())
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0].side, OrderSide.SELL)
        self.assertEqual(observed[0].kind.value, "time_exit")
        self.assertIsNone(observed[0].threshold)
        allocation = self.store.prototype_sell_allocation(observed[0].id)
        self.assertEqual(allocation["lot_id"], model_buy.id)
        self.assertEqual(service.submitted, [])

    def test_due_horizon_passes_full_preflight_and_sends_only_fake_demo_order(self):
        first = LOT_NOW.date()
        while session_on(Market.DOMESTIC, first) is None:
            first += timedelta(days=1)
        due = self.session_after(Market.DOMESTIC, first, 2)
        now = due.opened + timedelta(minutes=1)
        model_buy = self.buy("mark1-11-prototype", average="100")
        service = FakeTradingService()
        service.positions = (position(1, 1),)
        service.prices = [Decimal("100")]
        engine = AutoTrader(service, self.store, clock=lambda: now)
        engine.prototype_lots_enabled = True
        engine.enable_holdings_exits = True
        engine.holding_caps[Market.DOMESTIC] = Decimal("100000")
        engine.enable_orders("DEMO_AUTOTRADE")
        targets = engine.holding_exit_targets(service.positions[0])
        self.assertTrue(targets["reconciled"])
        self.assertIsNone(targets["lots"][0]["take_profit_price"])
        self.assertIsNone(targets["lots"][0]["stop_loss_price"])
        quote = service.quote(self.item.instrument)
        snapshot = MarketSnapshot(quote, service.history(self.item.instrument, 31), now)

        # This calls the real preflight, final-send guard and ledger claim; only
        # FakeTradingService.submit is reachable, never Kiwoom or a GUI worker.
        engine._lot_holdings_exits(self.item, service.positions[0], snapshot, targets, set())
        self.assertEqual(len(service.submitted), 1)
        self.assertEqual(service.submitted[0].side, OrderSide.SELL)
        self.assertEqual(service.submitted[0].quantity, 1)
        sale = next(row for row in self.store.order_history(limit=None) if row["side"] == "sell")
        self.assertEqual(sale["status"], "accepted")
        self.assertEqual(self.store.prototype_sell_allocation(sale["rule_id"])["lot_id"], model_buy.id)
