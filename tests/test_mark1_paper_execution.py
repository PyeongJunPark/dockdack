"""Normal AutoTrader integration: research checks do not grant order authority."""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from dockdack.autotrade import AutoTrader
from dockdack.exceptions import OrderNotSent
from dockdack.http import _ORDER_SEND_GUARD
from dockdack.models import OrderSide, TradingMode
from dockdack.signal_bridge import ExternalPolicy, export_charts, ingest_signals
from dockdack.watchlist import TriggerKind, TriggerRule, WatchItem, WatchStore
from test_autotrade import FakeTradingService, NOW, position
from test_trading_environment import RealFakeService


SOURCE = "mark1-prototype-demo-trigger"


class PacedFakeService(FakeTradingService):
    def __init__(self):
        super().__init__()
        self.before_guard = lambda: None

    def submit(self, request):
        self.before_guard()
        guard = _ORDER_SEND_GUARD.get()
        if guard is not None:
            guard()
        return super().submit(request)


class Mark1PaperExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.store = WatchStore(self.path / "paper.sqlite3")
        self.service = PacedFakeService()
        self.item = WatchItem(self.service.resolve("005930"), "삼성전자", 31)
        self.store.save_item(self.item)
        self.now = NOW
        self.engine = AutoTrader(self.service, self.store, clock=lambda: self.now)
        self.engine.snapshot(self.item)
        self.chart = export_charts(self.store, self.path / "charts.json", now=NOW)
        self.policy = ExternalPolicy(SOURCE, 1, Decimal("500"), Decimal("1000"))
        self.engine.external_only = True
        self.engine.external_policy = self.policy
        self.calls = []

    def validator(self, item, rule, fresh, actual_limit_price, *, stage):
        self.calls.append((stage, fresh.quote.price, actual_limit_price))

    def register(self, validator=None):
        self.engine.configure_source_validators({SOURCE: validator or self.validator})

    def ingest(self, **changes):
        entry = {"signal_id": "mark1-prototype:paper-decision", "export_id": self.chart["export_id"],
                 "market": "domestic", "symbol": "005930", "exchange": "KRX", "action": "buy",
                 "quantity": 1, "max_notional": "500", "generated_at": NOW.isoformat(),
                 "expires_at": (NOW + timedelta(minutes=2)).isoformat()}
        entry.update(changes)
        ingest_signals(self.store, {"schema_version": 1, "source_id": SOURCE, "trading_mode": "demo",
                                   "signals": [entry]}, self.policy, now=NOW)
        return self.store.rules(statuses=("ready",))[0]

    def test_connection_does_not_poll_or_arm_and_wrong_confirmation_cannot_arm(self):
        calls = self.service.quote_calls, self.service.history_calls
        self.register()
        self.ingest()
        self.assertFalse(self.engine.orders_enabled)
        self.assertEqual((self.service.quote_calls, self.service.history_calls), calls)
        with self.assertRaises(ValueError):
            self.engine.enable_orders("REAL_AUTOTRADE")
        self.assertEqual(self.service.submitted, [])

    def test_known_source_requires_local_validator_even_before_first_signal(self):
        with self.assertRaisesRegex(ValueError, "모델 주문 검증"):
            self.engine.enable_orders("DEMO_AUTOTRADE")
        self.assertFalse(self.engine.orders_enabled)

    def test_local_registration_cannot_change_while_orders_are_on(self):
        self.register()
        self.engine.enable_orders("DEMO_AUTOTRADE")
        with self.assertRaises(ValueError):
            self.engine.configure_source_validators({})
        self.assertEqual(self.calls, [])

    def test_demo_explicit_arm_rechecks_fresh_quote_and_rounded_limit_twice(self):
        self.register()
        self.ingest()
        self.service.prices = [Decimal("100.75")]
        self.engine.enable_orders("DEMO_AUTOTRADE")
        self.engine.poll()
        self.assertEqual(self.calls, [("preflight", Decimal("100.75"), Decimal("100")),
                                     ("final_send", Decimal("100.75"), Decimal("100"))])
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(self.service.submitted[0].quantity, 1)
        self.assertEqual(self.service.submitted[0].price, Decimal("100"))
        saved = self.store.exit_targets(self.item.id)
        self.assertEqual(saved["source"], SOURCE)
        self.assertEqual(saved["take_profit_price"], Decimal("101"))
        self.assertEqual(saved["stop_loss_price"], Decimal("99.1"))

    def test_rejecting_model_at_preflight_sends_nothing(self):
        def reject(*args, **kwargs):
            raise ValueError("모델 재추론 확률이 50% 이하")
        self.register(reject)
        self.ingest()
        self.engine.enable_orders("DEMO_AUTOTRADE")
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.attempts(), ())

    def test_rejecting_model_after_http_pacer_is_definitely_not_sent(self):
        def reject_final(*args, stage, **kwargs):
            if stage == "final_send":
                raise ValueError("검증 연결 변경")
        self.register(reject_final)
        self.ingest()
        self.engine.enable_orders("DEMO_AUTOTRADE")
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.attempts()[0]["status"], "not_sent")

    def test_deleting_validator_during_pacing_cannot_bypass_final_guard(self):
        self.register()
        self.ingest()
        self.service.before_guard = self.engine.source_validators.clear
        self.engine.enable_orders("DEMO_AUTOTRADE")
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.attempts()[0]["status"], "not_sent")

    def test_model_check_duration_counts_toward_final_quote_freshness(self):
        def delayed(*args, stage, **kwargs):
            if stage == "final_send":
                self.now += timedelta(seconds=16)
        self.register(delayed)
        self.ingest()
        self.engine.enable_orders("DEMO_AUTOTRADE")
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.attempts()[0]["status"], "not_sent")

    def test_cash_quantity_and_holding_guards_remain_in_force(self):
        self.register()
        self.ingest()
        self.service.positions = (position(),)
        self.engine.enable_orders("DEMO_AUTOTRADE")
        self.engine.poll()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.service.submitted, [])
        self.service.positions = ()
        self.service.available = Decimal("1")
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])

    def test_restart_does_not_inherit_permission_or_executable_hook(self):
        self.register()
        self.ingest()
        restarted = AutoTrader(self.service, WatchStore(self.store.path), clock=lambda: NOW)
        restarted.external_only = True
        restarted.external_policy = self.policy
        self.assertFalse(restarted.orders_enabled)
        with self.assertRaisesRegex(ValueError, "모델 주문 검증"):
            restarted.enable_orders("DEMO_AUTOTRADE")
        self.assertEqual(self.service.submitted, [])

    def test_mark1_owned_exit_uses_actual_average_not_candidate_quote(self):
        self.register()
        self.ingest()
        self.engine.enable_orders("DEMO_AUTOTRADE")
        self.engine.poll()
        held = replace(position(), average_price=Decimal("110"))
        targets = self.engine.holding_exit_targets(held)
        self.assertEqual(targets["take_profit_price"], Decimal("111.10"))
        self.assertEqual(targets["stop_loss_price"], Decimal("109.010"))
        self.assertIn("mark1", targets["source"])
        invalid = self.engine.holding_exit_targets(replace(held, average_price=Decimal("NaN")))
        self.assertIsNone(invalid["take_profit_price"])
        self.assertIsNone(invalid["stop_loss_price"])

    def test_unrelated_holdings_keep_existing_point_eight_percent_default(self):
        held = replace(position(), average_price=Decimal("110"))
        targets = self.engine.holding_exit_targets(held)
        self.assertEqual(targets["take_profit_price"], Decimal("111.10"))
        self.assertEqual(targets["stop_loss_price"], Decimal("109.120"))
        self.store.set_exit_targets(self.item.id, Decimal("120"), Decimal("90"), source="other-model",
                                    rule_id="other-buy", now=NOW)
        saved = self.engine.holding_exit_targets(held)
        self.assertEqual(saved["take_profit_price"], Decimal("120"))
        self.assertEqual(saved["stop_loss_price"], Decimal("90"))

    def test_mark1_ownership_without_original_buy_record_fails_closed(self):
        self.store.set_exit_targets(self.item.id, Decimal("101"), Decimal("99.1"), source=SOURCE,
                                    rule_id="missing-buy", now=NOW)
        with self.assertRaisesRegex(ValueError, "원본 매수"):
            self.engine.holding_exit_targets(position())

    def test_renamed_exit_marker_still_uses_original_mark1_buy_origin(self):
        self.register()
        self.ingest()
        self.engine.enable_orders("DEMO_AUTOTRADE")
        self.engine.poll()
        with self.store.connection() as db:
            db.execute("UPDATE position_exit_targets SET source='renamed-model'")
        held = replace(position(), average_price=Decimal("110"))
        targets = self.engine.holding_exit_targets(held)
        self.assertEqual(targets["take_profit_price"], Decimal("111.10"))
        self.assertEqual(targets["stop_loss_price"], Decimal("109.010"))

    def test_unfilled_buy_ownership_survives_empty_holdings_pass_and_restart(self):
        self.register()
        self.ingest()
        self.engine.enable_orders("DEMO_AUTOTRADE")
        self.engine.poll()
        self.engine.enable_holdings_exits = True
        self.engine.poll()
        restarted = AutoTrader(self.service, WatchStore(self.store.path), clock=lambda: NOW)
        targets = restarted.holding_exit_targets(replace(position(), average_price=Decimal("110")))
        self.assertEqual(targets["take_profit_price"], Decimal("111.10"))
        self.assertEqual(targets["stop_loss_price"], Decimal("109.010"))
        self.assertFalse(restarted.orders_enabled)


class Mark1RealBlockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.service = RealFakeService()
        self.store = WatchStore(self.path / "real.sqlite3", mode=TradingMode.REAL,
                                storage_scope=self.service.storage_scope)
        self.item = WatchItem(self.service.resolve("005930"), days=31)
        self.store.save_item(self.item)
        self.engine = AutoTrader(self.service, self.store, clock=lambda: NOW)

    def test_aliases_cannot_be_added_as_primary_extra_or_validator_in_real(self):
        for source in (SOURCE, "mark1-prototype-daily-barrier", "MARK1_PROTOTYPE"):
            with self.subTest(source=source):
                policy = ExternalPolicy(source, 1, Decimal("1000"), Decimal("1000"))
                with self.assertRaises(ValueError):
                    self.engine.configure_external_sources([(policy, lambda: None)])
                with self.assertRaises(ValueError):
                    self.engine.configure_source_validators({source: lambda *args, **kwargs: None})
                self.engine.external_only = True
                self.engine.external_policy = policy
                with self.assertRaises(ValueError):
                    self.engine.enable_orders("REAL_AUTOTRADE")
        self.assertEqual(self.service.submitted, [])

    def test_direct_extra_feed_assignment_cannot_bypass_real_arm(self):
        policy = ExternalPolicy(SOURCE, 1, Decimal("1000"), Decimal("1000"))
        self.engine.external_sources[SOURCE] = (policy, lambda: None)
        self.engine.external_only = True
        with self.assertRaises(ValueError):
            self.engine.enable_orders("REAL_AUTOTRADE")
        self.assertFalse(self.engine.orders_enabled)

    def test_persisted_renamed_strategy_origin_cannot_arm_real(self):
        self.engine.snapshot(self.item)
        chart = export_charts(self.store, self.path / "charts.json", now=NOW)
        policy = ExternalPolicy("renamed-model", 1, Decimal("1000"), Decimal("1000"))
        entry = {"signal_id": "normal-entry", "export_id": chart["export_id"], "market": "domestic",
                 "symbol": "005930", "exchange": "KRX", "action": "buy", "quantity": 1,
                 "max_notional": "1000", "generated_at": NOW.isoformat(),
                 "expires_at": (NOW + timedelta(minutes=2)).isoformat()}
        ingest_signals(self.store, {"schema_version": 1, "trading_mode": "real", "source_id": policy.source_id,
                                   "signals": [entry]}, policy, now=NOW)
        # Model provenance must survive a renamed source and persisted reload.
        for provenance in ({"origin_strategy": "mark1-prototype"},
                           {"strategy_id": "mark1-prototype"},
                           {"signal_id": "mark1-prototype:manifest:decision"}):
            with self.subTest(provenance=provenance):
                with self.store.connection() as db:
                    db.execute("UPDATE external_signals SET payload=?", (json.dumps({**entry, **provenance,
                                                                                   "trading_mode": "real"}),))
                self.engine.external_only = True
                self.engine.external_policy = policy
                with self.assertRaisesRegex(ValueError, "이름을 바꿔도"):
                    self.engine.enable_orders("REAL_AUTOTRADE")
        self.assertEqual(self.service.submitted, [])

    def test_real_final_send_rechecks_renamed_origin_after_manual_confirmation(self):
        fresh = self.engine.snapshot(self.item)
        chart = export_charts(self.store, self.path / "charts.json", now=NOW)
        policy = ExternalPolicy("ordinary-model", 1, Decimal("1000"), Decimal("1000"))
        entry = {"signal_id": "ordinary-decision", "export_id": chart["export_id"], "market": "domestic",
                 "symbol": "005930", "exchange": "KRX", "action": "buy", "quantity": 1,
                 "max_notional": "1000", "generated_at": NOW.isoformat(),
                 "expires_at": (NOW + timedelta(minutes=2)).isoformat()}
        ingest_signals(self.store, {"schema_version": 1, "trading_mode": "real", "source_id": policy.source_id,
                                   "signals": [entry]}, policy, now=NOW)
        self.engine.external_only, self.engine.external_policy = True, policy
        self.engine.enable_orders("REAL_AUTOTRADE")
        rule = self.store.rules(statuses=("ready",))[0]
        self.store.claim(rule, fresh.quote.price, NOW)
        with self.store.connection() as db:
            db.execute("UPDATE external_signals SET payload=?", (json.dumps({**entry, "trading_mode": "real",
                       "signal_id": "mark1-prototype:manifest:renamed"}),))
        with self.assertRaises(OrderNotSent):
            self.engine._before_order_send(self.item, rule, fresh)
        self.assertFalse(self.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_mark1_holding_marker_cannot_be_reused_by_generic_real_exit(self):
        self.store.set_exit_targets(self.item.id, Decimal("101"), Decimal("99.1"), source=SOURCE,
                                    rule_id="paper-buy", now=NOW)
        self.engine.enable_holdings_exits = True
        with self.assertRaisesRegex(ValueError, "모의|실전"):
            self.engine.holding_exit_targets(position())
        self.assertEqual(self.service.submitted, [])

    def test_real_holding_rule_is_blocked_again_at_final_paced_send(self):
        self.store.set_exit_targets(self.item.id, Decimal("101"), Decimal("99.1"), source=SOURCE,
                                    rule_id="paper-buy", now=NOW)
        rule = TriggerRule("holding-exit-forged-paper", self.item.id, TriggerKind.PRICE_GE,
                           OrderSide.SELL, 1, Decimal("1000"), Decimal("101"))
        self.store.save_holding_rule(self.item, rule)
        fresh = self.engine.snapshot(self.item)
        self.engine._armed.set()  # Deliberately bypass UI/manual confirmation in this guard test.
        self.store.claim(rule, fresh.quote.price, NOW)
        with self.assertRaisesRegex(OrderNotSent, "모의 환경"):
            self.engine._before_order_send(self.item, rule, fresh)
        self.assertFalse(self.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])


if __name__ == "__main__":
    unittest.main()
