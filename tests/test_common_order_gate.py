"""Legacy watchlist/rules must not bypass current common-company eligibility."""

from datetime import timedelta
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from dockdack.autotrade import AutoTrader
from dockdack.exceptions import BrokerAPIError
from dockdack.gui_service import TradingService
from dockdack.signal_bridge import ExternalPolicy, export_charts, ingest_signals
from dockdack.watchlist import TriggerRule, WatchItem, WatchStore
from test_autotrade import FakeTradingService, NOW, position


class CommonOrderGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.counter = 0

    def case(self, *, external=False, side="buy"):
        self.counter += 1
        path = Path(self.temp.name) / f"case-{self.counter}.sqlite3"
        store, service = WatchStore(path), FakeTradingService()
        item = WatchItem(service.resolve("005930"), "기존 관심종목")
        store.save_item(item)
        if side == "sell":
            service.positions = (position(),)
        engine = AutoTrader(service, store, clock=lambda: NOW)
        if external:
            engine.snapshot(item)
            chart = export_charts(store, Path(self.temp.name) / f"chart-{self.counter}.json", now=NOW)
            policy = ExternalPolicy("test-model", 1, Decimal(1000), Decimal(1000))
            payload = dict(schema_version=1, source_id="test-model", signals=[dict(
                signal_id="legacy-decision", export_id=chart["export_id"], market="domestic", symbol="005930",
                exchange="KRX", action=side, quantity=1, max_notional="1000",
                generated_at=NOW.isoformat(), expires_at=(NOW + timedelta(minutes=2)).isoformat())])
            ingest_signals(store, payload, policy, now=NOW)
            engine.external_policy, engine.external_only = policy, True
        else:
            store.add_rule(TriggerRule.create(item, "price_ge", side, 1, Decimal(1000), Decimal(95)))
        return store, service, item, engine

    def test_existing_manual_buy_and_sell_rules_cannot_submit_disallowed_products(self):
        for side in ("buy", "sell"):
            with self.subTest(side=side):
                store, service, item, engine = self.case(side=side)
                service.ensure_common_equity = Mock(side_effect=ValueError("not an ordinary company share"))
                engine.enable_orders("DEMO_AUTOTRADE")
                engine.poll()
                engine.poll()
                self.assertEqual(service.submitted, [])
                self.assertEqual(store.attempts(), ())
                self.assertEqual(store.rules()[0].status, "ready")
                self.assertEqual(service.ensure_common_equity.call_count, 2)
                service.ensure_common_equity.assert_called_with(item.instrument)

    def test_existing_external_buy_and_sell_rules_cannot_submit_disallowed_products(self):
        for side in ("buy", "sell"):
            with self.subTest(side=side):
                store, service, item, engine = self.case(external=True, side=side)
                service.ensure_common_equity = Mock(side_effect=ValueError("ETF/REIT/SPAC blocked"))
                engine.enable_orders("DEMO_AUTOTRADE")
                engine.poll()
                self.assertEqual(service.submitted, [])
                self.assertEqual(store.attempts(), ())
                service.ensure_common_equity.assert_called_once_with(item.instrument)

    def test_classification_api_or_schema_failure_never_prepares_claims_or_submits(self):
        for external in (False, True):
            for error in (BrokerAPIError("source incomplete"), TimeoutError("source timeout"), ValueError("unknown classification")):
                with self.subTest(external=external, error=type(error).__name__):
                    store, service, item, engine = self.case(external=external)
                    service.ensure_common_equity = Mock(side_effect=error)
                    service.prepare = Mock(wraps=service.prepare)
                    service.safety_account = Mock(wraps=service.safety_account)
                    engine.enable_orders("DEMO_AUTOTRADE")
                    engine.poll()
                    self.assertEqual(service.submitted, [])
                    self.assertEqual(store.attempts(), ())
                    service.prepare.assert_not_called()
                    service.safety_account.assert_not_called()

    def test_confirmed_common_share_still_executes_manual_and_external_orders(self):
        for external in (False, True):
            with self.subTest(external=external):
                store, service, item, engine = self.case(external=external)
                service.ensure_common_equity = Mock()
                engine.enable_orders("DEMO_AUTOTRADE")
                engine.poll()
                self.assertEqual(len(service.submitted), 1)
                self.assertEqual(store.attempts()[0]["status"], "accepted")
                service.ensure_common_equity.assert_called_once_with(item.instrument)

    def test_restarted_ready_rule_is_reclassified_not_trusted_from_previous_session(self):
        store, service, item, engine = self.case()
        service.ensure_common_equity = Mock(side_effect=ValueError("classification changed"))
        restarted = AutoTrader(service, WatchStore(store.path), clock=lambda: NOW)
        restarted.enable_orders("DEMO_AUTOTRADE")
        restarted.poll()
        self.assertEqual(service.submitted, [])
        self.assertEqual(store.attempts(), ())
        service.ensure_common_equity.assert_called_once_with(item.instrument)

    def test_stopping_during_classification_prevents_submission(self):
        store, service, item, engine = self.case()
        service.ensure_common_equity = Mock(side_effect=lambda instrument: engine.stop())
        engine.enable_orders("DEMO_AUTOTRADE")
        engine.poll()
        self.assertFalse(engine.orders_enabled)
        self.assertEqual(service.submitted, [])
        self.assertEqual(store.attempts(), ())

    def test_real_service_gate_demands_exact_current_membership_before_auto_submit(self):
        for membership in (frozenset(), frozenset({("005930", "ND")}), frozenset({("069500", "KRX")})):
            with self.subTest(membership=membership):
                store, service, item, engine = self.case()
                service.common_equities = Mock(return_value=membership)
                service.ensure_common_equity = lambda instrument: TradingService.ensure_common_equity(service, instrument)
                engine.enable_orders("DEMO_AUTOTRADE")
                engine.poll()
                self.assertEqual(service.submitted, [])
                self.assertEqual(store.attempts(), ())
                service.common_equities.assert_called_once_with(item.instrument.market, (("005930", "KRX"),))


if __name__ == "__main__":
    unittest.main()
