"""Offline environment isolation: all broker traffic uses an in-memory transport."""
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
from uuid import NAMESPACE_URL, uuid5

from dockdack import KiwoomBroker, Market, OrderResult, TradingMode
from dockdack.autotrade import AutoTrader
from dockdack.exceptions import ConfigurationError, OrderNotSent, OrderOutcomeUnknown
from dockdack.gui_service import LIVE_RISK_ACKNOWLEDGEMENT, TradingService
from dockdack.http import _ORDER_SEND_GUARD, order_send_guard
from dockdack.signal_bridge import ExternalPolicy, atomic_json, export_charts, ingest_signals
from dockdack.signal_status import inspect_signal_file
from dockdack.test_strategy import RandomDemoSignals
from dockdack.watchlist import TriggerRule, WatchItem, WatchStore
from test_autotrade import FakeTradingService, NOW
from test_kiwoom import FakeResponse, QueueTransport, config, token_response


class TradingEnvironmentServiceTests(unittest.TestCase):
    def service(self, *, mode=TradingMode.REAL, allow=True, transport=None):
        transport = transport or QueueTransport(token_response(), FakeResponse({"return_code": 0, "ord_no": "001"}))
        broker = KiwoomBroker(config(mode, allow_live_orders=allow), transport=transport)
        service = TradingService(lambda _: broker, mode=mode)
        return service, broker, transport

    @staticmethod
    def prepare(service):
        return service.prepare(service.resolve("AAPL", "ND"), "buy", 1, "limit", Decimal("100"))

    def test_default_remains_demo_and_mode_is_read_only(self):
        service = TradingService()
        self.assertIs(service.mode, TradingMode.DEMO)
        with self.assertRaises(AttributeError):
            service.mode = TradingMode.REAL
        with self.assertRaises(ValueError):
            service.acknowledge_live_risk(LIVE_RISK_ACKNOWLEDGEMENT)

    def test_real_ui_can_be_constructed_without_keys_and_does_not_fallback(self):
        with patch("dockdack.gui_service.KiwoomConfig.from_env", side_effect=ConfigurationError("키 필요")) as load:
            service = TradingService(mode=TradingMode.REAL)
        self.assertEqual(service.storage_scope, "unconfigured")
        self.assertFalse(service.live_risk_acknowledged)
        self.assertEqual(load.call_count, 2)
        self.assertTrue(all(call.args == (TradingMode.REAL,) for call in load.call_args_list))
        with self.assertRaises(ConfigurationError):
            service.broker(Market.US)

    def test_real_configuration_is_frozen_and_scope_changes_with_key(self):
        original = config(TradingMode.REAL)
        with patch("dockdack.gui_service.KiwoomConfig.from_env", return_value=original) as load:
            service = TradingService(mode=TradingMode.REAL)
            load.return_value = replace(original, app_key="other-account-key")
            self.assertEqual(service.broker(Market.US).us_config.app_key, original.app_key)
            other = TradingService(mode=TradingMode.REAL)
        self.assertNotEqual(service.storage_scope, other.storage_scope)
        self.assertNotIn(original.app_key, service.storage_scope)

    def test_real_order_needs_both_session_warning_and_persistent_allow_flag(self):
        for allow, acknowledged in ((True, False), (False, True)):
            with self.subTest(allow=allow, acknowledged=acknowledged):
                service, _, transport = self.service(allow=allow)
                if acknowledged:
                    service.acknowledge_live_risk(LIVE_RISK_ACKNOWLEDGEMENT)
                request = self.prepare(service)
                with self.assertRaises(OrderNotSent):
                    service.submit(request)
                self.assertEqual(transport.calls, [])

    def test_real_fake_transport_uses_only_real_endpoint_and_explicit_sdk_confirmation(self):
        service, _, transport = self.service()
        service.acknowledge_live_risk(LIVE_RISK_ACKNOWLEDGEMENT)
        request = self.prepare(service)
        result = service.submit(request)
        self.assertIs(result.mode, TradingMode.REAL)
        self.assertTrue(all(call["url"].startswith("https://api.kiwoom.com/") for call in transport.calls))
        self.assertEqual(sum(call["url"].endswith("/ordr") for call in transport.calls), 1)
        with self.assertRaises(OrderNotSent):
            service.submit(request)
        self.assertEqual(len(transport.calls), 2)

    def test_old_demo_preview_cannot_be_sent_by_real_service(self):
        demo, _, demo_transport = self.service(mode=TradingMode.DEMO)
        real, _, real_transport = self.service()
        real.acknowledge_live_risk(LIVE_RISK_ACKNOWLEDGEMENT)
        with self.assertRaises(OrderNotSent):
            real.submit(self.prepare(demo))
        self.assertEqual(demo_transport.calls + real_transport.calls, [])

    def test_config_and_actual_http_client_must_agree(self):
        service, broker, transport = self.service()
        broker._us_http.config = config(TradingMode.DEMO)
        with self.assertRaises(ValueError):
            service.broker(Market.US)
        self.assertEqual(transport.calls, [])

    def test_wrong_result_environment_is_unknown_not_retryable(self):
        service, broker, _ = self.service()
        service.acknowledge_live_risk(LIVE_RISK_ACKNOWLEDGEMENT)
        request = self.prepare(service)
        with patch.object(broker, "place_order", return_value=OrderResult(True, TradingMode.DEMO, request, "1", "접수")):
            with self.assertRaises(OrderOutcomeUnknown):
                service.submit(request)
        with self.assertRaises(OrderNotSent):
            service.submit(request)

    def test_permission_revoked_during_token_call_blocks_paced_order_transport(self):
        class RevokeTransport(QueueTransport):
            def request(inner, method, url, **kwargs):
                result = super().request(method, url, **kwargs)
                if url.endswith("/oauth2/token"):
                    service.revoke_live_risk()
                return result

        transport = RevokeTransport(token_response(), FakeResponse({"return_code": 0, "ord_no": "001"}))
        service, _, _ = self.service(transport=transport)
        service.acknowledge_live_risk(LIVE_RISK_ACKNOWLEDGEMENT)
        with self.assertRaises(OrderNotSent):
            service.submit(self.prepare(service))
        self.assertFalse(any(call["url"].endswith("/ordr") for call in transport.calls))

    def test_service_guard_keeps_outer_autotrader_guard(self):
        service, _, transport = self.service()
        service.acknowledge_live_risk(LIVE_RISK_ACKNOWLEDGEMENT)

        def stop():
            raise OrderNotSent("OFF")

        with order_send_guard(stop), self.assertRaisesRegex(OrderNotSent, "OFF"):
            service.submit(self.prepare(service))
        self.assertFalse(any(call["url"].endswith("/ordr") for call in transport.calls))

    def test_nested_guards_compose_and_restore(self):
        calls = []
        with order_send_guard(lambda: calls.append("outer")):
            with order_send_guard(lambda: calls.append("inner")):
                _ORDER_SEND_GUARD.get()()
            _ORDER_SEND_GUARD.get()()
        self.assertEqual(calls, ["outer", "inner", "outer"])
        self.assertIsNone(_ORDER_SEND_GUARD.get())


class RealFakeService(FakeTradingService):
    """No API client: exercises only engine decisions and journal transitions."""
    def __init__(self):
        super().__init__()
        self.mode = TradingMode.REAL
        self.storage_scope = "a" * 64
        self.live_risk_acknowledged = True

    def ensure_environment(self, inst):
        if self.mode is not TradingMode.REAL:
            raise ValueError("environment changed")

    def ensure_order_permission(self, inst):
        self.ensure_environment(inst)
        if not self.live_risk_acknowledged:
            raise ValueError("warning not acknowledged")

    def submit(self, request):
        self.submitted.append(request)
        return OrderResult(True, TradingMode.REAL, request, "0000200", "접수")


class RealAutoTraderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.service = RealFakeService()
        self.store = WatchStore(self.path / "real.sqlite3", mode=TradingMode.REAL, storage_scope=self.service.storage_scope)
        self.item = WatchItem(self.service.resolve("005930"))
        self.store.save_item(self.item)
        self.rule = TriggerRule.create(self.item, "price_ge", "buy", 1, Decimal(1000), Decimal(95))
        self.store.add_rule(self.rule)
        self.engine = AutoTrader(self.service, self.store, clock=lambda: NOW)

    def test_explicit_real_arm_can_execute_with_fake_service_and_correct_ledger(self):
        with self.assertRaises(ValueError):
            self.engine.enable_orders("DEMO_AUTOTRADE")
        self.engine.enable_orders("REAL_AUTOTRADE")
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(self.store.rules()[0].status, "accepted")

    def test_demo_ledger_and_different_real_key_ledger_block_arm(self):
        for mode, scope in ((TradingMode.DEMO, "demo"), (TradingMode.REAL, "b" * 64)):
            store = WatchStore(self.path / f"{mode.value}.other.sqlite3", mode=mode, storage_scope=scope)
            engine = AutoTrader(self.service, store, clock=lambda: NOW)
            with self.assertRaises(ValueError):
                engine.enable_orders("REAL_AUTOTRADE")
            self.assertFalse(engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_mode_or_warning_change_disarms_before_any_submission(self):
        for change in (lambda: setattr(self.service, "mode", TradingMode.DEMO),
                       lambda: setattr(self.service, "live_risk_acknowledged", False)):
            self.service.mode = TradingMode.REAL
            self.service.live_risk_acknowledged = True
            self.engine.enable_orders("REAL_AUTOTRADE")
            change()
            self.engine.poll()
            self.assertFalse(self.engine.orders_enabled)
            self.assertEqual(self.service.submitted, [])

    def test_demo_policy_is_blocked_even_with_live_keys(self):
        self.engine.external_only = True
        self.engine.external_policy = ExternalPolicy("random-demo", 1, Decimal(1000), Decimal(1000))
        with self.assertRaisesRegex(ValueError, "모의 테스트"):
            self.engine.enable_orders("REAL_AUTOTRADE")

    def test_renaming_demo_source_does_not_hide_generator_signal_identity(self):
        self.engine.snapshot(self.item)
        chart = export_charts(self.store, self.path / "charts.json", now=NOW)
        policy = ExternalPolicy("renamed-strategy", 1, Decimal(1000), Decimal(1000))
        entry = {"signal_id": "legitimate-model-decision",
                 "export_id": chart["export_id"], "market": "domestic", "symbol": "005930", "exchange": "KRX",
                 "action": "buy", "quantity": 1, "max_notional": "1000", "generated_at": NOW.isoformat(),
                 "expires_at": (NOW + timedelta(minutes=2)).isoformat()}
        payload = {"schema_version": 1, "trading_mode": "real", "source_id": policy.source_id, "signals": [entry]}
        ingest_signals(self.store, payload, policy, now=NOW)
        entry["signal_id"] = uuid5(NAMESPACE_URL, f"random-demo:{chart['export_id']}:{self.item.id}").hex
        with self.assertRaisesRegex(ValueError, "이름을 바꿔도"):
            ingest_signals(self.store, payload, policy, now=NOW)
        # Even bypassing ingestion by tampering with the saved record must not
        # evade the engine's last-moment provenance check.
        with self.store.connection() as db:
            db.execute("UPDATE external_signals SET payload=?", (json.dumps({**entry, "trading_mode": "real"}),))
        self.engine.external_only, self.engine.external_policy = True, policy
        with self.assertRaisesRegex(ValueError, "이름을 바꿔도"):
            self.engine.enable_orders("REAL_AUTOTRADE")
        self.assertEqual(self.service.submitted, [])

    def test_real_export_and_signal_require_explicit_matching_mode(self):
        self.engine.snapshot(self.item)
        chart = export_charts(self.store, self.path / "charts.json", now=NOW)
        self.assertEqual(chart["source"], "kiwoom_real")
        self.assertEqual(chart["trading_mode"], "real")
        policy = ExternalPolicy("external-model", 1, Decimal(1000), Decimal(1000))
        entry = {"signal_id": "decision-1", "export_id": chart["export_id"], "market": "domestic",
                 "symbol": "005930", "exchange": "KRX", "action": "buy", "quantity": 1,
                 "max_notional": "1000", "generated_at": NOW.isoformat(),
                 "expires_at": (NOW + timedelta(minutes=2)).isoformat()}
        payload = {"schema_version": 1, "source_id": policy.source_id, "signals": [entry]}
        for mode in (None, "demo", [], "invalid"):
            candidate = dict(payload)
            if mode is not None:
                candidate["trading_mode"] = mode
            with self.assertRaises(ValueError):
                ingest_signals(self.store, candidate, policy, now=NOW)
        self.assertEqual(len(self.store.rules()), 1)
        payload["trading_mode"] = "real"
        self.assertEqual(ingest_signals(self.store, payload, policy, now=NOW)["queued"], 1)
        self.assertEqual(ingest_signals(self.store, payload, policy, now=NOW)["duplicates"], 1)
        self.engine.external_only, self.engine.external_policy = True, policy
        self.engine.enable_orders("REAL_AUTOTRADE")
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 1)

    def test_live_signal_rejects_demo_source_and_saved_mode_tampering(self):
        policy = ExternalPolicy("random-demo", 1, Decimal(1000), Decimal(1000))
        with self.assertRaisesRegex(ValueError, "모의 테스트"):
            ingest_signals(self.store, {"schema_version": 1, "trading_mode": "real",
                           "source_id": "random-demo", "signals": []}, policy, now=NOW)
        demo_store = WatchStore(self.path / "demo-signals.sqlite3")
        with self.assertRaises(ValueError):
            ingest_signals(demo_store, {"schema_version": 1, "trading_mode": "real",
                           "source_id": "random-demo", "signals": []}, policy, now=NOW)

    def test_random_demo_generator_cannot_construct_or_publish_in_real_environment(self):
        policy = ExternalPolicy("renamed-strategy", 1, Decimal(1000), Decimal(1000))
        path = self.path / "signals.json"
        with self.assertRaisesRegex(ValueError, "모의투자 전용"):
            RandomDemoSignals(self.service, self.store, policy, path, clock=lambda: NOW)
        demo = FakeTradingService()
        demo_store = WatchStore(self.path / "demo.sqlite3")
        generator = RandomDemoSignals(demo, demo_store, policy, path, clock=lambda: NOW)
        generator.service = self.service
        with self.assertRaisesRegex(ValueError, "모의투자 전용"):
            generator.publish({"stocks": []})
        self.assertFalse(path.exists())

    def test_read_only_inspector_matches_live_receiver_environment_contract(self):
        policy = ExternalPolicy("external-model", 1, Decimal(1000), Decimal(1000))
        path = self.path / "inspect.json"
        payload = {"schema_version": 1, "source_id": policy.source_id, "signals": []}
        atomic_json(path, payload)
        self.assertEqual(inspect_signal_file(path, policy, now=NOW)["state"], "format_ok")
        self.assertEqual(inspect_signal_file(path, policy, now=NOW, mode=TradingMode.REAL)["state"], "error")
        payload["trading_mode"] = "real"
        atomic_json(path, payload)
        self.assertEqual(inspect_signal_file(path, policy, now=NOW, mode=TradingMode.REAL)["state"], "format_ok")
        self.assertEqual(inspect_signal_file(path, policy, now=NOW)["state"], "error")
        self.assertEqual(len(self.store.rules()), 1)
        self.assertEqual(self.service.submitted, [])
