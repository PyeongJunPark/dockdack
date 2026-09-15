"""Offline end-to-end tests of final checks after HTTP rate-limit waiting."""

from __future__ import annotations

import tempfile
import unittest
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from dockdack.autotrade import AutoTrader
from dockdack.config import KiwoomConfig
from dockdack.http import KiwoomHTTPClient
from dockdack.kiwoom import KiwoomBroker
from dockdack.signal_bridge import ExternalPolicy, SignalFileReader, atomic_json, export_charts, ingest_signals
from dockdack.watchlist import TriggerRule, WatchItem, WatchStore
from test_autotrade import FakeTradingService, NOW
from test_http import Response, Transport, token


class Clock:
    def __init__(self):
        self.seconds = 0.0
        self.after_sleep = lambda: None

    def __call__(self):
        return self.seconds

    def now(self):
        return NOW + timedelta(seconds=self.seconds)

    def sleep(self, seconds):
        self.seconds += seconds
        callback, self.after_sleep = self.after_sleep, lambda: None
        callback()


class PacedOrderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.store = WatchStore(self.folder / "watch.sqlite3")
        self.clock = Clock()
        self.service = FakeTradingService()
        self.item = WatchItem(self.service.resolve("005930"), "삼성전자")
        self.store.save_item(self.item)
        self.engine = AutoTrader(self.service, self.store, clock=self.clock.now)
        self.market = patch("dockdack.autotrade.regular_session", return_value=True)
        self.market_check = self.market.start()
        self.addCleanup(self.market.stop)

    def use_http(self, interval=2, outcome=None):
        config = KiwoomConfig("offline-key", "offline-secret", min_request_interval_seconds=interval)
        self.transport = Transport(self.clock, token(), outcome or Response({"return_code": 0, "ord_no": "demo-order"}))
        self.client = KiwoomHTTPClient(config, transport=self.transport,
                                      monotonic=self.clock, sleeper=self.clock.sleep)
        # All requests use this injected fake transport, including broker.place_order.
        broker = KiwoomBroker(config, transport=self.transport)
        broker._domestic_http = broker._us_http = self.client
        self.service.submit = broker.place_order
        self.client.get_access_token()

    def manual_rule(self):
        rule = TriggerRule.create(self.item, "price_ge", "buy", 1, Decimal(1000), threshold=Decimal(95))
        self.store.add_rule(rule)
        return rule

    def external_rule(self, expiry=60):
        snapshot = self.engine.snapshot(self.item)
        chart = export_charts(self.store, self.folder / "charts.json", now=self.clock.now())
        self.policy = ExternalPolicy("offline", 1, Decimal(1000), Decimal(1000))
        self.signal = dict(signal_id="buy-1", export_id=chart["export_id"], market="domestic",
                           symbol="005930", exchange="KRX", action="buy", quantity=1, max_notional="1000",
                           generated_at=NOW.isoformat(), expires_at=(NOW + timedelta(seconds=expiry)).isoformat())
        self.inbox = self.folder / "signals.json"
        self.publish(self.signal)
        self.engine.external_only = True
        self.engine.external_policy = self.policy
        self.engine.external_reader = SignalFileReader(self.store, self.inbox, self.policy, self.clock.now)
        self.engine.external_reader()
        return self.store.rules()[0], snapshot

    def publish(self, signal):
        atomic_json(self.inbox, {"schema_version": 1, "source_id": "offline", "signals": [signal]})

    def order_calls(self):
        return [call for call in self.transport.calls if call["headers"].get("api-id") == "kt10000"]

    def execute(self, rule, snapshot=None):
        self.engine.enable_orders("DEMO_AUTOTRADE")
        return self.engine._execute(self.item, rule, snapshot or self.engine.snapshot(self.item))

    def assert_not_sent_once(self, rule):
        self.assertEqual(self.order_calls(), [])
        self.assertEqual(self.store.attempts()[0]["status"], "not_sent")
        self.assertEqual(self.store.rules()[0].status, "not_sent")
        self.engine.poll()
        self.assertEqual(self.order_calls(), [])
        self.assertEqual(len(self.store.attempts()), 1)

    def test_off_during_wait_is_not_sent_and_scope_does_not_leak(self):
        rule = self.manual_rule()
        self.use_http()
        self.clock.after_sleep = self.engine.disarm
        self.assertTrue(self.execute(rule))
        self.assertFalse(self.engine.orders_enabled)
        self.assert_not_sent_once(rule)
        # Unscoped fake HTTP is independent of the previous denied auto-order.
        self.client.request(api_id="kt10000", path="/api/dostk/ordr")
        self.assertEqual(len(self.order_calls()), 1)

    def test_stop_during_wait_blocks_transport(self):
        rule = self.manual_rule()
        self.use_http()
        self.clock.after_sleep = self.engine.stop
        self.assertTrue(self.execute(rule))
        self.assert_not_sent_once(rule)

    def test_wait_longer_than_quote_freshness_is_terminal_not_sent(self):
        rule = self.manual_rule()
        self.use_http(interval=20)
        self.assertTrue(self.execute(rule))
        self.assertIn("15초", self.store.attempts()[0]["message"])
        self.assert_not_sent_once(rule)

    def test_market_close_during_wait_blocks_transport(self):
        rule = self.manual_rule()
        self.use_http()
        self.market_check.side_effect = lambda market, now: now < NOW + timedelta(seconds=1)
        self.assertTrue(self.execute(rule))
        self.assertIn("정규장", self.store.attempts()[0]["message"])
        self.assert_not_sent_once(rule)

    def test_external_signal_expiry_during_wait_blocks_transport(self):
        rule, snapshot = self.external_rule(expiry=1)
        self.use_http()
        self.assertTrue(self.execute(rule, snapshot))
        self.assertIn("만료", self.store.attempts()[0]["message"])
        self.assert_not_sent_once(rule)

    def test_new_hold_after_claim_supersedes_paced_buy_before_send(self):
        rule, snapshot = self.external_rule()
        self.use_http()
        def publish_hold():
            hold = {key: value for key, value in self.signal.items() if key not in {"quantity", "max_notional"}}
            hold.update(signal_id="hold-2", action="hold", generated_at=self.clock.now().isoformat())
            self.publish(hold)
        self.clock.after_sleep = publish_hold
        self.assertTrue(self.execute(rule, snapshot))
        self.assertIn("HOLD", self.store.attempts()[0]["message"])
        self.assert_not_sent_once(rule)

    def test_invalid_external_file_during_wait_disarms_without_transport(self):
        rule, snapshot = self.external_rule()
        self.use_http()
        self.clock.after_sleep = lambda: atomic_json(self.inbox, {"schema_version": 2})
        self.assertTrue(self.execute(rule, snapshot))
        self.assertFalse(self.engine.orders_enabled)
        self.assert_not_sent_once(rule)

    def test_valid_paced_order_sends_once_and_records_broker_acceptance(self):
        rule = self.manual_rule()
        self.use_http()
        self.assertTrue(self.execute(rule))
        self.assertEqual(len(self.order_calls()), 1)
        self.assertEqual(self.store.attempts()[0]["status"], "accepted")
        self.engine.poll()
        self.assertEqual(len(self.order_calls()), 1)

    def test_transport_timeout_remains_unknown_never_not_sent_or_retried(self):
        rule = self.manual_rule()
        self.use_http(outcome=TimeoutError("fake transport timeout"))
        self.assertTrue(self.execute(rule))
        self.assertEqual(len(self.order_calls()), 1)
        self.assertEqual(self.store.attempts()[0]["status"], "unknown")
        self.assertFalse(self.engine.orders_enabled)
        self.engine.poll()
        self.assertEqual(len(self.order_calls()), 1)

    def test_fake_service_without_http_guard_remains_compatible(self):
        rule = self.manual_rule()
        self.assertTrue(self.execute(rule))
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(self.store.attempts()[0]["status"], "accepted")


if __name__ == "__main__":
    unittest.main()
