"""Offline regressions for transient sweep failures; no broker/network calls."""

import sqlite3
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from requests.exceptions import ConnectionError as TransportConnectionError, Timeout as TransportTimeout

from dockdack.autotrade import AutoTrader
from dockdack.exceptions import BrokerAPIError, ConfigurationError, OrderOutcomeUnknown
from dockdack.models import TradingMode
from dockdack.watchlist import TriggerRule, WatchItem, WatchStore
from test_autotrade import FakeTradingService, NOW, position


class AutoOrderResilienceTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.store = WatchStore(Path(folder.name) / "offline.sqlite3")
        self.service = FakeTradingService()
        self.item = WatchItem(self.service.resolve("005930"))
        self.store.save_item(self.item)
        self.rule = TriggerRule.create(self.item, kind="price_ge", side="buy", quantity=1,
                                       max_notional=Decimal("1000"), threshold=Decimal("95"))
        self.store.add_rule(self.rule)
        self.engine = AutoTrader(self.service, self.store, clock=lambda: NOW)
        self.engine.isolated_symbol_errors = True
        sessions = patch("dockdack.autotrade.regular_session", return_value=True)
        sessions.start()
        self.addCleanup(sessions.stop)
        self.engine.enable_orders("DEMO_AUTOTRADE")

    def fail_checkpoint(self, error):
        def checkpoint():
            raise error
        with self.assertRaises(type(error)):
            self.engine.poll(checkpoint=checkpoint)

    def test_transient_checkpoint_preserves_on_then_next_sweep_succeeds_once(self):
        self.fail_checkpoint(TimeoutError("temporary account timeout"))
        self.assertTrue(self.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.attempts(), ())
        self.engine.poll()
        self.engine.poll()
        self.assertTrue(self.engine.orders_enabled)
        self.assertEqual(len(self.service.submitted), 1)

    def test_connection_and_broker_backpressure_preserve_on(self):
        for error in (ConnectionError("offline"), TransportTimeout("slow"),
                      TransportConnectionError("offline"),
                      BrokerAPIError("too many requests", status_code=429),
                      BrokerAPIError("unavailable", status_code=503),
                      BrokerAPIError("rate limit", status_code=200, return_code="1700"),
                      BrokerAPIError("[1700:허용된 API 요청 개수를 초과하였습니다]", status_code=200, return_code=1)):
            with self.subTest(error=str(error)):
                self.fail_checkpoint(error)
                self.assertTrue(self.engine.orders_enabled)
                self.assertEqual(self.service.submitted, [])

    def test_http_transport_wrapper_preserves_on(self):
        error = BrokerAPIError("connection failed")
        error.__cause__ = TransportConnectionError("network disconnected")
        self.fail_checkpoint(error)
        self.assertTrue(self.engine.orders_enabled)

    def test_unexpected_and_critical_failures_still_turn_off(self):
        for error in (sqlite3.OperationalError("database is locked"),
                      sqlite3.DatabaseError("database disk image malformed"),
                      ValueError("invalid ledger identity"),
                      RuntimeError("unexpected checkpoint failure"),
                      OSError("journal disk unavailable"),
                      ConfigurationError("missing key"),
                      BrokerAPIError("authentication required", status_code=401),
                      BrokerAPIError("permission denied", status_code=403),
                      BrokerAPIError("undocumented data error"),
                      BrokerAPIError("bad return code", status_code=200, return_code=1700.5),
                      OrderOutcomeUnknown("unconfirmed order", status_code=503)):
            with self.subTest(error=str(error)):
                self.engine.enable_orders("DEMO_AUTOTRADE")
                self.fail_checkpoint(error)
                self.assertFalse(self.engine.orders_enabled)
                self.assertEqual(self.service.submitted, [])

    def test_legacy_strict_mode_still_turns_off_on_timeout(self):
        self.engine.isolated_symbol_errors = False
        self.fail_checkpoint(TimeoutError("temporary failure"))
        self.assertFalse(self.engine.orders_enabled)

    def test_explicit_off_is_not_undone_by_transient_exception(self):
        def checkpoint():
            self.engine.disarm()
            raise TimeoutError("timeout after OFF")
        with self.assertRaises(TimeoutError):
            self.engine.poll(checkpoint=checkpoint)
        self.assertFalse(self.engine.orders_enabled)
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])

    def test_explicit_stop_is_not_undone_by_transient_exception(self):
        def checkpoint():
            self.engine.stop()
            raise TimeoutError("timeout after stop")
        with self.assertRaises(TimeoutError):
            self.engine.poll(checkpoint=checkpoint)
        self.assertFalse(self.engine.orders_enabled)
        self.assertTrue(self.engine._stop.is_set())
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])

    def test_environment_change_still_disarms_without_sending(self):
        self.service.mode = TradingMode.REAL
        results = self.engine.poll()
        self.assertIsInstance(results[self.item.id], ValueError)
        self.assertFalse(self.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_holding_checkpoint_timeout_keeps_approved_on(self):
        self.store.remove_item(self.item.id)
        self.engine.enable_holdings_exits = True
        self.service.positions = (position(),)
        self.fail_checkpoint(TimeoutError("temporary holdings helper outage"))
        self.assertTrue(self.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])


if __name__ == "__main__":
    unittest.main()
