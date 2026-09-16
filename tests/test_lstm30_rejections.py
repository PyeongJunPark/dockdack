"""Offline rejection quarantine and actual LSTM GUI-engine order-path proofs."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from dockdack import BrokerAPIError, Market, OrderOutcomeUnknown, OrderSide
from dockdack.gui_service import Instrument
from dockdack.http import _ORDER_SEND_GUARD
from dockdack.lstm30_gui import _LSTM30AutoTrader
from dockdack.lstm30_rejections import rejected_today
from dockdack.watchlist import TriggerRule, WatchItem, WatchStore
from test_autotrade import FakeTradingService, position


NOW = datetime(2026, 9, 14, 14, tzinfo=timezone.utc)


def seed_attempt(store, item, when, *, status="rejected", side="buy", rule_id=None):
    """Insert a historical broker response directly; never invoke an order API."""
    store.save_item(item)
    rule = TriggerRule.create(item, "price_ge", side, 1, Decimal(1000), Decimal(95))
    if rule_id:
        rule = replace(rule, id=rule_id)
    store.add_rule(rule)
    with store.connection() as db:
        db.execute("UPDATE rules SET status=? WHERE id=?", (status, rule.id))
        db.execute("INSERT INTO attempts(rule_id,watch_id,status,price,started_at) VALUES(?,?,?,?,?)",
                   (rule.id, item.id, status, "100", when.isoformat() if isinstance(when, datetime) else when))
    return rule


class GuardedService(FakeTradingService):
    def __init__(self):
        super().__init__()
        self.before_send = lambda: None
        self.reject_symbol = None

    def submit(self, request):
        self.before_send()
        guard = _ORDER_SEND_GUARD.get()
        if guard is not None:
            guard()
        if request.symbol == self.reject_symbol:
            self.submitted.append(request)
            raise BrokerAPIError("confirmed test rejection", return_code=2000, status_code=200)
        return super().submit(request)


class RejectedTodayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = WatchStore(Path(self.temp.name) / "ledger.sqlite3")
        self.item = WatchItem(Instrument(Market.US, "AAPL", "ND"), "Apple", 31)

    def test_both_sides_and_close_source_quarantine_survive_store_reopen(self):
        for side in ("buy", "sell"):
            seed_attempt(self.store, self.item, NOW, side=side, rule_id="close-" + uuid4().hex)
        self.assertTrue(rejected_today(WatchStore(self.store.path), self.item.instrument, NOW))
        self.assertFalse(rejected_today(self.store, self.item.instrument, NOW + timedelta(days=1)))

    def test_us_market_local_day_crosses_utc_midnight_without_reset(self):
        seed_attempt(self.store, self.item, datetime(2026, 9, 14, 23, 30, tzinfo=timezone.utc))
        self.assertTrue(rejected_today(self.store, self.item.instrument,
                                      datetime(2026, 9, 15, 0, 30, tzinfo=timezone.utc)))
        self.assertFalse(rejected_today(self.store, self.item.instrument,
                                       datetime(2026, 9, 15, 4, 0, tzinfo=timezone.utc)))

    def test_domestic_market_local_midnight_resets_within_same_utc_day(self):
        item = WatchItem(Instrument(Market.DOMESTIC, "005930", "KRX"), "Samsung", 31)
        seed_attempt(self.store, item, datetime(2026, 9, 14, 14, 30, tzinfo=timezone.utc))
        self.assertTrue(rejected_today(self.store, item.instrument,
                                      datetime(2026, 9, 14, 14, 59, tzinfo=timezone.utc)))
        self.assertFalse(rejected_today(self.store, item.instrument,
                                       datetime(2026, 9, 14, 15, 0, tzinfo=timezone.utc)))

    def test_quarantine_is_exact_market_exchange_and_symbol_identity(self):
        seed_attempt(self.store, self.item, NOW)
        for instrument in (Instrument(Market.US, "AAPL", "NY"), Instrument(Market.US, "MSFT", "ND"),
                           Instrument(Market.DOMESTIC, "005930", "KRX")):
            with self.subTest(instrument=instrument):
                self.assertFalse(rejected_today(self.store, instrument, NOW))

    def test_nonrejected_statuses_do_not_trigger_rejection_quarantine(self):
        for index, status in enumerate(("accepted", "filled", "cancelled", "not_sent", "unknown", "submitting", "reviewed")):
            item = WatchItem(Instrument(Market.US, f"S{index}", "ND"), status, 31)
            seed_attempt(self.store, item, NOW, status=status)
            self.assertFalse(rejected_today(self.store, item.instrument, NOW), status)

    def test_malformed_or_naive_rejected_time_fails_closed(self):
        for index, timestamp in enumerate(("invalid-date", "2026-09-14T14:00:00", "")):
            item = WatchItem(Instrument(Market.US, f"S{index}", "ND"), "Malformed", 31)
            seed_attempt(self.store, item, timestamp)
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                rejected_today(self.store, item.instrument, NOW)
        with self.assertRaises(ValueError):
            rejected_today(self.store, self.item.instrument, NOW.replace(tzinfo=None))


class GuiRejectionEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.now = NOW
        self.service = GuardedService()
        self.store = WatchStore(Path(self.temp.name) / "ledger.sqlite3")
        self.items = (WatchItem(Instrument(Market.US, "AAPL", "ND"), "Apple", 31),
                      WatchItem(Instrument(Market.US, "MSFT", "ND"), "Microsoft", 31))
        for item in self.items:
            self.store.save_item(item)
        self.engine = self.new_engine()
        self.network_patch = patch("requests.sessions.Session.request", side_effect=AssertionError("network forbidden"))
        self.network = self.network_patch.start()
        self.addCleanup(self.network_patch.stop)
        self.addCleanup(self.network.assert_not_called)

    def new_engine(self):
        return _LSTM30AutoTrader(self.service, self.store, items=self.items, clock=lambda: self.now)

    def rule(self, item=None, *, side="buy", quantity=1):
        rule = TriggerRule.create(item or self.items[0], "price_ge", side, quantity, Decimal(1000), Decimal(95))
        self.store.add_rule(rule)
        return rule

    def arm(self):
        self.engine.enable_orders("DEMO_AUTOTRADE")

    def test_confirmed_first_symbol_reject_allows_second_symbol_and_keeps_on(self):
        self.rule(self.items[0])
        self.rule(self.items[1])
        self.service.reject_symbol = "AAPL"
        self.arm()
        self.engine.poll()
        self.assertTrue(self.engine.orders_enabled)
        self.assertEqual([order.symbol for order in self.service.submitted], ["AAPL", "MSFT"])
        self.assertEqual([row["status"] for row in self.store.attempts()], ["rejected", "accepted"])
        self.assertEqual(self.engine._critical_attempts(), [])
        self.rule(self.items[0])
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 2)
        self.assertEqual(len(self.store.attempts(self.items[0].id)), 1)

    def test_new_sell_signals_are_quarantined_across_restart_but_allowed_next_local_day(self):
        seed_attempt(self.store, self.items[0], self.now, side="buy")
        self.service.positions = (replace(position(), market=Market.US, symbol="AAPL", exchange="ND", currency="USD"),)
        for restart in (False, True):
            if restart:
                self.engine = self.new_engine()
            rule = self.rule(side="sell")
            self.arm()
            self.engine.poll()
            self.assertTrue(self.engine.orders_enabled)
            self.assertEqual(self.service.submitted, [])
            self.assertEqual(next(row.status for row in self.store.rules() if row.id == rule.id), "paused")
        self.now += timedelta(days=1)
        self.rule(side="sell")
        self.engine.poll()
        self.assertEqual([(order.side, order.quantity) for order in self.service.submitted], [(OrderSide.SELL, 1)])

    def test_confirmed_rejection_does_not_rearm_manual_off_but_explicit_on_is_allowed(self):
        seed_attempt(self.store, self.items[0], self.now)
        self.rule(self.items[1])
        self.arm()
        self.engine.disarm()
        self.engine.poll()
        self.assertFalse(self.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])
        self.arm()
        self.engine.poll()
        self.assertTrue(self.engine.orders_enabled)
        self.assertEqual([order.symbol for order in self.service.submitted], ["MSFT"])

    def test_new_confirmed_rejection_in_final_pacer_blocks_send_without_global_off(self):
        rule = self.rule()
        self.arm()
        self.service.before_send = lambda: seed_attempt(self.store, self.items[0], self.now, side="sell")
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])
        self.assertTrue(self.engine.orders_enabled)
        self.assertEqual(next(row["status"] for row in self.store.attempts() if row["rule_id"] == rule.id), "not_sent")

    def test_ambiguous_or_unknown_broker_errors_still_disarm_before_other_symbols(self):
        failures = (OrderOutcomeUnknown("missing acknowledgement"), BrokerAPIError("no code", status_code=200),
                    BrokerAPIError("zero code", return_code=0, status_code=200),
                    BrokerAPIError("server error", return_code=2000, status_code=500))
        for index, failure in enumerate(failures):
            with self.subTest(failure=str(failure)):
                self.store = WatchStore(Path(self.temp.name) / f"unknown-{index}.sqlite3")
                self.service = GuardedService()
                for item in self.items:
                    self.store.save_item(item)
                    self.rule(item)
                self.engine = self.new_engine()
                self.arm()
                self.service.submit_error = failure
                self.engine.poll()
                self.assertFalse(self.engine.orders_enabled)
                self.assertEqual(len(self.service.submitted), 1)
                self.assertEqual(self.store.attempts()[0]["status"], "unknown")
                self.service.submit_error = None
                self.engine.poll()
                self.assertEqual(len(self.service.submitted), 1)
                with self.assertRaises(ValueError):
                    self.arm()

    def test_existing_submitting_intent_still_prevents_explicit_rearm(self):
        seed_attempt(self.store, self.items[0], self.now, status="submitting")
        self.rule(self.items[1])
        self.engine = self.new_engine()
        with self.assertRaises(ValueError):
            self.arm()
        self.assertFalse(self.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_malformed_rejected_timestamp_disarms_before_other_instruments(self):
        seed_attempt(self.store, self.items[0], "malformed")
        self.rule(self.items[0])
        self.rule(self.items[1])
        self.arm()
        self.engine.poll()
        self.assertFalse(self.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_malformed_rejection_in_final_guard_disarms_and_records_not_sent(self):
        rule = self.rule()
        self.arm()
        self.service.before_send = lambda: seed_attempt(self.store, self.items[0], "malformed", side="sell")
        self.engine.poll()
        self.assertFalse(self.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(next(row["status"] for row in self.store.attempts() if row["rule_id"] == rule.id), "not_sent")


if __name__ == "__main__":
    unittest.main()
