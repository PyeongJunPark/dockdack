"""Past accepted BUY records never become invented fills or live-order authority.

Every account, quote, execution and send here is an in-memory fake. All ledgers
are created in temporary directories; no configured broker is instantiated.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dockdack.autotrade import AutoTrader
from dockdack.exceptions import BrokerAPIError
from dockdack.http import _ORDER_SEND_GUARD
from dockdack.models import Market, OpenOrder, OrderSide, TradingMode
from dockdack.watchlist import TriggerRule, WatchItem, WatchStore
from test_autotrade import position
from test_v00_execution import V00Service


# New York: September 14, 11:00. Seoul: September 15, 00:00.
NOW = datetime(2026, 9, 14, 15, tzinfo=timezone.utc)
OLD = NOW - timedelta(days=3)


class GuardedOfflineService(V00Service):
    def __init__(self):
        super().__init__()
        self.before_send = lambda: None
        self.order_checks = []
        self.execution_checks = []

    def safety_orders(self, instrument):
        self.order_checks.append(instrument)
        return super().safety_orders(instrument)

    def safety_executions(self, instrument):
        self.execution_checks.append(instrument)
        return super().safety_executions(instrument)

    def submit(self, request):
        self.before_send()
        guard = _ORDER_SEND_GUARD.get()
        if guard is None:
            raise AssertionError("The fake transport requires the real final-send guard")
        guard()
        return super().submit(request)


class HistoricalBuyExitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dockdack-historical-buy-tests-")
        self.addCleanup(self.temp.cleanup)
        self.store = WatchStore(Path(self.temp.name) / "offline.sqlite3")
        self.service = GuardedOfflineService()
        self.now = NOW
        self.open = True
        self.item = WatchItem(self.service.resolve("AAPL"), "Offline holding", 30)
        self.store.save_item(self.item)
        self.service.positions = (replace(position(3, 3), market=Market.US, symbol="AAPL",
                                          exchange="ND", currency="USD"),)
        self.service.prices = [D(102)]
        self.engine = self.new_engine(self.store)
        self.sessions = patch("dockdack.autotrade.regular_session",
                              side_effect=lambda market, now: self.open and market is Market.US)
        self.sessions.start()
        self.addCleanup(self.sessions.stop)
        self.sequence = 100

    def new_engine(self, store):
        engine = AutoTrader(self.service, store, clock=lambda: self.now)
        engine.enable_holdings_exits = True
        engine.isolated_symbol_errors = True
        return engine

    def seed(self, *, side="buy", status="accepted", started=OLD, number=None):
        """Inject a persisted state, including states concurrent writers could add."""
        self.sequence += 1
        rule = TriggerRule.create(self.item, kind="price_ge", side=side, quantity=1,
                                  max_notional=D(10000), threshold=D(95))
        self.store.add_rule(rule)
        stamp = started.isoformat() if isinstance(started, datetime) else started
        number = str(self.sequence) if number is None else number
        with self.store.connection() as db:
            db.execute("INSERT INTO attempts(rule_id,watch_id,status,price,started_at,order_number) VALUES(?,?,?,?,?,?)",
                       (rule.id, self.item.id, status, "100", stamp, number))
            db.execute("UPDATE rules SET status=? WHERE id=?", (status, rule.id))
        return rule

    def saved(self, rule):
        with self.store.connection() as db:
            return (dict(db.execute("SELECT * FROM rules WHERE id=?", (rule.id,)).fetchone()),
                    dict(db.execute("SELECT * FROM attempts WHERE rule_id=?", (rule.id,)).fetchone()))

    def sell_rule(self, threshold=101):
        rule = TriggerRule.create(self.item, kind="price_ge", side="sell", quantity=1,
                                  max_notional=D(10000), threshold=D(threshold))
        self.store.add_rule(rule)
        return rule

    def arm(self):
        self.engine.enable_orders("DEMO_AUTOTRADE")

    def holdings(self):
        self.engine._holdings_pass(set())

    def order(self, side="buy", **changes):
        values = dict(market=Market.US, order_number="101", symbol="AAPL", name="fake",
                      exchange="ND", side=side, status="접수", order_quantity=D(1),
                      filled_quantity=D(0), remaining_quantity=D(1), order_price=D(100))
        values.update(changes)
        return OpenOrder(**values)

    def test_eligibility_returns_only_older_accepted_buy_ids_without_mutation(self):
        old = self.seed()
        same = self.seed(started=NOW)
        future = self.seed(started=NOW + timedelta(days=1))
        unknown = self.seed(status="unknown")
        submitting = self.seed(status="submitting")
        sell = self.seed(side="sell")
        before = {row.id: self.saved(row) for row in (old, same, future, unknown, submitting, sell)}
        self.assertEqual(self.store.historical_buy_attempt_ids(self.item.id, NOW), frozenset({old.id}))
        self.assertEqual(before, {row.id: self.saved(row) for row in (old, same, future, unknown, submitting, sell)})

    def test_same_us_day_is_not_old_despite_seoul_midnight(self):
        self.seed(started=NOW - timedelta(hours=2))
        self.assertEqual(self.store.historical_buy_attempt_ids(self.item.id, NOW), frozenset())
        self.arm()
        self.holdings()
        self.assertEqual(self.service.submitted, [])

    def test_malformed_naive_future_or_unacknowledged_records_are_not_exempt(self):
        for stamp, number in ((OLD.replace(tzinfo=None), "101"), ("not-a-time", "102"),
                              (NOW + timedelta(days=1), "103"), (OLD, "")):
            with self.subTest(started=stamp, order_number=number):
                row = self.seed(started=stamp, number=number)
                self.assertNotIn(row.id, self.store.historical_buy_attempt_ids(self.item.id, NOW))

    def test_inconsistent_rule_status_and_missing_rule_are_not_exempt(self):
        mismatched = self.seed()
        orphan = self.seed()
        with self.store.connection() as db:
            db.execute("UPDATE rules SET status='filled' WHERE id=?", (mismatched.id,))
        self.assertNotIn(mismatched.id, self.store.historical_buy_attempt_ids(self.item.id, NOW))
        # Remove the parent only in this isolated corruption fixture.
        with self.store.connection() as db:
            db.execute("PRAGMA foreign_keys=OFF")
            db.execute("DELETE FROM rules WHERE id=?", (orphan.id,))
        self.assertNotIn(orphan.id, self.store.historical_buy_attempt_ids(self.item.id, NOW))
        self.assertFalse(self.store.claim(self.sell_rule(), D(102), NOW, allow_historical_buys=True))

    def test_profit_exit_once_preserves_original_buy_across_poll_and_restart(self):
        buy = self.seed()
        original = self.saved(buy)
        self.arm()
        self.engine.poll()
        self.engine.poll()
        restarted = self.new_engine(WatchStore(self.store.path))
        restarted.enable_orders("DEMO_AUTOTRADE")
        restarted.poll()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual((self.service.submitted[0].side, self.service.submitted[0].quantity), (OrderSide.SELL, 3))
        self.assertEqual(self.saved(buy), original)
        self.assertTrue(self.service.order_checks)
        self.assertGreaterEqual(len(self.service.account_calls), 2)

    def test_stop_loss_exit_also_preserves_original_buy(self):
        buy = self.seed()
        original = self.saved(buy)
        self.service.prices = [D(99)]
        self.arm()
        self.holdings()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(self.service.submitted[0].side, OrderSide.SELL)
        self.assertEqual(self.saved(buy), original)

    def test_domestic_past_buy_profit_exit_preserves_original_too(self):
        self.sessions.stop()
        self.item = WatchItem(self.service.resolve("005930"), "Offline domestic", 30)
        self.store.save_item(self.item)
        self.service.positions = (position(3, 3),)
        self.now = datetime(2026, 9, 14, 1, tzinfo=timezone.utc)
        buy = self.seed()
        original = self.saved(buy)
        self.arm()
        with patch("dockdack.autotrade.regular_session", side_effect=lambda market, now: market is Market.DOMESTIC):
            self.holdings()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(self.service.submitted[0].market, Market.DOMESTIC)
        self.assertEqual(self.saved(buy), original)

    def test_generic_sell_rule_uses_same_narrow_exception(self):
        buy = self.seed()
        original = self.saved(buy)
        self.engine.enable_holdings_exits = False
        self.sell_rule()
        self.arm()
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(self.saved(buy), original)

    def test_same_day_buy_unknown_buy_submitting_buy_and_pending_sells_still_block(self):
        for side, status, started in (("buy", "accepted", NOW), ("buy", "unknown", OLD),
                                      ("buy", "submitting", OLD), ("sell", "accepted", OLD),
                                      ("sell", "submitting", OLD), ("sell", "unknown", OLD)):
            with self.subTest(side=side, status=status):
                row = self.seed(side=side, status=status, started=started)
                self.arm()
                self.holdings()
                self.assertEqual(self.service.submitted, [])
                # Terminal review is test fixture cleanup, not production behavior.
                self.store.finish(row.id, "reviewed", "offline test fixture end")

    def test_readonly_monitor_mode_never_sells(self):
        self.seed()
        self.holdings()
        self.assertEqual(self.service.submitted, [])

    def test_old_only_reconciliation_skips_today_execution_endpoint(self):
        self.seed()
        with patch.object(self.service, "safety_executions", side_effect=AssertionError("not relevant today")):
            self.engine._reconcile(self.item)

    def test_current_day_reconciliation_still_reads_today_execution_endpoint(self):
        self.seed(started=NOW)
        self.engine._reconcile(self.item)
        self.assertEqual(self.service.execution_checks, [self.item.instrument])

    def test_current_broker_open_buy_and_sell_each_block_historical_exception(self):
        self.seed()
        self.arm()
        for side in ("buy", "sell"):
            with self.subTest(side=side):
                self.service.open_orders = (self.order(side),)
                self.holdings()
                self.assertEqual(self.service.submitted, [])

    def test_broker_open_order_errors_or_malformed_remaining_do_not_send(self):
        self.seed()
        self.arm()
        with patch.object(self.service, "safety_orders", side_effect=BrokerAPIError("offline outage")):
            self.holdings()
        self.assertEqual(self.service.submitted, [])
        for remaining in (D("NaN"), D(-1)):
            self.service.open_orders = (self.order(remaining_quantity=remaining),)
            self.holdings()
            self.assertEqual(self.service.submitted, [])

    def test_no_sellable_shares_or_bad_position_identity_do_not_send(self):
        self.seed()
        self.arm()
        original = self.service.positions[0]
        for change in (dict(sellable_quantity=D(0)), dict(sellable_quantity=D(4)),
                       dict(currency="KRW"), dict(sellable_quantity=D("NaN"))):
            with self.subTest(change=change):
                self.service.positions = (replace(original, **change),)
                self.holdings()
                self.assertEqual(self.service.submitted, [])

    def test_changed_exchange_in_fresh_preflight_account_cannot_back_sell(self):
        self.seed()
        original = self.service.positions[0]
        def changed():
            if len(self.service.account_calls) >= 2:
                self.service.positions = (replace(original, exchange="NY"),)
        self.service.on_account = changed
        self.arm()
        self.holdings()
        self.assertEqual(self.service.submitted, [])

    def test_fresh_preflight_rechecks_available_shares_and_target(self):
        self.seed()
        original = self.service.positions[0]
        def changed():
            if len(self.service.account_calls) >= 2:
                self.service.positions = (replace(original, sellable_quantity=D(1)),)
        self.service.on_account = changed
        self.arm()
        self.holdings()
        self.assertEqual(self.service.submitted, [])

    def test_fresh_quote_below_target_does_not_send(self):
        self.seed()
        self.service.prices = [D(102), D(100)]
        self.arm()
        self.holdings()
        self.assertEqual(self.service.submitted, [])

    def test_atomic_claim_refuses_new_pending_inserted_after_preflight(self):
        buy = self.seed()
        before = self.saved(buy)
        original_claim = self.store.claim
        def competing_claim(rule, price, now, **options):
            self.seed(side="sell", status="submitting", started=self.now)
            return original_claim(rule, price, now, **options)
        self.arm()
        with patch.object(self.store, "claim", side_effect=competing_claim):
            self.holdings()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.saved(buy), before)

    def test_final_guard_refuses_new_ambiguous_buy_after_claim(self):
        self.seed()
        self.service.before_send = lambda: self.seed(status="unknown", started=self.now)
        self.arm()
        self.holdings()
        self.assertEqual(self.service.submitted, [])
        self.assertIn("not_sent", {row["status"] for row in self.store.attempts()})
        self.assertIn("unknown", {row["status"] for row in self.store.attempts()})

    def test_final_guard_refuses_new_accepted_sell_after_claim(self):
        self.seed()
        self.service.before_send = lambda: self.seed(side="sell", status="accepted", started=self.now)
        self.arm()
        self.holdings()
        self.assertEqual(self.service.submitted, [])
        self.assertIn("not_sent", {row["status"] for row in self.store.attempts()})

    def test_final_guard_refuses_new_historical_buy_not_in_preflight_evidence(self):
        self.seed()
        self.service.before_send = lambda: self.seed(started=OLD)
        self.arm()
        self.holdings()
        self.assertEqual(self.service.submitted, [])
        self.assertIn("not_sent", {row["status"] for row in self.store.attempts()})

    def test_final_guard_rechecks_quote_age(self):
        self.seed()
        self.service.before_send = lambda: setattr(self, "now", self.now + timedelta(seconds=16))
        self.arm()
        self.holdings()
        self.assertEqual(self.service.submitted, [])
        self.assertIn("not_sent", {row["status"] for row in self.store.attempts()})

    def test_fresh_quote_does_not_hide_stale_orders_account_checks(self):
        self.seed()
        def delayed():
            if len(self.service.account_calls) == 2:
                self.now += timedelta(seconds=16)
        self.service.on_account = delayed
        self.arm()
        self.holdings()
        self.assertEqual(self.service.submitted, [])

    def test_slow_final_calendar_check_cannot_outlive_account_evidence(self):
        self.seed()
        sending = {"started": False}
        def account_delay():
            if len(self.service.account_calls) == 2:
                self.now += timedelta(seconds=10)
        self.service.on_account = account_delay
        def pacing_delay():
            self.now += timedelta(seconds=4)
            sending["started"] = True
        self.service.before_send = pacing_delay
        def slow_calendar(market, now):
            if sending["started"]:
                self.now += timedelta(seconds=2)
            return self.open and market is Market.US
        self.arm()
        with patch("dockdack.autotrade.regular_session", side_effect=slow_calendar):
            self.holdings()
        # Quote age is only six seconds, but account/open-order evidence is 16.
        self.assertEqual(self.service.submitted, [])

    def test_legacy_post_parent_checks_cannot_outlive_account_evidence(self):
        from dockdack.lstm30_gui import _LSTM30AutoTrader

        buy = self.seed()
        original = self.saved(buy)
        self.engine = _LSTM30AutoTrader(self.service, self.store, items=(self.item,),
                                       clock=lambda: self.now)
        self.engine.enable_holdings_exits = True
        self.engine.isolated_symbol_errors = True
        sending = {"started": False, "rejection_checks": 0}
        def account_delay():
            if len(self.service.account_calls) == 2:
                self.now += timedelta(seconds=10)
        self.service.on_account = account_delay
        def pacing_delay():
            self.now += timedelta(seconds=4)
            sending["started"] = True
        self.service.before_send = pacing_delay
        def slow_rejection_check(*args):
            if sending["started"]:
                sending["rejection_checks"] += 1
                if sending["rejection_checks"] == 2:
                    # This second lookup is after AutoTrader's final guard.
                    self.now += timedelta(seconds=2)
            return False
        self.arm()
        with patch("dockdack.lstm30_gui.rejected_today", side_effect=slow_rejection_check):
            self.holdings()
        self.assertEqual(sending["rejection_checks"], 2)
        self.assertEqual(self.now, NOW + timedelta(seconds=16))
        # The quote is six seconds old; only the inherited broker evidence aged out.
        self.assertEqual(self.service.submitted, [])
        self.assertIn("not_sent", {row["status"] for row in self.store.attempts()})
        self.assertEqual(self.saved(buy), original)

    def test_final_off_prevents_send(self):
        self.seed()
        self.arm()
        self.service.before_send = self.engine.disarm
        self.holdings()
        self.assertEqual(self.service.submitted, [])
        self.assertIn("not_sent", {row["status"] for row in self.store.attempts()})

    def test_final_stop_prevents_send(self):
        self.seed()
        self.arm()
        self.service.before_send = self.engine._stop.set
        self.holdings()
        self.assertEqual(self.service.submitted, [])
        self.assertIn("not_sent", {row["status"] for row in self.store.attempts()})

    def test_final_market_close_prevents_send(self):
        self.seed()
        self.arm()
        self.service.before_send = lambda: setattr(self, "open", False)
        self.holdings()
        self.assertEqual(self.service.submitted, [])

    def test_claim_flag_cannot_unlock_another_buy(self):
        self.seed()
        candidate = TriggerRule.create(self.item, kind="price_ge", side="buy", quantity=1,
                                       max_notional=D(10000), threshold=D(95))
        self.store.add_rule(candidate)
        self.assertFalse(self.store.claim(candidate, D(102), NOW, allow_historical_buys=True))

    def test_second_independent_sell_claim_is_still_blocked(self):
        self.seed()
        first, second = self.sell_rule(), self.sell_rule(threshold=102)
        other = WatchStore(self.store.path)
        self.assertTrue(self.store.claim(first, D(102), NOW, allow_historical_buys=True))
        self.assertFalse(other.claim(second, D(102), NOW, allow_historical_buys=True))

    def test_real_ledger_never_exempts_historical_buy(self):
        self.store = WatchStore(Path(self.temp.name) / "real-offline.sqlite3",
                                mode=TradingMode.REAL, storage_scope="a" * 64)
        self.store.save_item(self.item)
        self.seed()
        self.assertEqual(self.store.historical_buy_attempt_ids(self.item.id, NOW), frozenset())
        self.assertFalse(self.store.claim(self.sell_rule(), D(102), NOW, allow_historical_buys=True))

    def test_unconfirmed_prototype_buy_cannot_fall_back_to_aggregate_exit(self):
        buy = self.seed()
        payload = dict(signal_id="fixture-prototype", action="buy", strategy_id="mark1-prototype",
                       market="us", exchange="ND", symbol="AAPL")
        with self.store.connection() as db:
            db.execute("INSERT INTO external_signals VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                       ("mark1-prototype-demo-trigger", payload["signal_id"], json.dumps(payload), buy.id,
                        self.item.id, OLD.isoformat(), (OLD + timedelta(hours=1)).isoformat(),
                        "offline-export", OLD.isoformat(), "buy", "accepted"))
        self.engine.prototype_lots_enabled = True
        self.arm()
        self.holdings()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.attempts()[0]["status"], "accepted")


if __name__ == "__main__":
    unittest.main()
