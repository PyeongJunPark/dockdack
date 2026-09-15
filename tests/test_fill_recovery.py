import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal as D
from pathlib import Path

from dockdack.fill_recovery import FillRecovery
from dockdack.gui_service import Instrument
from dockdack.history import market_time
from dockdack.models import ExecutionHistoryRecord, Market, OrderSide, TradingMode
from dockdack.watchlist import TriggerRule, WatchItem, WatchStore
from test_autotrade import FakeTradingService, NOW


class HistoryService(FakeTradingService):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.records = {}
        self.error = None

    def execution_history(self, market, day):
        self.calls.append((market, day))
        if self.error:
            raise self.error
        return self.records.get((market, day), ())


class FillRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = WatchStore(Path(self.temp.name) / "ledger.sqlite3")
        self.service = HistoryService()
        self.now = NOW
        self.recovery = FillRecovery(self.service, self.store, clock=lambda: self.now)
        self.serial = 0

    def order(self, status="filled", *, market=Market.DOMESTIC, day=None, exchange=None, quantity=1, side="buy", number=None):
        self.serial += 1
        inst = Instrument(market, "005930" if market is Market.DOMESTIC else "AAPL",
                          exchange or ("KRX" if market is Market.DOMESTIC else "ND"))
        item = WatchItem(inst)
        self.store.save_item(item)
        rule = TriggerRule.create(item, "price_ge", side, quantity, D(100000), D(90))
        self.store.add_rule(rule)
        self.assertTrue(self.store.claim(rule, D(999), day or NOW))
        number = number or f"{self.serial:07d}"
        if status in {"filled", "cancelled"}:
            self.store.finish(rule.id, "accepted", "saved acknowledgement", number)
        if status != "submitting":
            self.store.finish(rule.id, status, "saved status", number)
        return self.row(rule.id)

    def row(self, rule_id):
        return next(row for row in self.store.order_history(limit=None) if row["rule_id"] == rule_id)

    def record(self, row, **changes):
        market = Market(row["market"])
        day = market_time(market, datetime.fromisoformat(row["started_at"])).date()
        record = ExecutionHistoryRecord(market=market, order_date=day, order_number=row["order_number"],
                    symbol=row["symbol"], exchange=row["exchange"], side=OrderSide(row["side"]),
                    order_quantity=D(row["quantity"]), filled_quantity=D(row["quantity"]), remaining_quantity=D(0),
                    order_price=D(999), fill_price=D(101), reported_fill_price=D(101), price_basis="single_share",
                    order_time="090000", fill_time="090001", status="체결", currency=row["currency"],
                    source_api="kt00007" if market is Market.DOMESTIC else "ust21150")
        return replace(record, **changes)

    def supply(self, *records):
        for record in records:
            key = record.market, record.order_date
            self.service.records[key] = (*self.service.records.get(key, ()), record)

    def assert_no_orders(self):
        self.assertEqual(self.service.submitted, [])

    def test_recovers_legacy_fill_on_inactive_watchlist_and_keeps_attempt_state(self):
        row = self.order()
        self.store.remove_item(row["watch_id"])
        self.supply(self.record(row, order_number="1"))
        result = self.recovery.refresh_due()
        actual = self.row(row["rule_id"])
        self.assertEqual((result["enriched"], result["unresolved"]), (1, 0))
        self.assertEqual(actual["fill_price"], "101")
        self.assertEqual(actual["reference_price"], "999")
        self.assertEqual(actual["status"], "filled")
        self.assertEqual(actual["recovery_status"], "enriched")
        self.assertEqual(actual["recovery_price_basis"], "single_share")
        self.assertEqual(actual["recovery_source_api"], "kt00007")
        self.assertEqual(self.store.items(), ())
        self.assert_no_orders()

    def test_accepted_fill_evidence_does_not_unlock_order_gate(self):
        row = self.order("accepted")
        self.supply(self.record(row))
        self.recovery.refresh_due()
        actual = self.row(row["rule_id"])
        self.assertEqual(actual["status"], "accepted")
        self.assertEqual(actual["fill_price"], "101")
        self.assertEqual(len(self.store.attempts(pending_only=True)), 1)
        self.assertEqual(self.store.rules()[0].status, "accepted")
        self.assert_no_orders()

    def test_unknown_submitting_and_cancelled_without_fill_evidence_never_queried(self):
        self.order("cancelled")
        self.order("unknown")
        # Different venue avoids the same-symbol pending claim safety gate.
        self.order("submitting", market=Market.US)
        self.assertEqual(self.recovery.refresh_due()["checked_groups"], 0)
        self.assertEqual(self.service.calls, [])
        self.assert_no_orders()

    def test_missing_history_keeps_price_unknown_and_records_reason(self):
        row = self.order()
        result = self.recovery.refresh_due()
        actual = self.row(row["rule_id"])
        self.assertEqual(result["unresolved"], 1)
        self.assertIsNone(actual["fill_price"])
        self.assertEqual(actual["recovery_status"], "not_found")

    def test_zero_price_and_zero_filled_quantity_never_use_order_price(self):
        row = self.order("accepted")
        self.supply(self.record(row, fill_price=D(0), reported_fill_price=D(0), filled_quantity=D(0),
                                 remaining_quantity=D(1), price_basis="not_filled"))
        self.recovery.refresh_due()
        actual = self.row(row["rule_id"])
        self.assertIsNone(actual["fill_price"])
        self.assertEqual(actual["filled_quantity"], "0")
        self.assertEqual(actual["recovery_status"], "price_unknown")

    def test_partial_cancelled_multishare_reported_price_is_not_assumed_average(self):
        row = self.order("cancelled", quantity=2)
        self.store.record_execution(row["rule_id"], filled_quantity=D(1), remaining_quantity=D(0),
                                    fill_price=None, observed_at=self.now)
        self.supply(self.record(row, filled_quantity=D(1), fill_price=None, price_basis="unverified_multi_share"))
        self.recovery.refresh_due()
        actual = self.row(row["rule_id"])
        self.assertIsNone(actual["fill_price"])
        self.assertEqual(actual["recovery_reported_fill_price"], "101")
        self.assertEqual(actual["filled_quantity"], "1")
        self.assertEqual(actual["status"], "cancelled")

    def test_wrong_symbol_side_or_quantity_never_matches(self):
        row = self.order()
        self.supply(self.record(row, symbol="000660"), self.record(row, side=OrderSide.SELL),
                    self.record(row, order_quantity=D(2)))
        self.recovery.refresh_due()
        self.assertIsNone(self.row(row["rule_id"])["fill_price"])
        self.assertEqual(self.row(row["rule_id"])["recovery_status"], "not_found")

    def test_query_market_date_mismatch_is_error_and_keeps_history(self):
        row = self.order()
        record = self.record(row)
        self.service.records[(record.market, record.order_date)] = (replace(record, order_date=record.order_date-timedelta(days=1)),)
        self.assertEqual(self.recovery.refresh_due()["state"], "error")
        self.assertEqual(self.row(row["rule_id"])["recovery_status"], "error")
        self.assertIsNone(self.row(row["rule_id"])["fill_price"])

    def test_conflicting_multiple_records_fail_closed(self):
        row = self.order()
        self.supply(self.record(row), self.record(row, fill_price=D(102)))
        self.recovery.refresh_due()
        self.assertEqual(self.row(row["rule_id"])["recovery_status"], "ambiguous")
        self.assertIsNone(self.row(row["rule_id"])["fill_price"])

    def test_unknown_us_exchange_requires_unique_local_candidate(self):
        first = self.order(market=Market.US, exchange="ND", number="0001")
        second = self.order(market=Market.US, exchange="NY", number="1")
        self.supply(self.record(first, exchange=""))
        self.recovery.refresh_due()
        for row in (first, second):
            self.assertEqual(self.row(row["rule_id"])["recovery_status"], "ambiguous")
            self.assertIsNone(self.row(row["rule_id"])["fill_price"])

    def test_unique_us_order_keeps_price_unknown_without_independent_date_evidence(self):
        # 01:00 UTC is still the prior New York calendar day.
        for exchange in ("", "ND"):
            with self.subTest(exchange=exchange):
                row = self.order(market=Market.US)
                self.supply(self.record(row, exchange=exchange))
                result = self.recovery.refresh_due(force=True)
                self.assertEqual(self.service.calls[-1][1].isoformat(), "2026-09-13")
                actual = self.row(row["rule_id"])
                self.assertEqual(result["enriched"], 0)
                self.assertIsNone(actual["fill_price"])
                self.assertIsNone(actual["filled_quantity"])
                self.assertEqual(actual["recovery_status"], "ambiguous")
                self.assertEqual(actual["recovery_price_basis"], "date_scope_unverified")
                self.assertIn("시간대 근거 미확인", actual["recovery_message"])
                self.assertEqual(actual["recovery_reported_fill_price"], "101")
                self.assertEqual(actual["status"], "filled")
                self.now += timedelta(seconds=60)
        self.assert_no_orders()

    def test_us_adjacent_date_order_number_collision_cannot_create_fill_evidence(self):
        first = self.order(market=Market.US, day=NOW, number="0000042")
        second = self.order(market=Market.US, day=NOW + timedelta(days=1), number="42")
        # Each queried day returns the same order identity with a different
        # price. The adapter stamps only its QUERY date; there is no returned
        # broker date proving which New York order owns either price.
        self.supply(self.record(first, exchange="", fill_price=D(111), reported_fill_price=D(111)),
                    self.record(second, exchange="", fill_price=D(222), reported_fill_price=D(222)))
        result = self.recovery.refresh_due()
        self.assertEqual((result["checked_groups"], result["enriched"], result["unresolved"]), (2, 0, 2))
        for original in (first, second):
            actual = self.row(original["rule_id"])
            self.assertIsNone(actual["fill_price"])
            self.assertIsNone(actual["filled_quantity"])
            self.assertEqual(actual["recovery_price_basis"], "date_scope_unverified")
            self.assertEqual(actual["status"], "filled")
        with self.store.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM order_execution_snapshots").fetchone()[0], 0)
        self.assert_no_orders()

    def test_us_unverified_date_does_not_update_previously_confirmed_quantities(self):
        row = self.order("accepted", market=Market.US)
        self.store.record_execution(row["rule_id"], filled_quantity=D(0), remaining_quantity=D(1),
                                    fill_price=None, observed_at=self.now)
        self.supply(self.record(row))
        self.recovery.refresh_due()
        actual = self.row(row["rule_id"])
        self.assertEqual((actual["filled_quantity"], actual["remaining_quantity"]), ("0", "1"))
        self.assertIsNone(actual["fill_price"])
        self.assertEqual(actual["status"], "accepted")
        self.assertEqual(len(self.store.attempts(pending_only=True)), 1)
        self.assertEqual(actual["recovery_price_basis"], "date_scope_unverified")
        self.assert_no_orders()

    def test_known_exchange_mismatch_and_amendment_link_do_not_recover(self):
        row = self.order(market=Market.US)
        self.supply(self.record(row, exchange="NY"))
        self.recovery.refresh_due()
        self.assertEqual(self.row(row["rule_id"])["recovery_status"], "not_found")
        self.service.records.clear()
        self.supply(self.record(row, original_order_number="0000099"))
        self.now += timedelta(seconds=60)
        self.recovery.refresh_due(force=True)
        self.assertEqual(self.row(row["rule_id"])["recovery_status"], "ambiguous")

    def test_confirmed_filled_quantity_never_regresses_from_older_response(self):
        row = self.order()
        self.supply(self.record(row, filled_quantity=D(0), remaining_quantity=D(1), fill_price=None, price_basis="not_filled"))
        self.recovery.refresh_due()
        self.assertEqual(self.row(row["rule_id"])["recovery_status"], "quantity_conflict")
        self.assertIsNone(self.row(row["rule_id"])["filled_quantity"])
        self.assertEqual(self.row(row["rule_id"])["status"], "filled")

    def test_failed_requests_are_throttled_manual_never_bypasses_60_seconds(self):
        self.order()
        self.service.error = ValueError("read unavailable")
        self.recovery.refresh_due()
        self.now += timedelta(seconds=59)
        self.recovery.refresh_due(force=True)
        self.assertEqual(len(self.service.calls), 1)
        self.now += timedelta(seconds=1)
        self.recovery.refresh_due(force=True)
        self.assertEqual(len(self.service.calls), 2)
        self.now += timedelta(seconds=60)
        self.recovery.refresh_due()
        self.assertEqual(len(self.service.calls), 2)
        self.now += timedelta(seconds=240)
        self.recovery.refresh_due()
        self.assertEqual(len(self.service.calls), 3)
        self.assert_no_orders()

    def test_restart_preserves_saved_throttle_and_status_is_readonly(self):
        self.order()
        self.recovery.refresh_due()
        later = FillRecovery(self.service, self.store, clock=lambda: self.now)
        later.refresh_due(force=True)
        self.assertEqual(len(self.service.calls), 1)
        status = later.status()
        status["errors"]["injected"] = "not shared"
        self.assertEqual(later.status()["errors"], {})

    def test_at_most_four_groups_per_call_and_stopped_makes_no_calls(self):
        for days in range(5):
            self.order(day=NOW-timedelta(days=days))
        self.recovery.refresh_due(stopped=lambda: True)
        self.assertEqual(self.service.calls, [])
        self.assertEqual(self.recovery.refresh_due()["checked_groups"], 4)
        self.assertEqual(len(self.service.calls), 4)
        self.now += timedelta(seconds=60)
        self.assertEqual(self.recovery.refresh_due()["checked_groups"], 1)
        self.assertEqual(len(self.service.calls), 5)

    def test_record_execution_preserves_same_quantity_price_but_not_larger_unknown_average(self):
        row = self.order("accepted", quantity=2)
        def save(filled, remaining, price):
            self.store.record_execution(row["rule_id"], filled_quantity=D(filled), remaining_quantity=D(remaining),
                                        fill_price=None if price is None else D(price), observed_at=self.now)
        save(1, 1, 101)
        save(1, 1, None)
        self.assertEqual(self.row(row["rule_id"])["fill_price"], "101")
        with self.assertRaises(ValueError):
            save(0, 2, None)
        self.assertEqual(self.row(row["rule_id"])["filled_quantity"], "1")
        save(2, 0, None)
        self.assertIsNone(self.row(row["rule_id"])["fill_price"])

    def test_complete_ledger_includes_rows_older_than_the_ui_500_limit(self):
        old = self.order()
        with self.store.connection() as db:
            for index in range(501):
                rid = f"extra-{index}"
                db.execute("""INSERT INTO rules SELECT ?,watch_id,kind,side,quantity,max_notional,threshold,period,status
                              FROM rules WHERE id=?""", (rid, old["rule_id"]))
                # Rejected rows need no fill enrichment but still occupy UI history.
                db.execute("INSERT INTO attempts VALUES(?,?, 'rejected','999',?,?,'')",
                           (rid, old["watch_id"], (NOW+timedelta(seconds=index+1)).isoformat(), str(index+100)))
        self.assertEqual(len(self.store.order_history()), 500)
        all_rows = self.store.order_history(limit=None)
        self.assertEqual(len(all_rows), 502)
        self.assertEqual(all_rows[0]["rule_id"], old["rule_id"])
        self.supply(self.record(old))
        self.assertEqual(self.recovery.refresh_due()["enriched"], 1)
        self.assertEqual(self.row(old["rule_id"])["fill_price"], "101")

    def real_recovery(self, *, scope="a" * 64):
        self.service.mode = TradingMode.REAL
        self.service.storage_scope = scope
        self.store = WatchStore(Path(self.temp.name) / "real.sqlite3", mode=TradingMode.REAL, storage_scope=scope)
        self.recovery = FillRecovery(self.service, self.store, clock=lambda: self.now)

    def test_constructor_rejects_cross_mode_before_broker_or_ledger_write(self):
        row = self.order()
        before = self.store.order_history(limit=None)
        self.service.mode = TradingMode.REAL
        self.service.storage_scope = "a" * 64
        with self.assertRaisesRegex(ValueError, "모의/실전"):
            FillRecovery(self.service, self.store)
        self.assertEqual(self.service.calls, [])
        self.assertEqual(self.store.order_history(limit=None), before)
        self.assertIsNone(self.row(row["rule_id"])["fill_price"])

    def test_constructor_rejects_different_real_keys_even_for_same_mode(self):
        self.real_recovery()
        self.service.storage_scope = "b" * 64
        with self.assertRaisesRegex(ValueError, "계좌 키 범위"):
            FillRecovery(self.service, self.store)
        self.assertEqual(self.service.calls, [])

    def test_matching_unconfigured_real_can_render_but_cannot_refresh(self):
        self.real_recovery(scope="unconfigured")
        self.assertEqual(self.recovery.status()["state"], "idle")
        before = self.store.events()
        with self.assertRaisesRegex(ValueError, "미설정"):
            self.recovery.refresh_due(force=True)
        self.assertEqual(self.service.calls, [])
        self.assertEqual(self.store.events(), before)

    def test_real_matching_scope_recovers_using_fake_history_only(self):
        self.real_recovery()
        row = self.order()
        self.supply(self.record(row))
        self.assertEqual(self.recovery.refresh_due()["enriched"], 1)
        self.assertEqual(self.row(row["rule_id"])["fill_price"], "101")
        self.assert_no_orders()

    def test_changed_environment_after_construction_does_not_query_or_write(self):
        self.real_recovery()
        row = self.order()
        self.supply(self.record(row))
        before, events = self.store.order_history(limit=None), self.store.events()
        for scope in ("b" * 64, "unconfigured"):
            self.service.storage_scope = scope
            with self.assertRaises(ValueError):
                self.recovery.refresh_due(force=True)
            self.assertEqual(self.service.calls, [])
            self.assertEqual(self.store.order_history(limit=None), before)
            self.assertEqual(self.store.events(), events)
        self.service.storage_scope = "a" * 64
        self.service.mode = TradingMode.DEMO
        with self.assertRaises(ValueError):
            self.recovery.refresh_due(force=True)
        self.assertEqual(self.service.calls, [])

    def test_scope_change_during_history_response_does_not_persist_even_error(self):
        self.real_recovery()
        row = self.order()
        record = self.record(row)
        before, events = self.store.order_history(limit=None), self.store.events()
        def swapped_service(market, day):
            self.service.calls.append((market, day))
            self.service.storage_scope = "b" * 64
            return (record,)
        self.service.execution_history = swapped_service
        with self.assertRaisesRegex(ValueError, "계좌 키 범위"):
            self.recovery.refresh_due()
        self.assertEqual(len(self.service.calls), 1)
        self.assertEqual(self.store.order_history(limit=None), before)
        self.assertEqual(self.store.events(), events)
        self.assert_no_orders()


if __name__ == "__main__":
    unittest.main()
