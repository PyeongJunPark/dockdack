from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from dockdack.exceptions import BrokerAPIError, OrderNotSent, OrderOutcomeUnknown
from dockdack.gui_service import Instrument
from dockdack.manual_orders import reconcile_manual_executions, submit_manual_order
from dockdack.models import Market, OrderExecution, OrderRequest, OrderResult, OrderSide, Quote, TradingMode
from dockdack.watchlist import TriggerRule, WatchItem, WatchStore


D = Decimal
NOW = datetime(2026, 9, 15, 1, tzinfo=timezone.utc)
INST = Instrument(Market.DOMESTIC, "005930", "KRX")


class FakeService:
    def __init__(self, mode=TradingMode.DEMO):
        self.mode = mode
        self.submitted = []
        self.error = None
        self.result_override = None
        self.on_submit = lambda: None
        self.permission_calls = 0
        self.permission_error_at = None
        self.quote_calls = 0
        self.quote_value = Quote(INST.market, INST.symbol, "삼성전자", INST.exchange, D(99), "KRW")

    def ensure_order_permission(self, instrument):
        self.permission_calls += 1
        if self.permission_error_at == self.permission_calls:
            raise OrderNotSent("권한 취소")

    def quote(self, instrument):
        self.quote_calls += 1
        return self.quote_value

    def submit(self, request):
        self.submitted.append(request)
        self.on_submit()
        if self.error:
            raise self.error
        return self.result_override or OrderResult(True, self.mode, request, "000123", "접수")


class ManualOrderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = WatchStore(Path(self.temp.name) / "orders.sqlite3")
        self.service = FakeService()
        self.request = OrderRequest(INST.market, OrderSide.BUY, INST.symbol, 1, INST.exchange, "0", D(100))
        self.time_patch = patch("dockdack.manual_orders.utc_now", return_value=NOW)
        self.time_patch.start()
        self.addCleanup(self.time_patch.stop)

    def submit(self, request=None, reference=None):
        return submit_manual_order(self.service, self.store, request or self.request, reference)

    def row(self):
        return self.store.order_history(limit=None)[0]

    def execution(self, **changes):
        row = OrderExecution("123", INST.symbol, "매수", "체결", D(1), D(1), D(0), D(100), D(99), "100000")
        return replace(row, **changes)

    def reconcile(self, *rows):
        return reconcile_manual_executions(self.store, INST, tuple(rows))

    def test_intent_is_committed_before_single_send_and_is_never_ready(self):
        def inspect():
            pending = self.store.attempts(pending_only=True)
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0]["status"], "submitting")
            self.assertTrue(pending[0]["rule_id"].startswith("manual-"))
            self.assertEqual(self.store.rules()[0].status, "submitting")
            self.assertIn("체결가 아님", self.store.events(category="order")[0]["message"])
        self.service.on_submit = inspect
        result = self.submit()
        self.assertEqual(result.order_number, "000123")
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(self.row()["status"], "accepted")
        self.assertIsNone(self.row()["fill_price"])
        self.assertEqual(self.store.items()[0].days, 30)

    def test_existing_watch_details_and_inactive_state_are_preserved(self):
        item = WatchItem(INST, "사용자 이름", 83)
        self.store.save_item(item)
        self.store.remove_item(item.id)
        self.submit()
        with self.store.connection() as db:
            row = db.execute("SELECT * FROM watchlist WHERE id=?", (item.id,)).fetchone()
        self.assertEqual((row["name"], row["days"], row["active"]), ("사용자 이름", 83, 0))

    def test_pending_attempt_blocks_second_manual_click(self):
        self.submit()
        with self.assertRaises(OrderNotSent):
            self.submit()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(len(self.store.attempts()), 1)

    def test_automatic_pending_attempt_also_blocks_manual_order(self):
        item = WatchItem(INST)
        self.store.save_item(item)
        rule = TriggerRule.create(item, "price_ge", "buy", 1, D(1000), D(90))
        self.store.add_rule(rule)
        self.store.claim(rule, D(100), NOW)
        with self.assertRaises(OrderNotSent):
            self.submit()
        self.assertEqual(self.service.submitted, [])

    def test_market_order_gets_reference_quote_but_does_not_invent_fill(self):
        self.submit(replace(self.request, order_type="3", price=None))
        self.assertEqual(self.service.quote_calls, 1)
        self.assertEqual(self.row()["reference_price"], "99")
        self.assertIsNone(self.row()["fill_price"])

    def test_explicit_market_reference_avoids_extra_quote(self):
        self.submit(replace(self.request, order_type="3", price=None), D(101))
        self.assertEqual(self.service.quote_calls, 0)
        self.assertEqual(self.row()["reference_price"], "101")

    def test_invalid_reference_or_mismatched_quote_never_claims_or_sends(self):
        for reference in (D(0), D(-1), D("NaN"), D("Infinity")):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                self.submit(reference=reference)
        self.service.quote_value = replace(self.service.quote_value, symbol="000660")
        with self.assertRaises(OrderNotSent):
            self.submit(replace(self.request, order_type="3", price=None))
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.attempts(), ())

    def test_mode_mismatch_never_sends_or_claims(self):
        self.service.mode = TradingMode.REAL
        with self.assertRaises(OrderNotSent):
            self.submit()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.attempts(), ())

    def real_service_store(self):
        self.service = FakeService(TradingMode.REAL)
        self.service.storage_scope = "a" * 64
        self.store = WatchStore(Path(self.temp.name) / "real.sqlite3", mode=TradingMode.REAL,
                                storage_scope=self.service.storage_scope)

    def test_real_order_records_only_in_matching_mode_and_credential_scope(self):
        demo_store = self.store
        self.real_service_store()
        self.submit()
        self.assertEqual(self.row()["status"], "accepted")
        self.assertEqual(demo_store.attempts(), ())

    def test_real_credential_scope_mismatch_never_claims_or_sends(self):
        self.real_service_store()
        self.service.storage_scope = "b" * 64
        with self.assertRaises(OrderNotSent):
            self.submit()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.attempts(), ())

    def test_real_without_permission_guard_is_blocked_even_with_matching_db(self):
        self.real_service_store()
        self.service.ensure_order_permission = None
        with self.assertRaises(OrderNotSent):
            self.submit()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.attempts(), ())

    def test_unconfigured_real_scope_cannot_send(self):
        self.service = FakeService(TradingMode.REAL)
        self.service.storage_scope = "unconfigured"
        self.store = WatchStore(Path(self.temp.name) / "real.sqlite3", mode=TradingMode.REAL)
        with self.assertRaises(OrderNotSent):
            self.submit()
        self.assertEqual(self.service.submitted, [])

    def test_permission_before_claim_and_revocation_after_claim(self):
        self.service.permission_error_at = 1
        with self.assertRaises(OrderNotSent):
            self.submit()
        self.assertEqual(self.store.attempts(), ())
        self.service.permission_calls = 0
        self.service.permission_error_at = 2
        with self.assertRaises(OrderNotSent):
            self.submit()
        self.assertEqual(self.row()["status"], "not_sent")
        self.assertEqual(self.service.submitted, [])

    def test_known_rejection_is_recorded_and_never_retried(self):
        self.service.error = BrokerAPIError("명시적 거절", status_code=200, return_code="1701")
        with self.assertRaises(BrokerAPIError):
            self.submit()
        self.assertEqual(self.row()["status"], "rejected")
        self.assertEqual(len(self.service.submitted), 1)

    def test_unknown_send_blocks_any_later_submission(self):
        self.service.error = TimeoutError("응답 불명")
        with self.assertRaises(TimeoutError):
            self.submit()
        self.assertEqual(self.row()["status"], "unknown")
        with self.assertRaises(OrderNotSent):
            self.submit()
        self.assertEqual(len(self.service.submitted), 1)

    def test_final_guard_denial_is_not_sent(self):
        self.service.error = OrderNotSent("paced guard stopped")
        with self.assertRaises(OrderNotSent):
            self.submit()
        self.assertEqual(self.row()["status"], "not_sent")

    def test_wrong_mode_ack_is_unknown(self):
        self.service.result_override = OrderResult(True, TradingMode.REAL, self.request, "123", "접수")
        with self.assertRaises(OrderOutcomeUnknown):
            self.submit()
        self.assertEqual(self.row()["status"], "unknown")

    def test_intent_event_failure_rolls_back_everything_before_send(self):
        with patch.object(self.store, "_insert_event", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.submit()
        self.assertEqual(self.store.attempts(), ())
        self.assertEqual(self.store.rules(), ())
        self.assertEqual(self.store.items(), ())
        self.assertEqual(self.service.submitted, [])

    def test_result_journal_failure_retains_blocking_intent(self):
        with patch.object(self.store, "finish", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.submit()
        self.assertEqual(self.row()["status"], "submitting")
        with self.assertRaises(OrderNotSent):
            self.submit()
        self.assertEqual(len(self.service.submitted), 1)

    def test_empty_execution_list_never_means_filled(self):
        self.submit()
        self.assertEqual(self.reconcile(), 0)
        self.assertEqual(self.row()["status"], "accepted")
        self.assertIsNone(self.row()["fill_price"])

    def test_exact_fill_stores_actual_price_not_order_reference(self):
        self.submit()
        self.assertEqual(self.reconcile(self.execution()), 1)
        self.assertEqual((self.row()["status"], self.row()["fill_price"]), ("filled", "99"))
        self.assertEqual(self.row()["reference_price"], "100")

    def test_wrong_identity_not_borrowed(self):
        self.submit()
        for changes in ({"order_number": "999"}, {"symbol": "000660"}):
            self.assertEqual(self.reconcile(self.execution(**changes)), 0)
        for changes in ({"side": "매도"}, {"side": "매수정정"}, {"order_quantity": D(2)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.reconcile(self.execution(**changes))
        self.assertEqual(self.row()["status"], "accepted")
        self.assertIsNone(self.row()["fill_price"])

    def test_duplicate_execution_rows_fail_closed(self):
        self.submit()
        with self.assertRaises(ValueError):
            self.reconcile(self.execution(), self.execution())
        self.assertIsNone(self.row()["fill_price"])

    def test_prior_day_order_number_is_not_matched_to_today(self):
        with patch("dockdack.manual_orders.utc_now", return_value=NOW-timedelta(days=1)):
            self.submit()
        self.assertEqual(self.reconcile(self.execution()), 0)
        self.assertEqual(self.row()["status"], "accepted")

    def test_missing_execution_price_remains_unknown_then_can_be_enriched(self):
        self.submit()
        self.reconcile(self.execution(fill_price=D(0)))
        self.assertEqual(self.row()["status"], "filled")
        self.assertIsNone(self.row()["fill_price"])
        self.reconcile(self.execution())
        self.assertEqual(self.row()["fill_price"], "99")

    def test_partial_fill_remains_accepted_and_explicit_cancel_is_terminal(self):
        self.submit(replace(self.request, quantity=2))
        partial = self.execution(order_quantity=D(2), remaining_quantity=D(1))
        self.reconcile(partial)
        self.assertEqual(self.row()["status"], "accepted")
        self.reconcile(replace(partial, remaining_quantity=D(0), status="취소확인"))
        self.assertEqual(self.row()["status"], "cancelled")
        self.assertEqual(self.row()["filled_quantity"], "1")

    def test_cancel_request_or_denial_is_not_terminal(self):
        self.submit()
        for status in ("취소요청", "취소거부", "접수"):
            self.reconcile(self.execution(filled_quantity=D(0), remaining_quantity=D(0), status=status))
            self.assertEqual(self.row()["status"], "accepted")
            self.assertIsNone(self.row()["fill_price"])

    def test_decreasing_cumulative_quantity_is_not_saved(self):
        self.submit(replace(self.request, quantity=2))
        self.reconcile(self.execution(order_quantity=D(2), remaining_quantity=D(1)))
        with self.assertRaises(ValueError):
            self.reconcile(self.execution(order_quantity=D(2), filled_quantity=D(0), remaining_quantity=D(2)))
        self.assertEqual(self.row()["filled_quantity"], "1")

    def test_non_manual_orders_are_never_reconciled(self):
        item = WatchItem(INST)
        self.store.save_item(item)
        rule = TriggerRule.create(item, "price_ge", "buy", 1, D(1000), D(90))
        self.store.add_rule(rule)
        self.store.claim(rule, D(100), NOW)
        self.store.finish(rule.id, "accepted", "자동 주문 접수", "000123")
        self.assertEqual(self.reconcile(self.execution()), 0)
        self.assertEqual(self.row()["status"], "accepted")


if __name__ == "__main__":
    unittest.main()
