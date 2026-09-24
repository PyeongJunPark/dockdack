"""Adversarial runtime regressions: fake broker reads, temp ledgers, no orders."""
from datetime import datetime, timedelta
from decimal import Decimal as D
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from dockdack import BrokerAPIError, Market
from dockdack.fill_recovery import FillRecovery
from dockdack.http import _READ_ONLY_APIS
from dockdack.mark1_trigger import Mark1TriggerBridge, SharedPrototypeAccounts
from dockdack.mark1_1_trigger import Mark11TriggerBridge
from dockdack.prototype_external import MODEL_IDS, PrototypeProcessClient
from dockdack.strategy_lots import project_prototype_inventory
import test_execution_history as history
import test_fill_recovery as recovery
import test_mark1_trigger as bridge
import test_strategy_lots as lots
from test_mark1_adapter import mark1_chart
from test_kiwoom import FakeResponse


def fill_row(**overrides):
    return {**history.domestic_row(ord_qty="3", cntr_qty="1", cntr_uv="100"),
            "cntr_no": "0000001", "cntr_tm": "13:05:44", "orig_ord_no": "0000000",
            **overrides}


class DomesticWeightedFillTests(unittest.TestCase):
    service = history.ExecutionHistoryTests.service

    def snapshot(self, **changes):
        return FakeResponse({"acnt_ord_cntr_prps_dtl": [
            history.domestic_row(ord_qty="3", cntr_qty="3", cntr_uv="103", **changes)]})

    def test_unique_fill_ids_and_all_pages_prove_average_not_last_price(self):
        a, b = fill_row(), fill_row(cntr_no="0000002", cntr_qty="2", cntr_uv="103", cntr_tm="13:05:45")
        service, _, transport = self.service(self.snapshot(),
            FakeResponse({"acnt_ord_cntr_prst_array": [a]}, headers={"cont-yn": "Y", "next-key": "next"}),
            FakeResponse({"acnt_ord_cntr_prst_array": [a, b]}))
        result, = service.execution_history(Market.DOMESTIC, history.DAY)
        self.assertEqual((result.fill_price, result.fill_amount, result.filled_quantity), (D(102), D(306), D(3)))
        self.assertEqual(result.reported_fill_price, D(103))
        self.assertEqual((result.source_api, result.price_basis, result.fill_time),
                         ("kt00007+kt00009", "weighted_fills", "13:05:45"))
        self.assertEqual(transport.calls[-1]["json"]["ord_dt"], "20260915")
        self.assertEqual(transport.calls[-1]["json"]["qry_tp"], "1")
        self.assertEqual([call["headers"]["api-id"] for call in transport.calls[1:]],
                         ["kt00007", "kt00009", "kt00009"])

    def test_missing_or_moving_fill_totals_never_guess_average(self):
        for events in ([], [fill_row()], [fill_row(cntr_qty="2"), fill_row(cntr_no="2", cntr_qty="2")]):
            with self.subTest(events=events):
                service, _, _ = self.service(self.snapshot(), FakeResponse({"acnt_ord_cntr_prst_array": events}))
                record, = service.execution_history(Market.DOMESTIC, history.DAY)
                self.assertIsNone(record.fill_price)
                self.assertEqual(record.price_basis, "unverified_multi_share")

    def test_partial_fill_proves_only_current_cumulative_quantity(self):
        service, _, _ = self.service(
            FakeResponse({"acnt_ord_cntr_prps_dtl": [history.domestic_row(ord_qty="3", cntr_qty="2", ord_remnq="1")]}),
            FakeResponse({"acnt_ord_cntr_prst_array": [fill_row(cntr_qty="2", cntr_uv="101")]}))
        record, = service.execution_history(Market.DOMESTIC, history.DAY)
        self.assertEqual((record.fill_price, record.filled_quantity, record.remaining_quantity), (D(101), D(2), D(1)))

    def test_missing_fill_id_conflicts_identity_and_nonfinite_are_rejected(self):
        cases = [[fill_row(cntr_no="")], [fill_row(cntr_no="0000000")],
                 [fill_row(), fill_row(cntr_uv="101")], [fill_row(stk_cd="A000660")],
                 [fill_row(dmst_stex_tp="NXT")], [fill_row(io_tp_nm="현금매도")],
                 [fill_row(cntr_uv="NaN")], [fill_row(cntr_qty="1.5")],
                 [fill_row(cntr_tm="99:00:00")]]
        for events in cases:
            with self.subTest(events=events):
                service, _, _ = self.service(self.snapshot(), FakeResponse({"acnt_ord_cntr_prst_array": events}))
                with self.assertRaises(ValueError):
                    service.execution_history(Market.DOMESTIC, history.DAY)

    def test_correction_chain_never_gets_an_invented_average(self):
        service, _, _ = self.service(self.snapshot(), FakeResponse({"acnt_ord_cntr_prst_array": [
            fill_row(cntr_qty="3", orig_ord_no="0000004")]}))
        record, = service.execution_history(Market.DOMESTIC, history.DAY)
        self.assertIsNone(record.fill_price)

    def test_incomplete_individual_fill_pagination_fails_closed(self):
        _, broker, _ = self.service(self.snapshot(), FakeResponse({"acnt_ord_cntr_prst_array": [fill_row(cntr_qty="3")]},
            headers={"cont-yn": "Y", "next-key": "more"}))
        with self.assertRaises(BrokerAPIError):
            broker.list_execution_history(Market.DOMESTIC, history.DAY, max_pages=1)

    def test_exact_individual_fill_read_only_route(self):
        self.assertIn(("kt00009", "/api/dostk/acnt"), _READ_ONLY_APIS)
        self.assertNotIn(("kt00009", "/api/dostk/ordr"), _READ_ONLY_APIS)


class WeightedRecoveryTests(unittest.TestCase):
    setUp = recovery.FillRecoveryTests.setUp
    order = recovery.FillRecoveryTests.order
    row = recovery.FillRecoveryTests.row
    record = recovery.FillRecoveryTests.record
    supply = recovery.FillRecoveryTests.supply

    def test_domestic_multi_share_evidence_is_persisted_and_bound(self):
        row = self.order(quantity=3)
        self.supply(self.record(row, source_api="kt00007+kt00009", price_basis="weighted_fills",
                               fill_price=D(102), reported_fill_price=D(103), fill_amount=D(306)))
        self.recovery.refresh_due(force=True)
        actual = self.row(row["rule_id"])
        self.assertEqual((actual["recovery_status"], actual["fill_price"]), ("enriched", "102"))
        self.assertEqual((actual["price_basis_quantity"], actual["price_basis_price"]), ("3", "102"))
        self.assertEqual(actual["recovery_price_basis"], "weighted_fills")

    def test_domestic_last_price_or_wrong_evidence_source_does_not_qualify(self):
        for source, amount in (("kt00007", D(306)), ("kt00007+kt00009", D(309))):
            row = self.order(quantity=3)
            self.supply(self.record(row, source_api=source, price_basis="weighted_fills", fill_price=D(102), fill_amount=amount))
            self.now += timedelta(seconds=301)
            self.recovery.refresh_due(force=True)
            self.assertIsNone(self.row(row["rule_id"])["fill_price"])

    def test_demo_credential_scope_change_blocks_reads_and_writes(self):
        self.service.storage_scope = "demo-key-a"
        scoped_store = recovery.WatchStore(self.store.path.with_name("scoped.sqlite3"), storage_scope="demo-key-a")
        checker = FillRecovery(self.service, scoped_store, clock=lambda: self.now)
        self.service.storage_scope = "demo-key-b"
        with self.assertRaisesRegex(ValueError, "계좌 키 범위"):
            checker.refresh_due(force=True)
        self.assertEqual(self.service.calls, [])


class SharedSignalAccountTests(unittest.TestCase):
    setUp = bridge.BridgeTests.setUp

    def bridges(self):
        shared = SharedPrototypeAccounts(self.window)
        return [kind(self.window, predictors={}, account_snapshots=shared)
                for kind in (Mark1TriggerBridge, Mark11TriggerBridge)]

    def test_two_models_one_account_read_per_market_and_expiry(self):
        first, second = self.bridges()
        stock = mark1_chart()["stocks"][0]
        self.assertEqual(first._position(stock), second._position(stock))
        self.assertEqual(self.service.safety_account.call_count, 1)
        now = self.engine.clock()
        self.engine.clock = lambda: now + timedelta(seconds=10)
        second._position(stock)
        first._position(stock)
        self.assertEqual(self.service.safety_account.call_count, 2)
        first._position(mark1_chart(market="us")["stocks"][0])
        second._position(mark1_chart(market="us")["stocks"][0])
        self.assertEqual(self.service.safety_account.call_count, 3)
        self.service.submit.assert_not_called()

    def test_mode_scope_service_or_store_change_cannot_reuse_snapshots(self):
        first, second = self.bridges()
        stock = mark1_chart()["stocks"][0]
        first._position(stock)
        self.service.storage_scope = "different-key"
        with self.assertRaises(ValueError):
            second._position(stock)
        self.assertEqual(self.service.safety_account.call_count, 1)

    def test_changed_scope_during_read_is_not_cached(self):
        first, _ = self.bridges()
        original = self.service.safety_account.side_effect
        def change(inst):
            result = original(inst)
            self.service.storage_scope = "different-key"
            return result
        self.service.safety_account.side_effect = change
        with self.assertRaises(ValueError):
            first._position(mark1_chart()["stocks"][0])
        self.assertEqual(first.account_snapshots.accounts, {})

    def test_bad_or_failed_account_is_not_cached(self):
        first, second = self.bridges()
        stock = mark1_chart()["stocks"][0]
        original = self.service.safety_account.side_effect
        self.service.safety_account.side_effect = ValueError("fake unavailable")
        with self.assertRaises(ValueError):
            first._position(stock)
        self.service.safety_account.side_effect = original
        second._position(stock)
        self.assertEqual(self.service.safety_account.call_count, 2)

    def test_different_window_service_or_store_cannot_reuse_cache(self):
        first, second = self.bridges()
        stock = mark1_chart()["stocks"][0]
        first._position(stock)
        other = SimpleNamespace(service=self.service, store=object(), engine=self.engine)
        with self.assertRaises(ValueError):
            first.account_snapshots.get(stock, window=other)
        self.assertEqual(self.service.safety_account.call_count, 1)

    def test_concurrent_model_reads_share_one_inflight_account_read(self):
        first, second = self.bridges()
        stock = mark1_chart()["stocks"][0]
        original = self.service.safety_account.side_effect
        entered, release = threading.Event(), threading.Event()
        def slow_read(inst):
            entered.set()
            release.wait(2)
            return original(inst)
        self.service.safety_account.side_effect = slow_read
        results = []
        workers = [threading.Thread(target=lambda current=current: results.append(current._position(stock)))
                   for current in (first, second)]
        self.addCleanup(release.set)
        for worker in workers:
            worker.start()
        self.assertTrue(entered.wait(.5))
        release.set()
        for worker in workers:
            worker.join(1)
            self.assertFalse(worker.is_alive())
        self.assertEqual(len(results), 2)
        self.assertEqual(self.service.safety_account.call_count, 1)


class RpcDeadlineTests(unittest.TestCase):
    def process(self, writer=None, waiter=None):
        return SimpleNamespace(poll=Mock(return_value=None), terminate=Mock(), kill=Mock(), wait=waiter or Mock(),
            stdin=SimpleNamespace(write=writer or Mock(), flush=Mock(), close=Mock()),
            stdout=SimpleNamespace(readline=Mock(return_value=""), close=Mock()))

    def test_blocked_pipe_write_obeys_same_deadline_and_close_is_nonblocking(self):
        release, entered, terminated = threading.Event(), threading.Event(), threading.Event()
        def writer(line):
            entered.set()
            release.wait(2)
        client = PrototypeProcessClient(MODEL_IDS[0], timeout=.08)
        process = self.process(writer, waiter=lambda **kw: release.wait(2))
        process.terminate.side_effect = terminated.set
        client.process = process
        self.addCleanup(release.set)
        before = time.monotonic()
        with self.assertRaises(TimeoutError):
            client.request("health")
        elapsed = time.monotonic() - before
        self.assertTrue(entered.is_set())
        self.assertLess(elapsed, .35)
        self.assertTrue(terminated.wait(.5))
        self.assertFalse(client.is_alive)
        before = time.monotonic()
        client.close()
        self.assertLess(time.monotonic() - before, .1)
        release.set()

    def test_late_spawn_after_deadline_is_disposed_never_attached(self):
        release, entered, terminated = threading.Event(), threading.Event(), threading.Event()
        process = self.process()
        process.terminate.side_effect = terminated.set
        def spawn(*args, **kwargs):
            entered.set()
            release.wait(2)
            return process
        client = PrototypeProcessClient(MODEL_IDS[0], timeout=.08)
        self.addCleanup(release.set)
        with patch("dockdack.prototype_external.subprocess.Popen", side_effect=spawn):
            before = time.monotonic()
            with self.assertRaises(TimeoutError):
                client.request("health")
            self.assertLess(time.monotonic() - before, .35)
            self.assertTrue(entered.is_set())
            release.set()
            self.assertTrue(terminated.wait(.5))
        self.assertIsNone(client.process)

    def test_close_interrupts_active_response_without_waiting_for_rpc(self):
        entered, finished = threading.Event(), threading.Event()
        client = PrototypeProcessClient(MODEL_IDS[0], timeout=10)
        client.process = self.process(writer=lambda line: entered.set())
        errors = []
        def request():
            try:
                client.request("health")
            except Exception as exc:
                errors.append(exc)
            finally:
                finished.set()
        worker = threading.Thread(target=request, daemon=True)
        worker.start()
        self.assertTrue(entered.wait(.5))
        before = time.monotonic()
        client.close()
        self.assertLess(time.monotonic() - before, .1)
        self.assertTrue(finished.wait(.5))
        self.assertIsInstance(errors[0], ValueError)
        self.assertFalse(client.is_alive)

    def test_engine_stop_cancels_active_request(self):
        entered = threading.Event()
        client = PrototypeProcessClient(MODEL_IDS[0], timeout=10)
        client.cancel_event = threading.Event()
        client.process = self.process(writer=lambda line: (entered.set(), client.cancel_event.set()))
        before = time.monotonic()
        with self.assertRaisesRegex(ValueError, "cancelled"):
            client.request("health")
        self.assertLess(time.monotonic() - before, .3)
        self.assertFalse(client.is_alive)

    def test_start_write_and_response_do_not_each_receive_new_budgets(self):
        client = PrototypeProcessClient(MODEL_IDS[0], timeout=.1)
        def writer(line):
            time.sleep(.07)
        client.process = self.process(writer)
        before = time.monotonic()
        with self.assertRaises(TimeoutError):
            client.request("health")
        self.assertLess(time.monotonic() - before, .16)

    def test_serialization_queue_has_deadline_and_does_not_kill_active_request(self):
        client = PrototypeProcessClient(MODEL_IDS[0], timeout=.05)
        process = self.process()
        client.process = process
        client._request_lock.acquire()
        try:
            with self.assertRaises(TimeoutError):
                client.request("health")
            self.assertIs(client.process, process)
            process.terminate.assert_not_called()
        finally:
            client._request_lock.release()
            client.close()


class BulkCloseProjectionTests(unittest.TestCase):
    setUp = lots.StrategyLotTests.setUp
    tearDown = lots.StrategyLotTests.tearDown
    rule = lots.StrategyLotTests.rule
    accept = lots.StrategyLotTests.accept
    fill = lots.StrategyLotTests.fill
    buy = lots.StrategyLotTests.buy

    def test_domestic_weighted_recovery_enables_existing_multi_share_lot_barriers(self):
        rule = self.rule(lots.OLD, 3)
        self.accept(rule)
        with self.store.connection() as db:
            db.execute("UPDATE attempts SET order_number='0000001' WHERE rule_id=?", (rule.id,))
        self.fill(rule, average="103", verified=False)
        self.assertFalse(self.store.prototype_inventory(self.item.id, D(3), D(3))["reconciled"])
        row = self.store.order_history(limit=None)[0]
        record = recovery.FillRecoveryTests.record(self, row, source_api="kt00007+kt00009",
            price_basis="weighted_fills", fill_price=D(102), reported_fill_price=D(103), fill_amount=D(306))
        service = recovery.HistoryService()
        service.records[(record.market, record.order_date)] = (record,)
        FillRecovery(service, self.store, clock=lambda: lots.NOW).refresh_due(force=True)
        inventory = self.store.prototype_inventory(self.item.id, D(3), D(3))
        self.assertTrue(inventory["reconciled"], inventory["issues"])
        lot, = inventory["lots"]
        self.assertEqual((lot["average_price"], lot["take_profit_price"], lot["stop_loss_price"]),
                         (D(102), D("103.02"), D("101.082")))

    def fixture(self, filled, status="accepted", *, order_quantity=6):
        a, b = self.buy(lots.OLD, 2, "100"), self.buy(lots.NEW, 3, "110")
        rows = list(self.store.order_history(limit=None))
        rows.append({**rows[0], "rule_id": "bulk", "side": "sell", "quantity": order_quantity,
                     "started_at": (max(datetime.fromisoformat(row["started_at"]) for row in rows) + timedelta(seconds=1)).isoformat(),
                     "filled_quantity": str(filled) if filled is not None else None, "status": status})
        allocations = [dict(rule_id="bulk", lot_id=a.id, quantity="2", fill_offset="0", bulk=1, order_quantity=order_quantity),
                       dict(rule_id="bulk", lot_id=b.id, quantity="3", fill_offset="2", bulk=1, order_quantity=order_quantity)]
        return rows, allocations

    def test_partial_and_complete_bulk_fills_replay_across_two_models_and_manual_residual(self):
        rows, allocations = self.fixture(3)
        result = project_prototype_inventory(rows, allocations)
        first, second = sorted(result["lots"], key=lambda row: row["buy_started_at"])
        self.assertEqual(result["issues"], ())
        self.assertEqual((first["quantity_sold"], second["quantity_sold"]), (D(2), D(1)))
        self.assertEqual((first["quantity_reserved_sell"], second["quantity_reserved_sell"]), (D(0), D(2)))
        self.assertEqual(project_prototype_inventory(rows, allocations), result)
        rows[-1] = {**rows[-1], "filled_quantity": "6", "status": "filled"}
        complete = project_prototype_inventory(rows, allocations)
        self.assertEqual(complete["issues"], ())
        self.assertEqual(complete["expected_quantity"], D(0))
        self.assertEqual(sum(row["quantity_sold"] for row in complete["lots"]), D(5))

    def test_cancelled_partial_bulk_releases_only_unsold_reservations(self):
        rows, allocations = self.fixture(3, "cancelled")
        result = project_prototype_inventory(rows, allocations)
        self.assertEqual(result["issues"], ())
        self.assertEqual(result["expected_quantity"], D(2))
        self.assertEqual(sum(row["available_quantity"] for row in result["lots"]), D(2))

    def test_bulk_full_status_validates_original_order_quantity_not_lot_size(self):
        rows, allocations = self.fixture(5, "filled")
        result = project_prototype_inventory(rows, allocations)
        self.assertTrue(any("전체 체결 상태" in issue for issue in result["issues"]))

    def test_bulk_missing_fill_evidence_overlapping_ranges_and_wrong_totals_are_blocked(self):
        rows, allocations = self.fixture(None, "filled")
        self.assertTrue(project_prototype_inventory(rows, allocations)["issues"])
        rows[-1] = {**rows[-1], "filled_quantity": "3", "status": "accepted"}
        for change in ({"fill_offset": "1"}, {"fill_offset": "4"}, {"fill_offset": "0.5"}, {"order_quantity": 5}):
            with self.subTest(change=change):
                bad = [allocations[0], {**allocations[1], **change}]
                self.assertTrue(project_prototype_inventory(rows, bad)["issues"])


if __name__ == "__main__":
    unittest.main()
