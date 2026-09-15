from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from datetime import timedelta
from decimal import Decimal as D
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None
if HAS_QT:
    from PySide6.QtWidgets import QApplication
    from dockdack.operations_gui import OperationsPanel, OrderHistoryPanel

from dockdack.watchlist import TriggerRule, WatchItem, WatchStore
from dockdack.fill_recovery import FillRecovery
from dockdack.history import market_time
from dockdack.models import ExecutionHistoryRecord, Market, OrderSide
from unittest.mock import Mock
from test_autotrade import FakeTradingService, NOW


@unittest.skipUnless(HAS_QT, "Install the gui extra")
class OperationsGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = WatchStore(Path(self.temp.name) / "operations.sqlite3")
        self.service = FakeTradingService()
        self.orders = OrderHistoryPanel()
        self.operations = OperationsPanel()
        self.serial = 0

    def tearDown(self):
        for panel in (self.orders, self.operations):
            panel.close()
            panel.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def order(self, status, *, side="buy", quantity=1, symbol=None, filled=None, remaining=None, price=None):
        self.serial += 1
        symbol = symbol or f"{self.serial:06d}"
        item = WatchItem(self.service.resolve(symbol), f"종목 {self.serial}")
        self.store.save_item(item)
        rule = TriggerRule.create(item, "price_ge", side, quantity, D("10000"), D("95"))
        self.store.add_rule(rule)
        self.assertTrue(self.store.claim(rule, D("123"), NOW + timedelta(seconds=self.serial)))
        if status in {"filled", "cancelled"}:
            self.store.finish(rule.id, "accepted", "증권사 접수", str(self.serial))
        if status != "submitting":
            self.store.finish(rule.id, status, f"증권사 상태 {status}", str(self.serial))
        if filled is not None:
            self.store.record_execution(rule.id, filled_quantity=D(filled), remaining_quantity=D(remaining),
                                        fill_price=None if price is None else D(price), observed_at=NOW)
        return rule

    def select_filter(self, key):
        self.orders.filter.setCurrentIndex(self.orders.filter.findData(key))

    def test_hold_flood_goes_only_to_signal_log_and_never_creates_order_rows(self):
        self.store.event("SYSTEM", "서버 시작", category="system")
        self.store.event("domestic:KRX:005930", "시세/차트 조회 완료", category="monitor")
        for index in range(510):
            self.store.event("domestic:KRX:005930", f"외부 신호 HOLD · 주문 생성 없음 {index}", category="signal")
        self.orders.reload(self.store)
        self.operations.reload(self.store)
        self.assertEqual(self.orders.table.rowCount(), 0)
        self.assertEqual(self.orders.audit.table.rowCount(), 0)
        self.assertIn("매수 0건 / 매도 0건", self.orders.summary.text())
        self.assertIn("신호만 수신해도", self.orders.count_label.text())
        self.assertEqual(self.operations.logs["system"].table.rowCount(), 1)
        self.assertEqual(self.operations.logs["monitor"].table.rowCount(), 1)
        self.assertEqual(self.operations.logs["signal"].table.rowCount(), 500)
        self.assertIn("서버 시작", self.operations.logs["system"].table.item(0, 2).text())
        self.assertEqual(self.service.submitted, [])

    def test_accepted_is_not_filled_and_legacy_fill_price_is_explicitly_unknown(self):
        self.order("filled")
        self.order("accepted", side="sell")
        self.orders.reload(self.store)
        self.assertIn("매수 1건 / 매도 0건", self.orders.summary.text())
        self.assertIn("접수·확인 대기 1건", self.orders.summary.text())
        self.assertIn("접수", self.orders.table.item(0, 7).text())
        self.assertEqual(self.orders.table.item(0, 5).text(), "—")
        self.assertEqual(self.orders.table.item(1, 5).text(), "1")
        self.assertEqual(self.orders.table.item(1, 6).text(), "0")
        self.assertEqual(self.orders.table.item(1, 8).text(), "— (미확인)")
        self.assertIn("123 KRW (체결가 아님)", self.orders.table.item(1, 7).toolTip())
        self.select_filter("filled")
        self.assertEqual(self.orders.table.rowCount(), 1)
        self.assertEqual(self.orders.table.item(0, 3).text(), "매수")

    def test_cancelled_partial_fill_remains_in_fill_history_with_confirmed_quantity(self):
        self.order("cancelled", side="sell", quantity=2, filled="1", remaining="0", price="101")
        self.orders.reload(self.store)
        self.assertIn("매수 0건 / 매도 1건", self.orders.summary.text())
        self.assertIn("접수·확인 대기 0건", self.orders.summary.text())
        self.assertEqual(self.orders.table.item(0, 4).text(), "2")
        self.assertEqual(self.orders.table.item(0, 5).text(), "1")
        self.assertIn("취소", self.orders.table.item(0, 7).text())
        self.assertEqual(self.orders.table.item(0, 8).text(), "— (미확인)")
        self.assertIn("원본 응답 가격 101", self.orders.table.item(0, 8).toolTip())
        self.assertEqual(self.orders.table.item(0, 3).foreground().color().name(), "#7aa2ff")
        self.select_filter("filled")
        self.assertEqual(self.orders.table.rowCount(), 1)
        self.select_filter("pending")
        self.assertEqual(self.orders.table.rowCount(), 0)

    def test_status_filter_uses_order_state_and_fill_evidence_not_message_words(self):
        self.order("filled")
        self.order("accepted", side="sell", quantity=2, filled="1", remaining="1", price="100")
        self.order("unknown")
        self.order("submitting")
        self.order("cancelled")
        self.order("rejected")
        self.order("not_sent")
        self.store.event("SYSTEM", "BUY SELL 체결 확인처럼 보이는 신호 문구", category="signal")
        self.orders.reload(self.store)
        for key, expected in (("all", 7), ("filled", 2), ("pending", 3), ("other", 3)):
            with self.subTest(key=key):
                self.select_filter(key)
                self.assertEqual(self.orders.table.rowCount(), expected)
        self.assertIn("매수 1건 / 매도 1건", self.orders.summary.text())

    def test_us_partial_fill_requires_price_evidence_and_has_distinct_buy_color(self):
        self.order("accepted", symbol="AAPL", quantity=2, filled="1", remaining="1", price="123.4567")
        self.orders.reload(self.store)
        self.assertEqual(self.orders.table.item(0, 2).text(), "미국")
        self.assertEqual(self.orders.table.item(0, 3).text(), "매수")
        self.assertEqual(self.orders.table.item(0, 3).foreground().color().name(), "#ed7892")
        self.assertIn("부분체결", self.orders.table.item(0, 7).text())
        self.assertEqual(self.orders.table.item(0, 8).text(), "— (미확인)")
        self.assertIn("123.4567", self.orders.table.item(0, 8).toolTip())

    def test_confirmed_single_share_prices_and_realized_profit_are_visible(self):
        self.order("filled", symbol="AAPL", filled="1", remaining="0", price="100.1234")
        self.order("filled", side="sell", symbol="AAPL", filled="1", remaining="0", price="101.1234")
        self.orders.reload(self.store)
        self.assertEqual(self.orders.table.item(0, 8).text(), "101.1234 USD")
        self.assertEqual(self.orders.table.item(0, 10).text(), "100.12")
        self.assertEqual(self.orders.table.item(0, 11).text(), "1.00")
        self.assertEqual(self.orders.table.item(0, 12).text(), "+1.00%")
        self.assertIn("+1.00 USD", self.orders.performance_label.text())

    def test_sell_without_buy_history_is_unknown_not_zero(self):
        self.order("filled", side="sell", filled="1", remaining="0", price="101")
        self.orders.reload(self.store)
        self.assertEqual(self.orders.table.item(0, 8).text(), "101 KRW")
        self.assertEqual(self.orders.table.item(0, 11).text(), "미확인")
        self.assertIn("미확인", self.orders.performance_label.text())

    def test_manual_recovery_button_only_emits_readonly_request(self):
        requests = []
        self.orders.request_refresh.connect(lambda: requests.append(True))
        self.orders.refresh_button.click()
        self.assertEqual(requests, [True])
        self.assertFalse(self.service.submitted)
        self.orders.set_recovery_status({"state": "running", "message": "조회 중"})
        self.assertFalse(self.orders.refresh_button.isEnabled())
        self.orders.set_recovery_status({"state": "ok", "message": "완료"})
        self.assertTrue(self.orders.refresh_button.isEnabled())

    def test_recovered_legacy_prices_flow_into_visible_realized_profit(self):
        buy = self.order("filled", symbol="005930")
        sell = self.order("filled", side="sell", symbol="005930")
        day = market_time(Market.DOMESTIC, NOW).date()
        records = tuple(ExecutionHistoryRecord(
            market=Market.DOMESTIC, order_date=day, order_number=str(index),
            symbol="005930", exchange="KRX", side=side, order_quantity=D(1),
            filled_quantity=D(1), remaining_quantity=D(0), order_price=None,
            fill_price=price, reported_fill_price=price, price_basis="single_share",
            order_time="090001", fill_time="090002", status="체결", currency="KRW", source_api="kt00007",
        ) for index, side, price in ((1, OrderSide.BUY, D(100)), (2, OrderSide.SELL, D(101))))
        self.service.execution_history = Mock(return_value=records)
        self.orders.reload(self.store)
        self.assertEqual(self.orders.table.item(0, 11).text(), "미확인")
        result = FillRecovery(self.service, self.store, clock=lambda: NOW).refresh_due()
        self.orders.set_recovery_status(result)
        self.orders.reload(self.store)
        self.assertEqual(result["enriched"], 2)
        self.assertEqual(self.orders.table.item(0, 8).text(), "101 KRW")
        self.assertEqual(self.orders.table.item(0, 11).text(), "1")
        self.assertEqual(self.orders.table.item(0, 12).text(), "+1.00%")
        self.assertIn("kt00007", self.orders.table.item(0, 8).toolTip())
        self.assertEqual({r["rule_id"]: r["status"] for r in self.store.order_history()}, {buy.id: "filled", sell.id: "filled"})
        self.assertFalse(self.service.submitted)

    def test_signal_only_updates_do_not_rebuild_or_replace_order_cells(self):
        self.order("filled", filled="1", remaining="0", price="101")
        self.orders.reload(self.store)
        before = self.orders.table.item(0, 0)
        records = self.orders.records
        self.store.event("domestic:KRX:005930", "외부 신호 HOLD · 주문 생성 없음", category="signal")
        self.orders.reload(self.store)
        self.assertEqual(self.orders.records, records)
        self.assertIs(self.orders.table.item(0, 0), before)
        self.assertEqual(self.orders.table.rowCount(), 1)


if __name__ == "__main__":
    unittest.main()
