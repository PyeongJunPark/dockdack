from __future__ import annotations

import importlib.util
import os
import time
import tempfile
from pathlib import Path
import unittest
from decimal import Decimal

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None
if HAS_QT:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QDialogButtonBox
    from dockdack.gui import OrderDialog, TradingWindow

from dockdack import AccountSnapshot, Market, OrderOutcomeUnknown, OrderRequest, OrderResult, OrderSide, Quote, TradingMode
from dockdack.gui_service import Instrument
from dockdack.watchlist import WatchStore


class FakeService:
    def __init__(self):
        self.submitted = []
        self.fail_quote = False
        self.fail_submit = False
        self.unknown_submit = False
        self.fail_account = False

    def resolve(self, symbol, exchange=""):
        return Instrument(Market.DOMESTIC, symbol, "KRX")

    def quote(self, instrument):
        if self.fail_quote:
            raise ValueError("시세 조회 실패")
        return Quote(Market.DOMESTIC, instrument.symbol, "삼성전자", "KRX", Decimal("251250"), "KRW", Decimal("250"), Decimal("0.1"))

    def prepare(self, instrument, side, quantity, kind, price):
        return OrderRequest(instrument.market, OrderSide(side), instrument.symbol, quantity, instrument.exchange,
                            "0", Decimal("251250") if kind == "current" else price)

    def submit(self, request):
        self.submitted.append(request)
        if self.unknown_submit:
            raise OrderOutcomeUnknown("응답의 주문번호를 확인할 수 없습니다.")
        if self.fail_submit:
            raise ValueError("모의투자 장종료")
        return OrderResult(True, TradingMode.DEMO, request, "123", "모의주문 접수")

    def account(self, instrument):
        if self.fail_account:
            raise ValueError("잔고 조회 실패")
        return AccountSnapshot(Market.DOMESTIC, "KRW", (), cash=Decimal("10000000"), available_to_order=Decimal("9900000"))

    def orders(self, instrument):
        return ()

    def executions(self, instrument):
        return ()


@unittest.skipUnless(HAS_QT, "Install the gui extra to run Qt tests")
class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.service = FakeService()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = WatchStore(Path(self.temp.name) / 'gui.sqlite3')
        self.window = TradingWindow(self.service, store=self.store)
        self.window.timer.stop()
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.wait_idle()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def wait_idle(self):
        deadline = time.monotonic() + 5
        while self.window._worker is not None:
            QTest.qWait(10)
            self.assertLess(time.monotonic(), deadline, "Worker did not finish")
        self.app.processEvents()

    def select(self):
        QTest.mouseClick(self.window.search_button, Qt.MouseButton.LeftButton)
        self.wait_idle()

    def test_query_enables_orders_and_shows_price(self):
        self.assertFalse(self.window.buy_button.isEnabled())
        self.select()
        self.assertEqual(self.window.price_label.text(), "251,250")
        self.assertTrue(self.window.buy_button.isEnabled())

    def test_failed_symbol_query_clears_old_order_context(self):
        self.select()
        self.service.fail_quote = True
        self.window.symbol_input.setText("000660")
        self.window.search()
        self.wait_idle()
        self.assertIsNone(self.window.instrument)
        self.assertFalse(self.window.buy_button.isEnabled())
        self.assertIn("실패", self.window.message.text())

    def test_cancel_confirmation_sends_no_order(self):
        self.select()
        self.window.confirm_order = lambda request: False
        QTest.mouseClick(self.window.buy_button, Qt.MouseButton.LeftButton)
        self.wait_idle()
        self.assertEqual(self.service.submitted, [])

    def test_double_click_does_not_duplicate_order(self):
        self.select()
        self.window.confirm_order = lambda request: True
        self.window.buy_button.click()
        self.window.buy_button.click()
        self.wait_idle()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(self.service.submitted[0].price, Decimal("251250"))

    def test_rejected_order_does_not_retry(self):
        self.select()
        self.service.fail_submit = True
        self.window.confirm_order = lambda request: True
        self.window.sell_button.click()
        self.wait_idle()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertIn("장종료", self.window.message.text())

    def test_unknown_order_does_not_show_acceptance_or_retry(self):
        self.select()
        self.service.unknown_submit = True
        self.window.confirm_order = lambda request: True
        self.window.buy_button.click()
        self.wait_idle()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertIn("접수 여부 확인 필요", self.window.message.text())
        self.assertIn("주문·체결 내역", self.window.message.text())
        messages = [self.window.activity.item(row, 1).text() for row in range(self.window.activity.rowCount())]
        self.assertFalse(any("모의 매수 접수" in message for message in messages))

    def test_account_errors_are_visible(self):
        self.select()
        self.service.fail_account = True
        self.window.refresh_account()
        self.wait_idle()
        self.assertEqual(self.window.cash_label.text(), "조회 실패")
        self.assertIn("일부 조회 실패", self.window.message.text())

    def test_confirmation_dialog_defaults_to_cancel(self):
        request = OrderRequest(Market.US, OrderSide.BUY, "AAPL", 1, "ND", "00", Decimal("329.49"))
        dialog = OrderDialog(request, self.window)
        self.assertTrue(dialog.buttons.button(QDialogButtonBox.StandardButton.Cancel).isDefault())
        self.assertFalse(dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).autoDefault())
        dialog.close()


if __name__ == "__main__":
    unittest.main()
