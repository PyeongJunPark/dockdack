from __future__ import annotations

import importlib.util
import os
import unittest
from decimal import Decimal as D
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None
if HAS_QT:
    from PySide6.QtCore import QDate, QPoint
    from PySide6.QtWidgets import QApplication
    from dockdack.trade_journal_gui import DailyTradeJournalPanel

from test_trade_journal import order


@unittest.skipUnless(HAS_QT, "Install the gui extra")
class DailyTradeJournalGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.store = SimpleNamespace(mode="demo", order_history=Mock(return_value=()))
        self.panel = DailyTradeJournalPanel(self.store)
        self.panel.dates["domestic"].setDate(QDate(2026, 9, 15))
        self.panel.dates["us"].setDate(QDate(2026, 9, 14))

    def tearDown(self):
        self.panel.close()
        self.panel.deleteLater()
        self.app.processEvents()

    def test_market_tabs_have_independent_dates_rows_and_currencies(self):
        self.panel.refresh(records=[order(1), order(2, market="us", exchange="ND", symbol="AAPL", currency="USD")])
        self.assertEqual(self.panel.market_tabs.count(), 2)
        self.assertIn("국내", self.panel.market_tabs.tabText(0))
        self.assertIn("해외", self.panel.market_tabs.tabText(1))
        self.assertEqual(self.panel.values["domestic"]["buy"].text(), "100 KRW")
        self.assertEqual(self.panel.values["us"]["buy"].text(), "100.00 USD")
        self.assertEqual(self.panel.tables["domestic"].rowCount(), 1)
        self.assertEqual(self.panel.tables["us"].rowCount(), 1)
        self.panel.dates["us"].setDate(QDate(2026, 9, 15))
        self.assertEqual(self.panel.tables["us"].rowCount(), 0)
        self.assertEqual(self.panel.tables["domestic"].rowCount(), 1)
        self.store.order_history.assert_not_called()

    def test_known_profit_and_unknown_cost_are_visibly_distinct(self):
        self.panel.refresh(records=[order(1), order(2, "sell", "110"), order(3, "sell", "105", symbol="000001")])
        self.assertIn("+10 KRW", self.panel.values["domestic"]["profit"].text())
        self.assertIn("확인분만", self.panel.values["domestic"]["profit"].text())
        self.assertIn("미확인 1건", self.panel.summaries["domestic"].text())
        self.assertIn("계좌 전체 일수익률 아님", self.panel.values["domestic"]["return"].toolTip())
        self.assertEqual(self.panel.tables["domestic"].item(0, 6).text(), "미확인")

    def test_no_fill_does_not_show_reference_price_or_cash(self):
        self.panel.refresh(records=[order(1, status="accepted", filled_quantity=None, fill_price=None)])
        self.assertEqual(self.panel.values["domestic"]["buy"].text(), "0 KRW")
        self.assertEqual(self.panel.tables["domestic"].item(0, 5).text(), "—")
        self.assertIn("대기 1건", self.panel.summaries["domestic"].text())

    def test_revision_and_unrevisioned_reads_are_cached(self):
        self.store.order_history.return_value = (order(1),)
        self.assertTrue(self.panel.refresh(head=1))
        first = self.panel.tables["domestic"].item(0, 0)
        self.assertFalse(self.panel.refresh(head=1))
        self.assertFalse(self.panel.refresh())
        self.store.order_history.assert_called_once_with(limit=None)
        self.assertFalse(self.panel.refresh(head=2))
        self.assertEqual(self.store.order_history.call_count, 2)
        self.assertIs(self.panel.tables["domestic"].item(0, 0), first)

    def test_refresh_button_is_local_read_only_and_mode_is_visible(self):
        self.store.mode = SimpleNamespace(value="live")
        requested = Mock()
        self.panel.request_refresh.connect(requested)
        self.panel.refresh_button.click()
        requested.assert_called_once_with()
        self.store.order_history.assert_not_called()
        self.panel.refresh(records=())
        self.assertIn("REAL", self.panel.mode_badge.text())
        self.assertIn("API 조회나 매수·매도는 하지 않습니다", self.panel.refresh_button.toolTip())

    def test_missing_dates_are_visible_not_silently_aggregated(self):
        self.panel.refresh(records=[order(1, started_at="not-a-date")])
        self.assertIn("미확인 1건", self.panel.warning.text())
        self.assertEqual(self.panel.tables["domestic"].rowCount(), 0)

    def test_wide_layout_uses_four_cards_and_preserves_multiline_caveats(self):
        self.panel.resize(1280, 600)
        self.panel.show()
        self.panel.refresh(records=[order(1), order(2, "sell", "110"), order(3, "sell", "100", symbol="000001")])
        self.app.processEvents()
        self.assertEqual(self.panel._metric_columns, 4)
        grid = self.panel.metric_grids["domestic"]
        positions = [grid.getItemPosition(grid.indexOf(frame))[:2] for frame in self.panel.metric_cards["domestic"]]
        self.assertEqual(positions, [(0, 0), (0, 1), (0, 2), (0, 3)])
        for key in ("profit", "return"):
            value = self.panel.values["domestic"][key]
            self.assertIn("확인분만", value.text())
            self.assertGreaterEqual(value.height(), value.fontMetrics().lineSpacing() * 2)
        self.assertGreaterEqual(self.panel.tables["domestic"].height(), 190)
        self.store.order_history.assert_not_called()

    def test_narrow_short_layout_scrolls_instead_of_clipping_detail_rows(self):
        self.panel.resize(800, 500)
        self.panel.show()
        self.panel.refresh(records=[order(1), order(2, "sell", "110"), order(3, "sell", "100", symbol="000001")])
        self.app.processEvents()
        self.assertEqual(self.panel._metric_columns, 2)
        view = self.panel.tables["domestic"]
        scroll = self.panel.scroll_areas["domestic"]
        self.assertGreaterEqual(view.height(), 190)
        self.assertGreater(scroll.verticalScrollBar().maximum(), 0)
        scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
        self.app.processEvents()
        top = view.mapTo(scroll.viewport(), QPoint(0, 0)).y()
        self.assertGreaterEqual(top, 0)
        self.assertLessEqual(top + view.height(), scroll.viewport().height())
        self.store.order_history.assert_not_called()


if __name__ == "__main__":
    unittest.main()
