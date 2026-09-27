from __future__ import annotations

import importlib.util
import os
import threading
import time
import unittest
from datetime import date, timedelta
from decimal import Decimal as D
from types import SimpleNamespace
from unittest.mock import Mock

from dockdack.fx_reference import UsdKrwReference

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None
if HAS_QT:
    from PySide6.QtCore import QDate, QPoint
    from PySide6.QtWidgets import QApplication
    from dockdack.trade_journal_gui import DailyTradeJournalPanel

from test_trade_journal import START, order


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

    def wait_for_fx(self):
        deadline = time.monotonic() + 3
        while self.panel._fx_workers and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.app.processEvents()
        self.assertFalse(self.panel._fx_workers, "FX display worker did not finish")

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
        self.assertIn("계좌 전체 기간 수익률 아님", self.panel.values["domestic"]["return"].toolTip())
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
        self.assertEqual(self.panel.mode_badge.text(), "실전")
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

    def test_month_and_year_controls_filter_ledger_and_navigate_each_market_independently(self):
        records = [
            order(1),
            order(2, "sell", "110", started_at=(START + timedelta(days=1)).isoformat()),
            order(3, started_at="2026-10-02T00:00:00+00:00"),
            order(4, market="us", exchange="ND", symbol="AAPL", currency="USD"),
        ]
        self.panel.refresh(records=records)
        self.panel.periods["domestic"].setCurrentIndex(1)
        self.assertEqual(self.panel.dates["domestic"].displayFormat(), "yyyy-MM")
        self.assertEqual(self.panel.tables["domestic"].rowCount(), 2)
        self.assertIn("2026-09-16", self.panel.tables["domestic"].item(0, 0).text())
        self.assertEqual(self.panel.values["domestic"]["profit"].text(), "+10 KRW")
        self.assertIn("2026-09 주문일 기준", self.panel.summaries["domestic"].text())
        self.assertEqual(self.panel.tables["domestic"].horizontalHeaderItem(0).text(), "주문일·시각")
        self.panel.next_buttons["domestic"].click()
        self.assertEqual((self.panel.dates["domestic"].date().year(),
                          self.panel.dates["domestic"].date().month()), (2026, 10))
        self.assertEqual(self.panel.tables["domestic"].rowCount(), 1)
        self.panel.previous_buttons["domestic"].click()
        self.assertEqual(self.panel.tables["domestic"].rowCount(), 2)

        self.panel.periods["domestic"].setCurrentIndex(2)
        self.assertEqual(self.panel.dates["domestic"].displayFormat(), "yyyy")
        self.assertEqual(self.panel.tables["domestic"].rowCount(), 3)
        self.panel.previous_buttons["domestic"].click()
        self.assertEqual(self.panel.dates["domestic"].date().year(), 2025)
        self.assertEqual(self.panel.tables["domestic"].rowCount(), 0)
        self.panel.next_buttons["domestic"].click()
        self.assertEqual(self.panel.tables["domestic"].rowCount(), 3)
        self.assertEqual(self.panel.periods["us"].currentData(), "day")
        self.assertEqual(self.panel.tables["us"].rowCount(), 1)
        self.assertEqual(self.panel.values["us"]["buy"].text(), "100.00 USD")
        self.store.order_history.assert_not_called()

    def test_year_view_shows_partial_known_profit_without_silently_zeroing_unknown(self):
        records = [order(1), order(2, "sell", "110"),
                   order(3, "sell", "105", symbol="000001",
                         started_at="2026-10-02T00:00:00+00:00")]
        self.panel.refresh(records=records)
        self.panel.periods["domestic"].setCurrentIndex(2)
        self.assertIn("+10 KRW", self.panel.values["domestic"]["profit"].text())
        self.assertIn("확인분만", self.panel.values["domestic"]["profit"].text())
        self.assertIn("미확인 1건", self.panel.summaries["domestic"].text())
        self.assertEqual(self.panel.tables["domestic"].rowCount(), 3)
        self.store.order_history.assert_not_called()

    def test_usd_to_krw_toggle_is_async_display_only_with_dated_reference(self):
        threads = []
        updates = []
        reference = UsdKrwReference(date(2026, 9, 25), D(1200), D("1.25"), D(1500))

        def fetcher(*, timeout):
            threads.append(threading.get_ident())
            self.assertEqual(timeout, 3)
            return reference

        self.panel._fx_fetcher = fetcher
        finished = self.panel._fx_finished
        self.panel._fx_finished = lambda *args: (updates.append(threading.get_ident()),
                                                 finished(*args))
        self.panel.refresh(records=[
            order(1, market="us", exchange="ND", symbol="AAPL", currency="USD"),
            order(2, "sell", "110", market="us", exchange="ND", symbol="AAPL", currency="USD"),
        ])
        self.assertEqual(self.panel.values["us"]["profit"].text(), "+10.00 USD")
        self.panel.fx_toggle.setChecked(True)
        self.wait_for_fx()
        self.assertTrue(threads)
        self.assertNotEqual(threads[0], threading.get_ident())
        self.assertEqual(updates, [threading.get_ident()])
        self.assertIn("2026-09-25", self.panel.fx_status.text())
        self.assertIn("120,000 KRW", self.panel.values["us"]["buy"].text())
        self.assertIn("원본 100.00 USD", self.panel.values["us"]["buy"].text())
        self.assertIn("+12,000 KRW", self.panel.values["us"]["profit"].text())
        self.assertEqual(self.panel.values["us"]["return"].text(), "+10.00%")
        self.assertIn("실제 원화 실현손익 아님", self.panel.fx_status.text())
        self.assertIn("132,000 KRW", self.panel.tables["us"].item(0, 5).text())
        self.assertIn("참고 KRW", self.panel.tables["us"].horizontalHeaderItem(5).text())
        self.assertIn("원본 USD", self.panel.tables["us"].item(0, 5).toolTip())
        self.assertEqual(self.panel.values["domestic"]["buy"].text(), "0 KRW")
        self.store.order_history.assert_not_called()

        self.panel.fx_toggle.setChecked(False)
        self.assertEqual(self.panel.values["us"]["buy"].text(), "100.00 USD")
        self.assertEqual(self.panel.values["us"]["profit"].text(), "+10.00 USD")
        self.assertNotIn("참고 KRW", self.panel.tables["us"].horizontalHeaderItem(5).text())

    def test_failed_reference_never_fabricates_krw_and_usd_remains_visible(self):
        def failing_fetcher(*, timeout):
            raise TimeoutError("offline test")

        self.panel._fx_fetcher = failing_fetcher
        self.panel.refresh(records=[order(1, market="us", exchange="ND",
                                          symbol="AAPL", currency="USD")])
        self.panel.fx_toggle.setChecked(True)
        self.wait_for_fx()
        self.assertIn("환율 조회 실패", self.panel.fx_status.text())
        self.assertIn("USD 원본 유지", self.panel.fx_status.text())
        self.assertEqual(self.panel.values["us"]["buy"].text(), "100.00 USD")
        self.assertEqual(self.panel.tables["us"].item(0, 5).text(), "100.00 USD")
        self.store.order_history.assert_not_called()

    def test_conversion_keeps_unmatched_sale_profit_unknown(self):
        self.panel._fx_fetcher = lambda *, timeout: UsdKrwReference(
            date(2026, 9, 25), D(1200), D("1.25"), D(1500))
        self.panel.refresh(records=[order(1, "sell", "110", market="us",
                                          exchange="ND", symbol="AAPL", currency="USD")])
        self.panel.fx_toggle.setChecked(True)
        self.wait_for_fx()
        self.assertEqual(self.panel.values["us"]["profit"].text(), "미확인")
        self.assertEqual(self.panel.values["us"]["return"].text(), "미확인")
        self.assertEqual(self.panel.tables["us"].item(0, 6).text(), "미확인")
        self.assertIn("미확인 1건", self.panel.summaries["us"].text())


if __name__ == "__main__":
    unittest.main()
