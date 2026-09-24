from __future__ import annotations

import importlib.util
import os
import unittest
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal as D
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None
if HAS_QT:
    from PySide6.QtWidgets import QApplication
    from dockdack.portfolio_gui import PortfolioPanel

from dockdack.models import Market
from dockdack.portfolio import PortfolioMarketState
from test_portfolio import NOW, account, position


@unittest.skipUnless(HAS_QT, "Install the gui extra")
class PortfolioPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.panel = PortfolioPanel()

    def tearDown(self):
        self.panel.close()
        self.panel.deleteLater()
        self.app.processEvents()

    def test_unknown_is_not_displayed_as_zero_holdings_or_zero_cash(self):
        self.assertEqual(self.panel.table.rowCount(), 0)
        for labels in self.panel.market_labels.values():
            self.assertIn("미확인", labels["holdings"].text())
            self.assertIn("미확인", labels["cash"].text())
        self.assertIn("미확인", self.panel.summary_label.text())

    def test_separate_currency_cards_and_readable_holdings(self):
        kr = account(positions=(position(), position(symbol="000660", quantity="0")))
        us = account(Market.US, positions=(replace(position(Market.US, profit="-20"), profit_rate=D("-5")),))
        payload = {
            Market.DOMESTIC: PortfolioMarketState(Market.DOMESTIC, kr, NOW, NOW),
            Market.US: PortfolioMarketState(Market.US, us, NOW, NOW),
        }
        self.panel.apply(payload, now=NOW)
        kr_table = self.panel.tables[Market.DOMESTIC]
        us_table = self.panel.tables[Market.US]
        self.assertEqual(kr_table.rowCount(), 1)
        self.assertEqual(us_table.rowCount(), 1)
        self.assertIn("한국 1종목 / 미국 1종목", self.panel.brief_summary())
        kr_labels = self.panel.market_labels[Market.DOMESTIC]
        us_labels = self.panel.market_labels[Market.US]
        self.assertEqual(kr_labels["cash"].text(), "5,000 KRW")
        self.assertEqual(kr_labels["available"].text(), "4,500 KRW")
        self.assertEqual(us_labels["cash"].text(), "5,000.00 USD")
        self.assertEqual(kr_labels["cash"].accessibleName(), "예수금")
        self.assertEqual(kr_table.item(0, 1).text(), "삼성전자 · 005930")
        self.assertIn("AAPL", us_table.item(0, 1).text())
        self.assertEqual(kr_table.item(0, 2).text(), "2")
        self.assertEqual(kr_table.item(0, 3).text(), "2")
        self.assertEqual(kr_table.item(0, 7).foreground().color().name(), "#f08098")
        self.assertEqual(us_table.item(0, 7).foreground().color().name(), "#84a7ff")
        self.assertIn("-5.00%", us_table.item(0, 8).text())
        self.assertEqual(kr_table.item(0, 0).text(), "한국 · KRW")
        self.assertEqual(us_table.item(0, 0).text(), "미국 · USD")

    def test_error_keeps_rows_and_empty_success_is_explicit(self):
        self.panel.apply({
            Market.DOMESTIC: PortfolioMarketState(Market.DOMESTIC, account(), NOW, NOW, "요청 제한"),
            Market.US: PortfolioMarketState(Market.US, account(Market.US, positions=()), NOW, NOW),
        }, now=NOW)
        self.assertEqual(self.panel.table.rowCount(), 1)
        self.assertIn("이전 잔고 유지", self.panel.market_labels[Market.DOMESTIC]["status"].text())
        self.assertIn("이전 잔고", self.panel.table.item(0, 1).toolTip())
        self.assertIn("요청 제한", self.panel.market_labels[Market.DOMESTIC]["updated"].text())
        self.assertEqual(self.panel.market_labels[Market.US]["status"].text(), "보유종목 없음")
        self.assertNotIn("미국", self.panel.summary_label.text())
        self.panel.market_tabs.setCurrentIndex(1)
        self.assertEqual(self.panel.table.rowCount(), 0)
        self.assertIn("미국: 보유종목 없음", self.panel.summary_label.text())

    def test_stale_relabel_and_manual_refresh_are_pure_display_operations(self):
        payload = {Market.DOMESTIC: PortfolioMarketState(Market.DOMESTIC, account(), NOW, NOW)}
        self.panel.apply(payload, now=NOW + timedelta(seconds=121))
        self.assertIn("오래된 잔고", self.panel.market_labels[Market.DOMESTIC]["status"].text())
        calls = []
        self.panel.request_refresh.connect(lambda: calls.append("refresh"))
        self.panel.refresh_button.click()
        self.assertEqual(calls, ["refresh"])
        self.assertEqual(self.panel.table.rowCount(), 1)

    def test_unchanged_heartbeat_does_not_rebuild_table_or_clear_selection(self):
        payload = {Market.DOMESTIC: PortfolioMarketState(Market.DOMESTIC, account(), NOW, NOW)}
        self.panel.apply(payload, now=NOW)
        self.panel.table.selectRow(0)
        item = self.panel.table.item(0, 1)
        self.panel.apply(payload, now=NOW + timedelta(seconds=1))
        self.assertIs(self.panel.table.item(0, 1), item)
        self.assertEqual(len(self.panel.table.selectedItems()), 11)
        newer = {Market.DOMESTIC: replace(payload[Market.DOMESTIC], fetched_at=NOW + timedelta(seconds=60))}
        self.panel.apply(newer, now=NOW + timedelta(seconds=60))
        self.assertEqual(len(self.panel.table.selectedItems()), 11)

    def test_market_tabs_only_show_the_selected_markets_card_and_holdings(self):
        payload = {market: PortfolioMarketState(market, account(market), NOW, NOW) for market in Market}
        self.panel.apply(payload, now=NOW)
        self.panel.resize(1080, 600)
        self.panel.show()
        self.app.processEvents()
        calls = []
        self.panel.request_refresh.connect(lambda: calls.append("refresh"))
        self.assertEqual(self.panel.market_tabs.tabText(0), "한국 · KRW")
        self.assertEqual(self.panel.market_tabs.tabText(1), "미국 · USD")
        self.assertTrue(self.panel.market_cards[Market.DOMESTIC].isVisible())
        self.assertFalse(self.panel.market_cards[Market.US].isVisible())
        self.assertIs(self.panel.table, self.panel.tables[Market.DOMESTIC])
        self.panel.market_tabs.setCurrentIndex(1)
        self.app.processEvents()
        self.assertIs(self.panel.table, self.panel.tables[Market.US])
        self.assertEqual(self.panel.current_market, Market.US)
        self.assertFalse(self.panel.market_cards[Market.DOMESTIC].isVisible())
        self.assertTrue(self.panel.market_cards[Market.US].isVisible())
        self.assertIn("AAPL", self.panel.table.item(0, 1).text())
        self.assertNotIn("한국", self.panel.summary_label.text())
        self.assertEqual(calls, [])
        self.assertEqual(self.panel._payload, payload)

    def test_refresh_preserves_active_market_and_each_markets_selection_and_scroll(self):
        payload = {
            market: PortfolioMarketState(market, account(market, positions=(
                position(market, symbol=f"{index:06d}" if market is Market.DOMESTIC else f"TEST{index:02d}")
                for index in range(30)
            )), NOW, NOW)
            for market in Market
        }
        self.panel.resize(1080, 500)
        self.panel.apply(payload, now=NOW)
        self.panel.show()
        self.app.processEvents()
        saved = {}
        for index, market in enumerate(Market):
            self.panel.market_tabs.setCurrentIndex(index)
            self.app.processEvents()
            table = self.panel.table
            table.selectRow(15 + index)
            table.verticalScrollBar().setValue(12 + index)
            saved[market] = (table.item(15 + index, 1), table.verticalScrollBar().value())
        self.panel.apply(payload, now=NOW + timedelta(seconds=1))
        self.assertEqual(self.panel.current_market, Market.US)
        for index, market in enumerate(Market):
            table = self.panel.tables[market]
            self.assertIs(table.item(15 + index, 1), saved[market][0])
            self.assertEqual(table.verticalScrollBar().value(), saved[market][1])
            self.assertEqual(len(table.selectedItems()), 11)
        newer = {market: replace(state, fetched_at=NOW + timedelta(seconds=60)) for market, state in payload.items()}
        self.panel.apply(newer, now=NOW + timedelta(seconds=60))
        self.assertEqual(self.panel.current_market, Market.US)
        for market, table in self.panel.tables.items():
            self.assertEqual(table.verticalScrollBar().value(), saved[market][1])
            self.assertEqual(len(table.selectedItems()), 11)

    def test_market_error_unknown_and_stale_are_not_combined_with_other_market(self):
        self.panel.apply({
            Market.DOMESTIC: PortfolioMarketState(Market.DOMESTIC, account(), NOW, NOW),
            Market.US: PortfolioMarketState(Market.US, last_attempt=NOW, error="미국 조회 실패"),
        }, now=NOW + timedelta(seconds=121))
        self.assertIn("오래된 잔고", self.panel.summary_label.text())
        self.assertNotIn("미국 조회 실패", self.panel.summary_label.text())
        self.assertEqual(self.panel.table.rowCount(), 1)
        self.panel.market_tabs.setCurrentIndex(1)
        self.assertIn("보유 여부 미확인", self.panel.summary_label.text())
        self.assertIn("미국 조회 실패", self.panel.market_labels[Market.US]["updated"].text())
        self.assertEqual(self.panel.table.rowCount(), 0)

    def test_unchanged_state_skips_formatting_sorting_and_style_work(self):
        payload = {Market.DOMESTIC: PortfolioMarketState(Market.DOMESTIC, account(), NOW, NOW)}
        self.panel.apply(payload, now=NOW)
        with patch("dockdack.portfolio_gui._money", side_effect=AssertionError("unchanged heartbeat formatted amounts")), \
                patch.object(self.panel, "_apply_rows", side_effect=AssertionError("unchanged heartbeat rebuilt rows")):
            self.panel.apply(payload, now=NOW + timedelta(seconds=1))
            # An equal immutable copy is unchanged as well, not only identity.
            self.panel.apply({Market.DOMESTIC: replace(payload[Market.DOMESTIC])}, now=NOW + timedelta(seconds=2))
        self.panel.apply(payload, now=NOW + timedelta(seconds=121))
        self.assertIn("오래된 잔고", self.panel.market_labels[Market.DOMESTIC]["status"].text())

    def test_market_column_is_hidden_without_removing_data_or_currency_context(self):
        self.panel.apply({Market.DOMESTIC: PortfolioMarketState(Market.DOMESTIC, account(), NOW, NOW)}, now=NOW)
        self.assertTrue(self.panel.table.isColumnHidden(0))
        self.assertEqual(self.panel.table.item(0, 0).text(), "한국 · KRW")
        self.assertIn("KRW", self.panel.market_tabs.tabText(0))
        self.assertEqual(self.panel.market_labels[Market.DOMESTIC]["evaluation"].objectName(), "portfolioValue")


if __name__ == "__main__":
    unittest.main()
