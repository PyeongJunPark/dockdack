"""Reference FX is presentation-only for US holdings."""

from __future__ import annotations

import importlib.util
import os
import unittest
from datetime import date
from decimal import Decimal as D
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None
if HAS_QT:
    from PySide6.QtWidgets import QApplication
    from dockdack.fx_reference import UsdKrwReference
    from dockdack.portfolio_gui import PortfolioPanel

from dockdack.models import Market
from dockdack.portfolio import PortfolioMarketState
from test_portfolio import NOW, account


@unittest.skipUnless(HAS_QT, "Install the gui extra")
class PortfolioFxGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.panel = PortfolioPanel(fx_fetcher=lambda *, timeout: (_ for _ in ()).throw(
            AssertionError("No network in a GUI unit test")))
        self.payload = {market: PortfolioMarketState(market, account(market), NOW, NOW)
                        for market in Market}
        self.panel.apply(self.payload, now=NOW)
        self.panel.market_tabs.setCurrentIndex(1)
        self.reference = UsdKrwReference(date(2026, 9, 25), D(1200), D("1.25"), D(1500))

    def tearDown(self):
        self.panel.close()
        self.panel.deleteLater()
        self.app.processEvents()

    def _request_without_network(self):
        with patch("dockdack.ui.portfolio_gui.QThreadPool.globalInstance") as pool:
            self.panel.fx_toggle.click()
            pool.return_value.start.assert_called_once()
        return self.panel._fx_request_id

    def test_default_usd_and_successful_toggle_converts_display_not_account(self):
        us = self.panel.tables[Market.US]
        kr = self.panel.tables[Market.DOMESTIC]
        self.assertFalse(self.panel.fx_toggle.isChecked())
        self.assertEqual(self.panel.market_labels[Market.US]["cash"].text(), "5,000.00 USD")
        self.assertEqual(us.item(0, 4).text(), "100.0000")
        token = self._request_without_network()
        self.assertEqual(self.panel.market_labels[Market.US]["cash"].text(), "5,000.00 USD")
        self.panel._fx_finished(token, self.reference, None)
        self.assertEqual(self.panel.market_tabs.tabText(1), "미국 · KRW")
        self.assertEqual(self.panel.market_labels[Market.US]["cash"].text(), "6,000,000 KRW")
        self.assertEqual(us.item(0, 4).text(), "120,000")
        self.assertEqual(us.item(0, 5).text(), "240,000")
        self.assertEqual(us.item(0, 6).text(), "480,000")
        self.assertEqual(us.item(0, 9).text(), "≥ 121,200")
        self.assertIn("실제 매도 기준 ≥ 101.0000 USD", us.item(0, 9).toolTip())
        self.assertIn("원본 100.0000 USD", us.item(0, 4).toolTip())
        self.assertEqual(kr.item(0, 4).text(), "100")
        self.assertEqual(self.payload[Market.US].snapshot.cash, D(5000))
        self.assertEqual(self.payload[Market.US].positions[0].average_price, D(100))

        self.panel.fx_toggle.click()
        self.assertEqual(self.panel.market_tabs.tabText(1), "미국 · USD")
        self.assertEqual(self.panel.market_labels[Market.US]["cash"].text(), "5,000.00 USD")
        self.assertEqual(us.item(0, 9).text(), "≥ 101")

        with patch("dockdack.ui.portfolio_gui.QThreadPool.globalInstance") as pool:
            self.panel.fx_toggle.click()
            pool.return_value.start.assert_not_called()
        self.assertEqual(self.panel.market_tabs.tabText(1), "미국 · KRW")
        self.assertEqual(self.panel.market_labels[Market.US]["cash"].text(), "6,000,000 KRW")
        self.assertIn("2026-09-25", self.panel.fx_status.toolTip())
        with patch("dockdack.ui.portfolio_gui.QThreadPool.globalInstance") as pool:
            self.panel.fx_refresh_button.click()
            pool.return_value.start.assert_called_once()
        self.assertGreater(self.panel._fx_request_id, token)
        self.panel._fx_finished(self.panel._fx_request_id, self.reference, None)

    def test_failure_keeps_usd_and_late_worker_result_cannot_reenable_fx(self):
        token = self._request_without_network()
        self.panel._fx_finished(token, None, ValueError("offline"))
        self.assertIn("USD 유지", self.panel.fx_status.text())
        self.assertEqual(self.panel.tables[Market.US].item(0, 4).text(), "100.0000")
        self.panel.fx_toggle.click()
        self.panel._fx_finished(token, self.reference, None)
        self.assertFalse(self.panel.fx_toggle.isChecked())
        self.assertEqual(self.panel.market_labels[Market.US]["cash"].text(), "5,000.00 USD")

    def test_invalid_reference_date_cannot_render_krw(self):
        token = self._request_without_network()
        invalid = UsdKrwReference("not-a-date", D(1200), D("1.25"), D(1500))
        self.panel._fx_finished(token, invalid, None)
        self.assertIn("USD 유지", self.panel.fx_status.text())
        self.assertEqual(self.panel.market_labels[Market.US]["cash"].text(), "5,000.00 USD")

    def test_failed_explicit_refresh_preserves_last_dated_rate(self):
        token = self._request_without_network()
        self.panel._fx_finished(token, self.reference, None)
        with patch("dockdack.ui.portfolio_gui.QThreadPool.globalInstance") as pool:
            self.panel.fx_refresh_button.click()
            pool.return_value.start.assert_called_once()
        self.assertEqual(self.panel.market_labels[Market.US]["cash"].text(), "6,000,000 KRW")
        self.panel._fx_finished(self.panel._fx_request_id,
                                UsdKrwReference(date(2026, 9, 26), D("NaN"), D("1.25"), D(1500)), None)
        self.assertIn("갱신 실패 · 이전 환율", self.panel.fx_status.text())
        self.assertIn("2026-09-25", self.panel.fx_status.toolTip())
        self.assertEqual(self.panel.market_tabs.tabText(1), "미국 · KRW")
        self.assertEqual(self.panel.market_labels[Market.US]["cash"].text(), "6,000,000 KRW")
        self.assertIs(self.panel._fx_reference, self.reference)

    def test_fresh_holding_quote_is_converted_but_order_threshold_stays_usd_underneath(self):
        token = self._request_without_network()
        self.panel._fx_finished(token, self.reference, None)
        table = self.panel.tables[Market.US]
        from types import SimpleNamespace
        self.panel.apply_holding_quote({"watch_id": "us:ND:AAPL",
                                        "instrument": SimpleNamespace(market=Market.US, currency="USD"),
                                        "quote": SimpleNamespace(price=D(210)),
                                        "targets": {"take_profit_price": D(101), "stop_loss_price": D(99)}})
        self.assertEqual(table.item(0, 5).text(), "252,000")
        self.assertEqual(table.item(0, 9).text(), "≥ 121,200")
        self.assertIn("실제 매도 기준 ≥ 101 USD", table.item(0, 9).toolTip())
        self.assertEqual(self.payload[Market.US].positions[0].current_price, D(200))


if __name__ == "__main__":
    unittest.main()
