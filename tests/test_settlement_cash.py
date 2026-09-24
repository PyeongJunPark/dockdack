"""Offline account parsing, allocation, and signed-cash presentation regression."""

import importlib.util
import os
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal as D

from dockdack.execution_policy import account_equity, allocation_quantity
from dockdack.kiwoom import KiwoomBroker
from dockdack.models import AccountSnapshot, Market
from dockdack.portfolio import PortfolioMarketState
from test_kiwoom import FakeResponse, QueueTransport, config, token_response


def snapshot(**changes):
    values = dict(market=Market.DOMESTIC, currency="KRW", positions=(), cash=D("-1200"),
                  available_to_order=D("3000"), total_evaluation=D("7000"),
                  cash_d1=D("400"), cash_d2=D("3000"), cash_receivable=D("1200"),
                  cash_settlement_source="kt00001:d2_entra")
    values.update(changes)
    return AccountSnapshot(**values)


class SettlementCashParserTests(unittest.TestCase):
    def domestic(self, deposit):
        transport = QueueTransport(token_response(), FakeResponse({"acnt_evlt_remn_indv_tot": []}),
                                   FakeResponse(deposit))
        return KiwoomBroker(config(), transport=transport).account_domestic()

    def test_domestic_keeps_negative_cash_and_separate_settlement_values(self):
        account = self.domestic({"entr": "-00000000001200", "ord_alow_amt": "3000",
                                 "d1_entra": "400", "d2_entra": "3000", "ch_uncla": "1200"})
        self.assertEqual(account.cash, D("-1200"))
        self.assertEqual(account.available_to_order, D("3000"))
        self.assertEqual(account.cash_d1, D("400"))
        self.assertEqual(account.cash_d2, D("3000"))
        self.assertEqual(account.cash_receivable, D("1200"))
        self.assertEqual(account.cash_settlement_source, "kt00001:d2_entra")

    def test_missing_settlement_is_not_zero_or_available_to_order(self):
        account = self.domestic({"entr": "-500", "ord_alow_amt": "10000"})
        self.assertIsNone(account.cash_d1)
        self.assertIsNone(account.cash_d2)
        self.assertIsNone(account.cash_receivable)
        self.assertEqual(account.cash_settlement_source, "")

    def test_malformed_or_nonfinite_forecasts_remain_unknown(self):
        for value in ("NaN", "Infinity", "bad"):
            with self.subTest(value=value):
                account = self.domestic({"entr": "-100", "d1_entra": value,
                                         "d2_entra": value, "ch_uncla": value})
                self.assertIsNone(account.cash_d1)
                self.assertIsNone(account.cash_d2)
                self.assertIsNone(account.cash_receivable)
                self.assertEqual(account.cash_settlement_source, "")

    def test_us_uses_usd_row_receivable_not_top_level_krw(self):
        transport = QueueTransport(token_response(), FakeResponse({"result_list": []}),
                                   FakeResponse({"krw_entra": "90000000", "ch_uncla": "5000000",
                                                 "d2_entra": "99999999", "result_list": [
                                                     {"crnc_code": "JPY", "fc_entra": "100", "fc_ch_uncla": "20"},
                                                     {"crnc_code": "USD", "fc_entra": "2000",
                                                      "fc_ord_alowa": "1800.125", "fc_ch_uncla": "3.25"}]}))
        account = KiwoomBroker(config(), transport=transport).account_us()
        self.assertEqual(account.cash, D("2000"))
        self.assertEqual(account.available_to_order, D("1800.125"))
        self.assertEqual(account.cash_receivable, D("3.25"))
        self.assertIsNone(account.cash_d1)
        self.assertIsNone(account.cash_d2)
        self.assertEqual(account.cash_settlement_source, "")


class SettlementAllocationTests(unittest.TestCase):
    def quantity(self, account, price="100"):
        return allocation_quantity(account, D(price), D("10"), D("10000000"))

    def test_negative_current_cash_uses_verified_d2_plus_current_holdings(self):
        account = snapshot()
        self.assertEqual(account_equity(account), D("10000"))
        self.assertEqual(self.quantity(account), 10)
        self.assertEqual(account.cash, D("-1200"))

    def test_negative_cash_missing_or_untrusted_settlement_is_blocked(self):
        for changes in ({"cash_d2": None}, {"cash_settlement_source": ""},
                        {"cash_settlement_source": "external-signal"},
                        {"cash_d2": D("NaN")}, {"cash_d2": D("Infinity")},
                        {"cash_d2": D("-1")}, {"cash_d2": "3000"},
                        {"currency": "USD"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.quantity(snapshot(**changes))

    def test_raw_untyped_d2_or_buying_power_never_fills_missing_settlement(self):
        account = snapshot(cash_d2=None, cash_settlement_source="", raw={"deposit": {"d2_entra": "1000000"}})
        with self.assertRaises(ValueError):
            self.quantity(account)

    def test_zero_d2_cannot_spend_positive_margin_buying_power(self):
        with self.assertRaises(ValueError):
            self.quantity(snapshot(cash_d2=D(0), available_to_order=D("999999999")))

    def test_settlement_cash_caps_budget_below_ten_percent_of_equity(self):
        account = snapshot(cash_d2=D("150"), available_to_order=D("999999999"))
        self.assertEqual(self.quantity(account), 1)

    def test_available_funds_remain_an_independent_cap(self):
        self.assertEqual(self.quantity(snapshot(available_to_order=D("150"))), 1)
        with self.assertRaises(ValueError):
            self.quantity(snapshot(available_to_order=D(0)))

    def test_nonnegative_cash_behavior_unchanged_even_if_d2_is_present(self):
        account = snapshot(cash=D("40000"), cash_d2=D("900000"), total_evaluation=D("60000"),
                           available_to_order=D("40000"))
        self.assertEqual(account_equity(account), D("100000"))
        self.assertEqual(self.quantity(account, "1000"), 10)

    def test_us_cannot_use_domestic_settlement_even_if_accidentally_attached(self):
        account = snapshot(market=Market.US, currency="USD", cash=D("-10"))
        with self.assertRaises(ValueError):
            self.quantity(account)
        positive = replace(account, cash=D("2000"), total_evaluation=D("450.25"))
        self.assertEqual(account_equity(positive), D("2450.25"))


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None
if HAS_QT:
    from PySide6.QtWidgets import QApplication
    from dockdack.portfolio_gui import PortfolioPanel


@unittest.skipUnless(HAS_QT, "Install the gui extra")
class SettlementCashGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.panel = PortfolioPanel()
        self.now = datetime.now(timezone.utc)

    def tearDown(self):
        self.panel.close()
        self.panel.deleteLater()
        self.app.processEvents()

    def apply(self, account):
        self.panel.apply({account.market: PortfolioMarketState(account.market, account, self.now, self.now)}, now=self.now)

    def test_negative_current_cash_stays_visible_with_d1_d2_and_receivable(self):
        self.apply(snapshot())
        labels = self.panel.market_labels[Market.DOMESTIC]
        self.assertEqual(labels["cash"].text(), "-1,200 KRW")
        self.assertEqual(labels["available"].text(), "3,000 KRW")
        summary = self.panel.summaries[Market.DOMESTIC].text()
        self.assertIn("D+1 추정 400 KRW", summary)
        self.assertIn("D+2 추정 3,000 KRW", summary)
        self.assertIn("현금미수 1,200 KRW", summary)
        self.assertIn("현재 예수금 음수", summary)
        self.assertIn("D+2 추정예수금 + 보유 평가금액", labels["cash"].toolTip())
        self.assertIn("수수료·세금 차감 전", labels["cash"].toolTip())

    def test_unknown_settlement_shows_buy_calculation_hold_not_fake_positive_cash(self):
        self.apply(snapshot(cash_d1=None, cash_d2=None, cash_settlement_source=""))
        summary = self.panel.summaries[Market.DOMESTIC].text()
        self.assertIn("비중 매수 보류", summary)
        self.assertNotIn("D+2 추정 ", summary)
        self.assertEqual(self.panel.market_labels[Market.DOMESTIC]["cash"].text(), "-1,200 KRW")

    def test_us_context_stays_usd_without_domestic_forecasts(self):
        self.apply(snapshot(market=Market.US, currency="USD", cash=D("2000"), cash_d1=None,
                            cash_d2=None, cash_settlement_source="", cash_receivable=D("3.25")))
        summary = self.panel.summaries[Market.US].text()
        self.assertIn("외화현금미수 3.25 USD", summary)
        self.assertNotIn("KRW", summary)
        self.assertNotIn("D+2", summary)

    def test_mode_clear_removes_old_settlement_context_and_warning(self):
        self.apply(snapshot())
        self.panel.apply({}, now=self.now)
        self.assertNotIn("1,200", self.panel.summaries[Market.DOMESTIC].text())
        self.assertNotIn("1,200", self.panel.market_labels[Market.DOMESTIC]["cash"].toolTip())
        self.assertEqual(self.panel.market_labels[Market.DOMESTIC]["cash"].styleSheet(), "")


if __name__ == "__main__":
    unittest.main()
