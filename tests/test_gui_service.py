from __future__ import annotations

import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dockdack import KiwoomBroker, Market, TradingMode
from dockdack.gui_service import TradingService
from test_kiwoom import FakeResponse, QueueTransport, config, token_response


class GuiServiceTests(unittest.TestCase):
    def service(self, *responses, mode=TradingMode.DEMO):
        transport = QueueTransport(*responses)
        broker = KiwoomBroker(config(mode), transport=transport)
        return TradingService(lambda _: broker), transport

    def test_demo_only_guard(self):
        service, transport = self.service(mode=TradingMode.REAL)
        with self.assertRaisesRegex(ValueError, "모의투자 전용"):
            service.broker(Market.DOMESTIC)
        self.assertEqual(transport.calls, [])

    def test_us_resolve_and_current_order_preview(self):
        service, transport = self.service(
            token_response(), FakeResponse({"list": [{"stk_cd": "AAPL", "stex_tp": "ND"}]}),
            FakeResponse({"cur_prc": "329.49"}),
        )
        instrument = service.resolve("aapl")
        request = service.prepare(instrument, "buy", 1, "current")
        self.assertEqual(request.price, Decimal("329.49"))
        self.assertEqual(instrument.currency, "USD")
        self.assertFalse(any(c["url"].endswith("/ordr") for c in transport.calls))

    def test_us_market_order_fails_before_http(self):
        service, transport = self.service()
        with self.assertRaisesRegex(ValueError, "지정가만"):
            service.prepare(service.resolve("AAPL", "ND"), "buy", 1, "market")
        self.assertEqual(transport.calls, [])

    def test_domestic_fractional_limit_rejected(self):
        service, transport = self.service()
        with self.assertRaises(ValueError):
            service.prepare(service.resolve("005930"), "sell", 1, "limit", Decimal("10.1"))
        self.assertEqual(transport.calls, [])

    def test_domestic_execution_mapping(self):
        service, transport = self.service(token_response(), FakeResponse({"cntr": [{
            "ord_no": "0145934", "stk_cd": "005930", "ord_qty": "1", "cntr_qty": "1",
            "cntr_pric": "251250", "oso_qty": "0", "ord_stt": "체결", "io_tp_nm": "-매도",
        }]}))
        orders = service.executions(service.resolve("005930"))
        self.assertEqual(orders[0].fill_price, Decimal("251250"))
        self.assertEqual(orders[0].filled_quantity, 1)
        self.assertEqual(transport.calls[-1]["headers"]["api-id"], "ka10076")
        self.assertEqual(transport.calls[-1]["json"]["ord_no"], "")

    def test_us_execution_mapping_supports_documented_typo(self):
        for key in ("result_list", "result_lsit"):
            service, transport = self.service(token_response(), FakeResponse({key: [{
                "ord_no": "000000123", "stk_cd": "AAPL", "ord_qty": "000000000001", "cntr_qty": "000000000000",
                "cntr_uv": "0.0000", "ord_uv": "329.4900", "ord_remnq": "1", "ord_stat": "접수", "slby_tp_nm": "매수",
            }]}))
            orders = service.executions(service.resolve("AAPL", "ND"))
            self.assertEqual(orders[0].filled_quantity, 0)
            self.assertEqual(orders[0].remaining_quantity, 1)
            self.assertEqual(orders[0].status, "접수")
            self.assertEqual(transport.calls[-1]["headers"]["api-id"], "ust21510")

    def test_safety_orders_rejects_missing_or_malformed_lists(self):
        for body in ({}, {"oso": None}, {"oso": [{"stk_cd": "005930", "ord_no": "1"}]}):
            service, _ = self.service(token_response(), FakeResponse(body))
            with self.assertRaises(ValueError):
                service.safety_orders(service.resolve("005930"))

    def test_safety_account_rejects_missing_position_quantities(self):
        service, _ = self.service(token_response(), FakeResponse({"acnt_evlt_remn_indv_tot": [{"stk_cd": "005930"}]}),
                                  FakeResponse({"ord_alow_amt": "10000"}))
        with self.assertRaises(ValueError):
            service.safety_account(service.resolve("005930"))

    def test_us_non_usd_deposit_is_not_used_as_usd_available_funds(self):
        service, _ = self.service(token_response(), FakeResponse({"result_list": []}),
                                  FakeResponse({"result_list": [{"crnc_code": "JPY", "fc_entra": "100000", "fc_ord_alowa": "100000"}]}))
        account = service.safety_account(service.resolve("AAPL", "ND"))
        self.assertIsNone(account.available_to_order)
        self.assertIsNone(account.cash)

    def test_safety_executions_rejects_missing_quantity_or_list(self):
        for body in ({}, {"cntr": [{"ord_no": "1", "stk_cd": "005930", "ord_qty": "1", "cntr_qty": "1"}]}):
            service, _ = self.service(token_response(), FakeResponse(body))
            with self.assertRaises(ValueError):
                service.safety_executions(service.resolve("005930"))

    def test_protected_symbols_reads_all_market_holdings_and_orders(self):
        for market, exchange, currency in ((Market.DOMESTIC, "KRX", "KRW"), (Market.US, "%", "USD")):
            service = TradingService()
            account = SimpleNamespace(market=market, currency=currency, positions=(
                SimpleNamespace(market=market, currency=currency, symbol="HELD", quantity=Decimal(1)),))
            broker = Mock()
            broker.list_open_orders.return_value = (
                SimpleNamespace(symbol="PENDING", remaining_quantity=Decimal(1)),
                SimpleNamespace(symbol="DONE", remaining_quantity=Decimal(0)))
            with patch.object(service, "safety_account", return_value=account), patch.object(service, "broker", return_value=broker):
                self.assertEqual(service.protected_symbols(market), {"HELD", "PENDING"})
            broker.list_open_orders.assert_called_once_with(market, exchange=exchange, strict=True)

    def test_protected_symbols_fails_closed_for_unverifiable_quantity(self):
        service = TradingService()
        account = SimpleNamespace(market=Market.US, currency="USD", positions=(
            SimpleNamespace(market=Market.US, currency="USD", symbol="AAPL", quantity=Decimal("NaN")),))
        with patch.object(service, "safety_account", return_value=account), patch.object(service, "broker") as broker:
            with self.assertRaises(ValueError):
                service.protected_symbols(Market.US)
        broker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
