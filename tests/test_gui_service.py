from __future__ import annotations

import unittest
from decimal import Decimal

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


if __name__ == "__main__":
    unittest.main()
