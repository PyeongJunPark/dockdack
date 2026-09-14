from __future__ import annotations

import unittest
from decimal import Decimal

from dockdack import BrokerAPIError, KiwoomBroker, LiveOrderConfirmationRequired, TradingMode
from test_kiwoom import FakeResponse, QueueTransport, config, token_response


class CurrentPriceOrderTests(unittest.TestCase):
    def test_buy_and_sell_use_queried_price_for_both_markets(self):
        for market, symbol, price in (("domestic", "005930", "-251250"), ("us", "AAPL", "+329.4900")):
            for side in ("buy", "sell"):
                with self.subTest(market=market, side=side):
                    responses = [token_response()]
                    if market == "us":
                        responses.append(FakeResponse({"list": [{"stk_cd": symbol, "stex_tp": "ND"}]}))
                    responses += [FakeResponse({"stk_cd": symbol, "cur_prc": price}),
                                  FakeResponse({"return_code": 0, "ord_no": "123"})]
                    transport = QueueTransport(*responses)
                    broker = KiwoomBroker(config(), transport=transport)
                    result = getattr(broker, f"{side}_at_current_price")(
                        market=market, symbol=symbol.lower(), quantity=2,
                    )
                    order = transport.calls[-1]
                    self.assertEqual(order["json"]["ord_uv"], "251250" if market == "domestic" else "329.4900")
                    self.assertEqual(order["json"]["trde_tp"], "0" if market == "domestic" else "00")
                    self.assertEqual(order["json"]["ord_qty"], "2")
                    self.assertEqual(order["headers"]["api-id"],
                                     ("kt1000" if market == "domestic" else "ust2000") + ("0" if side == "buy" else "1"))
                    self.assertEqual(result.request.price, abs(Decimal(price)))
                    self.assertEqual(sum(call["url"].endswith("/ordr") for call in transport.calls), 1)

    def test_explicit_exchange_preview_does_not_resolve_or_submit(self):
        transport = QueueTransport(token_response(), FakeResponse({"stk_cd": "AAPL", "cur_prc": "329.49"}))
        broker = KiwoomBroker(config(), transport=transport)
        request = broker.build_order_at_current_price(
            market="us", side="buy", symbol="AAPL", quantity=1, exchange="NASDAQ",
        )
        self.assertEqual(request.price, Decimal("329.49"))
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(transport.calls[-1]["headers"]["api-id"], "usa20100")

    def test_invalid_quantity_prevents_any_network_request(self):
        for quantity in (0, -1, True, 1.5):
            transport = QueueTransport()
            broker = KiwoomBroker(config(), transport=transport)
            with self.assertRaises(ValueError):
                broker.buy_at_current_price(market="us", symbol="AAPL", quantity=quantity)
            self.assertEqual(transport.calls, [])

    def test_invalid_or_wrong_symbol_quote_does_not_order(self):
        for quote in ({"cur_prc": "0"}, {"cur_prc": "NaN"}, {"cur_prc": "Infinity"},
                      {"cur_prc": "oops"}, {}, {"stk_cd": "MSFT", "cur_prc": "329.49"}):
            with self.subTest(quote=quote):
                transport = QueueTransport(token_response(), FakeResponse(quote))
                broker = KiwoomBroker(config(), transport=transport)
                with self.assertRaises(ValueError):
                    broker.sell_at_current_price(market="us", symbol="AAPL", quantity=1, exchange="NASDAQ")
                self.assertEqual(len(transport.calls), 2)

    def test_quote_failure_prevents_order(self):
        transport = QueueTransport(token_response(), FakeResponse({"return_code": 1, "return_msg": "시세 조회 실패"}))
        broker = KiwoomBroker(config(), transport=transport)
        with self.assertRaises(BrokerAPIError):
            broker.buy_at_current_price(market="domestic", symbol="005930", quantity=1)
        self.assertEqual(len(transport.calls), 2)

    def test_closed_market_failure_is_not_retried(self):
        transport = QueueTransport(token_response(), FakeResponse({"cur_prc": "329.49"}),
                                   FakeResponse({"return_code": 2000, "return_msg": "RC4058:모의투자 장종료"}))
        broker = KiwoomBroker(config(), transport=transport)
        with self.assertRaisesRegex(BrokerAPIError, "장종료"):
            broker.buy_at_current_price(market="us", symbol="AAPL", quantity=1, exchange="NASDAQ")
        self.assertEqual(len(transport.calls), 3)

    def test_live_confirmation_checked_before_query(self):
        for side in ("buy", "sell"):
            for enabled, confirmation in ((False, "LIVE_ORDER"), (True, None)):
                transport = QueueTransport()
                broker = KiwoomBroker(config(TradingMode.REAL, allow_live_orders=enabled), transport=transport)
                with self.assertRaises(LiveOrderConfirmationRequired):
                    getattr(broker, f"{side}_at_current_price")(
                        market="us", symbol="AAPL", quantity=1, confirm_live_order=confirmation,
                    )
                self.assertEqual(transport.calls, [])


if __name__ == "__main__":
    unittest.main()
