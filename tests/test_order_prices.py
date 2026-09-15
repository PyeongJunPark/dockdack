from __future__ import annotations

from decimal import Decimal, localcontext
import unittest

from dockdack import KiwoomBroker, Market, OrderRequest, OrderSide
from dockdack.order_prices import current_limit_price, validate_us_order_price
from test_kiwoom import FakeResponse, QueueTransport, config, token_response


class CurrentLimitPriceTests(unittest.TestCase):
    def test_side_conservative_rounding_and_one_dollar_boundary(self):
        for quote, buy, sell in (
            ("330.8003", "330.80", "330.81"),
            ("330.8000", "330.80", "330.80"),
            ("0.123456", "0.1234", "0.1235"),
            ("0.99999", "0.9999", "1.00"),
            ("1.00001", "1.00", "1.01"),
            ("1", "1", "1"),
        ):
            for side, expected in (("buy", buy), ("sell", sell)):
                with self.subTest(quote=quote, side=side):
                    value = current_limit_price(Market.US, side, Decimal(quote))
                    self.assertEqual(value, Decimal(expected))
                    self.assertEqual(validate_us_order_price(value), value)
                    self.assertTrue(value <= Decimal(quote) if side == "buy" else value >= Decimal(quote))

    def test_domestic_quote_is_unchanged(self):
        quote = Decimal("251250")
        self.assertIs(current_limit_price("domestic", "buy", quote), quote)

    def test_invalid_values_and_buy_rounding_to_zero_are_rejected(self):
        for price in ("0", "-1", "NaN", "Infinity", "-Infinity"):
            for side in ("buy", "sell"):
                with self.subTest(price=price, side=side), self.assertRaises(ValueError):
                    current_limit_price("us", side, Decimal(price))
        with self.assertRaises(ValueError):
            current_limit_price("us", "buy", Decimal("0.00001"))
        self.assertEqual(current_limit_price("us", "sell", Decimal("0.00001")), Decimal("0.0001"))

    def test_low_decimal_context_does_not_change_rounding(self):
        with localcontext() as context:
            context.prec = 3
            self.assertEqual(current_limit_price("us", "sell", Decimal("330.8003")), Decimal("330.81"))


class BrokerPricePrecisionTests(unittest.TestCase):
    def test_current_orders_submit_once_with_legal_side_conservative_limit(self):
        for quote, side, expected in (
            ("330.8003", "buy", "330.80"), ("330.8003", "sell", "330.81"),
            ("0.123456", "buy", "0.1234"), ("0.123456", "sell", "0.1235"),
            ("0.99999", "sell", "1.00"), ("1.00001", "buy", "1.00"),
        ):
            with self.subTest(quote=quote, side=side):
                transport = QueueTransport(token_response(), FakeResponse({"stk_cd": "AAPL", "cur_prc": quote}),
                                           FakeResponse({"return_code": 0, "ord_no": "123"}))
                broker = KiwoomBroker(config(), transport=transport)
                result = getattr(broker, side + "_at_current_price")(
                    market="us", symbol="AAPL", quantity=1, exchange="ND",
                )
                orders = [call for call in transport.calls if call["url"].endswith("/ordr")]
                self.assertEqual(len(orders), 1)
                self.assertEqual(orders[0]["json"]["ord_uv"], expected)
                self.assertEqual(orders[0]["json"]["trde_tp"], "00")
                self.assertEqual(result.request.price, Decimal(expected))

    def test_manual_sub_tick_price_and_stop_reject_before_any_http(self):
        for value in ("330.8003", "1.0001", "0.12345", "0", "NaN", "Infinity"):
            for field in ("price", "stop_price"):
                with self.subTest(value=value, field=field):
                    transport = QueueTransport()
                    broker = KiwoomBroker(config(), transport=transport)
                    kwargs = dict(market="us", side="sell", symbol="AAPL", quantity=1,
                                  exchange="ND", order_type="34", price="330.8", stop_price="330.7")
                    kwargs[field] = Decimal(value)
                    with self.assertRaises(ValueError):
                        broker.build_order(**kwargs)
                    self.assertEqual(transport.calls, [])

    def test_direct_request_cannot_bypass_limit_or_stop_precision_validation(self):
        for price, stop in ((Decimal("330.8003"), Decimal("330.7")),
                            (Decimal("330.8"), Decimal("330.7003")),
                            (Decimal("0.12345"), Decimal("0.12"))):
            with self.subTest(price=price, stop=stop):
                transport = QueueTransport()
                broker = KiwoomBroker(config(), transport=transport)
                request = OrderRequest(Market.US, OrderSide.SELL, "AAPL", 1, "ND", "34", price, stop)
                with self.assertRaises(ValueError):
                    broker.place_order(request)
                self.assertEqual(transport.calls, [])

    def test_representable_manual_trailing_zeroes_are_canonical_without_rounding(self):
        transport = QueueTransport(token_response(), FakeResponse({"return_code": 0, "ord_no": "123"}))
        broker = KiwoomBroker(config(), transport=transport)
        request = broker.build_order(market="us", side="sell", symbol="AAPL", quantity=1,
                                     exchange="ND", order_type="34", price=Decimal("330.8000000000"),
                                     stop_price=Decimal("330.7000"))
        self.assertEqual(request.price, Decimal("330.8000000000"))
        self.assertEqual(request.stop_price, Decimal("330.7000"))
        broker.place_order(request)
        self.assertEqual(transport.calls[-1]["json"]["ord_uv"], "330.80")
        self.assertEqual(transport.calls[-1]["json"]["stop_pric"], "330.70")


if __name__ == "__main__":
    unittest.main()
