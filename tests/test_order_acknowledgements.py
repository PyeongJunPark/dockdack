from __future__ import annotations

import unittest
from unittest.mock import patch

from dockdack import BrokerAPIError, KiwoomBroker, Market, OrderOutcomeUnknown
from dockdack.http import RequestsTransport
from test_kiwoom import FakeResponse, QueueTransport, config, token_response


class OrderAcknowledgementTests(unittest.TestCase):
    def execute(self, broker, market, operation):
        args = {
            "market": market,
            "symbol": "005930" if market is Market.DOMESTIC else "AAPL",
            "exchange": "KRX" if market is Market.DOMESTIC else "ND",
            "quantity": 1,
        }
        if operation == "cancel":
            return broker.cancel_order(**args, original_order_number="0000140")
        return getattr(broker, operation)(**args, price="1000")

    def assert_unknown(self, body):
        for market in Market:
            for operation in ("buy", "sell", "cancel"):
                with self.subTest(market=market, operation=operation, body=body):
                    transport = QueueTransport(token_response(), FakeResponse(body))
                    broker = KiwoomBroker(config(), transport=transport)
                    with self.assertRaisesRegex(OrderOutcomeUnknown, "주문·체결 내역"):
                        self.execute(broker, market, operation)
                    # One token request and one submission, never an automatic retry.
                    self.assertEqual(len(transport.calls), 2)

    def test_empty_acknowledgement_is_unknown(self):
        self.assert_unknown({})

    def test_order_number_without_success_code_is_unknown(self):
        self.assert_unknown({"ord_no": "0000200"})

    def test_missing_order_number_is_unknown(self):
        self.assert_unknown({"return_code": 0})

    def test_invalid_order_number_is_unknown(self):
        for number in (None, "", "   ", False, 0, 200, [], {}):
            self.assert_unknown({"return_code": 0, "ord_no": number})

    def test_non_integer_success_codes_are_not_coerced_to_zero(self):
        for code in (None, "", False, 0.0, 0.5):
            self.assert_unknown({"return_code": code, "ord_no": "0000200"})

    def test_confirmed_response_preserves_order_number_leading_zeros(self):
        for code in (0, "0", " 0 "):
            for market in Market:
                for operation in ("buy", "sell", "cancel"):
                    with self.subTest(code=code, market=market, operation=operation):
                        transport = QueueTransport(token_response(), FakeResponse({
                            "return_code": code, "ord_no": " 0000200 ",
                        }))
                        result = self.execute(KiwoomBroker(config(), transport=transport), market, operation)
                        self.assertTrue(result.accepted)
                        number = result.cancel_order_number if operation == "cancel" else result.order_number
                        self.assertEqual(number, "0000200")
                        self.assertEqual(len(transport.calls), 2)

    def test_business_rejection_is_not_reported_as_unknown_or_success(self):
        for operation in ("buy", "sell", "cancel"):
            transport = QueueTransport(token_response(), FakeResponse({
                "return_code": 2000, "return_msg": "모의투자 장종료", "ord_no": "0000200",
            }))
            with self.assertRaisesRegex(BrokerAPIError, "장종료") as raised:
                self.execute(KiwoomBroker(config(), transport=transport), Market.US, operation)
            self.assertNotIsInstance(raised.exception, OrderOutcomeUnknown)
            self.assertEqual(len(transport.calls), 2)

    def test_http_errors_and_redirects_do_not_accept_or_retry_submissions(self):
        for status in (302, 401, 500):
            for market in Market:
                for operation in ("buy", "sell", "cancel"):
                    with self.subTest(status=status, market=market, operation=operation):
                        transport = QueueTransport(token_response(), FakeResponse(
                            {"return_code": 0, "ord_no": "0000200"}, status_code=status,
                        ))
                        with self.assertRaises(BrokerAPIError):
                            self.execute(KiwoomBroker(config(), transport=transport), market, operation)
                        self.assertEqual(len(transport.calls), 2)

    def test_cancel_timeout_does_not_retry(self):
        class TimeoutTransport(QueueTransport):
            def request(self, method, url, **kwargs):
                response = super().request(method, url, **kwargs)
                if url.endswith("/ordr"):
                    raise TimeoutError("response timed out")
                return response

        for market in Market:
            transport = TimeoutTransport(token_response(), FakeResponse({}))
            with self.assertRaises(BrokerAPIError):
                self.execute(KiwoomBroker(config(), transport=transport), market, "cancel")
            self.assertEqual(len(transport.calls), 2)

    def test_default_transport_does_not_follow_redirects(self):
        with patch("requests.Session") as session:
            transport = RequestsTransport()
            transport.request("POST", "https://mockapi.kiwoom.com/api/dostk/ordr",
                              headers={}, json={}, timeout=15)
            session.return_value.request.assert_called_once_with(
                method="POST", url="https://mockapi.kiwoom.com/api/dostk/ordr", headers={}, json={},
                timeout=15, allow_redirects=False,
            )

    def test_invalid_cancel_quantity_is_rejected_before_http(self):
        for quantity in (True, -1, 1.5, "1", None, 1_000_000_000_000):
            with self.subTest(quantity=quantity):
                transport = QueueTransport()
                broker = KiwoomBroker(config(), transport=transport)
                with self.assertRaises(ValueError):
                    broker.cancel_order(market=Market.DOMESTIC, original_order_number="0000140",
                                        symbol="005930", exchange="KRX", quantity=quantity)
                self.assertEqual(transport.calls, [])

    def test_invalid_original_order_number_is_rejected_before_http(self):
        for number in (None, "", "  ", 140, False):
            with self.subTest(number=number):
                transport = QueueTransport()
                broker = KiwoomBroker(config(), transport=transport)
                with self.assertRaises(ValueError):
                    broker.cancel_order(market=Market.DOMESTIC, original_order_number=number,
                                        symbol="005930", exchange="KRX")
                self.assertEqual(transport.calls, [])

    def test_domestic_zero_cancel_quantity_still_means_all_remaining(self):
        transport = QueueTransport(token_response(), FakeResponse({"return_code": 0, "ord_no": "0000200"}))
        result = KiwoomBroker(config(), transport=transport).cancel_order(
            market=Market.DOMESTIC, original_order_number="0000140", symbol="005930", exchange="KRX",
        )
        self.assertTrue(result.accepted)
        self.assertEqual(transport.calls[-1]["json"]["cncl_qty"], "0")


if __name__ == "__main__":
    unittest.main()
