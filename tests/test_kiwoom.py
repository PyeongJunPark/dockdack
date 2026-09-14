from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping

from dockdack import (
    DomesticExchange,
    KiwoomBroker,
    KiwoomConfig,
    LiveOrderConfirmationRequired,
    Market,
    OrderSide,
    TradingMode,
    USExchange,
)


@dataclass
class FakeResponse:
    body: dict[str, Any]
    status_code: int = 200
    headers: Mapping[str, str] = field(default_factory=dict)
    text: str = ""

    def json(self) -> dict[str, Any]:
        return self.body


class QueueTransport:
    def __init__(self, *responses: FakeResponse) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, Any],
        timeout: float,
    ) -> FakeResponse:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "json": dict(json),
                "timeout": timeout,
            }
        )
        if not self.responses:
            raise AssertionError("No queued response")
        return self.responses.pop(0)


def token_response() -> FakeResponse:
    return FakeResponse(
        {
            "return_code": 0,
            "return_msg": "정상",
            "token": "test-token",
            "expires_dt": "20991231235959",
        }
    )


def config(
    mode: TradingMode = TradingMode.DEMO,
    *,
    allow_live_orders: bool = False,
) -> KiwoomConfig:
    return KiwoomConfig(
        app_key="app-key",
        secret_key="secret-key",
        mode=mode,
        min_request_interval_seconds=0,
        allow_live_orders=allow_live_orders,
    )


class QuoteTests(unittest.TestCase):
    def test_separate_domestic_and_us_configs_issue_tokens_with_each_key(self) -> None:
        domestic_config = KiwoomConfig(
            app_key="domestic-app-key",
            secret_key="domestic-secret-key",
            min_request_interval_seconds=0,
        )
        us_config = KiwoomConfig(
            app_key="us-app-key",
            secret_key="us-secret-key",
            min_request_interval_seconds=0,
        )
        transport = QueueTransport(
            token_response(),
            FakeResponse(
                {
                    "return_code": 0,
                    "stk_cd": "005930",
                    "stk_nm": "삼성전자",
                    "cur_prc": "70000",
                }
            ),
            token_response(),
            FakeResponse(
                {
                    "return_code": 0,
                    "stex_tp": "ND",
                    "stk_cd": "AAPL",
                    "stk_enm": "Apple Inc.",
                    "cur_prc": "213.04",
                }
            ),
        )
        broker = KiwoomBroker(
            domestic_config,
            us_config=us_config,
            transport=transport,
        )

        broker.quote_domestic("005930")
        broker.quote_us("AAPL", exchange=USExchange.NASDAQ)

        self.assertEqual(transport.calls[0]["json"]["appkey"], "domestic-app-key")
        self.assertEqual(transport.calls[2]["json"]["appkey"], "us-app-key")

    def test_domestic_quote_uses_demo_endpoint_and_normalizes_signed_price(self) -> None:
        transport = QueueTransport(
            token_response(),
            FakeResponse(
                {
                    "return_code": 0,
                    "stk_cd": "005930",
                    "stk_nm": "삼성전자",
                    "cur_prc": "+70000",
                    "pred_pre": "+1200",
                    "flu_rt": "+1.74",
                    "trde_qty": "123456",
                }
            ),
        )
        broker = KiwoomBroker(config(), transport=transport)

        quote = broker.quote_domestic("005930")

        self.assertEqual(quote.name, "삼성전자")
        self.assertEqual(quote.price, Decimal("70000"))
        self.assertEqual(quote.change_rate, Decimal("1.74"))
        self.assertEqual(transport.calls[1]["url"], "https://mockapi.kiwoom.com/api/dostk/stkinfo")
        self.assertEqual(transport.calls[1]["headers"]["api-id"], "ka10001")

    def test_us_quote_maps_exchange_and_currency(self) -> None:
        transport = QueueTransport(
            token_response(),
            FakeResponse(
                {
                    "return_code": 0,
                    "stex_tp": "ND",
                    "stk_cd": "AAPL",
                    "stk_enm": "Apple Inc.",
                    "cur_prc": "213.04",
                    "flu_rt": "0.75",
                    "curr_unit": "USD",
                }
            ),
        )
        broker = KiwoomBroker(config(), transport=transport)

        quote = broker.quote_us("aapl", exchange=USExchange.NASDAQ)

        self.assertEqual(quote.symbol, "AAPL")
        self.assertEqual(quote.price, Decimal("213.04"))
        self.assertEqual(transport.calls[1]["json"], {"stex_tp": "ND", "stk_cd": "AAPL"})


class CatalogTests(unittest.TestCase):
    def test_search_domestic_stock_by_name(self) -> None:
        transport = QueueTransport(
            token_response(),
            FakeResponse(
                {
                    "return_code": 0,
                    "list": [
                        {"code": "005930", "name": "삼성전자", "marketName": "KOSPI", "lastPrice": "68000"},
                        {"code": "000660", "name": "SK하이닉스", "marketName": "KOSPI", "lastPrice": "190000"},
                    ],
                }
            ),
        )
        broker = KiwoomBroker(config(), transport=transport)

        result = broker.list_domestic_stocks(market_codes=("0",), query="삼성")

        self.assertEqual([stock.symbol for stock in result], ["005930"])
        self.assertEqual(result[0].previous_close, Decimal("68000"))


class AccountTests(unittest.TestCase):
    def test_domestic_account_combines_positions_and_cash(self) -> None:
        transport = QueueTransport(
            token_response(),
            FakeResponse(
                {
                    "return_code": 0,
                    "tot_pur_amt": "65000",
                    "tot_evlt_amt": "70000",
                    "tot_evlt_pl": "5000",
                    "tot_prft_rt": "7.69",
                    "acnt_evlt_remn_indv_tot": [
                        {
                            "stk_cd": "A005930",
                            "stk_nm": "삼성전자",
                            "rmnd_qty": "1",
                            "trde_able_qty": "1",
                            "pur_pric": "65000",
                            "cur_prc": "+70000",
                            "evlt_amt": "70000",
                            "evltv_prft": "5000",
                            "prft_rt": "7.69",
                        }
                    ],
                }
            ),
            FakeResponse({"return_code": 0, "entr": "1000000", "ord_alow_amt": "900000"}),
        )
        broker = KiwoomBroker(config(), transport=transport)

        account = broker.account_domestic(exchange=DomesticExchange.KRX)

        self.assertEqual(account.cash, Decimal("1000000"))
        self.assertEqual(account.total_evaluation, Decimal("70000"))
        self.assertEqual(account.positions[0].symbol, "005930")
        self.assertEqual(account.positions[0].current_price, Decimal("70000"))


class OrderTests(unittest.TestCase):
    def test_real_order_requires_both_safety_gates(self) -> None:
        blocked_transport = QueueTransport()
        blocked = KiwoomBroker(config(TradingMode.REAL), transport=blocked_transport)
        order = blocked.build_order(
            market=Market.DOMESTIC,
            side=OrderSide.BUY,
            symbol="005930",
            quantity=1,
            exchange=DomesticExchange.KRX,
            price="70000",
        )
        with self.assertRaises(LiveOrderConfirmationRequired):
            blocked.place_order(order, confirm_live_order="LIVE_ORDER")
        self.assertEqual(blocked_transport.calls, [])

        confirmation_transport = QueueTransport()
        confirmation = KiwoomBroker(
            config(TradingMode.REAL, allow_live_orders=True),
            transport=confirmation_transport,
        )
        order = confirmation.build_order(
            market=Market.US,
            side=OrderSide.SELL,
            symbol="AAPL",
            quantity=1,
            exchange=USExchange.NASDAQ,
            price="200",
        )
        with self.assertRaises(LiveOrderConfirmationRequired):
            confirmation.place_order(order)
        self.assertEqual(confirmation_transport.calls, [])

    def test_demo_buy_submits_correct_api_payload(self) -> None:
        transport = QueueTransport(
            token_response(),
            FakeResponse({"return_code": 0, "return_msg": "정상", "ord_no": "12345"}),
        )
        broker = KiwoomBroker(config(), transport=transport)

        result = broker.buy(
            market=Market.US,
            symbol="AAPL",
            quantity=2,
            exchange=USExchange.NASDAQ,
            price="210.50",
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.order_number, "12345")
        self.assertEqual(transport.calls[1]["headers"]["api-id"], "ust20000")
        self.assertEqual(
            transport.calls[1]["json"],
            {
                "stex_tp": "ND",
                "stk_cd": "AAPL",
                "ord_qty": "2",
                "trde_tp": "00",
                "ord_uv": "210.50",
            },
        )

    def test_demo_cancel_order_uses_domestic_cancel_api(self) -> None:
        transport = QueueTransport(
            token_response(),
            FakeResponse(
                {
                    "return_code": 0,
                    "return_msg": "정상",
                    "ord_no": "0000200",
                    "base_orig_ord_no": "0000140",
                    "cncl_qty": "1",
                }
            ),
        )
        broker = KiwoomBroker(config(), transport=transport)

        result = broker.cancel_order(
            market=Market.DOMESTIC,
            original_order_number="0000140",
            symbol="005930",
            exchange=DomesticExchange.KRX,
            quantity=1,
        )

        self.assertEqual(result.cancel_order_number, "0000200")
        self.assertEqual(result.cancelled_quantity, Decimal("1"))
        self.assertEqual(transport.calls[1]["headers"]["api-id"], "kt10003")

    def test_lists_open_orders_for_fill_verification(self) -> None:
        transport = QueueTransport(
            token_response(),
            FakeResponse(
                {
                    "return_code": 0,
                    "result_list": [
                        {
                            "ord_no": "000000047",
                            "stex_nm": "NASDAQ",
                            "stk_cd": "AAPL",
                            "frgn_stk_nm": "Apple Inc.",
                            "slby_tp_nm": "매수",
                            "ord_stat": "접수",
                            "ord_qty": "2",
                            "cntr_qty": "1",
                            "ord_remnq": "1",
                            "ord_uv": "210.50",
                        }
                    ],
                }
            ),
        )
        broker = KiwoomBroker(config(), transport=transport)

        orders = broker.list_open_orders(Market.US, exchange=USExchange.ALL)

        self.assertEqual(orders[0].symbol, "AAPL")
        self.assertEqual(orders[0].remaining_quantity, Decimal("1"))
        self.assertEqual(transport.calls[1]["headers"]["api-id"], "ust21050")


if __name__ == "__main__":
    unittest.main()
