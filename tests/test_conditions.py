from __future__ import annotations

import json
import unittest
from collections import deque
from decimal import Decimal
from typing import Any

from dockdack.conditions import KiwoomConditionClient
from dockdack.config import KiwoomConfig
from dockdack.models import Market


class FakeWebSocket:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages = deque(json.dumps(message, ensure_ascii=False) for message in messages)
        self.sent: list[dict[str, Any]] = []
        self.closed = False

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    async def recv(self) -> str:
        return self.messages.popleft()

    async def close(self) -> None:
        self.closed = True


class ConditionTests(unittest.IsolatedAsyncioTestCase):
    async def test_lists_saved_domestic_conditions(self) -> None:
        websocket = FakeWebSocket(
            [
                {"trnm": "LOGIN", "return_code": 0},
                {"trnm": "CNSRLST", "return_code": 0, "data": [["1", "거래량 급증"]]},
            ]
        )

        async def connect(uri: str, timeout: float) -> FakeWebSocket:
            self.assertEqual(uri, "wss://mockapi.kiwoom.com:10000/api/dostk/websocket")
            return websocket

        client = KiwoomConditionClient(
            KiwoomConfig(
                app_key="app",
                secret_key="secret",
                min_request_interval_seconds=0,
            ),
            lambda: "token",
            connect_factory=connect,
        )

        result = await client.list_conditions(Market.DOMESTIC)

        self.assertEqual(result[0].name, "거래량 급증")
        self.assertEqual(websocket.sent[0], {"trnm": "LOGIN", "token": "token"})
        self.assertTrue(websocket.closed)

    async def test_condition_result_contains_stock_name_and_price(self) -> None:
        websocket = FakeWebSocket(
            [
                {"trnm": "LOGIN", "return_code": 0},
                {
                    "trnm": "GCNSRREQ",
                    "return_code": 0,
                    "cont_yn": "N",
                    "data": [
                        {
                            "9001": "AAPL",
                            "302": "Apple Inc.",
                            "10": "213.04",
                            "12": "1.2",
                            "13": "1000000",
                            "stex_tp": "ND",
                        }
                    ],
                },
            ]
        )

        async def connect(uri: str, timeout: float) -> FakeWebSocket:
            return websocket

        client = KiwoomConditionClient(
            KiwoomConfig(
                app_key="app",
                secret_key="secret",
                min_request_interval_seconds=0,
            ),
            lambda: "token",
            connect_factory=connect,
        )

        result = await client.run_condition(Market.US, "0")

        self.assertEqual(result[0].symbol, "AAPL")
        self.assertEqual(result[0].name, "Apple Inc.")
        self.assertEqual(result[0].price, Decimal("213.04"))
        self.assertEqual(websocket.sent[1]["trnm"], "GCNSRREQ")


if __name__ == "__main__":
    unittest.main()
