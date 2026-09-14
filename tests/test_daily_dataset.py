from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from dockdack import DailyBar, KiwoomBroker, KiwoomConfig, Market, StockInfo
from dockdack.daily_dataset import DailyDatasetStore, Instrument


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

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, Any],
        timeout: float,
    ) -> FakeResponse:
        del method, url, headers, json, timeout
        return self.responses.pop(0)


def token_response() -> FakeResponse:
    return FakeResponse(
        {
            "return_code": 0,
            "token": "token",
            "expires_dt": "20991231235959",
        }
    )


class DailyBarTests(unittest.TestCase):
    def test_domestic_daily_bars_follow_continuation_and_normalize_prices(self) -> None:
        transport = QueueTransport(
            token_response(),
            FakeResponse(
                {
                    "return_code": 0,
                    "stk_dt_pole_chart_qry": [
                        {
                            "dt": "20260821",
                            "open_pric": "-70000",
                            "high_pric": "+71000",
                            "low_pric": "-69000",
                            "cur_prc": "+70500",
                            "trde_qty": "123",
                        }
                    ],
                },
                headers={"cont-yn": "Y", "next-key": "page-2"},
            ),
            FakeResponse(
                {
                    "return_code": 0,
                    "stk_dt_pole_chart_qry": [
                        {
                            "dt": "20260820",
                            "open_pric": "68000",
                            "high_pric": "70000",
                            "low_pric": "67000",
                            "cur_prc": "69000",
                            "trde_qty": "456",
                        }
                    ],
                }
            ),
        )
        config = KiwoomConfig(
            app_key="key",
            secret_key="secret",
            min_request_interval_seconds=0,
        )
        broker = KiwoomBroker(config, transport=transport)

        bars = broker.daily_bars_domestic("005930", max_pages=2)

        self.assertEqual([bar.trade_date.isoformat() for bar in bars], ["2026-08-21", "2026-08-20"])
        self.assertEqual(bars[0].open, Decimal("70000"))
        self.assertEqual(bars[0].close, Decimal("70500"))


class DatasetStoreTests(unittest.TestCase):
    def test_page_commit_is_queryable_and_marks_symbol_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "daily.sqlite3"
            with DailyDatasetStore(path, Market.DOMESTIC) as store:
                store.save_catalog(
                    [
                        StockInfo(
                            market=Market.DOMESTIC,
                            symbol="005930",
                            name="삼성전자",
                            exchange="KOSPI",
                        )
                    ]
                )
                instrument = Instrument("005930", "KRX", "삼성전자")
                store.start_fresh_pass(instrument)
                store.save_page(
                    instrument,
                    [
                        DailyBar(
                            market=Market.DOMESTIC,
                            symbol="005930",
                            exchange="KRX",
                            trade_date=date(2026, 8, 21),
                            open=Decimal("70000"),
                            high=Decimal("71000"),
                            low=Decimal("69000"),
                            close=Decimal("70500"),
                            volume=Decimal("123456"),
                            currency="KRW",
                        )
                    ],
                    has_next=False,
                    cont_yn=None,
                    next_key=None,
                )

                self.assertEqual(store.stats()["bars"], 1)
                self.assertEqual(store.progress(instrument)["status"], "complete")
                row = store.connection.execute(
                    "SELECT close, volume FROM daily_bars WHERE symbol='005930'"
                ).fetchone()
                self.assertEqual(tuple(row), ("70500", 123456))

    def test_us_catalog_excludes_pink_sheet_not_supported_by_daily_chart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "daily.sqlite3"
            with DailyDatasetStore(path, Market.US) as store:
                count = store.save_catalog(
                    [
                        StockInfo(Market.US, "AAPL", "Apple", "ND"),
                        StockInfo(Market.US, "AABB", "Pink Sheet", "NP"),
                    ]
                )
                self.assertEqual(count, 1)
                self.assertEqual(store.instruments(), (Instrument("AAPL", "ND", "Apple"),))


if __name__ == "__main__":
    unittest.main()
