"""Offline collector contract; no credentials, network or account access."""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from dockdack import BrokerAPIError, KiwoomBroker
from dockdack.models import Market, TradingMode
from examples.collect_minute_research import _rows, collect
from test_kiwoom import QueueTransport, config


def bar(stamp="2026-09-29T09:10:00+09:00", close="101"):
    return SimpleNamespace(
        timestamp=datetime.fromisoformat(stamp), open=Decimal("100"),
        high=Decimal("102"), low=Decimal("99"), close=Decimal(close),
        volume=Decimal("1000"))


class FakeBroker:
    mode = TradingMode.DEMO

    def __init__(self, bars, index_bars=()):
        self.bars = tuple(bars)
        self.index_bars = tuple(index_bars)
        self.calls = []

    def minute_bars_domestic(self, symbol, **kwargs):
        self.calls.append(("domestic", symbol, kwargs))
        return self.bars

    def minute_bars_us(self, symbol, **kwargs):
        self.calls.append(("us", symbol, kwargs))
        return self.bars

    def minute_bars_domestic_index(self, index_code, **kwargs):
        self.calls.append(("index", index_code, kwargs))
        return self.index_bars


class MinuteResearchCollectorTests(unittest.TestCase):
    def test_collect_demo_bar_with_receipt_and_no_secret_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "minute.jsonl"
            broker = FakeBroker((bar(), bar(), bar("2026-09-29T09:15:00+09:00")))
            receipt = collect(market=Market.DOMESTIC, exchange="KRX", symbols=("005930",),
                              interval_minutes=5, max_pages=2, output=output, broker=broker)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(receipt["exact_duplicates"], 1)
            self.assertEqual(receipt["accepted_bars"], 2)
            self.assertTrue(receipt["regular_session_filter"])
            self.assertFalse(receipt["holiday_and_early_close_calendar_verified"])
            self.assertIsNone(receipt["chart_truncated"])
            self.assertEqual(rows[0]["market"], "domestic")
            self.assertNotIn("app_key", output.read_text(encoding="utf-8"))
            self.assertTrue(output.with_suffix(".jsonl.receipt.json").exists())
            self.assertEqual(broker.calls[0][2]["max_pages"], 2)
            with self.assertRaises(FileExistsError):
                collect(market=Market.DOMESTIC, exchange="KRX", symbols=("005930",),
                        interval_minutes=5, max_pages=2, output=output, broker=broker)

    def test_conflicting_duplicate_timestamp_is_quarantined(self):
        broker = FakeBroker((bar(), bar(close="98"), bar("2026-09-29T09:15:00+09:00")))
        rows, stats = _rows(broker, Market.DOMESTIC, "KRX", ("005930",), 5, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(stats["conflicting_timestamps_dropped"], 1)

    def test_rejects_empty_naive_and_real_mode_without_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "minute.jsonl"
            args = dict(market=Market.DOMESTIC, exchange="KRX", symbols=("005930",),
                        interval_minutes=5, max_pages=1, output=output)
            with self.assertRaises(ValueError):
                collect(**args, broker=FakeBroker(()))
            with self.assertRaises(ValueError):
                collect(**args, broker=FakeBroker((bar("2026-09-29T09:05:00"),)))
            real = FakeBroker((bar(),))
            real.mode = TradingMode.REAL
            with self.assertRaises(ValueError):
                collect(**args, broker=real)
            self.assertEqual(real.calls, [])
            self.assertFalse(output.exists())

    def test_us_pipeline_formats_only_prevalidated_fake_bars(self):
        broker = FakeBroker((bar("2026-09-29T09:40:00-04:00"),))
        rows, _ = _rows(broker, Market.US, "ND", ("AAPL",), 5, 3)
        self.assertEqual(rows[0]["market"], "us")
        self.assertEqual(broker.calls, [("us", "AAPL", {"exchange": "ND",
                                                      "interval_minutes": 5, "max_pages": 3})])

    def test_real_us_adapter_fails_before_network_or_output(self):
        transport = QueueTransport()
        broker = KiwoomBroker(config(), transport=transport)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "us-minute.jsonl"
            with self.assertRaisesRegex(BrokerAPIError, "시간대"):
                collect(market=Market.US, exchange="ND", symbols=("AAPL",),
                        interval_minutes=5, max_pages=1, output=output,
                        broker=broker)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".jsonl.receipt.json").exists())
        self.assertEqual(transport.calls, [])

    def test_domestic_outside_and_uncertain_edge_bars_are_removed(self):
        broker = FakeBroker(tuple(bar(f"2026-09-29T{clock}:00+09:00") for clock in (
            "08:59", "09:00", "09:05", "09:10", "15:20", "15:25", "15:30", "15:31")))
        rows, stats = _rows(broker, Market.DOMESTIC, "KRX", ("005930",), 5, 1)
        self.assertEqual([row["timestamp"][11:16] for row in rows], ["09:10", "15:20"])
        self.assertEqual(stats["raw_bars"], 8)
        self.assertEqual(stats["outside_regular_hours_dropped"], 2)
        self.assertEqual(stats["ambiguous_session_boundary_dropped"], 4)

    def test_us_dst_offset_is_checked_and_mismatch_fails_closed(self):
        for stamp in ("2026-09-29T09:40:00-04:00", "2026-12-10T09:40:00-05:00"):
            with self.subTest(stamp=stamp):
                rows, _ = _rows(FakeBroker((bar(stamp),)), Market.US, "ND", ("AAPL",), 5, 1)
                self.assertEqual(len(rows), 1)
        with self.assertRaises(ValueError):
            _rows(FakeBroker((bar("2026-09-29T09:40:00-05:00"),)),
                  Market.US, "ND", ("AAPL",), 5, 1)

    def test_receipt_marks_real_broker_page_truncation(self):
        class LimitedBroker(FakeBroker):
            def minute_bars_domestic(self, symbol, **kwargs):
                result = super().minute_bars_domestic(symbol, **kwargs)

                class Slice(tuple):
                    page_count = 1
                    truncated = True

                return Slice(result)

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "minute.jsonl"
            receipt = collect(market=Market.DOMESTIC, exchange="KRX", symbols=("005930",),
                              interval_minutes=5, max_pages=1, output=output,
                              broker=LimitedBroker((bar(),)))
            self.assertTrue(receipt["chart_truncated"])
            self.assertEqual(receipt["chart_page_counts"], {"005930": 1})
            self.assertEqual(receipt["chart_truncated_symbols"], ["005930"])

    def test_optional_index_snapshot_is_distinct_from_tradable_equity(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "minute.jsonl"
            broker = FakeBroker((bar(),), (bar(close="252127"),))
            receipt = collect(market=Market.DOMESTIC, exchange="KRX", symbols=("005930",),
                              interval_minutes=5, max_pages=1, output=output,
                              broker=broker, index_code="201")
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({(row["exchange"], row["symbol"]) for row in rows},
                             {("KRX", "005930"), ("INDEX", "201")})
            self.assertEqual(receipt["index_code"], "201")
            self.assertIn("signed_abs_divided_by_100_points", receipt["index_price_basis"])
            self.assertEqual(broker.calls[-1], ("index", "201", {
                "interval_minutes": 5, "max_pages": 1,
            }))

    def test_us_market_cannot_request_domestic_index(self):
        broker = FakeBroker((bar("2026-09-29T09:40:00-04:00"),))
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                collect(market=Market.US, exchange="ND", symbols=("AAPL",),
                        interval_minutes=5, max_pages=1,
                        output=Path(folder) / "minute.jsonl", broker=broker,
                        index_code="201")
        self.assertEqual(broker.calls, [])

    def test_index_only_snapshot_never_calls_stock_chart(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "kospi200.jsonl"
            broker = FakeBroker((), (bar(close="252127"),))
            receipt = collect(market=Market.DOMESTIC, exchange="INDEX", symbols=(),
                              interval_minutes=5, max_pages=1, output=output,
                              broker=broker, index_code="201")
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(receipt["symbols"], [])
            self.assertEqual(receipt["exchange"], "INDEX")
            self.assertEqual([(row["exchange"], row["symbol"]) for row in rows],
                             [("INDEX", "201")])
            self.assertEqual([call[0] for call in broker.calls], ["index"])

    def test_empty_symbols_without_index_still_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                collect(market=Market.DOMESTIC, exchange="KRX", symbols=(),
                        interval_minutes=5, max_pages=1,
                        output=Path(folder) / "minute.jsonl", broker=FakeBroker(()))


if __name__ == "__main__":
    unittest.main()
