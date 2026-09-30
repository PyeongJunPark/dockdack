"""On-demand minute feed never turns chart/cache uncertainty into a trade input."""

from datetime import datetime, timedelta
from decimal import Decimal
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zoneinfo import ZoneInfo

from dockdack.broker.kiwoom import MinuteBars
from dockdack.minute_feed import DomesticMinuteFeed, MinuteFeedUnavailable
from dockdack.models import Market, MinuteBar, TradingMode


KST = ZoneInfo("Asia/Seoul")
DAY = datetime(2026, 9, 29, 9, 10, tzinfo=KST)


def bar(minutes: int, *, symbol: str = "005930", close: str = "70000",
        market: Market = Market.DOMESTIC, exchange: str = "KRX") -> MinuteBar:
    label = DAY + timedelta(minutes=minutes)
    price = Decimal(close)
    return MinuteBar(market, symbol, exchange, label, price, price,
                     price, price, Decimal("100"), "KRW")


class FakeBroker:
    mode = TradingMode.DEMO

    def __init__(self, responses=()):
        self.responses = list(responses)
        self.calls = []

    def minute_bars_domestic(self, symbol, **kwargs):
        self.calls.append((symbol, kwargs))
        if not self.responses:
            raise AssertionError("unexpected broker call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def page(*bars, truncated=False):
    return MinuteBars((bars,), truncated=truncated)


class MinuteFeedTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.now = datetime(2026, 9, 29, 10, 6, tzinfo=KST)

    def feed(self, broker):
        return DomesticMinuteFeed(broker, Path(self.temp.name), clock=lambda: self.now)

    def test_same_completed_slot_reuses_cache_for_many_models_and_restart(self):
        broker = FakeBroker([page(bar(40), bar(45), bar(50))])
        feed = self.feed(broker)
        first = feed.get_complete_bars("005930", "KRX", count=3)
        second = feed.get_complete_bars("005930", "KRX", count=2)
        reloaded = self.feed(broker).get_complete_bars("005930", "KRX", count=3)
        self.assertEqual(len(broker.calls), 1)
        self.assertFalse(first.from_cache)
        self.assertTrue(second.from_cache)
        self.assertTrue(reloaded.from_cache)
        self.assertEqual([item.timestamp for item in first.bars],
                         [bar(40).timestamp, bar(45).timestamp, bar(50).timestamp])
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(first.sha256, reloaded.sha256)
        self.assertTrue(first.jsonl_path.is_file())
        self.assertTrue(first.receipt_path.is_file())
        self.assertEqual(broker.calls[0][1]["max_pages"], 1)
        self.assertEqual(broker.calls[0][1]["base_date"], self.now.date())

    def test_next_slot_fetches_only_latest_page_and_merges_new_label(self):
        broker = FakeBroker([page(bar(40), bar(45), bar(50)),
                             page(bar(55), bar(50))])
        feed = self.feed(broker)
        feed.get_complete_bars("005930", "KRX", count=3)
        self.now = datetime(2026, 9, 29, 10, 11, tzinfo=KST)
        updated = feed.get_complete_bars("005930", "KRX", count=3)
        self.assertEqual(len(broker.calls), 2)
        self.assertEqual(updated.last_bar_label, bar(55).timestamp)
        self.assertEqual([item.timestamp for item in updated.bars],
                         [bar(45).timestamp, bar(50).timestamp, bar(55).timestamp])

    def test_conflicting_same_timestamp_does_not_overwrite_previous_receipt(self):
        broker = FakeBroker([page(bar(40), bar(45), bar(50)),
                             page(bar(55), bar(50, close="70100"))])
        feed = self.feed(broker)
        previous = feed.get_complete_bars("005930", "KRX", count=3)
        self.now = datetime(2026, 9, 29, 10, 11, tzinfo=KST)
        with self.assertRaisesRegex(MinuteFeedUnavailable, "조회·검증이 실패"):
            feed.get_complete_bars("005930", "KRX", count=3)
        self.assertEqual(previous.sha256,
                         self.feed(FakeBroker())._read(previous.jsonl_path,
                                                        previous.receipt_path,
                                                        "005930", self.now.date()).sha256)

    def test_gap_or_stale_bars_fail_closed(self):
        broker = FakeBroker([page(bar(30), bar(40), bar(50))])
        with self.assertRaisesRegex(MinuteFeedUnavailable, "연속"):
            self.feed(broker).get_complete_bars("005930", "KRX", count=3)
        late = FakeBroker([page(bar(30), bar(35), bar(40))])
        with self.assertRaisesRegex(MinuteFeedUnavailable, "지연"):
            self.feed(late).get_complete_bars("005930", "KRX", count=3)

    def test_forming_bar_boundary_and_wrong_identity_are_excluded_or_rejected(self):
        broker = FakeBroker([page(bar(45), bar(50), bar(55))])
        result = self.feed(broker).get_complete_bars("005930", "KRX", count=2)
        self.assertEqual(result.last_bar_label, bar(50).timestamp)
        wrong = FakeBroker([page(bar(45, symbol="005930"))])
        with self.assertRaisesRegex(MinuteFeedUnavailable, "조회·검증이 실패"):
            self.feed(wrong).get_complete_bars("000660", "KRX", count=1)
        boundary = FakeBroker([page(bar(-5, symbol="114800"))])  # 09:05 is not a safe label.
        with self.assertRaises(MinuteFeedUnavailable):
            self.feed(boundary).get_complete_bars("114800", "KRX", count=1)

    def test_cache_tamper_is_not_silently_served(self):
        broker = FakeBroker([page(bar(45), bar(50))])
        feed = self.feed(broker)
        snap = feed.get_complete_bars("005930", "KRX", count=2)
        snap.jsonl_path.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(MinuteFeedUnavailable, "무결성"):
            feed.get_complete_bars("005930", "KRX", count=2)
        with self.assertRaisesRegex(MinuteFeedUnavailable, "무결성"):
            self.feed(FakeBroker()).get_complete_bars("005930", "KRX", count=2)

    def test_replayed_receipt_future_and_non_utc_fetch_time_are_rejected(self):
        for symbol, changed in (("005930", "2026-09-29T01:07:00+00:00"),
                                ("000660", "2026-09-29T10:06:00+09:00")):
            with self.subTest(changed=changed):
                broker = FakeBroker([page(bar(45, symbol=symbol), bar(50, symbol=symbol))])
                snap = self.feed(broker).get_complete_bars(symbol, "KRX", count=2)
                receipt = json.loads(snap.receipt_path.read_text(encoding="utf-8"))
                receipt["fetched_at_utc"] = changed
                snap.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                with self.assertRaises(MinuteFeedUnavailable):
                    self.feed(FakeBroker()).get_complete_bars(symbol, "KRX", count=2)

    def test_loaded_cache_with_off_session_bar_is_rejected_even_if_rehashed(self):
        broker = FakeBroker([page(bar(45), bar(50))])
        snap = self.feed(broker).get_complete_bars("005930", "KRX", count=2)
        rows = [json.loads(line) for line in snap.jsonl_path.read_text(encoding="utf-8").splitlines()]
        rows[0]["timestamp"] = "2026-09-29T09:05:00+09:00"
        payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")) + "\n" for row in rows)
        from hashlib import sha256
        snap.jsonl_path.write_bytes(payload.encode("utf-8"))
        receipt = json.loads(snap.receipt_path.read_text(encoding="utf-8"))
        receipt["sha256"] = sha256(payload.encode("utf-8")).hexdigest()
        snap.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(MinuteFeedUnavailable, "시각·장 구간"):
            self.feed(FakeBroker()).get_complete_bars("005930", "KRX", count=2)

    def test_new_slot_requires_newly_completed_label(self):
        self.now = datetime(2026, 9, 29, 10, 10, tzinfo=KST)
        broker = FakeBroker([page(bar(45), bar(50))])
        with self.assertRaisesRegex(MinuteFeedUnavailable, "지연"):
            self.feed(broker).get_complete_bars("005930", "KRX", count=2)

    def test_late_completed_candle_retries_once_after_delay_in_same_slot(self):
        self.now = datetime(2026, 9, 29, 10, 10, tzinfo=KST)
        broker = FakeBroker([page(bar(45), bar(50)), page(bar(50), bar(55))])
        feed = self.feed(broker)
        with self.assertRaisesRegex(MinuteFeedUnavailable, "지연"):
            feed.get_complete_bars("005930", "KRX", count=2)
        self.now += timedelta(seconds=10)
        with self.assertRaisesRegex(MinuteFeedUnavailable, "이번 완료 슬롯"):
            feed.get_complete_bars("005930", "KRX", count=2)
        self.assertEqual(len(broker.calls), 1)
        self.now += timedelta(seconds=15)
        result = feed.get_complete_bars("005930", "KRX", count=2)
        self.assertEqual(len(broker.calls), 2)
        self.assertEqual(result.last_bar_label, bar(55).timestamp)
        self.assertEqual([row.timestamp for row in result.bars],
                         [bar(50).timestamp, bar(55).timestamp])

    def test_no_stale_fallback_on_network_failure(self):
        broker = FakeBroker([page(bar(45), bar(50)), RuntimeError("offline")])
        feed = self.feed(broker)
        feed.get_complete_bars("005930", "KRX", count=2)
        self.now = datetime(2026, 9, 29, 10, 11, tzinfo=KST)
        with self.assertRaisesRegex(MinuteFeedUnavailable, "조회·검증이 실패"):
            feed.get_complete_bars("005930", "KRX", count=2)
        with self.assertRaisesRegex(MinuteFeedUnavailable, "조회·검증이 실패"):
            feed.get_complete_bars("005930", "KRX", count=2)
        self.assertEqual(len(broker.calls), 2)

    def test_insufficient_window_is_not_requeried_for_each_model(self):
        broker = FakeBroker([page(bar(45), bar(50))])
        feed = self.feed(broker)
        with self.assertRaises(MinuteFeedUnavailable):
            feed.get_complete_bars("005930", "KRX", count=3)
        with self.assertRaisesRegex(MinuteFeedUnavailable, "이번 완료 슬롯"):
            feed.get_complete_bars("005930", "KRX", count=3)
        shorter = feed.get_complete_bars("005930", "KRX", count=2)
        self.assertTrue(shorter.from_cache)
        self.assertEqual(len(broker.calls), 1)

    def test_real_mode_us_market_and_holiday_never_call_chart(self):
        real = FakeBroker()
        real.mode = TradingMode.REAL
        with self.assertRaisesRegex(MinuteFeedUnavailable, "실전"):
            self.feed(real)
        demo = FakeBroker()
        feed = self.feed(demo)
        for symbol, exchange in (("AAPL", "KRX"), ("005930", "NXT")):
            with self.assertRaises(MinuteFeedUnavailable):
                feed.get_complete_bars(symbol, exchange, count=1)
        self.now = datetime(2026, 10, 3, 10, 6, tzinfo=KST)
        with self.assertRaises(MinuteFeedUnavailable):
            feed.get_complete_bars("005930", "KRX", count=1)
        self.assertEqual(demo.calls, [])

    def test_escalate_pages_only_if_contiguous_window_is_insufficient(self):
        broker = FakeBroker([
            page(bar(50), truncated=True),
            MinuteBars(((bar(50),), (bar(45), bar(40))), truncated=True),
        ])
        feed = self.feed(broker)
        result = feed.get_complete_bars("005930", "KRX", count=3)
        self.assertEqual(len(broker.calls), 2)
        self.assertEqual([call[1]["max_pages"] for call in broker.calls], [1, 2])
        self.assertEqual(len(result.bars), 3)


if __name__ == "__main__":
    unittest.main()
