"""Minute-chart access is read-only and never exposes a forming interval."""

from datetime import date, datetime, timezone
from decimal import Decimal
import unittest
from zoneinfo import ZoneInfo

from dockdack import BrokerAPIError, DomesticExchange, KiwoomBroker, Market, USExchange
from dockdack.broker.kiwoom import _minute_bar, _minute_complete
from dockdack.http import _READ_ONLY_APIS
from dockdack.models import MinuteBar
from test_kiwoom import FakeResponse, QueueTransport, config, token_response


def domestic_row(stamp="20260929100000", **changes):
    row = {
        "cntr_tm": stamp, "open_pric": "-78850", "high_pric": "-78900",
        "low_pric": "-78800", "cur_prc": "-78820", "trde_qty": "7913",
    }
    row.update(changes)
    return row


def us_row(stamp="20260928100000", **changes):
    row = {
        "cntr_tm": stamp, "bus_dt": stamp[:8], "open_pric": "201.3900",
        "high_pric": "202.0000", "low_pric": "201.0000",
        "cur_prc": "201.5000", "trde_qty": "2",
    }
    row.update(changes)
    return row


class MinuteBarTests(unittest.TestCase):
    def test_domestic_signed_prices_and_forming_bar_excluded(self):
        transport = QueueTransport(token_response(), FakeResponse({
            "stk_cd": "005930", "stk_min_pole_chart_qry": [
                domestic_row(), domestic_row("20260929095900"),
            ],
        }))
        broker = KiwoomBroker(config(), transport=transport)
        rows = broker.minute_bars_domestic(
            "005930", interval_minutes=1, base_date=date(2026, 9, 29),
            as_of=datetime(2026, 9, 29, 10, 0, 30, tzinfo=ZoneInfo("Asia/Seoul")),
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.page_count, 1)
        self.assertFalse(rows.truncated)
        self.assertIsInstance(rows[0], MinuteBar)
        self.assertEqual(rows[0].market, Market.DOMESTIC)
        self.assertEqual(rows[0].timestamp.isoformat(), "2026-09-29T09:59:00+09:00")
        self.assertEqual(rows[0].open, Decimal("78850"))
        self.assertEqual(rows[0].high, Decimal("78900"))
        self.assertEqual(rows[0].volume, Decimal("7913"))
        call = transport.calls[-1]
        self.assertEqual(call["headers"]["api-id"], "ka10080")
        self.assertEqual(call["url"], "https://mockapi.kiwoom.com/api/dostk/chart")
        self.assertEqual(call["json"], {
            "stk_cd": "005930", "tic_scope": "1", "upd_stkpc_tp": "1", "base_dt": "20260929",
        })

    def test_domestic_exchange_suffix_and_page_continuation(self):
        transport = QueueTransport(
            token_response(),
            FakeResponse({"stk_cd": "005930", "stk_min_pole_chart_qry": [
                domestic_row("20260929100000"),
            ]}, headers={"cont-yn": "Y", "next-key": "page2"}),
            FakeResponse({"stk_cd": "005930", "stk_min_pole_chart_qry": [
                domestic_row("20260929095500"),
            ]}),
        )
        broker = KiwoomBroker(config(), transport=transport)
        rows = broker.minute_bars_domestic(
            "005930", exchange=DomesticExchange.NXT, interval_minutes=5,
            adjusted=False,
            as_of=datetime(2026, 9, 29, 10, 6, tzinfo=ZoneInfo("Asia/Seoul")),
            max_pages=2,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].exchange, "NXT")
        self.assertEqual(transport.calls[1]["json"]["stk_cd"], "005930_NX")
        self.assertEqual(transport.calls[1]["json"]["upd_stkpc_tp"], "0")
        self.assertEqual(transport.calls[2]["headers"]["next-key"], "page2")

    def test_us_public_chart_is_disabled_before_network(self):
        transport = QueueTransport()
        broker = KiwoomBroker(config(), transport=transport)
        with self.assertRaisesRegex(BrokerAPIError, "시간대"):
            broker.minute_bars_us(
                "NVDA", exchange=USExchange.NASDAQ, interval_minutes=5,
                start_date="20260309",
                as_of=datetime(2026, 3, 9, 14, 6, tzinfo=timezone.utc),
            )
        with self.assertRaisesRegex(BrokerAPIError, "시간대"):
            list(broker.iter_minute_bars_us(
                "NVDA", exchange=USExchange.NASDAQ, interval_minutes=5,
                as_of=datetime(2026, 3, 9, 14, 6, tzinfo=timezone.utc),
            ))
        self.assertEqual(transport.calls, [])
        self.assertNotIn(("usa06011", "/api/us/chart"), _READ_ONLY_APIS)

    def test_us_parser_accepts_standard_clock_only_when_zone_is_explicit(self):
        row = us_row("20260309100000")
        with self.assertRaisesRegex(BrokerAPIError, "원본 시간대"):
            _minute_bar(row, Market.US, "NVDA", "ND", "USD")
        parsed = _minute_bar(row, Market.US, "NVDA", "ND", "USD",
                             source_timezone=ZoneInfo("America/New_York"))
        self.assertEqual(parsed.timestamp.isoformat(), "2026-03-09T10:00:00-04:00")
        self.assertEqual(parsed.currency, "USD")
        self.assertEqual(parsed.open, Decimal("201.3900"))
        self.assertTrue(_minute_complete(
            parsed, 5, datetime(2026, 3, 9, 14, 6, tzinfo=timezone.utc)))
        self.assertFalse(_minute_complete(
            parsed, 5, datetime(2026, 3, 9, 14, 3, tzinfo=timezone.utc)))

    def test_us_extended_business_day_hour_is_rejected_without_normalization(self):
        with self.assertRaisesRegex(BrokerAPIError, "유효한 날짜"):
            _minute_bar(us_row("20260928244500"), Market.US, "AAPL", "ND", "USD",
                        source_timezone=ZoneInfo("America/New_York"))

    def test_domestic_index_uses_ka20005_signed_values_and_point_scale(self):
        transport = QueueTransport(token_response(), FakeResponse({
            "inds_cd": "201", "inds_min_pole_qry": [{
                "cntr_tm": "20260929132000", "open_pric": "+108082",
                "high_pric": "+108249", "low_pric": "+108016",
                "cur_prc": "-108176", "trde_qty": "522",
            }],
        }))
        rows = KiwoomBroker(config(), transport=transport).minute_bars_domestic_index(
            "201", interval_minutes=5,
            as_of=datetime(2026, 9, 29, 13, 26, tzinfo=ZoneInfo("Asia/Seoul")),
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].symbol, "201")
        self.assertEqual(rows[0].exchange, "INDEX")
        self.assertEqual(rows[0].currency, "POINT")
        self.assertEqual(rows[0].close, Decimal("1081.76"))
        self.assertEqual(rows[0].high, Decimal("1082.49"))
        self.assertEqual(rows[0].raw["cur_prc"], "-108176")
        self.assertEqual(transport.calls[-1]["headers"]["api-id"], "ka20005")
        self.assertEqual(transport.calls[-1]["json"], {"inds_cd": "201", "tic_scope": "5"})

    def test_index_does_not_accept_daily_key_as_minute_rows(self):
        transport = QueueTransport(token_response(), FakeResponse({
            "inds_cd": "201", "inds_dt_pole_qry": [],
        }))
        with self.assertRaises(BrokerAPIError):
            KiwoomBroker(config(), transport=transport).minute_bars_domestic_index(
                "201", as_of=datetime(2026, 9, 29, 16, tzinfo=ZoneInfo("Asia/Seoul")),
            )

    def test_index_zero_price_is_not_a_tradeable_research_bar(self):
        transport = QueueTransport(token_response(), FakeResponse({
            "inds_min_pole_qry": [{
                "cntr_tm": "20260929132000", "open_pric": "0", "high_pric": "0",
                "low_pric": "0", "cur_prc": "0", "trde_qty": "1",
            }],
        }))
        with self.assertRaises(BrokerAPIError):
            KiwoomBroker(config(), transport=transport).minute_bars_domestic_index(
                "201", as_of=datetime(2026, 9, 29, 16, tzinfo=ZoneInfo("Asia/Seoul")),
            )

    def test_malformed_rows_are_rejected_instead_of_silently_dropped(self):
        failures = [
            domestic_row(cntr_tm="202609291000"),
            domestic_row(cntr_tm="20261329100000"),
            domestic_row(open_pric="NaN"),
            domestic_row(high_pric="-78800"),
            domestic_row(trde_qty="-1"),
            domestic_row(trde_qty="1.5"),
        ]
        for row in failures:
            with self.subTest(row=row):
                transport = QueueTransport(token_response(), FakeResponse({
                    "stk_min_pole_chart_qry": [row],
                }))
                with self.assertRaises(BrokerAPIError):
                    KiwoomBroker(config(), transport=transport).minute_bars_domestic(
                        "005930", as_of=datetime(2026, 9, 29, 11, tzinfo=timezone.utc),
                    )
        transport = QueueTransport(token_response(), FakeResponse({"stk_min_pole_chart_qry": {}}))
        with self.assertRaises(BrokerAPIError):
            KiwoomBroker(config(), transport=transport).minute_bars_domestic(
                "005930", as_of=datetime(2026, 9, 29, 11, tzinfo=timezone.utc),
            )

    def test_us_date_mismatch_or_negative_price_rejected(self):
        for row in (us_row(bus_dt="20260927"), us_row(open_pric="-201.3900")):
            with self.subTest(row=row):
                with self.assertRaises(BrokerAPIError):
                    _minute_bar(row, Market.US, "NVDA", "ND", "USD",
                                source_timezone=ZoneInfo("America/New_York"))

    def test_invalid_parameters_fail_before_network_and_page_limit_is_bounded(self):
        transport = QueueTransport()
        broker = KiwoomBroker(config(), transport=transport)
        common = {"as_of": datetime(2026, 9, 29, 10, tzinfo=timezone.utc)}
        for interval in (0, 2, True, 120):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                broker.minute_bars_domestic("005930", interval_minutes=interval, **common)
        for max_pages in (0, 101, True):
            with self.subTest(max_pages=max_pages), self.assertRaises(ValueError):
                broker.minute_bars_domestic("005930", max_pages=max_pages, **common)
        with self.assertRaises(ValueError):
            broker.minute_bars_us("NVDA", exchange="ND", as_of=datetime(2026, 9, 29, 10))
        self.assertEqual(transport.calls, [])

    def test_minute_bar_requires_aware_timestamp(self):
        with self.assertRaises(ValueError):
            MinuteBar(Market.DOMESTIC, "005930", "KRX", datetime(2026, 9, 29, 10),
                      *(Decimal("1") for _ in range(5)), "KRW")

    def test_page_cap_returns_explicitly_truncated_latest_slice(self):
        transport = QueueTransport(
            token_response(),
            FakeResponse({"stk_min_pole_chart_qry": [domestic_row("20260929095900")]},
                         headers={"cont-yn": "Y", "next-key": "next"}),
        )
        rows = KiwoomBroker(config(), transport=transport).minute_bars_domestic(
            "005930", max_pages=1,
            as_of=datetime(2026, 9, 29, 10, 1, tzinfo=ZoneInfo("Asia/Seoul")),
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.page_count, 1)
        self.assertTrue(rows.truncated)
        self.assertEqual(rows.pages, (tuple(rows),))

    def test_missing_continuation_key_rejects_ambiguous_slice(self):
        transport = QueueTransport(
            token_response(), FakeResponse({"stk_min_pole_chart_qry": [domestic_row()]},
                                           headers={"cont-yn": "Y"}),
        )
        with self.assertRaises(BrokerAPIError):
            KiwoomBroker(config(), transport=transport).minute_bars_domestic(
                "005930", max_pages=1,
                as_of=datetime(2026, 9, 29, 10, 1, tzinfo=ZoneInfo("Asia/Seoul")),
            )


if __name__ == "__main__":
    unittest.main()
