import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from dockdack.exceptions import BrokerAPIError
from dockdack.gui_service import Instrument
from dockdack.http import APIPage
from dockdack.models import Market
from dockdack.universe import RankedStock, top_turnover
from dockdack.watchlist import TriggerRule, WatchItem, WatchStore


class Pages:
    def __init__(self, *pages):
        self.pages, self.calls, self.read = pages, [], 0

    def iter_pages(self, **kwargs):
        self.calls.append(kwargs)
        for body in self.pages:
            self.read += 1
            yield APIPage(body, True, "next", "Y")


class UniverseTests(unittest.TestCase):
    def setUp(self):
        self.classifier = patch("dockdack.universe.common_equities", side_effect=lambda http, market, candidates: frozenset(candidates)).start()
        self.addCleanup(patch.stopall)

    def test_excluded_products_do_not_count_toward_limit(self):
        rows = [dict(stk_cd=s, stex_tp="ND", rank=str(i+1), trde_prica=str(100-i))
                for i, s in enumerate(("QQQ", "AAPL", "MSFT"))]
        self.classifier.side_effect = lambda http, market, candidates: frozenset(k for k in candidates if k[0] != "QQQ")
        http = Pages({"result_list": rows[:2]}, {"result_list": rows[2:]})
        ranks = top_turnover(http, Market.US, 2)
        self.assertEqual([r.symbol for r in ranks], ["AAPL", "MSFT"])
        self.assertEqual(http.read, 2)
        self.assertEqual([r.rank for r in ranks], [1, 2])

    def test_short_filtered_list_fails_without_padding(self):
        self.classifier.return_value = frozenset()
        self.classifier.side_effect = None
        with self.assertRaises(BrokerAPIError):
            top_turnover(Pages({"result_list": [dict(stk_cd="QQQ", stex_tp="ND", rank="1", trde_prica="100")]}), Market.US, 1)

    def test_domestic_100_stops_after_first_page_and_scales_million_krw(self):
        rows = [dict(stk_cd=f"{i:06d}", now_rank=str(i), stk_nm="종목", trde_prica=str(1000-i)) for i in range(1, 101)]
        http = Pages({"trde_prica_upper": rows}, {"bad": "must not read"})
        ranks = top_turnover(http, Market.DOMESTIC)
        self.assertEqual(len(ranks), 100)
        self.assertEqual(http.read, 1)
        self.assertEqual(http.calls[0]["api_id"], "ka10032")
        self.assertEqual(http.calls[0]["body"], {"mrkt_tp": "000", "mang_stk_incls": "0", "stex_tp": "1"})
        self.assertEqual(ranks[0].turnover, Decimal(999_000_000))
        self.assertEqual(ranks[0].currency, "KRW")

    def test_us_paginates_100_all_exchanges_and_scales_thousand_usd(self):
        pages = [{"result_list": [dict(stk_cd=f"S{i}", rank=str(i+1), stex_tp=("ND", "NY", "NA")[i % 3], trde_prica=str(1000-i))
                                 for i in range(start, start+20)]} for start in range(0, 100, 20)]
        http = Pages(*pages)
        ranks = top_turnover(http, Market.US)
        self.assertEqual(len(ranks), 100)
        self.assertEqual(http.read, 5)
        self.assertEqual(ranks[0].turnover, Decimal(1_000_000))
        self.assertEqual(http.calls[0]["body"]["stex_tp"], "0")
        self.assertEqual(http.calls[0]["body"]["stk_tp"], "1")

    def test_missing_rankings_and_malformed_values_fail_instead_of_padding(self):
        for body in ({}, {"result_list": []}, {"result_list": [dict(stk_cd="AAPL", stex_tp="BAD", rank="1", trde_prica="100")]},
                     {"result_list": [dict(stk_cd="AAPL", stex_tp="ND", rank="1", trde_prica="NaN")]}):
            with self.subTest(body=body), self.assertRaises(BrokerAPIError):
                top_turnover(Pages(body), Market.US)

    def test_case_sensitive_share_class_survives_rank_and_watchlist(self):
        from dockdack.cli import identify_symbol
        from dockdack import KiwoomBroker
        from test_kiwoom import config, token_response, QueueTransport, FakeResponse
        http = Pages({"result_list": [dict(stk_cd="BRKb", stex_tp="NY", rank="1", trde_prica="100")]})
        stock = top_turnover(http, Market.US, 1)[0]
        self.assertEqual(stock.symbol, "BRKb")
        self.assertEqual(identify_symbol(stock.symbol), (Market.US, "BRKb"))
        self.assertEqual(identify_symbol("aapl"), (Market.US, "AAPL"))
        item = WatchItem(Instrument(Market.US, stock.symbol, stock.exchange))
        self.assertEqual(item.id, "us:NY:BRKb")
        transport = QueueTransport(token_response(), FakeResponse({"cur_prc": "100", "stk_cd": "BRKb"}))
        broker = KiwoomBroker(config(), transport=transport)
        self.assertEqual(broker.quote_us(stock.symbol, exchange="NY").symbol, "BRKb")
        self.assertEqual(transport.calls[-1]["json"]["stk_cd"], "BRKb")
        self.assertEqual(broker.build_order(market="us", symbol="BRKb", side="buy", quantity=1,
                                          exchange="NY", order_type="limit", price=Decimal(100)).symbol, "BRKb")

    def test_duplicate_symbols_do_not_count_twice_and_turnover_sort_is_numeric(self):
        first = dict(stk_cd="AAPL", stex_tp="ND", rank="1", trde_prica="99")
        second = dict(stk_cd="MSFT", stex_tp="ND", rank="2", trde_prica="100")
        ranks = top_turnover(Pages({"result_list": [first, first]}, {"result_list": [second]}), Market.US, 2)
        self.assertEqual([r.symbol for r in ranks], ["MSFT", "AAPL"])

    def test_bulk_add_preserves_manual_stocks_existing_n_and_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            store = WatchStore(Path(directory) / "db.sqlite3")
            item = WatchItem(Instrument(Market.DOMESTIC, "005930", "KRX"), "삼성", 60)
            store.save_item(item)
            store.save_item(WatchItem(Instrument(Market.US, "AAPL", "ND")))
            rule = TriggerRule.create(item, "price_ge", "buy", 1, Decimal(1000), Decimal(100))
            store.add_rule(rule)
            ranks = [RankedStock(Market.DOMESTIC, f"{i:06d}", "KRX", "순위 종목", i, Decimal(1000-i), "KRW") for i in range(1, 101)]
            ranks[0] = RankedStock(Market.DOMESTIC, "005930", "KRX", "삼성전자", 1, Decimal(1000), "KRW")
            store.add_ranked(ranks)
            self.assertEqual(len(store.items()), 101)
            self.assertEqual(store.items()[0].days, 60)
            self.assertEqual(store.rules(), (rule,))
            self.assertEqual(len(store.rankings()), 100)
            store.add_ranked(ranks)
            self.assertEqual(len(store.items()), 101)


if __name__ == "__main__":
    unittest.main()
