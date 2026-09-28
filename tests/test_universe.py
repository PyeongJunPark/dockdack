import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from dockdack.exceptions import BrokerAPIError
from dockdack.gui_service import Instrument
from dockdack.http import APIPage
from dockdack.models import Market
from dockdack.universe import RankedStock, top_change, top_market_cap, top_turnover, top_watchlist
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

    def test_us_np_listing_before_100_is_excluded_without_remapping_or_stopping(self):
        classified = set()
        def classify(http, market, candidates):
            candidates = frozenset(candidates)
            classified.update(candidates)
            return candidates
        self.classifier.side_effect = classify
        rows = [dict(stk_cd=f"S{i}", rank=str(i + 1), stex_tp="ND", trde_prica=str(1000 - i))
                for i in range(99)]
        unsupported = dict(stk_cd="TCEHY", rank="100", stex_tp="NP", trde_prica="901")
        final = dict(stk_cd="LAST", rank="101", stex_tp="NY", trde_prica="900")
        http = Pages({"result_list": rows + [unsupported]}, {"result_list": [final]})
        ranks = top_turnover(http, Market.US, 100)
        self.assertEqual(len(ranks), 100)
        self.assertEqual(http.read, 2)
        self.assertEqual(ranks[-1].symbol, "LAST")
        self.assertNotIn("TCEHY", {r.symbol for r in ranks})
        self.assertTrue(all(r.exchange in {"ND", "NY", "NA"} for r in ranks))
        self.assertEqual(len(classified), 100)
        self.assertNotIn(("TCEHY", "NP"), classified)

    def test_us_np_does_not_pad_short_supported_common_share_list(self):
        rows = [dict(stk_cd="AAPL", stex_tp="ND", rank="1", trde_prica="100"),
                dict(stk_cd="TCEHY", stex_tp="NP", rank="2", trde_prica="99")]
        with self.assertRaisesRegex(BrokerAPIError, "1개뿐"):
            top_turnover(Pages({"result_list": rows}), Market.US, 2)

    def test_us_missing_blank_and_unknown_exchange_still_fail_closed(self):
        for exchange in (None, "", "  ", "BAD", "ZZ"):
            row = dict(stk_cd="AAPL", rank="1", trde_prica="100")
            if exchange is not None:
                row["stex_tp"] = exchange
            with self.subTest(exchange=exchange), self.assertRaisesRegex(BrokerAPIError, "거래소"):
                top_turnover(Pages({"result_list": [row]}), Market.US, 1)
        with self.assertRaisesRegex(BrokerAPIError, "종목"):
            top_turnover(Pages({"result_list": [dict(stk_cd="", stex_tp="NP", rank="1", trde_prica="100")]}), Market.US, 1)

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

    def test_change_and_us_market_cap_use_official_rank_contracts(self):
        domestic = Pages({"pred_pre_flu_rt_upper": [
            {"stk_cd": "005930", "stk_nm": "삼성전자", "flu_rt": "+7.5"}]})
        self.assertEqual(top_change(domestic, Market.DOMESTIC, gainers=True, limit=1)[0].symbol, "005930")
        self.assertEqual(domestic.calls[0]["api_id"], "ka10027")
        self.assertEqual(domestic.calls[0]["body"]["sort_tp"], "1")
        domestic_down = Pages({"pred_pre_flu_rt_upper": [
            {"stk_cd": "000660", "flu_rt": "-4.1"}]})
        self.assertEqual(top_change(domestic_down, Market.DOMESTIC, gainers=False, limit=1)[0].symbol, "000660")
        self.assertEqual(domestic_down.calls[0]["body"]["sort_tp"], "3")
        us_down = Pages({"result_list": [
            {"stk_cd": "AAPL", "stex_tp": "ND", "rank": "1", "flu_rt": "-2.5"}]})
        self.assertEqual(top_change(us_down, Market.US, gainers=False, limit=1)[0].symbol, "AAPL")
        self.assertEqual(us_down.calls[0]["api_id"], "usa20910")
        self.assertEqual(us_down.calls[0]["body"]["sort_tp"], "4")
        us_cap = Pages({"result_list": [
            {"stk_cd": "MSFT", "stex_tp": "ND", "rank": "1", "mac": "1000"}]})
        self.assertEqual(top_market_cap(us_cap, Market.US, 1)[0].symbol, "MSFT")
        self.assertEqual(us_cap.calls[0]["api_id"], "usa20550")
        self.assertEqual(top_market_cap(Pages(), Market.DOMESTIC), ())

    def test_incomplete_category_fails_closed(self):
        with self.assertRaises(BrokerAPIError):
            top_change(Pages({"pred_pre_flu_rt_upper": [
                {"stk_cd": "005930", "flu_rt": "1"}]}), Market.DOMESTIC,
                gainers=True, limit=2)
        with self.assertRaises(BrokerAPIError):
            top_market_cap(Pages({"result_list": [
                {"stk_cd": "MSFT", "stex_tp": "ND", "rank": "1", "mac": "NaN"}]}), Market.US, 1)

    def test_duplicate_heavy_categories_fill_to_100_from_traded_value(self):
        def stock(symbol, basis, ordinal):
            return RankedStock(Market.US, symbol, "ND", symbol, ordinal,
                               Decimal(1000 - ordinal), "USD", ranking_basis=basis)
        turnover = tuple(stock(f"S{i}", "turnover", i + 1) for i in range(100))
        volume = tuple(stock(f"S{i}", "volume", i + 1) for i in range(20))
        gainers = tuple(stock(f"S{i}", "gainers", i + 1) for i in range(20))
        decliners = tuple(stock(f"D{i}", "decliners", i + 1) for i in range(20))
        cap = tuple(stock(f"C{i}", "market_cap", i + 1) for i in range(20))
        with (patch("dockdack.universe.top_turnover", return_value=turnover),
              patch("dockdack.universe.top_volume", return_value=volume),
              patch("dockdack.universe.top_change", side_effect=(gainers, decliners)),
              patch("dockdack.universe.top_market_cap", return_value=cap)):
            selected = top_watchlist(object(), Market.US)
        self.assertEqual(len(selected), 100)
        self.assertEqual(len({(row.symbol, row.exchange) for row in selected}), 100)
        self.assertEqual([row.rank for row in selected], list(range(1, 101)))
        self.assertEqual([row.symbol for row in selected[:20]], [f"S{i}" for i in range(20)])
        self.assertEqual([row.symbol for row in selected[20:40]], [f"D{i}" for i in range(20)])
        self.assertEqual([row.symbol for row in selected[40:60]], [f"C{i}" for i in range(20)])
        self.assertEqual([row.symbol for row in selected[60:]], [f"S{i}" for i in range(20, 60)])
        self.assertTrue(all(row.ranking_basis == "turnover" for row in selected[60:]))

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
