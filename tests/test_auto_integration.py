"""Full auto-trader -> service -> broker -> fake HTTP contract checks; no network."""

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from dockdack import KiwoomBroker, Market
from dockdack.autotrade import AutoTrader
from dockdack.gui_service import TradingService
from dockdack.watchlist import TriggerRule, WatchItem, WatchStore
from dockdack.signal_bridge import ExternalPolicy, export_charts, ingest_signals
from test_history import row
from test_kiwoom import FakeResponse, QueueTransport, config, token_response


class AutoIntegrationTests(unittest.TestCase):
    def setUp(self):
        # Classification contracts have their own source/negative-control tests;
        # these fixtures exercise the existing quote/account/order HTTP contract.
        classifier = patch("dockdack.equity_policy.common_equities",
                           side_effect=lambda http, market, candidates: frozenset(candidates))
        classifier.start()
        self.addCleanup(classifier.stop)

    def test_domestic_external_market_order_contract_and_pre_http_journal(self):
        with tempfile.TemporaryDirectory() as folder:
            class AuditedTransport(QueueTransport):
                def request(self,method,url,**kwargs):
                    if url.endswith("/ordr"):
                        assert store.attempts()[0]["status"]=="submitting"
                        assert store.attempts()[0]["price"]=="100"
                    return super().request(method,url,**kwargs)
            transport=AuditedTransport(token_response(),
                FakeResponse({"stk_dt_pole_chart_qry":[row("20260911")]}),FakeResponse({"cur_prc":"100"}),
                FakeResponse({"cur_prc":"100"}),FakeResponse({"oso":[]}),
                FakeResponse({"acnt_evlt_remn_indv_tot":[]}),FakeResponse({"ord_alow_amt":"10000"}),
                FakeResponse({"cur_prc":"100"}),FakeResponse({"return_code":0,"ord_no":"123","return_msg":"접수"}))
            service=TradingService(lambda _:KiwoomBroker(config(),transport=transport))
            store=WatchStore(Path(folder)/"watch.sqlite3", storage_scope=service.storage_scope)
            item=WatchItem(service.resolve("005930","KRX"),days=1)
            store.save_item(item)
            now=datetime(2026,9,14,1,tzinfo=timezone.utc)
            engine=AutoTrader(service,store,clock=lambda:now)
            engine.snapshot(item)
            exported=export_charts(store,Path(folder)/"charts.json",now=now)
            policy=ExternalPolicy("test",1,Decimal(1000),Decimal(0),allow_market=True)
            ingest_signals(store,{"schema_version":1,"source_id":"test","signals":[{
                "signal_id":"market-1","export_id":exported["export_id"],"market":"domestic","symbol":"005930","exchange":"KRX",
                "action":"buy","quantity":1,"max_notional":"1000","order_type":"market",
                "generated_at":now.isoformat(),"expires_at":(now+timedelta(minutes=2)).isoformat()}]},policy,now=now)
            engine.external_only,engine.external_policy=True,policy
            engine.enable_orders("DEMO_AUTOTRADE")
            engine.poll()
            self.assertEqual(store.rules()[0].status,"accepted",store.events())
            call=next(c for c in transport.calls if c["url"].endswith("/ordr"))
            self.assertEqual(call["json"]["trde_tp"],"3")
            self.assertEqual(call["json"]["ord_uv"],"")

    def test_external_signals_reach_both_broker_order_contracts_once(self):
        for market in Market:
            with self.subTest(market=market), tempfile.TemporaryDirectory() as folder:
                domestic = market is Market.DOMESTIC
                symbol, exchange = ("005930", "KRX") if domestic else ("BRKb", "NY")
                class AuditedTransport(QueueTransport):
                    def request(self, method, url, **kwargs):
                        if url.endswith("/ordr"):
                            assert store.attempts()[0]["status"] == "submitting"
                        return super().request(method, url, **kwargs)
                transport = AuditedTransport(
                    token_response(),
                    FakeResponse({"stk_dt_pole_chart_qry" if domestic else "result_list": [row("20260911", us=not domestic)]}),
                    FakeResponse({"cur_prc": "100", "stk_cd": symbol}),
                    FakeResponse({"cur_prc": "100", "stk_cd": symbol}),
                    FakeResponse({"oso" if domestic else "result_list": []}),
                    FakeResponse({"acnt_evlt_remn_indv_tot" if domestic else "result_list": []}),
                    FakeResponse({"ord_alow_amt": "10000"} if domestic else {"result_list": [{"crnc_code": "USD", "fc_ord_alowa": "10000"}]}),
                    FakeResponse({"cur_prc": "100", "stk_cd": symbol}),
                    FakeResponse({"return_code": 0, "ord_no": "0000200", "return_msg": "접수"}),
                )
                service = TradingService(lambda _: KiwoomBroker(config(), transport=transport))
                store = WatchStore(Path(folder) / "watch.sqlite3", storage_scope=service.storage_scope)
                item = WatchItem(service.resolve(symbol, exchange), days=1)
                store.save_item(item)
                now = datetime(2026, 9, 14, 1 if domestic else 15, tzinfo=timezone.utc)
                engine = AutoTrader(service, store, clock=lambda: now)
                engine.snapshot(item)
                exported = export_charts(store, Path(folder) / "charts.json", now=now)
                policy = ExternalPolicy("model", 1, Decimal(200), Decimal(200))
                payload = {"schema_version": 1, "source_id": "model", "signals": [{
                    "signal_id": "model-run-1", "export_id": exported["export_id"], "market": market.value,
                    "symbol": symbol, "exchange": exchange, "action": "buy", "quantity": 1, "max_notional": "200",
                    "generated_at": now.isoformat(), "expires_at": (now+timedelta(minutes=2)).isoformat()}]}
                ingest_signals(store, payload, policy, now=now)
                engine.external_only, engine.external_policy = True, policy
                engine.enable_orders("DEMO_AUTOTRADE")
                engine.poll()
                self.assertEqual(store.rules()[0].status, "accepted", store.events())
                calls = [call for call in transport.calls if call["url"].endswith("/ordr")]
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["json"]["stk_cd"], symbol)
                self.assertEqual(calls[0]["json"]["ord_uv"], "100")
                self.assertEqual(calls[0]["headers"]["api-id"], "kt10000" if domestic else "ust20000")
                self.assertEqual(ingest_signals(store, payload, policy, now=now)["duplicates"], 1)

    def test_both_markets_use_real_broker_code_and_journal_before_fake_http_order(self):
        for market in Market:
            with self.subTest(market=market), tempfile.TemporaryDirectory() as folder:
                domestic = market is Market.DOMESTIC
                symbol, exchange = ("005930", "KRX") if domestic else ("AAPL", "ND")

                class AuditedTransport(QueueTransport):
                    def request(self, method, url, **kwargs):
                        if url.endswith("/ordr"):
                            attempts = store.attempts()
                            if len(attempts) != 1 or attempts[0]["status"] != "submitting":
                                raise AssertionError("Submission intent must be durable before HTTP")
                        return super().request(method, url, **kwargs)

                transport = AuditedTransport(
                    token_response(),
                    FakeResponse({"stk_dt_pole_chart_qry" if domestic else "result_list": [row("20260911", us=not domestic)]}),
                    FakeResponse({"cur_prc": "100"}),
                    FakeResponse({"oso" if domestic else "result_list": []}),
                    FakeResponse({"acnt_evlt_remn_indv_tot" if domestic else "result_list": []}),
                    FakeResponse({"ord_alow_amt": "10000"} if domestic else {"result_list": [{"crnc_code": "USD", "fc_ord_alowa": "10000"}]}),
                    FakeResponse({"cur_prc": "100"}),
                    FakeResponse({"return_code": 0, "ord_no": "0000200", "return_msg": "접수"}),
                )
                broker = KiwoomBroker(config(), transport=transport)
                service = TradingService(lambda _: broker)
                store = WatchStore(Path(folder) / "watch.sqlite3", storage_scope=service.storage_scope)
                item = WatchItem(service.resolve(symbol, exchange))
                store.save_item(item)
                store.add_rule(TriggerRule.create(item, "price_ge", "buy", 1, Decimal(200), Decimal(95)))
                now = datetime(2026, 9, 14, 1 if domestic else 15, tzinfo=timezone.utc)
                engine = AutoTrader(service, store, clock=lambda: now)
                engine.enable_orders("DEMO_AUTOTRADE")
                results = engine.poll()
                self.assertFalse(isinstance(results[item.id], Exception))
                self.assertEqual(store.rules()[0].status, "accepted", store.events())
                order_calls = [call for call in transport.calls if call["url"].endswith("/ordr")]
                self.assertEqual(len(order_calls), 1)
                self.assertEqual(order_calls[0]["json"]["ord_uv"], "100")
                self.assertEqual(order_calls[0]["json"]["trde_tp"], "0" if domestic else "00")
                self.assertEqual(order_calls[0]["headers"]["api-id"], "kt10000" if domestic else "ust20000")
                self.assertTrue(all(call["url"].startswith("https://mockapi.kiwoom.com/") for call in transport.calls))


if __name__ == "__main__":
    unittest.main()
