"""External JSON -> mock BUY fill -> independent holding SELL, entirely offline."""

import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from dockdack.autotrade import AutoTrader
from dockdack.models import AccountSnapshot, Market, OrderExecution, OrderResult, OrderSide, TradingMode
from dockdack.signal_bridge import ExternalPolicy, SignalFileReader, atomic_json, export_charts
from dockdack.watchlist import WatchItem, WatchStore
from test_autotrade import FakeTradingService, NOW, position


class RoundtripService(FakeTradingService):
    """Records Python objects only; never constructs a broker/HTTP client."""

    def __init__(self):
        super().__init__()
        self.cash = Decimal("40000")
        self.evaluation = Decimal("60000")
        self.available = Decimal("40000")

    def safety_account(self, instrument):
        return AccountSnapshot(instrument.market, instrument.currency,
                               tuple(p for p in self.positions if p.market is instrument.market),
                               cash=self.cash, total_evaluation=self.evaluation,
                               available_to_order=self.available)

    def submit(self, request):
        self.submitted.append(request)
        return OrderResult(True, TradingMode.DEMO, request, f"{len(self.submitted):07d}",
                           "offline fake acceptance, not a broker order")


class ExternalRoundtripV00Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.open_market = None
        sessions = patch("dockdack.autotrade.regular_session",
                         side_effect=lambda market, now: market is self.open_market)
        sessions.start()
        self.addCleanup(sessions.stop)
        network = patch("requests.sessions.Session.request",
                        side_effect=AssertionError("network is forbidden in offline roundtrip tests"))
        network.start()
        self.addCleanup(network.stop)

    def build(self, market, *, directory="flow", cap=None, available=Decimal("40000")):
        folder = self.root / directory / market.value
        folder.mkdir(parents=True)
        store = WatchStore(folder / "offline.sqlite3")
        service = RoundtripService()
        service.available = available
        item = WatchItem(service.resolve("005930" if market is Market.DOMESTIC else "AAPL"),
                         "삼성전자" if market is Market.DOMESTIC else "애플")
        store.save_item(item)
        engine = AutoTrader(service, store, clock=lambda: NOW)
        engine.enable_holdings_exits = True
        engine.equity_buy_percent = Decimal("10")
        engine.isolated_symbol_errors = True
        self.open_market = market
        engine.snapshot(item)
        chart = export_charts(store, folder / "charts.json", now=NOW)
        market_cap = cap if cap is not None else engine.holding_caps[market]
        policy = ExternalPolicy("offline-external-model", 999_999_999,
                                market_cap if market is Market.DOMESTIC else Decimal(0),
                                market_cap if market is Market.US else Decimal(0))
        signal_path = folder / "signals.json"
        atomic_json(signal_path, {
            "schema_version": 1, "source_id": policy.source_id,
            "signals": [{"signal_id": "buy-with-bracket", "export_id": chart["export_id"],
                         "market": market.value, "symbol": item.instrument.symbol,
                         "exchange": item.instrument.exchange, "action": "buy", "quantity": 1,
                         "max_notional": str(market_cap), "take_profit_price": "110",
                         "stop_loss_price": "90", "generated_at": NOW.isoformat(),
                         "expires_at": (NOW + timedelta(minutes=2)).isoformat()}],
        })
        reader = SignalFileReader(store, signal_path, policy, clock=lambda: NOW)
        engine.external_only = True
        engine.configure_external_sources([(policy, reader)])
        return service, store, item, engine, reader

    @staticmethod
    def held(item, quantity):
        return replace(position(quantity, quantity), market=item.instrument.market,
                       symbol=item.instrument.symbol, exchange=item.instrument.exchange,
                       currency=item.instrument.currency, name=item.name,
                       evaluation_amount=Decimal(quantity) * Decimal("100"))

    def test_both_markets_json_buy_full_fill_then_unwatched_holding_target_sell(self):
        for market in (Market.DOMESTIC, Market.US):
            with self.subTest(market=market):
                service, store, item, engine, reader = self.build(market)
                engine.enable_orders("DEMO_AUTOTRADE")
                engine.poll()
                self.assertEqual(reader.received_counts["queued"], 1)
                self.assertEqual(len(service.submitted), 1)
                buy = service.submitted[0]
                # (40,000 cash + 60,000 held evaluation) * 10% / 100 = 100 shares.
                self.assertEqual((buy.side, buy.quantity, buy.price), (OrderSide.BUY, 100, Decimal("100")))
                self.assertLessEqual(buy.quantity * buy.price, engine.holding_caps[market])
                self.assertEqual(store.attempts()[0]["status"], "accepted")
                targets = WatchStore(store.path).exit_targets(item.id)
                self.assertEqual((targets["take_profit_price"], targets["stop_loss_price"], targets["source"]),
                                 (Decimal("110"), Decimal("90"), "offline-external-model"))

                engine.poll()  # An unchanged JSON file or accepted-unfilled order cannot repeat BUY.
                self.assertEqual(len(service.submitted), 1)
                self.assertEqual(len(store.attempts()), 1)

                service.positions = (self.held(item, buy.quantity),)
                service.fills = (OrderExecution("1", item.instrument.symbol, "매수", "체결",
                                               Decimal(buy.quantity), Decimal(buy.quantity), Decimal(0),
                                               buy.price, Decimal("100"), "100000"),)
                engine.poll()
                self.assertEqual(store.attempts()[0]["status"], "filled")
                buy_history = store.order_history()[0]
                self.assertEqual((buy_history["filled_quantity"], buy_history["fill_price"]), ("100", "100"))
                self.assertEqual(len(service.submitted), 1)

                store.remove_item(item.id)
                self.assertEqual(store.items(), ())
                # Reopen the ledger/engine to prove the bracket is durable, not an in-memory signal.
                engine = AutoTrader(service, WatchStore(store.path), clock=lambda: NOW)
                engine.enable_holdings_exits = True
                engine.equity_buy_percent = Decimal("10")
                engine.isolated_symbol_errors = True
                engine.enable_orders("DEMO_AUTOTRADE")
                history_calls = service.history_calls
                service.prices = [Decimal("109")]
                engine.poll()
                self.assertEqual(len(service.submitted), 1)  # Below saved 110, despite +1% fallback.
                service.prices = [Decimal("110")]
                engine.poll()
                self.assertEqual(len(service.submitted), 2)
                sell = service.submitted[1]
                expected_quantity = min(buy.quantity, int(engine.holding_caps[market] / Decimal("110")))
                self.assertEqual((sell.side, sell.quantity, sell.price),
                                 (OrderSide.SELL, expected_quantity, Decimal("110")))
                self.assertLessEqual(sell.quantity * sell.price, engine.holding_caps[market])
                self.assertEqual(service.history_calls, history_calls)  # Holdings are quote-only.
                self.assertEqual(engine.store.items(), ())
                self.assertEqual([a["status"] for a in engine.store.attempts()], ["filled", "accepted"])
                engine.poll()
                engine.poll()
                self.assertEqual(len(service.submitted), 2)  # Pending SELL is never duplicated.
                self.assertEqual(len(engine.store.attempts()), 2)

    def test_off_and_closed_market_send_neither_external_buy_nor_holding_sell(self):
        for market in (Market.DOMESTIC, Market.US):
            for closed in (False, True):
                with self.subTest(market=market, closed=closed):
                    service, store, item, engine, reader = self.build(market, directory=f"closed-{closed}")
                    service.positions = (self.held(item, 1),)
                    service.prices = [Decimal("110")]
                    if closed:
                        engine.enable_orders("DEMO_AUTOTRADE")
                        self.open_market = None
                    engine.poll()
                    engine.poll()
                    self.assertEqual(reader.received_counts["queued"], 1)
                    self.assertEqual(service.submitted, [])
                    self.assertEqual(store.attempts(), ())

    def test_percentage_buy_respects_signal_cap_and_available_cash_headroom(self):
        for cap, available, expected in ((Decimal("550"), Decimal("40000"), 5),
                                         (Decimal("10000"), Decimal("500"), 4)):
            with self.subTest(cap=cap, available=available):
                service, store, item, engine, reader = self.build(
                    Market.US, directory=f"cap-{cap}", cap=cap, available=available)
                engine.enable_orders("DEMO_AUTOTRADE")
                engine.poll()
                self.assertEqual(len(service.submitted), 1)
                order = service.submitted[0]
                self.assertEqual(order.quantity, expected)
                self.assertLessEqual(order.quantity * order.price, cap)
                self.assertLessEqual(order.quantity * order.price * Decimal("1.01"), available)
                self.assertEqual(store.attempts()[0]["status"], "accepted")


if __name__ == "__main__":
    unittest.main()
