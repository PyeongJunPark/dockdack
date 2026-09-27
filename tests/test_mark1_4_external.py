"""Mark1.4 pre-open decision tests; fake model, no broker or orders."""
from datetime import date, timedelta
from decimal import Decimal
from types import SimpleNamespace
import unittest

from dockdack.market_schedule import session_on
from dockdack.models import Market
from dockdack.signals.prototype_external import PrototypeProcessClient, RemoteWorkerError
from dockdack.signals.mark1_4_external import (
    MODEL_ID, Mark14Worker, _expected_dates, _policy_enabled,
)


DAY = date(2026, 9, 28)
SESSION = session_on(Market.DOMESTIC, DAY)
KEY = "domestic:KRX:005930"


class FakePredictor:
    metadata = {"market": "domestic", "bundle_manifest_sha256": "a" * 64}

    def score_many(self, windows, symbols):
        if len(windows) != 100 or symbols[0] != ("005930", "KRX"):
            raise ValueError("fake universe mismatch")
        return [{"symbol": symbol, "exchange": exchange,
                 "score": 0.6 if index == 0 else 0.1,
                 "frozen_numeric_score_threshold": 0.45,
                 "above_frozen_threshold": index == 0,
                 "score_metric": "predicted_net_return_percent", "score_unit": "percent",
                 "in_training_universe": False, "out_of_training_universe": True}
                for index, (symbol, exchange) in enumerate(symbols)]


def candidate(symbol="005930"):
    dates = list(_expected_dates("domestic", DAY))
    return {"watch_id": f"domestic:KRX:{symbol}", "symbol": symbol, "exchange": "KRX",
            "dates": dates, "last_completed_date": dates[-1],
            "bars": [[100., 101., 99., 100., 1000.] for _ in dates]}


def candidates():
    return [candidate(), *(candidate(f"{index:06d}") for index in range(1, 100))]


def worker():
    return Mark14Worker(bundle_root="unused", predictors={"domestic": FakePredictor()})


def preopen_request(**changes):
    base = {"schema_version": 1, "model_id": MODEL_ID, "operation": "prepare_preopen",
            "market": "domestic", "now": (SESSION.opened - timedelta(minutes=9)).isoformat(),
            "session_open": SESSION.opened.isoformat(), "candidates": candidates()}
    base.update(changes)
    return base


def produce_request(now):
    stock = {"watch_id": KEY, "market": "domestic", "symbol": "005930",
             "exchange": "KRX", "currency": "KRW", "status": "ok",
             "price": "100", "quote_fetched_at": now.isoformat(),
             "quote_age_seconds": 0, "quote_stale": False,
             "bars": [{"date": day, "open": "100", "high": "101",
                       "low": "99", "close": "100", "volume": "1000"}
                      for day in _expected_dates("domestic", DAY)]}
    position = {"market": "domestic", "symbol": "005930", "exchange": "KRX",
                "currency": "KRW", "quantity": "0", "sellable_quantity": "0",
                "average_price": None, "fetched_at": now.isoformat()}
    return {"schema_version": 1, "model_id": MODEL_ID, "operation": "produce",
            "chart": {"schema_version": 1, "source": "kiwoom_demo", "trading_mode": "demo",
                      "created_at": now.isoformat(), "export_id": "test-export",
                      "adjusted_prices": True, "stocks": [stock]},
            "positions": {KEY: position}, "now": now.isoformat(),
            "max_krw": "500000", "max_usd": "1000"}


class Mark14WorkerTests(unittest.TestCase):
    def test_preopen_plan_is_frozen_and_open_only(self):
        owner = worker()
        prepared = owner.dispatch(preopen_request())
        self.assertEqual(prepared["state"], "prepared")
        self.assertEqual(prepared["selected_count"], 1)
        self.assertEqual(prepared["scored_count"], 100)
        self.assertTrue(prepared["candidates"][0]["out_of_training_universe"])
        repeated = owner.dispatch(preopen_request(candidates=[]))
        self.assertTrue(repeated["already_frozen"])
        self.assertEqual(repeated["plan_sha256"], prepared["plan_sha256"])
        at_open = SESSION.opened + timedelta(seconds=30)
        result = owner.dispatch(produce_request(at_open))
        signal = result["payload"]["signals"][0]
        self.assertEqual(signal["action"], "buy")
        self.assertEqual(signal["strategy_id"], MODEL_ID)
        checked = owner.dispatch({"schema_version": 1, "model_id": MODEL_ID,
                                  "operation": "validate_frozen", "now": at_open.isoformat(),
                                  "market": "domestic", "watch_id": KEY,
                                  "plan_sha256": prepared["plan_sha256"],
                                  "signal_id": signal["signal_id"], "export_id": signal["export_id"],
                                  "bars": produce_request(at_open)["chart"]["stocks"][0]["bars"]})
        self.assertTrue(checked["selected"])
        late = SESSION.opened + timedelta(minutes=5)
        self.assertEqual(owner.dispatch(produce_request(late))["payload"]["signals"][0]["action"], "hold")

    def test_restart_and_missing_dates_fail_closed(self):
        at_open = SESSION.opened + timedelta(seconds=30)
        self.assertEqual(worker().dispatch(produce_request(at_open))["payload"]["signals"][0]["action"], "hold")
        old = candidates()
        old[0]["dates"][-1] = "2026-01-01"
        with self.assertRaisesRegex(ValueError, "30 completed sessions"):
            worker().dispatch(preopen_request(candidates=old))
        with self.assertRaisesRegex(ValueError, "exactly 100"):
            worker().dispatch(preopen_request(candidates=[candidate()]))

    def test_revised_history_blocks_frozen_buy(self):
        owner = worker()
        owner.dispatch(preopen_request())
        at_open = SESSION.opened + timedelta(seconds=30)
        request = produce_request(at_open)
        request["chart"]["stocks"][0]["bars"][-1]["close"] = "100.5"
        row = owner.dispatch(request)
        self.assertEqual(row["payload"]["signals"][0]["action"], "hold")
        self.assertEqual(row["diagnostics"][0]["reason"], "CHART_CHANGED_SINCE_PREOPEN")

    def test_preopen_time_and_demo_policy_are_required(self):
        with self.assertRaisesRegex(ValueError, "final 10"):
            worker().dispatch(preopen_request(now=(SESSION.opened - timedelta(minutes=11)).isoformat()))
        engine = SimpleNamespace(equity_buy_percent=Decimal("10"), close_liquidator=None)
        self.assertTrue(_policy_enabled(engine))
        engine.equity_buy_percent = Decimal("9")
        self.assertFalse(_policy_enabled(engine))

    def test_remote_preparation_rejection_keeps_healthy_child(self):
        client = PrototypeProcessClient(MODEL_ID, timeout=15)
        try:
            self.assertEqual(client.request("health")["model_id"], MODEL_ID)
            with self.assertRaises(RemoteWorkerError):
                client.request("prepare_preopen", start=False,
                               preserve_remote_error=True, market="domestic",
                               now="invalid", session_open=SESSION.opened.isoformat(),
                               candidates=candidates())
            self.assertTrue(client.is_alive)
            self.assertEqual(client.request("health", start=False)["model_id"], MODEL_ID)
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
