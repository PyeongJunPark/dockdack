"""Mark1.3 sealed, data-only pre-open model; never reaches a broker."""
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from dockdack.mark1_3_preopen import Mark13Predictor, NUMERIC_GUARD_PERCENT
from dockdack.market_schedule import session_on
from dockdack.models import Market
from dockdack.prototype_external import PrototypeProcessClient, PrototypeWorker
from dockdack.signals.mark1_4_external import Mark14Worker, _expected_dates
from dockdack.signals.preopen_series import PREOPEN_MODELS
from dockdack.signals.preopen_series import PreopenExperimentalFeed
from dockdack.signal_bridge import ExternalPolicy
from dockdack.trading.model_exit_schedule import timed_exit_due
import test_mark1_trigger as bridge_fixtures


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "models" / "mark1_3"
BAR = [100, 101, 99, 100.5, 100_000]


class Mark13PreopenTests(unittest.TestCase):
    def test_sealed_models_score_completed_bars_as_net_return_not_probability(self):
        for market, symbol, exchange in (("domestic", "005930", "KRX"),
                                         ("us", "AAPL", "NASDAQ")):
            with self.subTest(market=market):
                predictor = Mark13Predictor(BUNDLE, market)
                result = predictor.score_one(np.asarray([BAR] * 30), symbol, exchange)
                self.assertTrue(np.isfinite(result["score"]))
                self.assertEqual(result["score_unit"], "percent")
                self.assertFalse(result["score_is_calibrated_probability"])
                self.assertEqual(result["above_frozen_threshold"], result["score"] > NUMERIC_GUARD_PERCENT)
                self.assertTrue(result["research_only"])
                self.assertFalse(result["deployment_allowed"])
                self.assertTrue(predictor.metadata["bundle_manifest_sha256"])
                with self.assertRaises(ValueError):
                    predictor.score_many([[BAR] * 29], [(symbol, exchange)])
                with self.assertRaises(ValueError):
                    predictor.score_many([[BAR] * 30, [BAR] * 30],
                                         [(symbol, exchange), (symbol, exchange)])

    def test_modified_artifact_is_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle = Path(folder) / "tampered"
            shutil.copytree(BUNDLE, bundle)
            path = bundle / "domestic.json"
            path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                Mark13Predictor(bundle, "domestic")

    def test_worker_and_family_identity_are_demo_preopen(self):
        spec = PREOPEN_MODELS["mark1-3-prototype"]
        self.assertEqual(spec.bundle_directory, "models/mark1_3")
        worker = PrototypeWorker("mark1-3-prototype", bundle_root=BUNDLE)
        self.assertIsInstance(worker.special, Mark14Worker)
        result = worker.dispatch({"schema_version": 1, "model_id": "mark1-3-prototype",
                                  "operation": "metadata", "market": "domestic"})
        self.assertEqual(result["score_unit"], "percent")
        self.assertEqual(result["training_device"], "cuda")

    def test_real_isolated_child_loads_sealed_bundle_without_broker(self):
        client = PrototypeProcessClient("mark1-3-prototype", bundle_root=BUNDLE, timeout=30)
        try:
            health = client.request("health")
            self.assertEqual(health["source_id"], "mark1-3-prototype-demo-trigger")
            self.assertEqual(health["trading_mode"], "demo")
            metadata = client.request("metadata", market="us")
            self.assertEqual(metadata["bundle_manifest_sha256"],
                             Mark13Predictor(BUNDLE, "us").metadata["bundle_manifest_sha256"])
            self.assertFalse(metadata["deployment_allowed"])
        finally:
            client.close()

    def test_frozen_preopen_selects_no_more_than_ten_and_exit_is_model_owned(self):
        market = Market.DOMESTIC
        session = session_on(market, date(2026, 9, 28))
        self.assertIsNotNone(session)
        expected = _expected_dates(market.value, date(2026, 9, 28))
        candidates = [{"market": market.value, "currency": "KRW", "symbol": f"{index:06d}",
                       "exchange": "KRX", "watch_id": f"domestic:KRX:{index:06d}", "dates": list(expected),
                       "last_completed_date": expected[-1], "bars": [BAR] * 30}
                      for index in range(100)]
        class FakePredictor:
            metadata = {"market": "domestic", "bundle_manifest_sha256": "a" * 64}

            def score_many(self, windows, symbols):
                return [{"symbol": symbol, "exchange": exchange,
                         "score": float(50 - index), "above_frozen_threshold": index < 30,
                         "frozen_numeric_score_threshold": 0.0,
                         "score_metric": "predicted_t_plus_1_net_open_to_close_return_percent",
                         "score_unit": "percent"}
                        for index, (symbol, exchange) in enumerate(symbols)]
        worker = Mark14Worker(bundle_root=BUNDLE, model_id="mark1-3-prototype",
                              predictors={market.value: FakePredictor()})
        result = worker.dispatch({"schema_version": 1, "model_id": "mark1-3-prototype",
                                  "operation": "prepare_preopen", "market": market.value,
                                  "session_open": session.opened.isoformat(),
                                  "now": (session.opened - timedelta(minutes=6)).isoformat(),
                                  "candidates": candidates})
        self.assertEqual(result["scored_count"], 100)
        self.assertEqual(result["selected_count"], 10)
        self.assertEqual(sum(bool(row["selected"]) for row in result["candidates"]), 10)
        now = session.opened + timedelta(seconds=30)
        bars = [{"date": day, "open": "100", "high": "101", "low": "99",
                 "close": "100.5", "volume": "100000"} for day in expected]
        stock = {"watch_id": "domestic:KRX:000000", "market": "domestic",
                 "symbol": "000000", "exchange": "KRX", "currency": "KRW", "status": "ok",
                 "price": "100", "quote_fetched_at": now.isoformat(), "quote_age_seconds": 0,
                 "quote_stale": False, "bars": bars}
        position = {"market": "domestic", "symbol": "000000", "exchange": "KRX",
                    "currency": "KRW", "quantity": "0", "sellable_quantity": "0",
                    "average_price": None, "fetched_at": now.isoformat()}
        produced = worker.dispatch({
            "schema_version": 1, "model_id": "mark1-3-prototype", "operation": "produce",
            "now": now.isoformat(), "max_krw": "500000", "max_usd": "1000",
            "chart": {"schema_version": 1, "source": "kiwoom_demo", "trading_mode": "demo",
                      "created_at": now.isoformat(), "export_id": "offline-1",
                      "adjusted_prices": True, "stocks": [stock]},
            "positions": {stock["watch_id"]: position},
        })
        signal = produced["payload"]["signals"][0]
        self.assertEqual(produced["payload"]["source_id"], "mark1-3-prototype-demo-trigger")
        self.assertEqual(signal["action"], "buy")
        self.assertEqual(signal["strategy_id"], "mark1-3-prototype")
        self.assertEqual(signal["model_manifest_sha256"], "a" * 64)
        self.assertTrue(worker.dispatch({
            "schema_version": 1, "model_id": "mark1-3-prototype",
            "operation": "validate_frozen", "market": "domestic",
            "watch_id": stock["watch_id"], "now": now.isoformat(),
            "plan_sha256": result["plan_sha256"], "signal_id": signal["signal_id"],
            "export_id": signal["export_id"], "bars": bars,
        })["selected"])
        lot = {"strategy_id": "mark1-3-prototype",
               "buy_fill_observed_at": (session.opened + timedelta(minutes=1)).isoformat()}
        self.assertFalse(timed_exit_due(lot, market, session.opened + timedelta(minutes=2)))
        self.assertTrue(timed_exit_due(lot, market, session.closed - timedelta(minutes=4)))
        self.assertFalse(timed_exit_due({**lot, "strategy_id": "manual"}, market,
                                        session.closed - timedelta(minutes=4)))


class Mark13FeedTests(unittest.TestCase):
    """The normal external feed publishes only data; no broker submit or arm."""

    setUp = bridge_fixtures.BridgeTests.setUp

    def test_gui_feed_prepares_and_publishes_demo_only_frozen_buy(self):
        session = session_on(Market.DOMESTIC, date(2026, 9, 28))
        expected = _expected_dates("domestic", date(2026, 9, 28))
        preopen = session.opened - timedelta(minutes=6)
        self.engine.clock = lambda: preopen
        self.engine.equity_buy_percent = Decimal("10")

        class FakePredictor:
            metadata = {"market": "domestic", "bundle_manifest_sha256": "a" * 64}

            def score_many(self, windows, symbols):
                return [{"symbol": symbol, "exchange": exchange,
                         "score": float(1 - index), "above_frozen_threshold": index == 0,
                         "frozen_numeric_score_threshold": NUMERIC_GUARD_PERCENT,
                         "score_metric": "predicted_t_plus_1_net_open_to_close_return_percent",
                         "score_unit": "percent"}
                        for index, (symbol, exchange) in enumerate(symbols)]

        class LocalClient:
            def __init__(self, model_id, *, bundle_root=None, state_path=None):
                self.worker = PrototypeWorker(model_id, bundle_root=BUNDLE,
                                              predictors={"domestic": FakePredictor()})
                self.is_alive = False

            def request(self, operation, *, start=True, **values):
                self.is_alive = True
                return self.worker.dispatch({"schema_version": 1, "model_id": self.worker.model_id,
                                             "operation": operation, **values})

            def close(self):
                self.is_alive = False

        policy = ExternalPolicy("mark1-3-prototype-demo-trigger", 3,
                                Decimal("500000"), Decimal("1000"))
        feed = PreopenExperimentalFeed(self.window, "mark1-3-prototype", policy,
                                       self.root / "mark13-signals.json",
                                       client_factory=LocalClient)
        self.addCleanup(feed.close)
        candidates = [{"watch_id": f"domestic:KRX:{index:06d}",
                       "symbol": f"{index:06d}", "exchange": "KRX", "dates": list(expected),
                       "last_completed_date": expected[-1], "bars": [BAR] * 30}
                      for index in range(100)]
        prepared = feed.prepare_preopen("domestic", candidates, now=preopen, session=session)
        self.assertEqual(prepared["selected_count"], 1)
        now = session.opened + timedelta(seconds=30)
        self.engine.clock = lambda: now
        stock = {"watch_id": "domestic:KRX:000000", "market": "domestic",
                 "symbol": "000000", "exchange": "KRX", "currency": "KRW", "status": "ok",
                 "price": "100", "quote_fetched_at": now.isoformat(),
                 "quote_age_seconds": 0, "quote_stale": False,
                 "bars": [{"date": day, "open": "100", "high": "101", "low": "99",
                           "close": "100.5", "volume": "100000"} for day in expected]}
        feed.publish({"schema_version": 1, "source": "kiwoom_demo", "trading_mode": "demo",
                      "created_at": now.isoformat(), "export_id": "feed-offline-1",
                      "adjusted_prices": True, "stocks": [stock]})
        payload = json.loads(feed.output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["source_id"], "mark1-3-prototype-demo-trigger")
        self.assertEqual(payload["signals"][0]["action"], "buy")
        self.assertEqual(payload["signals"][0]["strategy_id"], "mark1-3-prototype")
        self.service.submit.assert_not_called()
        self.engine.enable_orders.assert_not_called()


if __name__ == "__main__":
    unittest.main()
