"""Functional identity/integrity checks; not a profitability backtest."""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from dockdack.mark1_intraday_inference import MarkIntradayPredictor
from dockdack.mark1_intraday_models import VARIANTS, feature_matrix
from dockdack.mark1_intraday_extra_models import (
    VARIANTS as EXTRA_VARIANTS, EFFECTIVE_LOOKBACK as EXTRA_LOOKBACK,
    feature_matrix as extra_feature_matrix,
)
from dockdack.prototype_external import MODEL_IDS, PrototypeProcessClient, PrototypeWorker
from dockdack.signal_bridge import prototype_family
from dockdack.signals.mark1_intraday_trigger import MODEL_IDS as NEW_MODEL_IDS
from dockdack.market_schedule import calendar_for
from dockdack.models import Market


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "models" / "mark1_intraday"
EXTRA_BUNDLE = ROOT / "models" / "mark1_intraday_extra"
HISTORY = np.array([[float(100 + index), float(102 + index),
                     float(99 + index), float(101 + index), 1000.]
                    for index in range(30)])


class IntradayProxyModelsTest(unittest.TestCase):
    def test_ten_unique_demo_identities(self):
        self.assertEqual(NEW_MODEL_IDS, tuple(f"mark1-{n}-prototype" for n in range(13, 23)))
        self.assertTrue(set(NEW_MODEL_IDS).issubset(MODEL_IDS))
        for model_id in NEW_MODEL_IDS:
            family = prototype_family(model_id + "-demo-trigger")
            self.assertEqual(family.id, model_id)
            self.assertEqual(str(family.take_profit), "0.01")
            self.assertEqual(str(family.stop_loss), "0.009")

    def test_features_are_causal_and_variant_specific(self):
        vectors = [feature_matrix(HISTORY, 131., variant) for variant in VARIANTS]
        self.assertTrue(all(vector.shape == (1, 8) and np.isfinite(vector).all()
                            for vector in vectors))
        self.assertEqual(len({vector.tobytes() for vector in vectors}), 5)
        modified = HISTORY.copy()
        modified[-1, 1] += 3
        self.assertFalse(np.array_equal(vectors[2], feature_matrix(modified, 131., "mark1.15")))
        self.assertFalse(np.array_equal(vectors[0], feature_matrix(HISTORY, 132., "mark1.13")))
        early_volume = HISTORY.copy()
        early_volume[:9, 4] *= 7
        # Mark1.13's volume_slope reads only the last 20 bars. Its maximum
        # actual dependency is 21 bars for momentum_20, not all 30 inputs.
        np.testing.assert_array_equal(vectors[0],
                                      feature_matrix(early_volume, 131., "mark1.13"))
        self.assertFalse(np.array_equal(vectors[4],
                                        feature_matrix(early_volume, 131., "mark1.17")))
        bad = HISTORY.copy()
        bad[-1, 1] = bad[-1, 2] - 1
        with self.assertRaises(ValueError):
            feature_matrix(bad, 131., "mark1.13")
        extra = [extra_feature_matrix(HISTORY, 131., variant) for variant in EXTRA_VARIANTS]
        self.assertEqual(len({vector.tobytes() for vector in extra}), 5)
        self.assertTrue(all(vector.shape == (1, 8) and np.isfinite(vector).all()
                            for vector in extra))

    def test_all_twenty_saved_weights_are_distinct_and_finite(self):
        hashes = []
        for root, variants in ((BUNDLE, VARIANTS), (EXTRA_BUNDLE, EXTRA_VARIANTS)):
            manifest_bytes = (root / "manifest.json").read_bytes()
            digest = hashlib.sha256(manifest_bytes).hexdigest()
            self.assertEqual((root / "manifest.sha256").read_text("ascii"), digest)
            manifest = json.loads(manifest_bytes)
            self.assertFalse(manifest["risk_flags"]["deployment_allowed"])
            self.assertFalse(manifest["risk_flags"]["intraday_path_verified"])
            self.assertEqual(manifest["entry_training_basis"], "target_session_observed_open_only")
            lookbacks = EXTRA_LOOKBACK if root == EXTRA_BUNDLE else {
                "mark1.13": 21, "mark1.14": 30, "mark1.15": 30,
                "mark1.16": 21, "mark1.17": 30,
            }
            for market in ("domestic", "us"):
                count = manifest["markets"][market]["sample_count"]
                self.assertTrue(190000 <= count <= 200000)
                for variant in variants:
                    member = manifest["variants"][variant]["models"][market]
                    hashes.append(member["sha256"])
                    self.assertEqual(member["train_examples"], count)
                    self.assertEqual(member["epochs"], 8)
                    self.assertEqual(manifest["variants"][variant]["effective_lookback"],
                                     lookbacks[variant])
                    predictor = MarkIntradayPredictor(root, market, variant)
                    prediction = predictor.predict(HISTORY, current_price=131.)
                    self.assertTrue(0 <= prediction["probability_success"] <= 1)
                    self.assertFalse(prediction["intraday_path_verified"])
                    self.assertFalse(prediction["deployment_allowed"])
                    self.assertEqual(prediction["score_scope"],
                                     "daily_open_whole_session_proxy_not_intraday")
        self.assertEqual(len(set(hashes)), 20)

    def test_tampered_weights_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "mark1_intraday"
            shutil.copytree(BUNDLE, copied)
            path = copied / "domestic-mark1_13.pt"
            with path.open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                MarkIntradayPredictor(copied, "domestic", "mark1.13")

    def test_legacy_bundle_lookback_is_required_even_with_a_new_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "mark1_intraday"
            shutil.copytree(BUNDLE, copied)
            manifest_path = copied / "manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["variants"]["mark1.13"].pop("effective_lookback")
            raw = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")
            manifest_path.write_bytes(raw)
            (copied / "manifest.sha256").write_text(hashlib.sha256(raw).hexdigest(),
                                                    encoding="ascii")
            with self.assertRaisesRegex(ValueError, "variant identity"):
                MarkIntradayPredictor(copied, "domestic", "mark1.13")

    def test_worker_predict_has_no_order_or_broker_access(self):
        bars = HISTORY.tolist()
        for model_id in NEW_MODEL_IDS:
            root = BUNDLE if int(model_id.split("-")[1]) < 18 else EXTRA_BUNDLE
            worker = PrototypeWorker(model_id, bundle_root=root)
            self.assertIsNone(worker.special)
            health = worker.dispatch({"schema_version": 1, "model_id": model_id,
                                      "operation": "health"})
            self.assertEqual(health["trading_mode"], "demo")
            prediction = worker.dispatch({"schema_version": 1, "model_id": model_id,
                                          "operation": "predict", "market": "domestic",
                                          "bars": bars, "current_price": "131"})
            self.assertEqual(prediction["strategy_id"], model_id)
            self.assertFalse(prediction["deployment_allowed"])
            with self.assertRaises(ValueError):
                worker.dispatch({"schema_version": 1, "model_id": model_id,
                                 "operation": "place_order"})

    def test_saved_models_connect_to_demo_signal_production_only(self):
        now = datetime(2026, 9, 15, 1, 0, tzinfo=timezone.utc)
        days = [stamp.date() for stamp in calendar_for(Market.DOMESTIC, 2026).sessions
                if stamp.date() < date(2026, 9, 15)][-30:]
        self.assertEqual(len(days), 30)
        stock = {
            "watch_id": "domestic:KRX:005930", "market": "domestic",
            "exchange": "KRX", "symbol": "005930", "currency": "KRW",
            "status": "ok", "complete": True, "requested_days": 30,
            "available_days": 30, "price": "100", "quote_fetched_at": now.isoformat(),
            "quote_age_seconds": 0, "quote_stale": False,
            "bars": [{"date": day.isoformat(), "open": "100", "high": "103",
                      "low": "99", "close": "101", "volume": "1000",
                      "is_current_day": False} for day in days],
        }
        chart = {"schema_version": 1, "export_id": "intraday-proxy-test",
                 "trading_mode": "demo", "source": "kiwoom_demo",
                 "created_at": now.isoformat(), "adjusted_prices": True,
                 "stocks": [stock]}
        position = {"market": "domestic", "exchange": "KRX", "symbol": "005930",
                    "currency": "KRW", "quantity": "0", "sellable_quantity": "0",
                    "average_price": None, "fetched_at": now.isoformat()}
        for model_id in NEW_MODEL_IDS:
            root = BUNDLE if int(model_id.split("-")[1]) < 18 else EXTRA_BUNDLE
            worker = PrototypeWorker(model_id, bundle_root=root)
            response = worker.dispatch({"schema_version": 1, "model_id": model_id,
                                        "operation": "produce", "chart": chart,
                                        "positions": {stock["watch_id"]: position},
                                        "now": now.isoformat(), "max_krw": "10000",
                                        "max_usd": "1000"})
            self.assertEqual(response["payload"]["source_id"], model_id + "-demo-trigger")
            rows = response["payload"]["signals"]
            self.assertEqual(len(rows), 1)
            self.assertIn(rows[0]["action"], {"buy", "hold"})
            self.assertTrue(rows[0]["signal_id"].startswith(model_id + ":"))
            self.assertFalse(response["metadata"]["domestic"]["deployment_allowed"])

    def test_real_child_process_rpc_for_new_model(self):
        for model_id in NEW_MODEL_IDS:
            root = BUNDLE if int(model_id.split("-")[1]) < 18 else EXTRA_BUNDLE
            client = PrototypeProcessClient(model_id, bundle_root=root, timeout=15)
            try:
                health = client.request("health")
                self.assertEqual(health["trading_mode"], "demo")
                prediction = client.request("predict", market="us", bars=HISTORY.tolist(),
                                            current_price="131")
                self.assertFalse(prediction["intraday_path_verified"])
            finally:
                client.close()


if __name__ == "__main__":
    unittest.main()
