"""Offline mark1.1 wire, fresh-price validation and strategy-owned brackets."""
import copy
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from dockdack.mark1_1_trigger import (
    Mark11DemoSignalProducer, Mark11TriggerBridge, SOURCE_ID, STRATEGY_ID,
    decide_position,
)
from dockdack.mark1_trigger import Mark1DemoSignalProducer
from dockdack.models import TradingMode
from dockdack.signal_bridge import ExternalPolicy, ingest_signals
from test_lstm30_adapter import NOW, position
from test_mark1_adapter import mark1_chart, prediction
from test_mark1_trigger import predictor as old_predictor
import test_mark1_trigger as old_trigger_tests
import test_mark1_paper_execution as old_execution_tests


def new_prediction(probability=.7):
    return {**prediction(probability), "take_profit_pct": .5, "stop_loss_pct": .4,
            "policy_threshold": .5, "stop_probability_cap": 1., "strategy_id": STRATEGY_ID,
            "title": "mark1.1 prototype", "version": "20260924-v1", "bundle_manifest_sha256": "a" * 64}


def new_predictor(probability=.7):
    return SimpleNamespace(metadata={"market": "domestic", "model_name": "fixture", "strategy_id": STRATEGY_ID},
                           predict=Mock(return_value=new_prediction(probability)))


class Mark11ProducerTests(unittest.TestCase):
    def make(self, probability=.7, provider=position, **kwargs):
        model = new_predictor(probability)
        return Mark11DemoSignalProducer({"domestic": model}, position_provider=provider,
            quantity=1, max_krw="10000", max_usd="1000", clock=lambda: NOW, **kwargs), model

    def test_strict_threshold_and_durable_identity(self):
        for probability, action in ((.5, "hold"), (.50001, "buy")):
            producer, model = self.make(probability)
            payload, diagnostics = producer(mark1_chart())
            row = payload["signals"][0]
            self.assertEqual(row["action"], action)
            self.assertEqual(payload["source_id"], SOURCE_ID)
            self.assertTrue(row["signal_id"].startswith(STRATEGY_ID + ":"))
            self.assertEqual(diagnostics[0]["title"], "mark1.1 prototype")
            self.assertIn("1.005", diagnostics[0]["target_basis"])
            self.assertFalse(diagnostics[0]["deployment_allowed"])
            if action == "buy":
                self.assertEqual((row["take_profit_price"], row["stop_loss_price"]), ("100.5", "99.6"))
                self.assertEqual(row["model_title"], "mark1.1 prototype")
                self.assertEqual(row["strategy_id"], STRATEGY_ID)
                self.assertEqual(row["model_version"], "20260924-v1")
                self.assertEqual(row["model_manifest_sha256"], "a" * 64)

    def test_every_new_price_reinfers_but_exact_export_replay_is_immutable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "new-state.json"
            producer, model = self.make(state_path=path)
            chart = mark1_chart()
            first = producer(chart)
            replay, replay_model = self.make(state_path=path)
            self.assertEqual(replay(chart), first)
            replay_model.predict.assert_not_called()
            chart = copy.deepcopy(chart)
            chart["export_id"] = "changed-price"
            chart["stocks"][0]["price"] = "100.2"
            producer(chart)
            self.assertEqual(model.predict.call_count, 2)
            self.assertEqual(model.predict.call_args.kwargs["current_price"], Decimal("100.2"))

    def test_old_state_and_old_model_cannot_masquerade_as_new(self):
        producer, model = self.make()
        for changes in ({"take_profit_pct": 1.}, {"stop_loss_pct": .9}, {"strategy_id": "mark1-prototype"}):
            model.predict.return_value = {**new_prediction(), **changes}
            chart = mark1_chart()
            chart["export_id"] = "check-" + next(iter(changes))
            payload, rows = producer(chart)
            self.assertEqual(payload["signals"][0]["action"], "hold")
            self.assertEqual(rows[0]["reason"], "PREDICTION_UNAVAILABLE")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old-state.json"
            old = Mark1DemoSignalProducer({"domestic": old_predictor()}, position_provider=position,
                quantity=1, max_krw="10000", max_usd="1000", clock=lambda: NOW, state_path=path)
            old(mark1_chart())
            with self.assertRaisesRegex(ValueError, "own decision state"):
                self.make(state_path=path)

    def test_held_exits_use_new_bounds_but_only_gui_owns_selling(self):
        for price, reason in (("100.5", "TAKE_PROFIT_0_5PCT"), ("99.6", "STOP_LOSS_0_4PCT")):
            self.assertEqual(decide_position(current_price=price, quantity=1, sellable_quantity=1,
                                             average_price=100)["reason"], reason)
            producer, model = self.make(provider=lambda row: position(row, quantity="1", sellable="1", average="100"))
            chart = mark1_chart()
            chart["stocks"][0]["price"] = price
            payload, rows = producer(chart)
            self.assertEqual(payload["signals"][0]["action"], "hold")
            self.assertEqual(rows[0]["reason"], "POSITION_EXIT_MANAGED_BY_GUI")
            model.predict.assert_not_called()
        self.assertEqual(decide_position(current_price="100.4", quantity=1, sellable_quantity=1,
                                         average_price=100)["action"], "hold")

    def test_real_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, "모의"):
            self.make(trading_mode="real")


class Mark11BridgeTests(unittest.TestCase):
    # Reuse only the network-disabled fixture, not legacy model expectations.
    setUp = old_trigger_tests.BridgeTests.setUp
    execution_args = old_trigger_tests.BridgeTests.execution_args

    def new_bridge(self):
        self.engine.external_policy = ExternalPolicy(SOURCE_ID, 1, Decimal(10000), Decimal(1000))
        self.model = new_predictor()
        return Mark11TriggerBridge(self.window, predictors={"domestic": self.model})

    def test_source_state_and_status_are_independent(self):
        bridge = self.new_bridge()
        bridge.publish(mark1_chart())
        import json
        payload = json.loads(self.engine.external_reader.path.read_text())
        self.assertEqual(payload["source_id"], SOURCE_ID)
        self.assertEqual(payload["signals"][0]["model_title"], "mark1.1 prototype")
        self.assertIn("+0.5%", bridge.status)
        self.assertTrue((self.root / bridge.state_filename).is_file())
        self.assertFalse((self.root / "exchange/mark1-trigger-decisions-v1.json").exists())
        self.service.submit.assert_not_called()
        self.engine.enable_orders.assert_not_called()

    def test_actual_rounded_price_rechecked_with_correct_model_contract(self):
        bridge = self.new_bridge()
        item, rule, fresh = self.execution_args()
        bridge.validate_execution(item, rule, fresh, Decimal("100.1"))
        self.assertEqual([call.kwargs["current_price"] for call in self.model.predict.call_args_list],
                         [Decimal(100), Decimal("100.1")])
        self.model.predict.return_value = {**new_prediction(), "take_profit_pct": 1.}
        with self.assertRaisesRegex(ValueError, "정책"):
            bridge.validate_execution(item, rule, fresh, Decimal(100))
        self.service.safety_account.assert_not_called()

    def test_lazy_alias_load_and_wrong_environment_fail_closed(self):
        self.new_bridge()
        bridge = Mark11TriggerBridge(self.window)
        module = SimpleNamespace(Mark11PrototypePredictor=Mock(return_value=self.model))
        with patch.dict("sys.modules", {"dockdack.mark1_1_prototype_inference": module}):
            bridge.publish(mark1_chart())
        module.Mark11PrototypePredictor.assert_called_once_with(bridge.bundle_root, "domestic")
        self.service.mode = TradingMode.REAL
        with self.assertRaisesRegex(ValueError, "실전"):
            bridge._ensure_demo()


class Mark11PaperTests(unittest.TestCase):
    setUp = old_execution_tests.Mark1PaperExecutionTests.setUp
    validator = old_execution_tests.Mark1PaperExecutionTests.validator

    def queue(self, *, wrong_identity=False):
        self.policy = ExternalPolicy(SOURCE_ID, 1, Decimal("500"), Decimal("1000"))
        self.engine.external_policy = self.policy
        self.engine.configure_source_validators({SOURCE_ID: self.validator})
        from datetime import timedelta
        entry = {"signal_id": STRATEGY_ID + ":paper-proof", "export_id": self.chart["export_id"],
                 "market": "domestic", "symbol": "005930", "exchange": "KRX", "action": "buy",
                 "quantity": 1, "max_notional": "500", "generated_at": NOW.isoformat(),
                 "expires_at": (NOW + timedelta(minutes=2)).isoformat(), "strategy_id": STRATEGY_ID,
                 "model_title": "mark1.1 prototype", "model_version": "20260924-v1",
                 "model_manifest_sha256": "a" * 64}
        if wrong_identity:
            entry["signal_id"] = "mark1-prototype:wrong-family"
        return ingest_signals(self.store, {"schema_version": 1, "source_id": SOURCE_ID,
                              "trading_mode": "demo", "signals": [entry]}, self.policy, now=NOW)

    def test_fake_accepted_buy_persists_half_boundaries_and_model(self):
        self.queue()
        self.engine.enable_orders("DEMO_AUTOTRADE")  # Fake broker only, temporary ledger.
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 1)
        targets = self.store.exit_targets(self.item.id)
        self.assertEqual(targets["source"], SOURCE_ID)
        self.assertEqual(targets["take_profit_price"], Decimal("100.5"))
        self.assertEqual(targets["stop_loss_price"], Decimal("99.6"))
        record = self.store.external_for_rule(targets["rule_id"])
        import json
        saved = json.loads(record["payload"])
        self.assertEqual(saved["model_title"], "mark1.1 prototype")
        self.assertEqual(saved["model_version"], "20260924-v1")
        self.assertEqual(len(self.calls), 2)

    def test_conflicting_family_never_sends(self):
        try:
            self.queue(wrong_identity=True)
        except ValueError:
            pass  # Rejecting the malformed payload at ingestion is also correct.
        else:
            self.engine.enable_orders("DEMO_AUTOTRADE")
            self.engine.poll()
        self.assertEqual(self.service.submitted, [])

    def test_old_model_validator_cannot_authorize_new_model_order(self):
        from dockdack.mark1_trigger import Mark1TriggerBridge
        self.queue()
        model = old_predictor()
        window = SimpleNamespace(service=self.service, store=self.store, engine=self.engine)
        wrong_bridge = Mark1TriggerBridge(window, predictors={"domestic": model})
        self.engine.configure_source_validators({SOURCE_ID: wrong_bridge.validate_execution})
        self.engine.enable_orders("DEMO_AUTOTRADE")
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])
        model.predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
