"""External model worker tests: temporary files and fake accounts only."""
import copy
from datetime import timedelta
from decimal import Decimal
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from dockdack.lstm30_adapter import read_json
from dockdack.models import TradingMode
from dockdack.prototype_external import (
    ExternalPrototypeFeed, MODEL_IDS, PrototypeProcessClient, PrototypeWorker,
    child_environment, json_line, parse_message, serve,
)
from dockdack.signal_bridge import ExternalPolicy
import test_mark1_trigger as bridge_fixtures
from test_lstm30_adapter import NOW, position
from test_mark1_adapter import mark1_chart, prediction


def fake_predictor(model_id, market="domestic", probability=.7):
    result = prediction(probability)
    result.update(strategy_id=model_id, take_profit_pct=1. if model_id == MODEL_IDS[0] else .5,
                  stop_loss_pct=.9 if model_id == MODEL_IDS[0] else .4,
                  version="fixture-v1", bundle_manifest_sha256="a" * 64)
    return SimpleNamespace(metadata={"market": market, "strategy_id": model_id,
                                     "version": "fixture-v1", "bundle_manifest_sha256": "a" * 64},
                           predict=Mock(return_value=result))


def request_for(model_id, chart=None):
    chart = chart or mark1_chart()
    return {"schema_version": 1, "model_id": model_id, "operation": "produce", "chart": chart,
            "positions": {row["watch_id"]: position(row) for row in chart["stocks"]},
            "now": NOW.isoformat(), "max_krw": "10000", "max_usd": "1000"}


class WorkerTests(unittest.TestCase):
    def test_both_models_emit_distinct_buy_provenance_and_exit_brackets(self):
        results = []
        for model_id, take, stop in ((MODEL_IDS[0], "101", "99.1"), (MODEL_IDS[1], "100.5", "99.6")):
            worker = PrototypeWorker(model_id, predictors={"domestic": fake_predictor(model_id)})
            response = worker.dispatch(request_for(model_id))
            results.append(response)
            row = response["payload"]["signals"][0]
            self.assertEqual(row["action"], "buy")
            self.assertEqual((row["take_profit_price"], row["stop_loss_price"]), (take, stop))
            self.assertEqual(row["strategy_id"], model_id)
            self.assertTrue(row["signal_id"].startswith(model_id + ":"))
            self.assertEqual(row["model_manifest_sha256"], "a" * 64)
        self.assertNotEqual(results[0]["payload"]["source_id"], results[1]["payload"]["source_id"])

    def test_real_or_wrong_model_request_rejected(self):
        worker = PrototypeWorker(MODEL_IDS[0], predictors={})
        request = request_for(MODEL_IDS[1])
        with self.assertRaisesRegex(ValueError, "identity"):
            worker.dispatch(request)
        request["model_id"] = MODEL_IDS[0]
        request["chart"]["trading_mode"] = "real"
        with self.assertRaisesRegex(ValueError, "DEMO"):
            worker.dispatch(request)

    def test_missing_position_is_not_flat(self):
        worker = PrototypeWorker(MODEL_IDS[0], predictors={"domestic": fake_predictor(MODEL_IDS[0])})
        request = request_for(MODEL_IDS[0])
        request["positions"] = {}
        result = worker.dispatch(request)
        self.assertEqual(result["payload"]["signals"][0]["action"], "hold")
        self.assertEqual(result["diagnostics"][0]["reason"], "QUOTE_OR_POSITION_UNAVAILABLE")
        worker.predictors["domestic"].predict.assert_not_called()

    def test_signal_replay_is_immutable_across_child_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.json"
            first = PrototypeWorker(MODEL_IDS[1], state_path=state,
                                    predictors={"domestic": fake_predictor(MODEL_IDS[1])})
            request = request_for(MODEL_IDS[1])
            response = first.dispatch(request)
            other_model = fake_predictor(MODEL_IDS[1], probability=.1)
            second = PrototypeWorker(MODEL_IDS[1], state_path=state, predictors={"domestic": other_model})
            replay = second.dispatch(request)
            self.assertEqual(response["payload"], replay["payload"])
            self.assertEqual(response["diagnostics"], replay["diagnostics"])
            other_model.predict.assert_not_called()

    def test_new_current_price_gets_new_inference(self):
        model = fake_predictor(MODEL_IDS[1])
        worker = PrototypeWorker(MODEL_IDS[1], predictors={"domestic": model})
        request = request_for(MODEL_IDS[1])
        worker.dispatch(request)
        request["chart"]["export_id"] = "next-export"
        request["chart"]["stocks"][0]["price"] = "100.5"
        worker.dispatch(request)
        self.assertEqual(model.predict.call_count, 2)
        self.assertEqual(model.predict.call_args.kwargs["current_price"], Decimal("100.5"))

    def test_stale_quotes_and_positions_hold(self):
        for field in ("quote", "position"):
            request = request_for(MODEL_IDS[0])
            if field == "quote":
                request["chart"]["stocks"][0]["quote_fetched_at"] = (NOW - timedelta(seconds=16)).isoformat()
            else:
                next(iter(request["positions"].values()))["fetched_at"] = (NOW - timedelta(seconds=16)).isoformat()
            worker = PrototypeWorker(MODEL_IDS[0], predictors={"domestic": fake_predictor(MODEL_IDS[0])})
            self.assertEqual(worker.dispatch(request)["payload"]["signals"][0]["action"], "hold")

    def test_held_positions_never_sell_from_child(self):
        for model_id in MODEL_IDS:
            request = request_for(model_id)
            stock = request["chart"]["stocks"][0]
            stock["price"] = "102"
            request["positions"][stock["watch_id"]] = position(stock, quantity="2", sellable="2", average="100")
            model = fake_predictor(model_id)
            result = PrototypeWorker(model_id, predictors={"domestic": model}).dispatch(request)
            self.assertEqual(result["payload"]["signals"][0]["action"], "hold")
            model.predict.assert_not_called()

    def test_fixed_threshold_and_wrong_barrier_fail_closed(self):
        for probability, take in ((.5, .5), (.7, 1.)):
            model = fake_predictor(MODEL_IDS[1], probability=probability)
            model.predict.return_value["take_profit_pct"] = take
            result = PrototypeWorker(MODEL_IDS[1], predictors={"domestic": model}).dispatch(request_for(MODEL_IDS[1]))
            self.assertEqual(result["payload"]["signals"][0]["action"], "hold")

    def test_changed_limit_requires_restart(self):
        worker = PrototypeWorker(MODEL_IDS[0], predictors={"domestic": fake_predictor(MODEL_IDS[0])})
        request = request_for(MODEL_IDS[0])
        worker.dispatch(request)
        request["max_krw"] = "20000"
        with self.assertRaisesRegex(ValueError, "Restart"):
            worker.dispatch(request)

    def test_json_protocol_rejects_duplicate_keys_and_nonfinite(self):
        for value in ('{"x":1,"x":2}', '{"x":NaN}'):
            with self.assertRaises(ValueError):
                parse_message(value)
        with self.assertRaises(ValueError):
            json_line({"x": float("inf")})

    def test_serve_echoes_request_identity_and_has_no_eval(self):
        output = io.StringIO()
        request = {"schema_version": 1, "request_id": "test", "model_id": MODEL_IDS[0], "operation": "health"}
        serve(PrototypeWorker(MODEL_IDS[0], predictors={}), io.StringIO(json_line(request)), output)
        reply = json.loads(output.getvalue())
        self.assertTrue(reply["ok"])
        self.assertEqual(reply["request_id"], "test")
        self.assertEqual(reply["result"]["trading_mode"], "demo")


class ProcessTests(unittest.TestCase):
    def test_environment_does_not_forward_credentials(self):
        with patch.dict(os.environ, {"KIWOOM_APP_KEY": "secret", "OPENAI_API_KEY": "secret",
                                     "HTTP_PROXY": "secret", "ACCOUNT_NUMBER": "secret"}):
            environment = child_environment()
        for key in ("KIWOOM_APP_KEY", "OPENAI_API_KEY", "HTTP_PROXY", "ACCOUNT_NUMBER"):
            self.assertNotIn(key, environment)

    def test_real_subprocesses_are_separate_and_one_failure_does_not_stop_other(self):
        clients = [PrototypeProcessClient(model_id, timeout=15) for model_id in MODEL_IDS]
        for client in clients:
            self.addCleanup(client.close)
        self.assertFalse(any(client.is_alive for client in clients))
        first, second = [client.request("health") for client in clients]
        self.assertNotEqual(first["pid"], second["pid"])
        self.assertNotIn(os.getpid(), (first["pid"], second["pid"]))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            clients[0].request("arbitrary-python-code")
        self.assertFalse(clients[0].is_alive)
        self.assertEqual(clients[1].request("health")["pid"], second["pid"])

    def test_stopped_process_cannot_restart_for_revalidation(self):
        client = PrototypeProcessClient(MODEL_IDS[0])
        self.addCleanup(client.close)
        client.request("health")
        client.close()
        with self.assertRaisesRegex(ValueError, "not running"):
            client.request("health", start=False)
        self.assertFalse(client.is_alive)

    def test_stdio_cli_accepts_only_data_operations(self):
        request = {"schema_version": 1, "model_id": MODEL_IDS[1], "request_id": "cli", "operation": "health"}
        result = subprocess.run([sys.executable, "-m", "examples.run_prototype_signal", "--model", MODEL_IDS[1], "--stdio"],
                                input=json_line(request), capture_output=True, text=True, encoding="utf-8",
                                env=child_environment(), timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["result"]["model_id"], MODEL_IDS[1])

    def test_cli_network_guard_blocks_connections(self):
        from examples.run_prototype_signal import deny_network
        with self.assertRaisesRegex(RuntimeError, "no network"):
            deny_network(("127.0.0.1", 1))

    def test_timeout_kills_only_the_stalled_child(self):
        client = PrototypeProcessClient(MODEL_IDS[0], timeout=.001)
        process = SimpleNamespace(poll=Mock(return_value=None), terminate=Mock(), wait=Mock(),
                                  stdin=SimpleNamespace(write=Mock(), flush=Mock(), close=Mock()),
                                  stdout=SimpleNamespace(close=Mock()))
        client.process = process
        with self.assertRaisesRegex(TimeoutError, "timed out"):
            client.request("health")
        process.terminate.assert_called_once()
        self.assertFalse(client.is_alive)

    def test_wrong_reply_identity_kills_child_and_cannot_leak_other_model(self):
        client = PrototypeProcessClient(MODEL_IDS[0])
        def bad_reply(line):
            request = json.loads(line)
            client._responses.put(json_line({"schema_version": 1, "request_id": request["request_id"],
                                              "model_id": MODEL_IDS[1], "ok": True, "result": {}}))
        process = SimpleNamespace(poll=Mock(return_value=None), terminate=Mock(), wait=Mock(),
                                  stdin=SimpleNamespace(write=bad_reply, flush=Mock(), close=Mock()),
                                  stdout=SimpleNamespace(close=Mock()))
        client.process = process
        with self.assertRaisesRegex(ValueError, "identity"):
            client.request("health")
        process.terminate.assert_called_once()
        self.assertFalse(client.is_alive)

    def test_standalone_file_failure_clears_previous_signals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "signals.json"
            from dockdack.lstm30_adapter import atomic_json
            atomic_json(output, {"signals": [{"action": "buy"}]})
            result = subprocess.run(
                [sys.executable, "-m", "examples.run_prototype_signal", "--model", MODEL_IDS[1],
                 "--chart", str(root / "missing-chart.json"), "--positions", str(root / "positions.json"),
                 "--state", str(root / "state.json"), "--output", str(output), "--once"],
                capture_output=True, text=True, encoding="utf-8", env=child_environment(), timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(read_json(output)["signals"], [])


class LocalClient:
    def __init__(self, model_id, *, bundle_root=None, state_path=None):
        self.worker = PrototypeWorker(model_id, state_path=state_path,
                                      predictors={"domestic": fake_predictor(model_id)})
        self.is_alive = False
        self.requests = []

    def request(self, operation, *, start=True, **values):
        self.requests.append((operation, values))
        if not self.is_alive and not start:
            raise ValueError("not running")
        self.is_alive = True
        return self.worker.dispatch({"schema_version": 1, "model_id": self.worker.model_id,
                                     "operation": operation, **values})

    def close(self):
        self.is_alive = False


class FeedTests(unittest.TestCase):
    setUp = bridge_fixtures.BridgeTests.setUp
    execution_args = bridge_fixtures.BridgeTests.execution_args

    def feed(self, model_id=MODEL_IDS[0], name="external.json"):
        from dockdack.prototype_external import model_bridge_type
        policy = ExternalPolicy(model_bridge_type(model_id).source_id, 3, Decimal(10000), Decimal(1000))
        result = ExternalPrototypeFeed(self.window, model_id, policy, self.root / name, client_factory=LocalClient)
        self.addCleanup(result.close)
        return result

    def test_two_feeds_use_separate_files_state_and_model_identity(self):
        feeds = [self.feed(MODEL_IDS[0], "old.json"), self.feed(MODEL_IDS[1], "new.json")]
        for feed in feeds:
            feed.publish(mark1_chart())
            self.assertEqual(read_json(feed.output_path)["source_id"], feed.source_id)
            self.assertTrue(feed._ready)
        self.assertNotEqual(feeds[0].client.worker.state_path, feeds[1].client.worker.state_path)
        self.service.submit.assert_not_called()
        self.engine.enable_orders.assert_not_called()

    def test_market_by_market_publish_retains_both_latest_diagnostics_bounded(self):
        feed = self.feed(MODEL_IDS[1])
        feed.client.worker.predictors['us'] = fake_predictor(MODEL_IDS[1], market='us')
        feed.publish(mark1_chart())
        chart = mark1_chart(market='us')
        chart['export_id'] = 'us-export'
        feed.publish(chart)
        self.assertEqual(set(feed.diagnostics), {'domestic:KRX:005930', 'us:ND:AAPL'})
        self.assertTrue(all(row.get('prediction') for row in feed.diagnostics.values()))
        feed.diagnostics = {f'old-{index}': {'watch_id': f'old-{index}'} for index in range(501)}
        chart['export_id'] = 'next-us-export'
        feed.publish(chart)
        self.assertEqual(len(feed.diagnostics), 500)
        self.assertIn('us:ND:AAPL', feed.diagnostics)

    def test_failed_feed_replaces_previous_buy_with_hold_other_feed_works(self):
        old, new = self.feed(), self.feed(MODEL_IDS[1], "new.json")
        old.publish(mark1_chart())
        old.client.request = Mock(side_effect=TimeoutError("unavailable"))
        chart = mark1_chart()
        chart["export_id"] = "fresh-export"
        old.publish(chart)
        new.publish(chart)
        self.assertFalse(old._ready)
        self.assertEqual(read_json(old.output_path)["signals"][0]["action"], "hold")
        self.assertEqual(read_json(new.output_path)["signals"][0]["action"], "buy")

    def test_closed_or_dead_feed_cannot_validate_old_signal(self):
        feed = self.feed()
        feed.publish(mark1_chart())
        feed.client.close()
        with self.assertRaisesRegex(ValueError, "unavailable"):
            feed.validate_execution(None, None, None, None)
        feed.close()
        feed.publish(mark1_chart())
        self.assertFalse(feed.client.is_alive)

    def test_real_environment_fails_without_child_start(self):
        feed = self.feed()
        self.service.mode = TradingMode.REAL
        feed.publish(mark1_chart())
        self.assertFalse(feed._ready)
        self.assertFalse(feed.client.requests)
        self.assertEqual(read_json(feed.output_path)["signals"][0]["action"], "hold")

    def test_source_policy_cannot_be_other_model(self):
        with self.assertRaisesRegex(ValueError, "policy"):
            ExternalPrototypeFeed(self.window, MODEL_IDS[1], self.engine.external_policy, self.root / "wrong.json")

    def test_proxy_reinference_uses_child_and_actual_rounded_price(self):
        feed = self.feed(MODEL_IDS[1])
        feed.publish(mark1_chart())
        model = feed.bridge.predictors["domestic"]
        bars = [[100., 103., 99., 101., 1000.] for _ in range(30)]
        model.predict(bars, current_price=Decimal("100.1"))
        self.assertEqual(feed.client.requests[-1][0], "predict")
        self.assertEqual(feed.client.requests[-1][1]["current_price"], "100.1")

    def test_preflight_and_final_send_recheck_quote_and_rounded_price_in_child(self):
        feed = self.feed(MODEL_IDS[1])
        feed.publish(mark1_chart())
        self.service.safety_account.reset_mock()
        item, rule, fresh = self.execution_args()
        for stage in ("preflight", "final_send"):
            feed.validate_execution(item, rule, fresh, Decimal("100.1"), stage=stage)
        prices = [values["current_price"] for operation, values in feed.client.requests if operation == "predict"]
        self.assertEqual(prices, ["100", "100.1", "100", "100.1"])
        self.service.safety_account.assert_not_called()
        self.service.submit.assert_not_called()

    def test_other_models_confirmed_holding_does_not_block_own_entry(self):
        self.engine.prototype_lots_enabled = True
        old, new = self.feed(), self.feed(MODEL_IDS[1], "new.json")
        stock = mark1_chart()["stocks"][0]
        aggregate = position(stock, quantity="2", sellable="2", average="100")
        inventory = {"reconciled": True, "issues": [], "lots": [
            {"source_id": old.source_id, "strategy_id": old.strategy_id,
             "quantity_remaining": Decimal(2), "available_quantity": Decimal(2), "average_price": Decimal(100)}]}
        self.store.prototype_inventory = Mock(return_value=inventory)
        self.store.prototype_pending_buys = Mock(return_value=[])
        for feed in (old, new):
            feed.bridge._position = Mock(return_value=aggregate)
            feed.publish(mark1_chart())
        self.assertEqual(read_json(old.output_path)["signals"][0]["action"], "hold")
        self.assertEqual(read_json(new.output_path)["signals"][0]["action"], "buy")
        self.store.prototype_inventory.assert_called_with(stock["watch_id"], broker_quantity=Decimal(2), broker_sellable=Decimal(2))

    def test_strategy_inventory_uses_own_weighted_cost_and_sellable(self):
        self.engine.prototype_lots_enabled = True
        feed = self.feed(MODEL_IDS[1])
        stock = mark1_chart()["stocks"][0]
        feed.bridge._position = Mock(return_value=position(stock, quantity="6", sellable="5", average="999"))
        base = {"source_id": feed.source_id, "strategy_id": feed.strategy_id}
        self.store.prototype_inventory = Mock(return_value={"reconciled": True, "lots": [
            {**base, "quantity_remaining": 1, "available_quantity": 1, "average_price": 100},
            {**base, "quantity_remaining": 2, "available_quantity": 1, "average_price": 103},
            {"source_id": "mark1-prototype-demo-trigger", "strategy_id": MODEL_IDS[0],
             "quantity_remaining": 3, "available_quantity": 3, "average_price": 300}]})
        self.store.prototype_pending_buys = Mock(return_value=[])
        result = feed._position(stock)
        self.assertEqual((result["quantity"], result["sellable_quantity"], result["average_price"]), ("3", "2", "102"))

    def test_reconciliation_failure_and_own_pending_buy_are_unknown_not_flat(self):
        self.engine.prototype_lots_enabled = True
        for reconciled, pending in ((False, []), (True, [{"order_id": "pending"}])):
            feed = self.feed(MODEL_IDS[1], f"blocked-{reconciled}.json")
            self.store.prototype_inventory = Mock(return_value={"reconciled": reconciled, "lots": []})
            self.store.prototype_pending_buys = Mock(return_value=pending)
            feed.publish(mark1_chart())
            self.assertEqual(read_json(feed.output_path)["signals"][0]["action"], "hold")
            feed.client.worker.predictors["domestic"].predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
