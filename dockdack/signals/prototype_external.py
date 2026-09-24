"""Separate-process, data-only prototype feeds for the ordinary DEMO GUI.

The GUI alone reads verified positions and owns account/order permissions. Each
model has its own child process, immutable decision state and signal JSON file.
The child receives chart/position JSON, never credentials or a broker object.
"""
from __future__ import annotations

import contextlib
from datetime import timedelta
from decimal import Decimal
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from uuid import uuid4

import dockdack
from dockdack.runtime_paths import app_home, checkout_root, model_bundle

MODEL_IDS = ("mark1-prototype", "mark1-1-prototype")
MAX_RPC_BYTES = 12_000_000
# Package import root (checkout or site-packages), independent of this module's
# internal location. Model/artifact paths come from runtime_paths, never ROOT.
ROOT = Path(dockdack.__file__).resolve().parent.parent


def model_bridge_type(model_id):
    if model_id == "mark1-prototype":
        from dockdack.mark1_trigger import Mark1TriggerBridge
        return Mark1TriggerBridge
    if model_id == "mark1-1-prototype":
        from dockdack.mark1_1_trigger import Mark11TriggerBridge
        return Mark11TriggerBridge
    raise ValueError("Unknown prototype model identity")


def json_line(value):
    line = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    if len(line.encode("utf-8")) > MAX_RPC_BYTES:
        raise ValueError("External prototype message exceeds size limit")
    return line


def parse_message(line):
    if len(line.encode("utf-8")) > MAX_RPC_BYTES:
        raise ValueError("External prototype message exceeds size limit")
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate external prototype JSON key")
            value[key] = item
        return value
    def invalid(value):
        raise ValueError("Non-finite external prototype JSON number")
    return json.loads(line, object_pairs_hook=unique, parse_constant=invalid)


class PrototypeWorker:
    """Child-side model/producer owner; all inputs and outputs are JSON data."""

    def __init__(self, model_id, *, bundle_root=None, state_path=None, predictors=None):
        self.model_id = model_id
        self.bridge_type = model_bridge_type(model_id)
        self.bundle_root = Path(bundle_root) if bundle_root is not None else model_bundle(Path(self.bridge_type.bundle_directory).name)
        self.state_path = Path(state_path) if state_path else None
        self.predictors = dict(predictors or {})
        self.injected = predictors is not None
        self.producer = None
        self.positions = {}
        self.now = None
        self.started = 0.
        self.limits = None

    def _clock(self):
        return self.now + timedelta(seconds=max(0., time.monotonic() - self.started))

    def predictor(self, market):
        if market not in {"domestic", "us"}:
            raise ValueError("Unsupported prototype market")
        if market not in self.predictors:
            if self.injected:
                raise ValueError("Missing injected market model")
            if self.model_id == "mark1-prototype":
                from dockdack.mark1_prototype_inference import PrototypePredictor
                self.predictors[market] = PrototypePredictor(self.bundle_root, market)
            else:
                from dockdack.mark1_1_prototype_inference import Mark11PrototypePredictor
                self.predictors[market] = Mark11PrototypePredictor(self.bundle_root, market)
        return self.predictors[market]

    def dispatch(self, request):
        from dockdack.lstm30_adapter import timestamp
        if (not isinstance(request, dict) or request.get("schema_version") != 1
                or request.get("model_id") != self.model_id):
            raise ValueError("External prototype request identity mismatch")
        operation = request.get("operation")
        if operation == "health":
            return {"model_id": self.model_id, "source_id": self.bridge_type.source_id,
                    "trading_mode": "demo", "pid": os.getpid()}
        if operation == "metadata":
            return self.predictor(request.get("market")).metadata
        if operation == "predict":
            result = self.predictor(request.get("market")).predict(
                request.get("bars"), current_price=request.get("current_price"))
            self.bridge_type.producer_type.validate_prediction(result)
            return result
        if operation != "produce":
            raise ValueError("Unsupported external prototype operation")
        chart, positions = request.get("chart"), request.get("positions")
        if (not isinstance(chart, dict) or chart.get("trading_mode") != "demo"
                or chart.get("source") != "kiwoom_demo" or not isinstance(positions, dict)):
            raise ValueError("External prototype requires DEMO chart and position snapshots")
        self.now, self.started = timestamp(request.get("now")), time.monotonic()
        self.positions = positions
        limits = (str(request.get("max_krw")), str(request.get("max_usd")))
        if self.limits is not None and limits != self.limits:
            raise ValueError("Restart external prototype after changing notional limits")
        self.limits = limits
        errors = {}
        for market in {row.get("market") for row in chart.get("stocks", []) if isinstance(row, dict)}:
            try:
                self.predictor(market)
            except Exception as exc:
                errors[str(market)] = str(exc)[:500]
        if self.producer is None:
            self.producer = self.bridge_type.producer_type(
                self.predictors, position_provider=lambda stock: self.positions.get(stock["watch_id"]),
                quantity=1, max_krw=limits[0], max_usd=limits[1], state_path=self.state_path,
                clock=self._clock, trading_mode="demo")
        self.producer.predictors = dict(self.predictors)
        payload, diagnostics = self.producer(chart, now=self.now)
        return {"payload": payload, "diagnostics": diagnostics, "market_errors": errors,
                "metadata": {market: predictor.metadata for market, predictor in self.predictors.items()}}


def serve(worker, input_stream=None, output_stream=None):
    """One request/reply per line. Malformed input never becomes executable code."""
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    while True:
        line = input_stream.readline(MAX_RPC_BYTES + 1)
        if not line:
            break
        request = None
        try:
            request = parse_message(line)
            # Native-library messages cannot corrupt the JSON response stream.
            with contextlib.redirect_stdout(sys.stderr):
                result = worker.dispatch(request)
            reply = {"schema_version": 1, "request_id": request.get("request_id"),
                     "model_id": worker.model_id, "ok": True, "result": result}
        except Exception as exc:
            reply = {"schema_version": 1,
                     "request_id": request.get("request_id") if isinstance(request, dict) else None,
                     "model_id": worker.model_id, "ok": False,
                     "error": str(exc)[:1500] or type(exc).__name__}
        output_stream.write(json_line(reply))
        output_stream.flush()
        if len(line.encode("utf-8")) > MAX_RPC_BYTES:
            break


def child_environment():
    """Allowlist runtime paths; API keys, tokens and account settings are absent."""
    allowed = {"SYSTEMROOT", "WINDIR", "PATH", "PATHEXT", "TEMP", "TMP", "COMSPEC",
               "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "VIRTUAL_ENV",
               "DOCKDACK_HOME", "DOCKDACK_MODEL_ROOT"}
    result = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    paths = [str(ROOT)]
    if sys.version_info[:2] == (3, 13):
        for name in ("selective-deps", "prototype-gui-deps"):
            path = app_home() / "outputs" / "mark1" / name
            if path.is_dir():
                paths.append(str(path))
    result.update(PYTHONPATH=os.pathsep.join(paths), PYTHONNOUSERSITE="1", PYTHONUTF8="1",
                  PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1")
    return result


class PrototypeProcessClient:
    """Bounded synchronous RPC, one model per subprocess; timeout kills that child."""

    def __init__(self, model_id, *, bundle_root=None, state_path=None, timeout=30.):
        if model_id not in MODEL_IDS:
            raise ValueError("Unknown prototype model identity")
        self.model_id, self.bundle_root, self.state_path = model_id, bundle_root, state_path
        self.timeout = float(timeout)
        if not 0 < self.timeout <= 120:
            raise ValueError("External prototype timeout must be within 120 seconds")
        self.process = None
        self._responses = queue.Queue()
        self._lock = threading.RLock()
        self._request_lock = threading.Lock()
        self._generation = 0
        self._active = None
        self.cancel_event = None
        self._stderr = None

    @property
    def is_alive(self):
        process = self.process
        return process is not None and process.poll() is None

    @staticmethod
    def _dispose(process, stderr=None):
        """Never wait for a child or a blocked pipe while holding the GUI caller."""
        def cleanup():
            try:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                pass
            finally:
                for stream in ((getattr(process, "stdin", None), getattr(process, "stdout", None), stderr)):
                    if stream is not None:
                        with contextlib.suppress(OSError, ValueError):
                            stream.close()
        threading.Thread(target=cleanup, name="prototype-cleanup", daemon=True).start()

    def _start(self, cancelled):
        with self._lock:
            if cancelled.is_set():
                raise ValueError("External prototype request cancelled")
            if self.is_alive:
                return self.process, self._responses
            old, old_stderr = self.process, self._stderr
            self.process, self._stderr = None, None
        if old is not None or old_stderr is not None:
            self._dispose(old, old_stderr)
        command = [sys.executable, "-u", "-X", "utf8", "-m", "dockdack.signals.worker",
                   "--model", self.model_id, "--stdio"]
        stderr = None
        if self.bundle_root is not None:
            command.extend(("--bundle", str(self.bundle_root)))
        if self.state_path is not None:
            command.extend(("--state", str(self.state_path)))
            log = Path(self.state_path).with_suffix(".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            stderr = log.open("a", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command, cwd=checkout_root(), env=child_environment(), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=stderr or subprocess.DEVNULL,
                text=True, encoding="utf-8", bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception:
            if stderr is not None:
                stderr.close()
            raise
        responses = queue.Queue()
        with self._lock:
            if cancelled.is_set() or self._active is not cancelled:
                # Popen may finish AFTER its caller timed out/closed/restarted.
                # That late child must never replace the newer process.
                self._dispose(process, stderr)
                raise ValueError("External prototype startup cancelled")
            self.process, self._stderr, self._responses = process, stderr, responses
        stream = process.stdout
        def read_responses():
            try:
                while True:
                    line = stream.readline(MAX_RPC_BYTES + 1)
                    if not line:
                        break
                    responses.put(line)
                    if len(line.encode("utf-8")) > MAX_RPC_BYTES:
                        break
            except (OSError, ValueError):
                pass
            finally:
                responses.put(None)
        threading.Thread(target=read_responses, name=f"prototype-{self.model_id}", daemon=True).start()
        return process, responses

    def request(self, operation, *, start=True, **values):
        # One budget covers serialization, queueing, startup, pipe write and
        # response. Blocking OS/pipe operations live on a cancellable worker;
        # close() does not wait on this request's serialization lock.
        deadline = time.monotonic() + self.timeout
        with self._lock:
            generation = self._generation
        cancelled = threading.Event()

        def check():
            if (cancelled.is_set() or self._generation != generation
                    or self.cancel_event is not None and self.cancel_event.is_set()):
                raise ValueError("External prototype request cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("External prototype request timed out; this feed is blocked")
            return min(remaining, .02)

        acquired = False
        try:
            while not acquired:
                acquired = self._request_lock.acquire(timeout=check())
            with self._lock:
                check()
                self._active = cancelled
            completed = queue.Queue(maxsize=1)

            def exchange():
                try:
                    if start:
                        process, responses = self._start(cancelled)
                    else:
                        with self._lock:
                            if cancelled.is_set() or not self.is_alive:
                                raise ValueError("External prototype process is not running")
                            process, responses = self.process, self._responses
                    request_id = uuid4().hex
                    request = {"schema_version": 1, "request_id": request_id, "model_id": self.model_id,
                               "operation": operation, **values}
                    line = json_line(request)
                    if cancelled.is_set():
                        raise ValueError("External prototype request cancelled")
                    process.stdin.write(line)
                    process.stdin.flush()
                    line = responses.get()
                    result = self._reply(line, request_id)
                    completed.put((True, result))
                except Exception as exc:
                    completed.put((False, exc))

            threading.Thread(target=exchange, name=f"prototype-rpc-{self.model_id}", daemon=True).start()
            while True:
                try:
                    success, result = completed.get(timeout=check())
                    check()
                    if not success:
                        raise result
                    return result
                except queue.Empty:
                    continue
        except Exception:
            # A waiting second caller must not kill somebody else's request
            # merely because its own queueing budget expired.
            if acquired:
                self.close()
            raise
        finally:
            cancelled.set()
            if acquired:
                with self._lock:
                    if self._active is cancelled:
                        self._active = None
                self._request_lock.release()

    def _reply(self, line, request_id):
        if line is None:
            raise ValueError("External prototype process exited")
        reply = parse_message(line)
        if (not isinstance(reply, dict) or reply.get("schema_version") != 1
                or reply.get("request_id") != request_id or reply.get("model_id") != self.model_id
                or type(reply.get("ok")) is not bool):
            raise ValueError("External prototype response identity mismatch")
        if not reply["ok"]:
            raise ValueError(reply.get("error") or "External prototype rejected request")
        return reply["result"]

    def close(self):
        with self._lock:
            self._generation += 1
            if self._active is not None:
                self._active.set()
            process, self.process = self.process, None
            stderr, self._stderr = self._stderr, None
            self._responses.put(None)
        if process is not None or stderr is not None:
            self._dispose(process, stderr)


class _Proxy:
    def __init__(self, target, **overrides):
        self._target = target
        self.__dict__.update(overrides)

    def __getattr__(self, name):
        return getattr(self._target, name)


class _RemotePredictor:
    def __init__(self, client, market, metadata):
        self.client, self.market, self.metadata = client, market, metadata

    def predict(self, bars, *, current_price):
        return self.client.request("predict", start=False, market=self.market, bars=bars,
                                   current_price=str(current_price))


class ExternalPrototypeFeed:
    """GUI adapter with one isolated child/output/policy and original final checks."""

    def __init__(self, window, model_id, policy, output_path, *, bundle_root=None, client_factory=None,
                 account_snapshots=None):
        bridge_type = model_bridge_type(model_id)
        if policy.source_id != bridge_type.source_id:
            raise ValueError("External prototype source policy does not match its model")
        self.window, self.model_id, self.policy = window, model_id, policy
        self.output_path = Path(output_path)
        engine = _Proxy(window.engine, external_policy=policy,
                        external_reader=SimpleNamespace(path=self.output_path))
        self.bridge = bridge_type(_Proxy(window, engine=engine), predictors={}, bundle_root=bundle_root,
                                  account_snapshots=account_snapshots)
        self.source_id, self.strategy_id, self.title = bridge_type.source_id, model_id, bridge_type.title
        state = window.store.path.parent / "exchange" / f"{model_id}-external-decisions-v1.json"
        self.client = (client_factory or PrototypeProcessClient)(
            model_id, bundle_root=self.bridge.bundle_root, state_path=state)
        if isinstance(self.client, PrototypeProcessClient):
            self.client.cancel_event = window.engine._stop
        self.diagnostics = {}
        self.status = f"{self.title} 외장 연결 대기 · 첫 조회 때 별도 프로세스 시작"
        self._ready = False
        self._closed = False

    def _position(self, stock):
        """Use reconciled, confirmed strategy lots in simultaneous-model mode."""
        result = self.bridge._position(stock)
        if not getattr(self.window.engine, "prototype_lots_enabled", False):
            return result
        from dockdack.lstm30_adapter import instrument, number
        watch_id = instrument(stock)
        store = self.window.store
        inventory = store.prototype_inventory(
            watch_id, broker_quantity=number(result["quantity"], "broker quantity", zero=True),
            broker_sellable=number(result["sellable_quantity"], "broker sellable", zero=True))
        if inventory.get("reconciled") is not True:
            raise ValueError("모델별 보유수량과 증권사 잔고가 일치하지 않아 매수를 대기합니다.")
        if store.prototype_pending_buys(watch_id, strategy_id=self.strategy_id):
            raise ValueError("이 모델의 미확정 매수 주문이 있어 추가 매수를 대기합니다.")
        lots = [row for row in inventory["lots"] if row.get("strategy_id") == self.strategy_id]
        if any(row.get("source_id") != self.source_id for row in lots):
            raise ValueError("모델별 보유 기록의 신호 출처가 일치하지 않습니다.")
        quantity = sum((number(row["quantity_remaining"], "lot quantity", zero=True) for row in lots), Decimal(0))
        sellable = sum((number(row["available_quantity"], "lot available", zero=True) for row in lots), Decimal(0))
        cost = sum((number(row["quantity_remaining"], "lot quantity", zero=True)
                    * number(row["average_price"], "lot average") for row in lots
                    if number(row["quantity_remaining"], "lot quantity", zero=True)), Decimal(0))
        return {**result, "quantity": str(quantity), "sellable_quantity": str(sellable),
                "average_price": str(cost / quantity) if quantity else None}

    def publish(self, chart):
        if self._closed or self.window.engine._stop.is_set():
            return
        self._ready = False
        try:
            from dockdack.lstm30_adapter import instrument, atomic_json
            self.bridge._ensure_demo()
            if not isinstance(chart, dict) or chart.get("trading_mode") != "demo" or chart.get("source") != "kiwoom_demo":
                raise ValueError("External prototypes require explicit DEMO charts")
            positions = {}
            for stock in chart.get("stocks", []):
                if not isinstance(stock, dict) or stock.get("status") != "ok":
                    continue
                key = instrument(stock)
                try:
                    positions[key] = self._position(stock)
                except Exception:
                    # Missing is unknown, never zero; producer emits HOLD for it.
                    positions[key] = None
            result = self.client.request("produce", chart=chart, positions=positions,
                                         now=self.window.engine.clock().isoformat(),
                                         max_krw=str(self.policy.max_krw), max_usd=str(self.policy.max_usd))
            payload = result["payload"]
            if (payload.get("source_id") != self.source_id or payload.get("trading_mode") != "demo"
                    or not isinstance(payload.get("signals"), list)):
                raise ValueError("External prototype returned a different source or mode")
            for signal in payload["signals"]:
                if (not str(signal.get("signal_id", "")).startswith(self.strategy_id + ":")
                        or signal.get("action") not in {"buy", "hold"}):
                    raise ValueError("External prototype signal identity or action mismatch")
            self.bridge.predictors = {market: _RemotePredictor(self.client, market, metadata)
                                      for market, metadata in result.get("metadata", {}).items()
                                      if market in {"domestic", "us"} and metadata.get("market") == market}
            self.bridge._ensure_demo()
            atomic_json(self.output_path, payload)
            # Normal GUI sweeps may publish one market at a time. Preserve the
            # other market's latest diagnostics without retaining unbounded ranks.
            self.diagnostics.update({row["watch_id"]: row for row in result.get("diagnostics", [])})
            while len(self.diagnostics) > 500:
                self.diagnostics.pop(next(iter(self.diagnostics)))
            self._ready = True
            self.status = f"{self.title} 외장 프로세스 연결됨 · 완료 30봉 + 현재가마다 재추론\n{self.bridge.strategy_notice}\n{self.bridge.risk_notice}"
            if result.get("market_errors"):
                self.status += "\n모델 없음: 해당 시장 HOLD · " + str(result["market_errors"])[:500]
        except Exception as exc:
            self.client.close()
            self.bridge.predictors.clear()
            self.bridge._publish_unavailable(chart, exc)
            self.diagnostics, self.status = self.bridge.diagnostics, self.bridge.status

    def validate_execution(self, item, rule, fresh_snapshot, actual_limit_price, *, stage="preflight"):
        if self._closed or not self._ready or not self.client.is_alive:
            raise ValueError("External prototype is unavailable; previous file cannot authorize a buy")
        return self.bridge.validate_execution(item, rule, fresh_snapshot, actual_limit_price, stage=stage)

    def close(self):
        self._closed, self._ready = True, False
        self.client.close()
        self.bridge.predictors.clear()
