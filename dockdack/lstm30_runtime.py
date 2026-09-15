"""A single-writer, explicitly armed, DEMO-only headless LSTM30 session.

This runtime reuses AutoTrader's durable claims, account/quote preflight and
final send guard. It never enables REAL mode or rearms after a safety stop.
The dedicated directory is intentionally separate from the desktop GUI ledger.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from decimal import Decimal
import json
import os
from pathlib import Path
import signal
from threading import Event, Thread
import time
from uuid import uuid4

from dockdack.autotrade import AutoTrader
from dockdack.gui_service import Instrument, TradingService
from dockdack.history import market_time, regular_session
from dockdack.lstm30_adapter import LSTM30SignalProducer, SOURCE_ID, demo_position_provider
from dockdack.models import Market, OrderSide, TradingMode
from dockdack.signal_bridge import ExternalPolicy, SignalFileReader, atomic_json, export_charts
from dockdack.watchlist import WatchItem, WatchStore, utc_now


DEFAULT_RUNTIME_DIR = Path(".dockdack/lstm30-demo")
DEFAULT_ITEMS = (
    WatchItem(Instrument(Market.DOMESTIC, "005930", "KRX"), "Samsung Electronics", 31),
    WatchItem(Instrument(Market.US, "AAPL", "ND"), "Apple", 31),
)
_BAD_DIAGNOSTICS = frozenset({
    "UNREGISTERED_INVALID_CHART", "QUOTE_OR_POSITION_UNAVAILABLE",
    "PREDICTION_UNAVAILABLE", "QUOTE_EXPIRED_DURING_INFERENCE",
    "POSITION_EXPIRED_DURING_INFERENCE",
})


class SessionLock:
    """Advisory OS lock released even after process termination; never unlink it."""

    def __init__(self, path):
        self.path, self.stream = Path(path), None

    def acquire(self):
        if self.stream is not None:
            raise RuntimeError("Session lock is already held")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        try:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise RuntimeError("Another LSTM demo session owns this runtime directory") from exc
        self.stream = stream

    def release(self):
        if self.stream is not None:
            # Closing the descriptor releases the operating-system lock.
            self.stream.close()
            self.stream = None


def request_stop(runtime_dir=DEFAULT_RUNTIME_DIR):
    """Ask exactly the published session to stop; never kill or cancel orders."""
    root = Path(runtime_dir).resolve()
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    session = status.get("session_id")
    if not isinstance(session, str) or len(session) != 32 or any(c not in "0123456789abcdef" for c in session):
        raise ValueError("No valid running session identity was found")
    atomic_json(root / "stop.json", {"session_id": session, "requested_at": utc_now().isoformat()})
    return session


class DemoLSTMRuntime:
    """Inject service, predictors, positions and clock for order-free fake tests.

    ``start()`` warms every instrument with orders OFF. Only an explicit
    ``DEMO_AUTOTRADE`` confirmation can arm, once, after a successful warmup.
    Warmup failures leave a monitoring-only session; they do not auto-rearm.
    Call ``close()`` in a finally block, or use ``run_forever()``.
    """

    def __init__(self, runtime_dir=DEFAULT_RUNTIME_DIR, *, predictors, quantity,
                 max_krw, max_usd, service=None, items=None, position_provider=None,
                 interval_seconds=30, clock=utc_now, checkpoint_paths=None):
        if type(interval_seconds) is not int or not 15 <= interval_seconds <= 3600:
            raise ValueError("interval_seconds must be an integer from 15 to 3600")
        self.service = service if service is not None else TradingService(mode=TradingMode.DEMO)
        if getattr(self.service, "mode", None) is not TradingMode.DEMO:
            raise ValueError("The LSTM runtime accepts only an explicitly DEMO service")
        self.policy = ExternalPolicy(SOURCE_ID, quantity, Decimal(str(max_krw)), Decimal(str(max_usd)))
        self.items = tuple(DEFAULT_ITEMS if items is None else items)
        if not self.items or len({item.id for item in self.items}) != len(self.items):
            raise ValueError("Select at least one instrument without duplicates")
        if any(item.days < 31 for item in self.items):
            raise ValueError("Request at least 31 daily bars to retain 30 completed bars")
        self.predictors = dict(predictors)
        for market in {item.instrument.market.value for item in self.items}:
            predictor = self.predictors.get(market)
            if predictor is None or getattr(predictor, "metadata", {}).get("market") != market:
                raise ValueError(f"A matching {market} checkpoint is required at startup")
        self.root, self.clock = Path(runtime_dir).resolve(), clock
        self.interval_seconds = interval_seconds
        self.checkpoint_paths = {key: str(Path(value).resolve()) for key, value in (checkpoint_paths or {}).items()}
        self.position_provider = position_provider or self._demo_position
        self.session_id, self.started_at = uuid4().hex, self.clock()
        self.lock = SessionLock(self.root / "session.lock")
        self._stop_event = Event()
        self._stop_watcher = None
        self._started = False
        self._closed = False
        self._initial_attempts = set()
        self._armed_once = False
        self.engine = self.store = self.producer = None
        self.phase, self.cycles = "created", 0
        self.results, self.diagnostics, self.errors = {}, {}, []
        self.safety_latch = []

    def _demo_position(self, stock):
        market = Market(stock["market"])
        broker = self.service.broker(market)
        return demo_position_provider({market.value: broker}, clock=self.clock)(stock)

    def _check_demo(self):
        if getattr(self.service, "mode", None) is not TradingMode.DEMO:
            if self.engine is not None:
                self.engine.disarm()
            raise ValueError("The runtime cannot switch out of DEMO mode")
        if self.store is not None and self.store.mode is not TradingMode.DEMO:
            self.engine.disarm()
            raise ValueError("The runtime ledger must remain DEMO")
        if self.store is not None and {item.id for item in self.store.items()} != {item.id for item in self.items}:
            self.engine.disarm()
            raise ValueError("The dedicated watchlist changed during this session")
        for item in self.items:
            self.service.ensure_demo(item.instrument)

    def start(self, confirmation=None):
        if confirmation not in (None, "DEMO_AUTOTRADE"):
            raise ValueError("Only the exact DEMO_AUTOTRADE confirmation is accepted")
        if self._started or self._closed:
            raise RuntimeError("A runtime instance can only be started once")
        self.lock.acquire()
        try:
            self.store = WatchStore(self.root / "watchlist.sqlite3", mode=TradingMode.DEMO)
            existing = self.store.items()
            if existing and {item.id for item in existing} != {item.id for item in self.items}:
                raise ValueError("Existing runtime watchlist differs; choose another dedicated runtime directory")
            for item in self.items:
                self.store.save_item(item)
            self.engine = AutoTrader(self.service, self.store, clock=self.clock)
            self.engine.external_only = True
            self.engine.external_policy = self.policy
            self.engine.external_reader = SignalFileReader(
                self.store, self.root / "exchange/signals.json", self.policy, clock=self.clock)
            self.producer = LSTM30SignalProducer(
                self.predictors, position_provider=self.position_provider,
                quantity=self.policy.max_quantity, max_krw=self.policy.max_krw,
                max_usd=self.policy.max_usd, state_path=self.root / "exchange/decisions.json",
                clock=self.clock)
            self._initial_attempts = {row["rule_id"] for row in self.store.attempts()}
            # Old READY signals are never inherited as new-session authority.
            for rule in self.store.rules():
                if rule.status == "ready":
                    self.store.pause_rule(rule.id)
            atomic_json(self.root / "exchange/signals.json", {
                "schema_version": 1, "source_id": SOURCE_ID, "trading_mode": "demo", "signals": []})
            self._started, self.phase = True, "warming_up"
            self._write_status()
            self._check_demo()
            self._stop_watcher = Thread(target=self._watch_stop, name="lstm30-demo-stop", daemon=True)
            self._stop_watcher.start()
            external_errors_before = self.engine.external_error_count
            self.poll_once()
            unsafe = [dict(watch_id=key, reason=value["reason"])
                      for key, value in self.diagnostics.items() if value["reason"] in _BAD_DIAGNOSTICS]
            missing = set(item.id for item in self.items) - set(self.diagnostics)
            self.errors.extend(unsafe)
            self.errors.extend({"watch_id": key, "reason": "WARMUP_INCOMPLETE"} for key in sorted(missing))
            if self.engine.external_error_count != external_errors_before:
                self.errors.append({"watch_id": "SYSTEM", "reason": "EXTERNAL_SIGNAL_ERROR_DURING_WARMUP"})
            if self._stop_event.is_set():
                self.phase = "stopped"
            elif self.errors or self.engine.external_error:
                self.engine.disarm()
                self.phase = "blocked"
            elif confirmation == "DEMO_AUTOTRADE":
                try:
                    self.engine.enable_orders(confirmation)
                except Exception as exc:
                    self.errors.append({"watch_id": "SYSTEM", "reason": f"ARM_FAILED:{type(exc).__name__}"})
                    self.phase = "blocked"
                else:
                    self._armed_once = True
                    self.phase = "running"
            else:
                self.phase = "monitoring"
            if self.phase == "blocked":
                self.safety_latch = list(self.errors) or [{"watch_id": "SYSTEM", "reason": "EXTERNAL_SIGNAL_ERROR"}]
            return self._write_status()
        except Exception:
            self.close()
            raise

    def _buy_attempted_today(self, item):
        today = market_time(item.instrument.market, self.clock()).date()
        with self.store.connection() as db:
            attempts = db.execute("""SELECT a.started_at FROM attempts a JOIN rules r ON r.id=a.rule_id
                                     WHERE a.watch_id=? AND r.side=?""", (item.id, OrderSide.BUY.value)).fetchall()
        return any(market_time(item.instrument.market, datetime.fromisoformat(row["started_at"])).date() == today
                   for row in attempts)

    def _on_snapshot(self, item, snapshot):
        self._check_stop()
        if self._stop_event.is_set():
            return
        if item.id not in {selected.id for selected in self.items}:
            raise ValueError("The snapshot is outside the explicitly selected watchlist")
        charts = export_charts(self.store, self.root / "exchange/charts.json",
                               now=self.clock(), watch_ids={item.id})
        payload, diagnostics = self.producer(charts)
        atomic_json(self.root / "exchange/signals.json", payload)
        # Ingest before applying the runtime's one BUY attempt per local day gate.
        # The model's immutable wire decision is never rewritten by this gate.
        self.engine.external_reader()
        actions = {f"{row['market']}:{row['exchange']}:{row['symbol']}": row["action"]
                   for row in payload["signals"]}
        for row in diagnostics:
            safe = {key: value for key, value in row.items() if key not in {"error", "watch_id"}}
            safe["action"] = actions.get(row["watch_id"], "none")
            self.diagnostics[row["watch_id"]] = safe
        if self._buy_attempted_today(item):
            for rule in self.store.rules(item.id):
                if rule.status == "ready" and rule.side is OrderSide.BUY:
                    self.store.pause_rule(rule.id)
            if item.id in self.diagnostics:
                self.diagnostics[item.id]["execution_gate"] = "DAILY_BUY_ATTEMPT_LIMIT"
        atomic_json(self.root / "exchange/diagnostics.json", {
            "session_id": self.session_id, "updated_at": self.clock().isoformat(),
            "diagnostics": self.diagnostics})

    def _new_critical_attempts(self):
        return [row for row in self.store.attempts()
                if row["rule_id"] not in self._initial_attempts
                and row["status"] in {"rejected", "unknown", "submitting"}]

    def _progress(self, progress):
        if isinstance(progress[1], Exception) or self._new_critical_attempts():
            self.engine.disarm()
        self._check_stop()

    def poll_once(self):
        if not self._started or self._closed:
            raise RuntimeError("Start the session before polling")
        self._check_stop()
        if self._stop_event.is_set():
            self.phase = "stopped"
            return self._write_status()
        self.errors = []
        self.diagnostics = {}
        try:
            self._check_demo()
            raw = self.engine.poll(on_snapshot=self._on_snapshot, progress=self._progress)
            self.results = {}
            for key, value in raw.items():
                if isinstance(value, Exception):
                    reason = f"SNAPSHOT_OR_RECONCILIATION_FAILED:{type(value).__name__}"
                    self.results[key] = {"ok": False, "reason": reason}
                    self.errors.append({"watch_id": key, "reason": reason})
                    self.engine.disarm()
                else:
                    self.results[key] = {"ok": True, "price": str(value.quote.price),
                                         "currency": value.quote.currency, "fetched_at": value.fetched_at.isoformat()}
        except Exception as exc:
            self.engine.disarm()
            self.errors.append({"watch_id": "SYSTEM", "reason": f"POLL_FAILED:{type(exc).__name__}"})
        if self._new_critical_attempts():
            self.engine.disarm()
            self.errors.append({"watch_id": "SYSTEM", "reason": "ORDER_REJECTED_OR_OUTCOME_UNKNOWN"})
        self.cycles += 1
        if self._stop_event.is_set():
            self.phase = "stopped"
        elif self._armed_once and not self.engine.orders_enabled:
            self.phase = "disarmed"
            if not self.safety_latch:
                self.safety_latch = list(self.errors) or [{"watch_id": "SYSTEM", "reason": "ENGINE_SAFETY_DISARM"}]
        return self._write_status()

    def status(self):
        counts = Counter()
        orders = []
        thresholds, checkpoint_thresholds = {}, {}
        for market, predictor in self.predictors.items():
            metadata = getattr(predictor, "metadata", {})
            original = metadata.get("buy_threshold") if isinstance(metadata, dict) else None
            effective = getattr(predictor, "buy_threshold", original)
            checkpoint_thresholds[market] = (float(original) if type(original) in (int, float)
                                             and 0 < original < 1 else None)
            thresholds[market] = (float(effective) if type(effective) in (int, float)
                                  and 0 < effective < 1 else checkpoint_thresholds[market])
        if self.store is not None:
            for row in self.store.attempts():
                if row["rule_id"] not in self._initial_attempts:
                    counts[row["status"]] += 1
                    orders.append({key: row[key] for key in ("watch_id", "status", "order_number", "started_at")})
        markets = {item.instrument.market for item in self.items}
        return {
            "schema_version": 1, "trading_mode": "demo", "source_id": SOURCE_ID,
            "pid": os.getpid(), "session_id": self.session_id, "phase": self.phase,
            "started_at": self.started_at.isoformat(), "heartbeat": self.clock().isoformat(),
            "orders_enabled": bool(self.engine and self.engine.orders_enabled), "cycles": self.cycles,
            "interval_seconds": self.interval_seconds,
            "watchlist": [{"watch_id": item.id, "days": item.days} for item in self.items],
            "limits": {"quantity": self.policy.max_quantity, "max_krw": str(self.policy.max_krw),
                       "max_usd": str(self.policy.max_usd), "buy_attempts_per_symbol_local_day": 1},
            "regular_session": {market.value: regular_session(market, self.clock()) for market in markets},
            "checkpoints": self.checkpoint_paths, "results": self.results,
            "buy_thresholds": thresholds, "checkpoint_buy_thresholds": checkpoint_thresholds,
            "diagnostics": self.diagnostics, "errors": self.errors,
            "safety_latch": self.safety_latch,
            "session_order_counts": dict(counts), "session_orders": orders[-100:],
            "store_path": str(self.root / "watchlist.sqlite3"),
            "signals_path": str(self.root / "exchange/signals.json"),
        }

    def _write_status(self):
        state = self.status()
        atomic_json(self.root / "status.json", state)
        return state

    def _check_stop(self):
        marker = self.root / "stop.json"
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            # Corrupt stop control cannot leave order permission silently ON.
            self.stop()
            return
        if not isinstance(payload, dict):
            self.stop()
        elif payload.get("session_id") == self.session_id:
            self.stop()

    def _watch_stop(self):
        while not self._stop_event.wait(0.5):
            self._check_stop()

    def stop(self):
        self._stop_event.set()
        if self.engine is not None:
            self.engine.stop()

    def close(self):
        if self._closed:
            return
        self.stop()
        if self._stop_watcher is not None and self._stop_watcher.is_alive():
            self._stop_watcher.join(timeout=1)
        try:
            if self._started:
                self.phase = "stopped"
                self._write_status()
        finally:
            self._closed = True
            self.lock.release()

    def run_forever(self, confirmation=None):
        try:
            self.start(confirmation)
            while not self._stop_event.is_set():
                deadline = time.monotonic() + self.interval_seconds
                last_heartbeat = time.monotonic()
                while not self._stop_event.wait(min(1, max(0, deadline - time.monotonic()))):
                    self._check_stop()
                    if time.monotonic() - last_heartbeat >= 5:
                        self._write_status()
                        last_heartbeat = time.monotonic()
                    if time.monotonic() >= deadline:
                        break
                if not self._stop_event.is_set():
                    self.poll_once()
        finally:
            self.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--stop", action="store_true", help="Gracefully stop the current session; never cancel accepted orders")
    parser.add_argument("--once", action="store_true", help="Perform one orders-OFF warmup and exit (default without --arm)")
    parser.add_argument("--arm", choices=["DEMO_AUTOTRADE"], help="Explicitly enable continuous DEMO orders after valid warmup")
    parser.add_argument("--quantity", type=int)
    parser.add_argument("--max-krw")
    parser.add_argument("--max-usd")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--domestic-checkpoint", type=Path)
    parser.add_argument("--us-checkpoint", type=Path)
    parser.add_argument("--buy-threshold", type=float, default=0.4,
                        help="Inclusive runtime BUY probability threshold (default 0.4); does not retrain or modify weights")
    parser.add_argument("--symbol", action="append", help="Repeat MARKET:EXCHANGE:SYMBOL, e.g. domestic:KRX:005930 or us:ND:AAPL")
    args = parser.parse_args(argv)
    if args.stop:
        session = request_stop(args.runtime_dir)
        print(f"DEMO stop requested for session {session}; accepted orders are not cancelled.")
        return 0
    if args.once and args.arm:
        parser.error("--once is orders-OFF only and cannot be combined with --arm")
    if not 0 < args.buy_threshold < 1:
        parser.error("--buy-threshold must be a finite probability strictly between 0 and 1")
    if any(value is None for value in (args.quantity, args.max_krw, args.max_usd)):
        parser.error("Provide explicit --quantity, --max-krw and --max-usd limits")
    paths = {market: path for market, path in (("domestic", args.domestic_checkpoint), ("us", args.us_checkpoint)) if path}
    items = None
    if args.symbol:
        try:
            items = [WatchItem(Instrument(Market(market), symbol, exchange), days=31)
                     for market, exchange, symbol in (text.split(":") for text in args.symbol)]
        except (ValueError, TypeError) as exc:
            parser.error(f"Invalid --symbol selection: {exc}")
    from dockdack.ml30 import Predictor
    predictors = {market: Predictor(path, device="cpu", buy_threshold=args.buy_threshold) for market, path in paths.items()}
    runtime = DemoLSTMRuntime(
        args.runtime_dir, predictors=predictors, quantity=args.quantity, max_krw=args.max_krw,
        max_usd=args.max_usd, items=items, interval_seconds=args.interval, checkpoint_paths=paths)
    previous = {}
    try:
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is not None:
                previous[sig] = signal.signal(sig, lambda *_: runtime.stop())
        if not args.arm:
            state = runtime.start()
            print(json.dumps(state, ensure_ascii=False, indent=2))
            return 1 if state["phase"] == "blocked" else 0
        print(f"DEMO-only session {runtime.session_id}; status: {runtime.root / 'status.json'}", flush=True)
        runtime.run_forever(args.arm)
        return 0
    finally:
        runtime.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
