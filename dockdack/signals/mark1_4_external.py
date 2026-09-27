"""Mark1.4 DEMO-only, pre-open frozen external signal feed.

The model estimates a next-session open-to-close research score, not a
probability or an intraday barrier. Scoring and selection finish before open;
regular-session polls may only consume that immutable selection. A child
restart, missed preparation, stale quote, changed allocation or disabled
execution policy fails closed. This module never arms or submits an order.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from uuid import NAMESPACE_URL, uuid5

from dockdack.history import market_time
from dockdack.lstm30_adapter import _market_calendar, instrument, number, timestamp, validate_position
from dockdack.market_schedule import EXTRA_CLOSURES, session_on
from dockdack.models import Market, OrderSide
from dockdack.signal_bridge import atomic_json, prototype_record_family
from dockdack.signals.mark1_trigger import Mark1TriggerBridge
from dockdack.signals.prototype_external import ExternalPrototypeFeed, RemoteWorkerError


MODEL_ID = "mark1-4-prototype"
SOURCE_ID = "mark1-4-prototype-demo-trigger"
TITLE = "mark1.4 prototype"
STRATEGY_NOTICE = (
    "국내 E5 / 미국 E1 · 완료 30일봉으로 개장 전 후보 동결 · 장초반 5분만 모의 매수 · "
    "시장별 평가자산 10%씩 최대 10종목 · 체결 확인된 Mark1.4 보유분만 거래일 마감 5분 전 개별 매도 시도"
)
RISK_NOTICE = (
    "국내 압축특징망 / 미국 같은 날 순위망 · 점수는 확률이 아님 · 현재 TOP100은 "
    "훈련 당시 100종목과 다를 수 있음 · 계좌 전체 청산 없음 · 실제 시가 매수/마감 매도 체결은 보장되지 않음"
)
OPEN_BUY_WINDOW = timedelta(minutes=5)
SCORE_WINDOW = timedelta(minutes=10)


class Mark14TriggerBridge(Mark1TriggerBridge):
    """Reuse only DEMO account/position verification from the older bridge."""

    source_id = SOURCE_ID
    title = TITLE
    strategy_id = MODEL_ID
    strategy_notice = STRATEGY_NOTICE
    risk_notice = RISK_NOTICE
    bundle_directory = "models/mark1_4"
    state_filename = "exchange/mark1-4-external-decisions-v1.json"
    producer_type = None  # Price-dependent Mark1 producers are incompatible.

    def validate_execution(self, *args, **kwargs):
        raise ValueError("Mark1.4 requires a frozen pre-open child decision")


def _expected_dates(market: str, session_day: date) -> tuple[str, ...]:
    calendar = _market_calendar(market, session_day.year)
    identity = Market(market)
    dates = [stamp.date() for stamp in calendar.sessions
             if stamp.date() < session_day and (identity, stamp.date()) not in EXTRA_CLOSURES]
    if len(dates) < 30:
        raise ValueError("Mark1.4 exchange calendar has fewer than 30 completed sessions")
    return tuple(day.isoformat() for day in dates[-30:])


def _opening(market: str, value: str) -> datetime:
    opened = timestamp(value, "session_open")
    day = market_time(Market(market), opened).date()
    session = session_on(Market(market), day)
    if session is None or opened != session.opened.astimezone(timezone.utc):
        raise ValueError("Mark1.4 session_open does not match the exchange calendar")
    return opened


def _signal_id(export_id: str, watch_id: str, *, model_id=MODEL_ID, source_id=SOURCE_ID) -> str:
    return model_id + ":" + uuid5(NAMESPACE_URL, f"{source_id}:{export_id}:{watch_id}").hex


def _policy_enabled(engine) -> bool:
    return getattr(engine, "equity_buy_percent", None) == Decimal("10")


def _matches_preopen_bars(plan, key, market, bars):
    """Refuse a revised adjusted history after the decision was frozen."""
    if key not in plan["inputs"] or not isinstance(bars, list):
        return False
    opened = timestamp(plan["session_open"])
    session_day = market_time(Market(market), opened).date()
    completed = []
    try:
        for bar in bars:
            day = date.fromisoformat(bar["date"])
            if day >= session_day or bar.get("is_current_day") is True:
                continue
            values = tuple(float(number(bar[field], field, zero=field == "volume"))
                           for field in ("open", "high", "low", "close", "volume"))
            completed.append((day.isoformat(), values))
    except (KeyError, TypeError, ValueError):
        return False
    expected_dates, expected_values = plan["inputs"][key]
    tail = completed[-30:]
    return (len(tail) == 30 and tuple(day for day, _ in tail) == expected_dates
            and tuple(values for _, values in tail) == expected_values)


class Mark14Worker:
    """Child-side pure inference and session plan; no broker or GUI references."""

    def __init__(self, *, bundle_root, predictors=None, model_id=MODEL_ID,
                 predictor_loader=None):
        self.bundle_root = Path(bundle_root)
        self.predictors = dict(predictors or {})
        self.model_id = model_id
        self.source_id = model_id + "-demo-trigger"
        self.predictor_loader = predictor_loader
        self.plans = {}

    def predictor(self, market):
        if market not in {"domestic", "us"}:
            raise ValueError("Unsupported Mark1.4 market")
        if market not in self.predictors:
            if self.predictor_loader is None:
                from dockdack.mark1_4_inference import Mark14Predictor
                self.predictors[market] = Mark14Predictor(self.bundle_root, market)
            else:
                self.predictors[market] = self.predictor_loader(self.bundle_root, market)
        return self.predictors[market]

    def dispatch(self, request):
        if (not isinstance(request, dict) or request.get("schema_version") != 1
                or request.get("model_id") != self.model_id):
            raise ValueError("Mark1.4 worker identity mismatch")
        operation = request.get("operation")
        if operation == "health":
            return {"model_id": self.model_id, "source_id": self.source_id,
                    "trading_mode": "demo", "pid": os.getpid()}
        if operation == "metadata":
            return self.predictor(request.get("market")).metadata
        if operation == "prepare_preopen":
            return self.prepare_preopen(request)
        if operation == "produce":
            return self.produce(request)
        if operation == "validate_frozen":
            return self.validate_frozen(request)
        raise ValueError("Unsupported Mark1.4 operation")

    def prepare_preopen(self, request):
        started = time.monotonic()
        market = request.get("market")
        if market not in {"domestic", "us"}:
            raise ValueError("Unsupported Mark1.4 market")
        now = timestamp(request.get("now"), "now")
        opened = _opening(market, request.get("session_open"))
        if not opened - SCORE_WINDOW <= now < opened:
            raise ValueError("Mark1.4 preparation must occur in the final 10 pre-open minutes")
        day = market_time(Market(market), opened).date()
        old = self.plans.get(market)
        if old is not None and old["session_open"] == opened.isoformat():
            return {**old["summary"], "state": "prepared", "already_frozen": True}
        candidates = request.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 100:
            raise ValueError("Mark1.4 needs exactly 100 ranked pre-open candidates")
        expected = _expected_dates(market, day)
        symbols, windows, keys = [], [], []
        inputs = {}
        for item in candidates:
            if not isinstance(item, dict):
                raise ValueError("Mark1.4 candidate must be an object")
            key = instrument({**item, "market": market,
                              "currency": "KRW" if market == "domestic" else "USD"})
            dates = item.get("dates")
            if not isinstance(dates, list) or tuple(dates) != expected:
                raise ValueError(f"Mark1.4 requires the exact prior 30 completed sessions: {key}")
            if item.get("last_completed_date") != expected[-1]:
                raise ValueError(f"Mark1.4 last completed bar is stale: {key}")
            if key in keys:
                raise ValueError("Duplicate Mark1.4 candidate")
            keys.append(key)
            symbols.append((item["symbol"], item["exchange"]))
            windows.append(item.get("bars"))
        predictor = self.predictor(market)
        scores = predictor.score_many(windows, symbols)
        if len(scores) != len(keys):
            raise ValueError("Mark1.4 score count does not match ranked universe")
        for key, window in zip(keys, windows):
            inputs[key] = (expected, tuple(tuple(float(value) for value in bar) for bar in window))
        above = [(key, row) for key, row in zip(keys, scores)
                 if row.get("above_frozen_threshold") is True]
        above.sort(key=lambda pair: (-pair[1]["score"], pair[0]))
        selected = {key for key, _ in above[:10]}
        rows = [{"watch_id": key, "symbol": row["symbol"], "exchange": row["exchange"],
                 "score": row["score"], "threshold": row["frozen_numeric_score_threshold"],
                 "score_metric": row["score_metric"], "score_unit": row["score_unit"],
                 **({"equity_fraction": row["equity_fraction"]} if "equity_fraction" in row else {}),
                 "selected": key in selected,
                 "in_training_universe": row.get("in_training_universe"),
                 "out_of_training_universe": row.get("out_of_training_universe")}
                for key, row in zip(keys, scores)]
        elapsed = time.monotonic() - started
        finished = now + timedelta(seconds=max(0, elapsed))
        if finished >= opened:
            raise ValueError("Mark1.4 pre-open selection did not finish before market open")
        material = json.dumps({"market": market, "session_open": opened.isoformat(),
                               "prepared_at": finished.isoformat(), "rows": rows,
                               "inputs": inputs},
                              sort_keys=True, ensure_ascii=False, allow_nan=False)
        digest = sha256(material.encode("utf-8")).hexdigest()
        summary = {"market": market, "state": "prepared", "session_open": opened.isoformat(),
                   "prepared_at": finished.isoformat(), "selected_count": len(selected),
                   "scored_count": len(rows), "candidates": rows,
                   "plan_sha256": digest, "reason": "장전 선택 동결 · 주문은 별도 ON 필요"}
        self.plans[market] = {"session_open": opened.isoformat(), "prepared_at": finished,
                              "selected": selected, "rows": {row["watch_id"]: row for row in rows},
                              "summary": summary, "sha256": digest, "inputs": inputs}
        return summary

    def _active_plan(self, market, now):
        plan = self.plans.get(market)
        if plan is None:
            return None
        try:
            opened = _opening(market, plan["session_open"])
        except ValueError:
            return None
        if not opened <= now < opened + OPEN_BUY_WINDOW:
            return None
        if plan["prepared_at"] >= opened:
            return None
        return plan

    def produce(self, request):
        chart, positions = request.get("chart"), request.get("positions")
        if (not isinstance(chart, dict) or chart.get("schema_version") != 1
                or chart.get("trading_mode") != "demo" or chart.get("source") != "kiwoom_demo"
                or not isinstance(chart.get("stocks"), list) or not isinstance(positions, dict)):
            raise ValueError("Mark1.4 requires a DEMO chart and verified positions")
        now = timestamp(request.get("now"), "now")
        created = timestamp(chart.get("created_at"), "chart created_at")
        if not 0 <= (now - created).total_seconds() < 120:
            raise ValueError("Mark1.4 chart export is stale or future-dated")
        export_id = chart.get("export_id")
        if not isinstance(export_id, str) or not export_id:
            raise ValueError("Mark1.4 chart export identity is missing")
        caps = {"domestic": number(request.get("max_krw"), "KRW cap", zero=True),
                "us": number(request.get("max_usd"), "USD cap", zero=True)}
        signals, diagnostics, seen = [], [], set()
        for stock in chart["stocks"]:
            if not isinstance(stock, dict) or stock.get("status") != "ok":
                continue
            key = instrument(stock)
            if key in seen:
                raise ValueError("Duplicate Mark1.4 chart instrument")
            seen.add(key)
            market = stock["market"]
            plan = self._active_plan(market, now)
            detail = dict(plan["rows"].get(key, {})) if plan else {}
            action, reason = "hold", "NO_PREOPEN_PLAN_OR_OPEN_WINDOW"
            if plan and key in plan["selected"]:
                reason = "PREOPEN_SELECTED"
                try:
                    price = number(stock.get("price"), "current price")
                    quote_time = timestamp(stock.get("quote_fetched_at"), "quote fetched_at")
                    age = number(stock.get("quote_age_seconds"), "quote age", zero=True)
                    if (stock.get("quote_stale") is not False or age > 15
                            or not 0 <= (now - quote_time).total_seconds() <= 15):
                        raise ValueError("Mark1.4 opening quote is stale")
                    # A different strategy may already own this symbol. The
                    # engine reconciles its virtual lots and rejects only an
                    # additional BUY by this very same model.
                    validate_position(positions.get(key), stock, now)
                    if not _matches_preopen_bars(plan, key, market, stock.get("bars")):
                        reason = "CHART_CHANGED_SINCE_PREOPEN"
                    elif caps[market] < price:
                        reason = "MARKET_NOTIONAL_CAP"
                    elif chart.get("adjusted_prices") is not True:
                        reason = "UNADJUSTED_CHART"
                    else:
                        action = "buy"
                except Exception as exc:
                    reason = "QUOTE_OR_POSITION_UNAVAILABLE"
                    detail["error"] = str(exc)[:300]
            elif plan:
                reason = "BELOW_THRESHOLD_OR_OUTSIDE_TOP10" if key in plan["rows"] else "NOT_IN_FROZEN_PREOPEN_UNIVERSE"
            signal = {"signal_id": _signal_id(export_id, key, model_id=self.model_id,
                                               source_id=self.source_id), "export_id": export_id,
                      "market": market, "symbol": stock["symbol"], "exchange": stock["exchange"],
                      "action": action, "generated_at": created.isoformat(),
                      "expires_at": (created + timedelta(seconds=120)).isoformat()}
            if action == "buy":
                signal.update(quantity=1, max_notional=str(caps[market]),
                              strategy_id=self.model_id,
                              model_title=self.model_id.replace("mark1-", "mark1.").replace("-prototype", " prototype"),
                              model_version=self.model_id.replace("mark1-", "1.").replace("-prototype", ""),
                              model_manifest_sha256=self.predictor(market).metadata["bundle_manifest_sha256"])
                if "equity_fraction" in detail:
                    signal["target_equity_fraction"] = format(
                        Decimal(str(detail["equity_fraction"])).quantize(Decimal("0.00000001")), "f")
            signals.append(signal)
            diagnostics.append({"watch_id": key, "reason": reason, "emitted": True,
                                "plan_sha256": plan["sha256"] if plan else None,
                                **detail})
        return {"payload": {"schema_version": 1, "source_id": self.source_id,
                            "trading_mode": "demo", "signals": signals},
                "diagnostics": diagnostics,
                "metadata": {market: predictor.metadata for market, predictor in self.predictors.items()}}

    def validate_frozen(self, request):
        now = timestamp(request.get("now"), "now")
        market = request.get("market")
        key = request.get("watch_id")
        if market not in {"domestic", "us"} or not isinstance(key, str):
            raise ValueError("Invalid Mark1.4 order identity")
        plan = self._active_plan(market, now)
        if plan is None or key not in plan["selected"]:
            raise ValueError("Mark1.4 frozen pre-open BUY is unavailable")
        if request.get("plan_sha256") != plan["sha256"]:
            raise ValueError("Mark1.4 pre-open plan identity changed")
        if request.get("signal_id") != _signal_id(request.get("export_id"), key,
                                                    model_id=self.model_id, source_id=self.source_id):
            raise ValueError("Mark1.4 signal identity changed")
        if not _matches_preopen_bars(plan, key, market, request.get("bars")):
            raise ValueError("Mark1.4 daily chart changed after pre-open selection")
        return {"selected": True, "plan_sha256": plan["sha256"],
                "session_open": plan["session_open"]}


class Mark14ExternalFeed(ExternalPrototypeFeed):
    """Main-GUI adapter; only the trusted engine may turn a signal into an order."""

    expected_model_id = MODEL_ID
    strategy_notice = STRATEGY_NOTICE
    risk_notice = RISK_NOTICE

    def __init__(self, window, model_id, policy, output_path, *, bundle_root=None,
                 account_snapshots=None, client_factory=None):
        if model_id != self.expected_model_id:
            raise ValueError("Mark1.4 feed identity mismatch")
        super().__init__(window, model_id, policy, output_path,
                         bundle_root=bundle_root, account_snapshots=account_snapshots,
                         client_factory=client_factory)
        self.plans = {}
        self.status = (f"{self.title} · 장전 10분에 TOP100 완료 30봉 준비 대기\n"
                       f"{self.strategy_notice}\n{self.risk_notice}")

    def _require_policy(self):
        self.bridge._ensure_demo()
        if not _policy_enabled(self.window.engine):
            raise ValueError("Mark1.4는 평가자산 10% 모의 매수 설정이 필요합니다.")

    def prepare_preopen(self, market, candidates, *, now, session):
        """Score a whole ranked market once during the final pre-open ten minutes.

        The GUI's broker worker must have verified ranking membership and 30
        completed dates before this call. The child independently rechecks the
        exchange dates and freezes its result in memory until exit/restart.
        """
        if self._closed or self.window.engine._stop.is_set():
            raise ValueError("Mark1.4 feed is stopped")
        self._require_policy()
        opened = session.opened if hasattr(session, "opened") else session
        if isinstance(opened, datetime):
            opened = opened.isoformat()
        if not isinstance(opened, str):
            raise ValueError("Mark1.4 session opening time is missing")
        if not self.client.is_alive and self.plans:
            # A crashed/restarted child cannot retain the supposedly frozen plan.
            self.plans.clear()
            self._ready = False
        if (self.client.is_alive and market in self.plans
                and self.plans[market].get("session_open") == timestamp(opened).isoformat()):
            # A frozen choice cannot be silently replaced by a later ranking.
            return {**self.plans[market], "state": "prepared", "already_frozen": True}
        try:
            result = self.client.request("prepare_preopen", market=market, candidates=candidates,
                                         now=now.isoformat(), session_open=opened,
                                         preserve_remote_error=True)
            if (not isinstance(result, dict) or result.get("market") != market
                    or result.get("state") != "prepared" or result.get("session_open") != timestamp(opened).isoformat()
                    or not isinstance(result.get("candidates"), list)
                    or not isinstance(result.get("plan_sha256"), str)):
                raise ValueError("Mark1.4 pre-open worker returned an invalid plan")
            self.plans[market] = result
            self.diagnostics.update({row["watch_id"]: row for row in result["candidates"]})
            self.status = (f"{self.title} · {market} 장전 판단 완료 · {result['selected_count']}/{result['scored_count']}종목 선택"
                           f"\n{self.strategy_notice}\n{self.risk_notice}")
            return result
        except Exception as exc:
            if not self.client.is_alive:
                self.plans.clear()
                self._ready = False
            self.status = f"{self.title} · {market} 장전 준비 실패 · 당일 매수 HOLD: {exc}"
            raise

    def publish(self, chart):
        if self._closed or self.window.engine._stop.is_set():
            return
        self._ready = False
        try:
            self._require_policy()
            if (not isinstance(chart, dict) or chart.get("trading_mode") != "demo"
                    or chart.get("source") != "kiwoom_demo"):
                raise ValueError("Mark1.4 requires explicit DEMO charts")
            positions = {}
            for stock in chart.get("stocks", []):
                if not isinstance(stock, dict) or stock.get("status") != "ok":
                    continue
                key = instrument(stock)
                try:
                    positions[key] = self._position(stock)
                except Exception:
                    positions[key] = None  # Unknown balance is never flat.
            result = self.client.request("produce", chart=chart, positions=positions,
                                         now=self.window.engine.clock().isoformat(),
                                         max_krw=str(self.policy.max_krw),
                                         max_usd=str(self.policy.max_usd),
                                         preserve_remote_error=True)
            payload = result["payload"]
            if (payload.get("source_id") != self.source_id or payload.get("trading_mode") != "demo"
                    or not isinstance(payload.get("signals"), list)):
                raise ValueError("Mark1.4 returned a different source or trading mode")
            for signal in payload["signals"]:
                if (not str(signal.get("signal_id", "")).startswith(self.model_id + ":")
                        or signal.get("action") not in {"buy", "hold"}):
                    raise ValueError("Mark1.4 signal identity or direction mismatch")
            atomic_json(self.output_path, payload)
            self.diagnostics.update({row["watch_id"]: row for row in result.get("diagnostics", [])})
            while len(self.diagnostics) > 500:
                self.diagnostics.pop(next(iter(self.diagnostics)))
            self._ready = True
            active = sum(1 for signal in payload["signals"] if signal["action"] == "buy")
            self.status = (f"{self.title} · 장전 고정 판단 연결됨 · 이번 조회 매수 후보 {active}건"
                           f"\n{self.strategy_notice}\n{self.risk_notice}")
        except Exception as exc:
            if not isinstance(exc, RemoteWorkerError) or not self.client.is_alive:
                self.client.close()
                self.plans.clear()
            self.bridge._publish_unavailable(chart, exc)
            self.diagnostics, self.status = self.bridge.diagnostics, self.bridge.status

    def validate_execution(self, item, rule, fresh_snapshot, actual_limit_price, *, stage="preflight"):
        """Fail-closed preflight and paced-send check against the same child plan."""
        self._require_policy()
        if stage not in {"preflight", "final_send"}:
            raise ValueError("Unknown Mark1.4 validation stage")
        if rule.side is not OrderSide.BUY:
            raise ValueError("Mark1.4 external feed only generates BUY; closing policy owns SELL")
        if self._closed or not self._ready or not self.client.is_alive:
            raise ValueError("Mark1.4 pre-open decision process is unavailable")
        record = self.window.store.external_for_rule(rule.id)
        if record is None or record.get("source_id") != self.source_id:
            raise ValueError("Mark1.4 persisted BUY source is missing")
        family = prototype_record_family(record, watch_id=item.id, action="buy")
        if family is None or family.id != self.model_id:
            raise ValueError("Mark1.4 persisted BUY family does not match")
        now = self.window.engine.clock()
        market = item.instrument.market.value
        local_plan = self.plans.get(market)
        if local_plan is None:
            raise ValueError("Mark1.4 has no local pre-open plan")
        if self.model_id == "mark1-8-prototype":
            selected_row = next((row for row in local_plan.get("candidates", ())
                                 if row.get("watch_id") == item.id and row.get("selected") is True), None)
            if selected_row is None or "equity_fraction" not in selected_row:
                raise ValueError("Mark1.8 frozen model allocation is missing")
            payload = json.loads(record["payload"])
            expected_fraction = Decimal(str(selected_row["equity_fraction"])).quantize(Decimal("0.00000001"))
            if (payload.get("strategy_id") != self.model_id
                    or Decimal(str(payload.get("target_equity_fraction"))) != expected_fraction
                    or not Decimal(0) < expected_fraction <= Decimal("0.1")):
                raise ValueError("Mark1.8 signal allocation changed after pre-open scoring")
        opened = _opening(market, local_plan["session_open"])
        if not opened <= now < opened + OPEN_BUY_WINDOW:
            raise ValueError("Mark1.4 opening order window has closed")
        self.window.engine._validate_snapshot(item, fresh_snapshot)
        if not 0 <= (now - fresh_snapshot.fetched_at).total_seconds() <= 15:
            raise ValueError("Mark1.4 fresh quote is stale or future-dated")
        quote = number(fresh_snapshot.quote.price, "fresh quote")
        limit = number(actual_limit_price, "actual limit price")
        if abs(limit / quote - Decimal(1)) > Decimal("0.01"):
            raise ValueError("Mark1.4 actual limit price differs >1% from fresh quote")
        result = self.client.request("validate_frozen", start=False, market=market,
                                     watch_id=item.id, now=now.isoformat(),
                                     plan_sha256=local_plan["plan_sha256"],
                                     signal_id=record["signal_id"], export_id=record["export_id"],
                                     bars=[{"date": bar.day.isoformat(),
                                            **{field: str(getattr(bar, field))
                                               for field in ("open", "high", "low", "close", "volume")}}
                                           for bar in fresh_snapshot.history.bars])
        if (result.get("selected") is not True
                or result.get("plan_sha256") != local_plan["plan_sha256"]
                or result.get("session_open") != local_plan["session_open"]):
            raise ValueError("Mark1.4 child decision no longer matches frozen plan")
        self._require_policy()
        if not 0 <= (self.window.engine.clock() - fresh_snapshot.fetched_at).total_seconds() <= 15:
            raise ValueError("Mark1.4 quote expired during final validation")

    def close(self):
        super().close()
        self.plans.clear()
