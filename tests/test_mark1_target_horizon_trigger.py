"""Offline worker/bridge checks; no broker, orders, or operating ledger."""
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dockdack.application.trading_service import Instrument
from dockdack.history import DailyBar, DailyHistory
from dockdack.market_schedule import calendar_for
from dockdack.models import Market, OrderSide, Quote, TradingMode
from dockdack.persistence.watchlist import MarketSnapshot, TriggerKind, TriggerRule, WatchItem
from dockdack.prototype_external import MODEL_IDS, PrototypeWorker, model_bridge_type
from dockdack.signals.mark1_target_horizon_trigger import MODEL_IDS as HORIZON_IDS, MODEL_SPECS
from dockdack.trading.autotrade import AutoTrader
from dockdack.trading.model_exit_schedule import ModelExitSchedule, model_exit_schedule


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 9, 15, 1, 0, tzinfo=timezone.utc)
DAYS = [stamp.date() for stamp in calendar_for(Market.DOMESTIC, 2026).sessions
        if stamp.date() < date(2026, 9, 15)][-30:]


def stock():
    return {
        "watch_id": "domestic:KRX:005930", "market": "domestic", "exchange": "KRX",
        "symbol": "005930", "currency": "KRW", "status": "ok", "complete": True,
        "requested_days": 30, "available_days": 30, "price": "100",
        "quote_fetched_at": NOW.isoformat(), "quote_age_seconds": 0,
        "quote_stale": False,
        "bars": [{"date": day.isoformat(), "open": "100", "high": "103",
                  "low": "99", "close": "101", "volume": "1000",
                  "is_current_day": False} for day in DAYS],
    }


def chart(export_id="horizon-test"):
    return {"schema_version": 1, "export_id": export_id, "trading_mode": "demo",
            "source": "kiwoom_demo", "created_at": NOW.isoformat(),
            "adjusted_prices": True, "stocks": [stock()]}


def position(quantity="0"):
    return {"market": "domestic", "exchange": "KRX", "symbol": "005930",
            "currency": "KRW", "quantity": quantity, "sellable_quantity": quantity,
            "average_price": "100" if quantity != "0" else None,
            "fetched_at": NOW.isoformat()}


class CandidatePredictor:
    def __init__(self, model_id="mark1-23-prototype", *, reject_price=None):
        lookback, horizon, target_pct = MODEL_SPECS[model_id]
        self.model_id = model_id
        self.reject_price = reject_price
        self.queries = []
        self.metadata = {
            "strategy_id": model_id, "market": "domestic", "lookback": lookback,
            "horizon_sessions": horizon, "take_profit_pct": target_pct,
            "bundle_manifest_sha256": "a" * 64,
            "research_only": True, "research_qualified": False,
            "deployment_allowed": False, "intraday_path_verified": False,
        }

    def predict(self, bars, current_price=None):
        self.queries.append(Decimal(str(current_price)))
        price = float(current_price)
        _, horizon, target_pct = MODEL_SPECS[self.model_id]
        candidate = self.reject_price != Decimal(str(current_price))
        return {
            "title": self.model_id.replace("mark1-", "mark1.").replace("-prototype", " prototype"),
            "strategy_id": self.model_id, "market": "domestic",
            "probability_success": .6 if candidate else .4,
            "probability_stop": None, "expected_net_return": .02 if candidate else -.01,
            "predicts_success": candidate, "selected_research": candidate,
            "buy_threshold": .5, "policy_threshold": .5,
            "candidate_entry_price": price,
            "candidate_take_price": price * (1 + target_pct / 100),
            "candidate_stop_price": None, "take_profit_pct": target_pct,
            "stop_loss_pct": None, "horizon_sessions": horizon,
            "target": "daily_first_high_target_touch_else_Hth_close_no_stop_proxy",
            "score_scope": "next_open_proxy_not_intraday_verified",
            "bundle_manifest_sha256": "a" * 64,
            "research_only": True, "research_qualified": False,
            "deployment_allowed": False, "intraday_path_verified": False,
        }


class TargetHorizonTriggerTest(unittest.TestCase):
    def test_each_horizon_has_its_own_final_session_exit(self):
        for model_id, (_, horizon, _) in MODEL_SPECS.items():
            self.assertEqual(model_exit_schedule(model_id),
                             ModelExitSchedule(horizon - 1, "preclose"))

    def test_six_worker_identities_and_saved_predictors(self):
        self.assertEqual(HORIZON_IDS, tuple(f"mark1-{n}-prototype" for n in range(23, 29)))
        self.assertTrue(set(HORIZON_IDS).issubset(MODEL_IDS))
        for model_id in HORIZON_IDS:
            bridge = model_bridge_type(model_id)
            self.assertEqual(bridge.strategy_id, model_id)
            self.assertEqual(bridge.source_id, model_id + "-demo-trigger")
            worker = PrototypeWorker(model_id, bundle_root=ROOT / "models" / "mark1_target_horizon_v1")
            self.assertIsNone(worker.special)
            result = worker.dispatch({"schema_version": 1, "model_id": model_id,
                                      "operation": "predict", "market": "domestic",
                                      "bars": [[100., 103., 99., 101., 1000.]] * 30,
                                      "current_price": "100"})
            self.assertEqual(result["strategy_id"], model_id)
            self.assertIsNone(result["stop_loss_pct"])
            self.assertFalse(result["deployment_allowed"])
            with self.assertRaises(ValueError):
                worker.dispatch({"schema_version": 1, "model_id": model_id,
                                 "operation": "place_order"})

    def test_buy_requires_matching_exit_schedule_and_has_no_stop_bracket(self):
        model_id = "mark1-23-prototype"
        predictor = CandidatePredictor(model_id)
        request = {"schema_version": 1, "model_id": model_id, "operation": "produce",
                   "chart": chart(), "positions": {stock()["watch_id"]: position()},
                   "now": NOW.isoformat(), "max_krw": "10000", "max_usd": "1000"}
        worker = PrototypeWorker(model_id, predictors={"domestic": predictor})
        with patch("dockdack.trading.model_exit_schedule.model_exit_schedule", return_value=None):
            unscheduled = worker.dispatch(request)
        self.assertEqual(unscheduled["payload"]["signals"][0]["action"], "hold")
        self.assertEqual(unscheduled["diagnostics"][0]["reason"], "EXIT_SCHEDULE_UNAVAILABLE")
        with patch("dockdack.trading.model_exit_schedule.model_exit_schedule",
                   return_value=ModelExitSchedule(8, "preclose")):
            wrong = PrototypeWorker(model_id, predictors={"domestic": predictor}).dispatch(request)
        self.assertEqual(wrong["payload"]["signals"][0]["action"], "hold")
        with patch("dockdack.trading.model_exit_schedule.model_exit_schedule",
                   return_value=ModelExitSchedule(9, "elapsed")):
            wrong_timing = PrototypeWorker(model_id, predictors={"domestic": predictor}).dispatch(request)
        self.assertEqual(wrong_timing["payload"]["signals"][0]["action"], "hold")
        scheduled = PrototypeWorker(model_id, predictors={"domestic": predictor}).dispatch(request)
        signal = scheduled["payload"]["signals"][0]
        self.assertEqual(signal["action"], "buy")
        self.assertTrue(signal["signal_id"].startswith(model_id + ":"))
        self.assertEqual(signal["strategy_id"], model_id)
        self.assertNotIn("stop_loss_price", signal)
        self.assertNotIn("take_profit_price", signal)
        self.assertEqual(scheduled["metadata"]["domestic"]["horizon_sessions"], 10)

    def test_execution_reinfers_fresh_quote_and_actual_limit_then_fails_closed(self):
        model_id = "mark1-23-prototype"
        item = WatchItem(Instrument(Market.DOMESTIC, "005930", "KRX"))
        quote = Quote(Market.DOMESTIC, "005930", "Samsung", "KRX", Decimal("100"), "KRW")
        history = DailyHistory(Market.DOMESTIC, "005930", "KRX", "KRW", 30,
                               tuple(DailyBar(day, Decimal("100"), Decimal("103"),
                                              Decimal("99"), Decimal("101"), Decimal("1000"))
                                     for day in DAYS))
        snapshot = MarketSnapshot(quote, history, NOW)
        rule = TriggerRule("horizon-offline", item.id, TriggerKind.EXTERNAL,
                           OrderSide.BUY, 1, Decimal("10000"))
        engine = SimpleNamespace(clock=lambda: NOW, _validate_snapshot=AutoTrader._validate_snapshot)
        window = SimpleNamespace(service=SimpleNamespace(mode=TradingMode.DEMO),
                                 store=SimpleNamespace(mode=TradingMode.DEMO,
                                                       external_for_rule=lambda _: None), engine=engine)
        predictor = CandidatePredictor(model_id)
        bridge = model_bridge_type(model_id)(window, predictors={"domestic": predictor})
        with patch("dockdack.trading.model_exit_schedule.model_exit_schedule", return_value=None):
            with self.assertRaisesRegex(ValueError, "exit schedule"):
                bridge.validate_execution(item, rule, snapshot, Decimal("99"))
        self.assertEqual(predictor.queries, [])
        with patch("dockdack.trading.model_exit_schedule.model_exit_schedule",
                   return_value=ModelExitSchedule(9, "preclose")):
            bridge.validate_execution(item, rule, snapshot, Decimal("99"))
        self.assertEqual(predictor.queries, [Decimal("100"), Decimal("99")])
        predictor.queries.clear()
        predictor.reject_price = Decimal("99")
        with patch("dockdack.trading.model_exit_schedule.model_exit_schedule",
                   return_value=ModelExitSchedule(9, "preclose")):
            with self.assertRaisesRegex(ValueError, "policy fails"):
                bridge.validate_execution(item, rule, snapshot, Decimal("99"), stage="final_send")
        self.assertEqual(predictor.queries, [Decimal("100"), Decimal("99")])
        window.service.mode = TradingMode.REAL
        with self.assertRaises(ValueError):
            bridge.validate_execution(item, rule, snapshot, Decimal("99"))


if __name__ == "__main__":
    unittest.main()
