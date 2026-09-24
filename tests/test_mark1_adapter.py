"""No network, Torch or live ledger is needed for mark1 signal validation."""

import copy
import tempfile
import unittest
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from dockdack.lstm30_adapter import atomic_json
from dockdack.mark1_adapter import Mark1SignalProducer, SOURCE_ID, decide_position
from dockdack.market_schedule import EXTRA_CLOSURES, calendar_for
from dockdack.models import Market
from test_lstm30_adapter import NOW, chart, position


def mark1_chart(*, market="domestic", count=30):
    result = chart(market=market, count=count)
    last = date.fromisoformat(result["stocks"][0]["bars"][-1]["date"])
    calendar = calendar_for(Market(market), last.year)
    days = [stamp.date() for stamp in calendar.sessions
            if stamp.date() <= last and (Market(market), stamp.date()) not in EXTRA_CLOSURES][-count:]
    for row, day in zip(result["stocks"][0]["bars"], days):
        row["date"] = day.isoformat()
    return result


def prediction(probability=.7):
    return {"probability_success": probability, "buy_threshold": .5, "predicts_success": probability > .5}


def producer(*, probability=.7, provider=position, state_path=None, market="domestic"):
    model = SimpleNamespace(metadata={"market": market, "model_name": "fixture-gru", "buy_threshold": .5},
                            buy_threshold=.5, predict=Mock(return_value=prediction(probability)))
    return Mark1SignalProducer({market: model}, position_provider=provider, quantity=1,
                               max_krw="10000", max_usd="10000", state_path=state_path, clock=lambda: NOW), model


class Mark1RuleTests(unittest.TestCase):
    def test_buy_boundary_is_strict(self):
        for p, action in ((0, "hold"), (.49999, "hold"), (.5, "hold"), (.50001, "buy"), (1, "buy")):
            with self.subTest(probability=p):
                result = decide_position(current_price=100, quantity=0, sellable_quantity=0, prediction=prediction(p))
                self.assertEqual(result["action"], action)

    def test_cost_exit_exact_boundaries_and_mark0_loss_not_used(self):
        for price, action, reason in (("101", "sell", "TAKE_PROFIT_1PCT"),
                                     ("99.1", "sell", "STOP_LOSS_0_9PCT"),
                                     ("99.2", "hold", "POSITION_INSIDE_EXIT_BOUNDS"),
                                     ("99.100001", "hold", "POSITION_INSIDE_EXIT_BOUNDS"),
                                     ("100.999999", "hold", "POSITION_INSIDE_EXIT_BOUNDS")):
            with self.subTest(price=price):
                result = decide_position(current_price=price, quantity=2, sellable_quantity=1, average_price=100)
                self.assertEqual((result["action"], result["reason"]), (action, reason))
                if price == "99.1":
                    self.assertEqual(result["cost_loss_pct"], "0.9")

    def test_held_position_ignores_prediction(self):
        self.assertEqual(decide_position(current_price=100, quantity=1, sellable_quantity=1,
                                        average_price=100, prediction={"broken": True})["action"], "hold")
        self.assertEqual(decide_position(current_price=101, quantity=1, sellable_quantity=0,
                                        average_price=100)["reason"], "NO_SELLABLE_POSITION")

    def test_invalid_prediction_or_position_fails_closed(self):
        for invalid in ({**prediction(), "probability_success": "NaN"},
                        {**prediction(), "probability_success": 1.01},
                        {**prediction(), "buy_threshold": .4},
                        {**prediction(.5), "predicts_success": True},
                        {**prediction(), "predicts_success": 1}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                decide_position(current_price=100, quantity=0, sellable_quantity=0, prediction=invalid)
        for changes in ({"quantity": 1.5}, {"sellable_quantity": 2}, {"average_price": 0}):
            arguments = dict(current_price=100, quantity=1, sellable_quantity=1, average_price=100)
            arguments.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                decide_position(**arguments)


class Mark1AdapterTests(unittest.TestCase):
    def test_current_price_is_explicit_input_and_wire_is_compatible(self):
        adapter, model = producer()
        payload, diagnostics = adapter(mark1_chart())
        self.assertEqual(payload["source_id"], SOURCE_ID)
        self.assertEqual(payload["signals"][0]["action"], "buy")
        self.assertEqual(model.predict.call_args.kwargs, {"current_price": Decimal("100")})
        self.assertEqual(len(model.predict.call_args.args[0]), 30)
        self.assertEqual(diagnostics[0]["input_tokens"], 31)
        self.assertFalse(diagnostics[0]["intraday_path_verified"])
        self.assertEqual(diagnostics[0]["model_name"], "fixture-gru")
        self.assertNotIn("prediction", payload["signals"][0])

    def test_current_day_eventual_ohlcv_is_not_passed(self):
        charts = mark1_chart()
        stock = charts["stocks"][0]
        stock["bars"].append({"date": "2026-09-15", "open": "NaN", "high": "NaN", "low": "NaN",
                              "close": "NaN", "volume": "NaN", "is_current_day": True})
        stock.update(available_days=31, requested_days=31)
        adapter, model = producer()
        payload, _ = adapter(charts)
        self.assertEqual(payload["signals"][0]["action"], "buy")
        self.assertEqual(len(model.predict.call_args.args[0]), 30)

    def test_missing_session_blocks_entry_not_exit(self):
        charts = mark1_chart(count=31)
        charts["stocks"][0]["bars"].pop(-10)
        charts["stocks"][0].update(available_days=30, requested_days=30)
        adapter, model = producer()
        payload, diagnostic = adapter(charts)
        self.assertEqual(payload["signals"][0]["action"], "hold")
        self.assertIn("consecutive", diagnostic[0]["error"])
        model.predict.assert_not_called()
        charts["stocks"][0]["price"] = "99.1"
        adapter, model = producer(provider=lambda s: position(s, quantity="2", sellable="1", average="100"))
        adapter.predictors.clear()
        payload, _ = adapter(charts)
        self.assertEqual(payload["signals"][0]["cost_loss_pct"], "0.9")

    def test_stale_quote_remains_non_actionable(self):
        charts = mark1_chart()
        charts["stocks"][0]["quote_fetched_at"] = (NOW - timedelta(seconds=16)).isoformat()
        adapter, model = producer()
        payload, _ = adapter(charts)
        self.assertEqual(payload["signals"][0]["action"], "hold")
        model.predict.assert_not_called()

    def test_us_calendar_and_case_are_preserved(self):
        charts = mark1_chart(market="us")
        charts["stocks"][0].update(symbol="BRKb", watch_id="us:ND:BRKb")
        adapter, _ = producer(market="us")
        payload, _ = adapter(charts)
        self.assertEqual(payload["signals"][0]["symbol"], "BRKb")
        self.assertEqual(payload["signals"][0]["action"], "buy")

    def test_immutable_replay_and_other_strategy_state_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.json"
            adapter, model = producer(state_path=path)
            payload, diagnostic = adapter(mark1_chart())
            second, second_model = producer(state_path=path)
            self.assertEqual(second(mark1_chart()), (payload, diagnostic))
            second_model.predict.assert_not_called()
            changed = copy.deepcopy(adapter.state)
            changed["export-123"]["payload"]["source_id"] = "lstm30-mark0"
            atomic_json(path, changed)
            with self.assertRaisesRegex(ValueError, "own decision state"):
                producer(state_path=path)


if __name__ == "__main__":
    unittest.main()
