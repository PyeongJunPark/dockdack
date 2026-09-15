"""Rule and external-wire tests require neither Torch nor a broker connection."""

import subprocess
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dockdack.lstm30_adapter import (
    LSTM30SignalProducer, atomic_json, completed_bars, decide_position,
    demo_position_provider, previous_trading_day, read_json,
)
from dockdack.models import Market, TradingMode
from examples.publish_lstm30 import fixture_position_provider


NOW = datetime(2026, 9, 15, 1, 0, tzinfo=timezone.utc)


def chart(*, market="domestic", count=30, last_day=None):
    exchange, symbol, currency = ("KRX", "005930", "KRW") if market == "domestic" else ("ND", "AAPL", "USD")
    # NOW is September 15 in Seoul but September 14 in New York.
    days, day = [], last_day or (date(2026, 9, 14) if market == "domestic" else date(2026, 9, 11))
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day -= timedelta(days=1)
    days.reverse()
    stock = {"watch_id": f"{market}:{exchange}:{symbol}", "market": market, "symbol": symbol,
             "exchange": exchange, "currency": currency, "status": "ok", "complete": True,
             "requested_days": count, "available_days": count, "price": "100",
             "quote_fetched_at": NOW.isoformat(), "quote_age_seconds": 0, "quote_stale": False,
             "bars": [{"date": day.isoformat(), "open": "100", "high": "103", "low": "99",
                       "close": "101", "volume": "1000", "is_current_day": False} for day in days]}
    return {"schema_version": 1, "export_id": "export-123", "trading_mode": "demo", "source": "kiwoom_demo",
            "created_at": NOW.isoformat(), "adjusted_prices": True, "stocks": [stock]}


def position(stock, *, quantity="0", sellable="0", average=None):
    return {**{key: stock[key] for key in ("market", "symbol", "exchange", "currency")},
            "quantity": quantity, "sellable_quantity": sellable, "average_price": average,
            "fetched_at": NOW.isoformat()}


def prediction(probability=0.7):
    return {"probability_ge_1pct": probability, "buy_threshold": 0.5, "predicts_gain": probability >= 0.5}


def producer(*, state=None, provider=None, predictor=None, market="domestic", clock=lambda: NOW, **kwargs):
    predictor = predictor or SimpleNamespace(metadata={"market": market}, predict=Mock(return_value=prediction()))
    return LSTM30SignalProducer({market: predictor}, position_provider=provider or position,
                               quantity=kwargs.pop("quantity", 2), max_krw=kwargs.pop("max_krw", "10000"),
                               max_usd=kwargs.pop("max_usd", "10000"), state_path=state, clock=clock, **kwargs)


class RuleTests(unittest.TestCase):
    def test_flat_buys_threshold_inclusive(self):
        self.assertEqual(decide_position(current_price="100", quantity=0, sellable_quantity=0,
                                         prediction=prediction(0.5))["action"], "buy")

    def test_flat_low_probability_holds(self):
        self.assertEqual(decide_position(current_price=100, quantity=0, sellable_quantity=0,
                                         prediction=prediction(0.499))["action"], "hold")

    def test_profit_and_loss_boundaries_work_without_model(self):
        for current, field, value in (("101", "cost_profit_pct", "1"), ("99.2", "cost_loss_pct", "0.8")):
            with self.subTest(current=current):
                result = decide_position(current_price=current, quantity=2, sellable_quantity=2, average_price=100)
                self.assertEqual(result["action"], "sell")
                self.assertEqual(result[field], value)

    def test_neither_exit_boundary_no_additional_buy(self):
        for current in ("100.999999", "99.200001", "100"):
            self.assertEqual(decide_position(current_price=current, quantity=2, sellable_quantity=2,
                                             average_price=100, prediction=prediction())["action"], "hold")

    def test_non_sellable_position_holds(self):
        self.assertEqual(decide_position(current_price=101, quantity=2, sellable_quantity=0,
                                         average_price=100)["reason"], "NO_SELLABLE_POSITION")

    def test_invalid_quantities_average_probability_rejected(self):
        for change in ({"quantity": -1}, {"quantity": 1.5}, {"quantity": 0, "sellable_quantity": 1},
                       {"average_price": 0}, {"current_price": "NaN"}):
            kwargs = dict(current_price=100, quantity=2, sellable_quantity=2, average_price=100)
            kwargs.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                decide_position(**kwargs)
        with self.assertRaises(ValueError):
            decide_position(current_price=100, quantity=0, sellable_quantity=0,
                            prediction={**prediction(), "predicts_gain": False})


class AdapterTests(unittest.TestCase):
    def test_buy_wire_contract_no_unknown_fields(self):
        payload, diagnostics = producer()(chart())
        self.assertEqual(set(payload), {"schema_version", "source_id", "trading_mode", "signals"})
        signal = payload["signals"][0]
        self.assertEqual(signal["action"], "buy")
        self.assertEqual(signal["quantity"], 2)
        self.assertEqual(signal["max_notional"], "10000")
        self.assertNotIn("reason", signal)
        self.assertNotIn("prediction", signal)
        self.assertEqual(signal["generated_at"], NOW.isoformat())
        self.assertEqual(signal["expires_at"], (NOW + timedelta(minutes=2)).isoformat())
        self.assertIn("not current entry return", diagnostics[0]["target_basis"])

    def test_us_preserves_exported_symbol_case(self):
        charts = chart(market="us")
        charts["stocks"][0].update(symbol="BRKb", watch_id="us:ND:BRKb")
        payload, _ = producer(market="us")(charts)
        self.assertEqual(payload["signals"][0]["symbol"], "BRKb")

    def test_profit_and_loss_exits_ignore_missing_broken_model_history(self):
        for price, field in (("101", "cost_profit_pct"), ("99.2", "cost_loss_pct")):
            charts = chart()
            charts["stocks"][0].update(price=price, complete=False, bars=[])
            p = producer(provider=lambda s: position(s, quantity="5", sellable="3", average="100"))
            p.predictors.clear()
            result, _ = p(charts)
            self.assertEqual(result["signals"][0]["action"], "sell")
            self.assertIn(field, result["signals"][0])
            self.assertEqual(result["signals"][0]["quantity"], 2)

    def test_hold_contains_no_order_fields(self):
        result, _ = producer(predictor=SimpleNamespace(metadata={"market": "domestic"},
                                                       predict=Mock(return_value=prediction(0.1))))(chart())
        row = result["signals"][0]
        self.assertEqual(row["action"], "hold")
        self.assertFalse({"quantity", "max_notional", "cost_profit_pct", "cost_loss_pct"} & row.keys())

    def test_today_is_excluded_even_with_false_flag(self):
        charts = chart()
        charts["stocks"][0]["bars"][-1]["date"] = "2026-09-15"
        result, diagnostics = producer()(charts)
        self.assertEqual(result["signals"][0]["action"], "hold")
        self.assertIn("got 29", diagnostics[0]["error"])

    def test_31_exported_bars_supply_exactly_30_completed(self):
        charts = chart()
        stock = charts["stocks"][0]
        stock["bars"].append({**stock["bars"][-1], "date": "2026-09-15"})
        stock.update(available_days=31, requested_days=31)
        self.assertEqual(len(completed_bars(charts["stocks"][0], NOW)), 30)

    def test_old_history_with_fresh_quote_and_position_cannot_buy(self):
        for last_day in (date(2020, 9, 14), date(2026, 9, 11)):
            with self.subTest(last_day=last_day):
                payload, diagnostics = producer()(chart(last_day=last_day))
                self.assertEqual(payload["signals"][0]["action"], "hold")
                self.assertIn("Stale completed bars", diagnostics[0]["error"])

    def test_weekend_and_us_holiday_accept_the_actual_previous_session(self):
        for market, moment, expected in (
            ("domestic", datetime(2026, 9, 13, 1, tzinfo=timezone.utc), date(2026, 9, 11)),
            ("us", datetime(2026, 9, 8, 14, tzinfo=timezone.utc), date(2026, 9, 4)),
        ):
            with self.subTest(market=market, moment=moment):
                self.assertEqual(previous_trading_day(market, moment), expected)
                stock = chart(market=market, last_day=expected)["stocks"][0]
                self.assertEqual(len(completed_bars(stock, moment)), 30)

    def test_calendar_failure_blocks_buy_but_never_cost_based_exit(self):
        with patch("dockdack.lstm30_adapter._market_calendar", side_effect=ImportError("calendar missing")):
            payload, diagnostics = producer()(chart())
            self.assertEqual(payload["signals"][0]["action"], "hold")
            self.assertIn("Trading calendar unavailable", diagnostics[0]["error"])
            for price, field in (("101", "cost_profit_pct"), ("99.2", "cost_loss_pct")):
                charts = chart(last_day=date(2020, 9, 14))
                charts["stocks"][0]["price"] = price
                p = producer(provider=lambda s: position(s, quantity="2", sellable="2", average="100"))
                payload, _ = p(charts)
                self.assertEqual(payload["signals"][0]["action"], "sell")
                self.assertIn(field, payload["signals"][0])

    def test_market_local_day_used_for_us(self):
        charts = chart(market="us")
        charts["stocks"][0]["bars"][-1]["date"] = "2026-09-14"
        # 01:00 UTC is still September 14 in New York: this candle is incomplete.
        with self.assertRaisesRegex(ValueError, "got 29"):
            completed_bars(charts["stocks"][0], NOW)

    def test_bad_and_duplicate_bars_hold(self):
        for bad in ({"high": "99"}, {"volume": "-1"}, {"date": "2026-08-03"}):
            charts = chart()
            charts["stocks"][0]["bars"][-1].update(bad)
            result, _ = producer()(charts)
            self.assertEqual(result["signals"][0]["action"], "hold")

    def test_quote_stale_future_or_bad_position_holds(self):
        for change in ({"quote_stale": True}, {"quote_age_seconds": 16},
                       {"quote_fetched_at": (NOW - timedelta(seconds=16)).isoformat()},
                       {"quote_fetched_at": (NOW + timedelta(seconds=1)).isoformat()}):
            charts = chart()
            charts["stocks"][0].update(change)
            result, _ = producer()(charts)
            self.assertEqual(result["signals"][0]["action"], "hold")
        for provide in (lambda s: None, lambda s: {**position(s), "currency": "USD"},
                        lambda s: {**position(s), "fetched_at": (NOW - timedelta(seconds=16)).isoformat()}):
            result, _ = producer(provider=provide)(chart())
            self.assertEqual(result["signals"][0]["action"], "hold")

    def test_clock_rechecked_after_slow_model(self):
        times = iter([NOW, NOW, NOW + timedelta(seconds=16)])
        result, diagnostics = producer(clock=lambda: next(times))(chart())
        self.assertEqual(result["signals"][0]["action"], "hold")
        self.assertEqual(diagnostics[0]["reason"], "QUOTE_EXPIRED_DURING_INFERENCE")

    def test_missing_wrong_market_or_failed_model_holds(self):
        for model in (SimpleNamespace(metadata={"market": "us"}),
                      SimpleNamespace(metadata={"market": "domestic"}, predict=Mock(side_effect=RuntimeError("bad model")))):
            result, _ = producer(predictor=model)(chart())
            self.assertEqual(result["signals"][0]["action"], "hold")
        p = producer()
        p.predictors.clear()
        self.assertEqual(p(chart())[0]["signals"][0]["action"], "hold")

    def test_zero_cap_blocks_and_sell_respects_sellable_and_cash_cap(self):
        self.assertEqual(producer(max_krw="0")(chart())[0]["signals"][0]["action"], "hold")
        charts = chart()
        charts["stocks"][0]["price"] = "101"
        p = producer(quantity=10, max_krw="202", provider=lambda s: position(s, quantity="10", sellable="3", average="100"))
        self.assertEqual(p(charts)[0]["signals"][0]["quantity"], 2)

    def test_same_export_immutable_across_calls_and_restarts(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.json"
            first = producer(state=state)(chart())
            p = producer(state=state, provider=Mock(side_effect=AssertionError("must not re-query")),
                         clock=lambda: NOW + timedelta(seconds=40))
            self.assertEqual(first, p(chart()))
            changed = chart()
            changed["stocks"][0]["price"] = "90"
            with self.assertRaisesRegex(ValueError, "contents changed"):
                p(changed)

    def test_stale_state_pruned_and_expired_export_cannot_regenerate(self):
        p = producer()
        p(chart())
        future = NOW + timedelta(minutes=11)
        fresh = chart()
        fresh.update(export_id="next-export", created_at=future.isoformat())
        fresh["stocks"][0]["quote_fetched_at"] = future.isoformat()
        p.clock = lambda: future
        p(fresh)
        self.assertEqual(set(p.state), {"next-export"})
        with self.assertRaisesRegex(ValueError, "2 minutes"):
            p(chart())

    def test_nonregistered_error_chart_omitted(self):
        charts = chart()
        charts["stocks"][0]["status"] = "error"
        payload, diagnostics = producer()(charts)
        self.assertEqual(payload["signals"], [])
        self.assertFalse(diagnostics[0]["emitted"])

    def test_real_unknown_mode_and_identity_rejected(self):
        for change in ({"trading_mode": "real"}, {"source": "unknown"}, {"schema_version": True}):
            charts = chart()
            charts.update(change)
            with self.assertRaises(ValueError):
                producer()(charts)
        charts = chart()
        charts["stocks"][0]["currency"] = "USD"
        with self.assertRaises(ValueError):
            producer()(charts)

    def test_readonly_holdings_provider_aggregates_and_uses_symbol(self):
        stock = chart(market="us")["stocks"][0]
        rows = [SimpleNamespace(market=Market.US, currency="USD", exchange="NASDAQ", symbol="AAPL",
                                quantity=Decimal(qty), sellable_quantity=Decimal(qty), average_price=Decimal(avg))
                for qty, avg in (("2", "100"), ("1", "130"))]
        broker = SimpleNamespace(mode=TradingMode.DEMO, account_us=Mock(return_value=SimpleNamespace(
            market=Market.US, currency="USD", positions=rows)))
        result = demo_position_provider({"us": broker}, clock=lambda: NOW)(stock)
        self.assertEqual(result["quantity"], "3")
        self.assertEqual(result["average_price"], "110")
        broker.account_us.assert_called_once_with(exchange="ND", symbol="AAPL")
        rows[0].sellable_quantity = Decimal(3)
        rows[1].sellable_quantity = Decimal(0)
        with self.assertRaises(ValueError):
            demo_position_provider({"us": broker})(stock)
        broker.mode = TradingMode.REAL
        with self.assertRaises(ValueError):
            demo_position_provider({"us": broker})(stock)

    def test_fixture_missing_symbol_is_unknown_not_flat(self):
        provide = fixture_position_provider({"trading_mode": "demo", "positions": []})
        self.assertIsNone(provide(chart()["stocks"][0]))

    def test_atomic_json_roundtrip_and_duplicate_input_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.json"
            atomic_json(path, {"signals": []})
            self.assertEqual(read_json(path), {"signals": []})
            path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
            with self.assertRaises(ValueError):
                read_json(path)

    def test_cli_help_requires_explicit_policy_and_position_source(self):
        command = [sys.executable, "-m", "examples.publish_lstm30"]
        result = subprocess.run(command + ["--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--quantity", result.stdout)
        self.assertIn("--max-krw", result.stdout)
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)

    def test_cli_offline_exit_and_repeat_are_immutable_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            charts = chart()
            moment = datetime.now(timezone.utc).isoformat()
            charts["created_at"] = moment
            stock = charts["stocks"][0]
            stock.update(price="101", quote_fetched_at=moment)
            snapshot = position(stock, quantity="2", sellable="2", average="100")
            snapshot["fetched_at"] = moment
            atomic_json(folder / "charts.json", charts)
            atomic_json(folder / "positions.json", {"trading_mode": "demo", "positions": [snapshot]})
            output = folder / "signals.preview.json"
            command = [sys.executable, "-m", "examples.publish_lstm30", "--charts", str(folder / "charts.json"),
                       "--positions", str(folder / "positions.json"), "--output", str(output),
                       "--quantity", "2", "--max-krw", "10000", "--max-usd", "0"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            first = read_json(output)
            self.assertEqual(first["signals"][0]["cost_profit_pct"], "1")
            # Different position data on a repeated export must not rewrite its decision.
            atomic_json(folder / "positions.json", {"trading_mode": "demo", "positions": []})
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(first, read_json(output))


if __name__ == "__main__":
    unittest.main()
