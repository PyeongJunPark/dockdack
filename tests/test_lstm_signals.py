from datetime import date, timedelta
from decimal import Decimal
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from dockdack import Market, TradingMode


ML_AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "numpy"))
if ML_AVAILABLE:
    import numpy as np
    from examples.emit_lstm_signal import infer, live_signal, manual_signal, recent_completed_bars


@unittest.skipUnless(ML_AVAILABLE, "Install ml extra for LSTM signal integration tests")
class LSTMSignalTests(unittest.TestCase):
    def setUp(self):
        self.config = {"symbol": "005930", "exchange": "KRX", "lookback": 2,
                       "database": "unused.sqlite3", "start": "2020-01-01"}
        self.checkpoint = {"metadata": self.config}
        self.broker = Mock(mode=TradingMode.DEMO)
        self.broker.get_quote.return_value = SimpleNamespace(
            market=Market.DOMESTIC, symbol="005930", exchange="KRX", currency="KRW", price=Decimal("99"),
        )
        self.broker.account_domestic.return_value = SimpleNamespace(
            market=Market.DOMESTIC, currency="KRW", positions=[],
        )
        self.broker.place_order.side_effect = AssertionError("Signal code must not submit orders")
        self.broker.cancel_order.side_effect = AssertionError("Signal code must not cancel orders")

    def bar(self, day, price="100"):
        return SimpleNamespace(market=Market.DOMESTIC, symbol="005930", exchange="KRX", trade_date=day,
                               open=Decimal(price), high=Decimal(price), low=Decimal(price),
                               close=Decimal(price), volume=Decimal("1000"))

    def test_today_unfinished_bar_is_excluded_and_past_bars_are_sorted(self):
        today = date(2026, 9, 15)
        self.broker.iter_daily_bars_domestic.return_value = iter([
            tuple(self.bar(today - timedelta(days=i), str(100 - i)) for i in range(4)),
        ])
        dates, bars = recent_completed_bars(self.broker, self.config, Market.DOMESTIC, today)
        self.assertEqual(list(dates), ["2026-09-12", "2026-09-13", "2026-09-14"])
        self.assertEqual(list(bars[:, 3]), [97, 98, 99])
        self.assertEqual(self.broker.iter_daily_bars_domestic.call_args.kwargs["base_date"], today - timedelta(days=1))

    def test_take_profit_does_not_call_model_or_chart_and_never_places_order(self):
        self.broker.get_quote.return_value.price = Decimal("101")
        self.broker.account_domestic.return_value.positions = [SimpleNamespace(
            market=Market.DOMESTIC, symbol="005930", exchange="KRX", currency="KRW",
            quantity=Decimal("1"), average_price=Decimal("100"),
        )]
        result = live_signal(self.checkpoint, self.broker, date(2026, 9, 15))
        self.assertEqual(result["reason"], "TAKE_PROFIT_1PCT")
        self.assertEqual(result["action"], "SELL")
        self.broker.iter_daily_bars_domestic.assert_not_called()
        self.broker.place_order.assert_not_called()
        self.broker.cancel_order.assert_not_called()

    def test_current_quote_and_model_are_connected_to_buy_rule(self):
        self.broker.iter_daily_bars_domestic.return_value = iter([
            tuple(self.bar(date(2026, 9, day)) for day in (14, 13, 12)),
        ])
        prediction = {"as_of_date": "2026-09-14", "predicted_direction": "UP", "up_probability": 0.7}
        with patch("examples.emit_lstm_signal.infer", return_value=prediction):
            result = live_signal(self.checkpoint, self.broker, date(2026, 9, 15))
        self.assertEqual(result["action"], "BUY")
        self.assertEqual(result["previous_trade_date"], "2026-09-14")
        self.broker.place_order.assert_not_called()

    def test_wrong_quote_symbol_is_rejected(self):
        self.broker.get_quote.return_value.symbol = "000660"
        with self.assertRaisesRegex(ValueError, "Quote"):
            live_signal(self.checkpoint, self.broker, date(2026, 9, 15))

    def test_us_named_exchange_holding_and_symbol_specific_account_query(self):
        self.config.update(symbol="AAPL", exchange="ND")
        self.broker.account_us.return_value = SimpleNamespace(
            market=Market.US, currency="USD", positions=[SimpleNamespace(
                market=Market.US, symbol="AAPL", exchange="NASDAQ", currency="USD",
                quantity=Decimal("2"), average_price=Decimal("100"),
            )],
        )
        self.broker.get_quote.return_value = SimpleNamespace(
            market=Market.US, symbol="AAPL", exchange="ND", currency="USD", price=Decimal("101"),
        )
        result = live_signal(self.checkpoint, self.broker, date(2026, 9, 15))
        self.assertEqual(result["reason"], "TAKE_PROFIT_1PCT")
        self.assertEqual(result["position_quantity"], "2")
        self.broker.account_us.assert_called_once_with(exchange="ND", symbol="AAPL")
        self.broker.account_domestic.assert_not_called()
        self.broker.place_order.assert_not_called()

    def test_failed_prediction_returns_hold(self):
        with patch("examples.emit_lstm_signal.recent_completed_bars", side_effect=ValueError("missing data")):
            result = live_signal(self.checkpoint, self.broker, date(2026, 9, 15))
        self.assertEqual(result["action"], "HOLD")
        self.assertIn("missing data", result["prediction_error"])

    def test_stale_local_data_is_not_used_for_current_session(self):
        old_dates = np.array(["2026-08-20", "2026-08-21"])
        bars = np.array([[100, 100, 100, 100, 1], [100, 100, 100, 100, 1]])
        with patch("examples.emit_lstm_signal.load_bars", return_value=(old_dates, bars, 0)), \
             patch("examples.emit_lstm_signal.infer") as predict:
            result = manual_signal(self.checkpoint, Path("test.sqlite3"), 99, 100, date(2026, 9, 14), 0, None)
        self.assertEqual(result["action"], "HOLD")
        self.assertEqual(result["reason"], "STALE_OR_MISMATCHED_DAILY_DATA")
        predict.assert_not_called()

    def test_local_input_is_cut_off_at_previous_date(self):
        dates = np.array(["2026-09-11", "2026-09-14"])
        bars = np.array([[100, 100, 100, 100, 1], [100, 100, 100, 100, 1]])
        with patch("examples.emit_lstm_signal.load_bars", return_value=(dates, bars, 0)) as load, \
             patch("examples.emit_lstm_signal.infer", return_value={"predicted_direction": "UP"}):
            result = manual_signal(self.checkpoint, Path("test.sqlite3"), 99, 100, date(2026, 9, 14), 0, None)
        self.assertEqual(load.call_args.kwargs["end"], "2026-09-14")
        self.assertEqual(result["action"], "BUY")

    def test_invalid_scaler_is_rejected_before_inference(self):
        from examples.train_lstm_daily import FEATURE_NAMES
        checkpoint = {"metadata": {"feature_names": FEATURE_NAMES}, "mean": [0] * 5, "std": [0] * 5}
        with self.assertRaisesRegex(ValueError, "normalization"):
            infer(checkpoint, [], np.empty((0, 5)))

    def test_real_broker_is_not_accepted_by_demo_example(self):
        self.broker.mode = TradingMode.REAL
        with self.assertRaisesRegex(ValueError, "demo"):
            live_signal(self.checkpoint, self.broker, date(2026, 9, 15))


if __name__ == "__main__":
    unittest.main()
