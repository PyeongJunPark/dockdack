from decimal import Decimal
import unittest

from dockdack.signals import evaluate_signal


class SignalRuleTests(unittest.TestCase):
    def test_up_and_lower_than_previous_close_buys(self):
        signal = evaluate_signal(current_price="99", previous_close="100", predicted_direction="UP")
        self.assertEqual((signal.action, signal.reason), ("BUY", "UP_BELOW_PREVIOUS_CLOSE"))

    def test_not_up_and_higher_than_previous_close_sells_held_position(self):
        signal = evaluate_signal(current_price="101", previous_close="100", predicted_direction="NOT_UP",
                                 position_quantity="2", average_entry_price="105")
        self.assertEqual((signal.action, signal.reason), ("SELL", "NOT_UP_ABOVE_PREVIOUS_CLOSE"))

    def test_exactly_one_percent_takes_profit_even_when_model_says_buy(self):
        signal = evaluate_signal(current_price="303", previous_close="310", predicted_direction="UP",
                                 position_quantity="1", average_entry_price="300")
        self.assertEqual((signal.action, signal.reason), ("SELL", "TAKE_PROFIT_1PCT"))
        self.assertEqual(signal.unrealized_profit_pct, Decimal("1"))

    def test_just_below_one_percent_does_not_trigger_profit_exit(self):
        signal = evaluate_signal(current_price="302.999", previous_close="310", predicted_direction="NOT_UP",
                                 position_quantity="1", average_entry_price="300")
        self.assertEqual(signal.action, "HOLD")

    def test_profit_exit_does_not_need_model_or_previous_close(self):
        signal = evaluate_signal(current_price="102", position_quantity="1", average_entry_price="100")
        self.assertEqual((signal.action, signal.reason), ("SELL", "TAKE_PROFIT_1PCT"))

    def test_flat_position_never_emits_sell(self):
        signal = evaluate_signal(current_price="102", previous_close="100", predicted_direction="NOT_UP")
        self.assertEqual((signal.action, signal.reason), ("HOLD", "NO_POSITION_TO_SELL"))
        self.assertIsNone(signal.unrealized_profit_pct)

    def test_unmatched_direction_or_equal_price_holds(self):
        for price, direction in ((100, "UP"), (100, "NOT_UP"), (99, "NOT_UP"), (101, "UP")):
            with self.subTest(price=price, direction=direction):
                self.assertEqual(evaluate_signal(current_price=price, previous_close=100,
                                                 predicted_direction=direction).action, "HOLD")

    def test_buy_rule_also_works_with_existing_holding(self):
        signal = evaluate_signal(current_price=99, previous_close=100, predicted_direction="UP",
                                 position_quantity=2, average_entry_price=110)
        self.assertEqual(signal.action, "BUY")

    def test_position_gain_not_previous_day_gain_controls_profit_exit(self):
        signal = evaluate_signal(current_price=101, previous_close=90, predicted_direction="UP",
                                 position_quantity=1, average_entry_price=105)
        self.assertEqual(signal.action, "HOLD")
        self.assertLess(signal.unrealized_profit_pct, 0)

    def test_invalid_inputs_never_become_trade_signals(self):
        for changes in ({"current_price": "NaN"}, {"current_price": "Infinity"},
                        {"current_price": 0}, {"current_price": True},
                        {"position_quantity": -1}, {"position_quantity": "NaN"},
                        {"position_quantity": 1},
                        {"position_quantity": 1, "average_entry_price": 0},
                        {"previous_close": -1}, {"predicted_direction": "SELL"}):
            with self.subTest(changes=changes):
                inputs = dict(current_price=100, previous_close=100, predicted_direction="UP")
                inputs.update(changes)
                with self.assertRaises(ValueError):
                    evaluate_signal(**inputs)


if __name__ == "__main__":
    unittest.main()
