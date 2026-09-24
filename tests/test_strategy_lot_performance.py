"""A later model lot must not sell another model's cheaper FIFO purchase."""
from decimal import Decimal as D
import unittest

from dockdack.performance import realized_performance
from dockdack.trade_journal import daily_trade_journal
from test_performance import order


def sell(number, buy, quantity=1, price='120', **changes):
    return order(number, 'sell', quantity, price, prototype_lot_id=str(buy),
                 prototype_buy_rule_id=str(buy), **changes)


class StrategyLotPerformanceTests(unittest.TestCase):
    def test_later_model_purchase_is_used_instead_of_first_fifo_purchase(self):
        result = realized_performance([order(1, 'buy', 1, '100'), order(2, 'buy', 1, '110'),
                                       sell(3, 2), sell(4, 1)])
        self.assertEqual(result['by_rule_id']['3']['cost_basis'], D('110'))
        self.assertEqual(result['by_rule_id']['3']['realized_profit'], D('10'))
        self.assertEqual(result['by_rule_id']['4']['cost_basis'], D('100'))
        self.assertEqual(result['by_rule_id']['4']['realized_profit'], D('20'))
        self.assertEqual(result['by_rule_id']['3']['basis'], 'strategy_lot')

    def test_partial_sell_reserves_only_named_lot_and_general_fifo_remains(self):
        result = realized_performance([order(1, 'buy', 2, '100'), order(2, 'buy', 3, '110'),
                                       sell(3, 2, 1), order(4, 'sell', 2, '120'), sell(5, 2, 2)])['by_rule_id']
        self.assertEqual(result['3']['cost_basis'], D('110'))
        self.assertEqual(result['4']['cost_basis'], D('200'))
        self.assertEqual(result['5']['cost_basis'], D('220'))

    def test_unknown_or_exhausted_named_lot_cannot_borrow_other_models(self):
        for target in ('missing', '1'):
            with self.subTest(target=target):
                rows = [order(1, 'buy', 1, '100'), order(2, 'buy', 5, '110'), sell(3, target, 2)]
                metric = realized_performance(rows)['by_rule_id']['3']
                self.assertEqual(metric['status'], 'unknown')
                self.assertIsNone(metric['realized_profit'])

    def test_allocation_cannot_borrow_other_symbol_or_later_purchase(self):
        for rows in ([order(1, 'buy', symbol='OTHER'), sell(2, 1)],
                     [sell(1, 2), order(2, 'buy')]):
            with self.subTest(rows=rows):
                metric = next(value for value in realized_performance(rows)['by_rule_id'].values() if value['side'] == 'sell')
                self.assertEqual(metric['status'], 'unknown')

    def test_conflicting_allocation_is_unknown(self):
        row = sell(3, 1)
        row['prototype_buy_rule_id'] = '2'
        metric = realized_performance([order(1, 'buy'), order(2, 'buy'), row])['by_rule_id']['3']
        self.assertEqual(metric['reason_code'], 'invalid_lot_allocation')

    def test_unknown_fill_price_stays_unknown_not_broker_aggregate(self):
        metric = realized_performance([order(1, 'buy', fill_price=None), sell(2, 1)])['by_rule_id']['2']
        self.assertEqual(metric['status'], 'unknown')
        self.assertIsNone(metric['realized_profit'])

    def test_daily_journal_uses_same_specific_purchase(self):
        result = daily_trade_journal([order(1, 'buy', price='100'), order(2, 'buy', price='110'), sell(3, 2)])
        row = next(row for row in result['rows'] if row['rule_id'] == '3')
        self.assertEqual(row['metric']['realized_profit'], D('10'))


if __name__ == '__main__':
    unittest.main()
