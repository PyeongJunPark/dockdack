"""Same broker symbol is rendered as separate confirmed model acquisitions."""
from dataclasses import replace
from decimal import Decimal as D
import importlib.util
import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from dockdack.models import Market, Quote
from dockdack.gui_service import Instrument
from dockdack.portfolio import PortfolioMarketState
from test_portfolio import NOW, account, position


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Install GUI extra')
class StrategyLotPortfolioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from dockdack.portfolio_gui import PortfolioPanel
        self.panel = PortfolioPanel()
        self.held = replace(position(quantity='2'), average_price=D('105'), current_price=D('110'),
                            evaluation_amount=D('220'), profit_loss=D('10'))
        self.lots = ({'lot_id': 'buy-old', 'model_title': 'mark1 prototype', 'quantity': D(1),
                      'sellable_quantity': D(1), 'average_price': D(100),
                      'take_profit_price': D(101), 'stop_loss_price': D('99.1')},
                     {'lot_id': 'buy-new', 'model_title': 'mark1.1 prototype', 'quantity': D(1),
                      'sellable_quantity': D(1), 'average_price': D(110),
                      'take_profit_price': D('110.55'), 'stop_loss_price': D('109.56')})
        self.targets = {'lots': self.lots, 'reconciled': True}
        self.panel.set_exit_targets({'domestic:KRX:005930': self.targets})
        self.panel.apply({Market.DOMESTIC: PortfolioMarketState(Market.DOMESTIC,
                          account(positions=(self.held,)), NOW, NOW)}, now=NOW)

    def tearDown(self):
        self.panel.close()
        self.panel.deleteLater()
        self.app.processEvents()

    def test_two_rows_same_symbol_have_independent_fill_cost_and_targets(self):
        table = self.panel.table
        self.assertEqual(table.rowCount(), 2)
        self.assertEqual([table.item(row, 11).text() for row in range(2)], ['mark1 prototype', 'mark1.1 prototype'])
        self.assertEqual([table.item(row, 4).text() for row in range(2)], ['100', '110'])
        self.assertEqual([table.item(row, 2).text() for row in range(2)], ['1', '1'])
        self.assertEqual([table.item(row, 9).text() for row in range(2)], ['≥ 101', '≥ 110.55'])
        self.assertEqual(self.panel.market_labels[Market.DOMESTIC]['evaluation'].text(), '220 KRW')
        self.assertIn('buy-new', table.item(1, 11).toolTip())

    def test_quote_updates_both_rows_without_overwriting_each_lot_targets(self):
        inst = Instrument(Market.DOMESTIC, '005930', 'KRX')
        self.panel.apply_holding_quote({'instrument': inst, 'watch_id': 'domestic:KRX:005930',
            'quote': Quote(inst.market, inst.symbol, 'test', inst.exchange, D('111'), 'KRW'), 'targets': self.targets})
        self.assertEqual([self.panel.table.item(row, 5).text() for row in range(2)], ['111', '111'])
        self.assertEqual(self.panel.table.item(1, 9).text(), '≥ 110.55')

    def test_unreconciled_inventory_shows_broker_total_warning_not_fake_lots(self):
        self.panel.set_exit_targets({'domestic:KRX:005930': {**self.targets, 'reconciled': False,
                                                          'issues': ('unmatched broker quantity',)}})
        self.assertEqual(self.panel.table.rowCount(), 1)
        self.assertEqual(self.panel.table.item(0, 2).text(), '2')
        self.assertIn('보류', self.panel.table.item(0, 9).text())
        self.assertIn('대조', self.panel.table.item(0, 11).text())

    def test_shared_broker_sellable_limit_is_not_displayed_as_independent_quantity(self):
        lots = tuple({**lot, 'broker_sellable_quantity': D(1), 'sellable_is_shared': True} for lot in self.lots)
        self.panel.set_exit_targets({'domestic:KRX:005930': {**self.targets, 'lots': lots}})
        self.assertEqual([self.panel.table.item(row, 3).text() for row in range(2)], ['공유 1', '공유 1'])
        self.assertIn('더해 팔 수 있다는 뜻이 아닙니다', self.panel.table.item(0, 3).toolTip())

    def test_policy_never_claims_sellable_when_broker_reports_zero(self):
        from types import SimpleNamespace
        from dockdack.execution_policy import holding_exit_targets
        from dockdack.models import TradingMode
        source_lots = tuple({**lot, 'quantity_remaining': lot['quantity'], 'available_quantity': D(1)} for lot in self.lots)
        store = SimpleNamespace(mode=TradingMode.DEMO, exit_targets=lambda key: None,
            prototype_inventory=lambda *args, **kwargs: {'lots': source_lots, 'reconciled': True,
                                                         'has_prototype_history': True, 'issues': ()})
        result = holding_exit_targets(store, replace(self.held, sellable_quantity=D(0)), prototype_lots=True)
        self.assertEqual([lot['sellable_quantity'] for lot in result['lots']], [D(0), D(0)])


if __name__ == '__main__':
    unittest.main()
