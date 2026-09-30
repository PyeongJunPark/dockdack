"""Same broker symbol is rendered as separate confirmed model acquisitions."""
from dataclasses import replace
from decimal import Decimal as D
import importlib.util
import os
import unittest
from unittest.mock import patch

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

    def test_lot_quote_reuses_cells_and_preserves_unrelated_selection(self):
        other = replace(self.held, symbol='000660', name='unrelated', evaluation_amount=D(1))
        self.panel.apply({Market.DOMESTIC: PortfolioMarketState(Market.DOMESTIC,
            account(positions=(self.held, other)), NOW, NOW)}, now=NOW)
        view = self.panel.table
        cells = [[view.item(row, col) for col in range(12)] for row in range(3)]
        view.selectRow(2)
        inst = Instrument(Market.DOMESTIC, '005930', 'KRX')
        with patch.object(view, 'setItem', wraps=view.setItem) as write:
            self.panel.apply_holding_quote({'instrument': inst, 'watch_id': 'domestic:KRX:005930',
                'quote': Quote(inst.market, inst.symbol, 'test', inst.exchange, D('111'), 'KRW'), 'targets': self.targets})
        write.assert_not_called()
        for row in range(3):
            for col in range(12):
                self.assertIs(view.item(row, col), cells[row][col])
        self.assertEqual(view.currentRow(), 2)
        self.assertEqual(view.item(2, 5).text(), '110')

    def test_unreconciled_inventory_shows_broker_total_warning_not_fake_lots(self):
        self.panel.set_exit_targets({'domestic:KRX:005930': {**self.targets, 'reconciled': False,
                                                          'issues': ('unmatched broker quantity',)}})
        self.assertEqual(self.panel.table.rowCount(), 1)
        self.assertEqual(self.panel.table.item(0, 2).text(), '2')
        self.assertEqual([self.panel.table.item(0, col).text() for col in (9, 10, 12)], ['—'] * 3)
        self.assertNotIn('대조 필요', self.panel.table.item(0, 11).text())
        self.assertIn('자동매도 보류 1종목', self.panel.reconciliation_labels[Market.DOMESTIC].text())
        self.assertIn('unmatched broker quantity', self.panel.reconciliation_labels[Market.DOMESTIC].toolTip())

    def test_unreconciled_inventory_keeps_persisted_model_names_without_faking_cost_or_sell(self):
        lots = tuple({**lot, 'strategy_id': f'mark1-{index}-prototype',
                      'quantity_remaining': lot['quantity'], 'average_price': None}
                     for index, lot in enumerate(self.lots, 23))
        target = {'lots': lots, 'reconciled': False,
                  'issues': ('model cost unavailable', 'broker quantity unverified')}
        key = 'domestic:KRX:005930'
        self.panel.set_exit_targets({key: target})
        table = self.panel.table
        self.assertEqual(table.rowCount(), 1)
        self.assertEqual(table.item(0, 2).text(), '2')  # broker aggregate, not split lots
        self.assertIn('mark1 prototype', table.item(0, 11).text())
        self.assertIn('mark1.1 prototype', table.item(0, 11).toolTip())
        self.assertIn('증권사 보유주식의 모델별 배분 아님', table.item(0, 11).toolTip())
        self.assertIn('모델 매수분의 원가는 확인되지', table.item(0, 4).toolTip())
        self.assertEqual([table.item(0, col).text() for col in (9, 10, 12)], ['—'] * 3)
        self.assertIn('자동매도 보류 1종목', self.panel.reconciliation_labels[Market.DOMESTIC].text())
        inst = Instrument(Market.DOMESTIC, '005930', 'KRX')
        self.panel.apply_holding_quote({'instrument': inst, 'watch_id': key,
            'quote': Quote(inst.market, inst.symbol, 'test', inst.exchange, D('111'), 'KRW'), 'targets': target})
        self.assertIn('mark1 prototype', table.item(0, 11).text())
        self.assertIn('자동매도를 보류', table.item(0, 11).toolTip())
        self.assertEqual(table.item(0, 9).text(), '—')
        changed = {**target, 'issues': ('새 체결 근거 확인 필요',)}
        self.panel.apply_holding_quote({'instrument': inst, 'watch_id': key,
            'quote': Quote(inst.market, inst.symbol, 'test', inst.exchange, D('112'), 'KRW'), 'targets': changed})
        self.assertIn('새 체결 근거 확인 필요', self.panel.reconciliation_labels[Market.DOMESTIC].toolTip())

    def test_one_estimated_lot_retains_broker_account_values_but_marks_model_exit(self):
        lot = {**self.lots[0], 'quantity': D(2), 'sellable_quantity': D(2),
               'average_price_basis': 'demo_order_reference'}
        target = {'lots': (lot,), 'reconciled': True}
        key = 'domestic:KRX:005930'
        self.panel.set_exit_targets({key: target})
        table = self.panel.table
        self.assertEqual(table.rowCount(), 1)
        self.assertEqual(table.item(0, 4).text(), '105')  # broker aggregate, not the reference 100
        self.assertEqual(table.item(0, 7).text(), '+10')
        self.assertEqual(table.item(0, 11).text(), 'mark1 prototype · 원가 추정')
        self.assertEqual(table.item(0, 9).text(), '추정 ≥ 101')
        self.assertEqual(table.item(0, 10).text(), '추정 ≤ 99.1')
        self.assertIn('증권사 종목 합산 잔고', table.item(0, 4).toolTip())
        self.assertIn('모의투자 주문 당시 참고 시세', table.item(0, 9).toolTip())
        self.assertIn('다른 앱의 과거 매매가 없었다는 증명은 아닙니다', table.item(0, 9).toolTip())
        self.assertIn('매수 모델: mark1 prototype', table.item(0, 11).toolTip())
        self.assertFalse(self.panel.reconciliation_labels[Market.DOMESTIC].isVisible())
        inst = Instrument(Market.DOMESTIC, '005930', 'KRX')
        self.panel.apply_holding_quote({'instrument': inst, 'watch_id': key,
            'quote': Quote(inst.market, inst.symbol, 'test', inst.exchange, D('111'), 'KRW'), 'targets': target})
        self.assertEqual(table.item(0, 11).text(), 'mark1 prototype · 원가 추정')
        self.assertEqual(table.item(0, 9).text(), '추정 ≥ 101')

    def test_multiple_estimated_lots_mark_split_cost_and_profit_as_estimates(self):
        lots = tuple({**lot, 'average_price_basis': 'demo_order_reference'} for lot in self.lots)
        self.panel.set_exit_targets({'domestic:KRX:005930': {'lots': lots, 'reconciled': True}})
        table = self.panel.table
        self.assertEqual(table.rowCount(), 2)
        self.assertTrue(all('· 원가 추정' in table.item(row, 11).text() for row in range(2)))
        for row in range(2):
            self.assertTrue(table.item(row, 4).text().startswith('추정 '))
            self.assertTrue(table.item(row, 7).text().startswith('추정 '))
            self.assertTrue(table.item(row, 8).text().startswith('추정 '))
            self.assertTrue(table.item(row, 9).text().startswith('추정 ≥'))
            self.assertTrue(table.item(row, 10).text().startswith('추정 ≤'))
            self.assertFalse(table.item(row, 6).text().startswith('추정 '))
            self.assertIn('모델별 가상 분리 표시', table.item(row, 4).toolTip())
            self.assertIn('추정 원가를 사용한 값', table.item(row, 7).toolTip())

    def test_unreconciled_estimate_remains_blocked_with_one_visible_warning(self):
        lot = {**self.lots[0], 'average_price_basis': 'demo_order_reference',
               'quantity_remaining': D(2), 'strategy_id': 'mark1-0-prototype'}
        self.panel.set_exit_targets({'domestic:KRX:005930': {
            'lots': (lot,), 'reconciled': False, 'issues': ('매도 체결수량 대조 실패',)}})
        table = self.panel.table
        self.assertEqual(table.item(0, 11).text(), 'mark1 prototype · 원가 추정')
        self.assertEqual([table.item(0, col).text() for col in (9, 10, 12)], ['—'] * 3)
        self.assertIn('자동매도 보류 1종목', self.panel.reconciliation_labels[Market.DOMESTIC].text())
        self.assertIn('매도 체결수량 대조 실패', table.item(0, 11).toolTip())

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
