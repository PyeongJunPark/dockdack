"""US current-quote price steps keep share quantity and safeguards intact."""
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from dockdack.autotrade import AutoTrader
from dockdack.gui_service import Instrument
from dockdack.models import Market
from dockdack.watchlist import TriggerRule, WatchItem, WatchStore
from test_autotrade import FakeTradingService, position


class USAutomaticPriceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = WatchStore(Path(self.temp.name) / 'test.sqlite3')
        self.service = FakeTradingService()
        self.service.prices = [Decimal('330.8003')]
        self.item = WatchItem(Instrument(Market.US, 'AAPL', 'ND'), 'Apple', 30)
        self.store.save_item(self.item)
        self.engine = AutoTrader(self.service, self.store,
                                 clock=lambda: datetime(2026, 9, 14, 14, tzinfo=timezone.utc))

    def run_rule(self, side='buy', cap='10000'):
        if side == 'sell':
            self.service.positions = (replace(position(), market=Market.US, symbol='AAPL', exchange='ND', currency='USD'),)
        rule = TriggerRule.create(self.item, 'price_ge', side, 1, Decimal(cap), Decimal('0.00001'))
        self.store.add_rule(rule)
        self.engine.enable_orders('DEMO_AUTOTRADE')
        self.engine.poll()

    def test_buy_down_rounds_only_price_not_one_share_or_reference_quote(self):
        self.run_rule()
        order, = self.service.submitted
        self.assertEqual(order.price, Decimal('330.80'))
        self.assertEqual(order.quantity, 1)
        self.assertEqual(Decimal(self.store.attempts()[0]['price']), Decimal('330.8003'))
        self.assertTrue(any('330.8003 → 주문 지정가 330.80 USD' in row['message']
                            for row in self.store.events(category='order')))

    def test_sell_rounds_up_without_lowering_quote_limit(self):
        self.run_rule('sell')
        self.assertEqual(self.service.submitted[0].price, Decimal('330.81'))
        self.assertEqual(self.service.submitted[0].quantity, 1)

    def test_rounded_sell_notional_cannot_exceed_cap(self):
        self.run_rule('sell', '330.8003')
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.attempts(), ())

    def test_below_one_dollar_buy_uses_four_decimals(self):
        self.service.prices = [Decimal('0.123456')]
        self.run_rule()
        self.assertEqual(self.service.submitted[0].price, Decimal('0.1234'))

    def test_post_rounding_mismatch_from_service_still_blocked(self):
        original = self.service.prepare
        self.service.prepare = lambda *args: replace(original(*args), price=Decimal('330.81'))
        self.run_rule()
        self.assertEqual(self.service.submitted, [])

    def test_rounding_does_not_relax_funds_check(self):
        self.service.available = Decimal('100')
        self.run_rule()
        self.assertEqual(self.service.submitted, [])


if __name__ == '__main__':
    unittest.main()
