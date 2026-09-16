"""KRX common-equity ticks keep automatic quantities and safety checks intact."""
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dockdack.autotrade import AutoTrader
from dockdack.gui_service import Instrument
from dockdack.models import Market
from dockdack.watchlist import TriggerRule, WatchItem, WatchStore
from test_autotrade import FakeTradingService, NOW, position


class DomesticAutomaticPriceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = WatchStore(Path(self.temp.name) / "test.sqlite3")
        self.service = FakeTradingService()
        self.service.prices = [Decimal("310750")]
        self.service.available = Decimal("1000000")
        self.item = WatchItem(Instrument(Market.DOMESTIC, "108490", "KRX"), "로보티즈", 30)
        self.store.save_item(self.item)
        self.engine = AutoTrader(self.service, self.store, clock=lambda: NOW)

    def run_rule(self, side="buy", cap="1000000"):
        if side == "sell":
            self.service.positions = (replace(position(), symbol="108490", name="로보티즈"),)
        rule = TriggerRule.create(self.item, "price_ge", side, 1, Decimal(cap), Decimal("1"))
        self.store.add_rule(rule)
        self.engine.enable_orders("DEMO_AUTOTRADE")
        self.engine.poll()

    def test_buy_rounds_down_and_keeps_one_share_and_raw_reference(self):
        self.run_rule()
        order, = self.service.submitted
        self.assertEqual(order.price, Decimal("310500"))
        self.assertEqual(order.quantity, 1)
        self.assertEqual(Decimal(self.store.attempts()[0]["price"]), Decimal("310750"))
        self.assertTrue(any("310750 → 주문 지정가 310500 KRW" in row["message"]
                            for row in self.store.events(category="order")))

    def test_sell_rounds_up_and_keeps_one_share(self):
        self.run_rule("sell")
        order, = self.service.submitted
        self.assertEqual(order.price, Decimal("311000"))
        self.assertEqual(order.quantity, 1)

    def test_rounded_sell_notional_cannot_exceed_cap(self):
        self.run_rule("sell", "310750")
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.attempts(), ())

    def test_rounded_down_buy_does_not_relax_original_quote_cap(self):
        self.run_rule("buy", "310500")
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.attempts(), ())

    def test_insufficient_funds_remains_blocked(self):
        self.service.available = Decimal("310750")
        self.run_rule()
        self.assertEqual(self.service.submitted, [])

    def test_noncommon_instrument_rejected_before_stock_tick_rounding(self):
        with patch.object(self.service, "ensure_common_equity", side_effect=ValueError("not common equity")), \
                patch("dockdack.autotrade.current_common_equity_limit_price") as rounding:
            self.run_rule()
        rounding.assert_not_called()
        self.assertEqual(self.service.submitted, [])

    def test_post_rounding_request_mismatch_remains_blocked(self):
        original = self.service.prepare
        self.service.prepare = lambda *args: replace(original(*args), price=Decimal("310750"))
        self.run_rule()
        self.assertEqual(self.service.submitted, [])


if __name__ == "__main__":
    unittest.main()
