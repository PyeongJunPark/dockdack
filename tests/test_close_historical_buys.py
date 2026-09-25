"""Historical DEMO BUY records must not strand verified closing inventory."""

from datetime import timedelta
from decimal import Decimal
from unittest import TestCase
from unittest.mock import patch

from dockdack.models import Market, OpenOrder, OrderSide
from dockdack.watchlist import TriggerRule, WatchItem
import test_lstm30_close as close_tests


class HistoricalBuyCloseTests(TestCase):
    setUp = close_tests.CloseLiquidatorTests.setUp
    arm = close_tests.CloseLiquidatorTests.arm
    add_holding = close_tests.CloseLiquidatorTests.add_holding
    liquidator = close_tests.CloseLiquidatorTests.liquidator
    close_attempts = close_tests.CloseLiquidatorTests.close_attempts

    def seed(self, *, side="buy", status="accepted", at=None):
        item = WatchItem(self.service.resolve("HELD"))
        self.store.save_item(item)
        rule = TriggerRule.create(item, "price_ge", side, 1, Decimal(1000), Decimal(100))
        self.store.add_rule(rule)
        self.assertTrue(self.store.claim(rule, Decimal(100), at or self.now-timedelta(days=1)))
        if status != "submitting":
            self.store.finish(rule.id, status, "original record", "OLD-ORDER")
        return rule

    def run_close(self):
        close = self.liquidator()
        self.arm()
        close.tick()
        return close

    def test_old_buy_kept_and_full_sellable_closed_only_once(self):
        buy = self.seed()
        original = self.store.attempts()[0]
        self.add_holding(quantity=5, sellable=4)
        close = self.run_close()
        close.tick()
        self.liquidator().tick()  # Restart does not submit the close again.
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual((self.service.submitted[0].side, self.service.submitted[0].quantity), (OrderSide.SELL, 4))
        self.assertEqual(self.store.attempts()[0], original)
        self.assertEqual(self.close_attempts()[0]["status"], "accepted")
        self.assertEqual(self.store.historical_buy_attempt_ids(buy.watch_id, self.now), frozenset({buy.id}))

    def test_today_buy_still_blocks_close(self):
        self.seed(at=self.now-timedelta(minutes=1))
        self.add_holding()
        self.run_close()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.close_attempts(), ())

    def test_old_sell_still_blocks_close(self):
        self.seed(side="sell")
        self.add_holding()
        self.run_close()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.close_attempts(), ())

    def test_unknown_buy_still_blocks_close(self):
        self.seed(status="unknown")
        self.add_holding()
        self.engine.isolated_symbol_errors = True
        self.run_close()
        self.assertEqual(self.service.submitted, [])
        self.assertFalse(self.engine.orders_enabled)

    def test_broker_open_order_still_blocks_even_when_old_buy_eligible(self):
        self.seed()
        self.add_holding()
        self.service.open_orders = (OpenOrder(Market.US, "OLD-ORDER", "HELD", "", "ND", "buy",
            "accepted", Decimal(1), Decimal(0), Decimal(1), Decimal(100)),)
        self.run_close()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.close_attempts(), ())

    def test_sellable_zero_still_blocks_close(self):
        self.seed()
        self.add_holding(sellable=0)
        self.run_close()
        self.assertEqual(self.service.submitted, [])

    def test_old_buy_status_changed_before_atomic_claim_blocks_close(self):
        buy = self.seed()
        self.add_holding()
        close = self.liquidator()
        original = close._claim

        def changed(*args, **kwargs):
            with self.store.connection() as db:
                db.execute("UPDATE attempts SET status='unknown' WHERE rule_id=?", (buy.id,))
            return original(*args, **kwargs)

        self.arm()
        with patch.object(close, "_claim", side_effect=changed):
            close.tick()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.close_attempts(), ())

    def test_old_buy_status_changed_during_pacing_blocks_transport(self):
        buy = self.seed()
        self.add_holding()

        def changed():
            with self.store.connection() as db:
                db.execute("UPDATE attempts SET status='unknown' WHERE rule_id=?", (buy.id,))

        self.service.before_send = changed
        self.run_close()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.close_attempts()[0]["status"], "not_sent")

    def test_slow_open_order_check_cannot_hide_behind_fresh_account_and_quote(self):
        self.seed()
        self.add_holding()

        def delayed(instrument):
            self.now += timedelta(seconds=16)
            return ()

        with patch.object(self.service, "safety_orders", side_effect=delayed):
            self.run_close()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.close_attempts()[0]["status"], "not_sent")

    def test_off_during_pacing_still_blocks_transport(self):
        self.seed()
        self.add_holding()
        self.service.before_send = self.engine.disarm
        self.run_close()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.close_attempts()[0]["status"], "not_sent")
