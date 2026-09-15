"""Opt-in demo plumbing test: 10% entry, +1% profit / -0.8% loss exits.

This is not an investment strategy. Publishing does not arm or submit orders.
"""

import json
import random
from datetime import timedelta
from decimal import Decimal, ROUND_CEILING
from uuid import NAMESPACE_URL, uuid5

from dockdack.gui_service import Instrument
from dockdack.history import regular_session
from dockdack.models import Market, TradingMode
from dockdack.signal_bridge import atomic_json, timestamp


class RandomDemoSignals:
    def __init__(self, service, store, policy, path, *, clock, quantity=1, us_order_type="blocked", draw=None):
        self._require_demo(service, store)
        if type(quantity) is not int or not 1 <= quantity <= policy.max_quantity:
            raise ValueError("테스트 수량은 정책 한도 안의 정수여야 합니다.")
        if us_order_type not in {"blocked", "limit"}:
            raise ValueError("미국 모의 주문은 차단 또는 현재가 지정가만 가능합니다.")
        self.service, self.store, self.policy, self.path = service, store, policy, path
        self.clock, self.quantity, self.us_order_type = clock, quantity, us_order_type
        self.draw = draw or random.SystemRandom().random
        self.accounts = {}

    @staticmethod
    def _require_demo(service, store):
        if (TradingMode(getattr(service, "mode", TradingMode.DEMO)) is not TradingMode.DEMO or
                TradingMode(getattr(store, "mode", TradingMode.DEMO)) is not TradingMode.DEMO):
            raise ValueError("내장 무작위 테스트 신호기는 모의투자 전용입니다. 실전에서는 사용할 수 없습니다.")

    def _account(self, inst):
        self._require_demo(self.service, self.store)
        now = self.clock()
        cached = self.accounts.get(inst.market)
        if cached and 0 <= (now-cached[0]).total_seconds() < 30:
            return cached[1]
        account = self.service.safety_account(inst)
        if account.market is not inst.market or account.currency != inst.currency:
            raise ValueError("테스트 신호기 잔고의 시장/통화가 다릅니다.")
        for p in account.positions:
            if p.market is not inst.market or p.currency != inst.currency or any(not v.is_finite() or v < 0 for v in (p.quantity, p.sellable_quantity, p.average_price)):
                raise ValueError("테스트 신호기가 보유 수량/매입가를 확인할 수 없습니다.")
            if p.sellable_quantity > p.quantity or (p.quantity > 0 and p.average_price <= 0):
                raise ValueError("테스트 신호기에 유효한 평균 매입가와 매도가능 수량이 필요합니다.")
        self.accounts[inst.market] = (self.clock(), account)
        return account

    def publish(self, chart):
        self._require_demo(self.service, self.store)
        entries = []
        for stock in chart["stocks"]:
            self._require_demo(self.service, self.store)
            with self.store.connection() as db:
                existing = db.execute("SELECT payload FROM test_decisions WHERE export_id=? AND watch_id=?", (chart["export_id"], stock["watch_id"])).fetchone()
            if existing:
                entries.append(json.loads(existing[0]))
                continue
            now = self.clock()
            inst = Instrument(Market(stock["market"]), stock["symbol"], stock["exchange"])
            entry = {"signal_id": uuid5(NAMESPACE_URL, f"random-demo:{chart['export_id']}:{stock['watch_id']}").hex,
                     "export_id": chart["export_id"], "market": inst.market.value, "symbol": inst.symbol,
                     "exchange": inst.exchange, "action": "hold", "generated_at": now.isoformat(),
                     "expires_at": (now+timedelta(minutes=2)).isoformat()}
            supported = inst.market is Market.DOMESTIC or self.us_order_type == "limit"
            if stock.get("status") != "ok":
                continue  # No valid export membership; never manufacture a signal.
            if supported and self.policy.cap(inst.market) > 0 and regular_session(inst.market, now) and not self.store.attempts(stock["watch_id"], pending_only=True):
                self.service.ensure_common_equity(inst)
                account = self._account(inst)
                now = self.clock()
                age = (now-timestamp(stock["quote_fetched_at"], "quote_fetched_at")).total_seconds()
                price = Decimal(stock["price"])
                holdings = [p for p in account.positions if p.symbol == inst.symbol and p.quantity > 0]
                total = sum((p.quantity for p in holdings), Decimal(0))
                action, floor, exit_kind = "hold", None, None
                if 0 <= age <= 15 and price.is_finite() and price > 0 and price*self.quantity <= self.policy.cap(inst.market):
                    if total:
                        average = sum((p.average_price*p.quantity for p in holdings), Decimal(0))/total
                        if sum((p.sellable_quantity for p in holdings), Decimal(0)) >= self.quantity:
                            if price >= average*Decimal("1.01"):
                                action, floor, exit_kind = "sell", average*Decimal("1.01"), "profit"
                            elif price <= average*Decimal("0.992"):
                                action, exit_kind = "sell", "loss"
                    elif self.draw() < 0.1:
                        action = "buy"
                if action != "hold":
                    entry.update(action=action, quantity=self.quantity, max_notional=str(self.policy.cap(inst.market)),
                                 order_type="market" if inst.market is Market.DOMESTIC else "limit")
                    if exit_kind == "profit":
                        entry["min_sell_price"] = format(floor.quantize(Decimal("0.00000001"), rounding=ROUND_CEILING), "f")
                        entry["cost_profit_pct"] = "1"
                    elif exit_kind == "loss":
                        # No profit-price floor on a loss exit. The executor must
                        # confirm this condition using the latest actual cost/quote.
                        entry["cost_loss_pct"] = "0.8"
            encoded = json.dumps(entry, sort_keys=True)
            with self.store.connection() as db:
                db.execute("INSERT OR IGNORE INTO test_decisions VALUES(?,?,?)", (chart["export_id"], stock["watch_id"], encoded))
                # Concurrent repeat publication must use the first stored draw/decision.
                entry = json.loads(db.execute("SELECT payload FROM test_decisions WHERE export_id=? AND watch_id=?", (chart["export_id"], stock["watch_id"])).fetchone()[0])
            entries.append(entry)
        payload = {"schema_version": 1, "source_id": self.policy.source_id, "signals": entries}
        self._require_demo(self.service, self.store)
        atomic_json(self.path, payload)
        return payload
