"""Same-market holdings and pending BUY exposure limits, in native currency."""
from decimal import Decimal
from dockdack.execution_policy import account_equity


def _number(value):
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError('합산 매수 한도 계산에 필요한 금액·수량이 올바르지 않습니다.')
    return result


def remaining_buy_budget(store, account, watch_id, source_id, symbol_percent=0, model_percent=0, *, exclude_rule_id=None):
    symbol_percent, model_percent = _number(symbol_percent), _number(model_percent)
    if symbol_percent > 100 or model_percent > 100:
        raise ValueError('합산 비중 한도는 0~100%여야 합니다.')
    if not symbol_percent and not model_percent:
        return None
    equity = account_equity(account)
    positions = {}
    for position in account.positions:
        key = f'{position.market.value}:{position.exchange}:{position.symbol}'
        if key in positions:
            raise ValueError('중복 잔고로 합산 비중을 확정할 수 없습니다.')
        positions[key] = position
    symbol_used = sum((max(_number(p.evaluation_amount), _number(p.current_price) * _number(p.quantity))
                       for key, p in positions.items() if key == watch_id), Decimal(0))
    model_used = Decimal(0)
    checked_inventory = {}
    if model_percent and source_id:
        for lot in store.prototype_lots():
            if not lot['watch_id'].startswith(account.market.value + ':') or _number(lot['quantity_remaining']) == 0:
                continue
            lot_source = lot.get('source_id') or store.prototype_rule_source(lot['buy_rule_id'])
            if lot_source != source_id:
                continue
            position = positions.get(lot['watch_id'])
            if position is None or lot.get('issues') or _number(position.current_price) == 0:
                raise ValueError('모델 보유 장부·현재가 확인 후 합산 비중을 계산할 수 있습니다.')
            if lot['watch_id'] not in checked_inventory:
                checked_inventory[lot['watch_id']] = store.prototype_inventory(
                    lot['watch_id'], position.quantity, position.sellable_quantity)
            inventory = checked_inventory[lot['watch_id']]
            if not inventory['reconciled']:
                raise ValueError('모델 보유 수량 대조 후 합산 비중을 계산할 수 있습니다.')
            model_used += _number(lot['quantity_remaining']) * _number(position.current_price)
    for row in store.order_history(limit=None, market=account.market, side='buy', statuses=('accepted', 'submitting', 'unknown')):
        if row['rule_id'] == exclude_rule_id or row['market'] != account.market.value or row['side'] != 'buy' or row['status'] not in {'accepted', 'submitting', 'unknown'}:
            continue
        remaining = max(Decimal(0), _number(row['quantity']) - _number(row['filled_quantity'] or 0))
        if not remaining:
            continue
        price = _number(row['reference_price'])
        if not price:
            raise ValueError('미체결 예약 가격을 확인해야 합산 비중을 계산할 수 있습니다.')
        reserved = remaining * price * Decimal('1.01')
        if row['watch_id'] == watch_id:
            symbol_used += reserved
        if source_id and store.prototype_rule_source(row['rule_id']) == source_id:
            model_used += reserved
    budgets = []
    if symbol_percent:
        budgets.append(equity * symbol_percent / 100 - symbol_used)
    if model_percent and source_id:
        budgets.append(equity * model_percent / 100 - model_used)
    return max(Decimal(0), min(budgets)) if budgets else None
