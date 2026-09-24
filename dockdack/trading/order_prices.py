"""Order-price precision, kept separate from unrounded market quotes.

Kiwoom's US order endpoint accepts cents at $1 or above and four decimal
places below $1. A current-price BUY never raises the quoted limit; a SELL
never lowers it. Explicit user prices are validated, never rounded.
KRX common-equity ticks apply only through the separately scoped helper.
"""

from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR, localcontext

from dockdack.models import Market, OrderSide


def _positive(price: Decimal, label: str) -> Decimal:
    if not isinstance(price, Decimal) or not price.is_finite() or price <= 0:
        raise ValueError(f"{label}은 0보다 큰 유한한 숫자여야 합니다.")
    return price


def _step(price: Decimal) -> Decimal:
    return Decimal("0.01") if price >= 1 else Decimal("0.0001")


def _quantize(price: Decimal, step: Decimal, rounding: str) -> Decimal:
    try:
        with localcontext() as context:
            # Supported API prices fit in 12 characters; do not depend on a
            # caller's reduced Decimal precision or default rounding policy.
            context.prec = max(28, len(price.as_tuple().digits) + 4)
            return price.quantize(step, rounding=rounding)
    except InvalidOperation as exc:
        raise ValueError("미국 주문 가격이 API 입력 범위를 벗어났습니다.") from exc


def validate_us_order_price(price: Decimal, label: str = "미국 주문 가격") -> Decimal:
    """Preserve an explicit numeric price, rejecting sub-tick precision.

    Excess trailing zeroes are removed before serialization/range checks;
    for example Decimal('330.8000') becomes Decimal('330.80'), not a new limit.
    Already-valid representations such as '210.50' and '100' are preserved.
    """
    _positive(price, label)
    step = _step(price)
    rounded = _quantize(price, step, ROUND_FLOOR)
    if rounded != price:
        raise ValueError(f"{label}: $1 이상은 소수점 2자리, $1 미만은 소수점 4자리까지만 가능합니다.")
    return rounded if price.as_tuple().exponent < step.as_tuple().exponent else price


def current_limit_price(market: Market | str, side: OrderSide | str,
                        quote_price: Decimal) -> Decimal:
    """Return a side-conservative executable limit, without changing a quote."""
    selected_market, selected_side = Market(market), OrderSide(side)
    _positive(quote_price, "현재가")
    if selected_market is Market.DOMESTIC:
        return quote_price
    rounding = ROUND_FLOOR if selected_side is OrderSide.BUY else ROUND_CEILING
    price = _quantize(quote_price, _step(quote_price), rounding)
    # A sub-dollar SELL can round up to exactly $1: validate against its new
    # bracket. A sub-tick BUY can round down to zero and must not be sent.
    return validate_us_order_price(quote_price if price == quote_price else price)


def _common_equity_tick(price: Decimal) -> Decimal:
    """KRX common-equity tick grid; not an ETF/ETN or generic instrument rule."""
    for ceiling, tick in ((2000, 1), (5000, 5), (20000, 10), (50000, 50),
                          (200000, 100), (500000, 500)):
        if price < ceiling:
            return Decimal(tick)
    return Decimal(1000)


def _tick_multiple(price: Decimal, tick: Decimal, rounding: str) -> Decimal:
    try:
        with localcontext() as context:
            # Quantizing to Decimal('500') only removes fractional digits;
            # division into tick units is required for an actual multiple.
            # Include integer places for positive-exponent Decimal inputs.
            context.prec = max(28, len(price.as_tuple().digits) + 4, price.adjusted() + 4)
            return (price / tick).to_integral_value(rounding=rounding) * tick
    except InvalidOperation as exc:
        raise ValueError("국내 보통주 주문 가격을 호가 단위로 계산할 수 없습니다.") from exc


def current_common_equity_limit_price(market: Market | str, side: OrderSide | str,
                                      quote_price: Decimal) -> Decimal:
    """Side-conservative limit after the caller verified a common equity.

    This deliberately does not change generic/current manual-price handling.
    Korean ETFs, ETNs and other instruments must not use this stock tick grid.
    """
    selected_market, selected_side = Market(market), OrderSide(side)
    if selected_market is Market.US:
        return current_limit_price(selected_market, selected_side, quote_price)
    price = _positive(quote_price, "현재가")
    rounding = ROUND_FLOOR if selected_side is OrderSide.BUY else ROUND_CEILING
    tick = _common_equity_tick(price)
    # A SELL can cross into the next price band. Re-evaluate the resulting
    # price's tick rather than assuming the original band is still valid.
    for _ in range(7):
        price = _positive(_tick_multiple(price, tick, rounding), "국내 보통주 주문 가격")
        new_tick = _common_equity_tick(price)
        if new_tick == tick:
            return price
        tick = new_tick
    raise ValueError("국내 보통주 주문 가격의 호가 단위를 확인할 수 없습니다.")
