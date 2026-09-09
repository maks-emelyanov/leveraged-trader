from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_FLOOR, Context, Decimal, DecimalException, localcontext
from math import isfinite

_ALPACA_DOLLAR_PRICE_TICK = Decimal("0.01")
_ALPACA_SUBDOLLAR_PRICE_TICK = Decimal("0.0001")
_ALPACA_MAX_PRICE_ADJUSTED_EXPONENT = 25
_ALPACA_ORDER_QUANTITY_DECIMAL_PLACES = 9


def _isolated_price_context(*values: Decimal) -> Context:
    """Return an ambient-context-independent context large enough for prices.

    Decimal's context is thread-local and application code is free to lower its
    precision or enable traps.  Broker pricing must not inherit either choice.
    The dynamic integral allowance also lets a finite binary64-scale price be
    quantized to cents without the coefficient overflowing the context.
    """
    significant_digits = sum(max(len(value.as_tuple().digits), 1) for value in values)
    product_integral_digits = sum(max(value.adjusted() + 1, 0) for value in values)
    precision = max(64, significant_digits + 8, product_integral_digits + 8)
    return Context(prec=precision)


def _alpaca_price_tick(price: Decimal) -> Decimal:
    return _ALPACA_SUBDOLLAR_PRICE_TICK if price < Decimal("1") else _ALPACA_DOLLAR_PRICE_TICK


def canonical_alpaca_limit_price(value: object, *, field_name: str = "Alpaca limit price") -> float:
    """Return ``value`` only when it is already an exactly serializable Alpaca tick.

    This differs from the directional pricing helpers below: persisted broker
    intent must never be rounded a second time when it is later submitted.
    Rejecting an off-tick stored value keeps the durable intent identical to
    the price serialized on the wire.
    """
    try:
        price = Decimal(str(value))
    except (DecimalException, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive finite number on an Alpaca price tick.") from exc
    if not price.is_finite() or price <= 0 or price.adjusted() > _ALPACA_MAX_PRICE_ADJUSTED_EXPONENT:
        raise ValueError(f"{field_name} must be a positive finite number on an Alpaca price tick.")
    try:
        with localcontext(_isolated_price_context(price)):
            quantized = price.quantize(_alpaca_price_tick(price), rounding=ROUND_FLOOR)
    except DecimalException as exc:
        raise ValueError(f"{field_name} cannot be represented at Alpaca tick precision.") from exc
    if price != quantized:
        raise ValueError(f"{field_name} must already be aligned to an Alpaca price tick.")
    try:
        result = float(price)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field_name} cannot be represented at Alpaca tick precision.") from exc
    if not isfinite(result) or Decimal(str(result)) != price:
        raise ValueError(f"{field_name} cannot be represented at Alpaca tick precision.")
    return result


def canonical_alpaca_order_quantity(
    value: object,
    *,
    field_name: str = "Alpaca order quantity",
) -> float:
    """Return a positive quantity only when Alpaca's wire format preserves it exactly."""
    try:
        quantity = Decimal(str(value))
        result = float(quantity)
    except (DecimalException, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a positive finite number.") from exc
    if not quantity.is_finite() or quantity <= 0 or not isfinite(result):
        raise ValueError(f"{field_name} must be a positive finite number.")
    if Decimal(str(result)) != quantity:
        raise ValueError(f"{field_name} must be exactly representable for persistence.")
    formatted = f"{result:.{_ALPACA_ORDER_QUANTITY_DECIMAL_PLACES}f}".rstrip("0").rstrip(".") or "0"
    if Decimal(formatted) != quantity:
        raise ValueError(
            f"{field_name} must already be aligned to Alpaca's "
            f"{_ALPACA_ORDER_QUANTITY_DECIMAL_PLACES}-decimal quantity precision."
        )
    return result


def target_sell_price(filled_avg_price: float, profit_target_multiple: float) -> float:
    """Return Alpaca's upward-tick-rounded target using decimal input semantics."""
    try:
        entry = Decimal(str(filled_avg_price))
        multiple = Decimal(str(profit_target_multiple))
    except DecimalException as exc:
        raise ValueError("Target price inputs must be numeric.") from exc
    if not entry.is_finite() or entry <= 0:
        raise ValueError("filled_avg_price must be a positive finite number.")
    if not multiple.is_finite() or not Decimal("1") < multiple <= Decimal("100"):
        raise ValueError("profit_target_multiple must be a finite number greater than 1.0 and at most 100.0.")

    try:
        with localcontext(_isolated_price_context(entry, multiple)):
            target = entry * multiple
            if target.adjusted() > _ALPACA_MAX_PRICE_ADJUSTED_EXPONENT:
                raise ValueError("Target price cannot be represented at Alpaca tick precision.")
            quantized = target.quantize(_alpaca_price_tick(target), rounding=ROUND_CEILING)
            if quantized >= Decimal("1"):
                quantized = quantized.quantize(_ALPACA_DOLLAR_PRICE_TICK, rounding=ROUND_CEILING)
            result = float(quantized)
    except (DecimalException, OverflowError) as exc:
        raise ValueError("Target price cannot be represented at Alpaca tick precision.") from exc
    if not isfinite(result):
        raise ValueError("Target price cannot be represented as a finite Alpaca limit price.")
    if Decimal(str(result)) != quantized:
        raise ValueError("Target price cannot be represented at Alpaca tick precision.")
    return result


def buffered_buy_limit_price(latest_market_price: float, buy_limit_buffer_bps: float) -> float:
    """Return Alpaca's downward-tick-rounded protected buy limit."""
    try:
        market_price = Decimal(str(latest_market_price))
        buffer_bps = Decimal(str(buy_limit_buffer_bps))
    except DecimalException as exc:
        raise ValueError("Buy limit price inputs must be numeric.") from exc
    if not market_price.is_finite() or market_price <= 0:
        raise ValueError("latest_market_price must be a positive finite number.")
    if not buffer_bps.is_finite() or not Decimal("0") <= buffer_bps <= Decimal("10000"):
        raise ValueError("buy_limit_buffer_bps must be a finite number between 0 and 10000.")

    try:
        with localcontext(_isolated_price_context(market_price, buffer_bps)):
            protected = market_price * (Decimal("1") + buffer_bps / Decimal("10000"))
            if protected.adjusted() > _ALPACA_MAX_PRICE_ADJUSTED_EXPONENT:
                raise ValueError("Buy limit price cannot be represented at Alpaca tick precision.")
            quantized = protected.quantize(_alpaca_price_tick(protected), rounding=ROUND_FLOOR)
            if quantized >= Decimal("1"):
                quantized = quantized.quantize(_ALPACA_DOLLAR_PRICE_TICK, rounding=ROUND_FLOOR)
            result = float(quantized)
    except (DecimalException, OverflowError) as exc:
        raise ValueError("Buy limit price cannot be represented at Alpaca tick precision.") from exc
    if not isfinite(result) or result <= 0:
        raise ValueError("Buy limit price cannot be represented as a positive finite Alpaca limit price.")
    if Decimal(str(result)) != quantized:
        raise ValueError("Buy limit price cannot be represented at Alpaca tick precision.")
    return result
