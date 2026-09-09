from __future__ import annotations

import math

# Quantities close to zero need an absolute floor, while ordinary and large
# positions need a relative allowance for floating-point subtraction.  Keep
# this policy shared by state transitions and reporting so a position that can
# be closed is not later omitted from realized-P/L totals.
MANAGED_QUANTITY_ABSOLUTE_TOLERANCE = 1e-12
MANAGED_QUANTITY_RELATIVE_TOLERANCE = 1e-8
# A scale-aware tolerance must never grow large enough to hide an economically
# meaningful whole-share imbalance.  A cap just below half a share leaves ample
# room for floating-point noise in fractional accounting while ensuring that a
# half-share or whole-share residual is always observable.
MANAGED_QUANTITY_MAX_TOLERANCE = 0.499999999

# Monetary reconciliation is only intended to absorb the few ULPs introduced
# by adding the same finite fills in a different order.  The hard currency cap
# prevents a large notional from turning that floating-point allowance into an
# economically meaningful discrepancy.
MANAGED_VALUE_MAX_RECONCILIATION_TOLERANCE = 0.00005

# A residual position may be treated as zero only when both its quantity and
# marked value are indistinguishable from floating-point roundoff.  This
# relative bound is at most roughly 64 binary64 ULPs (depending on the
# significand) and remains subject to the hard currency cap above.  The
# explicit subnormal floor lets the Python and SQLite implementations agree at
# the bottom of the binary64 range.
MANAGED_NOTIONAL_RELATIVE_TOLERANCE = 32.0 * (2.0**-52)
MANAGED_NOTIONAL_MIN_ULP_ALLOWANCE = 32.0 * math.ulp(0.0)


def managed_quantity_tolerance(quantity_scale: float) -> float:
    """Return the completion tolerance for a non-negative quantity scale."""
    return min(
        MANAGED_QUANTITY_MAX_TOLERANCE,
        MANAGED_QUANTITY_ABSOLUTE_TOLERANCE + (MANAGED_QUANTITY_RELATIVE_TOLERANCE * abs(float(quantity_scale))),
    )


def managed_value_reconciliation_tolerance(value_scale: float) -> float:
    """Return the bounded roundoff allowance for a finite monetary value."""
    numeric_scale = abs(float(value_scale))
    if not math.isfinite(numeric_scale):
        return 0.0
    return min(
        MANAGED_VALUE_MAX_RECONCILIATION_TOLERANCE,
        32.0 * math.ulp(numeric_scale),
    )


def managed_residual_notional_tolerance(value_scale: float) -> float:
    """Return the bounded monetary allowance for a residual position."""
    numeric_scale = abs(float(value_scale))
    if not math.isfinite(numeric_scale):
        return 0.0
    return min(
        MANAGED_VALUE_MAX_RECONCILIATION_TOLERANCE,
        max(
            MANAGED_NOTIONAL_MIN_ULP_ALLOWANCE,
            MANAGED_NOTIONAL_RELATIVE_TOLERANCE * numeric_scale,
        ),
    )


def managed_residual_quantity_is_negligible(
    residual_quantity: float,
    *,
    quantity_scale: float,
    mark_price: float,
    value_scale: float,
) -> bool:
    """Whether a quantity residual is negligible in shares *and* currency."""
    try:
        quantity = abs(float(residual_quantity))
        normalized_quantity_scale = abs(float(quantity_scale))
        price = float(mark_price)
        normalized_value_scale = abs(float(value_scale))
    except (TypeError, ValueError, OverflowError):
        return False
    if quantity == 0.0:
        return True
    if (
        not math.isfinite(quantity)
        or not math.isfinite(normalized_quantity_scale)
        or not math.isfinite(price)
        or price <= 0.0
        or not math.isfinite(normalized_value_scale)
        or quantity > managed_quantity_tolerance(normalized_quantity_scale)
    ):
        return False
    residual_notional = quantity * price
    if not math.isfinite(residual_notional):
        return False
    return residual_notional <= managed_residual_notional_tolerance(max(normalized_value_scale, residual_notional))
