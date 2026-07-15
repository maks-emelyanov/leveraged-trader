from __future__ import annotations

import re
import sqlite3
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, InvalidOperation
from math import floor, isfinite
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf
from requests import HTTPError

from .config import AlpacaOrderConfig
from .storage import (
    active_alpaca_managed_symbols,
    attach_alpaca_managed_sell_order_if_current,
    claim_alpaca_managed_sell_renewal,
    claim_alpaca_managed_sell_replacement,
    claim_alpaca_managed_sell_submission_retry,
    claim_alpaca_managed_buy_intent,
    close_alpaca_managed_position,
    close_alpaca_managed_position_if_current_and_complete,
    fail_alpaca_managed_buy_submission_if_pending,
    load_alpaca_managed_positions,
    mark_alpaca_managed_buy_filled,
    mark_alpaca_managed_sell_filled,
    mark_alpaca_managed_sell_filled_if_current,
    migrate_alpaca_managed_position_symbol,
    record_alpaca_managed_sell_order,
    save_alpaca_managed_buy_order,
    SellFillQuantityRegressionError,
    update_alpaca_managed_buy_status,
    update_alpaca_managed_sell_status_if_current,
)

BUY_TERMINAL_STATUSES = {"canceled", "done_for_day", "expired", "rejected", "stopped", "suspended"}
SELL_INACTIVE_STATUSES = {"canceled", "done_for_day", "expired", "rejected", "stopped", "suspended"}
SELL_REPLACEMENT_SAFE_STATUSES = {"canceled", "expired"}
SELL_RENEWABLE_STATUSES = {"accepted", "accepted_for_bidding", "new", "partially_filled", "pending_new"}
SELL_CANCEL_PENDING_STATUSES = {"pending_cancel"}
_NEW_YORK = ZoneInfo("America/New_York")
_MANAGED_QTY_TOLERANCE = 1e-8
_BUY_SUBMISSION_VISIBILITY_LEASE = timedelta(minutes=5)
_SELL_RENEWAL_CLAIM_LEASE = timedelta(minutes=5)
_SELL_SUBMISSION_RETRY_CLAIM_LEASE = timedelta(minutes=5)
_ALPACA_DOLLAR_PRICE_TICK = Decimal("0.01")
_ALPACA_SUBDOLLAR_PRICE_TICK = Decimal("0.0001")
_BUY_BATCH_CASH_FRACTION_PER_ELIGIBLE_SIGNAL = 0.05


class SellFillMetadataError(ValueError):
    def __init__(self, observed_qty: float) -> None:
        super().__init__("filled Alpaca sell order has invalid fill metadata")
        self.observed_qty = observed_qty


class SellOrderIdentityError(ValueError):
    pass
_BUY_BATCH_CASH_FRACTION_MAX = 0.50


def _alpaca_headers(cfg: AlpacaOrderConfig) -> dict[str, str]:
    if not cfg.api_key_id or not cfg.api_secret_key:
        raise ValueError(
            "Alpaca paper order submission requires ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY, "
            "or matching CLI arguments."
        )
    if (
        cfg.api_key_id == "your_alpaca_paper_api_key_id"
        or cfg.api_secret_key == "your_alpaca_paper_api_secret_key"
    ):
        raise ValueError("Replace the placeholder Alpaca credentials in .env before submitting orders.")
    return {
        "APCA-API-KEY-ID": cfg.api_key_id,
        "APCA-API-SECRET-KEY": cfg.api_secret_key,
        "Content-Type": "application/json",
    }


def _alpaca_client_order_id(side: str, symbol: str, signal_date: str) -> str:
    safe_symbol = re.sub(r"[^A-Za-z0-9-]", "-", symbol.upper())
    safe_date = re.sub(r"[^0-9]", "", signal_date)
    return f"rsi-{side}-{safe_symbol}-{safe_date}"


def _alpaca_exit_client_order_id(symbol: str, position_id: int, renewal_count: int = 0) -> str:
    safe_symbol = re.sub(r"[^A-Za-z0-9-]", "-", symbol.upper())
    base_id = f"rsi-exit-{safe_symbol}-{position_id}"
    if renewal_count <= 0:
        return base_id
    return f"{base_id}-r{renewal_count}"


def _alpaca_exit_client_order_generation(symbol: str, position_id: int, client_order_id: str) -> int | None:
    base_id = _alpaca_exit_client_order_id(symbol, position_id)
    if client_order_id == base_id:
        return 0
    match = re.fullmatch(rf"{re.escape(base_id)}-r([1-9][0-9]*)", client_order_id)
    if match is not None:
        return int(match.group(1))
    # Existing client IDs retain the ticker that was current when the order was
    # submitted.  Position ID and renewal suffix remain stable across renames.
    alias_match = re.fullmatch(rf"rsi-exit-.+-{position_id}(?:-r([1-9][0-9]*))?", client_order_id)
    if alias_match is None:
        return None
    return 0 if alias_match.group(1) is None else int(alias_match.group(1))


def _format_order_number(value: float, places: int) -> str:
    formatted = f"{value:.{places}f}".rstrip("0").rstrip(".")
    return formatted or "0"


def _alpaca_limit_price_tick(price: Decimal) -> Decimal:
    return _ALPACA_SUBDOLLAR_PRICE_TICK if price < Decimal("1") else _ALPACA_DOLLAR_PRICE_TICK


def _quantize_alpaca_limit_price(value: float | Decimal, *, rounding: str) -> Decimal:
    """Round a positive price to Alpaca's valid limit-price increment."""
    try:
        price = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"Alpaca limit price must be numeric; got {value!r}.") from exc
    if not price.is_finite() or price <= 0:
        raise ValueError(f"Alpaca limit price must be positive; got {value!r}.")
    quantized = price.quantize(_alpaca_limit_price_tick(price), rounding=rounding)
    # A sub-dollar value can round upward across $1.  Re-quantize it using the
    # dollar-or-more tick so the serialized order is valid at the boundary.
    if quantized >= Decimal("1"):
        quantized = quantized.quantize(_ALPACA_DOLLAR_PRICE_TICK, rounding=rounding)
    return quantized


def _format_alpaca_limit_price(value: float | Decimal, *, rounding: str) -> str:
    quantized = _quantize_alpaca_limit_price(value, rounding=rounding)
    formatted = format(quantized, "f").rstrip("0").rstrip(".")
    return formatted or "0"


def _target_sell_price(filled_avg_price: float, profit_target_multiple: float) -> float:
    target = Decimal(str(filled_avg_price)) * Decimal(str(profit_target_multiple))
    return float(_quantize_alpaca_limit_price(target, rounding=ROUND_CEILING))


def _alpaca_limit_price_tolerance(value: float) -> float:
    return float(_alpaca_limit_price_tick(Decimal(str(value))) / Decimal("2"))


def _is_whole_share_qty(value: float) -> bool:
    try:
        qty = Decimal(str(value))
    except InvalidOperation:
        return False
    return qty == qty.to_integral_value()


def _alpaca_dynamic_batch_cash_fraction(eligible_buy_signal_count: int) -> float:
    if eligible_buy_signal_count < 1:
        raise ValueError(f"Eligible buy signal count must be positive; got {eligible_buy_signal_count}.")
    return min(
        eligible_buy_signal_count * _BUY_BATCH_CASH_FRACTION_PER_ELIGIBLE_SIGNAL,
        _BUY_BATCH_CASH_FRACTION_MAX,
    )


def _alpaca_cash_notional(
    cfg: AlpacaOrderConfig,
    headers: dict[str, str],
    *,
    eligible_buy_signal_count: int,
) -> float:
    base_url = cfg.base_url.rstrip("/")
    account_resp = requests.get(
        f"{base_url}/v2/account",
        headers=headers,
        timeout=cfg.timeout_seconds,
    )
    account_resp.raise_for_status()
    account = account_resp.json()
    cash = float(account["cash"])
    cash_fraction = _alpaca_dynamic_batch_cash_fraction(eligible_buy_signal_count)
    notional = round(cash * cash_fraction, 2)
    if notional <= 0:
        raise ValueError(f"Alpaca account cash must be positive to submit buy orders; got {cash}.")
    return notional


def _latest_market_price(symbol: str) -> float:
    ticker = yf.Ticker(symbol)
    fast_info = getattr(ticker, "fast_info", None)
    if fast_info is not None:
        try:
            price = fast_info.get("last_price")
        except AttributeError:
            price = getattr(fast_info, "last_price", None)
        if price is not None and float(price) > 0:
            return float(price)

    history = ticker.history(period="5d", interval="1d", auto_adjust=True)
    if history.empty or "Close" not in history:
        raise ValueError(f"Could not determine latest market price for {symbol}.")

    close = history["Close"].dropna()
    if close.empty or float(close.iloc[-1]) <= 0:
        raise ValueError(f"Could not determine latest market price for {symbol}.")
    return float(close.iloc[-1])


def _alpaca_position_qty(symbol: str, cfg: AlpacaOrderConfig, headers: dict[str, str]) -> float:
    base_url = cfg.base_url.rstrip("/")
    position_resp = requests.get(
        f"{base_url}/v2/positions/{symbol}",
        headers=headers,
        timeout=cfg.timeout_seconds,
    )
    if position_resp.status_code == 404:
        return 0.0
    position_resp.raise_for_status()
    position = position_resp.json()
    return float(position["qty"])


def _alpaca_asset(symbol: str, cfg: AlpacaOrderConfig, headers: dict[str, str]) -> dict:
    base_url = cfg.base_url.rstrip("/")
    asset_resp = requests.get(
        f"{base_url}/v2/assets/{symbol}",
        headers=headers,
        timeout=cfg.timeout_seconds,
    )
    asset_resp.raise_for_status()
    return asset_resp.json()


def _alpaca_has_open_order(
    symbol: str,
    side: str,
    cfg: AlpacaOrderConfig,
    headers: dict[str, str],
) -> bool:
    return _orders_have_open_order(_alpaca_open_orders(cfg, headers), symbol, side)


def _alpaca_open_orders(cfg: AlpacaOrderConfig, headers: dict[str, str]) -> list[dict]:
    base_url = cfg.base_url.rstrip("/")
    orders_resp = requests.get(
        f"{base_url}/v2/orders",
        headers=headers,
        params={
            "status": "open",
            "limit": 500,
            "direction": "desc",
        },
        timeout=cfg.timeout_seconds,
    )
    orders_resp.raise_for_status()
    return orders_resp.json()


def _orders_have_open_order(orders: list[dict], symbol: str, side: str) -> bool:
    symbol = symbol.upper()
    side = side.lower()
    for order in orders:
        if str(order.get("symbol", "")).upper() == symbol and str(order.get("side", "")).lower() == side:
            return True
    return False


def _matching_managed_open_sell_orders(
    orders: list[dict],
    symbol: str,
    position_id: int,
    *,
    expected_alpaca_asset_id: str | None = None,
    excluded_client_order_id: str | None = None,
    excluded_alpaca_order_id: str | None = None,
) -> list[dict]:
    symbol = symbol.upper()
    matches: list[dict] = []
    for order in orders:
        if str(order.get("side", "")).lower() != "sell":
            continue
        order_asset_id = _optional_str(order.get("asset_id"))
        if expected_alpaca_asset_id is not None and order_asset_id is not None:
            identity_matches = order_asset_id == expected_alpaca_asset_id
        else:
            identity_matches = str(order.get("symbol", "")).upper() == symbol
        if not identity_matches:
            continue
        client_order_id = _optional_str(order.get("client_order_id"))
        alpaca_order_id = _optional_str(order.get("id"))
        if (
            client_order_id == excluded_client_order_id
            or (excluded_alpaca_order_id is not None and alpaca_order_id == excluded_alpaca_order_id)
        ):
            continue
        if client_order_id is not None and _alpaca_exit_client_order_generation(
            symbol, position_id, client_order_id
        ) is not None:
            matches.append(order)
    return matches


def _optional_float(value: object) -> float | None:
    if value is None or value == "" or pd.isna(value):
        return None
    return float(value)


def _optional_positive_float(value: object) -> float | None:
    try:
        parsed = _optional_float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed is not None and isfinite(parsed) and parsed > 0 else None


def _optional_str(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: object) -> int:
    if value is None or value == "" or pd.isna(value):
        return 0
    return int(value)


def _signal_workflow(signal: dict) -> str | None:
    return _optional_str(signal.get("Workflow"))


def _managed_remaining_qty(position: dict, buy_filled_qty: float) -> float:
    sold_qty = _optional_float(position.get("sold_qty")) or 0.0
    return float(buy_filled_qty) - sold_qty


def _managed_sell_is_complete(remaining_qty: float) -> bool:
    return abs(remaining_qty) <= _MANAGED_QTY_TOLERANCE


def _sell_fill_metadata(sell_order: dict) -> tuple[float, float | None, bool]:
    raw_filled_qty = sell_order.get("filled_qty")
    if raw_filled_qty is None or raw_filled_qty == "":
        filled_qty = 0.0
    else:
        try:
            filled_qty = float(raw_filled_qty)
        except (TypeError, ValueError):
            return 0.0, None, False
        if not isfinite(filled_qty) or filled_qty < 0:
            return 0.0, None, False

    if filled_qty <= _MANAGED_QTY_TOLERANCE:
        return filled_qty, None, True
    filled_avg_price = _optional_positive_float(sell_order.get("filled_avg_price"))
    return filled_qty, filled_avg_price, filled_avg_price is not None


def _sell_order_identity_is_consistent(
    sell_order: dict,
    *,
    expected_sell_client_order_id: str,
    expected_sell_alpaca_order_id: str | None = None,
) -> bool:
    observed_client_order_id = _optional_str(sell_order.get("client_order_id"))
    observed_alpaca_order_id = _optional_str(sell_order.get("id"))
    return not (
        (observed_client_order_id is not None and observed_client_order_id != expected_sell_client_order_id)
        or (
            expected_sell_alpaca_order_id is not None
            and observed_alpaca_order_id is not None
            and observed_alpaca_order_id != expected_sell_alpaca_order_id
        )
    )


def _sell_order_matches_submission_intent(
    sell_order: dict,
    *,
    requested_qty: float,
    target_sell_price: float,
) -> bool:
    raw_qty = sell_order.get("qty")
    if raw_qty is not None and raw_qty != "":
        observed_qty = _optional_positive_float(raw_qty)
        if observed_qty is None or abs(observed_qty - requested_qty) > _MANAGED_QTY_TOLERANCE:
            return False
    raw_limit_price = sell_order.get("limit_price")
    if raw_limit_price is not None and raw_limit_price != "":
        observed_limit_price = _optional_positive_float(raw_limit_price)
        if observed_limit_price is None or (
            abs(observed_limit_price - target_sell_price)
            > _alpaca_limit_price_tolerance(target_sell_price)
        ):
            return False
    return True


def _append_incomplete_sell_fill_result(
    *,
    conn: sqlite3.Connection,
    rows: list[dict],
    position_id: int,
    symbol: str,
    buy_client_order_id: str,
    sell_client_order_id: str | None,
    observed_sell_qty: float,
    target_sell_price: float,
    sell_alpaca_order_id: str | None,
) -> bool:
    status_updated = bool(sell_client_order_id) and update_alpaca_managed_sell_status_if_current(
        conn,
        position_id,
        expected_sell_client_order_id=sell_client_order_id or "",
        sell_status="incomplete_fill_metadata",
        notes=(
            "Alpaca reported a partial managed sell fill without a valid average fill price; "
            "automatic renewal is blocked pending manual review"
        ),
    )
    _append_reconciliation_result(
        rows,
        position_id=position_id,
        symbol=symbol,
        action="sell",
        status="incomplete_fill_metadata" if status_updated else "superseded",
        buy_client_order_id=buy_client_order_id,
        sell_client_order_id=sell_client_order_id,
        qty=observed_sell_qty,
        limit_price=target_sell_price,
        alpaca_order_id=sell_alpaca_order_id,
        message=(
            "managed sell fill is missing a valid average price; automatic renewal is blocked for review"
            if status_updated
            else "fill observation belongs to a superseded managed GTC sell generation"
        ),
    )
    return status_updated


def _append_incomplete_sell_order_metadata_result(
    *,
    conn: sqlite3.Connection,
    rows: list[dict],
    position_id: int,
    symbol: str,
    buy_client_order_id: str,
    sell_client_order_id: str,
    target_sell_price: float,
    sell_alpaca_order_id: str | None,
) -> None:
    status_updated = update_alpaca_managed_sell_status_if_current(
        conn,
        position_id,
        expected_sell_client_order_id=sell_client_order_id,
        sell_status="incomplete_order_metadata",
        notes=(
            "Alpaca did not report a valid positive quantity and limit price for the active managed sell; "
            "automatic renewal is blocked pending review"
        ),
    )
    _append_reconciliation_result(
        rows,
        position_id=position_id,
        symbol=symbol,
        action="sell",
        status="incomplete_order_metadata" if status_updated else "superseded",
        buy_client_order_id=buy_client_order_id,
        sell_client_order_id=sell_client_order_id,
        qty=None,
        limit_price=target_sell_price,
        alpaca_order_id=sell_alpaca_order_id,
        message=(
            "managed sell quantity or limit price is unavailable or invalid; automatic renewal is blocked for review"
            if status_updated
            else "order metadata belongs to a superseded managed GTC sell generation"
        ),
    )


def _append_incomplete_sell_order_identity_result(
    *,
    conn: sqlite3.Connection,
    rows: list[dict],
    position_id: int,
    symbol: str,
    buy_client_order_id: str,
    sell_client_order_id: str,
    observed_sell_qty: float,
    target_sell_price: float,
    sell_alpaca_order_id: str | None = None,
    identity_issue: str = "managed sell fill has no broker order ID",
    expected_sell_client_order_id: str | None = None,
    match_missing_sell_client_order_id: bool = False,
) -> None:
    status_updated = update_alpaca_managed_sell_status_if_current(
        conn,
        position_id,
        expected_sell_client_order_id=(
            None
            if match_missing_sell_client_order_id
            else expected_sell_client_order_id or sell_client_order_id
        ),
        sell_status="incomplete_order_metadata",
        notes=(
            f"Alpaca reported inconsistent managed sell order identity ({identity_issue}); "
            "accounting and automatic renewal are blocked pending review"
        ),
    )
    _append_reconciliation_result(
        rows,
        position_id=position_id,
        symbol=symbol,
        action="sell",
        status="incomplete_order_metadata" if status_updated else "superseded",
        buy_client_order_id=buy_client_order_id,
        sell_client_order_id=sell_client_order_id,
        qty=observed_sell_qty,
        limit_price=target_sell_price,
        alpaca_order_id=sell_alpaca_order_id,
        message=(
            f"{identity_issue}; accounting and automatic renewal are blocked for review"
            if status_updated
            else "order identity belongs to a superseded managed GTC sell generation"
        ),
    )


def _append_incomplete_sell_expiration_metadata_result(
    *,
    conn: sqlite3.Connection,
    rows: list[dict],
    position_id: int,
    symbol: str,
    buy_client_order_id: str,
    sell_client_order_id: str,
    target_sell_price: float,
    sell_alpaca_order_id: str,
    expiration_issue: str = "managed GTC sell has no valid expiration",
) -> None:
    status_updated = update_alpaca_managed_sell_status_if_current(
        conn,
        position_id,
        expected_sell_client_order_id=sell_client_order_id,
        sell_status="incomplete_order_metadata",
        notes=(
            f"{expiration_issue}; "
            "automatic renewal is blocked pending review"
        ),
    )
    _append_reconciliation_result(
        rows,
        position_id=position_id,
        symbol=symbol,
        action="sell",
        status="incomplete_order_metadata" if status_updated else "superseded",
        buy_client_order_id=buy_client_order_id,
        sell_client_order_id=sell_client_order_id,
        qty=None,
        limit_price=target_sell_price,
        alpaca_order_id=sell_alpaca_order_id,
        message=(
            f"{expiration_issue}; automatic renewal is blocked for review"
            if status_updated
            else "expiration metadata belongs to a superseded managed GTC sell generation"
        ),
    )


def _append_sell_fill_quantity_regression_result(
    *,
    conn: sqlite3.Connection,
    rows: list[dict],
    position_id: int,
    symbol: str,
    buy_client_order_id: str,
    sell_client_order_id: str,
    observed_sell_qty: float,
    target_sell_price: float,
    sell_alpaca_order_id: str | None,
) -> None:
    status_updated = update_alpaca_managed_sell_status_if_current(
        conn,
        position_id,
        expected_sell_client_order_id=sell_client_order_id,
        sell_status="fill_quantity_regression",
        notes=(
            "Alpaca reported a cumulative managed sell fill below the previously recorded quantity; "
            "automatic renewal is blocked pending review"
        ),
    )
    _append_reconciliation_result(
        rows,
        position_id=position_id,
        symbol=symbol,
        action="sell",
        status="fill_quantity_regression" if status_updated else "superseded",
        buy_client_order_id=buy_client_order_id,
        sell_client_order_id=sell_client_order_id,
        qty=observed_sell_qty,
        limit_price=target_sell_price,
        alpaca_order_id=sell_alpaca_order_id,
        message=(
            "broker cumulative sell fill moved backward; existing accounting was preserved for review"
            if status_updated
            else "fill regression belongs to a superseded managed GTC sell generation"
        ),
    )


def _append_overfilled_sell_result(
    *,
    conn: sqlite3.Connection,
    rows: list[dict],
    position_id: int,
    symbol: str,
    buy_client_order_id: str,
    sell_client_order_id: str | None,
    observed_sell_qty: float,
    target_sell_price: float,
    sell_alpaca_order_id: str | None,
) -> bool:
    status_updated = bool(sell_client_order_id) and update_alpaca_managed_sell_status_if_current(
        conn,
        position_id,
        expected_sell_client_order_id=sell_client_order_id or "",
        sell_status="quantity_mismatch",
        notes="cumulative managed sell quantity exceeds the managed buy quantity; manual review required",
    )
    _append_reconciliation_result(
        rows,
        position_id=position_id,
        symbol=symbol,
        action="sell",
        status="quantity_mismatch" if status_updated else "superseded",
        buy_client_order_id=buy_client_order_id,
        sell_client_order_id=sell_client_order_id,
        qty=observed_sell_qty,
        limit_price=target_sell_price,
        alpaca_order_id=sell_alpaca_order_id,
        message=(
            "managed sell quantity exceeds the managed buy quantity; position remains active for review"
            if status_updated
            else "fill observation belongs to a superseded managed GTC sell generation"
        ),
    )
    return status_updated


def _append_active_sell_completion_mismatch_result(
    *,
    conn: sqlite3.Connection,
    rows: list[dict],
    position_id: int,
    symbol: str,
    buy_client_order_id: str,
    sell_client_order_id: str,
    observed_sell_qty: float,
    target_sell_price: float,
    sell_alpaca_order_id: str | None,
) -> None:
    status_updated = update_alpaca_managed_sell_status_if_current(
        conn,
        position_id,
        expected_sell_client_order_id=sell_client_order_id,
        sell_status="quantity_mismatch",
        notes=(
            "managed sell cumulative fills cover the position but Alpaca still reports an active order; "
            "position closure is blocked pending review"
        ),
    )
    _append_reconciliation_result(
        rows,
        position_id=position_id,
        symbol=symbol,
        action="sell",
        status="quantity_mismatch" if status_updated else "superseded",
        buy_client_order_id=buy_client_order_id,
        sell_client_order_id=sell_client_order_id,
        qty=observed_sell_qty,
        limit_price=target_sell_price,
        alpaca_order_id=sell_alpaca_order_id,
        message=(
            "managed quantity is fully sold but the broker order remains active; closure is blocked for review"
            if status_updated
            else "active-order completion mismatch belongs to a superseded managed GTC sell generation"
        ),
    )


def _append_sell_submission_intent_mismatch_result(
    *,
    conn: sqlite3.Connection,
    rows: list[dict],
    position_id: int,
    symbol: str,
    buy_client_order_id: str,
    sell_client_order_id: str,
    requested_qty: float,
    target_sell_price: float,
    sell_order: dict,
) -> None:
    sell_alpaca_order_id = _optional_str(sell_order.get("id"))
    observed_order_qty = _optional_positive_float(sell_order.get("qty"))
    status_updated = update_alpaca_managed_sell_status_if_current(
        conn,
        position_id,
        expected_sell_client_order_id=sell_client_order_id,
        sell_status="quantity_mismatch",
        sell_alpaca_order_id=sell_alpaca_order_id,
        sell_order_qty=observed_order_qty,
        notes=(
            "Alpaca replacement response does not match the submitted quantity or frozen target price; "
            "automatic renewal is blocked pending review"
        ),
    )
    _append_reconciliation_result(
        rows,
        position_id=position_id,
        symbol=symbol,
        action="sell",
        status="quantity_mismatch" if status_updated else "superseded",
        buy_client_order_id=buy_client_order_id,
        sell_client_order_id=sell_client_order_id,
        qty=requested_qty,
        limit_price=target_sell_price,
        alpaca_order_id=sell_alpaca_order_id,
        message=(
            "replacement order quantity or limit price conflicts with the submitted intent; review required"
            if status_updated
            else "submission-intent mismatch belongs to a superseded managed GTC sell generation"
        ),
    )


def _response_json(resp: requests.Response) -> dict:
    try:
        payload = resp.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_alpaca_datetime(value: object) -> datetime | None:
    text = _optional_str(value)
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _optional_alpaca_datetime_str(value: object) -> str | None:
    text = _optional_str(value)
    return text if text is not None and _parse_alpaca_datetime(text) is not None else None


def _buy_submission_visibility_lease_active(position: dict, cfg: AlpacaOrderConfig) -> bool:
    """Keep a claimed buy intent active while Alpaca may still expose it late."""
    claimed_at = _parse_alpaca_datetime(position.get("buy_submission_claimed_at"))
    if claimed_at is None:
        claimed_at = _parse_alpaca_datetime(position.get("created_at"))
    # A row without a usable local timestamp cannot safely prove that the
    # broker has not accepted the order, so leave it active for review.
    if claimed_at is None:
        return True
    lease = max(
        _BUY_SUBMISSION_VISIBILITY_LEASE,
        timedelta(seconds=max(cfg.timeout_seconds, 0) + 60),
    )
    return _utc_now() < claimed_at + lease


def _prior_calendar_session(calendar: list[dict], next_open: datetime) -> date | None:
    next_open_date = next_open.astimezone(_NEW_YORK).date()
    sessions = []
    for session in calendar:
        try:
            session_date = pd.Timestamp(session.get("date")).date()
        except (TypeError, ValueError):
            continue
        if session_date < next_open_date:
            sessions.append(session_date)
    return max(sessions) if sessions else None


def _buy_submission_window(
    clock: dict,
    signal_date: str,
    prior_session: date | None,
) -> tuple[bool, str]:
    """Allow only the exact prior-session signal in the premarket window."""
    if bool(clock.get("is_open", False)):
        return False, "market is open; close-based buy deferred until the next closed submission window"

    next_open = _parse_alpaca_datetime(clock.get("next_open"))
    if next_open is None:
        raise ValueError("Alpaca clock did not provide a valid next_open timestamp")
    clock_timestamp = _parse_alpaca_datetime(clock.get("timestamp"))
    if clock_timestamp is None:
        raise ValueError("Alpaca clock did not provide a valid timestamp")

    try:
        signal_session = pd.Timestamp(signal_date).date()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Buy signal has an invalid date: {signal_date!r}") from exc

    next_open_session = next_open.astimezone(_NEW_YORK).date()
    current_session = clock_timestamp.astimezone(_NEW_YORK).date()
    if next_open_session != current_session:
        return False, "buy submissions are limited to the premarket window for the next regular-session open"
    if prior_session is None:
        raise ValueError("Alpaca calendar did not provide a prior trading session")
    if signal_session != prior_session:
        return False, "signal date is not the immediately preceding trading session"
    return True, ""


def _gtc_sell_renewal_due(
    sell_order: dict,
    cfg: AlpacaOrderConfig,
    *,
    persisted_expires_at: object = None,
) -> bool:
    if not cfg.gtc_sell_renewal_enabled or cfg.gtc_sell_renewal_days_before_expiration < 0:
        return False

    sell_status = str(sell_order.get("status", "")).lower()
    if sell_status not in SELL_RENEWABLE_STATUSES:
        return False

    expires_at = _parse_alpaca_datetime(sell_order.get("expires_at"))
    if expires_at is None:
        expires_at = _parse_alpaca_datetime(persisted_expires_at)
    if expires_at is None:
        return False

    renewal_cutoff = _utc_now() + timedelta(days=cfg.gtc_sell_renewal_days_before_expiration)
    return expires_at <= renewal_cutoff


def _sell_renewal_claim_lease_active(position: dict) -> bool:
    requested_at = _parse_alpaca_datetime(position.get("sell_renewal_requested_at"))
    if requested_at is None:
        return False
    return requested_at > _utc_now() - _SELL_RENEWAL_CLAIM_LEASE


class AlpacaClient:
    def __init__(self, cfg: AlpacaOrderConfig):
        self.cfg = cfg
        self.base_url = cfg.base_url.rstrip("/")
        self.headers = _alpaca_headers(cfg)

    def get_order_by_client_order_id(self, client_order_id: str) -> requests.Response:
        return requests.get(
            f"{self.base_url}/v2/orders:by_client_order_id",
            headers=self.headers,
            params={"client_order_id": client_order_id},
            timeout=self.cfg.timeout_seconds,
        )

    def get_order(self, order_id: str) -> requests.Response:
        return requests.get(
            f"{self.base_url}/v2/orders/{order_id}",
            headers=self.headers,
            timeout=self.cfg.timeout_seconds,
        )

    def cancel_order(self, order_id: str) -> requests.Response:
        return requests.delete(
            f"{self.base_url}/v2/orders/{order_id}",
            headers=self.headers,
            timeout=self.cfg.timeout_seconds,
        )

    def market_clock(self) -> dict:
        response = requests.get(
            f"{self.base_url}/v2/clock",
            headers=self.headers,
            timeout=self.cfg.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Alpaca clock returned an invalid response")
        return payload

    def calendar(self, start: date, end: date) -> list[dict]:
        response = requests.get(
            f"{self.base_url}/v2/calendar",
            headers=self.headers,
            params={"start": start.isoformat(), "end": end.isoformat()},
            timeout=self.cfg.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise ValueError("Alpaca calendar returned an invalid response")
        return payload

    def cash_notional(self, *, eligible_buy_signal_count: int) -> float:
        return _alpaca_cash_notional(
            self.cfg,
            self.headers,
            eligible_buy_signal_count=eligible_buy_signal_count,
        )

    def position_qty(self, symbol: str) -> float:
        return _alpaca_position_qty(symbol, self.cfg, self.headers)

    def positions(self) -> list[dict]:
        response = requests.get(
            f"{self.base_url}/v2/positions",
            headers=self.headers,
            timeout=self.cfg.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise ValueError("Alpaca positions returned an invalid response")
        return payload

    def asset(self, symbol: str) -> dict:
        return _alpaca_asset(symbol, self.cfg, self.headers)

    def has_open_order(self, symbol: str, side: str) -> bool:
        return _alpaca_has_open_order(symbol, side, self.cfg, self.headers)

    def open_orders(self) -> list[dict]:
        return _alpaca_open_orders(self.cfg, self.headers)

    def submit_limit_buy_order(
        self,
        *,
        symbol: str,
        qty: int,
        limit_price: float,
        client_order_id: str,
    ) -> requests.Response:
        return requests.post(
            f"{self.base_url}/v2/orders",
            headers=self.headers,
            json={
                "symbol": symbol,
                "side": "buy",
                "type": "limit",
                "time_in_force": "day",
                "extended_hours": False,
                "client_order_id": client_order_id,
                "qty": str(qty),
                "limit_price": _format_alpaca_limit_price(limit_price, rounding=ROUND_FLOOR),
            },
            timeout=self.cfg.timeout_seconds,
        )

    def submit_limit_sell_order(
        self,
        *,
        symbol: str,
        qty: float,
        limit_price: float,
        client_order_id: str,
    ) -> requests.Response:
        return requests.post(
            f"{self.base_url}/v2/orders",
            headers=self.headers,
            json={
                "symbol": symbol,
                "side": "sell",
                "type": "limit",
                "time_in_force": "gtc",
                "extended_hours": False,
                "client_order_id": client_order_id,
                "qty": _format_order_number(qty, 8),
                "limit_price": _format_alpaca_limit_price(limit_price, rounding=ROUND_CEILING),
            },
            timeout=self.cfg.timeout_seconds,
        )


def _append_result(
    rows: list[dict],
    *,
    symbol: str,
    signal_date: str,
    client_order_id: str,
    amount_key: str,
    amount: float | None,
    status: str,
    alpaca_order_id: str | None,
    message: str,
) -> None:
    rows.append(
        {
            "Asset": symbol,
            "Date": signal_date,
            "Client Order ID": client_order_id,
            amount_key: amount,
            "Status": status,
            "Alpaca Order ID": alpaca_order_id,
            "Message": message,
        }
    )


def _recover_managed_buy_submission(
    *,
    conn: sqlite3.Connection,
    client: AlpacaClient,
    rows: list[dict],
    signal_index: int,
    signal: dict,
    symbol: str,
    signal_date: str,
    client_order_id: str,
    notional: float,
    qty: int,
    limit_price: float,
    message: str,
) -> bool:
    try:
        lookup = client.get_order_by_client_order_id(client_order_id)
        if lookup.status_code != 200:
            return False
        recovered = _response_json(lookup)
    except Exception:
        return False

    save_alpaca_managed_buy_order(
        conn,
        workflow=_signal_workflow(signal),
        symbol=symbol,
        signal_symbol=str(signal["RSI Symbol"]),
        buy_rsi=float(signal["Buy RSI"]),
        profit_target_multiple=float(signal["Sell Return Multiple"]),
        buy_signal_date=signal_date,
        buy_client_order_id=client_order_id,
        buy_alpaca_order_id=_optional_str(recovered.get("id")),
        buy_submitted_at=_optional_str(recovered.get("submitted_at")),
        buy_status=str(recovered.get("status", "existing")).lower(),
        notes="managed buy recovered by client order ID after an ambiguous submission result",
    )
    rows.append(
        {
            "Workflow": _signal_workflow(signal),
            "Asset": symbol,
            "Date": signal_date,
            "Client Order ID": client_order_id,
            "Notional": notional,
            "Qty": qty,
            "Limit Price": limit_price,
            "Status": "existing",
            "Alpaca Order ID": _optional_str(recovered.get("id")),
            "Message": f"recovered after ambiguous submission result: {message}",
            "_signal_index": signal_index,
        }
    )
    return True


def _http_error_message(exc: HTTPError) -> str:
    response = exc.response
    if response is None:
        return str(exc)

    detail = ""
    try:
        payload = response.json()
    except ValueError:
        detail = response.text.strip()
    else:
        if isinstance(payload, dict):
            detail = str(payload.get("message") or payload.get("error") or payload)
        else:
            detail = str(payload)

    return f"{exc}: {detail}" if detail else str(exc)


def _ambiguous_submission_http_error(exc: HTTPError) -> bool:
    """Whether an HTTP error could still mean Alpaca accepted the order."""
    response = exc.response
    if response is None:
        return True
    if response.status_code >= 500 or response.status_code in {408, 409, 425, 429}:
        return True
    detail = _http_error_message(exc).lower()
    mentions_client_order_id = any(
        marker in detail for marker in {"client_order_id", "client order id", "client order"}
    )
    return mentions_client_order_id and any(
        marker in detail for marker in {"exist", "duplicate", "already"}
    )


def _fetch_order_payload(client: AlpacaClient, order_id: str | None, client_order_id: str) -> dict:
    response = client.get_order(order_id) if order_id else client.get_order_by_client_order_id(client_order_id)
    if response.status_code == 404 and order_id:
        response = client.get_order_by_client_order_id(client_order_id)
    response.raise_for_status()
    return response.json()


def _append_reconciliation_result(
    rows: list[dict],
    *,
    position_id: int,
    symbol: str,
    action: str,
    status: str,
    buy_client_order_id: str,
    sell_client_order_id: str | None,
    qty: float | None,
    limit_price: float | None,
    alpaca_order_id: str | None,
    message: str,
) -> None:
    rows.append(
        {
            "Position ID": position_id,
            "Asset": symbol,
            "Action": action,
            "Status": status,
            "Buy Client Order ID": buy_client_order_id,
            "Sell Client Order ID": sell_client_order_id,
            "Qty": qty,
            "Limit Price": limit_price,
            "Alpaca Order ID": alpaca_order_id,
            "Message": message,
        }
    )


def _record_filled_managed_sell(
    *,
    conn: sqlite3.Connection,
    position_id: int,
    sell_order: dict,
    sell_alpaca_order_id: str | None,
    sell_client_order_id: str,
    sell_status: str = "filled",
    close_on_complete: bool = True,
) -> tuple[float, float, bool]:
    sell_filled_qty, sell_filled_avg_price, fill_metadata_is_valid = _sell_fill_metadata(sell_order)
    if not fill_metadata_is_valid or sell_filled_qty <= _MANAGED_QTY_TOLERANCE:
        raise SellFillMetadataError(sell_filled_qty)
    if not sell_alpaca_order_id:
        raise SellOrderIdentityError("filled Alpaca sell order is missing its broker order ID")

    remaining_qty, generation_is_current = mark_alpaca_managed_sell_filled_if_current(
        conn,
        position_id,
        expected_sell_client_order_id=sell_client_order_id,
        sell_status=sell_status,
        sell_filled_qty=sell_filled_qty,
        sell_filled_avg_price=sell_filled_avg_price,
        sell_filled_at=_optional_str(sell_order.get("filled_at")),
        sell_alpaca_order_id=sell_alpaca_order_id,
        sell_submitted_at=_optional_str(sell_order.get("submitted_at")),
        sell_expires_at=_optional_alpaca_datetime_str(sell_order.get("expires_at")),
    )
    if not generation_is_current:
        return sell_filled_qty, remaining_qty, False
    if sell_status == "filled" and _managed_sell_is_complete(remaining_qty) and close_on_complete:
        closed = close_alpaca_managed_position_if_current_and_complete(
            conn,
            position_id,
            expected_sell_client_order_id=sell_client_order_id,
            closed_at=_optional_str(sell_order.get("filled_at")),
            notes="managed target sell filled",
        )
        if not closed:
            return sell_filled_qty, remaining_qty, False
    elif sell_status == "filled" and not _managed_sell_is_complete(remaining_qty):
        mismatch_generation_is_current = update_alpaca_managed_sell_status_if_current(
            conn,
            position_id,
            expected_sell_client_order_id=sell_client_order_id,
            sell_status="quantity_mismatch",
            notes="filled sell quantity did not close the managed buy quantity; manual review required",
        )
        if not mismatch_generation_is_current:
            return sell_filled_qty, remaining_qty, False
    return sell_filled_qty, remaining_qty, True


def _record_observed_managed_sell_order(
    *,
    conn: sqlite3.Connection,
    position_id: int,
    sell_client_order_id: str,
    sell_order: dict,
) -> tuple[str, str | None, bool]:
    sell_status = str(sell_order.get("status", "submitted")).lower()
    sell_alpaca_order_id = _optional_str(sell_order.get("id"))
    generation_is_current = update_alpaca_managed_sell_status_if_current(
        conn,
        position_id,
        expected_sell_client_order_id=sell_client_order_id,
        sell_alpaca_order_id=sell_alpaca_order_id,
        sell_submitted_at=_optional_str(sell_order.get("submitted_at")),
        sell_status=sell_status,
        sell_expires_at=_optional_alpaca_datetime_str(sell_order.get("expires_at")),
        sell_order_qty=_optional_float(sell_order.get("qty")),
        clear_sell_submission_retry_claim=True,
    )
    return sell_status, sell_alpaca_order_id, generation_is_current


def _recover_managed_sell_submission(
    *,
    conn: sqlite3.Connection,
    client: AlpacaClient,
    rows: list[dict],
    position_id: int,
    symbol: str,
    buy_client_order_id: str,
    sell_client_order_id: str,
    filled_qty: float,
    target_sell_price: float,
    message: str,
    close_on_complete: bool,
) -> bool:
    try:
        lookup = client.get_order_by_client_order_id(sell_client_order_id)
        if lookup.status_code != 200:
            return False
        recovered = _response_json(lookup)
    except Exception:
        return False

    if not _sell_order_identity_is_consistent(
        recovered,
        expected_sell_client_order_id=sell_client_order_id,
    ):
        _append_incomplete_sell_order_identity_result(
            conn=conn,
            rows=rows,
            position_id=position_id,
            symbol=symbol,
            buy_client_order_id=buy_client_order_id,
            sell_client_order_id=sell_client_order_id,
            observed_sell_qty=_sell_fill_metadata(recovered)[0],
            target_sell_price=target_sell_price,
            sell_alpaca_order_id=_optional_str(recovered.get("id")),
            identity_issue="broker response does not match the managed sell client-order generation",
        )
        return True
    submission_intent_matches = _sell_order_matches_submission_intent(
        recovered,
        requested_qty=filled_qty,
        target_sell_price=target_sell_price,
    )

    sell_status, sell_alpaca_order_id, generation_is_current = _record_observed_managed_sell_order(
        conn=conn,
        position_id=position_id,
        sell_client_order_id=sell_client_order_id,
        sell_order=recovered,
    )
    if not generation_is_current:
        _append_reconciliation_result(
            rows,
            position_id=position_id,
            symbol=symbol,
            action="sell",
            status="superseded",
            buy_client_order_id=buy_client_order_id,
            sell_client_order_id=sell_client_order_id,
            qty=filled_qty,
            limit_price=target_sell_price,
            alpaca_order_id=sell_alpaca_order_id,
            message="submission recovery belongs to a superseded managed GTC sell generation",
        )
        return True
    remaining_qty: float | None = None
    recovered_sell_qty = _sell_fill_metadata(recovered)[0]
    if sell_status == "filled" or recovered_sell_qty > _MANAGED_QTY_TOLERANCE:
        try:
            _, remaining_qty, generation_is_current = _record_filled_managed_sell(
                conn=conn,
                position_id=position_id,
                sell_order=recovered,
                sell_alpaca_order_id=sell_alpaca_order_id,
                sell_client_order_id=sell_client_order_id,
                sell_status=sell_status,
                close_on_complete=close_on_complete and submission_intent_matches,
            )
        except SellFillMetadataError as exc:
            _append_incomplete_sell_fill_result(
                conn=conn,
                rows=rows,
                position_id=position_id,
                symbol=symbol,
                buy_client_order_id=buy_client_order_id,
                sell_client_order_id=sell_client_order_id,
                observed_sell_qty=exc.observed_qty,
                target_sell_price=target_sell_price,
                sell_alpaca_order_id=sell_alpaca_order_id,
            )
            return True
        except SellOrderIdentityError:
            _append_incomplete_sell_order_identity_result(
                conn=conn,
                rows=rows,
                position_id=position_id,
                symbol=symbol,
                buy_client_order_id=buy_client_order_id,
                sell_client_order_id=sell_client_order_id,
                observed_sell_qty=_sell_fill_metadata(recovered)[0],
                target_sell_price=target_sell_price,
            )
            return True
        except SellFillQuantityRegressionError:
            _append_sell_fill_quantity_regression_result(
                conn=conn,
                rows=rows,
                position_id=position_id,
                symbol=symbol,
                buy_client_order_id=buy_client_order_id,
                sell_client_order_id=sell_client_order_id,
                observed_sell_qty=_sell_fill_metadata(recovered)[0],
                target_sell_price=target_sell_price,
                sell_alpaca_order_id=sell_alpaca_order_id,
            )
            return True
        if not generation_is_current:
            return True
        if remaining_qty < -_MANAGED_QTY_TOLERANCE:
            _append_overfilled_sell_result(
                conn=conn,
                rows=rows,
                position_id=position_id,
                symbol=symbol,
                buy_client_order_id=buy_client_order_id,
                sell_client_order_id=sell_client_order_id,
                observed_sell_qty=recovered_sell_qty,
                target_sell_price=target_sell_price,
                sell_alpaca_order_id=sell_alpaca_order_id,
            )
            return True
        if sell_status != "filled" and _managed_sell_is_complete(remaining_qty):
            _append_active_sell_completion_mismatch_result(
                conn=conn,
                rows=rows,
                position_id=position_id,
                symbol=symbol,
                buy_client_order_id=buy_client_order_id,
                sell_client_order_id=sell_client_order_id,
                observed_sell_qty=recovered_sell_qty,
                target_sell_price=target_sell_price,
                sell_alpaca_order_id=sell_alpaca_order_id,
            )
            return True
    if not submission_intent_matches:
        _append_sell_submission_intent_mismatch_result(
            conn=conn,
            rows=rows,
            position_id=position_id,
            symbol=symbol,
            buy_client_order_id=buy_client_order_id,
            sell_client_order_id=sell_client_order_id,
            requested_qty=filled_qty,
            target_sell_price=target_sell_price,
            sell_order=recovered,
        )
        return True
    if sell_status in SELL_RENEWABLE_STATUSES and _gtc_sell_renewal_due(recovered, client.cfg):
        _append_incomplete_sell_expiration_metadata_result(
            conn=conn,
            rows=rows,
            position_id=position_id,
            symbol=symbol,
            buy_client_order_id=buy_client_order_id,
            sell_client_order_id=sell_client_order_id,
            target_sell_price=target_sell_price,
            sell_alpaca_order_id=sell_alpaca_order_id or "",
            expiration_issue="recovered managed GTC sell expiration remains inside the renewal window",
        )
        return True
    result_message = f"recovered managed sell after ambiguous submission result: {message}"
    if sell_status == "filled":
        if remaining_qty is not None and _managed_sell_is_complete(remaining_qty):
            result_message = (
                "managed target sell filled; position closed"
                if close_on_complete
                else "managed sell filled; awaiting final status of the still-open parent buy"
            )
        else:
            result_message = (
                "managed sell fill quantity does not match the managed buy; position remains active for review"
            )
    _append_reconciliation_result(
        rows,
        position_id=position_id,
        symbol=symbol,
        action="sell",
        status=sell_status,
        buy_client_order_id=buy_client_order_id,
        sell_client_order_id=sell_client_order_id,
        qty=filled_qty,
        limit_price=target_sell_price,
        alpaca_order_id=sell_alpaca_order_id,
        message=result_message,
    )
    return True


def _submit_managed_gtc_sell(
    *,
    conn: sqlite3.Connection,
    client: AlpacaClient,
    rows: list[dict],
    position_id: int,
    symbol: str,
    buy_client_order_id: str,
    sell_client_order_id: str,
    filled_qty: float,
    target_sell_price: float,
    increment_renewal_count: bool,
    replacement_message: str,
    close_on_complete: bool = True,
    intent_already_persisted: bool = False,
) -> None:
    if not intent_already_persisted:
        record_alpaca_managed_sell_order(
            conn,
            position_id,
            sell_client_order_id=sell_client_order_id,
            sell_alpaca_order_id=None,
            sell_submitted_at=None,
            sell_status="submission_pending",
            sell_expires_at=None,
            sell_order_qty=filled_qty,
            increment_renewal_count=increment_renewal_count,
            notes="managed GTC sell submission intent persisted before broker request",
        )
    try:
        sell_resp = client.submit_limit_sell_order(
            symbol=symbol,
            qty=float(filled_qty),
            limit_price=float(target_sell_price),
            client_order_id=sell_client_order_id,
        )
        sell_resp.raise_for_status()
    except HTTPError as exc:
        if _ambiguous_submission_http_error(exc):
            if _recover_managed_sell_submission(
                conn=conn,
                client=client,
                rows=rows,
                position_id=position_id,
                symbol=symbol,
                buy_client_order_id=buy_client_order_id,
                sell_client_order_id=sell_client_order_id,
                filled_qty=filled_qty,
                target_sell_price=target_sell_price,
                message=_http_error_message(exc),
                close_on_complete=close_on_complete,
            ):
                return
            state_updated = update_alpaca_managed_sell_status_if_current(
                conn,
                position_id,
                expected_sell_client_order_id=sell_client_order_id,
                sell_status="submission_unknown",
                notes="broker returned an ambiguous error after managed sell submission; recovery is pending",
            )
            if not state_updated:
                _append_reconciliation_result(
                    rows, position_id=position_id, symbol=symbol, action="sell", status="superseded",
                    buy_client_order_id=buy_client_order_id, sell_client_order_id=sell_client_order_id,
                    qty=filled_qty, limit_price=target_sell_price, alpaca_order_id=None,
                    message="submission result belongs to a superseded managed GTC sell generation",
                )
                return
            _append_reconciliation_result(
                rows,
                position_id=position_id,
                symbol=symbol,
                action="sell",
                status="submission_unknown",
                buy_client_order_id=buy_client_order_id,
                sell_client_order_id=sell_client_order_id,
                qty=filled_qty,
                limit_price=target_sell_price,
                alpaca_order_id=None,
                message="broker response is ambiguous; managed sell recovery will retry by client order ID",
            )
            return
        state_updated = update_alpaca_managed_sell_status_if_current(
            conn,
            position_id,
            expected_sell_client_order_id=sell_client_order_id,
            sell_status="submission_failed",
            notes="managed sell submission failed before an Alpaca order was accepted",
        )
        if not state_updated:
            _append_reconciliation_result(
                rows, position_id=position_id, symbol=symbol, action="sell", status="superseded",
                buy_client_order_id=buy_client_order_id, sell_client_order_id=sell_client_order_id,
                qty=filled_qty, limit_price=target_sell_price, alpaca_order_id=None,
                message="submission result belongs to a superseded managed GTC sell generation",
            )
            return
        _append_reconciliation_result(
            rows,
            position_id=position_id,
            symbol=symbol,
            action="sell",
            status="error",
            buy_client_order_id=buy_client_order_id,
            sell_client_order_id=sell_client_order_id,
            qty=filled_qty,
            limit_price=target_sell_price,
            alpaca_order_id=None,
            message=_http_error_message(exc),
        )
        return
    except requests.RequestException as exc:
        if _recover_managed_sell_submission(
            conn=conn,
            client=client,
            rows=rows,
            position_id=position_id,
            symbol=symbol,
            buy_client_order_id=buy_client_order_id,
            sell_client_order_id=sell_client_order_id,
            filled_qty=filled_qty,
            target_sell_price=target_sell_price,
            message=str(exc),
            close_on_complete=close_on_complete,
        ):
            return
        state_updated = update_alpaca_managed_sell_status_if_current(
            conn,
            position_id,
            expected_sell_client_order_id=sell_client_order_id,
            sell_status="submission_unknown",
            notes="managed sell submission transport failed; recovery is pending",
        )
        if not state_updated:
            _append_reconciliation_result(
                rows, position_id=position_id, symbol=symbol, action="sell", status="superseded",
                buy_client_order_id=buy_client_order_id, sell_client_order_id=sell_client_order_id,
                qty=filled_qty, limit_price=target_sell_price, alpaca_order_id=None,
                message="submission result belongs to a superseded managed GTC sell generation",
            )
            return
        _append_reconciliation_result(
            rows,
            position_id=position_id,
            symbol=symbol,
            action="sell",
            status="submission_unknown",
            buy_client_order_id=buy_client_order_id,
            sell_client_order_id=sell_client_order_id,
            qty=filled_qty,
            limit_price=target_sell_price,
            alpaca_order_id=None,
            message="managed sell submission is ambiguous; recovery will retry by client order ID",
        )
        return
    except Exception as exc:
        if _recover_managed_sell_submission(
            conn=conn,
            client=client,
            rows=rows,
            position_id=position_id,
            symbol=symbol,
            buy_client_order_id=buy_client_order_id,
            sell_client_order_id=sell_client_order_id,
            filled_qty=filled_qty,
            target_sell_price=target_sell_price,
            message=str(exc),
            close_on_complete=close_on_complete,
        ):
            return
        state_updated = update_alpaca_managed_sell_status_if_current(
            conn,
            position_id,
            expected_sell_client_order_id=sell_client_order_id,
            sell_status="submission_unknown",
            notes="managed sell submission ended unexpectedly; recovery is pending",
        )
        if not state_updated:
            _append_reconciliation_result(
                rows, position_id=position_id, symbol=symbol, action="sell", status="superseded",
                buy_client_order_id=buy_client_order_id, sell_client_order_id=sell_client_order_id,
                qty=filled_qty, limit_price=target_sell_price, alpaca_order_id=None,
                message="submission result belongs to a superseded managed GTC sell generation",
            )
            return
        _append_reconciliation_result(
            rows,
            position_id=position_id,
            symbol=symbol,
            action="sell",
            status="submission_unknown",
            buy_client_order_id=buy_client_order_id,
            sell_client_order_id=sell_client_order_id,
            qty=filled_qty,
            limit_price=target_sell_price,
            alpaca_order_id=None,
            message="managed sell submission is ambiguous; recovery will retry by client order ID",
        )
        return
    sell_order = _response_json(sell_resp)
    if not _sell_order_identity_is_consistent(
        sell_order,
        expected_sell_client_order_id=sell_client_order_id,
    ):
        _append_incomplete_sell_order_identity_result(
            conn=conn,
            rows=rows,
            position_id=position_id,
            symbol=symbol,
            buy_client_order_id=buy_client_order_id,
            sell_client_order_id=sell_client_order_id,
            observed_sell_qty=_sell_fill_metadata(sell_order)[0],
            target_sell_price=target_sell_price,
            sell_alpaca_order_id=_optional_str(sell_order.get("id")),
            identity_issue="broker response does not match the submitted sell client-order generation",
        )
        return
    submission_intent_matches = _sell_order_matches_submission_intent(
        sell_order,
        requested_qty=filled_qty,
        target_sell_price=target_sell_price,
    )
    sell_status, sell_alpaca_order_id, generation_is_current = _record_observed_managed_sell_order(
        conn=conn,
        position_id=position_id,
        sell_client_order_id=sell_client_order_id,
        sell_order=sell_order,
    )
    if not generation_is_current:
        _append_reconciliation_result(
            rows, position_id=position_id, symbol=symbol, action="sell", status="superseded",
            buy_client_order_id=buy_client_order_id, sell_client_order_id=sell_client_order_id,
            qty=filled_qty, limit_price=target_sell_price, alpaca_order_id=sell_alpaca_order_id,
            message="submission response belongs to a superseded managed GTC sell generation",
        )
        return
    remaining_qty: float | None = None
    submitted_sell_qty = _sell_fill_metadata(sell_order)[0]
    if sell_status == "filled" or submitted_sell_qty > _MANAGED_QTY_TOLERANCE:
        try:
            _, remaining_qty, generation_is_current = _record_filled_managed_sell(
                conn=conn,
                position_id=position_id,
                sell_order=sell_order,
                sell_alpaca_order_id=sell_alpaca_order_id,
                sell_client_order_id=sell_client_order_id,
                sell_status=sell_status,
                close_on_complete=close_on_complete and submission_intent_matches,
            )
        except SellFillMetadataError as exc:
            _append_incomplete_sell_fill_result(
                conn=conn,
                rows=rows,
                position_id=position_id,
                symbol=symbol,
                buy_client_order_id=buy_client_order_id,
                sell_client_order_id=sell_client_order_id,
                observed_sell_qty=exc.observed_qty,
                target_sell_price=target_sell_price,
                sell_alpaca_order_id=sell_alpaca_order_id,
            )
            return
        except SellOrderIdentityError:
            _append_incomplete_sell_order_identity_result(
                conn=conn,
                rows=rows,
                position_id=position_id,
                symbol=symbol,
                buy_client_order_id=buy_client_order_id,
                sell_client_order_id=sell_client_order_id,
                observed_sell_qty=_sell_fill_metadata(sell_order)[0],
                target_sell_price=target_sell_price,
            )
            return
        except SellFillQuantityRegressionError:
            _append_sell_fill_quantity_regression_result(
                conn=conn,
                rows=rows,
                position_id=position_id,
                symbol=symbol,
                buy_client_order_id=buy_client_order_id,
                sell_client_order_id=sell_client_order_id,
                observed_sell_qty=_sell_fill_metadata(sell_order)[0],
                target_sell_price=target_sell_price,
                sell_alpaca_order_id=sell_alpaca_order_id,
            )
            return
        if not generation_is_current:
            return
        if remaining_qty < -_MANAGED_QTY_TOLERANCE:
            _append_overfilled_sell_result(
                conn=conn,
                rows=rows,
                position_id=position_id,
                symbol=symbol,
                buy_client_order_id=buy_client_order_id,
                sell_client_order_id=sell_client_order_id,
                observed_sell_qty=submitted_sell_qty,
                target_sell_price=target_sell_price,
                sell_alpaca_order_id=sell_alpaca_order_id,
            )
            return
        if sell_status != "filled" and _managed_sell_is_complete(remaining_qty):
            _append_active_sell_completion_mismatch_result(
                conn=conn,
                rows=rows,
                position_id=position_id,
                symbol=symbol,
                buy_client_order_id=buy_client_order_id,
                sell_client_order_id=sell_client_order_id,
                observed_sell_qty=submitted_sell_qty,
                target_sell_price=target_sell_price,
                sell_alpaca_order_id=sell_alpaca_order_id,
            )
            return
    if not submission_intent_matches:
        _append_sell_submission_intent_mismatch_result(
            conn=conn,
            rows=rows,
            position_id=position_id,
            symbol=symbol,
            buy_client_order_id=buy_client_order_id,
            sell_client_order_id=sell_client_order_id,
            requested_qty=filled_qty,
            target_sell_price=target_sell_price,
            sell_order=sell_order,
        )
        return
    message = replacement_message
    if sell_status == "filled":
        if remaining_qty is not None and _managed_sell_is_complete(remaining_qty):
            message = (
                "managed target sell filled; position closed"
                if close_on_complete
                else "managed sell filled; awaiting final status of the still-open parent buy"
            )
        else:
            message = "managed sell fill quantity does not match the managed buy; position remains active for review"
    renewal_submission = increment_renewal_count or intent_already_persisted
    renewal_is_active = sell_status in SELL_RENEWABLE_STATUSES
    if renewal_submission and renewal_is_active and _gtc_sell_renewal_due(sell_order, client.cfg):
        _append_incomplete_sell_expiration_metadata_result(
            conn=conn,
            rows=rows,
            position_id=position_id,
            symbol=symbol,
            buy_client_order_id=buy_client_order_id,
            sell_client_order_id=sell_client_order_id,
            target_sell_price=target_sell_price,
            sell_alpaca_order_id=sell_alpaca_order_id or "",
            expiration_issue="replacement managed GTC sell expiration remains inside the renewal window",
        )
        return
    if renewal_submission and sell_status != "filled" and not renewal_is_active:
        message = (
            f"managed GTC sell replacement returned {sell_status}; "
            "no active replacement was confirmed and manual review is required"
        )
    _append_reconciliation_result(
        rows,
        position_id=position_id,
        symbol=symbol,
        action="sell",
        status=(
            "renewed"
            if renewal_submission and renewal_is_active
            else sell_status
        ),
        buy_client_order_id=buy_client_order_id,
        sell_client_order_id=sell_client_order_id,
        qty=filled_qty,
        limit_price=target_sell_price,
        alpaca_order_id=sell_alpaca_order_id,
        message=message,
    )


def _append_fractional_sell_result(
    *,
    conn: sqlite3.Connection,
    rows: list[dict],
    position_id: int,
    symbol: str,
    buy_client_order_id: str,
    sell_client_order_id: str | None,
    filled_qty: float,
    target_sell_price: float,
) -> None:
    status_updated = update_alpaca_managed_sell_status_if_current(
        conn,
        position_id,
        expected_sell_client_order_id=sell_client_order_id,
        sell_status="fractional_qty",
        notes="filled quantity is fractional; no GTC limit sell submitted",
    )
    _append_reconciliation_result(
        rows,
        position_id=position_id,
        symbol=symbol,
        action="sell",
        status="fractional_qty" if status_updated else "superseded",
        buy_client_order_id=buy_client_order_id,
        sell_client_order_id=sell_client_order_id,
        qty=filled_qty,
        limit_price=target_sell_price,
        alpaca_order_id=None,
        message=(
            "filled quantity is fractional; no GTC limit sell submitted"
            if status_updated
            else "fractional quantity result belongs to a superseded managed GTC sell generation"
        ),
    )


def _append_open_sell_result(
    *,
    rows: list[dict],
    position_id: int,
    symbol: str,
    buy_client_order_id: str,
    sell_client_order_id: str | None,
    filled_qty: float,
    target_sell_price: float,
) -> None:
    _append_reconciliation_result(
        rows,
        position_id=position_id,
        symbol=symbol,
        action="sell",
        status="open_order",
        buy_client_order_id=buy_client_order_id,
        sell_client_order_id=sell_client_order_id,
        qty=filled_qty,
        limit_price=target_sell_price,
        alpaca_order_id=None,
        message="open sell order already exists for symbol in Alpaca account",
    )


def _append_multiple_managed_open_sells_result(
    *,
    rows: list[dict],
    position_id: int,
    symbol: str,
    buy_client_order_id: str,
    sell_client_order_id: str | None,
    filled_qty: float,
    target_sell_price: float,
) -> None:
    _append_reconciliation_result(
        rows,
        position_id=position_id,
        symbol=symbol,
        action="sell",
        status="managed_order_conflict",
        buy_client_order_id=buy_client_order_id,
        sell_client_order_id=sell_client_order_id,
        qty=filled_qty,
        limit_price=target_sell_price,
        alpaca_order_id=None,
        message="multiple managed GTC sell generations are open; no order was attached or submitted",
    )


def _append_recovered_open_sell_result(
    *,
    conn: sqlite3.Connection,
    rows: list[dict],
    position_id: int,
    symbol: str,
    buy_client_order_id: str,
    sell_order: dict,
    filled_qty: float,
    target_sell_price: float,
    expected_sell_client_order_id: str | None,
    expected_sell_alpaca_order_id: str | None,
    expected_renewal_count: int,
    cfg: AlpacaOrderConfig,
    close_on_complete: bool = True,
) -> None:
    sell_client_order_id = _optional_str(sell_order.get("client_order_id")) or _alpaca_exit_client_order_id(
        symbol,
        position_id,
    )
    sell_status = str(sell_order.get("status", "submitted")).lower()
    sell_alpaca_order_id = _optional_str(sell_order.get("id"))
    sell_renewal_count = _alpaca_exit_client_order_generation(symbol, position_id, sell_client_order_id)
    if sell_alpaca_order_id is None:
        _append_incomplete_sell_order_identity_result(
            conn=conn,
            rows=rows,
            position_id=position_id,
            symbol=symbol,
            buy_client_order_id=buy_client_order_id,
            sell_client_order_id=sell_client_order_id,
            observed_sell_qty=_sell_fill_metadata(sell_order)[0],
            target_sell_price=target_sell_price,
            identity_issue="recovered open managed sell has no broker order ID",
            expected_sell_client_order_id=expected_sell_client_order_id,
            match_missing_sell_client_order_id=expected_sell_client_order_id is None,
        )
        return
    intent_matches = _sell_order_matches_submission_intent(
        sell_order,
        requested_qty=filled_qty,
        target_sell_price=target_sell_price,
    )
    attached = sell_renewal_count is not None and attach_alpaca_managed_sell_order_if_current(
        conn,
        position_id,
        expected_sell_client_order_id=expected_sell_client_order_id,
        expected_sell_alpaca_order_id=expected_sell_alpaca_order_id,
        expected_renewal_count=expected_renewal_count,
        sell_renewal_count=sell_renewal_count,
        sell_client_order_id=sell_client_order_id,
        sell_alpaca_order_id=sell_alpaca_order_id,
        sell_submitted_at=_optional_str(sell_order.get("submitted_at")),
        sell_status=sell_status,
        sell_expires_at=_optional_alpaca_datetime_str(sell_order.get("expires_at")),
        sell_order_qty=_optional_float(sell_order.get("qty")),
    )
    if not attached:
        _append_reconciliation_result(
            rows,
            position_id=position_id,
            symbol=symbol,
            action="sell",
            status="superseded",
            buy_client_order_id=buy_client_order_id,
            sell_client_order_id=sell_client_order_id,
            qty=filled_qty,
            limit_price=target_sell_price,
            alpaca_order_id=sell_alpaca_order_id,
            message="open-order recovery belongs to a superseded managed GTC sell generation",
        )
        return

    observed_sell_qty, _, fill_metadata_is_valid = _sell_fill_metadata(sell_order)
    if not fill_metadata_is_valid or (
        sell_status == "filled" and observed_sell_qty <= _MANAGED_QTY_TOLERANCE
    ):
        _append_incomplete_sell_fill_result(
            conn=conn,
            rows=rows,
            position_id=position_id,
            symbol=symbol,
            buy_client_order_id=buy_client_order_id,
            sell_client_order_id=sell_client_order_id,
            observed_sell_qty=observed_sell_qty,
            target_sell_price=target_sell_price,
            sell_alpaca_order_id=sell_alpaca_order_id,
        )
        return
    remaining_qty = filled_qty
    if observed_sell_qty > _MANAGED_QTY_TOLERANCE:
        try:
            _, remaining_qty, generation_is_current = _record_filled_managed_sell(
                conn=conn,
                position_id=position_id,
                sell_order=sell_order,
                sell_alpaca_order_id=sell_alpaca_order_id,
                sell_client_order_id=sell_client_order_id,
                sell_status=sell_status,
                close_on_complete=close_on_complete and intent_matches,
            )
        except SellFillQuantityRegressionError:
            _append_sell_fill_quantity_regression_result(
                conn=conn,
                rows=rows,
                position_id=position_id,
                symbol=symbol,
                buy_client_order_id=buy_client_order_id,
                sell_client_order_id=sell_client_order_id,
                observed_sell_qty=observed_sell_qty,
                target_sell_price=target_sell_price,
                sell_alpaca_order_id=sell_alpaca_order_id,
            )
            return
        if not generation_is_current:
            return
        if remaining_qty < -_MANAGED_QTY_TOLERANCE:
            _append_overfilled_sell_result(
                conn=conn,
                rows=rows,
                position_id=position_id,
                symbol=symbol,
                buy_client_order_id=buy_client_order_id,
                sell_client_order_id=sell_client_order_id,
                observed_sell_qty=observed_sell_qty,
                target_sell_price=target_sell_price,
                sell_alpaca_order_id=sell_alpaca_order_id,
            )
            return
        if sell_status != "filled" and _managed_sell_is_complete(remaining_qty):
            _append_active_sell_completion_mismatch_result(
                conn=conn,
                rows=rows,
                position_id=position_id,
                symbol=symbol,
                buy_client_order_id=buy_client_order_id,
                sell_client_order_id=sell_client_order_id,
                observed_sell_qty=observed_sell_qty,
                target_sell_price=target_sell_price,
                sell_alpaca_order_id=sell_alpaca_order_id,
            )
            return
    if not intent_matches:
        _append_sell_submission_intent_mismatch_result(
            conn=conn,
            rows=rows,
            position_id=position_id,
            symbol=symbol,
            buy_client_order_id=buy_client_order_id,
            sell_client_order_id=sell_client_order_id,
            requested_qty=filled_qty,
            target_sell_price=target_sell_price,
            sell_order=sell_order,
        )
        return
    if sell_status in SELL_RENEWABLE_STATUSES and _gtc_sell_renewal_due(sell_order, cfg):
        _append_incomplete_sell_expiration_metadata_result(
            conn=conn,
            rows=rows,
            position_id=position_id,
            symbol=symbol,
            buy_client_order_id=buy_client_order_id,
            sell_client_order_id=sell_client_order_id,
            target_sell_price=target_sell_price,
            sell_alpaca_order_id=sell_alpaca_order_id,
            expiration_issue="recovered open managed GTC sell expiration remains inside the renewal window",
        )
        return
    _append_reconciliation_result(
        rows,
        position_id=position_id,
        symbol=symbol,
        action="sell",
        status=sell_status,
        buy_client_order_id=buy_client_order_id,
        sell_client_order_id=sell_client_order_id,
        qty=filled_qty,
        limit_price=target_sell_price,
        alpaca_order_id=sell_alpaca_order_id,
        message="recovered existing managed GTC sell order from Alpaca open orders",
    )


def _submit_replacement_gtc_sell(
    *,
    conn: sqlite3.Connection,
    client: AlpacaClient,
    rows: list[dict],
    position: dict,
    symbol: str,
    buy_client_order_id: str,
    filled_qty: float,
    target_sell_price: float,
    replacement_message: str,
    check_open_order: bool,
    close_on_complete: bool = True,
) -> None:
    position_id = int(position["id"])
    if not _is_whole_share_qty(float(filled_qty)):
        _append_fractional_sell_result(
            conn=conn,
            rows=rows,
            position_id=position_id,
            symbol=symbol,
            buy_client_order_id=buy_client_order_id,
            sell_client_order_id=_optional_str(position.get("sell_client_order_id")),
            filled_qty=filled_qty,
            target_sell_price=target_sell_price,
        )
        return

    if check_open_order:
        open_orders = client.open_orders()
        matching_orders = _matching_managed_open_sell_orders(
            open_orders,
            symbol,
            position_id,
            expected_alpaca_asset_id=_optional_str(position.get("alpaca_asset_id")),
            excluded_client_order_id=_optional_str(position.get("sell_client_order_id")),
            excluded_alpaca_order_id=_optional_str(position.get("sell_alpaca_order_id")),
        )
        if len(matching_orders) > 1:
            _append_multiple_managed_open_sells_result(
                rows=rows,
                position_id=position_id,
                symbol=symbol,
                buy_client_order_id=buy_client_order_id,
                sell_client_order_id=_optional_str(position.get("sell_client_order_id")),
                filled_qty=filled_qty,
                target_sell_price=target_sell_price,
            )
            return
        if len(matching_orders) == 1:
            _append_recovered_open_sell_result(
                conn=conn,
                rows=rows,
                position_id=position_id,
                symbol=symbol,
                buy_client_order_id=buy_client_order_id,
                sell_order=matching_orders[0],
                filled_qty=filled_qty,
                target_sell_price=target_sell_price,
                expected_sell_client_order_id=_optional_str(position.get("sell_client_order_id")),
                expected_sell_alpaca_order_id=_optional_str(position.get("sell_alpaca_order_id")),
                expected_renewal_count=_optional_int(position.get("sell_renewal_count")),
                cfg=client.cfg,
                close_on_complete=close_on_complete,
            )
            return
        if _orders_have_open_order(open_orders, symbol, "sell"):
            _append_open_sell_result(
                rows=rows,
                position_id=position_id,
                symbol=symbol,
                buy_client_order_id=buy_client_order_id,
                sell_client_order_id=_optional_str(position.get("sell_client_order_id")),
                filled_qty=filled_qty,
                target_sell_price=target_sell_price,
            )
            return

    next_renewal_count = _optional_int(position.get("sell_renewal_count")) + 1
    sell_client_order_id = _alpaca_exit_client_order_id(symbol, position_id, next_renewal_count)
    prior_sell_client_order_id = _optional_str(position.get("sell_client_order_id"))
    prior_sell_alpaca_order_id = _optional_str(position.get("sell_alpaca_order_id"))
    authoritative_remaining_qty = None
    if prior_sell_client_order_id and prior_sell_alpaca_order_id:
        authoritative_remaining_qty = claim_alpaca_managed_sell_replacement(
        conn,
        position_id,
        prior_sell_client_order_id=prior_sell_client_order_id or "",
        prior_sell_alpaca_order_id=prior_sell_alpaca_order_id or "",
        prior_renewal_count=next_renewal_count - 1,
        replacement_sell_client_order_id=sell_client_order_id,
        requested_remaining_qty=filled_qty,
        notes="managed GTC sell replacement submission claimed before broker request",
        )
    if authoritative_remaining_qty is None:
        _append_reconciliation_result(
            rows,
            position_id=position_id,
            symbol=symbol,
            action="sell",
            status="submission_pending",
            buy_client_order_id=buy_client_order_id,
            sell_client_order_id=sell_client_order_id,
            qty=filled_qty,
            limit_price=target_sell_price,
            alpaca_order_id=None,
            message="managed GTC sell replacement is already claimed by another workflow",
        )
        return
    _submit_managed_gtc_sell(
        conn=conn,
        client=client,
        rows=rows,
        position_id=position_id,
        symbol=symbol,
        buy_client_order_id=buy_client_order_id,
        sell_client_order_id=sell_client_order_id,
        filled_qty=authoritative_remaining_qty,
        target_sell_price=target_sell_price,
        increment_renewal_count=False,
        replacement_message=replacement_message,
        close_on_complete=close_on_complete,
        intent_already_persisted=True,
    )


def _request_gtc_sell_renewal(
    *,
    conn: sqlite3.Connection,
    client: AlpacaClient,
    rows: list[dict],
    position: dict,
    symbol: str,
    buy_client_order_id: str,
    filled_qty: float,
    target_sell_price: float,
    sell_alpaca_order_id: str,
    replacement_message: str = "renewed managed GTC limit sell before Alpaca aged-order expiration",
    renewal_note: str = "managed GTC sell renewal requested before expiration",
    pending_message: str = (
        "managed GTC sell expires soon; cancellation requested and replacement "
        "will be submitted after Alpaca confirms cancellation"
    ),
    close_on_complete: bool = True,
) -> None:
    position_id = int(position["id"])
    sell_client_order_id = _optional_str(position.get("sell_client_order_id"))
    now = _utc_now()
    requested_at = now.isoformat().replace("+00:00", "Z")
    reclaim_before = (now - _SELL_RENEWAL_CLAIM_LEASE).isoformat().replace("+00:00", "Z")
    if not sell_client_order_id or not claim_alpaca_managed_sell_renewal(
        conn,
        position_id,
        sell_client_order_id=sell_client_order_id,
        sell_alpaca_order_id=sell_alpaca_order_id,
        requested_at=requested_at,
        reclaim_before=reclaim_before,
        notes=renewal_note,
    ):
        _append_reconciliation_result(
            rows,
            position_id=position_id,
            symbol=symbol,
            action="sell",
            status="pending_cancel",
            buy_client_order_id=buy_client_order_id,
            sell_client_order_id=_optional_str(position.get("sell_client_order_id")),
            qty=filled_qty,
            limit_price=target_sell_price,
            alpaca_order_id=sell_alpaca_order_id,
            message="managed GTC sell renewal is already claimed by another workflow",
        )
        return
    cancel_error: requests.RequestException | None = None
    try:
        cancel_resp = client.cancel_order(sell_alpaca_order_id)
        cancel_resp.raise_for_status()
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
        cancel_error = exc
    except HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code not in {404, 422}:
            raise
        cancel_error = exc

    try:
        refreshed_resp = client.get_order(sell_alpaca_order_id)
        refreshed_resp.raise_for_status()
    except requests.RequestException:
        if cancel_error is not None:
            raise cancel_error
        raise
    refreshed_order = _response_json(refreshed_resp)
    refreshed_status = str(refreshed_order.get("status", "pending_cancel")).lower()
    prior_sell_client_order_id = _optional_str(position.get("sell_client_order_id"))
    if prior_sell_client_order_id and not _sell_order_identity_is_consistent(
        refreshed_order,
        expected_sell_client_order_id=prior_sell_client_order_id,
        expected_sell_alpaca_order_id=sell_alpaca_order_id,
    ):
        _append_incomplete_sell_order_identity_result(
            conn=conn,
            rows=rows,
            position_id=position_id,
            symbol=symbol,
            buy_client_order_id=buy_client_order_id,
            sell_client_order_id=prior_sell_client_order_id,
            observed_sell_qty=_sell_fill_metadata(refreshed_order)[0],
            target_sell_price=target_sell_price,
            sell_alpaca_order_id=_optional_str(refreshed_order.get("id")),
            identity_issue="cancellation refresh does not match the managed sell order",
        )
        return
    if not prior_sell_client_order_id or not update_alpaca_managed_sell_status_if_current(
        conn,
        position_id,
        expected_sell_client_order_id=prior_sell_client_order_id,
        sell_status=refreshed_status,
        sell_alpaca_order_id=_optional_str(refreshed_order.get("id")) or sell_alpaca_order_id,
        sell_submitted_at=_optional_str(refreshed_order.get("submitted_at")),
        sell_expires_at=_optional_alpaca_datetime_str(refreshed_order.get("expires_at")),
        sell_renewal_requested_at=requested_at,
        notes=renewal_note,
    ):
        _append_reconciliation_result(
            rows,
            position_id=position_id,
            symbol=symbol,
            action="sell",
            status="superseded",
            buy_client_order_id=buy_client_order_id,
            sell_client_order_id=prior_sell_client_order_id,
            qty=filled_qty,
            limit_price=target_sell_price,
            alpaca_order_id=sell_alpaca_order_id,
            message="broker observation belongs to a superseded managed GTC sell generation",
        )
        return
    remaining_qty = _managed_remaining_qty(position, filled_qty)
    observed_sell_qty, observed_sell_avg_price, fill_metadata_is_valid = _sell_fill_metadata(refreshed_order)
    if not fill_metadata_is_valid or (
        refreshed_status == "filled" and observed_sell_qty <= _MANAGED_QTY_TOLERANCE
    ):
        _append_incomplete_sell_fill_result(
            conn=conn,
            rows=rows,
            position_id=position_id,
            symbol=symbol,
            buy_client_order_id=buy_client_order_id,
            sell_client_order_id=_optional_str(position.get("sell_client_order_id")),
            observed_sell_qty=observed_sell_qty,
            target_sell_price=target_sell_price,
            sell_alpaca_order_id=sell_alpaca_order_id,
        )
        return
    if observed_sell_qty > _MANAGED_QTY_TOLERANCE:
        try:
            remaining_qty, fill_generation_is_current = mark_alpaca_managed_sell_filled_if_current(
                conn,
                position_id,
                expected_sell_client_order_id=prior_sell_client_order_id,
                sell_status=refreshed_status,
                sell_filled_qty=observed_sell_qty,
                sell_filled_avg_price=observed_sell_avg_price,
                sell_filled_at=_optional_str(refreshed_order.get("filled_at")),
                sell_alpaca_order_id=_optional_str(refreshed_order.get("id")) or sell_alpaca_order_id,
                sell_submitted_at=_optional_str(refreshed_order.get("submitted_at")),
                sell_expires_at=_optional_alpaca_datetime_str(refreshed_order.get("expires_at")),
            )
        except SellFillQuantityRegressionError:
            _append_sell_fill_quantity_regression_result(
                conn=conn,
                rows=rows,
                position_id=position_id,
                symbol=symbol,
                buy_client_order_id=buy_client_order_id,
                sell_client_order_id=prior_sell_client_order_id,
                observed_sell_qty=observed_sell_qty,
                target_sell_price=target_sell_price,
                sell_alpaca_order_id=sell_alpaca_order_id,
            )
            return
        if not fill_generation_is_current:
            _append_reconciliation_result(
                rows,
                position_id=position_id,
                symbol=symbol,
                action="sell",
                status="superseded",
                buy_client_order_id=buy_client_order_id,
                sell_client_order_id=prior_sell_client_order_id,
                qty=observed_sell_qty,
                limit_price=target_sell_price,
                alpaca_order_id=sell_alpaca_order_id,
                message="fill observation belongs to a superseded managed GTC sell generation",
            )
            return
    if remaining_qty < -_MANAGED_QTY_TOLERANCE:
        _append_overfilled_sell_result(
            conn=conn,
            rows=rows,
            position_id=position_id,
            symbol=symbol,
            buy_client_order_id=buy_client_order_id,
            sell_client_order_id=_optional_str(position.get("sell_client_order_id")),
            observed_sell_qty=observed_sell_qty,
            target_sell_price=target_sell_price,
            sell_alpaca_order_id=sell_alpaca_order_id,
        )
        return
    if refreshed_status != "filled" and _managed_sell_is_complete(remaining_qty):
        _append_active_sell_completion_mismatch_result(
            conn=conn,
            rows=rows,
            position_id=position_id,
            symbol=symbol,
            buy_client_order_id=buy_client_order_id,
            sell_client_order_id=prior_sell_client_order_id,
            observed_sell_qty=observed_sell_qty,
            target_sell_price=target_sell_price,
            sell_alpaca_order_id=sell_alpaca_order_id,
        )
        return
    if _managed_sell_is_complete(remaining_qty):
        if close_on_complete:
            closed = close_alpaca_managed_position_if_current_and_complete(
                conn,
                position_id,
                expected_sell_client_order_id=prior_sell_client_order_id,
                closed_at=_optional_str(refreshed_order.get("filled_at")),
                notes="managed target sell filled",
            )
            if not closed:
                _append_reconciliation_result(
                    rows,
                    position_id=position_id,
                    symbol=symbol,
                    action="sell",
                    status="superseded",
                    buy_client_order_id=buy_client_order_id,
                    sell_client_order_id=prior_sell_client_order_id,
                    qty=observed_sell_qty,
                    limit_price=target_sell_price,
                    alpaca_order_id=sell_alpaca_order_id,
                    message="position closure belongs to a superseded managed GTC sell generation",
                )
                return
        _append_reconciliation_result(
            rows,
            position_id=position_id,
            symbol=symbol,
            action="sell",
            status="filled",
            buy_client_order_id=buy_client_order_id,
            sell_client_order_id=_optional_str(position.get("sell_client_order_id")),
            qty=observed_sell_qty,
            limit_price=target_sell_price,
            alpaca_order_id=sell_alpaca_order_id,
            message=(
                "managed target sell filled; position closed"
                if close_on_complete
                else "managed sell filled; awaiting final status of the still-open parent buy"
            ),
        )
        return
    if refreshed_status == "filled":
        mismatch_generation_is_current = update_alpaca_managed_sell_status_if_current(
            conn,
            position_id,
            expected_sell_client_order_id=prior_sell_client_order_id,
            sell_status="quantity_mismatch",
            notes="filled sell quantity did not close the managed buy quantity; manual review required",
        )
        if not mismatch_generation_is_current:
            _append_reconciliation_result(
                rows,
                position_id=position_id,
                symbol=symbol,
                action="sell",
                status="superseded",
                buy_client_order_id=buy_client_order_id,
                sell_client_order_id=prior_sell_client_order_id,
                qty=observed_sell_qty,
                limit_price=target_sell_price,
                alpaca_order_id=sell_alpaca_order_id,
                message="quantity mismatch belongs to a superseded managed GTC sell generation",
            )
            return
        _append_reconciliation_result(
            rows,
            position_id=position_id,
            symbol=symbol,
            action="sell",
            status=refreshed_status,
            buy_client_order_id=buy_client_order_id,
            sell_client_order_id=_optional_str(position.get("sell_client_order_id")),
            qty=observed_sell_qty,
            limit_price=target_sell_price,
            alpaca_order_id=sell_alpaca_order_id,
            message="managed sell fill quantity does not match the managed buy; position remains active for review",
        )
        return
    if refreshed_status in SELL_REPLACEMENT_SAFE_STATUSES:
        _submit_replacement_gtc_sell(
            conn=conn,
            client=client,
            rows=rows,
            position=position,
            symbol=symbol,
            buy_client_order_id=buy_client_order_id,
            filled_qty=remaining_qty,
            target_sell_price=target_sell_price,
            replacement_message=replacement_message,
            check_open_order=True,
            close_on_complete=close_on_complete,
        )
        return

    if refreshed_status in SELL_INACTIVE_STATUSES:
        pending_message = (
            f"managed GTC sell entered {refreshed_status}; replacement was not submitted because "
            "Alpaca has not confirmed cancellation or expiration"
        )

    _append_reconciliation_result(
        rows,
        position_id=position_id,
        symbol=symbol,
        action="sell",
        status="pending_cancel" if refreshed_status in SELL_CANCEL_PENDING_STATUSES else refreshed_status,
        buy_client_order_id=buy_client_order_id,
        sell_client_order_id=_optional_str(position.get("sell_client_order_id")),
        qty=filled_qty,
        limit_price=target_sell_price,
        alpaca_order_id=sell_alpaca_order_id,
        message=pending_message,
    )


def reconcile_alpaca_managed_positions(
    conn: sqlite3.Connection,
    cfg: AlpacaOrderConfig,
) -> pd.DataFrame:
    columns = [
        "Position ID",
        "Workflow",
        "Asset",
        "Action",
        "Status",
        "Buy Client Order ID",
        "Sell Client Order ID",
        "Qty",
        "Limit Price",
        "Alpaca Order ID",
        "Message",
    ]
    if not (cfg.enabled or cfg.sell_enabled):
        return pd.DataFrame(columns=columns)

    positions = load_alpaca_managed_positions(conn, active_only=True)
    if positions.empty:
        return pd.DataFrame(columns=columns)
    workflow_by_position_id = {
        int(position["id"]): _optional_str(position.get("workflow"))
        for position in positions.to_dict("records")
    }

    client = AlpacaClient(cfg)
    rows = []
    for position in positions.to_dict("records"):
        position_id = int(position["id"])
        symbol = str(position["symbol"])
        buy_client_order_id = str(position["buy_client_order_id"])
        sell_client_order_id = _optional_str(position.get("sell_client_order_id"))
        current_sell_status = str(position.get("sell_status") or "").lower()
        try:
            filled_qty = _optional_float(position.get("filled_qty"))
            filled_avg_price = _optional_float(position.get("filled_avg_price"))
            target_sell_price = _optional_float(position.get("target_sell_price"))
            prior_filled_qty = filled_qty
            parent_buy_fill_increased = False

            buy_status = str(position.get("buy_status", "submission_pending")).lower()
            buy_is_open = buy_status not in {"filled", *BUY_TERMINAL_STATUSES}
            if filled_qty is None or filled_avg_price is None or target_sell_price is None or buy_is_open:
                try:
                    buy_order = _fetch_order_payload(
                        client,
                        _optional_str(position.get("buy_alpaca_order_id")),
                        buy_client_order_id,
                    )
                except HTTPError as exc:
                    if (
                        buy_status in {"submission_pending", "submission_unknown"}
                        and exc.response is not None
                        and exc.response.status_code == 404
                    ):
                        if _buy_submission_visibility_lease_active(position, cfg):
                            _append_reconciliation_result(
                                rows,
                                position_id=position_id,
                                symbol=symbol,
                                action="buy",
                                status=buy_status,
                                buy_client_order_id=buy_client_order_id,
                                sell_client_order_id=sell_client_order_id,
                                qty=None,
                                limit_price=None,
                                alpaca_order_id=None,
                                message=(
                                    "managed buy submission is not visible yet; "
                                    "the owner visibility lease is still active"
                                ),
                            )
                            continue
                        update_alpaca_managed_buy_status(
                            conn,
                            position_id,
                            buy_status="submission_not_found",
                            notes="managed buy submission was not found by client order ID after recovery",
                        )
                        close_alpaca_managed_position(
                            conn,
                            position_id,
                            closed_at=None,
                            notes="managed buy submission did not reach Alpaca",
                        )
                        _append_reconciliation_result(
                            rows,
                            position_id=position_id,
                            symbol=symbol,
                            action="buy",
                            status="submission_not_found",
                            buy_client_order_id=buy_client_order_id,
                            sell_client_order_id=sell_client_order_id,
                            qty=None,
                            limit_price=None,
                            alpaca_order_id=None,
                            message="managed buy submission was not found; intent closed without an Alpaca order",
                        )
                        continue
                    raise
                buy_status = str(buy_order.get("status", "unknown")).lower()
                buy_alpaca_order_id = _optional_str(buy_order.get("id"))
                buy_submitted_at = _optional_str(buy_order.get("submitted_at"))
                order_filled_qty = _optional_float(buy_order.get("filled_qty")) or 0.0

                buy_is_open = buy_status not in {"filled", *BUY_TERMINAL_STATUSES}
                if order_filled_qty > _MANAGED_QTY_TOLERANCE:
                    order_filled_avg_price = _optional_float(buy_order.get("filled_avg_price"))
                    if order_filled_qty <= 0 or order_filled_avg_price is None or order_filled_avg_price <= 0:
                        raise ValueError("filled Alpaca buy order is missing filled_qty or filled_avg_price")
                    parent_buy_fill_increased = (
                        prior_filled_qty is not None
                        and order_filled_qty > prior_filled_qty + _MANAGED_QTY_TOLERANCE
                    )
                    filled_qty = order_filled_qty
                    filled_avg_price = order_filled_avg_price
                    target_sell_price = _target_sell_price(
                        filled_avg_price,
                        float(position["profit_target_multiple"]),
                    )
                    buy_fill_is_current = mark_alpaca_managed_buy_filled(
                        conn,
                        position_id,
                        buy_status=buy_status,
                        filled_qty=filled_qty,
                        filled_avg_price=filled_avg_price,
                        filled_at=_optional_str(buy_order.get("filled_at")),
                        target_sell_price=target_sell_price,
                        buy_alpaca_order_id=buy_alpaca_order_id,
                        buy_submitted_at=buy_submitted_at,
                    )
                    if not buy_fill_is_current:
                        _append_reconciliation_result(
                            rows,
                            position_id=position_id,
                            symbol=symbol,
                            action="buy",
                            status="superseded",
                            buy_client_order_id=buy_client_order_id,
                            sell_client_order_id=sell_client_order_id,
                            qty=order_filled_qty,
                            limit_price=None,
                            alpaca_order_id=buy_alpaca_order_id,
                            message="parent-buy fill observation moved backward and was ignored",
                        )
                        continue
                    position["filled_qty"] = filled_qty
                    position["filled_avg_price"] = filled_avg_price
                    position["target_sell_price"] = target_sell_price
                    position["buy_status"] = buy_status
                    _append_reconciliation_result(
                        rows,
                        position_id=position_id,
                        symbol=symbol,
                        action="buy",
                        status="partially_filled" if buy_is_open else buy_status,
                        buy_client_order_id=buy_client_order_id,
                        sell_client_order_id=sell_client_order_id,
                        qty=filled_qty,
                        limit_price=target_sell_price,
                        alpaca_order_id=buy_alpaca_order_id,
                        message=(
                            "buy partially filled; managed target sell covers the confirmed shares"
                            if buy_is_open
                            else "buy filled; target sell price frozen from actual fill"
                        ),
                    )
                elif buy_status in BUY_TERMINAL_STATUSES:
                    update_alpaca_managed_buy_status(
                        conn,
                        position_id,
                        buy_status=buy_status,
                        buy_alpaca_order_id=buy_alpaca_order_id,
                        buy_submitted_at=buy_submitted_at,
                    )
                    close_alpaca_managed_position(
                        conn,
                        position_id,
                        closed_at=_optional_str(buy_order.get("updated_at")),
                        notes="buy order terminated without a filled position",
                    )
                    _append_reconciliation_result(
                        rows,
                        position_id=position_id,
                        symbol=symbol,
                        action="buy",
                        status=buy_status,
                        buy_client_order_id=buy_client_order_id,
                        sell_client_order_id=sell_client_order_id,
                        qty=None,
                        limit_price=None,
                        alpaca_order_id=buy_alpaca_order_id,
                        message="buy order terminated without a filled position",
                    )
                    continue
                else:
                    update_alpaca_managed_buy_status(
                        conn,
                        position_id,
                        buy_status=buy_status,
                        buy_alpaca_order_id=buy_alpaca_order_id,
                        buy_submitted_at=buy_submitted_at,
                    )
                    _append_reconciliation_result(
                        rows,
                        position_id=position_id,
                        symbol=symbol,
                        action="buy",
                        status=buy_status,
                        buy_client_order_id=buy_client_order_id,
                        sell_client_order_id=sell_client_order_id,
                        qty=order_filled_qty if order_filled_qty > 0 else None,
                        limit_price=None,
                        alpaca_order_id=buy_alpaca_order_id,
                        message="buy order is not filled yet",
                    )
                    continue

            if not cfg.sell_enabled:
                _append_reconciliation_result(
                    rows,
                    position_id=position_id,
                    symbol=symbol,
                    action="sell",
                    status="disabled",
                    buy_client_order_id=buy_client_order_id,
                    sell_client_order_id=sell_client_order_id,
                    qty=filled_qty,
                    limit_price=target_sell_price,
                    alpaca_order_id=None,
                    message="managed limit sell submission is disabled",
                )
                continue

            if sell_client_order_id:
                try:
                    sell_order = _fetch_order_payload(
                        client,
                        _optional_str(position.get("sell_alpaca_order_id")),
                        sell_client_order_id,
                    )
                except HTTPError as exc:
                    if (
                        current_sell_status in {
                            "submission_pending", "submission_unknown", "submission_not_found", "submission_retrying"
                        }
                        and exc.response is not None
                        and exc.response.status_code == 404
                    ):
                        retry_now = _utc_now()
                        recovery_qty = claim_alpaca_managed_sell_submission_retry(
                            conn,
                            position_id,
                            sell_client_order_id=sell_client_order_id,
                            claimed_at=retry_now.isoformat().replace("+00:00", "Z"),
                            reclaim_before=(
                                retry_now - _SELL_SUBMISSION_RETRY_CLAIM_LEASE
                            ).isoformat().replace("+00:00", "Z"),
                            notes="managed sell submission was not found by client order ID after recovery",
                        )
                        if recovery_qty is None:
                            _append_reconciliation_result(
                                rows,
                                position_id=position_id,
                                symbol=symbol,
                                action="sell",
                                status="superseded",
                                buy_client_order_id=buy_client_order_id,
                                sell_client_order_id=sell_client_order_id,
                                qty=None,
                                limit_price=target_sell_price,
                                alpaca_order_id=None,
                                message="404 recovery was superseded or its safe retry quantity changed",
                            )
                            continue
                        _submit_managed_gtc_sell(
                            conn=conn,
                            client=client,
                            rows=rows,
                            position_id=position_id,
                            symbol=symbol,
                            buy_client_order_id=buy_client_order_id,
                            sell_client_order_id=sell_client_order_id,
                            filled_qty=recovery_qty,
                            target_sell_price=float(target_sell_price),
                            increment_renewal_count=False,
                            replacement_message=(
                                "resubmitted managed GTC limit sell after prior submission was not found"
                            ),
                            close_on_complete=not buy_is_open,
                            intent_already_persisted=True,
                        )
                        continue
                    raise
                sell_status = str(sell_order.get("status", "unknown")).lower()
                persisted_sell_alpaca_order_id = _optional_str(position.get("sell_alpaca_order_id"))
                if not _sell_order_identity_is_consistent(
                    sell_order,
                    expected_sell_client_order_id=sell_client_order_id,
                    expected_sell_alpaca_order_id=persisted_sell_alpaca_order_id,
                ):
                    _append_incomplete_sell_order_identity_result(
                        conn=conn,
                        rows=rows,
                        position_id=position_id,
                        symbol=symbol,
                        buy_client_order_id=buy_client_order_id,
                        sell_client_order_id=sell_client_order_id,
                        observed_sell_qty=_sell_fill_metadata(sell_order)[0],
                        target_sell_price=float(target_sell_price),
                        sell_alpaca_order_id=_optional_str(sell_order.get("id")),
                        identity_issue="broker response does not match the persisted managed sell order",
                    )
                    continue
                sell_alpaca_order_id = (
                    _optional_str(sell_order.get("id"))
                    or persisted_sell_alpaca_order_id
                )
                sell_expires_at = _optional_alpaca_datetime_str(sell_order.get("expires_at"))
                sell_order_qty = _optional_positive_float(sell_order.get("qty"))
                sell_order_limit_price = _optional_positive_float(sell_order.get("limit_price"))
                observation_is_current = update_alpaca_managed_sell_status_if_current(
                    conn,
                    position_id,
                    expected_sell_client_order_id=sell_client_order_id,
                    sell_status=sell_status,
                    sell_alpaca_order_id=sell_alpaca_order_id,
                    sell_submitted_at=_optional_str(sell_order.get("submitted_at")),
                    sell_expires_at=sell_expires_at,
                )
                if not observation_is_current:
                    _append_reconciliation_result(
                        rows,
                        position_id=position_id,
                        symbol=symbol,
                        action="sell",
                        status="superseded",
                        buy_client_order_id=buy_client_order_id,
                        sell_client_order_id=sell_client_order_id,
                        qty=None,
                        limit_price=target_sell_price,
                        alpaca_order_id=sell_alpaca_order_id,
                        message="broker observation belongs to a superseded managed GTC sell generation",
                    )
                    continue
                remaining_qty = _managed_remaining_qty(position, float(filled_qty))
                observed_sell_qty, observed_sell_avg_price, fill_metadata_is_valid = _sell_fill_metadata(sell_order)
                if not fill_metadata_is_valid or (
                    sell_status == "filled" and observed_sell_qty <= _MANAGED_QTY_TOLERANCE
                ):
                    _append_incomplete_sell_fill_result(
                        conn=conn,
                        rows=rows,
                        position_id=position_id,
                        symbol=symbol,
                        buy_client_order_id=buy_client_order_id,
                        sell_client_order_id=sell_client_order_id,
                        observed_sell_qty=observed_sell_qty,
                        target_sell_price=float(target_sell_price),
                        sell_alpaca_order_id=sell_alpaca_order_id,
                    )
                    continue
                if observed_sell_qty > _MANAGED_QTY_TOLERANCE and not sell_alpaca_order_id:
                    _append_incomplete_sell_order_identity_result(
                        conn=conn,
                        rows=rows,
                        position_id=position_id,
                        symbol=symbol,
                        buy_client_order_id=buy_client_order_id,
                        sell_client_order_id=sell_client_order_id,
                        observed_sell_qty=observed_sell_qty,
                        target_sell_price=float(target_sell_price),
                    )
                    continue
                if observed_sell_qty > _MANAGED_QTY_TOLERANCE:
                    try:
                        remaining_qty, fill_generation_is_current = mark_alpaca_managed_sell_filled_if_current(
                            conn,
                            position_id,
                            expected_sell_client_order_id=sell_client_order_id,
                            sell_status=sell_status,
                            sell_filled_qty=observed_sell_qty,
                            sell_filled_avg_price=observed_sell_avg_price,
                            sell_filled_at=_optional_str(sell_order.get("filled_at")),
                            sell_alpaca_order_id=sell_alpaca_order_id,
                            sell_submitted_at=_optional_str(sell_order.get("submitted_at")),
                            sell_expires_at=sell_expires_at,
                        )
                    except SellFillQuantityRegressionError:
                        _append_sell_fill_quantity_regression_result(
                            conn=conn,
                            rows=rows,
                            position_id=position_id,
                            symbol=symbol,
                            buy_client_order_id=buy_client_order_id,
                            sell_client_order_id=sell_client_order_id,
                            observed_sell_qty=observed_sell_qty,
                            target_sell_price=float(target_sell_price),
                            sell_alpaca_order_id=sell_alpaca_order_id,
                        )
                        continue
                    if not fill_generation_is_current:
                        _append_reconciliation_result(
                            rows,
                            position_id=position_id,
                            symbol=symbol,
                            action="sell",
                            status="superseded",
                            buy_client_order_id=buy_client_order_id,
                            sell_client_order_id=sell_client_order_id,
                            qty=observed_sell_qty,
                            limit_price=target_sell_price,
                            alpaca_order_id=sell_alpaca_order_id,
                            message="fill observation belongs to a superseded managed GTC sell generation",
                        )
                        continue
                if remaining_qty < -_MANAGED_QTY_TOLERANCE:
                    _append_overfilled_sell_result(
                        conn=conn,
                        rows=rows,
                        position_id=position_id,
                        symbol=symbol,
                        buy_client_order_id=buy_client_order_id,
                        sell_client_order_id=sell_client_order_id,
                        observed_sell_qty=observed_sell_qty,
                        target_sell_price=float(target_sell_price),
                        sell_alpaca_order_id=sell_alpaca_order_id,
                    )
                    continue
                if sell_status != "filled" and _managed_sell_is_complete(remaining_qty):
                    _append_active_sell_completion_mismatch_result(
                        conn=conn,
                        rows=rows,
                        position_id=position_id,
                        symbol=symbol,
                        buy_client_order_id=buy_client_order_id,
                        sell_client_order_id=sell_client_order_id,
                        observed_sell_qty=observed_sell_qty,
                        target_sell_price=float(target_sell_price),
                        sell_alpaca_order_id=sell_alpaca_order_id,
                    )
                    continue
                if _managed_sell_is_complete(remaining_qty):
                    if not buy_is_open:
                        closed = close_alpaca_managed_position_if_current_and_complete(
                            conn,
                            position_id,
                            expected_sell_client_order_id=sell_client_order_id,
                            closed_at=_optional_str(sell_order.get("filled_at")),
                            notes="managed target sell filled",
                        )
                        if not closed:
                            _append_reconciliation_result(
                                rows,
                                position_id=position_id,
                                symbol=symbol,
                                action="sell",
                                status="superseded",
                                buy_client_order_id=buy_client_order_id,
                                sell_client_order_id=sell_client_order_id,
                                qty=observed_sell_qty,
                                limit_price=target_sell_price,
                                alpaca_order_id=sell_alpaca_order_id,
                                message="position closure belongs to a superseded managed GTC sell generation",
                            )
                            continue
                    _append_reconciliation_result(
                        rows,
                        position_id=position_id,
                        symbol=symbol,
                        action="sell",
                        status="filled",
                        buy_client_order_id=buy_client_order_id,
                        sell_client_order_id=sell_client_order_id,
                        qty=observed_sell_qty,
                        limit_price=target_sell_price,
                        alpaca_order_id=sell_alpaca_order_id,
                        message=(
                            "managed sell filled; awaiting final status of the still-open parent buy"
                            if buy_is_open
                            else "managed target sell filled; position closed"
                        ),
                    )
                    continue
                if sell_status == "filled":
                    if buy_is_open or parent_buy_fill_increased:
                        _submit_replacement_gtc_sell(
                            conn=conn,
                            client=client,
                            rows=rows,
                            position=position,
                            symbol=symbol,
                            buy_client_order_id=buy_client_order_id,
                            filled_qty=remaining_qty,
                            target_sell_price=float(target_sell_price),
                            replacement_message=(
                                "parent buy filled additional shares after the prior managed sell completed; "
                                "submitted a replacement for the remaining shares"
                            ),
                            check_open_order=True,
                            close_on_complete=not buy_is_open,
                        )
                        continue
                    mismatch_generation_is_current = update_alpaca_managed_sell_status_if_current(
                        conn,
                        position_id,
                        expected_sell_client_order_id=sell_client_order_id,
                        sell_status="quantity_mismatch",
                        notes="filled sell quantity did not close the managed buy quantity; manual review required",
                    )
                    if not mismatch_generation_is_current:
                        _append_reconciliation_result(
                            rows,
                            position_id=position_id,
                            symbol=symbol,
                            action="sell",
                            status="superseded",
                            buy_client_order_id=buy_client_order_id,
                            sell_client_order_id=sell_client_order_id,
                            qty=observed_sell_qty,
                            limit_price=target_sell_price,
                            alpaca_order_id=sell_alpaca_order_id,
                            message="quantity mismatch belongs to a superseded managed GTC sell generation",
                        )
                        continue
                    _append_reconciliation_result(
                        rows,
                        position_id=position_id,
                        symbol=symbol,
                        action="sell",
                        status=sell_status,
                        buy_client_order_id=buy_client_order_id,
                        sell_client_order_id=sell_client_order_id,
                        qty=observed_sell_qty,
                        limit_price=target_sell_price,
                        alpaca_order_id=sell_alpaca_order_id,
                        message=(
                            "managed sell fill quantity does not match the managed buy; "
                            "position remains active for review"
                        ),
                    )
                    continue
                if sell_status in SELL_RENEWABLE_STATUSES and (
                    sell_order_qty is None or sell_order_limit_price is None
                ):
                    _append_incomplete_sell_order_metadata_result(
                        conn=conn,
                        rows=rows,
                        position_id=position_id,
                        symbol=symbol,
                        buy_client_order_id=buy_client_order_id,
                        sell_client_order_id=sell_client_order_id,
                        target_sell_price=float(target_sell_price),
                        sell_alpaca_order_id=sell_alpaca_order_id,
                    )
                    continue
                open_sell_remaining_qty = (
                    None
                    if sell_order_qty is None
                    else max(sell_order_qty - observed_sell_qty, 0.0)
                )
                needs_sell_replacement = (
                    (
                        open_sell_remaining_qty is not None
                        and abs(open_sell_remaining_qty - remaining_qty) > _MANAGED_QTY_TOLERANCE
                    )
                    or (
                        sell_order_limit_price is not None
                        and abs(sell_order_limit_price - float(target_sell_price))
                        > _alpaca_limit_price_tolerance(float(target_sell_price))
                    )
                )
                if sell_status in SELL_RENEWABLE_STATUSES and sell_alpaca_order_id and needs_sell_replacement:
                    _request_gtc_sell_renewal(
                        conn=conn,
                        client=client,
                        rows=rows,
                        position=position,
                        symbol=symbol,
                        buy_client_order_id=buy_client_order_id,
                        filled_qty=float(filled_qty),
                        target_sell_price=float(target_sell_price),
                        sell_alpaca_order_id=sell_alpaca_order_id,
                        replacement_message=(
                            "updated managed GTC sell after the managed buy fill quantity or price changed"
                        ),
                        renewal_note="managed GTC sell replacement requested after managed buy fill changed",
                        pending_message=(
                            "managed GTC sell replacement requested; awaiting Alpaca cancellation confirmation"
                        ),
                        close_on_complete=not buy_is_open,
                    )
                    continue
                renewal_was_requested = _optional_str(position.get("sell_renewal_requested_at")) is not None
                if sell_status in SELL_INACTIVE_STATUSES:
                    should_resubmit = (
                        sell_status == "expired"
                        and (cfg.gtc_sell_renewal_enabled or renewal_was_requested)
                    ) or (
                        sell_status == "canceled" and renewal_was_requested
                    )
                    if should_resubmit:
                        _submit_replacement_gtc_sell(
                            conn=conn,
                            client=client,
                            rows=rows,
                            position=position,
                            symbol=symbol,
                            buy_client_order_id=buy_client_order_id,
                            filled_qty=remaining_qty,
                            target_sell_price=float(target_sell_price),
                            replacement_message=(
                                "prior managed GTC sell expired; submitted replacement at frozen target price"
                                if sell_status == "expired"
                                else "renewed managed GTC limit sell after Alpaca confirmed cancellation"
                            ),
                            check_open_order=True,
                            close_on_complete=not buy_is_open,
                        )
                        continue
                    _append_reconciliation_result(
                        rows,
                        position_id=position_id,
                        symbol=symbol,
                        action="sell",
                        status=sell_status,
                        buy_client_order_id=buy_client_order_id,
                        sell_client_order_id=sell_client_order_id,
                        qty=filled_qty,
                        limit_price=target_sell_price,
                        alpaca_order_id=sell_alpaca_order_id,
                        message="managed GTC sell is no longer active; no automatic resubmission",
                    )
                    continue
                if sell_status in SELL_CANCEL_PENDING_STATUSES:
                    if (
                        (cfg.gtc_sell_renewal_enabled or renewal_was_requested)
                        and sell_alpaca_order_id
                        and not _sell_renewal_claim_lease_active(position)
                    ):
                        _request_gtc_sell_renewal(
                            conn=conn,
                            client=client,
                            rows=rows,
                            position=position,
                            symbol=symbol,
                            buy_client_order_id=buy_client_order_id,
                            filled_qty=float(filled_qty),
                            target_sell_price=float(target_sell_price),
                            sell_alpaca_order_id=sell_alpaca_order_id,
                            pending_message=(
                                "managed GTC sell cancellation is still pending after retry; "
                                "replacement not submitted yet"
                            ),
                            close_on_complete=not buy_is_open,
                        )
                        continue
                    _append_reconciliation_result(
                        rows,
                        position_id=position_id,
                        symbol=symbol,
                        action="sell",
                        status="pending_cancel",
                        buy_client_order_id=buy_client_order_id,
                        sell_client_order_id=sell_client_order_id,
                        qty=filled_qty,
                        limit_price=target_sell_price,
                        alpaca_order_id=sell_alpaca_order_id,
                        message="managed GTC sell cancellation is pending; replacement not submitted yet",
                    )
                    continue
                if (
                    cfg.gtc_sell_renewal_enabled
                    and cfg.gtc_sell_renewal_days_before_expiration >= 0
                    and sell_status in SELL_RENEWABLE_STATUSES
                    and sell_alpaca_order_id
                    and _parse_alpaca_datetime(sell_expires_at) is None
                    and _parse_alpaca_datetime(position.get("sell_expires_at")) is None
                ):
                    _append_incomplete_sell_expiration_metadata_result(
                        conn=conn,
                        rows=rows,
                        position_id=position_id,
                        symbol=symbol,
                        buy_client_order_id=buy_client_order_id,
                        sell_client_order_id=sell_client_order_id,
                        target_sell_price=float(target_sell_price),
                        sell_alpaca_order_id=sell_alpaca_order_id,
                    )
                    continue
                if sell_alpaca_order_id and _gtc_sell_renewal_due(
                    sell_order,
                    cfg,
                    persisted_expires_at=position.get("sell_expires_at"),
                ):
                    _request_gtc_sell_renewal(
                        conn=conn,
                        client=client,
                        rows=rows,
                        position=position,
                        symbol=symbol,
                        buy_client_order_id=buy_client_order_id,
                        filled_qty=float(filled_qty),
                        target_sell_price=float(target_sell_price),
                        sell_alpaca_order_id=sell_alpaca_order_id,
                        close_on_complete=not buy_is_open,
                    )
                    continue
                _append_reconciliation_result(
                    rows,
                    position_id=position_id,
                    symbol=symbol,
                    action="sell",
                    status=sell_status,
                    buy_client_order_id=buy_client_order_id,
                    sell_client_order_id=sell_client_order_id,
                    qty=filled_qty,
                    limit_price=target_sell_price,
                    alpaca_order_id=sell_alpaca_order_id,
                    message="managed GTC sell order already submitted",
                )
                continue

            open_orders = client.open_orders()
            matching_orders = _matching_managed_open_sell_orders(
                open_orders,
                symbol,
                position_id,
                expected_alpaca_asset_id=_optional_str(position.get("alpaca_asset_id")),
            )
            if len(matching_orders) > 1:
                _append_multiple_managed_open_sells_result(
                    rows=rows,
                    position_id=position_id,
                    symbol=symbol,
                    buy_client_order_id=buy_client_order_id,
                    sell_client_order_id=sell_client_order_id,
                    filled_qty=float(filled_qty),
                    target_sell_price=float(target_sell_price),
                )
                continue
            if len(matching_orders) == 1:
                _append_recovered_open_sell_result(
                    conn=conn,
                    rows=rows,
                    position_id=position_id,
                    symbol=symbol,
                    buy_client_order_id=buy_client_order_id,
                    sell_order=matching_orders[0],
                    filled_qty=float(filled_qty),
                    target_sell_price=float(target_sell_price),
                    expected_sell_client_order_id=_optional_str(position.get("sell_client_order_id")),
                    expected_sell_alpaca_order_id=_optional_str(position.get("sell_alpaca_order_id")),
                    expected_renewal_count=_optional_int(position.get("sell_renewal_count")),
                    cfg=cfg,
                )
                continue

            if _orders_have_open_order(open_orders, symbol, "sell"):
                _append_open_sell_result(
                    rows=rows,
                    position_id=position_id,
                    symbol=symbol,
                    buy_client_order_id=buy_client_order_id,
                    sell_client_order_id=sell_client_order_id,
                    filled_qty=float(filled_qty),
                    target_sell_price=float(target_sell_price),
                )
                continue

            if not _is_whole_share_qty(float(filled_qty)):
                _append_fractional_sell_result(
                    conn=conn,
                    rows=rows,
                    position_id=position_id,
                    symbol=symbol,
                    buy_client_order_id=buy_client_order_id,
                    sell_client_order_id=None,
                    filled_qty=float(filled_qty),
                    target_sell_price=float(target_sell_price),
                )
                continue

            sell_client_order_id = _alpaca_exit_client_order_id(symbol, position_id)
            _submit_managed_gtc_sell(
                conn=conn,
                client=client,
                rows=rows,
                position_id=position_id,
                symbol=symbol,
                buy_client_order_id=buy_client_order_id,
                sell_client_order_id=sell_client_order_id,
                filled_qty=float(filled_qty),
                target_sell_price=float(target_sell_price),
                increment_renewal_count=False,
                replacement_message="submitted managed GTC limit sell at frozen target price",
                close_on_complete=not buy_is_open,
            )
        except HTTPError as exc:
            _append_reconciliation_result(
                rows,
                position_id=position_id,
                symbol=symbol,
                action="reconcile",
                status="error",
                buy_client_order_id=buy_client_order_id,
                sell_client_order_id=sell_client_order_id,
                qty=None,
                limit_price=None,
                alpaca_order_id=None,
                message=_http_error_message(exc),
            )
        except Exception as exc:
            _append_reconciliation_result(
                rows,
                position_id=position_id,
                symbol=symbol,
                action="reconcile",
                status="error",
                buy_client_order_id=buy_client_order_id,
                sell_client_order_id=sell_client_order_id,
                qty=None,
                limit_price=None,
                alpaca_order_id=None,
                message=str(exc),
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=columns)
    result.insert(1, "Workflow", result["Position ID"].map(workflow_by_position_id))
    return result.reindex(columns=columns)


def migrate_alpaca_managed_position_symbols(
    conn: sqlite3.Connection,
    cfg: AlpacaOrderConfig,
) -> dict[str, str]:
    """Migrate active managed tickers using Alpaca's stable asset IDs.

    Returns an old-to-current symbol mapping. Existing broker order and client
    IDs are intentionally retained; only future submissions use the new ticker.
    """
    if not (cfg.enabled or cfg.sell_enabled):
        return {}
    positions = load_alpaca_managed_positions(conn, active_only=True)
    if positions.empty:
        return {}
    client = AlpacaClient(cfg)
    live_by_asset_id = {
        str(item["asset_id"]): str(item["symbol"]).upper()
        for item in client.positions()
        if item.get("asset_id") and item.get("symbol")
    }
    migration_plan: list[tuple[int, str, str, str]] = []
    for position in positions.to_dict("records"):
        prior_symbol = str(position["symbol"]).upper()
        asset_id = _optional_str(position.get("alpaca_asset_id"))
        current_symbol = live_by_asset_id.get(asset_id) if asset_id is not None else None
        if current_symbol is not None and current_symbol.upper() == prior_symbol:
            continue
        for order_id, expected_client_order_id, order_role in (
            (
                _optional_str(position.get("sell_alpaca_order_id")),
                _optional_str(position.get("sell_client_order_id")),
                "sell",
            ),
            (
                _optional_str(position.get("buy_alpaca_order_id")),
                _optional_str(position.get("buy_client_order_id")),
                "buy",
            ),
        ):
            if order_id is None:
                continue
            response = client.get_order(order_id)
            if response.status_code == 404:
                continue
            response.raise_for_status()
            payload = _response_json(response)
            observed_order_id = _optional_str(payload.get("id"))
            observed_client_order_id = _optional_str(payload.get("client_order_id"))
            if (
                expected_client_order_id is None
                or observed_order_id != order_id
                or observed_client_order_id != expected_client_order_id
                or (
                    order_role == "sell"
                    and _alpaca_exit_client_order_generation(
                        str(position["symbol"]),
                        int(position["id"]),
                        observed_client_order_id or "",
                    )
                    is None
                )
            ):
                raise ValueError(
                    f"Managed position {int(position['id'])} {order_role} order identity "
                    "does not match the persisted order; symbol migration is blocked for review"
                )
            observed_asset_id = _optional_str(payload.get("asset_id"))
            if observed_asset_id is None or (asset_id is not None and observed_asset_id != asset_id):
                raise ValueError(
                    f"Managed position {int(position['id'])} {order_role} order asset identity "
                    "does not match the persisted asset; symbol migration is blocked for review"
                )
            asset_id = observed_asset_id
            break
        if asset_id is None:
            continue
        current_symbol = live_by_asset_id.get(asset_id)
        if current_symbol is None:
            asset = client.asset(asset_id)
            observed_asset_id = _optional_str(asset.get("id"))
            current_symbol = _optional_str(asset.get("symbol"))
            if observed_asset_id != asset_id or current_symbol is None:
                raise ValueError(
                    f"Alpaca asset lookup for managed position {int(position['id'])} "
                    "does not match the persisted asset identity; symbol migration is blocked for review"
                )
        if current_symbol is None:
            continue
        migration_plan.append((int(position["id"]), prior_symbol, asset_id, current_symbol.upper()))

    asset_owners: dict[str, int] = {}
    symbol_owners: dict[str, int] = {}
    for position_id, _prior_symbol, asset_id, current_symbol in migration_plan:
        conflicting_position_id = asset_owners.get(asset_id) or symbol_owners.get(current_symbol)
        if conflicting_position_id is not None and conflicting_position_id != position_id:
            raise ValueError(
                f"Managed positions {conflicting_position_id} and {position_id} resolve to the same "
                f"Alpaca asset or current symbol ({current_symbol}); symbol migration was not applied"
            )
        asset_owners[asset_id] = position_id
        symbol_owners[current_symbol] = position_id

    migrations: dict[str, str] = {}
    conn.execute("SAVEPOINT alpaca_symbol_migration")
    try:
        for position_id, prior_symbol, asset_id, current_symbol in migration_plan:
            migrated = migrate_alpaca_managed_position_symbol(
                conn,
                position_id,
                alpaca_asset_id=asset_id,
                current_symbol=current_symbol,
                commit=False,
            )
            if not migrated:
                raise ValueError(
                    f"Could not safely migrate managed position {position_id} "
                    f"from {prior_symbol} to {current_symbol}; conflicting active symbol or asset identity"
                )
            if prior_symbol != current_symbol:
                migrations[prior_symbol] = current_symbol
    except Exception:
        conn.execute("ROLLBACK TO SAVEPOINT alpaca_symbol_migration")
        conn.execute("RELEASE SAVEPOINT alpaca_symbol_migration")
        raise
    conn.execute("RELEASE SAVEPOINT alpaca_symbol_migration")
    return migrations


def submit_alpaca_paper_buy_orders(
    buy_signals: pd.DataFrame,
    cfg: AlpacaOrderConfig,
    conn: sqlite3.Connection | None = None,
) -> pd.DataFrame:
    columns = [
        "Workflow",
        "Asset",
        "Date",
        "Client Order ID",
        "Notional",
        "Qty",
        "Limit Price",
        "Status",
        "Alpaca Order ID",
        "Message",
    ]
    if not cfg.enabled:
        return pd.DataFrame(columns=columns)
    if buy_signals.empty:
        return pd.DataFrame(columns=columns)

    managed_symbols = active_alpaca_managed_symbols(conn) if conn is not None else set()
    client: AlpacaClient | None = None
    market_clock: dict | None = None
    prior_session: date | None = None
    calendar_loaded = False
    rows = []
    eligible_orders: list[dict[str, object]] = []
    pending_batch_symbols: set[str] = set()
    pending_batch_client_order_ids: set[str] = set()
    workflow_by_signal_index: dict[int, str | None] = {}

    def append_preflight_result(
        signal_index: int,
        *,
        symbol: str,
        signal_date: str,
        client_order_id: str,
        status: str,
        message: str,
        alpaca_order_id: str | None = None,
        notional: float | None = None,
        qty: int | None = None,
        limit_price: float | None = None,
    ) -> None:
        rows.append(
            {
                "Workflow": workflow_by_signal_index.get(signal_index),
                "Asset": symbol,
                "Date": signal_date,
                "Client Order ID": client_order_id,
                "Notional": notional,
                "Qty": qty,
                "Limit Price": limit_price,
                "Status": status,
                "Alpaca Order ID": alpaca_order_id,
                "Message": message,
                "_signal_index": signal_index,
            }
        )

    # Phase one performs every per-symbol safety check, including quoting a
    # protected limit.  Only signals that can actually reach submission share
    # the account-level batch budget below.
    for signal_index, signal in enumerate(buy_signals.to_dict("records")):
        workflow_by_signal_index[signal_index] = _signal_workflow(signal)
        symbol = str(signal["Asset"])
        symbol_key = symbol.upper()
        signal_date = str(signal["Date"])
        client_order_id = _alpaca_client_order_id("buy", symbol, signal_date)

        try:
            if symbol_key in pending_batch_symbols or client_order_id in pending_batch_client_order_ids:
                append_preflight_result(
                    signal_index,
                    symbol=symbol,
                    signal_date=signal_date,
                    client_order_id=client_order_id,
                    status="duplicate_signal",
                    message="duplicate buy signal for symbol or client order ID in this batch; order skipped",
                )
                continue

            if symbol_key in managed_symbols:
                append_preflight_result(
                    signal_index,
                    symbol=symbol,
                    signal_date=signal_date,
                    client_order_id=client_order_id,
                    status="managed",
                    message="symbol already has an active managed Alpaca position",
                )
                continue

            if client is None:
                client = AlpacaClient(cfg)
            existing_resp = client.get_order_by_client_order_id(client_order_id)
            if existing_resp.status_code == 200:
                existing = existing_resp.json()
                if conn is not None and {"RSI Symbol", "Buy RSI", "Sell Return Multiple"}.issubset(signal):
                    save_alpaca_managed_buy_order(
                        conn,
                        workflow=_signal_workflow(signal),
                        symbol=symbol,
                        signal_symbol=str(signal["RSI Symbol"]),
                        buy_rsi=float(signal["Buy RSI"]),
                        profit_target_multiple=float(signal["Sell Return Multiple"]),
                        buy_signal_date=signal_date,
                        buy_client_order_id=client_order_id,
                        buy_alpaca_order_id=existing.get("id"),
                        buy_submitted_at=existing.get("submitted_at"),
                        buy_status=str(existing.get("status", "existing")).lower(),
                    )
                    managed_symbols.add(symbol.upper())
                append_preflight_result(
                    signal_index,
                    symbol=symbol,
                    signal_date=signal_date,
                    client_order_id=client_order_id,
                    status="existing",
                    alpaca_order_id=existing.get("id"),
                    message=existing.get("status", "order already exists"),
                )
                continue
            if existing_resp.status_code != 404:
                existing_resp.raise_for_status()

            open_orders = client.open_orders()
            if _orders_have_open_order(open_orders, symbol, "buy"):
                append_preflight_result(
                    signal_index,
                    symbol=symbol,
                    signal_date=signal_date,
                    client_order_id=client_order_id,
                    status="open_order",
                    message="open buy order already exists for symbol in Alpaca account",
                )
                continue

            if _orders_have_open_order(open_orders, symbol, "sell"):
                append_preflight_result(
                    signal_index,
                    symbol=symbol,
                    signal_date=signal_date,
                    client_order_id=client_order_id,
                    status="open_sell_order",
                    message="open sell order already exists for symbol in Alpaca account",
                )
                continue

            position_qty = client.position_qty(symbol)
            if position_qty != 0:
                append_preflight_result(
                    signal_index,
                    symbol=symbol,
                    signal_date=signal_date,
                    client_order_id=client_order_id,
                    status="held",
                    message=f"symbol already has an Alpaca position: qty={position_qty}",
                )
                continue

            asset = client.asset(symbol)
            if not asset.get("tradable", False):
                append_preflight_result(
                    signal_index,
                    symbol=symbol,
                    signal_date=signal_date,
                    client_order_id=client_order_id,
                    status="not_tradable",
                    message=f"symbol is not tradable through Alpaca: status={asset.get('status', 'unknown')}",
                )
                continue
            if asset.get("status") not in {None, "active"}:
                append_preflight_result(
                    signal_index,
                    symbol=symbol,
                    signal_date=signal_date,
                    client_order_id=client_order_id,
                    status="inactive",
                    message=f"symbol is not active through Alpaca: status={asset.get('status')}",
                )
                continue
            if market_clock is None:
                market_clock = client.market_clock()
            if not bool(market_clock.get("is_open", False)):
                next_open = _parse_alpaca_datetime(market_clock.get("next_open"))
                clock_timestamp = _parse_alpaca_datetime(market_clock.get("timestamp"))
                if next_open is None or clock_timestamp is None:
                    raise ValueError("Alpaca clock did not provide valid session timestamps")
                next_open_date = next_open.astimezone(_NEW_YORK).date()
                clock_date = clock_timestamp.astimezone(_NEW_YORK).date()
                if not calendar_loaded and next_open_date == clock_date:
                    calendar = client.calendar(
                        start=clock_timestamp.astimezone(_NEW_YORK).date() - timedelta(days=14),
                        end=next_open.astimezone(_NEW_YORK).date(),
                    )
                    prior_session = _prior_calendar_session(calendar, next_open)
                    calendar_loaded = True
            can_submit, defer_message = _buy_submission_window(market_clock, signal_date, prior_session)
            if not can_submit:
                append_preflight_result(
                    signal_index,
                    symbol=symbol,
                    signal_date=signal_date,
                    client_order_id=client_order_id,
                    status="deferred",
                    message=defer_message,
                )
                continue
            latest_price = _latest_market_price(symbol)
            if latest_price <= 0:
                raise ValueError(f"Latest market price must be positive; got {latest_price} for {symbol}.")
            if cfg.buy_limit_buffer_bps < 0:
                raise ValueError("Alpaca buy limit buffer must not be negative.")
            protected_limit = Decimal(str(latest_price)) * (
                Decimal("1") + Decimal(str(cfg.buy_limit_buffer_bps)) / Decimal("10000")
            )
            limit_price = float(_quantize_alpaca_limit_price(protected_limit, rounding=ROUND_FLOOR))
            if limit_price <= 0:
                raise ValueError(f"Calculated buy limit price must be positive; got {limit_price} for {symbol}.")
            pending_batch_symbols.add(symbol_key)
            pending_batch_client_order_ids.add(client_order_id)
            eligible_orders.append(
                {
                    "signal_index": signal_index,
                    "signal": signal,
                    "symbol": symbol,
                    "signal_date": signal_date,
                    "client_order_id": client_order_id,
                    "limit_price": limit_price,
                }
            )
        except HTTPError as exc:
            append_preflight_result(
                signal_index,
                symbol=symbol,
                signal_date=signal_date,
                client_order_id=client_order_id,
                status="error",
                message=_http_error_message(exc),
            )
        except Exception as exc:
            append_preflight_result(
                signal_index,
                symbol=symbol,
                signal_date=signal_date,
                client_order_id=client_order_id,
                status="error",
                message=str(exc),
            )

    if eligible_orders:
        assert client is not None
        try:
            batch_notional = client.cash_notional(eligible_buy_signal_count=len(eligible_orders))
        except HTTPError as exc:
            for order in eligible_orders:
                append_preflight_result(
                    int(order["signal_index"]),
                    symbol=str(order["symbol"]),
                    signal_date=str(order["signal_date"]),
                    client_order_id=str(order["client_order_id"]),
                    status="error",
                    message=_http_error_message(exc),
                    limit_price=float(order["limit_price"]),
                )
        except Exception as exc:
            for order in eligible_orders:
                append_preflight_result(
                    int(order["signal_index"]),
                    symbol=str(order["symbol"]),
                    signal_date=str(order["signal_date"]),
                    client_order_id=str(order["client_order_id"]),
                    status="error",
                    message=str(exc),
                    limit_price=float(order["limit_price"]),
                )
        else:
            remaining_batch_notional = batch_notional
            per_signal_budget = batch_notional / len(eligible_orders)
            for order in eligible_orders:
                signal_index = int(order["signal_index"])
                signal = order["signal"]
                assert isinstance(signal, dict)
                symbol = str(order["symbol"])
                signal_date = str(order["signal_date"])
                client_order_id = str(order["client_order_id"])
                limit_price = float(order["limit_price"])
                order_notional = min(per_signal_budget, remaining_batch_notional)
                order_qty: int | None = None
                if order_notional <= 0:
                    append_preflight_result(
                        signal_index,
                        symbol=symbol,
                        signal_date=signal_date,
                        client_order_id=client_order_id,
                        notional=0.0,
                        limit_price=limit_price,
                        status="batch_budget_exhausted",
                        message="the configured Alpaca buy-batch budget is exhausted",
                    )
                    continue

                order_qty = floor(order_notional / limit_price)
                if order_qty < 1:
                    append_preflight_result(
                        signal_index,
                        symbol=symbol,
                        signal_date=signal_date,
                        client_order_id=client_order_id,
                        notional=order_notional,
                        qty=order_qty,
                        limit_price=limit_price,
                        status="insufficient_notional",
                        message=(
                            "allocated buy-batch budget is below one whole share "
                            f"at protected limit price {limit_price:.4f}; buy order skipped"
                        ),
                    )
                    continue

                order_notional = round(order_qty * limit_price, 2)
                # Reserve capacity before the API request.  If the response is
                # ambiguous or fails after Alpaca accepted it, retrying cannot
                # accidentally over-allocate the configured batch budget.
                remaining_batch_notional = max(remaining_batch_notional - order_notional, 0.0)
                is_managed_signal = (
                    conn is not None and {"RSI Symbol", "Buy RSI", "Sell Return Multiple"}.issubset(signal)
                )
                managed_position_id: int | None = None
                submission_claimed = False
                if is_managed_signal:
                    managed_position_id, submission_claimed = claim_alpaca_managed_buy_intent(
                        conn,
                        workflow=_signal_workflow(signal),
                        symbol=symbol,
                        signal_symbol=str(signal["RSI Symbol"]),
                        buy_rsi=float(signal["Buy RSI"]),
                        profit_target_multiple=float(signal["Sell Return Multiple"]),
                        buy_signal_date=signal_date,
                        buy_client_order_id=client_order_id,
                        # Phase one already looked up this deterministic ID and
                        # received a 404, so only a previously verified missing
                        # intent may be reclaimed here.
                        allow_retry_after_not_found=True,
                    )
                    managed_symbols.add(symbol.upper())

                if is_managed_signal and not submission_claimed:
                    try:
                        existing_resp = client.get_order_by_client_order_id(client_order_id)
                    except requests.RequestException as exc:
                        append_preflight_result(
                            signal_index,
                            symbol=symbol,
                            signal_date=signal_date,
                            client_order_id=client_order_id,
                            notional=order_notional,
                            qty=order_qty,
                            limit_price=limit_price,
                            status="submission_pending",
                            message=(
                                "another workflow owns this managed buy intent; "
                                f"broker lookup is pending: {exc}"
                            ),
                        )
                        continue
                    if existing_resp.status_code == 200:
                        existing = _response_json(existing_resp)
                        save_alpaca_managed_buy_order(
                            conn,
                            workflow=_signal_workflow(signal),
                            symbol=symbol,
                            signal_symbol=str(signal["RSI Symbol"]),
                            buy_rsi=float(signal["Buy RSI"]),
                            profit_target_multiple=float(signal["Sell Return Multiple"]),
                            buy_signal_date=signal_date,
                            buy_client_order_id=client_order_id,
                            buy_alpaca_order_id=_optional_str(existing.get("id")),
                            buy_submitted_at=_optional_str(existing.get("submitted_at")),
                            buy_status=str(existing.get("status", "existing")).lower(),
                            notes="managed buy recovered by client order ID after another workflow claimed it",
                        )
                        append_preflight_result(
                            signal_index,
                            symbol=symbol,
                            signal_date=signal_date,
                            client_order_id=client_order_id,
                            notional=order_notional,
                            qty=order_qty,
                            limit_price=limit_price,
                            status="existing",
                            alpaca_order_id=_optional_str(existing.get("id")),
                            message="another workflow already submitted this managed buy",
                        )
                    else:
                        append_preflight_result(
                            signal_index,
                            symbol=symbol,
                            signal_date=signal_date,
                            client_order_id=client_order_id,
                            notional=order_notional,
                            qty=order_qty,
                            limit_price=limit_price,
                            status="submission_pending",
                            message=(
                                "another workflow owns this managed buy intent; "
                                "waiting for its broker submission to become visible"
                            ),
                        )
                    continue

                try:
                    submit_resp = client.submit_limit_buy_order(
                        symbol=symbol,
                        qty=order_qty,
                        limit_price=limit_price,
                        client_order_id=client_order_id,
                    )
                    submit_resp.raise_for_status()
                    submitted = submit_resp.json()
                    if is_managed_signal:
                        save_alpaca_managed_buy_order(
                            conn,
                            workflow=_signal_workflow(signal),
                            symbol=symbol,
                            signal_symbol=str(signal["RSI Symbol"]),
                            buy_rsi=float(signal["Buy RSI"]),
                            profit_target_multiple=float(signal["Sell Return Multiple"]),
                            buy_signal_date=signal_date,
                            buy_client_order_id=client_order_id,
                            buy_alpaca_order_id=submitted.get("id"),
                            buy_submitted_at=submitted.get("submitted_at"),
                            buy_status=str(submitted.get("status", "submitted")).lower(),
                        )
                        managed_symbols.add(symbol.upper())
                    append_preflight_result(
                        signal_index,
                        symbol=symbol,
                        signal_date=signal_date,
                        client_order_id=client_order_id,
                        notional=order_notional,
                        qty=order_qty,
                        limit_price=limit_price,
                        status="submitted",
                        alpaca_order_id=submitted.get("id"),
                        message=(
                            f"{submitted.get('status', 'submitted')}; whole-share day limit protected "
                            "by the configured price buffer"
                        ),
                    )
                except HTTPError as exc:
                    if managed_position_id is not None and _ambiguous_submission_http_error(exc):
                        if _recover_managed_buy_submission(
                            conn=conn,
                            client=client,
                            rows=rows,
                            signal_index=signal_index,
                            signal=signal,
                            symbol=symbol,
                            signal_date=signal_date,
                            client_order_id=client_order_id,
                            notional=order_notional,
                            qty=order_qty,
                            limit_price=limit_price,
                            message=_http_error_message(exc),
                        ):
                            continue
                        update_alpaca_managed_buy_status(
                            conn,
                            managed_position_id,
                            buy_status="submission_unknown",
                            notes=(
                                "broker returned an ambiguous error after managed buy submission; recovery is pending"
                            ),
                        )
                        append_preflight_result(
                            signal_index,
                            symbol=symbol,
                            signal_date=signal_date,
                            client_order_id=client_order_id,
                            notional=order_notional,
                            qty=order_qty,
                            limit_price=limit_price,
                            status="submission_unknown",
                            message=(
                                "broker response is ambiguous; managed order recovery will retry by client order ID"
                            ),
                        )
                        continue
                    if managed_position_id is not None:
                        fail_alpaca_managed_buy_submission_if_pending(
                            conn,
                            managed_position_id,
                            notes="managed buy submission failed before an Alpaca order was accepted",
                        )
                    append_preflight_result(
                        signal_index,
                        symbol=symbol,
                        signal_date=signal_date,
                        client_order_id=client_order_id,
                        notional=order_notional,
                        qty=order_qty,
                        limit_price=limit_price,
                        status="error",
                            message=_http_error_message(exc),
                        )
                except requests.RequestException as exc:
                    if managed_position_id is not None and _recover_managed_buy_submission(
                        conn=conn,
                        client=client,
                        rows=rows,
                        signal_index=signal_index,
                        signal=signal,
                        symbol=symbol,
                        signal_date=signal_date,
                        client_order_id=client_order_id,
                        notional=order_notional,
                        qty=order_qty,
                        limit_price=limit_price,
                        message=str(exc),
                    ):
                        continue
                    if managed_position_id is not None:
                        update_alpaca_managed_buy_status(
                            conn,
                            managed_position_id,
                            buy_status="submission_unknown",
                            notes="managed buy submission transport failed; recovery is pending",
                        )
                    append_preflight_result(
                        signal_index,
                        symbol=symbol,
                        signal_date=signal_date,
                        client_order_id=client_order_id,
                        notional=order_notional,
                        qty=order_qty,
                        limit_price=limit_price,
                        status="submission_unknown" if managed_position_id is not None else "error",
                        message=(
                            "managed order submission is ambiguous; recovery will retry by client order ID"
                            if managed_position_id is not None
                            else str(exc)
                        ),
                    )
                except Exception as exc:
                    if managed_position_id is not None and _recover_managed_buy_submission(
                        conn=conn,
                        client=client,
                        rows=rows,
                        signal_index=signal_index,
                        signal=signal,
                        symbol=symbol,
                        signal_date=signal_date,
                        client_order_id=client_order_id,
                        notional=order_notional,
                        qty=order_qty,
                        limit_price=limit_price,
                        message=str(exc),
                    ):
                        continue
                    if managed_position_id is not None:
                        update_alpaca_managed_buy_status(
                            conn,
                            managed_position_id,
                            buy_status="submission_unknown",
                            notes="managed buy submission ended unexpectedly; recovery is pending",
                        )
                    append_preflight_result(
                        signal_index,
                        symbol=symbol,
                        signal_date=signal_date,
                        client_order_id=client_order_id,
                        notional=order_notional,
                        qty=order_qty,
                        limit_price=limit_price,
                        status="submission_unknown" if managed_position_id is not None else "error",
                        message=(
                            "managed order submission is ambiguous; recovery will retry by client order ID"
                            if managed_position_id is not None
                            else str(exc)
                        ),
                    )

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=columns)
    return result.sort_values("_signal_index", kind="stable").reindex(columns=columns).reset_index(drop=True)


def submit_alpaca_paper_sell_orders(
    sell_signals: pd.DataFrame,
    cfg: AlpacaOrderConfig,
) -> pd.DataFrame:
    columns = [
        "Asset",
        "Date",
        "Client Order ID",
        "Qty",
        "Status",
        "Alpaca Order ID",
        "Message",
    ]
    if not cfg.sell_enabled:
        return pd.DataFrame(columns=columns)
    if sell_signals.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for signal in sell_signals.itertuples(index=False):
        symbol = str(signal.Asset)
        signal_date = str(signal.Date)
        client_order_id = _alpaca_client_order_id("sell", symbol, signal_date)
        _append_result(
            rows,
            symbol=symbol,
            signal_date=signal_date,
            client_order_id=client_order_id,
            amount_key="Qty",
            amount=None,
            status="managed_only",
            alpaca_order_id=None,
            message=(
                "direct sell-signal submissions are disabled; "
                "managed reconciliation handles GTC limit sells"
            ),
        )

    return pd.DataFrame(rows, columns=columns)
