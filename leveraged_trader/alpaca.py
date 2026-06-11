from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from math import floor
from typing import Optional

import pandas as pd
import requests
import yfinance as yf
from requests import HTTPError

from .config import AlpacaOrderConfig
from .storage import (
    active_alpaca_managed_symbols,
    close_alpaca_managed_position,
    load_alpaca_managed_positions,
    mark_alpaca_managed_buy_filled,
    mark_alpaca_managed_sell_filled,
    record_alpaca_managed_sell_order,
    save_alpaca_managed_buy_order,
    update_alpaca_managed_buy_status,
    update_alpaca_managed_sell_status,
)


BUY_TERMINAL_STATUSES = {"canceled", "done_for_day", "expired", "rejected"}
SELL_INACTIVE_STATUSES = {"canceled", "done_for_day", "expired", "rejected", "stopped", "suspended"}
SELL_RENEWABLE_STATUSES = {"accepted", "accepted_for_bidding", "new", "partially_filled", "pending_new"}
SELL_CANCEL_PENDING_STATUSES = {"pending_cancel"}


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


def _format_order_number(value: float, places: int) -> str:
    formatted = f"{value:.{places}f}".rstrip("0").rstrip(".")
    return formatted or "0"


def _is_whole_share_qty(value: float) -> bool:
    try:
        qty = Decimal(str(value))
    except InvalidOperation:
        return False
    return qty == qty.to_integral_value()


def _alpaca_cash_notional(cfg: AlpacaOrderConfig, headers: dict[str, str]) -> float:
    base_url = cfg.base_url.rstrip("/")
    account_resp = requests.get(
        f"{base_url}/v2/account",
        headers=headers,
        timeout=cfg.timeout_seconds,
    )
    account_resp.raise_for_status()
    account = account_resp.json()
    cash = float(account["cash"])
    notional = round(cash * cfg.cash_fraction, 2)
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


def _optional_float(value: object) -> Optional[float]:
    if value is None or value == "" or pd.isna(value):
        return None
    return float(value)


def _optional_str(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: object) -> int:
    if value is None or value == "" or pd.isna(value):
        return 0
    return int(value)


def _response_json(resp: requests.Response) -> dict:
    try:
        payload = resp.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_alpaca_datetime(value: object) -> Optional[datetime]:
    text = _optional_str(value)
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _gtc_sell_renewal_due(sell_order: dict, cfg: AlpacaOrderConfig) -> bool:
    if not cfg.gtc_sell_renewal_enabled or cfg.gtc_sell_renewal_days_before_expiration < 0:
        return False

    sell_status = str(sell_order.get("status", "")).lower()
    if sell_status not in SELL_RENEWABLE_STATUSES:
        return False

    expires_at = _parse_alpaca_datetime(sell_order.get("expires_at"))
    if expires_at is None:
        return False

    renewal_cutoff = _utc_now() + timedelta(days=cfg.gtc_sell_renewal_days_before_expiration)
    return expires_at <= renewal_cutoff


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

    def cash_notional(self) -> float:
        return _alpaca_cash_notional(self.cfg, self.headers)

    def position_qty(self, symbol: str) -> float:
        return _alpaca_position_qty(symbol, self.cfg, self.headers)

    def asset(self, symbol: str) -> dict:
        return _alpaca_asset(symbol, self.cfg, self.headers)

    def has_open_order(self, symbol: str, side: str) -> bool:
        return _alpaca_has_open_order(symbol, side, self.cfg, self.headers)

    def open_orders(self) -> list[dict]:
        return _alpaca_open_orders(self.cfg, self.headers)

    def submit_market_order(self, order: dict[str, str | bool]) -> requests.Response:
        return requests.post(
            f"{self.base_url}/v2/orders",
            headers=self.headers,
            json=order,
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
                "limit_price": _format_order_number(limit_price, 2),
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
    amount: Optional[float],
    status: str,
    alpaca_order_id: Optional[str],
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


def _fetch_order_payload(client: AlpacaClient, order_id: Optional[str], client_order_id: str) -> dict:
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
    sell_client_order_id: Optional[str],
    qty: Optional[float],
    limit_price: Optional[float],
    alpaca_order_id: Optional[str],
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
    fallback_qty: float,
    sell_alpaca_order_id: Optional[str],
) -> float:
    sell_filled_qty = _optional_float(sell_order.get("filled_qty")) or fallback_qty
    sell_filled_avg_price = _optional_float(sell_order.get("filled_avg_price"))
    if sell_filled_avg_price is None or sell_filled_avg_price <= 0:
        raise ValueError("filled Alpaca sell order is missing filled_avg_price")

    mark_alpaca_managed_sell_filled(
        conn,
        position_id,
        sell_status="filled",
        sell_filled_qty=sell_filled_qty,
        sell_filled_avg_price=sell_filled_avg_price,
        sell_filled_at=_optional_str(sell_order.get("filled_at")),
        sell_alpaca_order_id=sell_alpaca_order_id,
        sell_submitted_at=_optional_str(sell_order.get("submitted_at")),
        sell_expires_at=_optional_str(sell_order.get("expires_at")),
    )
    close_alpaca_managed_position(
        conn,
        position_id,
        closed_at=_optional_str(sell_order.get("filled_at")),
        notes="managed target sell filled",
    )
    return sell_filled_qty


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
) -> None:
    sell_resp = client.submit_limit_sell_order(
        symbol=symbol,
        qty=float(filled_qty),
        limit_price=float(target_sell_price),
        client_order_id=sell_client_order_id,
    )
    sell_resp.raise_for_status()
    sell_order = _response_json(sell_resp)
    sell_status = str(sell_order.get("status", "submitted")).lower()
    sell_alpaca_order_id = _optional_str(sell_order.get("id"))
    record_alpaca_managed_sell_order(
        conn,
        position_id,
        sell_client_order_id=sell_client_order_id,
        sell_alpaca_order_id=sell_alpaca_order_id,
        sell_submitted_at=_optional_str(sell_order.get("submitted_at")),
        sell_status=sell_status,
        sell_expires_at=_optional_str(sell_order.get("expires_at")),
        increment_renewal_count=increment_renewal_count,
    )
    if sell_status == "filled":
        _record_filled_managed_sell(
            conn=conn,
            position_id=position_id,
            sell_order=sell_order,
            fallback_qty=filled_qty,
            sell_alpaca_order_id=sell_alpaca_order_id,
        )
    _append_reconciliation_result(
        rows,
        position_id=position_id,
        symbol=symbol,
        action="sell",
        status="renewed" if increment_renewal_count and sell_status != "filled" else sell_status,
        buy_client_order_id=buy_client_order_id,
        sell_client_order_id=sell_client_order_id,
        qty=filled_qty,
        limit_price=target_sell_price,
        alpaca_order_id=sell_alpaca_order_id,
        message=(
            "managed target sell filled; position closed"
            if sell_status == "filled"
            else replacement_message
        ),
    )


def _append_fractional_sell_result(
    *,
    conn: sqlite3.Connection,
    rows: list[dict],
    position_id: int,
    symbol: str,
    buy_client_order_id: str,
    sell_client_order_id: Optional[str],
    filled_qty: float,
    target_sell_price: float,
) -> None:
    update_alpaca_managed_sell_status(
        conn,
        position_id,
        sell_status="fractional_qty",
        notes="filled quantity is fractional; no GTC limit sell submitted",
    )
    _append_reconciliation_result(
        rows,
        position_id=position_id,
        symbol=symbol,
        action="sell",
        status="fractional_qty",
        buy_client_order_id=buy_client_order_id,
        sell_client_order_id=sell_client_order_id,
        qty=filled_qty,
        limit_price=target_sell_price,
        alpaca_order_id=None,
        message="filled quantity is fractional; no GTC limit sell submitted",
    )


def _append_open_sell_result(
    *,
    rows: list[dict],
    position_id: int,
    symbol: str,
    buy_client_order_id: str,
    sell_client_order_id: Optional[str],
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

    if check_open_order and client.has_open_order(symbol, "sell"):
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
    _submit_managed_gtc_sell(
        conn=conn,
        client=client,
        rows=rows,
        position_id=position_id,
        symbol=symbol,
        buy_client_order_id=buy_client_order_id,
        sell_client_order_id=sell_client_order_id,
        filled_qty=filled_qty,
        target_sell_price=target_sell_price,
        increment_renewal_count=True,
        replacement_message=replacement_message,
    )


def _renewal_requested_at() -> str:
    return _utc_now().isoformat().replace("+00:00", "Z")


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
) -> None:
    position_id = int(position["id"])
    cancel_resp = client.cancel_order(sell_alpaca_order_id)
    cancel_resp.raise_for_status()

    refreshed_resp = client.get_order(sell_alpaca_order_id)
    refreshed_resp.raise_for_status()
    refreshed_order = _response_json(refreshed_resp)
    refreshed_status = str(refreshed_order.get("status", "pending_cancel")).lower()
    update_alpaca_managed_sell_status(
        conn,
        position_id,
        sell_status=refreshed_status,
        sell_alpaca_order_id=_optional_str(refreshed_order.get("id")) or sell_alpaca_order_id,
        sell_submitted_at=_optional_str(refreshed_order.get("submitted_at")),
        sell_expires_at=_optional_str(refreshed_order.get("expires_at")),
        sell_renewal_requested_at=_renewal_requested_at(),
        notes="managed GTC sell renewal requested before expiration",
    )
    if refreshed_status == "filled":
        filled_sell_qty = _record_filled_managed_sell(
            conn=conn,
            position_id=position_id,
            sell_order=refreshed_order,
            fallback_qty=filled_qty,
            sell_alpaca_order_id=sell_alpaca_order_id,
        )
        _append_reconciliation_result(
            rows,
            position_id=position_id,
            symbol=symbol,
            action="sell",
            status=refreshed_status,
            buy_client_order_id=buy_client_order_id,
            sell_client_order_id=_optional_str(position.get("sell_client_order_id")),
            qty=filled_sell_qty,
            limit_price=target_sell_price,
            alpaca_order_id=sell_alpaca_order_id,
            message="managed target sell filled; position closed",
        )
        return
    if refreshed_status in SELL_INACTIVE_STATUSES:
        _submit_replacement_gtc_sell(
            conn=conn,
            client=client,
            rows=rows,
            position=position,
            symbol=symbol,
            buy_client_order_id=buy_client_order_id,
            filled_qty=filled_qty,
            target_sell_price=target_sell_price,
            replacement_message="renewed managed GTC limit sell before Alpaca aged-order expiration",
            check_open_order=False,
        )
        return

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
        message=(
            "managed GTC sell expires soon; cancellation requested and replacement "
            "will be submitted after Alpaca confirms cancellation"
        ),
    )


def reconcile_alpaca_managed_positions(
    conn: sqlite3.Connection,
    cfg: AlpacaOrderConfig,
) -> pd.DataFrame:
    columns = [
        "Position ID",
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

    client = AlpacaClient(cfg)
    rows = []
    for position in positions.to_dict("records"):
        position_id = int(position["id"])
        symbol = str(position["symbol"])
        buy_client_order_id = str(position["buy_client_order_id"])
        sell_client_order_id = _optional_str(position.get("sell_client_order_id"))
        try:
            filled_qty = _optional_float(position.get("filled_qty"))
            filled_avg_price = _optional_float(position.get("filled_avg_price"))
            target_sell_price = _optional_float(position.get("target_sell_price"))

            if filled_qty is None or filled_avg_price is None or target_sell_price is None:
                buy_order = _fetch_order_payload(
                    client,
                    _optional_str(position.get("buy_alpaca_order_id")),
                    buy_client_order_id,
                )
                buy_status = str(buy_order.get("status", "unknown")).lower()
                buy_alpaca_order_id = _optional_str(buy_order.get("id"))
                buy_submitted_at = _optional_str(buy_order.get("submitted_at"))
                order_filled_qty = _optional_float(buy_order.get("filled_qty")) or 0.0

                if buy_status == "filled" or (buy_status in BUY_TERMINAL_STATUSES and order_filled_qty > 0):
                    order_filled_avg_price = _optional_float(buy_order.get("filled_avg_price"))
                    if order_filled_qty <= 0 or order_filled_avg_price is None or order_filled_avg_price <= 0:
                        raise ValueError("filled Alpaca buy order is missing filled_qty or filled_avg_price")
                    filled_qty = order_filled_qty
                    filled_avg_price = order_filled_avg_price
                    target_sell_price = round(filled_avg_price * float(position["profit_target_multiple"]), 2)
                    mark_alpaca_managed_buy_filled(
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
                    _append_reconciliation_result(
                        rows,
                        position_id=position_id,
                        symbol=symbol,
                        action="buy",
                        status=buy_status,
                        buy_client_order_id=buy_client_order_id,
                        sell_client_order_id=sell_client_order_id,
                        qty=filled_qty,
                        limit_price=target_sell_price,
                        alpaca_order_id=buy_alpaca_order_id,
                        message="buy filled; target sell price frozen from actual fill",
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
                sell_order = _fetch_order_payload(
                    client,
                    _optional_str(position.get("sell_alpaca_order_id")),
                    sell_client_order_id,
                )
                sell_status = str(sell_order.get("status", "unknown")).lower()
                sell_alpaca_order_id = _optional_str(sell_order.get("id"))
                sell_expires_at = _optional_str(sell_order.get("expires_at"))
                update_alpaca_managed_sell_status(
                    conn,
                    position_id,
                    sell_status=sell_status,
                    sell_alpaca_order_id=sell_alpaca_order_id,
                    sell_submitted_at=_optional_str(sell_order.get("submitted_at")),
                    sell_expires_at=sell_expires_at,
                )
                if sell_status == "filled":
                    filled_sell_qty = _record_filled_managed_sell(
                        conn=conn,
                        position_id=position_id,
                        sell_order=sell_order,
                        fallback_qty=float(filled_qty),
                        sell_alpaca_order_id=sell_alpaca_order_id,
                    )
                    _append_reconciliation_result(
                        rows,
                        position_id=position_id,
                        symbol=symbol,
                        action="sell",
                        status=sell_status,
                        buy_client_order_id=buy_client_order_id,
                        sell_client_order_id=sell_client_order_id,
                        qty=filled_sell_qty,
                        limit_price=target_sell_price,
                        alpaca_order_id=sell_alpaca_order_id,
                        message="managed target sell filled; position closed",
                    )
                    continue
                if sell_status in SELL_INACTIVE_STATUSES:
                    renewal_was_requested = _optional_str(position.get("sell_renewal_requested_at")) is not None
                    should_resubmit = cfg.gtc_sell_renewal_enabled and (
                        sell_status == "expired"
                        or (sell_status == "canceled" and renewal_was_requested)
                    )
                    if should_resubmit:
                        _submit_replacement_gtc_sell(
                            conn=conn,
                            client=client,
                            rows=rows,
                            position=position,
                            symbol=symbol,
                            buy_client_order_id=buy_client_order_id,
                            filled_qty=float(filled_qty),
                            target_sell_price=float(target_sell_price),
                            replacement_message=(
                                "prior managed GTC sell expired; submitted replacement at frozen target price"
                                if sell_status == "expired"
                                else "renewed managed GTC limit sell after Alpaca confirmed cancellation"
                            ),
                            check_open_order=True,
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
                if sell_alpaca_order_id and _gtc_sell_renewal_due(sell_order, cfg):
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

            if client.has_open_order(symbol, "sell"):
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

    return pd.DataFrame(rows, columns=columns)


def submit_alpaca_paper_buy_orders(
    buy_signals: pd.DataFrame,
    cfg: AlpacaOrderConfig,
    conn: Optional[sqlite3.Connection] = None,
) -> pd.DataFrame:
    columns = [
        "Asset",
        "Date",
        "Client Order ID",
        "Notional",
        "Qty",
        "Status",
        "Alpaca Order ID",
        "Message",
    ]
    if not cfg.enabled:
        return pd.DataFrame(columns=columns)
    if buy_signals.empty:
        return pd.DataFrame(columns=columns)

    managed_symbols = active_alpaca_managed_symbols(conn) if conn is not None else set()
    client: Optional[AlpacaClient] = None
    rows = []
    for signal in buy_signals.to_dict("records"):
        symbol = str(signal["Asset"])
        signal_date = str(signal["Date"])
        client_order_id = _alpaca_client_order_id("buy", symbol, signal_date)
        order_notional: Optional[float] = None
        order_qty: Optional[int] = None

        try:
            if symbol.upper() in managed_symbols:
                _append_result(
                    rows,
                    symbol=symbol,
                    signal_date=signal_date,
                    client_order_id=client_order_id,
                    amount_key="Notional",
                    amount=None,
                    status="managed",
                    alpaca_order_id=None,
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
                _append_result(
                    rows,
                    symbol=symbol,
                    signal_date=signal_date,
                    client_order_id=client_order_id,
                    amount_key="Notional",
                    amount=None,
                    status="existing",
                    alpaca_order_id=existing.get("id"),
                    message=existing.get("status", "order already exists"),
                )
                continue
            if existing_resp.status_code != 404:
                existing_resp.raise_for_status()

            open_orders = client.open_orders()
            if _orders_have_open_order(open_orders, symbol, "buy"):
                _append_result(
                    rows,
                    symbol=symbol,
                    signal_date=signal_date,
                    client_order_id=client_order_id,
                    amount_key="Notional",
                    amount=None,
                    status="open_order",
                    alpaca_order_id=None,
                    message="open buy order already exists for symbol in Alpaca account",
                )
                continue

            if _orders_have_open_order(open_orders, symbol, "sell"):
                _append_result(
                    rows,
                    symbol=symbol,
                    signal_date=signal_date,
                    client_order_id=client_order_id,
                    amount_key="Notional",
                    amount=None,
                    status="open_sell_order",
                    alpaca_order_id=None,
                    message="open sell order already exists for symbol in Alpaca account",
                )
                continue

            position_qty = client.position_qty(symbol)
            if position_qty > 0:
                _append_result(
                    rows,
                    symbol=symbol,
                    signal_date=signal_date,
                    client_order_id=client_order_id,
                    amount_key="Notional",
                    amount=None,
                    status="held",
                    alpaca_order_id=None,
                    message=f"symbol already held in Alpaca account: qty={position_qty}",
                )
                continue

            asset = client.asset(symbol)
            if not asset.get("tradable", False):
                _append_result(
                    rows,
                    symbol=symbol,
                    signal_date=signal_date,
                    client_order_id=client_order_id,
                    amount_key="Notional",
                    amount=None,
                    status="not_tradable",
                    alpaca_order_id=None,
                    message=f"symbol is not tradable through Alpaca: status={asset.get('status', 'unknown')}",
                )
                continue
            if asset.get("status") not in {None, "active"}:
                _append_result(
                    rows,
                    symbol=symbol,
                    signal_date=signal_date,
                    client_order_id=client_order_id,
                    amount_key="Notional",
                    amount=None,
                    status="inactive",
                    alpaca_order_id=None,
                    message=f"symbol is not active through Alpaca: status={asset.get('status')}",
                )
                continue
            order_notional = client.cash_notional()
            latest_price = _latest_market_price(symbol)
            order_qty = floor(order_notional / latest_price)
            if order_qty < 1:
                _append_result(
                    rows,
                    symbol=symbol,
                    signal_date=signal_date,
                    client_order_id=client_order_id,
                    amount_key="Notional",
                    amount=order_notional,
                    status="insufficient_notional",
                    alpaca_order_id=None,
                    message=(
                        "10% cash allocation is below one whole share "
                        f"at latest price {latest_price:.4f}; buy order skipped"
                    ),
                )
                continue
            order = {
                "symbol": symbol,
                "side": "buy",
                "type": "market",
                "time_in_force": "day",
                "extended_hours": False,
                "client_order_id": client_order_id,
                "qty": str(order_qty),
            }

            submit_resp = client.submit_market_order(order)
            submit_resp.raise_for_status()
            submitted = submit_resp.json()
            if conn is not None and {"RSI Symbol", "Buy RSI", "Sell Return Multiple"}.issubset(signal):
                save_alpaca_managed_buy_order(
                    conn,
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
            row = {
                "Asset": symbol,
                "Date": signal_date,
                "Client Order ID": client_order_id,
                "Notional": order_notional,
                "Qty": order_qty,
                "Status": "submitted",
                "Alpaca Order ID": submitted.get("id"),
                "Message": submitted.get("status", "submitted"),
            }
            if order_qty is not None:
                row["Message"] = f"{row['Message']}; whole-share qty order from latest price estimate"
            rows.append(row)
        except HTTPError as exc:
            rows.append(
                {
                    "Asset": symbol,
                    "Date": signal_date,
                    "Client Order ID": client_order_id,
                    "Notional": order_notional,
                    "Qty": order_qty,
                    "Status": "error",
                    "Alpaca Order ID": None,
                    "Message": _http_error_message(exc),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "Asset": symbol,
                    "Date": signal_date,
                    "Client Order ID": client_order_id,
                    "Notional": order_notional,
                    "Qty": order_qty,
                    "Status": "error",
                    "Alpaca Order ID": None,
                    "Message": str(exc),
                }
            )

    return pd.DataFrame(rows, columns=columns)


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
        symbol = str(getattr(signal, "Asset"))
        signal_date = str(getattr(signal, "Date"))
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
