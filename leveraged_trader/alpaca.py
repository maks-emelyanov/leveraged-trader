from __future__ import annotations

import re
from math import floor
from typing import Optional

import pandas as pd
import requests
import yfinance as yf
from requests import HTTPError

from .config import AlpacaOrderConfig


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
    symbol = symbol.upper()
    side = side.lower()
    for order in orders_resp.json():
        if str(order.get("symbol", "")).upper() == symbol and str(order.get("side", "")).lower() == side:
            return True
    return False


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

    def cash_notional(self) -> float:
        return _alpaca_cash_notional(self.cfg, self.headers)

    def position_qty(self, symbol: str) -> float:
        return _alpaca_position_qty(symbol, self.cfg, self.headers)

    def asset(self, symbol: str) -> dict:
        return _alpaca_asset(symbol, self.cfg, self.headers)

    def has_open_order(self, symbol: str, side: str) -> bool:
        return _alpaca_has_open_order(symbol, side, self.cfg, self.headers)

    def submit_market_order(self, order: dict[str, str | bool]) -> requests.Response:
        return requests.post(
            f"{self.base_url}/v2/orders",
            headers=self.headers,
            json=order,
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


def submit_alpaca_paper_buy_orders(
    buy_signals: pd.DataFrame,
    cfg: AlpacaOrderConfig,
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

    client = AlpacaClient(cfg)
    rows = []
    for signal in buy_signals.itertuples(index=False):
        symbol = str(getattr(signal, "Asset"))
        signal_date = str(getattr(signal, "Date"))
        client_order_id = _alpaca_client_order_id("buy", symbol, signal_date)
        order_notional: Optional[float] = None
        order_qty: Optional[int] = None

        try:
            existing_resp = client.get_order_by_client_order_id(client_order_id)
            if existing_resp.status_code == 200:
                existing = existing_resp.json()
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

            if client.has_open_order(symbol, "buy"):
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
            if not asset.get("fractionable", False):
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
            else:
                order_notional = client.cash_notional()
                order_qty = None
                order = {
                    "symbol": symbol,
                    "side": "buy",
                    "type": "market",
                    "time_in_force": "day",
                    "extended_hours": False,
                    "client_order_id": client_order_id,
                    "notional": str(order_notional),
                }

            submit_resp = client.submit_market_order(order)
            submit_resp.raise_for_status()
            submitted = submit_resp.json()
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

    client = AlpacaClient(cfg)
    rows = []
    for signal in sell_signals.itertuples(index=False):
        symbol = str(getattr(signal, "Asset"))
        signal_date = str(getattr(signal, "Date"))
        client_order_id = _alpaca_client_order_id("sell", symbol, signal_date)
        position_qty: Optional[float] = None

        try:
            existing_resp = client.get_order_by_client_order_id(client_order_id)
            if existing_resp.status_code == 200:
                existing = existing_resp.json()
                rows.append(
                    {
                        "Asset": symbol,
                        "Date": signal_date,
                        "Client Order ID": client_order_id,
                        "Qty": None,
                        "Status": "existing",
                        "Alpaca Order ID": existing.get("id"),
                        "Message": existing.get("status", "order already exists"),
                    }
                )
                continue
            if existing_resp.status_code != 404:
                existing_resp.raise_for_status()

            if client.has_open_order(symbol, "sell"):
                rows.append(
                    {
                        "Asset": symbol,
                        "Date": signal_date,
                        "Client Order ID": client_order_id,
                        "Qty": None,
                        "Status": "open_order",
                        "Alpaca Order ID": None,
                        "Message": "open sell order already exists for symbol in Alpaca account",
                    }
                )
                continue

            position_qty = client.position_qty(symbol)
            if position_qty <= 0:
                rows.append(
                    {
                        "Asset": symbol,
                        "Date": signal_date,
                        "Client Order ID": client_order_id,
                        "Qty": position_qty,
                        "Status": "not_held",
                        "Alpaca Order ID": None,
                        "Message": "symbol is not currently held in Alpaca account",
                    }
                )
                continue

            order = {
                "symbol": symbol,
                "side": "sell",
                "type": "market",
                "time_in_force": "day",
                "extended_hours": False,
                "client_order_id": client_order_id,
                "qty": str(position_qty),
            }

            submit_resp = client.submit_market_order(order)
            submit_resp.raise_for_status()
            submitted = submit_resp.json()
            _append_result(
                rows,
                symbol=symbol,
                signal_date=signal_date,
                client_order_id=client_order_id,
                amount_key="Qty",
                amount=position_qty,
                status="submitted",
                alpaca_order_id=submitted.get("id"),
                message=submitted.get("status", "submitted"),
            )
        except HTTPError as exc:
            _append_result(
                rows,
                symbol=symbol,
                signal_date=signal_date,
                client_order_id=client_order_id,
                amount_key="Qty",
                amount=position_qty,
                status="error",
                alpaca_order_id=None,
                message=_http_error_message(exc),
            )
        except Exception as exc:
            _append_result(
                rows,
                symbol=symbol,
                signal_date=signal_date,
                client_order_id=client_order_id,
                amount_key="Qty",
                amount=position_qty,
                status="error",
                alpaca_order_id=None,
                message=str(exc),
            )

    return pd.DataFrame(rows, columns=columns)
