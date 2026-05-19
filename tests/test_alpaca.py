from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import pandas as pd
from requests import HTTPError

from leveraged_trader.alpaca import submit_alpaca_paper_buy_orders, submit_alpaca_paper_sell_orders
from leveraged_trader.config import AlpacaOrderConfig


def response(status_code: int, payload: object) -> Mock:
    resp = Mock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.text = str(payload)
    resp.raise_for_status.return_value = None
    return resp


def error_response(status_code: int, payload: object) -> Mock:
    resp = response(status_code, payload)
    error = HTTPError(f"{status_code} Client Error")
    error.response = resp
    resp.raise_for_status.side_effect = error
    return resp


class AlpacaTests(unittest.TestCase):
    def cfg(self, *, buy: bool = False, sell: bool = False) -> AlpacaOrderConfig:
        return AlpacaOrderConfig(
            enabled=buy,
            sell_enabled=sell,
            api_key_id="key",
            api_secret_key="secret",
        )

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_buy_order_uses_10_percent_cash_notional(self, mock_get: Mock, mock_post: Mock) -> None:
        mock_get.side_effect = [
            response(404, {}),
            response(200, []),
            response(404, {}),
            response(200, {"symbol": "TQQQ", "status": "active", "tradable": True, "fractionable": True}),
            response(200, {"cash": "12345.67"}),
        ]
        mock_post.return_value = response(200, {"id": "order-1", "status": "accepted"})

        result = submit_alpaca_paper_buy_orders(
            pd.DataFrame([{"Asset": "TQQQ", "Date": "2026-01-02"}]),
            self.cfg(buy=True),
        )

        self.assertEqual(result.loc[0, "Status"], "submitted")
        self.assertEqual(result.loc[0, "Notional"], 1234.57)
        self.assertEqual(mock_post.call_args.kwargs["json"]["notional"], "1234.57")
        self.assertFalse(mock_post.call_args.kwargs["json"]["extended_hours"])

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_sell_order_sells_full_position_qty(self, mock_get: Mock, mock_post: Mock) -> None:
        mock_get.side_effect = [
            response(404, {}),
            response(200, []),
            response(200, {"qty": "12.5"}),
        ]
        mock_post.return_value = response(200, {"id": "order-2", "status": "accepted"})

        result = submit_alpaca_paper_sell_orders(
            pd.DataFrame([{"Asset": "TQQQ", "Date": "2026-01-02"}]),
            self.cfg(sell=True),
        )

        self.assertEqual(result.loc[0, "Status"], "submitted")
        self.assertEqual(result.loc[0, "Qty"], 12.5)
        self.assertEqual(mock_post.call_args.kwargs["json"]["qty"], "12.5")
        self.assertFalse(mock_post.call_args.kwargs["json"]["extended_hours"])

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_buy_skips_when_open_buy_order_exists(self, mock_get: Mock, mock_post: Mock) -> None:
        mock_get.side_effect = [
            response(404, {}),
            response(200, [{"symbol": "TQQQ", "side": "buy"}]),
        ]

        result = submit_alpaca_paper_buy_orders(
            pd.DataFrame([{"Asset": "TQQQ", "Date": "2026-01-02"}]),
            self.cfg(buy=True),
        )

        self.assertEqual(result.loc[0, "Status"], "open_order")
        mock_post.assert_not_called()

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_buy_skips_when_symbol_is_not_tradable(self, mock_get: Mock, mock_post: Mock) -> None:
        mock_get.side_effect = [
            response(404, {}),
            response(200, []),
            response(404, {}),
            response(200, {"symbol": "FCXG", "status": "inactive", "tradable": False, "fractionable": False}),
        ]

        result = submit_alpaca_paper_buy_orders(
            pd.DataFrame([{"Asset": "FCXG", "Date": "2026-01-02"}]),
            self.cfg(buy=True),
        )

        self.assertEqual(result.loc[0, "Status"], "not_tradable")
        self.assertIn("not tradable", result.loc[0, "Message"])
        mock_post.assert_not_called()

    @patch("leveraged_trader.alpaca._latest_market_price")
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_buy_uses_whole_share_qty_when_symbol_does_not_support_notional_orders(
        self,
        mock_get: Mock,
        mock_post: Mock,
        mock_latest_market_price: Mock,
    ) -> None:
        mock_get.side_effect = [
            response(404, {}),
            response(200, []),
            response(404, {}),
            response(200, {"symbol": "ABC", "status": "active", "tradable": True, "fractionable": False}),
            response(200, {"cash": "12345.67"}),
        ]
        mock_latest_market_price.return_value = 100.0
        mock_post.return_value = response(200, {"id": "order-3", "status": "accepted"})

        result = submit_alpaca_paper_buy_orders(
            pd.DataFrame([{"Asset": "ABC", "Date": "2026-01-02"}]),
            self.cfg(buy=True),
        )

        self.assertEqual(result.loc[0, "Status"], "submitted")
        self.assertEqual(result.loc[0, "Notional"], 1234.57)
        self.assertEqual(result.loc[0, "Qty"], 12)
        self.assertEqual(mock_post.call_args.kwargs["json"]["qty"], "12")
        self.assertNotIn("notional", mock_post.call_args.kwargs["json"])

    @patch("leveraged_trader.alpaca._latest_market_price")
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_buy_skips_non_fractionable_when_allocation_is_less_than_one_share(
        self,
        mock_get: Mock,
        mock_post: Mock,
        mock_latest_market_price: Mock,
    ) -> None:
        mock_get.side_effect = [
            response(404, {}),
            response(200, []),
            response(404, {}),
            response(200, {"symbol": "ABC", "status": "active", "tradable": True, "fractionable": False}),
            response(200, {"cash": "100.00"}),
        ]
        mock_latest_market_price.return_value = 20.0

        result = submit_alpaca_paper_buy_orders(
            pd.DataFrame([{"Asset": "ABC", "Date": "2026-01-02"}]),
            self.cfg(buy=True),
        )

        self.assertEqual(result.loc[0, "Status"], "insufficient_notional")
        self.assertIn("below one whole share", result.loc[0, "Message"])
        mock_post.assert_not_called()

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_buy_error_includes_alpaca_response_message(self, mock_get: Mock, mock_post: Mock) -> None:
        mock_get.side_effect = [
            response(404, {}),
            response(200, []),
            response(404, {}),
            response(200, {"symbol": "TQQQ", "status": "active", "tradable": True, "fractionable": True}),
            response(200, {"cash": "12345.67"}),
        ]
        mock_post.return_value = error_response(403, {"message": "account is not allowed to trade this asset"})

        result = submit_alpaca_paper_buy_orders(
            pd.DataFrame([{"Asset": "TQQQ", "Date": "2026-01-02"}]),
            self.cfg(buy=True),
        )

        self.assertEqual(result.loc[0, "Status"], "error")
        self.assertIn("account is not allowed", result.loc[0, "Message"])


if __name__ == "__main__":
    unittest.main()
