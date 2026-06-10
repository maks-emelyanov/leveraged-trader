from __future__ import annotations

import sqlite3
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from requests import HTTPError

from leveraged_trader.alpaca import (
    reconcile_alpaca_managed_positions,
    submit_alpaca_paper_buy_orders,
    submit_alpaca_paper_sell_orders,
)
from leveraged_trader.config import AlpacaOrderConfig
from leveraged_trader.storage import (
    active_alpaca_managed_symbols,
    init_state_db,
    load_alpaca_managed_positions,
    mark_alpaca_managed_buy_filled,
    record_alpaca_managed_sell_order,
    save_alpaca_managed_buy_order,
)


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

    @patch("leveraged_trader.alpaca._latest_market_price")
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_buy_order_uses_10_percent_cash_whole_share_qty(
        self,
        mock_get: Mock,
        mock_post: Mock,
        mock_latest_market_price: Mock,
    ) -> None:
        mock_get.side_effect = [
            response(404, {}),
            response(200, []),
            response(404, {}),
            response(200, {"symbol": "TQQQ", "status": "active", "tradable": True, "fractionable": True}),
            response(200, {"cash": "12345.67"}),
        ]
        mock_latest_market_price.return_value = 100.0
        mock_post.return_value = response(200, {"id": "order-1", "status": "accepted"})

        result = submit_alpaca_paper_buy_orders(
            pd.DataFrame([{"Asset": "TQQQ", "Date": "2026-01-02"}]),
            self.cfg(buy=True),
        )

        self.assertEqual(result.loc[0, "Status"], "submitted")
        self.assertEqual(result.loc[0, "Notional"], 1234.57)
        self.assertEqual(result.loc[0, "Qty"], 12)
        self.assertEqual(mock_post.call_args.kwargs["json"]["qty"], "12")
        self.assertNotIn("notional", mock_post.call_args.kwargs["json"])
        self.assertFalse(mock_post.call_args.kwargs["json"]["extended_hours"])

    def test_direct_sell_signal_submission_is_disabled(self) -> None:
        result = submit_alpaca_paper_sell_orders(
            pd.DataFrame([{"Asset": "TQQQ", "Date": "2026-01-02"}]),
            self.cfg(sell=True),
        )

        self.assertEqual(result.loc[0, "Status"], "managed_only")
        self.assertIn("managed reconciliation", result.loc[0, "Message"])

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
    def test_buy_uses_whole_share_qty_for_all_symbols(
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
    def test_buy_skips_when_allocation_is_less_than_one_share(
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

    @patch("leveraged_trader.alpaca._latest_market_price")
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_buy_error_includes_alpaca_response_message(
        self,
        mock_get: Mock,
        mock_post: Mock,
        mock_latest_market_price: Mock,
    ) -> None:
        mock_get.side_effect = [
            response(404, {}),
            response(200, []),
            response(404, {}),
            response(200, {"symbol": "TQQQ", "status": "active", "tradable": True, "fractionable": True}),
            response(200, {"cash": "12345.67"}),
        ]
        mock_latest_market_price.return_value = 100.0
        mock_post.return_value = error_response(403, {"message": "account is not allowed to trade this asset"})

        result = submit_alpaca_paper_buy_orders(
            pd.DataFrame([{"Asset": "TQQQ", "Date": "2026-01-02"}]),
            self.cfg(buy=True),
        )

        self.assertEqual(result.loc[0, "Status"], "error")
        self.assertIn("account is not allowed", result.loc[0, "Message"])

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_filled_managed_buy_creates_limit_sell_from_original_multiple(
        self,
        mock_get: Mock,
        mock_post: Mock,
    ) -> None:
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            save_alpaca_managed_buy_order(
                conn,
                symbol="TQQQ",
                signal_symbol="QQQ",
                buy_rsi=30,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-20260102",
                buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-01-02T14:30:00Z",
                buy_status="accepted",
            )
            mock_get.side_effect = [
                response(
                    200,
                    {
                        "id": "buy-1",
                        "status": "filled",
                        "submitted_at": "2026-01-02T14:30:00Z",
                        "filled_at": "2026-01-02T14:31:00Z",
                        "filled_qty": "2",
                        "filled_avg_price": "100.00",
                    },
                ),
                response(200, []),
            ]
            mock_post.return_value = response(200, {"id": "sell-1", "status": "accepted"})

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.iloc[-1]["Action"], "sell")
        self.assertEqual(result.iloc[-1]["Status"], "accepted")
        self.assertEqual(managed.loc[0, "target_sell_price"], 150.0)
        self.assertEqual(mock_post.call_args.kwargs["json"]["type"], "limit")
        self.assertEqual(mock_post.call_args.kwargs["json"]["time_in_force"], "gtc")
        self.assertEqual(mock_post.call_args.kwargs["json"]["qty"], "2")
        self.assertEqual(mock_post.call_args.kwargs["json"]["limit_price"], "150")

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_fractional_managed_buy_does_not_create_gtc_sell(
        self,
        mock_get: Mock,
        mock_post: Mock,
    ) -> None:
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            save_alpaca_managed_buy_order(
                conn,
                symbol="TQQQ",
                signal_symbol="QQQ",
                buy_rsi=30,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-20260102",
                buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-01-02T14:30:00Z",
                buy_status="accepted",
            )
            mock_get.side_effect = [
                response(
                    200,
                    {
                        "id": "buy-1",
                        "status": "filled",
                        "submitted_at": "2026-01-02T14:30:00Z",
                        "filled_at": "2026-01-02T14:31:00Z",
                        "filled_qty": "2.5",
                        "filled_avg_price": "100.00",
                    },
                ),
                response(200, []),
            ]

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.iloc[-1]["Status"], "fractional_qty")
        self.assertEqual(managed.loc[0, "sell_status"], "fractional_qty")
        mock_post.assert_not_called()

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_managed_symbol_suppresses_new_buy_signal(self, mock_get: Mock, mock_post: Mock) -> None:
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            save_alpaca_managed_buy_order(
                conn,
                symbol="TQQQ",
                signal_symbol="QQQ",
                buy_rsi=30,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-20260102",
                buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-01-02T14:30:00Z",
                buy_status="accepted",
            )

            result = submit_alpaca_paper_buy_orders(
                pd.DataFrame(
                    [
                        {
                            "Asset": "TQQQ",
                            "RSI Symbol": "QQQ",
                            "Date": "2026-01-03",
                            "Buy RSI": 31,
                            "Sell Return Multiple": 2.0,
                        }
                    ]
                ),
                self.cfg(buy=True),
                conn=conn,
            )

        self.assertEqual(result.loc[0, "Status"], "managed")
        mock_get.assert_not_called()
        mock_post.assert_not_called()

    def test_existing_managed_buy_does_not_rewrite_original_sell_multiple(self) -> None:
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            save_alpaca_managed_buy_order(
                conn,
                symbol="TQQQ",
                signal_symbol="QQQ",
                buy_rsi=30,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-20260102",
                buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-01-02T14:30:00Z",
                buy_status="accepted",
            )
            save_alpaca_managed_buy_order(
                conn,
                symbol="TQQQ",
                signal_symbol="QQQ",
                buy_rsi=31,
                profit_target_multiple=2.0,
                buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-20260102",
                buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-01-02T14:30:00Z",
                buy_status="accepted",
            )
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(managed.loc[0, "buy_rsi"], 30)
        self.assertEqual(managed.loc[0, "profit_target_multiple"], 1.5)

    @patch("leveraged_trader.alpaca.requests.get")
    def test_filled_managed_sell_closes_position(self, mock_get: Mock) -> None:
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            save_alpaca_managed_buy_order(
                conn,
                symbol="TQQQ",
                signal_symbol="QQQ",
                buy_rsi=30,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-20260102",
                buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-01-02T14:30:00Z",
                buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn,
                1,
                buy_status="filled",
                filled_qty=2.5,
                filled_avg_price=100.0,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150.0,
            )
            record_alpaca_managed_sell_order(
                conn,
                1,
                sell_client_order_id="rsi-exit-TQQQ-1-20260102143100",
                sell_alpaca_order_id="sell-1",
                sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="accepted",
            )
            mock_get.return_value = response(
                200,
                {
                    "id": "sell-1",
                    "status": "filled",
                    "submitted_at": "2026-01-02T14:32:00Z",
                    "filled_at": "2026-01-03T15:00:00Z",
                    "filled_qty": "2.5",
                },
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)
            active_symbols = active_alpaca_managed_symbols(conn)

        self.assertEqual(result.loc[0, "Status"], "filled")
        self.assertIsNotNone(managed.loc[0, "closed_at"])
        self.assertNotIn("TQQQ", active_symbols)

    @patch("leveraged_trader.alpaca.requests.get")
    def test_rejected_managed_sell_remains_active_to_block_new_buys(self, mock_get: Mock) -> None:
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            save_alpaca_managed_buy_order(
                conn,
                symbol="TQQQ",
                signal_symbol="QQQ",
                buy_rsi=30,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-20260102",
                buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-01-02T14:30:00Z",
                buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn,
                1,
                buy_status="filled",
                filled_qty=2.5,
                filled_avg_price=100.0,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150.0,
            )
            record_alpaca_managed_sell_order(
                conn,
                1,
                sell_client_order_id="rsi-exit-TQQQ-1-20260102143100",
                sell_alpaca_order_id="sell-1",
                sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="accepted",
            )
            mock_get.return_value = response(
                200,
                {
                    "id": "sell-1",
                    "status": "rejected",
                    "submitted_at": "2026-01-02T14:32:00Z",
                },
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            active_symbols = active_alpaca_managed_symbols(conn)

        self.assertEqual(result.loc[0, "Status"], "rejected")
        self.assertIn("TQQQ", active_symbols)

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_expired_managed_sell_is_not_resubmitted(self, mock_get: Mock, mock_post: Mock) -> None:
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            save_alpaca_managed_buy_order(
                conn,
                symbol="TQQQ",
                signal_symbol="QQQ",
                buy_rsi=30,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-20260102",
                buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-01-02T14:30:00Z",
                buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn,
                1,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=100.0,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150.0,
            )
            record_alpaca_managed_sell_order(
                conn,
                1,
                sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id="sell-1",
                sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="accepted",
            )
            mock_get.return_value = response(
                200,
                {
                    "id": "sell-1",
                    "status": "expired",
                    "submitted_at": "2026-01-02T14:32:00Z",
                },
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            active_symbols = active_alpaca_managed_symbols(conn)

        self.assertEqual(result.loc[0, "Status"], "expired")
        self.assertIn("no automatic resubmission", result.loc[0, "Message"])
        self.assertIn("TQQQ", active_symbols)
        mock_post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
