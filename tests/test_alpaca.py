from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests
from requests import HTTPError

from leveraged_trader.alpaca import (
    AlpacaClient,
    _alpaca_dynamic_batch_cash_fraction,
    _record_observed_managed_sell_order,
    _recover_managed_sell_submission,
    migrate_alpaca_managed_position_symbols,
    reconcile_alpaca_managed_positions,
    submit_alpaca_paper_buy_orders,
    submit_alpaca_paper_sell_orders,
)
from leveraged_trader.config import AlpacaOrderConfig
from leveraged_trader.storage import (
    active_alpaca_managed_symbols,
    attach_alpaca_managed_sell_order_if_current,
    claim_alpaca_managed_sell_renewal,
    claim_alpaca_managed_sell_replacement,
    claim_alpaca_managed_sell_submission_retry,
    close_alpaca_managed_position_if_current_and_complete,
    claim_alpaca_managed_buy_intent,
    fail_alpaca_managed_buy_submission_if_pending,
    init_state_db,
    load_alpaca_managed_positions,
    mark_alpaca_managed_buy_filled,
    mark_alpaca_managed_sell_filled,
    mark_alpaca_managed_sell_filled_if_current,
    record_alpaca_managed_sell_order,
    save_alpaca_managed_buy_order,
    update_alpaca_managed_sell_status,
    update_alpaca_managed_sell_status_if_current,
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
    def cfg(self, *, buy: bool = False, sell: bool = False, **kwargs: object) -> AlpacaOrderConfig:
        return AlpacaOrderConfig(
            enabled=buy,
            sell_enabled=sell,
            api_key_id="key",
            api_secret_key="secret",
            **kwargs,
        )

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_ticker_rename_migrates_managed_position_and_renews_with_current_symbol(
        self, mock_get: Mock, mock_post: Mock,
    ) -> None:
        asset_id = "asset-echo"
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            save_alpaca_managed_buy_order(
                conn,
                symbol="SATG",
                signal_symbol="SATS",
                buy_rsi=30,
                profit_target_multiple=1.1,
                buy_signal_date="2026-06-18",
                buy_client_order_id="rsi-buy-SATG-20260618",
                buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-06-22T13:30:00Z",
                buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn,
                1,
                buy_status="filled",
                filled_qty=73,
                filled_avg_price=12.203151,
                filled_at="2026-06-22T13:36:51Z",
                target_sell_price=13.42,
            )
            record_alpaca_managed_sell_order(
                conn,
                1,
                sell_client_order_id="rsi-exit-SATG-1",
                sell_alpaca_order_id="sell-1",
                sell_submitted_at="2026-06-22T17:30:44Z",
                sell_status="new",
                sell_expires_at="2026-09-18T20:00:00Z",
            )
            mock_get.side_effect = [
                response(200, [{"asset_id": asset_id, "symbol": "ECHX", "qty": "73"}]),
                response(
                    200,
                    {
                        "id": "sell-1",
                        "client_order_id": "rsi-exit-SATG-1",
                        "asset_id": asset_id,
                        "symbol": "SATG",
                    },
                ),
                response(
                    200,
                    {
                        "id": "sell-1",
                        "client_order_id": "rsi-exit-SATG-1",
                        "asset_id": asset_id,
                        "symbol": "SATG",
                        "status": "expired",
                    },
                ),
                response(200, []),
            ]
            mock_post.return_value = response(
                200,
                {
                    "id": "sell-2",
                    "client_order_id": "rsi-exit-ECHX-1-r1",
                    "asset_id": asset_id,
                    "symbol": "ECHX",
                    "status": "accepted",
                    "expires_at": "2027-06-30T20:00:00Z",
                },
            )

            migrations = migrate_alpaca_managed_position_symbols(conn, self.cfg(buy=True, sell=True))
            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)
            protected_symbols = active_alpaca_managed_symbols(conn)

        self.assertEqual(migrations, {"SATG": "ECHX"})
        self.assertEqual(result.loc[0, "Status"], "renewed")
        self.assertEqual(managed.loc[0, "symbol"], "ECHX")
        self.assertEqual(managed.loc[0, "alpaca_asset_id"], asset_id)
        self.assertEqual(managed.loc[0, "sell_client_order_id"], "rsi-exit-ECHX-1-r1")
        self.assertEqual(protected_symbols, {"SATG", "ECHX"})
        self.assertEqual(mock_post.call_args.kwargs["json"]["symbol"], "ECHX")

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_ticker_rename_recovers_unattached_open_sell_by_asset_id(
        self, mock_get: Mock, mock_post: Mock,
    ) -> None:
        asset_id = "asset-echo"
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            save_alpaca_managed_buy_order(
                conn,
                symbol="SATG",
                signal_symbol="SATS",
                buy_rsi=30,
                profit_target_multiple=1.1,
                buy_signal_date="2026-06-18",
                buy_client_order_id="rsi-buy-SATG-20260618",
                buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-06-22T13:30:00Z",
                buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn,
                1,
                buy_status="filled",
                filled_qty=73,
                filled_avg_price=12.203151,
                filled_at="2026-06-22T13:36:51Z",
                target_sell_price=13.42,
            )
            mock_get.side_effect = [
                response(200, [{"asset_id": asset_id, "symbol": "ECHX", "qty": "73"}]),
                response(
                    200,
                    {
                        "id": "buy-1",
                        "client_order_id": "rsi-buy-SATG-20260618",
                        "asset_id": asset_id,
                        "symbol": "SATG",
                    },
                ),
                response(
                    200,
                    [
                        {
                            "id": "sell-1",
                            "client_order_id": "rsi-exit-SATG-1",
                            "asset_id": asset_id,
                            "symbol": "SATG",
                            "side": "sell",
                            "status": "new",
                            "qty": "73",
                            "filled_qty": "0",
                            "limit_price": "13.42",
                            "expires_at": "2027-06-30T20:00:00Z",
                        }
                    ],
                ),
            ]

            migrate_alpaca_managed_position_symbols(conn, self.cfg(buy=True, sell=True))
            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "new")
        self.assertIn("recovered existing", result.loc[0, "Message"])
        self.assertEqual(managed.loc[0, "symbol"], "ECHX")
        self.assertEqual(managed.loc[0, "sell_client_order_id"], "rsi-exit-SATG-1")
        self.assertEqual(managed.loc[0, "sell_alpaca_order_id"], "sell-1")
        mock_post.assert_not_called()

    @patch("leveraged_trader.alpaca.requests.get")
    def test_ticker_migration_rejects_two_active_rows_resolving_to_same_asset(
        self, mock_get: Mock,
    ) -> None:
        asset_id = "asset-echo"
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            for symbol, order_id, signal_date in [
                ("SATG", "buy-1", "2026-06-18"),
                ("LEGACY", "buy-2", "2026-06-19"),
            ]:
                save_alpaca_managed_buy_order(
                    conn,
                    symbol=symbol,
                    signal_symbol="SATS",
                    buy_rsi=30,
                    profit_target_multiple=1.1,
                    buy_signal_date=signal_date,
                    buy_client_order_id=f"rsi-buy-{symbol}-{signal_date.replace('-', '')}",
                    buy_alpaca_order_id=order_id,
                    buy_submitted_at="2026-06-22T13:30:00Z",
                    buy_status="filled",
                )
            mock_get.side_effect = [
                response(200, [{"asset_id": asset_id, "symbol": "ECHX", "qty": "73"}]),
                response(200, {"id": "buy-1", "client_order_id": "rsi-buy-SATG-20260618", "asset_id": asset_id, "symbol": "SATG"}),
                response(200, {"id": "buy-2", "client_order_id": "rsi-buy-LEGACY-20260619", "asset_id": asset_id, "symbol": "LEGACY"}),
            ]

            with self.assertRaisesRegex(ValueError, "resolve to the same Alpaca asset"):
                migrate_alpaca_managed_position_symbols(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)
            aliases = conn.execute("SELECT COUNT(*) FROM alpaca_symbol_aliases").fetchone()[0]

        self.assertEqual(managed["symbol"].tolist(), ["SATG", "LEGACY"])
        self.assertTrue(managed["alpaca_asset_id"].isna().all())
        self.assertEqual(aliases, 0)

    @patch("leveraged_trader.alpaca.requests.get")
    def test_ticker_migration_rolls_back_entire_batch_when_later_update_fails(
        self, mock_get: Mock,
    ) -> None:
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            for symbol, order_id, signal_date in [
                ("OLD1", "buy-1", "2026-06-18"),
                ("OLD2", "buy-2", "2026-06-19"),
            ]:
                save_alpaca_managed_buy_order(
                    conn,
                    symbol=symbol,
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.1,
                    buy_signal_date=signal_date,
                    buy_client_order_id=f"rsi-buy-{symbol}-{signal_date.replace('-', '')}",
                    buy_alpaca_order_id=order_id,
                    buy_submitted_at="2026-06-22T13:30:00Z",
                    buy_status="filled",
                )
            conn.execute(
                """
                CREATE TRIGGER fail_second_symbol_migration
                BEFORE UPDATE OF symbol ON alpaca_managed_positions
                WHEN NEW.id = 2
                BEGIN
                    SELECT RAISE(ABORT, 'forced second migration failure');
                END
                """
            )
            mock_get.side_effect = [
                response(
                    200,
                    [
                        {"asset_id": "asset-1", "symbol": "NEW1", "qty": "1"},
                        {"asset_id": "asset-2", "symbol": "NEW2", "qty": "1"},
                    ],
                ),
                response(200, {"id": "buy-1", "client_order_id": "rsi-buy-OLD1-20260618", "asset_id": "asset-1", "symbol": "OLD1"}),
                response(200, {"id": "buy-2", "client_order_id": "rsi-buy-OLD2-20260619", "asset_id": "asset-2", "symbol": "OLD2"}),
            ]

            with self.assertRaisesRegex(sqlite3.IntegrityError, "forced second migration failure"):
                migrate_alpaca_managed_position_symbols(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)
            aliases = conn.execute("SELECT COUNT(*) FROM alpaca_symbol_aliases").fetchone()[0]

        self.assertEqual(managed["symbol"].tolist(), ["OLD1", "OLD2"])
        self.assertTrue(managed["alpaca_asset_id"].isna().all())
        self.assertEqual(aliases, 0)

    @patch("leveraged_trader.alpaca.requests.get")
    def test_ticker_migration_blocks_mismatched_order_identity_without_fallback(
        self, mock_get: Mock,
    ) -> None:
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            save_alpaca_managed_buy_order(
                conn,
                symbol="SATG",
                signal_symbol="SATS",
                buy_rsi=30,
                profit_target_multiple=1.1,
                buy_signal_date="2026-06-18",
                buy_client_order_id="rsi-buy-SATG-20260618",
                buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-06-22T13:30:00Z",
                buy_status="filled",
            )
            record_alpaca_managed_sell_order(
                conn,
                1,
                sell_client_order_id="rsi-exit-SATG-1",
                sell_alpaca_order_id="sell-1",
                sell_submitted_at="2026-06-22T17:30:44Z",
                sell_status="new",
            )
            mock_get.side_effect = [
                response(200, [{"asset_id": "wrong-asset", "symbol": "WRONG", "qty": "1"}]),
                response(
                    200,
                    {
                        "id": "sell-1",
                        "client_order_id": "unrelated-sell",
                        "asset_id": "wrong-asset",
                        "symbol": "WRONG",
                    },
                ),
            ]

            with self.assertRaisesRegex(ValueError, "identity does not match"):
                migrate_alpaca_managed_position_symbols(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)
            aliases = conn.execute("SELECT COUNT(*) FROM alpaca_symbol_aliases").fetchone()[0]

        self.assertEqual(managed.loc[0, "symbol"], "SATG")
        self.assertTrue(pd.isna(managed.loc[0, "alpaca_asset_id"]))
        self.assertEqual(aliases, 0)
        self.assertEqual(mock_get.call_count, 2)

    @patch("leveraged_trader.alpaca.requests.get")
    def test_ticker_migration_revalidates_persisted_asset_id_against_attached_order(
        self, mock_get: Mock,
    ) -> None:
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            save_alpaca_managed_buy_order(
                conn,
                symbol="SATG",
                signal_symbol="SATS",
                buy_rsi=30,
                profit_target_multiple=1.1,
                buy_signal_date="2026-06-18",
                buy_client_order_id="rsi-buy-SATG-20260618",
                buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-06-22T13:30:00Z",
                buy_status="filled",
            )
            conn.execute("UPDATE alpaca_managed_positions SET alpaca_asset_id = 'persisted-asset'")
            conn.commit()
            mock_get.side_effect = [
                response(200, [{"asset_id": "persisted-asset", "symbol": "ECHX", "qty": "73"}]),
                response(
                    200,
                    {
                        "id": "buy-1",
                        "client_order_id": "rsi-buy-SATG-20260618",
                        "asset_id": "different-asset",
                        "symbol": "SATG",
                    },
                ),
            ]

            with self.assertRaisesRegex(ValueError, "asset identity does not match"):
                migrate_alpaca_managed_position_symbols(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(managed.loc[0, "symbol"], "SATG")
        self.assertEqual(managed.loc[0, "alpaca_asset_id"], "persisted-asset")

    @patch("leveraged_trader.alpaca.requests.get")
    def test_ticker_migration_blocks_mismatched_asset_lookup_response(
        self, mock_get: Mock,
    ) -> None:
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            save_alpaca_managed_buy_order(
                conn,
                symbol="OLD",
                signal_symbol="QQQ",
                buy_rsi=30,
                profit_target_multiple=1.1,
                buy_signal_date="2026-06-18",
                buy_client_order_id="rsi-buy-OLD-20260618",
                buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-06-22T13:30:00Z",
                buy_status="filled",
            )
            mock_get.side_effect = [
                response(200, []),
                response(
                    200,
                    {
                        "id": "buy-1",
                        "client_order_id": "rsi-buy-OLD-20260618",
                        "asset_id": "expected-asset",
                        "symbol": "OLD",
                    },
                ),
                response(200, {"id": "different-asset", "symbol": "WRONG"}),
            ]

            with self.assertRaisesRegex(ValueError, "asset lookup.*does not match"):
                migrate_alpaca_managed_position_symbols(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)
            aliases = conn.execute("SELECT COUNT(*) FROM alpaca_symbol_aliases").fetchone()[0]

        self.assertEqual(managed.loc[0, "symbol"], "OLD")
        self.assertTrue(pd.isna(managed.loc[0, "alpaca_asset_id"]))
        self.assertEqual(aliases, 0)

    @patch("leveraged_trader.alpaca.requests.get")
    def test_ticker_migration_skips_order_lookup_for_unchanged_persisted_asset(
        self, mock_get: Mock,
    ) -> None:
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            save_alpaca_managed_buy_order(
                conn,
                symbol="ECHX",
                signal_symbol="ECHO",
                buy_rsi=30,
                profit_target_multiple=1.1,
                buy_signal_date="2026-06-18",
                buy_client_order_id="rsi-buy-SATG-20260618",
                buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-06-22T13:30:00Z",
                buy_status="filled",
            )
            conn.execute("UPDATE alpaca_managed_positions SET alpaca_asset_id = 'asset-echo'")
            conn.commit()
            mock_get.return_value = response(
                200,
                [{"asset_id": "asset-echo", "symbol": "ECHX", "qty": "73"}],
            )

            migrations = migrate_alpaca_managed_position_symbols(conn, self.cfg(buy=True, sell=True))

        self.assertEqual(migrations, {})
        self.assertEqual(mock_get.call_count, 1)
        self.assertTrue(mock_get.call_args.args[0].endswith("/v2/positions"))

    def test_managed_sell_renewal_claim_is_exclusive_until_lease_expires(self) -> None:
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
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
            )
            record_alpaca_managed_sell_order(
                conn,
                1,
                sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id="sell-1",
                sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="accepted",
            )

            first = claim_alpaca_managed_sell_renewal(
                conn,
                1,
                sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id="sell-1",
                requested_at="2026-03-25T14:32:00Z",
                reclaim_before="2026-03-25T14:27:00Z",
                notes="first claim",
            )
            overlapping = claim_alpaca_managed_sell_renewal(
                conn,
                1,
                sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id="sell-1",
                requested_at="2026-03-25T14:33:00Z",
                reclaim_before="2026-03-25T14:28:00Z",
                notes="overlapping claim",
            )
            reclaimed = claim_alpaca_managed_sell_renewal(
                conn,
                1,
                sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id="sell-1",
                requested_at="2026-03-25T14:38:00Z",
                reclaim_before="2026-03-25T14:33:00Z",
                notes="reclaimed",
            )
            conn.execute(
                """
                UPDATE alpaca_managed_positions
                SET sell_status = 'pending_cancel', sell_renewal_requested_at = 'not-a-timestamp'
                WHERE id = 1
                """
            )
            malformed_timestamp_reclaimed = claim_alpaca_managed_sell_renewal(
                conn,
                1,
                sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id="sell-1",
                requested_at="2026-03-25T14:39:00Z",
                reclaim_before="2026-03-25T14:34:00Z",
                notes="reclaimed malformed timestamp",
            )
            conn.execute(
                """
                UPDATE alpaca_managed_positions
                SET sell_status = 'pending_cancel', sell_renewal_requested_at = '2026-03-25 14:30:00'
                WHERE id = 1
                """
            )
            legacy_timestamp_reclaimed = claim_alpaca_managed_sell_renewal(
                conn,
                1,
                sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id="sell-1",
                requested_at="2026-03-25T14:40:00Z",
                reclaim_before="2026-03-25T14:35:00Z",
                notes="reclaimed legacy timestamp",
            )
            conn.execute(
                "UPDATE alpaca_managed_positions SET sell_client_order_id = ? WHERE id = 1",
                ("rsi-exit-TQQQ-1-r1",),
            )
            stale_generation = claim_alpaca_managed_sell_renewal(
                conn,
                1,
                sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id="sell-1",
                requested_at="2026-03-25T14:44:00Z",
                reclaim_before="2026-03-25T14:39:00Z",
                notes="stale generation",
            )
            conn.execute(
                """
                UPDATE alpaca_managed_positions
                SET sell_client_order_id = ?, closed_at = CURRENT_TIMESTAMP,
                    sell_status = 'accepted', sell_renewal_requested_at = NULL
                WHERE id = 1
                """,
                ("rsi-exit-TQQQ-1",),
            )
            closed_position = claim_alpaca_managed_sell_renewal(
                conn,
                1,
                sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id="sell-1",
                requested_at="2026-03-25T14:44:00Z",
                reclaim_before="2026-03-25T14:39:00Z",
                notes="closed position",
            )
            conn.execute(
                """
                UPDATE alpaca_managed_positions
                SET closed_at = NULL, remaining_qty = 0, sell_status = 'accepted'
                WHERE id = 1
                """
            )
            completed_position = claim_alpaca_managed_sell_renewal(
                conn,
                1,
                sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id="sell-1",
                requested_at="2026-03-25T14:44:00Z",
                reclaim_before="2026-03-25T14:39:00Z",
                notes="completed position",
            )
            conn.execute(
                """
                UPDATE alpaca_managed_positions
                SET remaining_qty = 2, sell_status = 'incomplete_fill_metadata'
                WHERE id = 1
                """
            )
            safety_blocked = claim_alpaca_managed_sell_renewal(
                conn,
                1,
                sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id="sell-1",
                requested_at="2026-03-25T14:44:00Z",
                reclaim_before="2026-03-25T14:39:00Z",
                notes="safety-blocked position",
            )

        self.assertTrue(first)
        self.assertFalse(overlapping)
        self.assertTrue(reclaimed)
        self.assertTrue(malformed_timestamp_reclaimed)
        self.assertTrue(legacy_timestamp_reclaimed)
        self.assertFalse(stale_generation)
        self.assertFalse(closed_position)
        self.assertFalse(completed_position)
        self.assertFalse(safety_blocked)

    def test_managed_sell_replacement_claim_is_atomic_across_connections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "state.db"
            with sqlite3.connect(db_path) as setup_conn:
                init_state_db(setup_conn)
                save_alpaca_managed_buy_order(
                    setup_conn,
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
                    setup_conn,
                    1,
                    buy_status="filled",
                    filled_qty=2,
                    filled_avg_price=100,
                    filled_at="2026-01-02T14:31:00Z",
                    target_sell_price=150,
                )
                record_alpaca_managed_sell_order(
                    setup_conn,
                    1,
                    sell_client_order_id="rsi-exit-TQQQ-1",
                    sell_alpaca_order_id="sell-1",
                    sell_submitted_at="2026-01-02T14:32:00Z",
                    sell_status="expired",
                )

            with sqlite3.connect(db_path) as first_conn, sqlite3.connect(db_path) as second_conn:
                first = claim_alpaca_managed_sell_replacement(
                    first_conn,
                    1,
                    prior_sell_client_order_id="rsi-exit-TQQQ-1",
                    prior_sell_alpaca_order_id="sell-1",
                    prior_renewal_count=0,
                    replacement_sell_client_order_id="rsi-exit-TQQQ-1-r1",
                    requested_remaining_qty=2,
                    notes="first replacement claim",
                )
                overlapping = claim_alpaca_managed_sell_replacement(
                    second_conn,
                    1,
                    prior_sell_client_order_id="rsi-exit-TQQQ-1",
                    prior_sell_alpaca_order_id="sell-1",
                    prior_renewal_count=0,
                    replacement_sell_client_order_id="rsi-exit-TQQQ-1-r1",
                    requested_remaining_qty=2,
                    notes="overlapping replacement claim",
                )
                stale_observation_updated = update_alpaca_managed_sell_status_if_current(
                    second_conn,
                    1,
                    expected_sell_client_order_id="rsi-exit-TQQQ-1",
                    sell_status="expired",
                    sell_alpaca_order_id="sell-1",
                    sell_expires_at="2026-03-30T20:15:00Z",
                )
                stale_manual_review_updated = update_alpaca_managed_sell_status_if_current(
                    second_conn,
                    1,
                    expected_sell_client_order_id="rsi-exit-TQQQ-1",
                    sell_status="quantity_mismatch",
                    notes="stale manual review result",
                )
                stale_null_generation_updated = update_alpaca_managed_sell_status_if_current(
                    second_conn,
                    1,
                    expected_sell_client_order_id=None,
                    sell_status="fractional_qty",
                    notes="stale initial-submission result",
                )
                _, _, stale_submission_response_updated = _record_observed_managed_sell_order(
                    conn=second_conn,
                    position_id=1,
                    sell_client_order_id="rsi-exit-TQQQ-1",
                    sell_order={
                        "id": "sell-1",
                        "status": "accepted",
                        "submitted_at": "2026-03-30T19:30:00Z",
                    },
                )
                stale_open_order_attached = attach_alpaca_managed_sell_order_if_current(
                    second_conn,
                    1,
                    expected_sell_client_order_id="rsi-exit-TQQQ-1",
                    expected_sell_alpaca_order_id="sell-1",
                    expected_renewal_count=0,
                    sell_renewal_count=0,
                    sell_client_order_id="rsi-exit-TQQQ-1",
                    sell_alpaca_order_id="sell-1",
                    sell_submitted_at="2026-01-02T14:32:00Z",
                    sell_status="accepted",
                    sell_expires_at="2026-03-30T20:15:00Z",
                    notes="stale open-order recovery",
                )
                remaining_qty, stale_fill_updated_generation = mark_alpaca_managed_sell_filled_if_current(
                    second_conn,
                    1,
                    expected_sell_client_order_id="rsi-exit-TQQQ-1",
                    sell_status="expired",
                    sell_filled_qty=1,
                    sell_filled_avg_price=150,
                    sell_filled_at="2026-03-30T19:00:00Z",
                    sell_alpaca_order_id="sell-1",
                    sell_expires_at="2026-03-30T20:15:00Z",
                )
                oversized_retry = claim_alpaca_managed_sell_submission_retry(
                    second_conn,
                    1,
                    sell_client_order_id="rsi-exit-TQQQ-1-r1",
                    claimed_at="2026-03-30T19:01:00Z",
                    reclaim_before="2026-03-30T18:56:00Z",
                    notes="over-sized retry",
                )
                row = second_conn.execute(
                    """
                    SELECT sell_client_order_id, sell_alpaca_order_id, sell_status,
                           sell_renewal_count, sold_qty, remaining_qty
                    FROM alpaca_managed_positions WHERE id = 1
                    """
                ).fetchone()

        self.assertTrue(first)
        self.assertFalse(overlapping)
        self.assertFalse(stale_observation_updated)
        self.assertFalse(stale_manual_review_updated)
        self.assertFalse(stale_null_generation_updated)
        self.assertFalse(stale_submission_response_updated)
        self.assertFalse(stale_open_order_attached)
        self.assertFalse(stale_fill_updated_generation)
        self.assertIsNone(oversized_retry)
        self.assertEqual(remaining_qty, 1)
        self.assertEqual(row, ("rsi-exit-TQQQ-1-r1", None, "submission_pending", 1, 1, 1))

    def test_managed_sell_replacement_claim_rejects_concurrently_closed_position(self) -> None:
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
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
            )
            record_alpaca_managed_sell_order(
                conn,
                1,
                sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id="sell-1",
                sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="canceled",
            )
            conn.execute(
                "UPDATE alpaca_managed_positions SET closed_at = CURRENT_TIMESTAMP WHERE id = 1"
            )

            claimed_qty = claim_alpaca_managed_sell_replacement(
                conn,
                1,
                prior_sell_client_order_id="rsi-exit-TQQQ-1",
                prior_sell_alpaca_order_id="sell-1",
                prior_renewal_count=0,
                replacement_sell_client_order_id="rsi-exit-TQQQ-1-r1",
                requested_remaining_qty=2,
                notes="stale replacement after concurrent close",
            )
            conn.execute(
                """
                UPDATE alpaca_managed_positions
                SET closed_at = NULL, sell_status = 'stopped'
                WHERE id = 1
                """
            )
            stopped_claimed_qty = claim_alpaca_managed_sell_replacement(
                conn,
                1,
                prior_sell_client_order_id="rsi-exit-TQQQ-1",
                prior_sell_alpaca_order_id="sell-1",
                prior_renewal_count=0,
                replacement_sell_client_order_id="rsi-exit-TQQQ-1-r1",
                requested_remaining_qty=2,
                notes="unsafe stopped replacement",
            )
            row = conn.execute(
                "SELECT sell_client_order_id, sell_status, sell_renewal_count FROM alpaca_managed_positions WHERE id = 1"
            ).fetchone()

        self.assertIsNone(claimed_qty)
        self.assertIsNone(stopped_claimed_qty)
        self.assertEqual(row, ("rsi-exit-TQQQ-1", "stopped", 0))

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_recovered_r2_managed_sell_advances_next_replacement_to_r3(
        self, mock_get: Mock, mock_post: Mock,
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
                buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn,
                1,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
            )
            attached = attach_alpaca_managed_sell_order_if_current(
                conn,
                1,
                expected_sell_client_order_id=None,
                expected_sell_alpaca_order_id=None,
                expected_renewal_count=0,
                sell_renewal_count=2,
                sell_client_order_id="rsi-exit-TQQQ-1-r2",
                sell_alpaca_order_id="sell-2",
                sell_submitted_at="2026-03-25T14:32:00Z",
                sell_status="canceled",
                sell_expires_at="2026-03-30T20:15:00Z",
            )
            backward_attached = attach_alpaca_managed_sell_order_if_current(
                conn,
                1,
                expected_sell_client_order_id="rsi-exit-TQQQ-1-r2",
                expected_sell_alpaca_order_id="sell-2",
                expected_renewal_count=2,
                sell_renewal_count=1,
                sell_client_order_id="rsi-exit-TQQQ-1-r1",
                sell_alpaca_order_id="sell-1",
                sell_submitted_at="2026-03-25T14:31:00Z",
                sell_status="accepted",
                sell_expires_at="2026-03-30T20:15:00Z",
            )
            update_alpaca_managed_sell_status_if_current(
                conn,
                1,
                expected_sell_client_order_id="rsi-exit-TQQQ-1-r2",
                sell_status="canceled",
                sell_renewal_requested_at="2026-03-25T14:32:00Z",
            )
            mock_get.side_effect = [
                response(200, {"id": "sell-2", "status": "canceled"}),
                response(200, []),
            ]
            mock_post.return_value = response(200, {"id": "sell-3", "status": "accepted"})

            result = reconcile_alpaca_managed_positions(
                conn,
                self.cfg(buy=True, sell=True, gtc_sell_renewal_enabled=False),
            )
            managed = load_alpaca_managed_positions(conn)

        self.assertTrue(attached)
        self.assertFalse(backward_attached)
        self.assertEqual(result.loc[0, "Status"], "renewed")
        self.assertEqual(mock_post.call_args.kwargs["json"]["client_order_id"], "rsi-exit-TQQQ-1-r3")
        self.assertEqual(managed.loc[0, "sell_renewal_count"], 3)
        self.assertEqual(managed.loc[0, "sell_client_order_id"], "rsi-exit-TQQQ-1-r3")

    def test_open_sell_attachment_rejects_closed_or_completed_position(self) -> None:
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
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
            )
            conn.execute(
                "UPDATE alpaca_managed_positions SET closed_at = CURRENT_TIMESTAMP WHERE id = 1"
            )
            closed_attached = attach_alpaca_managed_sell_order_if_current(
                conn,
                1,
                expected_sell_client_order_id=None,
                expected_sell_alpaca_order_id=None,
                expected_renewal_count=0,
                sell_renewal_count=1,
                sell_client_order_id="rsi-exit-TQQQ-1-r1",
                sell_alpaca_order_id="sell-1",
                sell_submitted_at="2026-03-25T14:32:00Z",
                sell_status="accepted",
                sell_expires_at="2026-06-23T20:15:00Z",
            )
            conn.execute(
                "UPDATE alpaca_managed_positions SET closed_at = NULL, remaining_qty = 0 WHERE id = 1"
            )
            completed_attached = attach_alpaca_managed_sell_order_if_current(
                conn,
                1,
                expected_sell_client_order_id=None,
                expected_sell_alpaca_order_id=None,
                expected_renewal_count=0,
                sell_renewal_count=1,
                sell_client_order_id="rsi-exit-TQQQ-1-r1",
                sell_alpaca_order_id="sell-1",
                sell_submitted_at="2026-03-25T14:32:00Z",
                sell_status="accepted",
                sell_expires_at="2026-06-23T20:15:00Z",
            )
            row = conn.execute(
                "SELECT sell_client_order_id, sell_alpaca_order_id, sell_renewal_count FROM alpaca_managed_positions WHERE id = 1"
            ).fetchone()

        self.assertFalse(closed_attached)
        self.assertFalse(completed_attached)
        self.assertEqual(row, (None, None, 0))

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_initial_recovery_rejects_multiple_managed_sell_generations(
        self, mock_get: Mock, mock_post: Mock,
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
                buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn,
                1,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
            )
            mock_get.return_value = response(
                200,
                [
                    {"id": "sell-1", "symbol": "TQQQ", "side": "sell", "client_order_id": "rsi-exit-TQQQ-1-r1"},
                    {"id": "sell-2", "symbol": "TQQQ", "side": "sell", "client_order_id": "rsi-exit-TQQQ-1-r2"},
                ],
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "managed_order_conflict")
        self.assertTrue(pd.isna(managed.loc[0, "sell_client_order_id"]))
        self.assertEqual(managed.loc[0, "sell_renewal_count"], 0)
        mock_post.assert_not_called()

    def test_final_fill_blocks_stale_replacement_claim_and_allows_guarded_close(self) -> None:
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
                conn, 1, buy_status="filled", filled_qty=2, filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z", target_sell_price=150,
            )
            record_alpaca_managed_sell_order(
                conn, 1, sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id="sell-1", sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="expired",
            )
            remaining, current = mark_alpaca_managed_sell_filled_if_current(
                conn, 1, expected_sell_client_order_id="rsi-exit-TQQQ-1",
                sell_status="filled", sell_filled_qty=2, sell_filled_avg_price=150,
                sell_filled_at="2026-03-30T19:00:00Z", sell_alpaca_order_id="sell-1",
            )
            stale_claim = claim_alpaca_managed_sell_replacement(
                conn, 1, prior_sell_client_order_id="rsi-exit-TQQQ-1",
                prior_sell_alpaca_order_id="sell-1", prior_renewal_count=0,
                replacement_sell_client_order_id="rsi-exit-TQQQ-1-r1",
                requested_remaining_qty=2, notes="stale replacement",
            )
            stale_retry = claim_alpaca_managed_sell_submission_retry(
                conn,
                1,
                sell_client_order_id="rsi-exit-TQQQ-1",
                claimed_at="2026-03-30T19:01:00Z",
                reclaim_before="2026-03-30T18:56:00Z",
                notes="stale 404 retry",
            )
            closed = close_alpaca_managed_position_if_current_and_complete(
                conn, 1, expected_sell_client_order_id="rsi-exit-TQQQ-1",
                closed_at="2026-03-30T19:00:00Z", notes="filled",
            )

        self.assertTrue(current)
        self.assertEqual(remaining, 0)
        self.assertIsNone(stale_claim)
        self.assertIsNone(stale_retry)
        self.assertTrue(closed)

    def test_parent_buy_fill_cannot_move_backward(self) -> None:
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
                buy_status="partially_filled",
            )
            current = mark_alpaca_managed_buy_filled(
                conn, 1, buy_status="partially_filled", filled_qty=3,
                filled_avg_price=101, filled_at=None, target_sell_price=151.5,
            )
            stale = mark_alpaca_managed_buy_filled(
                conn, 1, buy_status="partially_filled", filled_qty=2,
                filled_avg_price=100, filled_at=None, target_sell_price=150,
            )
            row = conn.execute(
                "SELECT filled_qty, filled_avg_price, target_sell_price, remaining_qty "
                "FROM alpaca_managed_positions WHERE id = 1"
            ).fetchone()

        self.assertTrue(current)
        self.assertFalse(stale)
        self.assertEqual(row, (3, 101, 151.5, 3))

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_replacement_404_recovery_uses_persisted_remaining_quantity(
        self, mock_get: Mock, mock_post: Mock,
    ) -> None:
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            save_alpaca_managed_buy_order(
                conn, symbol="TQQQ", signal_symbol="QQQ", buy_rsi=30,
                profit_target_multiple=1.5, buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-20260102", buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-01-02T14:30:00Z", buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn, 1, buy_status="filled", filled_qty=2, filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z", target_sell_price=150,
            )
            record_alpaca_managed_sell_order(
                conn, 1, sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id="sell-1", sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="expired", sell_order_qty=2,
            )
            mark_alpaca_managed_sell_filled(
                conn, 1, sell_status="expired", sell_filled_qty=1,
                sell_filled_avg_price=150, sell_filled_at="2026-03-30T19:00:00Z",
                sell_alpaca_order_id="sell-1",
            )
            claimed_qty = claim_alpaca_managed_sell_replacement(
                conn, 1, prior_sell_client_order_id="rsi-exit-TQQQ-1",
                prior_sell_alpaca_order_id="sell-1", prior_renewal_count=0,
                replacement_sell_client_order_id="rsi-exit-TQQQ-1-r1",
                requested_remaining_qty=1, notes="replacement claimed",
            )
            conn.execute(
                """
                UPDATE alpaca_managed_positions
                SET sell_status = 'submission_retrying',
                    sell_submission_retry_claimed_at = 'not-a-timestamp'
                WHERE id = 1
                """
            )
            malformed_retry_qty = claim_alpaca_managed_sell_submission_retry(
                conn, 1, sell_client_order_id="rsi-exit-TQQQ-1-r1",
                claimed_at="2026-03-30T19:10:00Z",
                reclaim_before="2026-03-30T19:05:00Z",
                notes="reclaimed malformed retry timestamp",
            )
            conn.execute(
                """
                UPDATE alpaca_managed_positions
                SET sell_status = 'submission_retrying',
                    sell_submission_retry_claimed_at = '2026-03-30 19:00:00'
                WHERE id = 1
                """
            )
            legacy_retry_qty = claim_alpaca_managed_sell_submission_retry(
                conn, 1, sell_client_order_id="rsi-exit-TQQQ-1-r1",
                claimed_at="2026-03-30T19:11:00Z",
                reclaim_before="2026-03-30T19:06:00Z",
                notes="reclaimed legacy retry timestamp",
            )
            conn.execute(
                """
                UPDATE alpaca_managed_positions
                SET sell_status = 'submission_pending', sell_submission_retry_claimed_at = NULL
                WHERE id = 1
                """
            )
            parent_fill_increased = mark_alpaca_managed_buy_filled(
                conn, 1, buy_status="filled", filled_qty=3, filled_avg_price=100,
                filled_at="2026-01-02T14:31:30Z", target_sell_price=150,
            )
            mock_get.return_value = error_response(404, {})
            competing_claims: list[float | None] = []

            def submit_while_competing_claim_is_attempted(*args: object, **kwargs: object) -> Mock:
                competing_claims.append(
                    claim_alpaca_managed_sell_submission_retry(
                        conn, 1, sell_client_order_id="rsi-exit-TQQQ-1-r1",
                        claimed_at="2099-01-01T00:00:00Z",
                        reclaim_before="2000-01-01T00:00:00Z",
                        notes="competing retry",
                    )
                )
                return response(200, {"id": "sell-2", "status": "accepted", "qty": "1"})

            mock_post.side_effect = submit_while_competing_claim_is_attempted

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))

        self.assertEqual(claimed_qty, 1)
        self.assertEqual(malformed_retry_qty, 1)
        self.assertEqual(legacy_retry_qty, 1)
        self.assertTrue(parent_fill_increased)
        self.assertEqual(result.loc[0, "Status"], "renewed")
        self.assertEqual(mock_post.call_args.kwargs["json"]["qty"], "1")
        self.assertEqual(competing_claims, [None])

    @patch("leveraged_trader.alpaca._latest_market_price")
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_buy_order_uses_5_percent_cash_for_one_eligible_signal(
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
            response(
                200,
                {"is_open": False, "timestamp": "2026-01-05T13:30:00Z", "next_open": "2026-01-05T14:30:00Z"},
            ),
            response(200, [{"date": "2026-01-02"}, {"date": "2026-01-05"}]),
            response(200, {"cash": "12345.67"}),
        ]
        mock_latest_market_price.return_value = 100.0
        mock_post.return_value = response(200, {"id": "order-1", "status": "accepted"})

        result = submit_alpaca_paper_buy_orders(
            pd.DataFrame([{"Asset": "TQQQ", "Date": "2026-01-02"}]),
            self.cfg(buy=True),
        )

        self.assertEqual(result.loc[0, "Status"], "submitted")
        self.assertEqual(result.loc[0, "Notional"], 525.0)
        self.assertEqual(result.loc[0, "Qty"], 5)
        self.assertEqual(result.loc[0, "Limit Price"], 105.0)
        self.assertEqual(mock_post.call_args.kwargs["json"]["qty"], "5")
        self.assertEqual(mock_post.call_args.kwargs["json"]["type"], "limit")
        self.assertEqual(mock_post.call_args.kwargs["json"]["limit_price"], "105")
        self.assertFalse(mock_post.call_args.kwargs["json"]["extended_hours"])

    @patch("leveraged_trader.alpaca._latest_market_price")
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_buy_order_preserves_workflow_side_in_results_and_managed_position(
        self,
        mock_get: Mock,
        mock_post: Mock,
        mock_latest_market_price: Mock,
    ) -> None:
        mock_get.side_effect = [
            response(404, {}),
            response(200, []),
            response(404, {}),
            response(200, {"symbol": "SQQQ", "status": "active", "tradable": True}),
            response(
                200,
                {"is_open": False, "timestamp": "2026-01-05T13:30:00Z", "next_open": "2026-01-05T14:30:00Z"},
            ),
            response(200, [{"date": "2026-01-02"}, {"date": "2026-01-05"}]),
            response(200, {"cash": "3000"}),
        ]
        mock_latest_market_price.return_value = 100.0
        mock_post.return_value = response(
            200,
            {"id": "order-short-1", "status": "accepted", "submitted_at": "2026-01-05T13:31:00Z"},
        )

        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            result = submit_alpaca_paper_buy_orders(
                pd.DataFrame(
                    [
                        {
                            "Workflow": "Short",
                            "Asset": "SQQQ",
                            "RSI Symbol": "QQQ",
                            "Date": "2026-01-02",
                            "Buy RSI": 70.0,
                            "Sell Return Multiple": 1.5,
                        }
                    ]
                ),
                self.cfg(buy=True),
                conn=conn,
            )
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Workflow"], "Short")
        self.assertEqual(result.loc[0, "Status"], "submitted")
        self.assertEqual(managed.loc[0, "workflow"], "Short")
        self.assertEqual(managed.loc[0, "symbol"], "SQQQ")

    def test_dynamic_batch_cash_fraction_scales_with_eligible_signal_count(self) -> None:
        self.assertEqual(_alpaca_dynamic_batch_cash_fraction(1), 0.05)
        self.assertEqual(_alpaca_dynamic_batch_cash_fraction(2), 0.10)
        self.assertEqual(_alpaca_dynamic_batch_cash_fraction(9), 0.45)
        self.assertEqual(_alpaca_dynamic_batch_cash_fraction(10), 0.50)
        self.assertEqual(_alpaca_dynamic_batch_cash_fraction(11), 0.50)

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
    def test_buy_is_deferred_while_market_is_open(self, mock_get: Mock, mock_post: Mock) -> None:
        mock_get.side_effect = [
            response(404, {}),
            response(200, []),
            response(404, {}),
            response(200, {"symbol": "TQQQ", "status": "active", "tradable": True}),
            response(200, {"is_open": True, "next_open": "2026-01-05T14:30:00Z"}),
        ]

        result = submit_alpaca_paper_buy_orders(
            pd.DataFrame([{"Asset": "TQQQ", "Date": "2026-01-02"}]),
            self.cfg(buy=True),
        )

        self.assertEqual(result.loc[0, "Status"], "deferred")
        self.assertIn("market is open", result.loc[0, "Message"])
        mock_post.assert_not_called()

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_buy_is_deferred_when_a_prior_session_signal_is_stale(self, mock_get: Mock, mock_post: Mock) -> None:
        mock_get.side_effect = [
            response(404, {}),
            response(200, []),
            response(404, {}),
            response(200, {"symbol": "TQQQ", "status": "active", "tradable": True}),
            response(
                200,
                {"is_open": False, "timestamp": "2026-01-05T13:30:00Z", "next_open": "2026-01-05T14:30:00Z"},
            ),
            response(200, [{"date": "2026-01-02"}, {"date": "2026-01-05"}]),
        ]

        result = submit_alpaca_paper_buy_orders(
            pd.DataFrame([{"Asset": "TQQQ", "Date": "2026-01-01"}]),
            self.cfg(buy=True),
        )

        self.assertEqual(result.loc[0, "Status"], "deferred")
        self.assertIn("immediately preceding", result.loc[0, "Message"])
        mock_post.assert_not_called()

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_buy_skips_when_symbol_is_short(self, mock_get: Mock, mock_post: Mock) -> None:
        mock_get.side_effect = [
            response(404, {}),
            response(200, []),
            response(200, {"symbol": "TQQQ", "qty": "-3"}),
        ]

        result = submit_alpaca_paper_buy_orders(
            pd.DataFrame([{"Asset": "TQQQ", "Date": "2026-01-02"}]),
            self.cfg(buy=True),
        )

        self.assertEqual(result.loc[0, "Status"], "held")
        self.assertIn("qty=-3.0", result.loc[0, "Message"])
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
            response(
                200,
                {"is_open": False, "timestamp": "2026-01-05T13:30:00Z", "next_open": "2026-01-05T14:30:00Z"},
            ),
            response(200, [{"date": "2026-01-02"}, {"date": "2026-01-05"}]),
            response(200, {"cash": "12345.67"}),
        ]
        mock_latest_market_price.return_value = 100.0
        mock_post.return_value = response(200, {"id": "order-3", "status": "accepted"})

        result = submit_alpaca_paper_buy_orders(
            pd.DataFrame([{"Asset": "ABC", "Date": "2026-01-02"}]),
            self.cfg(buy=True),
        )

        self.assertEqual(result.loc[0, "Status"], "submitted")
        self.assertEqual(result.loc[0, "Notional"], 525.0)
        self.assertEqual(result.loc[0, "Qty"], 5)
        self.assertEqual(mock_post.call_args.kwargs["json"]["qty"], "5")
        self.assertEqual(mock_post.call_args.kwargs["json"]["type"], "limit")

    @patch("leveraged_trader.alpaca._latest_market_price", return_value=25.0)
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_buy_batch_reserves_5_percent_cash_per_eligible_signal(
        self,
        mock_get: Mock,
        mock_post: Mock,
        _mock_latest_market_price: Mock,
    ) -> None:
        mock_get.side_effect = [
            response(404, {}),
            response(200, []),
            response(404, {}),
            response(200, {"symbol": "AAA", "status": "active", "tradable": True}),
            response(200, {"is_open": False, "timestamp": "2026-01-05T13:30:00Z", "next_open": "2026-01-05T14:30:00Z"}),
            response(200, [{"date": "2026-01-02"}, {"date": "2026-01-05"}]),
            response(404, {}),
            response(200, []),
            response(404, {}),
            response(200, {"symbol": "BBB", "status": "active", "tradable": True}),
            response(200, {"cash": "1000"}),
        ]
        mock_post.side_effect = [
            response(200, {"id": "order-1", "status": "accepted"}),
            response(200, {"id": "order-2", "status": "accepted"}),
        ]

        result = submit_alpaca_paper_buy_orders(
            pd.DataFrame([
                {"Asset": "AAA", "Date": "2026-01-02"},
                {"Asset": "BBB", "Date": "2026-01-02"},
            ]),
            self.cfg(buy=True),
        )

        self.assertEqual(result["Status"].tolist(), ["submitted", "submitted"])
        self.assertLessEqual(result["Notional"].sum(), 100.0)
        self.assertEqual([call.kwargs["json"]["type"] for call in mock_post.call_args_list], ["limit", "limit"])

    @patch("leveraged_trader.alpaca._latest_market_price", return_value=25.0)
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_preflight_rejections_do_not_dilute_eligible_buy_budget(
        self,
        mock_get: Mock,
        mock_post: Mock,
        _mock_latest_market_price: Mock,
    ) -> None:
        mock_get.side_effect = [
            response(404, {}),
            response(200, []),
            response(404, {}),
            response(200, {"symbol": "SKIP", "status": "inactive", "tradable": False}),
            response(404, {}),
            response(200, []),
            response(404, {}),
            response(200, {"symbol": "BUY", "status": "active", "tradable": True}),
            response(200, {"is_open": False, "timestamp": "2026-01-05T13:30:00Z", "next_open": "2026-01-05T14:30:00Z"}),
            response(200, [{"date": "2026-01-02"}, {"date": "2026-01-05"}]),
            response(200, {"cash": "1000"}),
        ]
        mock_post.return_value = response(200, {"id": "order-1", "status": "accepted"})

        result = submit_alpaca_paper_buy_orders(
            pd.DataFrame([
                {"Asset": "SKIP", "Date": "2026-01-02"},
                {"Asset": "BUY", "Date": "2026-01-02"},
            ]),
            self.cfg(buy=True),
        )

        self.assertEqual(result["Status"].tolist(), ["not_tradable", "submitted"])
        self.assertEqual(result.loc[1, "Qty"], 1)
        self.assertEqual(result.loc[1, "Notional"], 26.25)

    @patch("leveraged_trader.alpaca._latest_market_price", return_value=100.0)
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_duplicate_same_symbol_buy_signal_is_not_submitted_twice(
        self,
        mock_get: Mock,
        mock_post: Mock,
        _mock_latest_market_price: Mock,
    ) -> None:
        mock_get.side_effect = [
            response(404, {}),
            response(200, []),
            response(404, {}),
            response(200, {"symbol": "TQQQ", "status": "active", "tradable": True}),
            response(200, {"is_open": False, "timestamp": "2026-01-05T13:30:00Z", "next_open": "2026-01-05T14:30:00Z"}),
            response(200, [{"date": "2026-01-02"}, {"date": "2026-01-05"}]),
            response(200, {"cash": "10000"}),
        ]
        mock_post.return_value = response(200, {"id": "buy-1", "status": "accepted"})

        result = submit_alpaca_paper_buy_orders(
            pd.DataFrame(
                [
                    {"Asset": "TQQQ", "Date": "2026-01-02"},
                    {"Asset": "TQQQ", "Date": "2026-01-02"},
                ]
            ),
            self.cfg(buy=True),
        )

        self.assertEqual(result["Status"].tolist(), ["submitted", "duplicate_signal"])
        self.assertIn("duplicate buy signal", result.loc[1, "Message"])
        self.assertEqual(mock_post.call_count, 1)

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
            response(
                200,
                {"is_open": False, "timestamp": "2026-01-05T13:30:00Z", "next_open": "2026-01-05T14:30:00Z"},
            ),
            response(200, [{"date": "2026-01-02"}, {"date": "2026-01-05"}]),
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
            response(
                200,
                {"is_open": False, "timestamp": "2026-01-05T13:30:00Z", "next_open": "2026-01-05T14:30:00Z"},
            ),
            response(200, [{"date": "2026-01-02"}, {"date": "2026-01-05"}]),
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

    @patch("leveraged_trader.alpaca._latest_market_price", return_value=100.0)
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_ambiguous_buy_submission_persists_and_recovers_managed_intent(
        self,
        mock_get: Mock,
        mock_post: Mock,
        _mock_latest_price: Mock,
    ) -> None:
        mock_get.side_effect = [
            response(404, {}),
            response(200, []),
            response(404, {}),
            response(200, {"symbol": "TQQQ", "status": "active", "tradable": True}),
            response(200, {"is_open": False, "timestamp": "2026-01-05T13:30:00Z", "next_open": "2026-01-05T14:30:00Z"}),
            response(200, [{"date": "2026-01-02"}, {"date": "2026-01-05"}]),
            response(200, {"cash": "3000"}),
            response(200, {"id": "buy-1", "status": "accepted", "submitted_at": "2026-01-05T13:31:00Z"}),
        ]
        mock_post.side_effect = requests.Timeout("response lost after broker acceptance")

        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            result = submit_alpaca_paper_buy_orders(
                pd.DataFrame(
                    [{
                        "Asset": "TQQQ",
                        "RSI Symbol": "QQQ",
                        "Date": "2026-01-02",
                        "Buy RSI": 30.0,
                        "Sell Return Multiple": 1.5,
                    }]
                ),
                self.cfg(buy=True),
                conn=conn,
            )
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "existing")
        self.assertEqual(managed.loc[0, "buy_status"], "accepted")
        self.assertEqual(managed.loc[0, "buy_alpaca_order_id"], "buy-1")

    @patch("leveraged_trader.alpaca._latest_market_price", return_value=100.0)
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_unresolved_ambiguous_buy_submission_remains_active_during_visibility_lease(
        self,
        mock_get: Mock,
        mock_post: Mock,
        _mock_latest_price: Mock,
    ) -> None:
        mock_get.side_effect = [
            response(404, {}),
            response(200, []),
            response(404, {}),
            response(200, {"symbol": "TQQQ", "status": "active", "tradable": True}),
            response(200, {"is_open": False, "timestamp": "2026-01-05T13:30:00Z", "next_open": "2026-01-05T14:30:00Z"}),
            response(200, [{"date": "2026-01-02"}, {"date": "2026-01-05"}]),
            response(200, {"cash": "3000"}),
            response(404, {}),
        ]
        mock_post.side_effect = requests.Timeout("response lost")

        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            result = submit_alpaca_paper_buy_orders(
                pd.DataFrame(
                    [{
                        "Asset": "TQQQ",
                        "RSI Symbol": "QQQ",
                        "Date": "2026-01-02",
                        "Buy RSI": 30.0,
                        "Sell Return Multiple": 1.5,
                    }]
                ),
                self.cfg(buy=True),
                conn=conn,
            )
            self.assertEqual(result.loc[0, "Status"], "submission_unknown")

            with patch(
                "leveraged_trader.alpaca.requests.get",
                return_value=error_response(404, {"message": "order not found"}),
            ):
                reconciliation = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(reconciliation.loc[0, "Status"], "submission_unknown")
        self.assertTrue(pd.isna(managed.loc[0, "closed_at"]))
        self.assertIn("visibility lease", reconciliation.loc[0, "Message"])

    @patch("leveraged_trader.alpaca.requests.get")
    def test_pending_buy_intent_cannot_be_closed_by_another_workflow_during_its_lease(
        self,
        mock_get: Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite"
            with sqlite3.connect(db_path) as setup_conn:
                init_state_db(setup_conn)
            with sqlite3.connect(db_path) as owner_conn, sqlite3.connect(db_path) as observer_conn:
                position_id, claimed = claim_alpaca_managed_buy_intent(
                    owner_conn,
                    symbol="TQQQ",
                    signal_symbol="QQQ",
                    buy_rsi=30.0,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id="rsi-buy-TQQQ-20260102",
                )
                mock_get.return_value = error_response(404, {"message": "order not found"})
                reconciliation = reconcile_alpaca_managed_positions(
                    observer_conn,
                    self.cfg(buy=True, sell=True),
                )
                save_alpaca_managed_buy_order(
                    owner_conn,
                    symbol="TQQQ",
                    signal_symbol="QQQ",
                    buy_rsi=30.0,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id="rsi-buy-TQQQ-20260102",
                    buy_alpaca_order_id="buy-1",
                    buy_submitted_at="2026-01-02T14:30:00Z",
                    buy_status="accepted",
                )
                managed = load_alpaca_managed_positions(observer_conn)

        self.assertTrue(claimed)
        self.assertEqual(reconciliation.loc[0, "Status"], "submission_pending")
        self.assertEqual(managed.loc[0, "id"], position_id)
        self.assertEqual(managed.loc[0, "buy_status"], "accepted")
        self.assertEqual(managed.loc[0, "buy_alpaca_order_id"], "buy-1")
        self.assertTrue(pd.isna(managed.loc[0, "closed_at"]))

    @patch("leveraged_trader.alpaca.requests.get")
    def test_expired_buy_submission_lease_closes_a_missing_intent(self, mock_get: Mock) -> None:
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            position_id, _claimed = claim_alpaca_managed_buy_intent(
                conn,
                symbol="TQQQ",
                signal_symbol="QQQ",
                buy_rsi=30.0,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-20260102",
            )
            conn.execute(
                "UPDATE alpaca_managed_positions SET buy_submission_claimed_at = '2020-01-01 00:00:00' WHERE id = ?",
                (position_id,),
            )
            conn.commit()
            mock_get.return_value = error_response(404, {"message": "order not found"})

            reconciliation = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(reconciliation.loc[0, "Status"], "submission_not_found")
        self.assertFalse(pd.isna(managed.loc[0, "closed_at"]))

    @patch("leveraged_trader.alpaca._latest_market_price", return_value=100.0)
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_closed_missing_buy_intent_is_reclaimed_and_submitted_once(
        self,
        mock_get: Mock,
        mock_post: Mock,
        _mock_latest_price: Mock,
    ) -> None:
        mock_get.side_effect = [
            response(404, {}),
            response(200, []),
            response(404, {}),
            response(200, {"symbol": "TQQQ", "status": "active", "tradable": True}),
            response(200, {"is_open": False, "timestamp": "2026-01-05T13:30:00Z", "next_open": "2026-01-05T14:30:00Z"}),
            response(200, [{"date": "2026-01-02"}, {"date": "2026-01-05"}]),
            response(200, {"cash": "3000"}),
        ]
        mock_post.return_value = response(200, {"id": "buy-retry", "status": "accepted"})

        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            position_id, _claimed = claim_alpaca_managed_buy_intent(
                conn,
                symbol="TQQQ",
                signal_symbol="QQQ",
                buy_rsi=30.0,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-20260102",
            )
            conn.execute(
                """
                UPDATE alpaca_managed_positions
                SET buy_status = 'submission_not_found', closed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (position_id,),
            )
            conn.commit()

            result = submit_alpaca_paper_buy_orders(
                pd.DataFrame(
                    [{
                        "Asset": "TQQQ",
                        "RSI Symbol": "QQQ",
                        "Date": "2026-01-02",
                        "Buy RSI": 30.0,
                        "Sell Return Multiple": 1.5,
                    }]
                ),
                self.cfg(buy=True),
                conn=conn,
            )
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "submitted")
        self.assertEqual(mock_post.call_count, 1)
        self.assertEqual(managed.loc[0, "buy_status"], "accepted")
        self.assertEqual(managed.loc[0, "buy_alpaca_order_id"], "buy-retry")
        self.assertEqual(managed.loc[0, "buy_submission_attempt_count"], 2)
        self.assertTrue(pd.isna(managed.loc[0, "closed_at"]))

    def test_only_one_workflow_can_reclaim_a_closed_missing_buy_intent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite"
            with sqlite3.connect(db_path) as setup_conn:
                init_state_db(setup_conn)
            with sqlite3.connect(db_path) as first_conn, sqlite3.connect(db_path) as second_conn:
                position_id, _claimed = claim_alpaca_managed_buy_intent(
                    first_conn,
                    symbol="TQQQ",
                    signal_symbol="QQQ",
                    buy_rsi=30.0,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id="rsi-buy-TQQQ-20260102",
                )
                first_conn.execute(
                    """
                    UPDATE alpaca_managed_positions
                    SET buy_status = 'submission_not_found', closed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (position_id,),
                )
                first_conn.commit()

                first_position_id, first_claimed = claim_alpaca_managed_buy_intent(
                    first_conn,
                    symbol="TQQQ",
                    signal_symbol="QQQ",
                    buy_rsi=30.0,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id="rsi-buy-TQQQ-20260102",
                    allow_retry_after_not_found=True,
                )
                second_position_id, second_claimed = claim_alpaca_managed_buy_intent(
                    second_conn,
                    symbol="TQQQ",
                    signal_symbol="QQQ",
                    buy_rsi=30.0,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id="rsi-buy-TQQQ-20260102",
                    allow_retry_after_not_found=True,
                )
                managed = load_alpaca_managed_positions(second_conn)

        self.assertTrue(first_claimed)
        self.assertFalse(second_claimed)
        self.assertEqual(first_position_id, position_id)
        self.assertEqual(second_position_id, position_id)
        self.assertEqual(managed.loc[0, "buy_status"], "submission_pending")
        self.assertEqual(managed.loc[0, "buy_submission_attempt_count"], 2)
        self.assertTrue(pd.isna(managed.loc[0, "closed_at"]))

    def test_retry_claim_does_not_reopen_other_closed_buy_failures(self) -> None:
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            position_id, _claimed = claim_alpaca_managed_buy_intent(
                conn,
                symbol="TQQQ",
                signal_symbol="QQQ",
                buy_rsi=30.0,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-20260102",
            )
            conn.execute(
                """
                UPDATE alpaca_managed_positions
                SET buy_status = 'submission_failed', closed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (position_id,),
            )
            conn.commit()

            retried_position_id, retried = claim_alpaca_managed_buy_intent(
                conn,
                symbol="TQQQ",
                signal_symbol="QQQ",
                buy_rsi=30.0,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-20260102",
                allow_retry_after_not_found=True,
            )
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(retried_position_id, position_id)
        self.assertFalse(retried)
        self.assertEqual(managed.loc[0, "buy_status"], "submission_failed")
        self.assertEqual(managed.loc[0, "buy_submission_attempt_count"], 1)
        self.assertFalse(pd.isna(managed.loc[0, "closed_at"]))

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_visible_broker_order_wins_over_closed_missing_intent_retry(
        self,
        mock_get: Mock,
        mock_post: Mock,
    ) -> None:
        mock_get.return_value = response(200, {"id": "buy-visible", "status": "accepted"})

        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            position_id, _claimed = claim_alpaca_managed_buy_intent(
                conn,
                symbol="TQQQ",
                signal_symbol="QQQ",
                buy_rsi=30.0,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-20260102",
            )
            conn.execute(
                """
                UPDATE alpaca_managed_positions
                SET buy_status = 'submission_not_found', closed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (position_id,),
            )
            conn.commit()

            result = submit_alpaca_paper_buy_orders(
                pd.DataFrame(
                    [{
                        "Asset": "TQQQ",
                        "RSI Symbol": "QQQ",
                        "Date": "2026-01-02",
                        "Buy RSI": 30.0,
                        "Sell Return Multiple": 1.5,
                    }]
                ),
                self.cfg(buy=True),
                conn=conn,
            )
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "existing")
        self.assertEqual(managed.loc[0, "buy_status"], "accepted")
        self.assertEqual(managed.loc[0, "buy_alpaca_order_id"], "buy-visible")
        self.assertEqual(managed.loc[0, "buy_submission_attempt_count"], 1)
        self.assertTrue(pd.isna(managed.loc[0, "closed_at"]))
        mock_post.assert_not_called()

    def test_broker_observation_reopens_a_stale_closed_buy_intent(self) -> None:
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            position_id, _claimed = claim_alpaca_managed_buy_intent(
                conn,
                symbol="TQQQ",
                signal_symbol="QQQ",
                buy_rsi=30.0,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-20260102",
            )
            conn.execute(
                "UPDATE alpaca_managed_positions SET closed_at = CURRENT_TIMESTAMP WHERE id = ?",
                (position_id,),
            )
            conn.commit()
            save_alpaca_managed_buy_order(
                conn,
                symbol="TQQQ",
                signal_symbol="QQQ",
                buy_rsi=30.0,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-20260102",
                buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-01-02T14:30:00Z",
                buy_status="accepted",
            )
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(managed.loc[0, "buy_status"], "accepted")
        self.assertEqual(managed.loc[0, "buy_alpaca_order_id"], "buy-1")
        self.assertTrue(pd.isna(managed.loc[0, "closed_at"]))

    @patch("leveraged_trader.alpaca.requests.post")
    def test_limit_order_price_precision_uses_alpaca_ticks_and_side_safe_rounding(
        self,
        mock_post: Mock,
    ) -> None:
        client = AlpacaClient(self.cfg(buy=True, sell=True))

        client.submit_limit_buy_order(
            symbol="PENNY",
            qty=1,
            limit_price=0.12349,
            client_order_id="buy-penny",
        )
        client.submit_limit_sell_order(
            symbol="PENNY",
            qty=1,
            limit_price=0.12341,
            client_order_id="sell-penny",
        )
        client.submit_limit_buy_order(
            symbol="BOUNDARY",
            qty=1,
            limit_price=0.9999,
            client_order_id="buy-boundary",
        )
        client.submit_limit_sell_order(
            symbol="DOLLAR",
            qty=1,
            limit_price=1.001,
            client_order_id="sell-dollar",
        )

        self.assertEqual(mock_post.call_args_list[0].kwargs["json"]["limit_price"], "0.1234")
        self.assertEqual(mock_post.call_args_list[1].kwargs["json"]["limit_price"], "0.1235")
        self.assertEqual(mock_post.call_args_list[2].kwargs["json"]["limit_price"], "0.9999")
        self.assertEqual(mock_post.call_args_list[3].kwargs["json"]["limit_price"], "1.01")

    @patch("leveraged_trader.alpaca._latest_market_price", return_value=100.0)
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_duplicate_client_order_error_keeps_managed_intent_active_for_recovery(
        self,
        mock_get: Mock,
        mock_post: Mock,
        _mock_latest_price: Mock,
    ) -> None:
        mock_get.side_effect = [
            response(404, {}),
            response(200, []),
            response(404, {}),
            response(200, {"symbol": "TQQQ", "status": "active", "tradable": True}),
            response(200, {"is_open": False, "timestamp": "2026-01-05T13:30:00Z", "next_open": "2026-01-05T14:30:00Z"}),
            response(200, [{"date": "2026-01-02"}, {"date": "2026-01-05"}]),
            response(200, {"cash": "3000"}),
            response(404, {}),
        ]
        mock_post.return_value = error_response(422, {"message": "client_order_id already exists"})

        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            result = submit_alpaca_paper_buy_orders(
                pd.DataFrame(
                    [{
                        "Asset": "TQQQ",
                        "RSI Symbol": "QQQ",
                        "Date": "2026-01-02",
                        "Buy RSI": 30.0,
                        "Sell Return Multiple": 1.5,
                    }]
                ),
                self.cfg(buy=True),
                conn=conn,
            )
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "submission_unknown")
        self.assertEqual(managed.loc[0, "buy_status"], "submission_unknown")
        self.assertTrue(pd.isna(managed.loc[0, "closed_at"]))

    @patch("leveraged_trader.alpaca.requests.get")
    def test_zero_fill_stopped_or_suspended_buy_closes_managed_position(self, mock_get: Mock) -> None:
        for buy_status in ["stopped", "suspended"]:
            with self.subTest(buy_status=buy_status), sqlite3.connect(":memory:") as conn:
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
                mock_get.reset_mock()
                mock_get.return_value = response(
                    200,
                    {
                        "id": "buy-1",
                        "status": buy_status,
                        "submitted_at": "2026-01-02T14:30:00Z",
                        "updated_at": "2026-01-02T14:35:00Z",
                        "filled_qty": "0",
                    },
                )

                result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
                managed = load_alpaca_managed_positions(conn)

                self.assertEqual(result.loc[0, "Status"], buy_status)
                self.assertEqual(managed.loc[0, "buy_status"], buy_status)
                self.assertFalse(pd.isna(managed.loc[0, "closed_at"]))

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
    def test_subdollar_managed_target_preserves_the_valid_profit_tick(
        self,
        mock_get: Mock,
        mock_post: Mock,
    ) -> None:
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            save_alpaca_managed_buy_order(
                conn,
                symbol="PENNY",
                signal_symbol="PENNY",
                buy_rsi=30,
                profit_target_multiple=1.001,
                buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-PENNY-20260102",
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
                        "filled_qty": "2",
                        "filled_avg_price": "0.1234",
                    },
                ),
                response(200, []),
            ]
            mock_post.return_value = response(200, {"id": "sell-1", "status": "accepted"})

            reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(managed.loc[0, "target_sell_price"], 0.1236)
        self.assertEqual(mock_post.call_args.kwargs["json"]["limit_price"], "0.1236")

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_partial_managed_buy_creates_target_sell_for_confirmed_shares(
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
                        "status": "partially_filled",
                        "submitted_at": "2026-01-02T14:30:00Z",
                        "filled_qty": "2",
                        "filled_avg_price": "100.00",
                    },
                ),
                response(200, []),
            ]
            mock_post.return_value = response(200, {"id": "sell-1", "status": "accepted"})

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result["Status"].tolist(), ["partially_filled", "accepted"])
        self.assertEqual(managed.loc[0, "sell_status"], "accepted")
        self.assertEqual(mock_post.call_args.kwargs["json"]["qty"], "2")

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_terminal_buy_with_partial_fill_still_creates_protective_sell(
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
                        "status": "stopped",
                        "submitted_at": "2026-01-02T14:30:00Z",
                        "filled_qty": "2",
                        "filled_avg_price": "100.00",
                    },
                ),
                response(200, []),
            ]
            mock_post.return_value = response(200, {"id": "sell-1", "status": "accepted"})

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result["Status"].tolist(), ["stopped", "accepted"])
        self.assertEqual(managed.loc[0, "filled_qty"], 2.0)
        self.assertEqual(managed.loc[0, "sell_status"], "accepted")
        self.assertTrue(pd.isna(managed.loc[0, "closed_at"]))

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.delete")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_partial_managed_buy_replaces_sell_when_more_shares_fill(
        self,
        mock_get: Mock,
        mock_delete: Mock,
        mock_post: Mock,
    ) -> None:
        """A still-open parent buy keeps its protective sell sized to all fills."""
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
                        "status": "partially_filled",
                        "filled_qty": "2",
                        "filled_avg_price": "100",
                    },
                ),
                response(200, []),
                response(
                    200,
                    {
                        "id": "buy-1",
                        "status": "partially_filled",
                        "filled_qty": "3",
                        "filled_avg_price": "101",
                    },
                ),
                response(
                    200,
                    {
                        "id": "sell-1",
                        "status": "accepted",
                        "qty": "2",
                        "filled_qty": "0",
                        "limit_price": "150",
                    },
                ),
                response(
                    200,
                    {
                        "id": "sell-1",
                        "status": "canceled",
                        "qty": "2",
                        "filled_qty": "0",
                        "limit_price": "150",
                    },
                ),
                response(200, []),
            ]
            mock_delete.return_value = response(200, {})
            mock_post.side_effect = [
                response(200, {"id": "sell-1", "status": "accepted"}),
                response(200, {"id": "sell-2", "status": "accepted"}),
            ]

            reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result["Status"].tolist(), ["partially_filled", "renewed"])
        self.assertEqual(mock_post.call_args_list[0].kwargs["json"]["qty"], "2")
        self.assertEqual(mock_post.call_args_list[1].kwargs["json"]["qty"], "3")
        self.assertEqual(mock_post.call_args_list[1].kwargs["json"]["limit_price"], "151.5")
        self.assertEqual(managed.loc[0, "target_sell_price"], 151.5)
        self.assertEqual(managed.loc[0, "sell_alpaca_order_id"], "sell-2")
        mock_delete.assert_called_once()

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_later_parent_fill_is_protected_after_prior_partial_exit_filled(
        self,
        mock_get: Mock,
        mock_post: Mock,
    ) -> None:
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            position_id = save_alpaca_managed_buy_order(
                conn,
                symbol="TQQQ",
                signal_symbol="QQQ",
                buy_rsi=30,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-20260102",
                buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-01-02T14:30:00Z",
                buy_status="partially_filled",
            )
            mark_alpaca_managed_buy_filled(
                conn,
                position_id,
                buy_status="partially_filled",
                filled_qty=2.0,
                filled_avg_price=100.0,
                filled_at=None,
                target_sell_price=150.0,
            )
            record_alpaca_managed_sell_order(
                conn,
                position_id,
                sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id="sell-1",
                sell_submitted_at=None,
                sell_status="filled",
            )
            mark_alpaca_managed_sell_filled(
                conn,
                position_id,
                sell_status="filled",
                sell_filled_qty=2.0,
                sell_filled_avg_price=150.0,
                sell_filled_at=None,
                sell_alpaca_order_id="sell-1",
            )
            mock_get.side_effect = [
                response(
                    200,
                    {
                        "id": "buy-1",
                        "status": "partially_filled",
                        "filled_qty": "3",
                        "filled_avg_price": "100",
                    },
                ),
                response(
                    200,
                    {
                        "id": "sell-1",
                        "status": "filled",
                        "filled_qty": "2",
                        "filled_avg_price": "150",
                    },
                ),
                response(200, []),
            ]
            mock_post.return_value = response(200, {"id": "sell-2", "status": "accepted"})

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result["Status"].tolist(), ["partially_filled", "renewed"])
        self.assertEqual(mock_post.call_args.kwargs["json"]["qty"], "1")
        self.assertEqual(managed.loc[0, "remaining_qty"], 1.0)
        self.assertEqual(managed.loc[0, "sell_alpaca_order_id"], "sell-2")
        self.assertTrue(pd.isna(managed.loc[0, "closed_at"]))

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_terminal_parent_fill_is_protected_after_prior_partial_exit_filled(
        self,
        mock_get: Mock,
        mock_post: Mock,
    ) -> None:
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            position_id = save_alpaca_managed_buy_order(
                conn,
                symbol="TQQQ",
                signal_symbol="QQQ",
                buy_rsi=30,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-20260102",
                buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-01-02T14:30:00Z",
                buy_status="partially_filled",
            )
            mark_alpaca_managed_buy_filled(
                conn,
                position_id,
                buy_status="partially_filled",
                filled_qty=2.0,
                filled_avg_price=100.0,
                filled_at=None,
                target_sell_price=150.0,
            )
            record_alpaca_managed_sell_order(
                conn,
                position_id,
                sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id="sell-1",
                sell_submitted_at=None,
                sell_status="filled",
            )
            mark_alpaca_managed_sell_filled(
                conn,
                position_id,
                sell_status="filled",
                sell_filled_qty=2.0,
                sell_filled_avg_price=150.0,
                sell_filled_at=None,
                sell_alpaca_order_id="sell-1",
            )
            mock_get.side_effect = [
                response(
                    200,
                    {
                        "id": "buy-1",
                        "status": "filled",
                        "filled_qty": "3",
                        "filled_avg_price": "100",
                    },
                ),
                response(
                    200,
                    {
                        "id": "sell-1",
                        "status": "filled",
                        "filled_qty": "2",
                        "filled_avg_price": "150",
                    },
                ),
                response(200, []),
            ]
            mock_post.return_value = response(200, {"id": "sell-2", "status": "accepted"})

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result["Status"].tolist(), ["filled", "renewed"])
        self.assertEqual(mock_post.call_args.kwargs["json"]["qty"], "1")
        self.assertEqual(managed.loc[0, "remaining_qty"], 1.0)
        self.assertEqual(managed.loc[0, "sell_alpaca_order_id"], "sell-2")
        self.assertTrue(pd.isna(managed.loc[0, "closed_at"]))

    def test_competing_buy_intent_cannot_downgrade_or_close_accepted_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite"
            with sqlite3.connect(db_path) as first_conn:
                init_state_db(first_conn)
            with sqlite3.connect(db_path) as first_conn, sqlite3.connect(db_path) as second_conn:
                position_id, first_claimed = claim_alpaca_managed_buy_intent(
                    first_conn,
                    symbol="TQQQ",
                    signal_symbol="QQQ",
                    buy_rsi=30.0,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id="rsi-buy-TQQQ-20260102",
                )
                second_position_id, second_claimed = claim_alpaca_managed_buy_intent(
                    second_conn,
                    symbol="TQQQ",
                    signal_symbol="QQQ",
                    buy_rsi=30.0,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id="rsi-buy-TQQQ-20260102",
                )
                save_alpaca_managed_buy_order(
                    first_conn,
                    symbol="TQQQ",
                    signal_symbol="QQQ",
                    buy_rsi=30.0,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id="rsi-buy-TQQQ-20260102",
                    buy_alpaca_order_id="buy-1",
                    buy_submitted_at="2026-01-02T14:30:00Z",
                    buy_status="accepted",
                )
                was_closed = fail_alpaca_managed_buy_submission_if_pending(
                    second_conn,
                    second_position_id,
                    notes="duplicate client order ID",
                )
                managed = load_alpaca_managed_positions(second_conn)

        self.assertTrue(first_claimed)
        self.assertFalse(second_claimed)
        self.assertEqual(position_id, second_position_id)
        self.assertFalse(was_closed)
        self.assertEqual(managed.loc[0, "buy_status"], "accepted")
        self.assertTrue(pd.isna(managed.loc[0, "closed_at"]))

    @patch("leveraged_trader.alpaca._latest_market_price", return_value=100.0)
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    @patch("leveraged_trader.alpaca.active_alpaca_managed_symbols", return_value=set())
    def test_competing_intent_never_submits_a_second_buy_order(
        self,
        _mock_managed_symbols: Mock,
        mock_get: Mock,
        mock_post: Mock,
        _mock_latest_price: Mock,
    ) -> None:
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            claim_alpaca_managed_buy_intent(
                conn,
                symbol="TQQQ",
                signal_symbol="QQQ",
                buy_rsi=30.0,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-20260102",
            )
            mock_get.side_effect = [
                response(404, {}),
                response(200, []),
                response(404, {}),
                response(200, {"symbol": "TQQQ", "status": "active", "tradable": True}),
                response(
                    200,
                    {"is_open": False, "timestamp": "2026-01-05T13:30:00Z", "next_open": "2026-01-05T14:30:00Z"},
                ),
                response(200, [{"date": "2026-01-02"}, {"date": "2026-01-05"}]),
                response(200, {"cash": "3000"}),
                response(404, {}),
            ]

            result = submit_alpaca_paper_buy_orders(
                pd.DataFrame(
                    [{
                        "Asset": "TQQQ",
                        "RSI Symbol": "QQQ",
                        "Date": "2026-01-02",
                        "Buy RSI": 30.0,
                        "Sell Return Multiple": 1.5,
                    }]
                ),
                self.cfg(buy=True),
                conn=conn,
            )
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "submission_pending")
        self.assertEqual(managed.loc[0, "buy_status"], "submission_pending")
        mock_post.assert_not_called()

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
                    "filled_avg_price": "151.25",
                },
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)
            active_symbols = active_alpaca_managed_symbols(conn)

        self.assertEqual(result.loc[0, "Status"], "filled")
        self.assertIsNotNone(managed.loc[0, "closed_at"])
        self.assertEqual(managed.loc[0, "sell_filled_qty"], 2.5)
        self.assertEqual(managed.loc[0, "sell_filled_avg_price"], 151.25)
        self.assertEqual(managed.loc[0, "realized_pl"], 128.125)
        self.assertAlmostEqual(managed.loc[0, "realized_pl_pct"], 51.25)
        self.assertNotIn("TQQQ", active_symbols)

    @patch("leveraged_trader.alpaca.requests.get")
    def test_mismatched_filled_sell_qty_remains_active_for_review(self, mock_get: Mock) -> None:
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
                filled_qty=2.0,
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
                    "status": "filled",
                    "submitted_at": "2026-01-02T14:32:00Z",
                    "filled_at": "2026-01-02T14:33:00Z",
                    "filled_qty": "1",
                    "filled_avg_price": "150",
                },
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "filled")
        self.assertIn("remains active", result.loc[0, "Message"])
        self.assertTrue(pd.isna(managed.loc[0, "closed_at"]))
        self.assertEqual(managed.loc[0, "sell_status"], "quantity_mismatch")
        self.assertEqual(managed.loc[0, "sold_qty"], 1.0)
        self.assertEqual(managed.loc[0, "remaining_qty"], 1.0)
        self.assertEqual(managed.loc[0, "realized_pl"], 50.0)

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_immediate_partial_managed_sell_reports_active_for_review(
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
                buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn,
                1,
                buy_status="filled",
                filled_qty=2.0,
                filled_avg_price=100.0,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150.0,
            )
            mock_get.return_value = response(200, [])
            mock_post.return_value = response(
                200,
                {
                    "id": "sell-1",
                    "status": "filled",
                    "filled_at": "2026-01-02T14:33:00Z",
                    "filled_qty": "1",
                    "filled_avg_price": "150",
                },
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.iloc[-1]["Status"], "filled")
        self.assertIn("remains active", result.iloc[-1]["Message"])
        self.assertTrue(pd.isna(managed.loc[0, "closed_at"]))
        self.assertEqual(managed.loc[0, "sell_status"], "quantity_mismatch")
        self.assertEqual(managed.loc[0, "remaining_qty"], 1.0)

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_immediate_filled_sell_with_nonfinite_metadata_preserves_accounting(
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
                buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn,
                1,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
            )
            mock_get.return_value = response(200, [])
            mock_post.return_value = response(
                200,
                {
                    "id": "sell-1",
                    "status": "filled",
                    "filled_qty": "nan",
                    "filled_avg_price": "inf",
                },
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.iloc[-1]["Status"], "incomplete_fill_metadata")
        self.assertEqual(managed.loc[0, "sell_status"], "incomplete_fill_metadata")
        self.assertEqual(managed.loc[0, "sold_qty"], 0)
        self.assertEqual(managed.loc[0, "remaining_qty"], 2)
        self.assertTrue(pd.isna(managed.loc[0, "realized_pl"]))

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_immediate_filled_sell_without_broker_id_preserves_accounting(
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
                buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn,
                1,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
            )
            mock_get.return_value = response(200, [])
            mock_post.return_value = response(
                200,
                {
                    "status": "filled",
                    "filled_qty": "2",
                    "filled_avg_price": "150",
                },
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.iloc[-1]["Status"], "incomplete_order_metadata")
        self.assertEqual(managed.loc[0, "sell_status"], "incomplete_order_metadata")
        self.assertEqual(managed.loc[0, "sold_qty"], 0)
        self.assertEqual(managed.loc[0, "remaining_qty"], 2)
        self.assertTrue(pd.isna(managed.loc[0, "closed_at"]))

    @patch("leveraged_trader.alpaca.requests.get")
    def test_polled_sell_fill_without_broker_id_preserves_accounting(
        self,
        mock_get: Mock,
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
                buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn,
                1,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
            )
            record_alpaca_managed_sell_order(
                conn,
                1,
                sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id=None,
                sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="accepted",
            )
            mock_get.return_value = response(
                200,
                {
                    "status": "filled",
                    "filled_qty": "2",
                    "filled_avg_price": "150",
                },
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "incomplete_order_metadata")
        self.assertEqual(managed.loc[0, "sell_status"], "incomplete_order_metadata")
        self.assertEqual(managed.loc[0, "sold_qty"], 0)
        self.assertEqual(managed.loc[0, "remaining_qty"], 2)
        self.assertTrue(pd.isna(managed.loc[0, "closed_at"]))

    @patch("leveraged_trader.alpaca.requests.get")
    def test_conflicting_polled_sell_identity_preserves_managed_generation(
        self,
        mock_get: Mock,
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
                buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn,
                1,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
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
                    "id": "different-sell",
                    "client_order_id": "different-generation",
                    "status": "filled",
                    "filled_qty": "2",
                    "filled_avg_price": "150",
                },
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "incomplete_order_metadata")
        self.assertEqual(managed.loc[0, "sell_status"], "incomplete_order_metadata")
        self.assertEqual(managed.loc[0, "sell_alpaca_order_id"], "sell-1")
        self.assertEqual(managed.loc[0, "sell_client_order_id"], "rsi-exit-TQQQ-1")
        self.assertEqual(managed.loc[0, "sold_qty"], 0)
        self.assertTrue(pd.isna(managed.loc[0, "closed_at"]))

    @patch("leveraged_trader.alpaca.requests.get")
    def test_submission_recovery_fill_regression_preserves_accounting_for_review(
        self,
        mock_get: Mock,
    ) -> None:
        rows: list[dict] = []
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
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
            )
            record_alpaca_managed_sell_order(
                conn,
                1,
                sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id="sell-1",
                sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="partially_filled",
            )
            mark_alpaca_managed_sell_filled(
                conn,
                1,
                sell_status="partially_filled",
                sell_filled_qty=1,
                sell_filled_avg_price=150,
                sell_filled_at="2026-01-02T14:33:00Z",
                sell_alpaca_order_id="sell-1",
            )
            mock_get.return_value = response(
                200,
                {
                    "id": "sell-1",
                    "status": "filled",
                    "filled_qty": "0.5",
                    "filled_avg_price": "150",
                },
            )

            recovered = _recover_managed_sell_submission(
                conn=conn,
                client=AlpacaClient(self.cfg(sell=True)),
                rows=rows,
                position_id=1,
                symbol="TQQQ",
                buy_client_order_id="rsi-buy-TQQQ-20260102",
                sell_client_order_id="rsi-exit-TQQQ-1",
                filled_qty=2,
                target_sell_price=150,
                message="submission response lost",
                close_on_complete=True,
            )
            managed = load_alpaca_managed_positions(conn)

        self.assertTrue(recovered)
        self.assertEqual(rows[0]["Status"], "fill_quantity_regression")
        self.assertEqual(managed.loc[0, "sell_status"], "fill_quantity_regression")
        self.assertEqual(managed.loc[0, "sold_qty"], 1)
        self.assertEqual(managed.loc[0, "remaining_qty"], 1)
        self.assertEqual(managed.loc[0, "realized_pl"], 50)

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_ambiguous_managed_sell_submission_persists_intent_for_recovery(
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
                buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn,
                1,
                buy_status="filled",
                filled_qty=2.0,
                filled_avg_price=100.0,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150.0,
            )
            mock_get.side_effect = [
                response(200, []),
                response(404, {}),
                response(
                    200,
                    {
                        "id": "sell-1",
                        "client_order_id": "rsi-exit-TQQQ-1",
                        "status": "accepted",
                        "submitted_at": "2026-01-02T14:32:00Z",
                        "qty": "2",
                        "limit_price": "150",
                        "expires_at": "2027-06-30T20:15:00Z",
                    },
                ),
            ]
            mock_post.side_effect = requests.exceptions.Timeout("accepted but response lost")

            first_result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            after_timeout = load_alpaca_managed_positions(conn)
            mock_post.reset_mock()
            second_result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            recovered = load_alpaca_managed_positions(conn)

        self.assertEqual(first_result.loc[0, "Status"], "submission_unknown")
        self.assertEqual(after_timeout.loc[0, "sell_client_order_id"], "rsi-exit-TQQQ-1")
        self.assertEqual(after_timeout.loc[0, "sell_status"], "submission_unknown")
        self.assertEqual(second_result.loc[0, "Status"], "accepted")
        self.assertEqual(recovered.loc[0, "sell_alpaca_order_id"], "sell-1")
        self.assertEqual(recovered.loc[0, "sell_status"], "accepted")
        mock_post.assert_not_called()

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_open_managed_sell_order_is_recovered_when_local_intent_is_missing(
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
                buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn,
                1,
                buy_status="filled",
                filled_qty=2.0,
                filled_avg_price=100.0,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150.0,
            )
            mock_get.return_value = response(
                200,
                [
                    {
                        "id": "sell-1",
                        "symbol": "TQQQ",
                        "side": "sell",
                        "client_order_id": "rsi-exit-TQQQ-1",
                        "status": "accepted",
                        "submitted_at": "2026-01-02T14:32:00Z",
                    }
                ],
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "accepted")
        self.assertIn("recovered", result.loc[0, "Message"])
        self.assertEqual(managed.loc[0, "sell_client_order_id"], "rsi-exit-TQQQ-1")
        self.assertEqual(managed.loc[0, "sell_alpaca_order_id"], "sell-1")
        mock_post.assert_not_called()

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_open_sell_recovery_without_broker_id_is_blocked_for_review(
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
                buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn,
                1,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
            )
            mock_get.return_value = response(
                200,
                [
                    {
                        "symbol": "TQQQ",
                        "side": "sell",
                        "client_order_id": "rsi-exit-TQQQ-1",
                        "status": "accepted",
                    }
                ],
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "incomplete_order_metadata")
        self.assertEqual(managed.loc[0, "sell_status"], "incomplete_order_metadata")
        self.assertTrue(pd.isna(managed.loc[0, "sell_client_order_id"]))
        self.assertTrue(pd.isna(managed.loc[0, "sell_alpaca_order_id"]))
        mock_post.assert_not_called()

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_open_sell_recovery_accounts_fill_before_intent_mismatch(
        self, mock_get: Mock, mock_post: Mock,
    ) -> None:
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            save_alpaca_managed_buy_order(
                conn, symbol="TQQQ", signal_symbol="QQQ", buy_rsi=30,
                profit_target_multiple=1.5, buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-20260102", buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-01-02T14:30:00Z", buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn, 1, buy_status="filled", filled_qty=2, filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z", target_sell_price=150,
            )
            mock_get.return_value = response(
                200,
                [
                    {
                        "id": "sell-1", "symbol": "TQQQ", "side": "sell",
                        "client_order_id": "rsi-exit-TQQQ-1", "status": "partially_filled",
                        "qty": "3", "limit_price": "149", "filled_qty": "1",
                        "filled_avg_price": "149",
                    }
                ],
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "quantity_mismatch")
        self.assertEqual(managed.loc[0, "sell_status"], "quantity_mismatch")
        self.assertEqual(managed.loc[0, "sell_alpaca_order_id"], "sell-1")
        self.assertEqual(managed.loc[0, "sold_qty"], 1)
        self.assertEqual(managed.loc[0, "remaining_qty"], 1)
        self.assertEqual(managed.loc[0, "realized_pl"], 49)
        mock_post.assert_not_called()

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_expired_partial_sell_without_average_price_blocks_renewal(
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
                buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn,
                1,
                buy_status="filled",
                filled_qty=2.0,
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
                    "filled_qty": "1",
                },
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "incomplete_fill_metadata")
        self.assertIn("renewal is blocked", result.loc[0, "Message"])
        self.assertEqual(managed.loc[0, "sell_status"], "incomplete_fill_metadata")
        self.assertTrue(pd.isna(managed.loc[0, "closed_at"]))
        mock_post.assert_not_called()

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_invalid_numeric_sell_fill_metadata_is_blocked_without_accounting_update(
        self, mock_get: Mock, mock_post: Mock,
    ) -> None:
        cases = [
            ("-1", "150"),
            ("not-a-quantity", "150"),
            ("NaN", "150"),
            ("1", "Infinity"),
        ]
        for filled_qty, filled_avg_price in cases:
            with self.subTest(filled_qty=filled_qty, filled_avg_price=filled_avg_price), sqlite3.connect(
                ":memory:"
            ) as conn:
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
                    filled_avg_price=100,
                    filled_at="2026-01-02T14:31:00Z",
                    target_sell_price=150,
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
                        "status": "partially_filled",
                        "qty": "2",
                        "limit_price": "150",
                        "filled_qty": filled_qty,
                        "filled_avg_price": filled_avg_price,
                    },
                )

                result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
                managed = load_alpaca_managed_positions(conn)

            self.assertEqual(result.loc[0, "Status"], "incomplete_fill_metadata")
            self.assertEqual(managed.loc[0, "sell_status"], "incomplete_fill_metadata")
            self.assertEqual(managed.loc[0, "sold_qty"], 0)
            self.assertEqual(managed.loc[0, "remaining_qty"], 2)
            mock_post.assert_not_called()

    @patch("leveraged_trader.alpaca.requests.get")
    def test_cumulative_sell_fill_regression_preserves_accounting_for_review(self, mock_get: Mock) -> None:
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            save_alpaca_managed_buy_order(
                conn, symbol="TQQQ", signal_symbol="QQQ", buy_rsi=30,
                profit_target_multiple=1.5, buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-20260102", buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-01-02T14:30:00Z", buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn, 1, buy_status="filled", filled_qty=2, filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z", target_sell_price=150,
            )
            record_alpaca_managed_sell_order(
                conn, 1, sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id="sell-1", sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="partially_filled", sell_order_qty=2,
            )
            mark_alpaca_managed_sell_filled(
                conn, 1, sell_status="partially_filled", sell_filled_qty=1,
                sell_filled_avg_price=150, sell_filled_at="2026-01-02T14:33:00Z",
                sell_alpaca_order_id="sell-1",
            )
            mock_get.return_value = response(
                200,
                {
                    "id": "sell-1", "status": "partially_filled", "qty": "2",
                    "limit_price": "150", "filled_qty": "0.5", "filled_avg_price": "150",
                },
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "fill_quantity_regression")
        self.assertEqual(managed.loc[0, "sell_status"], "fill_quantity_regression")
        self.assertEqual(managed.loc[0, "sold_qty"], 1)
        self.assertEqual(managed.loc[0, "remaining_qty"], 1)
        self.assertEqual(managed.loc[0, "realized_pl"], 50)

    @patch("leveraged_trader.alpaca.requests.get")
    def test_overfilled_managed_sell_remains_active_for_review(self, mock_get: Mock) -> None:
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
                filled_qty=2.0,
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
                    "status": "filled",
                    "filled_qty": "3",
                    "filled_avg_price": "150",
                },
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "quantity_mismatch")
        self.assertIn("exceeds", result.loc[0, "Message"])
        self.assertEqual(managed.loc[0, "remaining_qty"], -1.0)
        self.assertTrue(pd.isna(managed.loc[0, "closed_at"]))

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_expired_partially_filled_sell_replaces_only_remaining_qty(
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
                buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn,
                1,
                buy_status="filled",
                filled_qty=2.0,
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
            mock_get.side_effect = [
                response(
                    200,
                    {
                        "id": "sell-1",
                        "status": "expired",
                        "submitted_at": "2026-01-02T14:32:00Z",
                        "filled_qty": "1",
                        "filled_avg_price": "150",
                    },
                ),
                response(200, []),
            ]
            mock_post.return_value = response(200, {"id": "sell-2", "status": "accepted"})

            reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(mock_post.call_args.kwargs["json"]["qty"], "1")
        self.assertEqual(managed.loc[0, "sold_qty"], 1.0)
        self.assertEqual(managed.loc[0, "remaining_qty"], 1.0)
        self.assertTrue(pd.isna(managed.loc[0, "closed_at"]))

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

    @patch("leveraged_trader.alpaca.requests.delete")
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_expired_managed_sell_is_resubmitted(
        self,
        mock_get: Mock,
        mock_post: Mock,
        mock_delete: Mock,
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
            mock_get.side_effect = [
                response(
                    200,
                    {
                        "id": "sell-1",
                        "status": "expired",
                        "submitted_at": "2026-01-02T14:32:00Z",
                    },
                ),
                response(200, []),
            ]
            mock_post.return_value = response(
                200,
                {
                    "id": "sell-2",
                    "status": "accepted",
                    "submitted_at": "2026-04-01T14:32:00Z",
                    "expires_at": "2027-06-30T20:15:00Z",
                },
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)
            active_symbols = active_alpaca_managed_symbols(conn)

        self.assertEqual(result.loc[0, "Status"], "renewed")
        self.assertIn("expired", result.loc[0, "Message"])
        self.assertIn("TQQQ", active_symbols)
        self.assertEqual(managed.loc[0, "sell_client_order_id"], "rsi-exit-TQQQ-1-r1")
        self.assertEqual(managed.loc[0, "sell_alpaca_order_id"], "sell-2")
        self.assertEqual(managed.loc[0, "sell_renewal_count"], 1)
        self.assertEqual(managed.loc[0, "sell_expires_at"], "2027-06-30T20:15:00Z")
        self.assertEqual(mock_post.call_args.kwargs["json"]["client_order_id"], "rsi-exit-TQQQ-1-r1")
        self.assertEqual(mock_post.call_args.kwargs["json"]["limit_price"], "150")
        mock_delete.assert_not_called()

    @patch("leveraged_trader.alpaca._utc_now")
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_replacement_expiring_inside_renewal_window_is_blocked_for_review(
        self, mock_get: Mock, mock_post: Mock, mock_utc_now: Mock,
    ) -> None:
        mock_utc_now.return_value = datetime(2026, 4, 1, tzinfo=UTC)
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            save_alpaca_managed_buy_order(
                conn, symbol="TQQQ", signal_symbol="QQQ", buy_rsi=30,
                profit_target_multiple=1.5, buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-20260102", buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-01-02T14:30:00Z", buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn, 1, buy_status="filled", filled_qty=2, filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z", target_sell_price=150,
            )
            record_alpaca_managed_sell_order(
                conn, 1, sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id="sell-1", sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="expired",
            )
            mock_get.side_effect = [
                response(200, {"id": "sell-1", "status": "expired"}),
                response(200, []),
            ]
            mock_post.return_value = response(
                200,
                {
                    "id": "sell-2", "client_order_id": "rsi-exit-TQQQ-1-r1",
                    "status": "accepted", "expires_at": "2026-04-05T20:15:00Z",
                },
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "incomplete_order_metadata")
        self.assertEqual(managed.loc[0, "sell_status"], "incomplete_order_metadata")
        self.assertEqual(managed.loc[0, "sell_alpaca_order_id"], "sell-2")
        self.assertIn("inside the renewal window", result.loc[0, "Message"])
        mock_post.assert_called_once()

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_rejected_replacement_is_not_reported_as_renewed(
        self, mock_get: Mock, mock_post: Mock,
    ) -> None:
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            save_alpaca_managed_buy_order(
                conn, symbol="TQQQ", signal_symbol="QQQ", buy_rsi=30,
                profit_target_multiple=1.5, buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-20260102", buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-01-02T14:30:00Z", buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn, 1, buy_status="filled", filled_qty=2, filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z", target_sell_price=150,
            )
            record_alpaca_managed_sell_order(
                conn, 1, sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id="sell-1", sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="expired",
            )
            mock_get.side_effect = [
                response(200, {"id": "sell-1", "status": "expired"}),
                response(200, []),
            ]
            mock_post.return_value = response(
                200,
                {
                    "id": "sell-2",
                    "client_order_id": "rsi-exit-TQQQ-1-r1",
                    "status": "rejected",
                },
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "rejected")
        self.assertIn("no active replacement", result.loc[0, "Message"])
        self.assertEqual(managed.loc[0, "sell_status"], "rejected")
        self.assertEqual(managed.loc[0, "sell_client_order_id"], "rsi-exit-TQQQ-1-r1")
        mock_post.assert_called_once()

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_partially_filled_replacement_is_accounted_immediately(
        self, mock_get: Mock, mock_post: Mock,
    ) -> None:
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            save_alpaca_managed_buy_order(
                conn, symbol="TQQQ", signal_symbol="QQQ", buy_rsi=30,
                profit_target_multiple=1.5, buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-20260102", buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-01-02T14:30:00Z", buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn, 1, buy_status="filled", filled_qty=2, filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z", target_sell_price=150,
            )
            record_alpaca_managed_sell_order(
                conn, 1, sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id="sell-1", sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="expired",
            )
            mock_get.side_effect = [
                response(200, {"id": "sell-1", "status": "expired"}),
                response(200, []),
            ]
            mock_post.return_value = response(
                200,
                {
                    "id": "sell-2",
                    "client_order_id": "rsi-exit-TQQQ-1-r1",
                    "status": "partially_filled",
                    "filled_qty": "1",
                    "filled_avg_price": "150",
                    "qty": "2",
                    "limit_price": "150",
                },
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "renewed")
        self.assertEqual(managed.loc[0, "sell_status"], "partially_filled")
        self.assertEqual(managed.loc[0, "sold_qty"], 1)
        self.assertEqual(managed.loc[0, "remaining_qty"], 1)
        self.assertEqual(managed.loc[0, "realized_pl"], 50)

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_active_partial_replacement_covering_position_does_not_close(
        self, mock_get: Mock, mock_post: Mock,
    ) -> None:
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            save_alpaca_managed_buy_order(
                conn, symbol="TQQQ", signal_symbol="QQQ", buy_rsi=30,
                profit_target_multiple=1.5, buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-20260102", buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-01-02T14:30:00Z", buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn, 1, buy_status="filled", filled_qty=2, filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z", target_sell_price=150,
            )
            record_alpaca_managed_sell_order(
                conn, 1, sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id="sell-1", sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="expired",
            )
            mock_get.side_effect = [
                response(200, {"id": "sell-1", "status": "expired"}),
                response(200, []),
            ]
            mock_post.return_value = response(
                200,
                {
                    "id": "sell-2",
                    "client_order_id": "rsi-exit-TQQQ-1-r1",
                    "status": "partially_filled",
                    "filled_qty": "2",
                    "filled_avg_price": "150",
                    "qty": "2",
                    "limit_price": "150",
                },
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "quantity_mismatch")
        self.assertEqual(managed.loc[0, "sell_status"], "quantity_mismatch")
        self.assertEqual(managed.loc[0, "remaining_qty"], 0)
        self.assertTrue(pd.isna(managed.loc[0, "closed_at"]))

    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_oversized_replacement_response_is_blocked_before_accounting(
        self, mock_get: Mock, mock_post: Mock,
    ) -> None:
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            save_alpaca_managed_buy_order(
                conn, symbol="TQQQ", signal_symbol="QQQ", buy_rsi=30,
                profit_target_multiple=1.5, buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-20260102", buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-01-02T14:30:00Z", buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn, 1, buy_status="filled", filled_qty=2, filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z", target_sell_price=150,
            )
            record_alpaca_managed_sell_order(
                conn, 1, sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id="sell-1", sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="expired",
            )
            mock_get.side_effect = [
                response(200, {"id": "sell-1", "status": "expired"}),
                response(200, []),
            ]
            mock_post.return_value = response(
                200,
                {
                    "id": "sell-2", "client_order_id": "rsi-exit-TQQQ-1-r1",
                    "status": "partially_filled", "qty": "3", "limit_price": "149",
                    "filled_qty": "1", "filled_avg_price": "149",
                },
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "quantity_mismatch")
        self.assertEqual(managed.loc[0, "sell_status"], "quantity_mismatch")
        self.assertEqual(managed.loc[0, "sell_alpaca_order_id"], "sell-2")
        self.assertEqual(managed.loc[0, "remaining_qty"], 1)
        self.assertEqual(managed.loc[0, "sold_qty"], 1)
        self.assertEqual(managed.loc[0, "realized_pl"], 49)

    @patch("leveraged_trader.alpaca._utc_now")
    @patch("leveraged_trader.alpaca.requests.delete")
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_managed_sell_near_expiration_is_canceled_and_renewed(
        self,
        mock_get: Mock,
        mock_post: Mock,
        mock_delete: Mock,
        mock_utc_now: Mock,
    ) -> None:
        mock_utc_now.return_value = datetime(2026, 3, 25, tzinfo=UTC)
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
            mock_get.side_effect = [
                response(
                    200,
                    {
                        "id": "sell-1",
                        "status": "accepted",
                        "submitted_at": "2026-01-02T14:32:00Z",
                        "expires_at": "2026-03-30T20:15:00Z",
                        "qty": "2",
                        "limit_price": "150",
                    },
                ),
                response(
                    200,
                    {
                        "id": "sell-1",
                        "status": "canceled",
                        "submitted_at": "2026-01-02T14:32:00Z",
                        "expires_at": "2026-03-30T20:15:00Z",
                    },
                ),
                response(200, []),
            ]
            mock_delete.return_value = response(204, {})
            mock_post.return_value = response(
                200,
                {
                    "id": "sell-2",
                    "status": "accepted",
                    "submitted_at": "2026-03-25T14:32:00Z",
                    "expires_at": "2026-06-23T20:15:00Z",
                },
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "renewed")
        self.assertIn("before Alpaca aged-order expiration", result.loc[0, "Message"])
        self.assertEqual(managed.loc[0, "sell_client_order_id"], "rsi-exit-TQQQ-1-r1")
        self.assertEqual(managed.loc[0, "sell_alpaca_order_id"], "sell-2")
        self.assertEqual(managed.loc[0, "sell_renewal_count"], 1)
        self.assertTrue(pd.isna(managed.loc[0, "sell_renewal_requested_at"]))
        self.assertEqual(mock_delete.call_args.args[0], "https://paper-api.alpaca.markets/v2/orders/sell-1")
        self.assertEqual(mock_post.call_args.kwargs["json"]["qty"], "2")

    @patch("leveraged_trader.alpaca._utc_now")
    @patch("leveraged_trader.alpaca.requests.delete")
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_cancellation_refresh_fill_regression_blocks_replacement(
        self, mock_get: Mock, mock_post: Mock, mock_delete: Mock, mock_utc_now: Mock,
    ) -> None:
        mock_utc_now.return_value = datetime(2026, 3, 25, tzinfo=UTC)
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            save_alpaca_managed_buy_order(
                conn, symbol="TQQQ", signal_symbol="QQQ", buy_rsi=30,
                profit_target_multiple=1.5, buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-20260102", buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-01-02T14:30:00Z", buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn, 1, buy_status="filled", filled_qty=2, filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z", target_sell_price=150,
            )
            record_alpaca_managed_sell_order(
                conn, 1, sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id="sell-1", sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="partially_filled", sell_order_qty=2,
            )
            mark_alpaca_managed_sell_filled(
                conn, 1, sell_status="partially_filled", sell_filled_qty=1,
                sell_filled_avg_price=150, sell_filled_at="2026-01-02T14:33:00Z",
                sell_alpaca_order_id="sell-1",
            )
            mock_get.side_effect = [
                response(
                    200,
                    {
                        "id": "sell-1", "status": "partially_filled", "qty": "2",
                        "limit_price": "150", "filled_qty": "1", "filled_avg_price": "150",
                        "expires_at": "2026-03-30T20:15:00Z",
                    },
                ),
                response(
                    200,
                    {
                        "id": "sell-1", "status": "canceled", "filled_qty": "0.5",
                        "filled_avg_price": "150",
                    },
                ),
            ]
            mock_delete.return_value = response(204, {})

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "fill_quantity_regression")
        self.assertEqual(managed.loc[0, "sell_status"], "fill_quantity_regression")
        self.assertEqual(managed.loc[0, "sold_qty"], 1)
        self.assertEqual(managed.loc[0, "remaining_qty"], 1)
        mock_delete.assert_called_once()
        mock_post.assert_not_called()

    @patch("leveraged_trader.alpaca._utc_now")
    @patch("leveraged_trader.alpaca.requests.delete")
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_managed_sell_renewal_uses_persisted_expiration_when_broker_value_is_invalid(
        self,
        mock_get: Mock,
        mock_post: Mock,
        mock_delete: Mock,
        mock_utc_now: Mock,
    ) -> None:
        mock_utc_now.return_value = datetime(2026, 3, 25, tzinfo=UTC)
        for broker_expires_at in [None, "not-a-timestamp"]:
            with self.subTest(broker_expires_at=broker_expires_at), sqlite3.connect(":memory:") as conn:
                mock_get.reset_mock()
                mock_post.reset_mock()
                mock_delete.reset_mock()
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
                    filled_avg_price=100,
                    filled_at="2026-01-02T14:31:00Z",
                    target_sell_price=150,
                )
                record_alpaca_managed_sell_order(
                    conn,
                    1,
                    sell_client_order_id="rsi-exit-TQQQ-1",
                    sell_alpaca_order_id="sell-1",
                    sell_submitted_at="2026-01-02T14:32:00Z",
                    sell_status="accepted",
                    sell_expires_at="2026-03-30T20:15:00Z",
                )
                broker_order = {
                    "id": "sell-1", "status": "accepted", "qty": "2", "limit_price": "150"
                }
                if broker_expires_at is not None:
                    broker_order["expires_at"] = broker_expires_at
                mock_get.side_effect = [
                    response(200, broker_order),
                    response(200, {"id": "sell-1", "status": "canceled"}),
                    response(200, []),
                ]
                mock_delete.return_value = response(204, {})
                mock_post.return_value = response(200, {"id": "sell-2", "status": "accepted"})

                result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
                managed = load_alpaca_managed_positions(conn)

            self.assertEqual(result.loc[0, "Status"], "renewed")
            self.assertEqual(managed.loc[0, "sell_client_order_id"], "rsi-exit-TQQQ-1-r1")
            mock_delete.assert_called_once()
            mock_post.assert_called_once()

    @patch("leveraged_trader.alpaca.requests.delete")
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_active_sell_without_any_valid_expiration_is_blocked_for_review(
        self, mock_get: Mock, mock_post: Mock, mock_delete: Mock,
    ) -> None:
        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            save_alpaca_managed_buy_order(
                conn, symbol="TQQQ", signal_symbol="QQQ", buy_rsi=30,
                profit_target_multiple=1.5, buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-20260102", buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-01-02T14:30:00Z", buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn, 1, buy_status="filled", filled_qty=2, filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z", target_sell_price=150,
            )
            record_alpaca_managed_sell_order(
                conn, 1, sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id="sell-1", sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="accepted", sell_expires_at=None,
            )
            mock_get.return_value = response(
                200,
                {
                    "id": "sell-1", "status": "accepted", "qty": "2",
                    "limit_price": "150", "expires_at": "not-a-timestamp",
                },
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "incomplete_order_metadata")
        self.assertEqual(managed.loc[0, "sell_status"], "incomplete_order_metadata")
        mock_delete.assert_not_called()
        mock_post.assert_not_called()

    @patch("leveraged_trader.alpaca.requests.delete")
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_active_managed_sell_with_invalid_order_metadata_is_blocked_for_review(
        self, mock_get: Mock, mock_post: Mock, mock_delete: Mock,
    ) -> None:
        cases = [
            (None, "150"),
            ("not-a-quantity", "150"),
            ("0", "150"),
            ("2", "-1"),
        ]
        for qty, limit_price in cases:
            with self.subTest(qty=qty, limit_price=limit_price), sqlite3.connect(":memory:") as conn:
                mock_get.reset_mock()
                mock_post.reset_mock()
                mock_delete.reset_mock()
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
                    filled_avg_price=100,
                    filled_at="2026-01-02T14:31:00Z",
                    target_sell_price=150,
                )
                record_alpaca_managed_sell_order(
                    conn,
                    1,
                    sell_client_order_id="rsi-exit-TQQQ-1",
                    sell_alpaca_order_id="sell-1",
                    sell_submitted_at="2026-01-02T14:32:00Z",
                    sell_status="accepted",
                )
                broker_order = {
                    "id": "sell-1",
                    "status": "accepted",
                    "limit_price": limit_price,
                    "expires_at": "2026-06-30T20:15:00Z",
                }
                if qty is not None:
                    broker_order["qty"] = qty
                mock_get.return_value = response(200, broker_order)

                result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
                managed = load_alpaca_managed_positions(conn)

            self.assertEqual(result.loc[0, "Status"], "incomplete_order_metadata")
            self.assertEqual(managed.loc[0, "sell_status"], "incomplete_order_metadata")
            mock_delete.assert_not_called()
            mock_post.assert_not_called()

    @patch("leveraged_trader.alpaca.requests.delete")
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_partial_fill_is_recorded_before_invalid_order_metadata_blocks_renewal(
        self, mock_get: Mock, mock_post: Mock, mock_delete: Mock,
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
                buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn,
                1,
                buy_status="filled",
                filled_qty=3,
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
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
                    "status": "partially_filled",
                    "filled_qty": "1",
                    "filled_avg_price": "150",
                    "limit_price": "150",
                    "expires_at": "2026-06-30T20:15:00Z",
                },
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "incomplete_order_metadata")
        self.assertEqual(managed.loc[0, "sell_status"], "incomplete_order_metadata")
        self.assertEqual(managed.loc[0, "sold_qty"], 1)
        self.assertEqual(managed.loc[0, "remaining_qty"], 2)
        self.assertEqual(managed.loc[0, "realized_pl"], 50)
        mock_delete.assert_not_called()
        mock_post.assert_not_called()

    @patch("leveraged_trader.alpaca._utc_now")
    @patch("leveraged_trader.alpaca.requests.delete")
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_managed_sell_renewal_rejects_multiple_newer_generations(
        self,
        mock_get: Mock,
        mock_post: Mock,
        mock_delete: Mock,
        mock_utc_now: Mock,
    ) -> None:
        mock_utc_now.return_value = datetime(2026, 3, 25, tzinfo=UTC)
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
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
            )
            record_alpaca_managed_sell_order(
                conn,
                1,
                sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id="sell-1",
                sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="accepted",
            )
            mock_get.side_effect = [
                response(
                    200,
                    {
                        "id": "sell-1",
                        "status": "accepted",
                        "expires_at": "2026-03-30T20:15:00Z",
                        "qty": "2",
                        "limit_price": "150",
                    },
                ),
                response(200, {"id": "sell-1", "status": "canceled"}),
                response(
                    200,
                    [
                        {
                            "id": "sell-1",
                            "symbol": "TQQQ",
                            "side": "sell",
                            "client_order_id": "rsi-exit-TQQQ-1",
                        },
                        {
                            "id": "sell-2",
                            "symbol": "TQQQ",
                            "side": "sell",
                            "client_order_id": "rsi-exit-TQQQ-1-r1",
                        },
                        {
                            "id": "sell-3",
                            "symbol": "TQQQ",
                            "side": "sell",
                            "client_order_id": "rsi-exit-TQQQ-1-r2",
                        },
                    ],
                ),
            ]
            mock_delete.return_value = response(204, {})

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "managed_order_conflict")
        self.assertEqual(managed.loc[0, "sell_client_order_id"], "rsi-exit-TQQQ-1")
        self.assertEqual(managed.loc[0, "sell_status"], "canceled")
        self.assertFalse(pd.isna(managed.loc[0, "sell_renewal_requested_at"]))
        mock_delete.assert_called_once()
        mock_post.assert_not_called()

    @patch("leveraged_trader.alpaca._utc_now")
    @patch("leveraged_trader.alpaca.requests.delete")
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_managed_sell_renewal_does_not_replace_stopped_or_suspended_order(
        self,
        mock_get: Mock,
        mock_post: Mock,
        mock_delete: Mock,
        mock_utc_now: Mock,
    ) -> None:
        mock_utc_now.return_value = datetime(2026, 3, 25, tzinfo=UTC)
        for broker_status in ["stopped", "suspended"]:
            with self.subTest(broker_status=broker_status), sqlite3.connect(":memory:") as conn:
                mock_get.reset_mock()
                mock_post.reset_mock()
                mock_delete.reset_mock()
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
                    filled_avg_price=100,
                    filled_at="2026-01-02T14:31:00Z",
                    target_sell_price=150,
                )
                record_alpaca_managed_sell_order(
                    conn,
                    1,
                    sell_client_order_id="rsi-exit-TQQQ-1",
                    sell_alpaca_order_id="sell-1",
                    sell_submitted_at="2026-01-02T14:32:00Z",
                    sell_status="accepted",
                )
                mock_get.side_effect = [
                    response(
                        200,
                        {
                            "id": "sell-1",
                            "status": "accepted",
                            "expires_at": "2026-03-30T20:15:00Z",
                            "qty": "2",
                            "limit_price": "150",
                        },
                    ),
                    response(200, {"id": "sell-1", "status": broker_status}),
                ]
                mock_delete.return_value = response(204, {})

                result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
                managed = load_alpaca_managed_positions(conn)

            self.assertEqual(result.loc[0, "Status"], broker_status)
            self.assertIn("replacement was not submitted", result.loc[0, "Message"])
            self.assertEqual(managed.loc[0, "sell_client_order_id"], "rsi-exit-TQQQ-1")
            self.assertEqual(managed.loc[0, "sell_status"], broker_status)
            mock_delete.assert_called_once()
            mock_post.assert_not_called()

    @patch("leveraged_trader.alpaca._utc_now")
    @patch("leveraged_trader.alpaca.requests.delete")
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_managed_sell_cancel_timeout_refreshes_and_renews_immediately(
        self,
        mock_get: Mock,
        mock_post: Mock,
        mock_delete: Mock,
        mock_utc_now: Mock,
    ) -> None:
        mock_utc_now.return_value = datetime(2026, 3, 25, tzinfo=UTC)
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
            mock_get.side_effect = [
                response(
                    200,
                    {
                        "id": "sell-1",
                        "status": "accepted",
                        "submitted_at": "2026-01-02T14:32:00Z",
                        "expires_at": "2026-03-30T20:15:00Z",
                        "qty": "2",
                        "limit_price": "150",
                    },
                ),
                response(
                    200,
                    {
                        "id": "sell-1",
                        "status": "canceled",
                        "submitted_at": "2026-01-02T14:32:00Z",
                        "expires_at": "2026-03-30T20:15:00Z",
                    },
                ),
                response(200, []),
            ]
            mock_delete.side_effect = requests.exceptions.Timeout("cancel response lost")
            mock_post.return_value = response(
                200,
                {
                    "id": "sell-2",
                    "status": "accepted",
                    "submitted_at": "2026-03-25T14:32:00Z",
                    "expires_at": "2026-06-23T20:15:00Z",
                },
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))
            renewed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "renewed")
        self.assertEqual(renewed.loc[0, "sell_client_order_id"], "rsi-exit-TQQQ-1-r1")
        self.assertEqual(renewed.loc[0, "sell_alpaca_order_id"], "sell-2")
        self.assertTrue(pd.isna(renewed.loc[0, "sell_renewal_requested_at"]))
        mock_delete.assert_called_once()
        mock_post.assert_called_once()

    @patch("leveraged_trader.alpaca._utc_now")
    @patch("leveraged_trader.alpaca.requests.delete")
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_managed_sell_pending_cancel_does_not_submit_duplicate_replacement(
        self,
        mock_get: Mock,
        mock_post: Mock,
        mock_delete: Mock,
        mock_utc_now: Mock,
    ) -> None:
        mock_utc_now.return_value = datetime(2026, 3, 25, tzinfo=UTC)
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
            update_alpaca_managed_sell_status(
                conn,
                1,
                sell_status="pending_cancel",
                sell_renewal_requested_at="2026-03-25T14:32:00Z",
            )
            mock_get.return_value = response(
                200,
                {
                    "id": "sell-1",
                    "status": "pending_cancel",
                    "submitted_at": "2026-01-02T14:32:00Z",
                    "expires_at": "2026-03-30T20:15:00Z",
                },
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))

        self.assertEqual(result.loc[0, "Status"], "pending_cancel")
        self.assertIn("replacement not submitted yet", result.loc[0, "Message"])
        mock_delete.assert_not_called()
        mock_post.assert_not_called()

    @patch("leveraged_trader.alpaca._utc_now")
    @patch("leveraged_trader.alpaca.requests.delete")
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_stale_pending_cancel_claim_retries_cancellation_and_replaces_order(
        self,
        mock_get: Mock,
        mock_post: Mock,
        mock_delete: Mock,
        mock_utc_now: Mock,
    ) -> None:
        mock_utc_now.return_value = datetime(2026, 3, 25, 15, 0, tzinfo=UTC)
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
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
            )
            record_alpaca_managed_sell_order(
                conn,
                1,
                sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id="sell-1",
                sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="accepted",
            )
            update_alpaca_managed_sell_status(
                conn,
                1,
                sell_status="pending_cancel",
                sell_renewal_requested_at="2026-03-25T14:50:00Z",
            )
            mock_get.side_effect = [
                response(
                    200,
                    {
                        "id": "sell-1",
                        "status": "pending_cancel",
                        "qty": "2",
                        "limit_price": "150",
                    },
                ),
                response(200, {"id": "sell-1", "status": "canceled"}),
                response(200, []),
            ]
            mock_delete.return_value = response(204, {})
            mock_post.return_value = response(
                200,
                {
                    "id": "sell-2",
                    "status": "accepted",
                    "submitted_at": "2026-03-25T15:00:00Z",
                    "expires_at": "2026-06-23T20:15:00Z",
                },
            )

            result = reconcile_alpaca_managed_positions(
                conn,
                self.cfg(buy=True, sell=True, gtc_sell_renewal_enabled=False),
            )
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "renewed")
        self.assertEqual(managed.loc[0, "sell_client_order_id"], "rsi-exit-TQQQ-1-r1")
        self.assertEqual(managed.loc[0, "sell_alpaca_order_id"], "sell-2")
        mock_delete.assert_called_once()
        mock_post.assert_called_once()

    @patch("leveraged_trader.alpaca.requests.delete")
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_expired_sell_with_persisted_renewal_intent_is_replaced_when_renewal_is_disabled(
        self,
        mock_get: Mock,
        mock_post: Mock,
        mock_delete: Mock,
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
                buy_status="filled",
            )
            mark_alpaca_managed_buy_filled(
                conn,
                1,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
            )
            record_alpaca_managed_sell_order(
                conn,
                1,
                sell_client_order_id="rsi-exit-TQQQ-1",
                sell_alpaca_order_id="sell-1",
                sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="pending_cancel",
            )
            update_alpaca_managed_sell_status(
                conn,
                1,
                sell_status="pending_cancel",
                sell_renewal_requested_at="2026-03-25T14:50:00Z",
            )
            mock_get.side_effect = [
                response(200, {"id": "sell-1", "status": "expired"}),
                response(200, []),
            ]
            mock_post.return_value = response(
                200,
                {
                    "id": "sell-2",
                    "status": "accepted",
                    "submitted_at": "2026-04-01T14:32:00Z",
                    "expires_at": "2027-06-30T20:15:00Z",
                },
            )

            result = reconcile_alpaca_managed_positions(
                conn,
                self.cfg(buy=True, sell=True, gtc_sell_renewal_enabled=False),
            )
            managed = load_alpaca_managed_positions(conn)

        self.assertEqual(result.loc[0, "Status"], "renewed")
        self.assertEqual(managed.loc[0, "sell_client_order_id"], "rsi-exit-TQQQ-1-r1")
        self.assertEqual(managed.loc[0, "sell_alpaca_order_id"], "sell-2")
        mock_delete.assert_not_called()
        mock_post.assert_called_once()

    @patch("leveraged_trader.alpaca.requests.delete")
    @patch("leveraged_trader.alpaca.requests.post")
    @patch("leveraged_trader.alpaca.requests.get")
    def test_manually_canceled_managed_sell_is_not_resubmitted(
        self,
        mock_get: Mock,
        mock_post: Mock,
        mock_delete: Mock,
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
                    "status": "canceled",
                    "submitted_at": "2026-01-02T14:32:00Z",
                },
            )

            result = reconcile_alpaca_managed_positions(conn, self.cfg(buy=True, sell=True))

        self.assertEqual(result.loc[0, "Status"], "canceled")
        self.assertIn("no automatic resubmission", result.loc[0, "Message"])
        mock_delete.assert_not_called()
        mock_post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
