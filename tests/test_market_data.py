from __future__ import annotations

import unittest
from datetime import datetime
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance.shared as yf_shared

from leveraged_trader.config import TradierMarketDataConfig
from leveraged_trader.market_data import (
    TRADIER_RECOVERED_SYMBOLS_ATTR,
    MarketDataDownloadError,
    exclude_unfinalized_daily_bar,
    load_market_data,
)


class MarketDataTests(unittest.TestCase):
    def test_current_session_daily_bar_is_excluded_before_finalization(self) -> None:
        index = pd.to_datetime(["2026-01-02", "2026-01-05"])
        data = pd.DataFrame({"AAA_Close": [100.0, 110.0]}, index=index)
        data.attrs[TRADIER_RECOVERED_SYMBOLS_ATTR] = ["AAA"]

        finalized = exclude_unfinalized_daily_bar(
            data,
            now=datetime(2026, 1, 5, 15, 30, tzinfo=ZoneInfo("America/New_York")),
        )

        self.assertEqual(finalized.index.tolist(), [pd.Timestamp("2026-01-02")])
        self.assertEqual(finalized.attrs[TRADIER_RECOVERED_SYMBOLS_ATTR], ["AAA"])

    def test_current_session_daily_bar_is_excluded_after_close_too(self) -> None:
        index = pd.to_datetime(["2026-01-02", "2026-01-05"])
        data = pd.DataFrame({"AAA_Close": [100.0, 110.0]}, index=index)

        finalized = exclude_unfinalized_daily_bar(
            data,
            now=datetime(2026, 1, 5, 16, 15, tzinfo=ZoneInfo("America/New_York")),
        )

        self.assertEqual(finalized.index.tolist(), [pd.Timestamp("2026-01-02")])

    def test_yfinance_errors_become_single_human_readable_download_error(self) -> None:
        def fake_download(**_: object) -> pd.DataFrame:
            yf_shared._ERRORS = {
                "PRICE": "$PRICE: possibly delisted; no timezone found",
                "CLOUD": "$CLOUD: possibly delisted; no timezone found",
            }
            return pd.DataFrame()

        with (
            patch("leveraged_trader.market_data.yf.download", side_effect=fake_download),
            self.assertRaises(MarketDataDownloadError) as raised,
        ):
            load_market_data(symbols=["PRICE", "CLOUD"])

        message = str(raised.exception)
        self.assertEqual(raised.exception.symbols, ["CLOUD", "PRICE"])
        self.assertIn("Yahoo Finance did not return usable daily data", message)
        self.assertIn("CLOUD, PRICE: no timezone found", message)
        self.assertNotIn("$PRICE", message)
        self.assertNotIn("Failed download", message)

    @patch("leveraged_trader.market_data.yf.download")
    def test_missing_downloaded_symbol_identifies_impacted_symbol(self, mock_download: Mock) -> None:
        index = pd.to_datetime(["2026-01-02"])
        columns = pd.MultiIndex.from_product([["AAA"], ["Open", "High", "Low", "Close", "Volume"]])
        mock_download.return_value = pd.DataFrame(
            [[1.0, 2.0, 0.5, 1.5, 1000]],
            index=index,
            columns=columns,
        )

        with self.assertRaises(MarketDataDownloadError) as raised:
            load_market_data(symbols=["AAA", "MISSING"])

        message = str(raised.exception)
        self.assertEqual(raised.exception.symbols, ["MISSING"])
        self.assertIn("MISSING", message)
        self.assertIn("missing from the Yahoo Finance response", message)

    @patch("leveraged_trader.market_data.requests.get")
    @patch("leveraged_trader.market_data.yf.download")
    def test_incomplete_yahoo_ohlcv_frame_uses_tradier_fallback(
        self,
        mock_download: Mock,
        mock_get: Mock,
    ) -> None:
        index = pd.to_datetime(["2026-01-02"])
        mock_download.return_value = pd.DataFrame(
            [[10.5]],
            index=index,
            columns=pd.MultiIndex.from_tuples([("AAA", "Close")]),
        )
        mock_get.return_value = Mock(
            status_code=200,
            text="",
            json=Mock(
                return_value={
                    "history": {
                        "day": {
                            "date": "2026-01-02",
                            "open": 10,
                            "high": 11,
                            "low": 9,
                            "close": 10.5,
                            "volume": 1000,
                        }
                    }
                }
            ),
        )

        data = load_market_data(
            symbols=["AAA"],
            tradier_cfg=TradierMarketDataConfig(access_token="token"),
        )

        self.assertEqual(data.columns.tolist(), [
            "AAA_Open", "AAA_High", "AAA_Low", "AAA_Close", "AAA_Volume"
        ])
        self.assertEqual(data.attrs[TRADIER_RECOVERED_SYMBOLS_ATTR], ["AAA"])

    @patch("leveraged_trader.market_data.requests.get")
    @patch("leveraged_trader.market_data.yf.download")
    def test_tradier_recovers_yfinance_failed_symbol(self, mock_download: Mock, mock_get: Mock) -> None:
        index = pd.to_datetime(["2026-01-02", "2026-01-05"])
        columns = pd.MultiIndex.from_product([["AAA"], ["Open", "High", "Low", "Close", "Volume"]])

        def fake_download(**_: object) -> pd.DataFrame:
            yf_shared._ERRORS = {"MISSING": "$MISSING: possibly delisted; no timezone found"}
            return pd.DataFrame(
                [
                    [10.0, 11.0, 9.0, 10.5, 1000],
                    [10.5, 12.0, 10.0, 11.5, 1100],
                ],
                index=index,
                columns=columns,
            )

        mock_download.side_effect = fake_download
        mock_get.return_value = Mock(
            status_code=200,
            text="",
            json=Mock(
                return_value={
                    "history": {
                        "day": [
                            {"date": "2026-01-02", "open": 20, "high": 21, "low": 19, "close": 20.5, "volume": 2000},
                            {"date": "2026-01-05", "open": 21, "high": 22, "low": 20, "close": 21.5, "volume": 2100},
                        ]
                    }
                }
            ),
        )

        data = load_market_data(
            symbols=["AAA", "MISSING"],
            tradier_cfg=TradierMarketDataConfig(access_token="token"),
        )

        self.assertEqual(list(data.columns), [
            "AAA_Open",
            "AAA_High",
            "AAA_Low",
            "AAA_Close",
            "AAA_Volume",
            "MISSING_Open",
            "MISSING_High",
            "MISSING_Low",
            "MISSING_Close",
            "MISSING_Volume",
        ])
        self.assertEqual(data.attrs[TRADIER_RECOVERED_SYMBOLS_ATTR], ["MISSING"])
        self.assertEqual(mock_get.call_args.kwargs["params"]["symbol"], "MISSING")
        self.assertEqual(mock_get.call_args.kwargs["headers"]["Authorization"], "Bearer token")

    @patch("leveraged_trader.market_data.requests.get")
    @patch("leveraged_trader.market_data.yf.download")
    def test_tradier_recovers_yfinance_non_overlapping_pair(self, mock_download: Mock, mock_get: Mock) -> None:
        index = pd.to_datetime(["2026-01-02", "2026-01-05"])
        fields = ["Open", "High", "Low", "Close", "Volume"]
        columns = pd.MultiIndex.from_product([["AAA", "BBB"], fields])
        mock_download.return_value = pd.DataFrame(
            [
                [10.0, 11.0, 9.0, 10.5, 1000, None, None, None, None, None],
                [None, None, None, None, None, 20.0, 21.0, 19.0, 20.5, 2000],
            ],
            index=index,
            columns=columns,
        )

        def fake_tradier_get(_url: str, **kwargs: object) -> Mock:
            params = kwargs["params"]
            self.assertIsInstance(params, dict)
            symbol = params["symbol"]
            open_price = 30 if symbol == "AAA" else 40
            return Mock(
                status_code=200,
                text="",
                json=Mock(
                    return_value={
                        "history": {
                            "day": {
                                "date": "2026-01-05",
                                "open": open_price,
                                "high": open_price + 1,
                                "low": open_price - 1,
                                "close": open_price + 0.5,
                                "volume": 3000,
                            }
                        }
                    }
                ),
            )

        mock_get.side_effect = fake_tradier_get

        data = load_market_data(
            symbols=["AAA", "BBB"],
            tradier_cfg=TradierMarketDataConfig(access_token="token"),
        )

        self.assertEqual(list(data.index), [pd.Timestamp("2026-01-05")])
        self.assertEqual(data.loc[pd.Timestamp("2026-01-05"), "AAA_Open"], 30)
        self.assertEqual(data.loc[pd.Timestamp("2026-01-05"), "BBB_Open"], 40)
        self.assertEqual(data.attrs[TRADIER_RECOVERED_SYMBOLS_ATTR], ["AAA", "BBB"])

    @patch("leveraged_trader.market_data.requests.get")
    @patch("leveraged_trader.market_data.yf.download")
    def test_tradier_symbol_uses_slash_for_class_shares(self, mock_download: Mock, mock_get: Mock) -> None:
        def fake_download(**_: object) -> pd.DataFrame:
            yf_shared._ERRORS = {"BRK-B": "$BRK-B: possibly delisted; no timezone found"}
            return pd.DataFrame()

        mock_download.side_effect = fake_download
        mock_get.return_value = Mock(
            status_code=200,
            text="",
            json=Mock(
                return_value={
                    "history": {
                        "day": {
                            "date": "2026-01-02",
                            "open": 20,
                            "high": 21,
                            "low": 19,
                            "close": 20.5,
                            "volume": 2000,
                        }
                    }
                }
            ),
        )

        load_market_data(
            symbols=["BRK-B"],
            tradier_cfg=TradierMarketDataConfig(access_token="token"),
        )

        self.assertEqual(mock_get.call_args.kwargs["params"]["symbol"], "BRK/B")

    @patch("leveraged_trader.market_data.requests.get")
    @patch("leveraged_trader.market_data.yf.download")
    def test_tradier_partial_failure_reports_only_unresolved_symbols(
        self,
        mock_download: Mock,
        mock_get: Mock,
    ) -> None:
        def fake_download(**_: object) -> pd.DataFrame:
            yf_shared._ERRORS = {
                "PRICE": "$PRICE: possibly delisted; no timezone found",
                "CLOUD": "$CLOUD: possibly delisted; no timezone found",
            }
            return pd.DataFrame()

        def fake_tradier_get(_url: str, **kwargs: object) -> Mock:
            params = kwargs["params"]
            self.assertIsInstance(params, dict)
            symbol = params["symbol"]
            if symbol == "CLOUD":
                return Mock(
                    status_code=200,
                    text="",
                    json=Mock(
                        return_value={
                            "history": {
                                "day": {
                                    "date": "2026-01-02",
                                    "open": 20,
                                    "high": 21,
                                    "low": 19,
                                    "close": 20.5,
                                    "volume": 2000,
                                }
                            }
                        }
                    ),
                )
            return Mock(status_code=200, text="", json=Mock(return_value={"history": {"day": []}}))

        mock_download.side_effect = fake_download
        mock_get.side_effect = fake_tradier_get

        with self.assertRaises(MarketDataDownloadError) as raised:
            load_market_data(
                symbols=["PRICE", "CLOUD"],
                tradier_cfg=TradierMarketDataConfig(access_token="token"),
            )

        message = str(raised.exception)
        self.assertEqual(raised.exception.symbols, ["PRICE"])
        self.assertIn("Yahoo Finance and Tradier did not return usable daily data", message)
        self.assertIn("PRICE: Yahoo Finance: no timezone found", message)
        self.assertIn("Tradier fallback: No historical daily data returned", message)
        self.assertNotIn("$PRICE", message)
        self.assertNotIn("CLOUD:", message)


if __name__ == "__main__":
    unittest.main()
