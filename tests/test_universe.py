from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import pandas as pd

from leveraged_trader.config import UniverseConfig
from leveraged_trader.universe import (
    AUDIT_UNIVERSE_SOURCES,
    ISSUER_UNIVERSE_SOURCES,
    WORKFLOW_ETN_SOURCES,
    ActiveListedSymbols,
    UniverseSource,
    _defiance_json_to_universe,
    _etracs_leverage_table_to_universe,
    _graniteshares_html_to_universe,
    _html_cards_to_universe,
    _issuer_table_to_universe,
    _js_ticker_name_to_universe,
    _merge_universe_sources,
    _microsectors_html_to_universe,
    _rex_menu_html_to_universe,
    _sec_company_tickers_to_universe,
    _sec_exchange_tickers_to_universe,
    _sec_mutual_fund_tickers_to_universe,
    _volatilityshares_html_to_universe,
    _with_audit_metadata,
    build_nasdaq_universe_table,
    build_universe_audit_report,
    determine_workflow_asset_groups,
    determine_workflow_assets,
    infer_leverage_and_direction,
    infer_rsi_mapping,
    infer_rsi_symbol,
    is_long_leveraged_name,
    is_short_leveraged_name,
    leveraged_name_filter,
    load_active_listed_symbols,
    load_current_etf_universe,
    load_etn_universe,
    load_issuer_etf_universe,
    select_short_workflow_universe,
)


class UniverseTests(unittest.TestCase):
    @patch("leveraged_trader.universe.load_current_etf_universe")
    def test_invalid_workflow_universe_limit_fails_before_discovery(self, mock_load_current: Mock) -> None:
        for invalid_limit in [0, -1, True, 1.5]:
            with (
                self.subTest(top_n=invalid_limit),
                self.assertRaisesRegex(ValueError, "positive integer or None"),
            ):
                determine_workflow_assets(
                    UniverseConfig(sqlite_db_path="state.sqlite", top_n=invalid_limit)
                )

        mock_load_current.assert_not_called()

    def test_positive_workflow_universe_limit_preserves_workflow_metadata(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [
                {"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"},
                {"symbol": "UPRO", "name": "ProShares UltraPro S&P500", "fund_type": "ETF"},
            ]
        )
        empty_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        empty_rows.attrs["workflow_source_status"] = []

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=empty_rows),
            patch(
                "leveraged_trader.universe.load_active_listed_symbols",
                return_value={"TQQQ", "UPRO", "QQQ", "SPY"},
            ),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite"),
        ):
            workflow_assets = determine_workflow_assets(
                UniverseConfig(sqlite_db_path="state.sqlite", top_n=1)
            )

        self.assertEqual(len(workflow_assets), 1)
        self.assertIn("rsi_symbol", workflow_assets.columns)

    def test_workflow_asset_groups_include_short_inverse_products(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [
                {"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"},
                {"symbol": "SQQQ", "name": "ProShares UltraPro Short QQQ", "fund_type": "ETF"},
            ]
        )
        empty_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        empty_rows.attrs["workflow_source_status"] = []

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=empty_rows),
            patch(
                "leveraged_trader.universe.load_active_listed_symbols",
                return_value={"TQQQ", "SQQQ", "QQQ"},
            ),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite"),
        ):
            workflow_asset_groups = determine_workflow_asset_groups(
                UniverseConfig(sqlite_db_path="state.sqlite")
            )

        self.assertEqual(workflow_asset_groups["long"]["symbol"].tolist(), ["TQQQ"])
        self.assertEqual(workflow_asset_groups["short"]["symbol"].tolist(), ["SQQQ"])
        self.assertEqual(workflow_asset_groups["short"].loc[0, "rsi_symbol"], "QQQ")
        self.assertEqual(workflow_asset_groups["short"].loc[0, "direction"], "inverse")

    def test_workflow_asset_groups_allow_short_only_universe(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [
                {"symbol": "QQQ", "name": "Invesco QQQ Trust", "fund_type": "ETF"},
                {"symbol": "SQQQ", "name": "ProShares UltraPro Short QQQ", "fund_type": "ETF"},
            ]
        )
        empty_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        empty_rows.attrs["workflow_source_status"] = []

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=empty_rows),
            patch(
                "leveraged_trader.universe.load_active_listed_symbols",
                return_value={"SQQQ", "QQQ"},
            ),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite"),
        ):
            workflow_asset_groups = determine_workflow_asset_groups(
                UniverseConfig(sqlite_db_path="state.sqlite")
            )

        self.assertTrue(workflow_asset_groups["long"].empty)
        self.assertEqual(workflow_asset_groups["short"]["symbol"].tolist(), ["SQQQ"])
        self.assertEqual(
            workflow_asset_groups["short"].attrs["universe_counts"]["Executable short leveraged ETFs/ETNs selected"],
            1,
        )

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value={"SQQQ", "QQQ"}),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite"),
            self.assertRaisesRegex(RuntimeError, "no executable long leveraged"),
        ):
            determine_workflow_assets(UniverseConfig(sqlite_db_path="state.sqlite"))

    @patch("leveraged_trader.universe._read_nasdaq_symbol_file")
    def test_active_listing_status_marks_partial_download_non_authoritative(
        self,
        mock_read: Mock,
    ) -> None:
        mock_read.side_effect = [
            pd.DataFrame({"Symbol": ["TQQQ"]}),
            RuntimeError("otherlisted unavailable"),
        ]

        active_symbols = load_active_listed_symbols()

        self.assertEqual(active_symbols, {"TQQQ"})
        self.assertFalse(active_symbols.is_complete)
        self.assertEqual(active_symbols.source_status[1]["status"], "error")

    def test_nasdaq_metadata_wins_duplicate_issuer_symbol(self) -> None:
        merged = _merge_universe_sources(
            pd.DataFrame([
                {"symbol": "TQQQ", "name": "Nasdaq Current Name", "fund_type": "ETF"},
            ]),
            pd.DataFrame([
                {
                    "symbol": "TQQQ",
                    "name": "Issuer Stale Name",
                    "fund_type": "ETF (Issuer)",
                    "source": "Issuer table",
                },
            ]),
        )

        self.assertEqual(merged.loc[0, "name"], "Nasdaq Current Name")
        self.assertEqual(merged.loc[0, "source"], "Nasdaq ETF definitions")

    def test_infers_underlying_symbol_from_leveraged_name(self) -> None:
        self.assertEqual(infer_rsi_symbol("TQQQ", "ProShares UltraPro QQQ"), "QQQ")

    def test_falls_back_to_asset_symbol_when_no_underlying_is_found(self) -> None:
        self.assertEqual(infer_rsi_symbol("XYZ", "Plain Fund Name"), "XYZ")

    def test_nasdaq_is_not_used_as_inferred_rsi_symbol(self) -> None:
        self.assertEqual(infer_rsi_symbol("QQQU", "Defiance Daily Target 2X Long NASDAQ ETF"), "QQQU")

    def test_direct_generic_inference_uses_safe_default_allowlist(self) -> None:
        mapping = infer_rsi_mapping("ENGY", "Ultra Energy")

        self.assertEqual(mapping.rsi_symbol, "ENGY")
        self.assertEqual(mapping.confidence, "fallback_to_self")
        self.assertEqual(infer_rsi_symbol("TQQQ", "ProShares UltraPro QQQ"), "QQQ")

    def test_inferred_underlying_must_be_known_when_known_symbols_are_provided(self) -> None:
        self.assertEqual(
            infer_rsi_symbol("ENGY", "ProShares Ultra Energy", known_symbols={"ENGY", "QQQ"}),
            "ENGY",
        )
        self.assertEqual(
            infer_rsi_symbol("AAPU", "Direxion Daily AAPL Bull 2X ETF", known_symbols={"AAPU", "AAPL"}),
            "AAPL",
        )
        self.assertEqual(
            infer_rsi_symbol("SPYU", "SPY 4X Daily ETF", known_symbols={"SPYU", "SPY"}),
            "SPY",
        )
        self.assertEqual(
            infer_rsi_symbol("FIVE", "5X Long QQQ Daily ETF", known_symbols={"FIVE", "QQQ"}),
            "QQQ",
        )
        self.assertEqual(
            infer_rsi_symbol("FIVP", "500% Long QQQ Daily ETF", known_symbols={"FIVP", "QQQ"}),
            "QQQ",
        )

    def test_normalizes_brkb_to_yahoo_symbol(self) -> None:
        self.assertEqual(infer_rsi_symbol("BRKU", "2X Long BRKB Daily ETF"), "BRK-B")
        self.assertEqual(infer_rsi_symbol("BRKU", "2X Long BRK.B Daily ETF"), "BRK-B")

    def test_curated_company_name_underlying_mappings(self) -> None:
        cases = {
            "AAPX": ("T-REX 2X Long Apple Daily Target ETF", "AAPL"),
            "ETNG": ("2x Long ETN Daily ETF", "ETN"),
            "GOOX": ("T-REX 2X Long Alphabet Daily Target ETF", "GOOG"),
            "MSFX": ("T-REX 2X Long Microsoft Daily Target ETF", "MSFT"),
            "NVDQ": ("Tradr 2X Short NVDA Daily ETF", "NVDA"),
            "NVDX": ("T-REX 2X Long NVIDIA Daily Target ETF", "NVDA"),
            "TSLZ": ("T-Rex 2X Inverse Tesla Daily Target ETF", "TSLA"),
            "TSLT": ("T-REX 2X Long Tesla Daily Target ETF", "TSLA"),
            "BULG": ("Leverage Shares 2X Long BULL Daily ETF", "BULL"),
            "MST": ("Defiance Leveraged Long + Income MSTR ETF", "MSTR"),
            "MSOX": ("MSOS Daily Leveraged ETF", "MSOS"),
            "SATG": ("Leverage Shares 2X Long SATS Daily ETF", "SATS"),
            "MQQQ": ("Tradr 2X Long Innovation 100 Monthly ETF", "QQQ"),
            "QQQP": ("Tradr 2X Long Innovation 100 Quarterly ETF", "QQQ"),
            "WLDU": ("2x Long World Daily ETF", "VT"),
            "AIQU": ("MicroSectors Artificial Intelligence (AI) 3X Long Exposure ETN", "AIQ"),
            "BDCX": ("ETRACS Quarterly Pay 1.5x Leveraged MarketVector BDC Liquid Index ETN", "BIZD"),
            "BIB": ("ProShares Ultra Nasdaq Biotechnology", "IBB"),
            "BNKU": ("MicroSectors Big Banks 3X Leveraged Exposure ETN", "KBWB"),
            "BULZ": ("MicroSectors Fang & Innovation 3X Leveraged Exposure ETN", "FNGS"),
            "CEFD": ("ETRACS Monthly Pay 1.5X Leveraged Closed-End Fund Index ETN", "CEFS"),
            "DIG": ("Ultra Energy", "XLE"),
            "DRNL": ("Defiance 2X Daily Long Pure Drone and Aerial Automation ETF", "DRNZ"),
            "EET": ("Ultra MSCI Emerging Markets", "EEM"),
            "EFO": ("Ultra MSCI EAFE", "EFA"),
            "EZJ": ("Ultra MSCI Japan", "EWJ"),
            "FDRX": ("Founder-Led 2X Daily ETF", "FDRS"),
            "FLYU": ("MicroSectors Travel 3X Leveraged Exposure ETN", "PEJ"),
            "FNGO": ("MicroSectors Fang+ 2X Leveraged Exposure ETN", "FNGS"),
            "FNGU": ("MicroSectors Fang+ 3X Leveraged Exposure ETN", "FNGS"),
            "HDLB": (
                "ETRACS Monthly Pay 2xLeveraged US High Dividend Low Volatility ETN Series B",
                "SPHD",
            ),
            "IWDL": ("ETRACS 2x Leveraged US Value Factor TR ETN", "IWD"),
            "IWFL": ("ETRACS 2x Leveraged US Growth Factor TR ETN", "IWF"),
            "IWML": ("ETRACS 2x Leveraged US Size Factor TR ETN", "SIZE"),
            "LTL": ("Ultra Communication Services", "XLC"),
            "MAGX": ("Daily 2X Long Magnificent Seven ETF", "MAGS"),
            "MLPR": ("ETRACS Quarterly Pay 1.5x Leveraged Alerian MLP Index ETN", "AMLP"),
            "MTUL": ("ETRACS 2x Leveraged MSCI US Momentum Factor TR ETN", "MTUM"),
            "MVRL": ("ETRACS Monthly Pay 1.5x Leveraged Mortgage REIT ETN", "REM"),
            "MVV": ("Ultra MidCap400", "MDY"),
            "NRGU": ("MicroSectors Big Oil 3X Leveraged Exposure ETN", "XLE"),
            "OILU": ("MicroSectors Oil & Gas Exploration & Production 3X Leveraged Exposure ETN", "XOP"),
            "PFFL": ("ETRACS Monthly Pay 2xLeveraged Preferred Stock ETN", "PFF"),
            "QPUX": ("Defiance 2X Daily Long Pure Quantum ETF", "QTUM"),
            "QULL": ("ETRACS 2x Leveraged MSCI US Quality Factor TR ETN", "QUAL"),
            "ROM": ("Ultra Technology", "XLK"),
            "RXL": ("Ultra Health Care", "XLV"),
            "SAA": ("Ultra SmallCap600", "IJR"),
            "SCDL": ("ETRACS 2x Leveraged US Dividend Factor TR ETN", "SCHD"),
            "SKYU": ("ProShares Ultra Cloud Computing", "SKYY"),
            "SMHB": (
                "ETRACS Monthly Pay 2xLeveraged US Small Cap High Dividend ETN Series B",
                "DES",
            ),
            "SPCL": ("Defiance Daily 2X Space ETF", "UFO"),
            "TARK": ("Tradr 2X Long Innovation ETF", "ARKK"),
            "UCC": ("Ultra Consumer Discretionary", "XLY"),
            "UCYB": ("ProShares Ultra Nasdaq Cybersecurity", "CIBR"),
            "UBR": ("Ultra MSCI Brazil Capped", "EWZ"),
            "UGE": ("Ultra Consumer Staples", "XLP"),
            "UJB": ("Ultra High Yield", "HYG"),
            "UMDD": ("UltraPro MidCap 400", "MDY"),
            "UPV": ("Ultra FTSE Europe", "VGK"),
            "UPW": ("Ultra Utilities", "XLU"),
            "URE": ("Ultra Real Estate", "IYR"),
            "USD": ("Ultra Semiconductors", "SOXX"),
            "USML": ("ETRACS 2x Leveraged MSCI US Minimum Volatility Factor TR ETN", "USMV"),
            "UVIX": ("2x Long VIX Futures ETF", "VIXY"),
            "UXI": ("Ultra Industrials", "XLI"),
            "UXRP": ("Ultra XRP ETF", "XRP-USD"),
            "UYG": ("Ultra Financials", "XLF"),
            "UYM": ("Ultra Materials", "XLB"),
            "XPP": ("Ultra FTSE China 50", "FXI"),
            "XRPT": ("Volatility Shares Trust XRP 2X ETF", "XRP-USD"),
            "BOIL": ("Ultra Bloomberg Natural Gas", "UNG"),
            "COPZ": ("Defiance Daily Target 2X Long Copper Miners ETF", "COPX"),
            "UCO": ("Ultra Bloomberg Crude Oil", "USO"),
            "UCOP": ("Ultra Copper K-1 Free ETF", "CPER"),
            "ULE": ("Ultra Euro", "FXE"),
            "UPAL": ("Ultra Palladium K-1 Free ETF", "PALL"),
            "UPLT": ("Ultra Platinum K-1 Free ETF", "PPLT"),
            "WTIU": ("MicroSectors Energy 3X Leveraged Exposure ETN", "XLE"),
            "YCL": ("Ultra Yen", "FXY"),
            "AVAZ": ("2x Avalanche ETF", "AVAX-USD"),
            "CHNU": ("2x Chainlink ETF", "LINK-USD"),
            "CRDX": ("2x Cardano ETF", "ADA-USD"),
            "STLU": ("2x Stellar ETF", "XLM-USD"),
            "SUIL": ("2x Sui ETF", "SUI20947-USD"),
            "TXXD": ("21Shares 2x Long Dogecoin ETF", "DOGE-USD"),
            "TXXH": ("21Shares 2x Long HYPE ETF", "HYPE32196-USD"),
        }

        for asset_symbol, (name, expected) in cases.items():
            with self.subTest(asset_symbol=asset_symbol):
                mapping = infer_rsi_mapping(asset_symbol, name)
                self.assertEqual(mapping.rsi_symbol, expected)
                self.assertEqual(mapping.confidence, "curated")
                self.assertEqual(mapping.mapping_source, "symbol_override")

    def test_curated_benchmark_and_asset_proxy_mappings_win_before_regex(self) -> None:
        cases = {
            "UPRO": ("ProShares UltraPro S&P500", "SPY"),
            "SSO": ("Ultra S&P500", "SPY"),
            "URSP": ("Ultra S&P 500 Equal Weight", "RSP"),
            "UGL": ("Ultra Gold", "GLD"),
            "BITX": ("2x Bitcoin ETF", "BTC-USD"),
            "DOGU": ("2x Dogecoin ETF", "DOGE-USD"),
            "AVAU": ("2x Avalanche ETF", "AVAX-USD"),
            "LNKU": ("2x Chainlink ETF", "LINK-USD"),
            "CARD": ("2x Cardano ETF", "ADA-USD"),
            "ETHU": ("2x Ether ETF", "ETH-USD"),
            "SOLT": ("2x Solana ETF", "SOL-USD"),
            "XRPU": ("2x XRP ETF", "XRP-USD"),
            "SUIX": ("2x Sui ETF", "SUI20947-USD"),
            "XLMU": ("2x Stellar ETF", "XLM-USD"),
            "HYPX": ("2x Hyperliquid ETF", "HYPE32196-USD"),
        }

        for asset_symbol, (name, expected) in cases.items():
            with self.subTest(asset_symbol=asset_symbol):
                mapping = infer_rsi_mapping(asset_symbol, name)
                self.assertEqual(mapping.rsi_symbol, expected)
                self.assertEqual(mapping.confidence, "curated")

    def test_common_words_that_are_tickers_do_not_win_generic_rsi_inference(self) -> None:
        cases = {
            "PAYT": "ETRACS Quarterly Pay 1.5x Leveraged Example Index ETN",
            "SERB": "ETRACS High Dividend ETN Series B 2X Leveraged",
            "MSCT": "Ultra MSCI Example Index",
            "HIGHX": "Ultra High Yield",
            "REALX": "Ultra Real Estate",
        }
        known_symbols = {"PAYT", "SERB", "MSCT", "HIGHX", "REALX", "PAY", "B", "MSCI", "HIGH", "REAL"}

        for asset_symbol, name in cases.items():
            with self.subTest(asset_symbol=asset_symbol):
                mapping = infer_rsi_mapping(asset_symbol, name, known_symbols=known_symbols)
                self.assertEqual(mapping.rsi_symbol, asset_symbol)
                self.assertEqual(mapping.confidence, "fallback_to_self")

    def test_resolved_basket_products_use_explicit_rsi_mappings(self) -> None:
        cases = {
            "BEGS": (
                "Rareview 2X Bull Cryptocurrency & Precious Metals ETF",
                "BEGS",
                "fallback_to_self",
                "self_fallback_override",
            ),
            "DRNL": (
                "Defiance 2X Daily Long Pure Drone and Aerial Automation ETF",
                "DRNZ",
                "curated",
                "symbol_override",
            ),
            "FDRX": (
                "Founder-Led 2X Daily ETF",
                "FDRS",
                "curated",
                "symbol_override",
            ),
            "FLYU": (
                "MicroSectors Travel 3X Leveraged Exposure ETN",
                "PEJ",
                "curated",
                "symbol_override",
            ),
        }

        for asset_symbol, (name, expected, confidence, mapping_source) in cases.items():
            with self.subTest(asset_symbol=asset_symbol):
                mapping = infer_rsi_mapping(asset_symbol, name, known_symbols={asset_symbol})
                self.assertEqual(mapping.rsi_symbol, expected)
                self.assertEqual(mapping.confidence, confidence)
                self.assertEqual(mapping.mapping_source, mapping_source)

    def test_unresolved_single_stock_style_mapping_needs_review(self) -> None:
        cases = [
            (
                "FOOU",
                "T-REX 2X Long ExampleCorp Daily Target ETF",
                "ETF (Tuttle Capital)",
            ),
            (
                "FOUR",
                "4X Long ExampleCorp Daily ETF",
                "ETF (Example Issuer)",
            ),
            (
                "FRAC",
                "1.25X Long ExampleCorp Daily ETF",
                "ETF (Example Issuer)",
            ),
            (
                "P150",
                "150% Long ExampleCorp Daily ETF",
                "ETF (Example Issuer)",
            ),
            (
                "TXGU",
                "10x Genomics 2X Long Daily ETF",
                "ETF (Example Issuer)",
            ),
            (
                "AAPU",
                "Direxion Daily AAPL Bull 2X ETF",
                "ETF (Direxion)",
            ),
            (
                "FIVE",
                "Direxion Daily NVDA Bull 5X Shares",
                "ETF (Direxion)",
            ),
            (
                "FOOD",
                "T-REX 2X Inverse ExampleCorp Daily Target ETF",
                "ETF (Example Issuer)",
            ),
            (
                "FOOS",
                "2X Short ExampleCorp Daily ETF",
                "ETF (Example Issuer)",
            ),
            (
                "FOOB",
                "Direxion Daily AAPL Bear 2X Shares",
                "ETF (Direxion)",
            ),
        ]

        for asset_symbol, name, fund_type in cases:
            with self.subTest(asset_symbol=asset_symbol):
                mapping = infer_rsi_mapping(
                    asset_symbol,
                    name,
                    known_symbols={asset_symbol},
                    fund_type=fund_type,
                )

                self.assertEqual(mapping.rsi_symbol, asset_symbol)
                self.assertEqual(mapping.confidence, "needs_review")
                self.assertEqual(mapping.mapping_source, "unresolved_single_stock")

    def test_spacex_products_map_to_spcx_before_regex(self) -> None:
        cases = {
            "SPAL": ("GraniteShares 2x Long SpaceX Daily ETF", "ETF (GraniteShares)"),
            "SPAX": ("T-REX 2X Long SpaceX Daily Target ETF", "ETF (Tuttle Capital)"),
            "SPCF": ("Ultra SpaceX", "ETF (ProShares)"),
            "SPCM": ("Tradr 2X Long SpaceX Daily ETF", "ETF (Tradr)"),
        }

        for asset_symbol, (name, fund_type) in cases.items():
            with self.subTest(asset_symbol=asset_symbol):
                mapping = infer_rsi_mapping(asset_symbol, name, fund_type=fund_type)
                self.assertEqual(mapping.rsi_symbol, "SPCX")
                self.assertEqual(mapping.underlying_name, "Space Exploration Technologies Corp. Class A")
                self.assertEqual(mapping.confidence, "curated")
                self.assertEqual(mapping.mapping_source, "name_proxy")

    def test_saved_universe_table_maps_spacex_products_to_spcx(self) -> None:
        universe = build_nasdaq_universe_table(
            pd.DataFrame(
                [
                    {
                        "symbol": "SPAL",
                        "name": "GraniteShares 2x Long SpaceX Daily ETF",
                        "fund_type": "ETF (GraniteShares)",
                        "source": "GraniteShares issuer table",
                    },
                    {
                        "symbol": "TQQQ",
                        "name": "ProShares UltraPro QQQ",
                        "fund_type": "ETF",
                        "source": "Nasdaq ETF definitions",
                    },
                ]
            )
        )

        spal = universe.loc[universe["symbol"] == "SPAL"].iloc[0]
        self.assertEqual(spal["rsi_symbol"], "SPCX")
        self.assertEqual(spal["underlying_name"], "Space Exploration Technologies Corp. Class A")
        self.assertEqual(spal["confidence"], "curated")
        self.assertNotIn("SPACEX", universe["rsi_symbol"].tolist())

    def test_saved_universe_table_validates_generic_rsi_mappings(self) -> None:
        universe = build_nasdaq_universe_table(
            pd.DataFrame(
                [
                    {"symbol": "ENGY", "name": "Ultra Energy", "fund_type": "ETF"},
                    {"symbol": "QQQ", "name": "Invesco QQQ Trust", "fund_type": "ETF"},
                    {"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"},
                ]
            )
        )

        dig = universe.loc[universe["symbol"] == "ENGY"].iloc[0]
        tqqq = universe.loc[universe["symbol"] == "TQQQ"].iloc[0]
        self.assertEqual(dig["rsi_symbol"], "ENGY")
        self.assertEqual(dig["confidence"], "fallback_to_self")
        self.assertEqual(tqqq["rsi_symbol"], "QQQ")
        self.assertNotIn("ENERGY", universe["rsi_symbol"].tolist())

    def test_long_duration_names_are_not_leveraged(self) -> None:
        false_positive_names = [
            "Vanguard Long-Term Corporate Bond ETF",
            "Baillie Gifford Long Term Global Growth ETF",
            "Innovator U.S. Equity Ultra Buffer ETF",
            "ProShares UltraShort Term Bond ETF",
            "ProShares Ultra Short-Term Bond ETF",
            "ProShares Ultra-Short-Term Bond ETF",
            "ProShares UltraShort-Term Bond ETF",
            "YieldMax Ultra Option Income Strategy ETF",
        ]
        for name in false_positive_names:
            with self.subTest(name=name):
                self.assertEqual(infer_leverage_and_direction(name), (None, None))
                self.assertFalse(leveraged_name_filter(name))
                self.assertFalse(is_long_leveraged_name(name))

        self.assertFalse(is_long_leveraged_name("MicroSectors FANG+ 1X Long Exposure ETN"))
        self.assertTrue(is_long_leveraged_name("GraniteShares 2x Long NVDA Daily ETF"))

    def test_explicit_leveraged_duration_names_are_not_suppressed(self) -> None:
        cases = [
            "Example 2X Long Long-Term Treasury ETF",
            "Example 2X Long-Term Treasury Bull ETF",
            "Example 2X Long Municipal Bond ETF",
        ]

        for name in cases:
            with self.subTest(name=name):
                self.assertEqual(infer_leverage_and_direction(name), (2.0, "long"))
                self.assertTrue(leveraged_name_filter(name))
                self.assertTrue(is_long_leveraged_name(name))

    def test_short_term_underlying_names_are_not_inverse(self) -> None:
        long_vix = "ProShares Ultra VIX Short-Term Futures ETF"

        self.assertEqual(infer_leverage_and_direction(long_vix), (2.0, "long"))
        self.assertTrue(is_long_leveraged_name(long_vix))

        self.assertEqual(
            infer_leverage_and_direction("ProShares Short VIX Short-Term Futures ETF"),
            (None, "inverse"),
        )
        self.assertFalse(is_long_leveraged_name("ProShares Short VIX Short-Term Futures ETF"))

    def test_signed_inverse_multiples_are_not_long_leveraged(self) -> None:
        cases = {
            "-2X Daily MSTR ETF": 2.0,
            "-3X Tesla Daily ETF": 3.0,
            "-4X Daily SPY ETF": 4.0,
            "MAX S&P 500 -5X Leveraged ETN": 5.0,
            "Example -1.5X Daily QQQ ETF": 1.5,
            "Example -150% Daily QQQ ETF": 1.5,
        }

        for name, leverage in cases.items():
            with self.subTest(name=name):
                self.assertEqual(infer_leverage_and_direction(name), (leverage, "inverse"))
                self.assertFalse(is_long_leveraged_name(name))
                self.assertTrue(is_short_leveraged_name(name))

    def test_short_leveraged_workflow_selection_uses_inverse_products(self) -> None:
        products = pd.DataFrame(
            [
                {"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"},
                {"symbol": "SQQQ", "name": "ProShares UltraPro Short QQQ", "fund_type": "ETF"},
                {"symbol": "SVIX", "name": "ProShares Short VIX Short-Term Futures ETF", "fund_type": "ETF"},
            ]
        )

        short_universe = select_short_workflow_universe(products)

        self.assertEqual(short_universe["symbol"].tolist(), ["SQQQ"])
        self.assertTrue(is_short_leveraged_name("ProShares UltraPro Short QQQ"))
        self.assertFalse(is_short_leveraged_name("ProShares Short VIX Short-Term Futures ETF"))

    def test_percent_based_leverage_names_are_classified(self) -> None:
        self.assertEqual(
            infer_leverage_and_direction("ProShares 200% Long QQQ ETF"),
            (2.0, "long"),
        )
        self.assertEqual(
            infer_leverage_and_direction("ProShares 150% Long QQQ ETF"),
            (1.5, "long"),
        )
        self.assertEqual(
            infer_leverage_and_direction("ProShares 125% Long QQQ ETF"),
            (1.25, "long"),
        )
        self.assertEqual(
            infer_leverage_and_direction("ProShares 300% Long QQQ ETF"),
            (3.0, "long"),
        )
        self.assertEqual(
            infer_leverage_and_direction("ProShares 400% Long SPY ETF"),
            (4.0, "long"),
        )
        self.assertEqual(
            infer_leverage_and_direction("ProShares 500% Long QQQ ETF"),
            (5.0, "long"),
        )
        self.assertTrue(is_long_leveraged_name("ProShares 150% Long QQQ ETF"))
        self.assertTrue(is_long_leveraged_name("ProShares 125% Long QQQ ETF"))
        self.assertTrue(is_long_leveraged_name("ProShares 200% Long QQQ ETF"))
        self.assertTrue(is_long_leveraged_name("ProShares 300% Long QQQ ETF"))
        self.assertTrue(is_long_leveraged_name("ProShares 400% Long SPY ETF"))
        self.assertTrue(is_long_leveraged_name("ProShares 500% Long QQQ ETF"))
        self.assertFalse(is_long_leveraged_name("ProShares 100% Long QQQ ETF"))
        self.assertFalse(is_long_leveraged_name("ProShares 50% Leveraged QQQ ETF"))

        self.assertEqual(
            infer_leverage_and_direction("Example 500% Short QQQ ETF"),
            (5.0, "inverse"),
        )
        self.assertFalse(is_long_leveraged_name("Example 500% Short QQQ ETF"))

    def test_numeric_multiples_above_supported_cap_are_not_leveraged(self) -> None:
        for name in [
            "10X Long QQQ ETF",
            "6X Long QQQ ETF",
            "5.5X Long QQQ ETF",
            "MAX S&P 500 10X Leveraged ETN",
            "MAX S&P 500 -10X Leveraged ETN",
            "S&P 500 6X Daily ETF",
            "600% Leveraged QQQ ETF",
            "1000% Leveraged QQQ ETF",
            "MAX S&P 500 1000% Leveraged ETN",
            "10X Long QQQ ETF 2X",
            "600% 2X Long QQQ ETF",
        ]:
            with self.subTest(name=name):
                leverage, _direction = infer_leverage_and_direction(name)
                self.assertIsNone(leverage)
                self.assertFalse(leveraged_name_filter(name))
                self.assertFalse(is_long_leveraged_name(name))

    def test_company_name_multiplier_does_not_hide_product_leverage(self) -> None:
        name = "10x Genomics 2X Long Daily ETF"

        self.assertEqual(infer_leverage_and_direction(name), (2.0, "long"))
        self.assertTrue(leveraged_name_filter(name))
        self.assertTrue(is_long_leveraged_name(name))

    def test_embedded_company_name_multipliers_are_not_leverage(self) -> None:
        for name in [
            "8X8 INC /DE/",
            "10x Genomics, Inc.",
            "V2X, Inc.",
            "Ultragenyx Pharmaceutical Inc.",
        ]:
            self.assertEqual(infer_leverage_and_direction(name), (None, None))
            self.assertFalse(leveraged_name_filter(name))
            self.assertFalse(is_long_leveraged_name(name))

    def test_bare_leveraged_names_need_known_leverage_to_be_long(self) -> None:
        self.assertEqual(infer_leverage_and_direction("Example Leveraged ETF"), (None, None))
        self.assertTrue(leveraged_name_filter("Example Leveraged ETF"))
        self.assertFalse(is_long_leveraged_name("Example Leveraged ETF"))

        self.assertEqual(infer_leverage_and_direction("Example Daily Leveraged Exposure ETF"), (None, "long"))
        self.assertTrue(leveraged_name_filter("Example Daily Leveraged Exposure ETF"))
        self.assertFalse(is_long_leveraged_name("Example Daily Leveraged Exposure ETF"))

    def test_msox_has_curated_leverage_and_rsi_underlying(self) -> None:
        name = "MSOS Daily Leveraged ETF"
        mapping = infer_rsi_mapping("MSOX", name, known_symbols={"MSOX", "MSOS"})

        self.assertEqual(infer_leverage_and_direction(name), (2.0, "long"))
        self.assertTrue(is_long_leveraged_name(name))
        self.assertEqual(mapping.rsi_symbol, "MSOS")
        self.assertEqual(mapping.confidence, "curated")
        self.assertEqual(mapping.mapping_source, "symbol_override")

    def test_mst_has_curated_leverage_and_rsi_underlying(self) -> None:
        name = "Defiance Leveraged Long + Income MSTR ETF"
        mapping = infer_rsi_mapping("MST", name, known_symbols={"MST", "MSTR"})

        self.assertEqual(infer_leverage_and_direction(name), (1.75, "long"))
        self.assertTrue(is_long_leveraged_name(name))
        self.assertEqual(mapping.rsi_symbol, "MSTR")
        self.assertEqual(mapping.confidence, "curated")
        self.assertEqual(mapping.mapping_source, "symbol_override")

    def test_extra_issuer_sources_are_registered(self) -> None:
        issuer_names = {source.name for source in ISSUER_UNIVERSE_SOURCES}

        for issuer in [
            "AdvisorShares",
            "AXS Investments",
            "Kurv",
            "Innovator",
            "Tuttle Capital",
            "Leverage Shares",
            "YieldMax",
            "Tidal",
            "Roundhill",
            "Themes",
            "Simplify",
        ]:
            self.assertIn(issuer, issuer_names)

    def test_etn_sources_are_registered(self) -> None:
        source_names = {source.name for source in WORKFLOW_ETN_SOURCES}

        self.assertIn("MicroSectors", source_names)
        self.assertIn("UBS ETRACS", source_names)

    def test_etn_source_status_uses_registered_source_type(self) -> None:
        with (
            patch(
                "leveraged_trader.universe.WORKFLOW_ETN_SOURCES",
                [
                    UniverseSource(
                        "Test ETN",
                        "https://etn.test",
                        source_type="etn_issuer",
                    )
                ],
            ),
            patch("leveraged_trader.universe.requests.get", side_effect=RuntimeError("offline")),
        ):
            etn_universe = load_etn_universe()

        status = etn_universe.attrs["workflow_source_status"]
        self.assertEqual(status[0]["source_type"], "etn_issuer")

    def test_audit_sources_are_registered_without_being_authoritative(self) -> None:
        source_names = {source.name for source in AUDIT_UNIVERSE_SOURCES}

        for source_name in [
            "NYSE exchange-traded products directory",
            "Nasdaq funds/ETFs directory",
            "Cboe listed products",
            "ETFdb leveraged ETF directory",
            "VettaFi ETF database",
            "ETF.com ETF finder",
            "SEC EDGAR company ticker registry",
            "SEC EDGAR exchange ticker registry",
            "SEC EDGAR mutual fund ticker registry",
            "SEC EDGAR full-text search",
        ]:
            self.assertIn(source_name, source_names)

        self.assertTrue(
            all(source.source_type != "issuer" for source in AUDIT_UNIVERSE_SOURCES)
        )

    def test_microsectors_parser_builds_etn_rows(self) -> None:
        source = UniverseSource(
            "MicroSectors",
            "https://example.test",
            source_type="etn_issuer",
            parser="microsectors_html",
        )
        html = """
            <div class="item"><a href="/fang">
                <div class="suite-name">Fang+</div>
                <div class="products">
                    <div class="product">
                        <div class="product-symbol">FNGU</div>
                        <div class="product-description">3X Leveraged Exposure</div>
                    </div>
                    <div class="product">
                        <div class="product-symbol">FNGS</div>
                        <div class="product-description">1X Long Exposure</div>
                    </div>
                    <div class="product">
                        <div class="product-symbol">FNGD</div>
                        <div class="product-description">-3X Inverse Leveraged Exposure</div>
                    </div>
                </div>
            </a></div>
        """

        out = _microsectors_html_to_universe(html, source)

        self.assertEqual(out["symbol"].tolist(), ["FNGU", "FNGD"])
        self.assertTrue(out["fund_type"].str.startswith("ETN").all())

    def test_etracs_parser_extracts_ticker_from_hidden_url_cell(self) -> None:
        source = UniverseSource(
            "UBS ETRACS",
            "https://example.test",
            source_type="etn_issuer",
            parser="etracs_leverage_table",
        )
        html = """
            <table>
                <tr><th>Ticker symbol</th><th>Name</th><th>Leverage</th></tr>
                <tr>
                    <td><div>/product/detail/index/ussymbol/BDCX</div><span>BDCX</span></td>
                    <td>ETRACS Quarterly Pay 1.5x Leveraged MarketVector BDC Liquid Index ETN</td>
                    <td>1.50x</td>
                </tr>
                <tr>
                    <td><div>/product/detail/index/ussymbol/BDCY</div><span>BDCY</span></td>
                    <td>ETRACS Quarterly Pay MarketVector BDC Liquid Index ETN</td>
                    <td>1.50x</td>
                </tr>
                <tr>
                    <td><div>/product/detail/index/ussymbol/BDCZ</div><span>BDCZ</span></td>
                    <td>ETRACS Quarterly Pay MarketVector BDC Liquid Index ETN</td>
                    <td>-1.50x</td>
                </tr>
                <tr>
                    <td><div>/product/detail/index/ussymbol/PLAIN</div><span>PLAIN</span></td>
                    <td>ETRACS Plain Index ETN</td>
                    <td>--</td>
                </tr>
            </table>
        """

        out = _etracs_leverage_table_to_universe(html, source)

        self.assertEqual(out["symbol"].tolist(), ["BDCX", "BDCY", "BDCZ"])
        self.assertEqual(
            out["fund_type"].tolist(),
            ["ETN (UBS ETRACS)", "ETN (UBS ETRACS)", "ETN (UBS ETRACS)"],
        )
        self.assertTrue(
            out.loc[out["symbol"].eq("BDCY"), "name"].item().endswith("1.5X Leveraged")
        )
        self.assertTrue(
            out.loc[out["symbol"].eq("BDCZ"), "name"].item().endswith("1.5X Inverse Leveraged")
        )
        self.assertFalse(is_long_leveraged_name(out.loc[out["symbol"].eq("BDCZ"), "name"].item()))

    def test_sec_company_tickers_parser_builds_audit_rows(self) -> None:
        source = UniverseSource(
            "SEC EDGAR company ticker registry",
            "https://example.test",
            source_type="filing_audit",
            parser="sec_company_tickers",
        )
        json_text = """
            {
                "0": {"cik_str": 927971, "ticker": "FNGU", "title": "BANK OF MONTREAL /CAN/"},
                "1": {"cik_str": 1114446, "ticker": "BDCX", "title": "UBS AG"}
            }
        """

        out = _sec_company_tickers_to_universe(json_text, source)

        self.assertEqual(out["symbol"].tolist(), ["FNGU", "BDCX"])
        self.assertEqual(out["source"].tolist(), ["SEC EDGAR company ticker registry audit source"] * 2)

    def test_sec_exchange_tickers_parser_builds_audit_rows(self) -> None:
        source = UniverseSource(
            "SEC EDGAR exchange ticker registry",
            "https://example.test",
            source_type="filing_audit",
            parser="sec_exchange_tickers",
        )
        json_text = """
            {
                "fields": ["cik", "name", "ticker", "exchange"],
                "data": [
                    [123, "ProShares UltraPro QQQ", "TQQQ", "Nasdaq"],
                    [456, "Plain Company", "PLAIN", "NYSE"]
                ]
            }
        """

        out = _sec_exchange_tickers_to_universe(json_text, source)

        self.assertEqual(out["symbol"].tolist(), ["TQQQ", "PLAIN"])
        self.assertEqual(out["name"].tolist(), ["ProShares UltraPro QQQ", "Plain Company"])
        self.assertEqual(out["source"].tolist(), ["SEC EDGAR exchange ticker registry audit source"] * 2)

    def test_sec_mutual_fund_tickers_parser_builds_symbol_only_audit_rows(self) -> None:
        source = UniverseSource(
            "SEC EDGAR mutual fund ticker registry",
            "https://example.test",
            source_type="filing_audit",
            parser="sec_mutual_fund_tickers",
        )
        json_text = """
            {
                "fields": ["cik", "seriesId", "classId", "symbol"],
                "data": [
                    [123, "S0001", "C0001", "TQQQ"],
                    [456, "S0002", "C0002", null],
                    [789, "S0003", "C0003", "PLAIN"]
                ]
            }
        """

        out = _sec_mutual_fund_tickers_to_universe(json_text, source)

        self.assertEqual(out["symbol"].tolist(), ["TQQQ", "PLAIN"])
        self.assertEqual(out["name"].tolist(), ["TQQQ", "PLAIN"])
        self.assertEqual(out["fund_type"].tolist(), ["SEC MF (SEC EDGAR mutual fund ticker registry)"] * 2)

    def test_sec_entity_audit_metadata_requires_product_context(self) -> None:
        source = UniverseSource(
            "SEC EDGAR company ticker registry",
            "https://example.test",
            source_type="filing_audit",
            parser="sec_company_tickers",
        )
        rows = pd.DataFrame(
            [
                {
                    "symbol": "EGHT",
                    "name": "8X8 INC /DE/",
                    "fund_type": "SEC (SEC EDGAR company ticker registry)",
                    "source": "SEC EDGAR company ticker registry audit source",
                },
                {
                    "symbol": "UCTT",
                    "name": "Ultra Clean Holdings, Inc.",
                    "fund_type": "SEC (SEC EDGAR company ticker registry)",
                    "source": "SEC EDGAR company ticker registry audit source",
                },
                {
                    "symbol": "UPRO",
                    "name": "ProShares UltraPro S&P500",
                    "fund_type": "SEC (SEC EDGAR company ticker registry)",
                    "source": "SEC EDGAR company ticker registry audit source",
                },
                {
                    "symbol": "XXXX",
                    "name": "MAX S&P 500 4X Leveraged Exchange Traded Note",
                    "fund_type": "SEC (SEC EDGAR company ticker registry)",
                    "source": "SEC EDGAR company ticker registry audit source",
                },
            ]
        )

        out = _with_audit_metadata(rows, source)

        self.assertEqual(out.loc[out["symbol"] == "EGHT", "is_leveraged_candidate"].item(), False)
        self.assertEqual(out.loc[out["symbol"] == "UCTT", "is_leveraged_candidate"].item(), False)
        self.assertEqual(out.loc[out["symbol"] == "UPRO", "is_leveraged_candidate"].item(), True)
        self.assertEqual(out.loc[out["symbol"] == "XXXX", "leverage"].item(), 4.0)

    def test_webflow_card_parser_finds_static_fund_rows(self) -> None:
        html = """
            <div class="tag is-ticker on-dark-bg">MSTU</div></div>
            <div class="grid_table_cell">
                <div aria-hidden="true" class="u-weight-medium u-text-balance">
                    T-REX 2X Long MSTR Daily Target ETF
                </div>
            </div>
            <div fs-cmssort-field="IDENTIFIER" class="text-weight-xbold">TSLZ</div>
            <div role="cell" class="table3_column">
                <div fs-cmssort-field="IDENTIFIER">T-REX 2X Inverse Tesla Daily Target ETF</div>
            </div>
            <a href="/etf/aapy" class="nav_dropdown_link w-inline-block">
                <div class="u-display-inline">AAPY</div>
                <div class="nav_dropdown_link_caption">Kurv Yield Premium Strategy Apple ETF</div>
            </a>
        """

        out = _html_cards_to_universe(
            html,
            "Example",
            source_label="Example issuer table",
            require_leveraged=True,
        )

        self.assertEqual(out["symbol"].tolist(), ["MSTU", "TSLZ"])

    def test_defiance_json_parser_builds_fund_rows(self) -> None:
        source = UniverseSource(
            "Defiance",
            "https://example.test",
            source_type="issuer_etf",
            parser="defiance_json",
        )
        json_text = """
            [
                {"ticker": "AMA", "name": "Defiance Daily Target 2X Long AMAT ETF"},
                {"ticker": "AIX", "name": "Defiance US 100 Tech AI Moat ETF"},
                {"ticker": "SPCQ", "name": "Defiance Daily Target 2X Short SPCX ETF"}
            ]
        """

        out = _defiance_json_to_universe(json_text, source)

        self.assertEqual(out["symbol"].tolist(), ["AMA", "SPCQ"])

    def test_script_ticker_name_parser_handles_name_before_and_after_ticker(self) -> None:
        source = UniverseSource(
            "Leverage Shares",
            "https://example.test",
            source_type="issuer_etf",
            parser="js_ticker_name",
        )
        html = """
            <script>
            window.featuredEtfData = [
                {"name":"Leverage Shares 2x Long NVDA Daily ETF","ticker":"NVDG"},
                { ticker: 'AALG', fund: " 2x Long AAL Daily ETF" },
                { ticker: 'QUOT', fund: "2x Long Bob\\"s Daily ETF" },
                { ticker: 'APOS', fund: '2x Long Bob\\'s Daily ETF' },
                { ticker: 'PLAIN', fund: "Plain Equity ETF" }
            ];
            </script>
        """

        out = _js_ticker_name_to_universe(html, source)

        self.assertEqual(out["symbol"].tolist(), ["NVDG", "AALG", "QUOT", "APOS"])
        self.assertIn("2x Long NVDA", out.loc[out["symbol"] == "NVDG", "name"].item())
        self.assertEqual(out.loc[out["symbol"] == "QUOT", "name"].item(), '2x Long Bob"s Daily ETF')
        self.assertEqual(out.loc[out["symbol"] == "APOS", "name"].item(), "2x Long Bob's Daily ETF")

    def test_graniteshares_parser_extracts_static_table_cells(self) -> None:
        source = UniverseSource(
            "GraniteShares",
            "https://example.test",
            source_type="issuer_etf",
            parser="graniteshares_html",
        )
        html = """
            <span class="etf-table-cell--ticker__symbol">AAPB</span>
            <span class="etf-table-cell--name-title Body1">
                GraniteShares 2x Long AAPL Daily ETF
            </span>
            <span class="etf-table-cell--ticker__symbol">PLAIN</span>
            <span class="etf-table-cell--name-title Body1">Plain Equity ETF</span>
        """

        out = _graniteshares_html_to_universe(html, source)

        self.assertEqual(out["symbol"].tolist(), ["AAPB"])

    def test_rex_menu_parser_builds_generated_names(self) -> None:
        source = UniverseSource(
            "REX Shares",
            "https://example.test",
            source_type="issuer_etf",
            parser="rex_menu_html",
        )
        html = """
            <a href="https://www.rexshares.com/mstu/">MSTU | +2X Daily MSTR</a>
            <a href="https://www.rexshares.com/mstz/">MSTZ | -2X Daily MSTR</a>
        """

        out = _rex_menu_html_to_universe(html, source)

        self.assertEqual(out["symbol"].tolist(), ["MSTU", "MSTZ"])
        self.assertIn("Inverse", out.loc[out["symbol"] == "MSTZ", "name"].item())

    def test_volatilityshares_parser_extracts_list_items(self) -> None:
        source = UniverseSource(
            "Volatility Shares",
            "https://example.test",
            source_type="issuer_etf",
            parser="volatilityshares_html",
        )
        html = """
            <li><a href="/bitx"><h4>BITX</h4><p>2x Bitcoin ETF</p></a></li>
            <li><a href="/plain"><h4>PLAIN</h4><p>Plain Bitcoin ETF</p></a></li>
        """

        out = _volatilityshares_html_to_universe(html, source)

        self.assertEqual(out["symbol"].tolist(), ["BITX"])

    def test_issuer_table_parser_skips_invalid_symbol_cells(self) -> None:
        table = pd.DataFrame(
            [
                {"Ticker": float("nan"), "Fund Name": "Missing Symbol 2X Long ETF"},
                {"Ticker": 123.0, "Fund Name": "Numeric Symbol 2X Long ETF"},
                {"Ticker": "GOOD", "Fund Name": "Example 2X Long GOOD Daily ETF"},
                {"Ticker": "PLAIN", "Fund Name": "Plain Equity ETF"},
            ]
        )

        out = _issuer_table_to_universe(table, "Example")

        self.assertEqual(out["symbol"].tolist(), ["GOOD"])
        self.assertEqual(out["source"].tolist(), ["Example issuer table"])

    def test_audit_report_includes_leveraged_candidates_missing_from_merged_universe(self) -> None:
        audit_rows = pd.DataFrame(
            [
                {
                    "symbol": "MISSING",
                    "name": "Example 2X Long MISSING Daily ETF",
                    "fund_type": "ETF (Audit)",
                    "source": "Audit source",
                    "audit_source_type": "third_party_audit",
                    "source_url": "https://example.test",
                    "is_leveraged_candidate": True,
                    "is_long_leveraged_candidate": True,
                    "leverage": 2.0,
                    "direction": "long",
                },
                {
                    "symbol": "TQQQ",
                    "name": "ProShares UltraPro QQQ",
                    "fund_type": "ETF (Audit)",
                    "source": "Audit source",
                    "audit_source_type": "third_party_audit",
                    "source_url": "https://example.test",
                    "is_leveraged_candidate": True,
                    "is_long_leveraged_candidate": True,
                    "leverage": 3.0,
                    "direction": "long",
                },
                {
                    "symbol": "SHORTY",
                    "name": "Example 2X Short Missing ETF",
                    "fund_type": "ETF (Audit)",
                    "source": "Audit source",
                    "audit_source_type": "third_party_audit",
                    "source_url": "https://example.test",
                    "is_leveraged_candidate": True,
                    "is_long_leveraged_candidate": False,
                    "leverage": 2.0,
                    "direction": "inverse",
                },
                {
                    "symbol": "PLAIN",
                    "name": "Plain Equity ETF",
                    "fund_type": "ETF (Audit)",
                    "source": "Audit source",
                    "audit_source_type": "third_party_audit",
                    "source_url": "https://example.test",
                    "is_leveraged_candidate": False,
                    "is_long_leveraged_candidate": False,
                    "leverage": None,
                    "direction": None,
                },
            ]
        )
        merged = pd.DataFrame([{"symbol": "TQQQ"}])
        workflow = pd.DataFrame([{"symbol": "TQQQ"}])

        report = build_universe_audit_report(audit_rows, merged, workflow)

        self.assertEqual(report["symbol"].tolist(), ["MISSING", "SHORTY"])
        short_row = report.loc[report["symbol"].eq("SHORTY")].iloc[0]
        self.assertIn("inverse leveraged-looking", short_row["audit_reason"])

    def test_inverse_self_rsi_fallback_is_excluded_for_review(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [
                {"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"},
                {"symbol": "MYST", "name": "Example 2X Inverse Mystery Index ETF", "fund_type": "ETF"},
            ]
        )
        empty_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        empty_rows.attrs["workflow_source_status"] = []

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value={"TQQQ", "QQQ", "MYST"}),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite"),
        ):
            groups = determine_workflow_asset_groups(UniverseConfig(sqlite_db_path="state.sqlite"))

        self.assertTrue(groups["short"].empty)
        review = groups["long"].attrs["rsi_mapping_review"]
        self.assertEqual([row["symbol"] for row in review], ["MYST"])
        self.assertEqual(review[0]["confidence"], "needs_review")
        self.assertIn("requires an underlying RSI proxy", review[0]["mapping_reason"])

    @patch("leveraged_trader.universe.requests.get")
    def test_load_current_etf_universe_filters_exclusions_and_normalizes_brkb(self, mock_get: Mock) -> None:
        response = Mock()
        response.text = """
            <table>
                <tr><th>Symbol</th><th>Fund Name</th><th>Fund Type</th></tr>
                <tr><td>NASDAQ</td><td>Bad Nasdaq Row</td><td>ETF</td></tr>
                <tr><td>BGGG</td><td>Baillie Gifford Long Term Global Growth ETF</td><td>ETF</td></tr>
                <tr><td>BRKB</td><td>Berkshire Test Row</td><td>ETF</td></tr>
                <tr><td>TQQQ</td><td>ProShares UltraPro QQQ</td><td>ETF</td></tr>
            </table>
        """
        mock_get.return_value = response

        universe = load_current_etf_universe()

        self.assertNotIn("NASDAQ", universe["symbol"].tolist())
        self.assertNotIn("BGGG", universe["symbol"].tolist())
        self.assertIn("BRK-B", universe["symbol"].tolist())
        self.assertIn("TQQQ", universe["symbol"].tolist())

    @patch("builtins.print")
    @patch(
        "leveraged_trader.universe.load_active_listed_symbols",
        return_value={"TQQQ", "BRKU", "BRK-B", "QQQ", "EXTRA", "BDCX"},
    )
    @patch("leveraged_trader.universe.load_audit_universe_sources")
    @patch("leveraged_trader.universe.load_etn_universe")
    @patch("leveraged_trader.universe.load_issuer_etf_universe")
    @patch("leveraged_trader.universe.save_table_to_sqlite")
    @patch("leveraged_trader.universe.load_current_etf_universe")
    def test_determine_workflow_assets_returns_universe_metadata_without_printing(
        self,
        mock_load_universe: Mock,
        mock_save_table: Mock,
        mock_load_issuer_universe: Mock,
        mock_load_etn_universe: Mock,
        mock_load_audit_sources: Mock,
        _mock_load_active_symbols: Mock,
        mock_print: Mock,
    ) -> None:
        mock_load_universe.return_value = pd.DataFrame(
            [
                {
                    "symbol": "TQQQ",
                    "name": "ProShares UltraPro QQQ",
                    "fund_type": "ETF",
                },
                {
                    "symbol": "BRKU",
                    "name": "2X Long BRKB Daily ETF",
                    "fund_type": "ETF (Single Stock)",
                },
                {
                    "symbol": "BGGG",
                    "name": "Baillie Gifford Long Term Global Growth ETF",
                    "fund_type": "ETF",
                },
            ]
        )
        mock_load_issuer_universe.return_value = pd.DataFrame(
            [
                {
                    "symbol": "EXTRA",
                    "name": "Example 2X Leveraged Broad Market ETF",
                    "fund_type": "ETF (Example Issuer)",
                    "source": "Example issuer table",
                },
            ]
        )
        mock_load_etn_universe.return_value = pd.DataFrame(
            [
                {
                    "symbol": "BDCX",
                    "name": "ETRACS Quarterly Pay 1.5x Leveraged MarketVector BDC Liquid Index ETN",
                    "fund_type": "ETN (UBS ETRACS)",
                    "source": "UBS ETRACS ETN issuer table",
                },
            ]
        )
        mock_load_audit_sources.return_value = (
            pd.DataFrame(
                [
                    {
                        "symbol": "MISSING",
                        "name": "Example 2X Long MISSING Daily ETF",
                        "fund_type": "ETF (Audit)",
                        "source": "Audit source",
                        "audit_source_type": "third_party_audit",
                        "source_url": "https://example.test",
                        "is_leveraged_candidate": True,
                        "is_long_leveraged_candidate": True,
                        "leverage": 2.0,
                        "direction": "long",
                    },
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "source": "Audit source",
                        "source_type": "third_party_audit",
                        "url": "https://example.test",
                        "parser": "html",
                        "enabled": True,
                        "status": "loaded",
                        "row_count": 1,
                        "error": "",
                        "notes": "",
                    }
                ]
            ),
        )

        workflow_assets = determine_workflow_assets(
            UniverseConfig(sqlite_db_path="state.sqlite")
        )

        self.assertEqual(
            workflow_assets.attrs["universe_title"],
            "Executable Long Leveraged ETFs/ETNs From Merged Universe",
        )
        self.assertEqual(workflow_assets.attrs["universe_counts"]["Current ETFs in Nasdaq table"], 3)
        self.assertEqual(workflow_assets.attrs["universe_counts"]["Current issuer-discovered leveraged ETFs found"], 1)
        self.assertEqual(workflow_assets.attrs["universe_counts"]["Current issuer-discovered leveraged ETNs found"], 1)
        self.assertEqual(workflow_assets.attrs["universe_counts"]["Merged current ETFs/ETNs"], 5)
        self.assertEqual(workflow_assets.attrs["universe_counts"]["Current long leveraged ETFs/ETNs found"], 4)
        self.assertEqual(workflow_assets.attrs["universe_counts"]["Audit rows parsed"], 1)
        self.assertEqual(
            workflow_assets.attrs["universe_counts"]["Audit leveraged candidates missing from merged universe"],
            1,
        )
        self.assertEqual(workflow_assets.attrs["universe_db_path"], "state.sqlite")
        self.assertEqual(workflow_assets["symbol"].tolist(), ["BDCX", "BRKU", "EXTRA", "TQQQ"])
        self.assertEqual(workflow_assets["rsi_symbol"].tolist(), ["BIZD", "BRK-B", "EXTRA", "QQQ"])
        self.assertIn("confidence", workflow_assets.columns)
        saved_table_names = [call.args[2] for call in mock_save_table.call_args_list]
        self.assertEqual(
            saved_table_names,
            [
                "nasdaq_etf_universe",
                "universe_inactive_discovered_products",
                "universe_active_listing_source_status",
                "universe_workflow_source_status",
                "universe_audit_rows",
                "universe_audit_missing_candidates",
                "universe_audit_source_status",
                "universe_rsi_mapping_review",
            ],
        )
        mock_print.assert_not_called()

    def test_unresolved_single_stock_mapping_is_saved_for_review(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [
                {
                    "symbol": "TQQQ",
                    "name": "ProShares UltraPro QQQ",
                    "fund_type": "ETF",
                },
                {
                    "symbol": "FOOU",
                    "name": "T-REX 2X Long ExampleCorp Daily Target ETF",
                    "fund_type": "ETF (Single Stock)",
                },
            ]
        )
        empty_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        empty_rows.attrs["workflow_source_status"] = []

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value={"FOOU", "TQQQ", "QQQ"}),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
        ):
            workflow_assets = determine_workflow_assets(UniverseConfig(sqlite_db_path="state.sqlite"))

        self.assertEqual(workflow_assets["symbol"].tolist(), ["TQQQ"])
        self.assertEqual(workflow_assets["rsi_symbol"].tolist(), ["QQQ"])
        self.assertEqual(workflow_assets.attrs["universe_counts"]["RSI mappings needing review"], 1)
        self.assertEqual(workflow_assets.attrs["universe_counts"]["RSI mappings excluded pending review"], 1)
        self.assertEqual(len(workflow_assets.attrs["rsi_mapping_review"]), 1)

        review_call = next(
            call
            for call in mock_save_table.call_args_list
            if call.args[2] == "universe_rsi_mapping_review"
        )
        review_df = review_call.args[0]
        self.assertEqual(review_df["symbol"].tolist(), ["FOOU"])
        self.assertEqual(review_df["confidence"].tolist(), ["needs_review"])

    def test_unresolved_short_single_stock_mapping_is_saved_for_review(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [
                {
                    "symbol": "TQQQ",
                    "name": "ProShares UltraPro QQQ",
                    "fund_type": "ETF",
                },
                {
                    "symbol": "SQQQ",
                    "name": "ProShares UltraPro Short QQQ",
                    "fund_type": "ETF",
                },
                {
                    "symbol": "FOOS",
                    "name": "2X Short ExampleCorp Daily ETF",
                    "fund_type": "ETF (Example Issuer)",
                },
            ]
        )
        empty_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        empty_rows.attrs["workflow_source_status"] = []

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=empty_rows),
            patch(
                "leveraged_trader.universe.load_active_listed_symbols",
                return_value={"FOOS", "QQQ", "SQQQ", "TQQQ"},
            ),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
        ):
            workflow_asset_groups = determine_workflow_asset_groups(
                UniverseConfig(sqlite_db_path="state.sqlite")
            )

        self.assertEqual(workflow_asset_groups["long"]["symbol"].tolist(), ["TQQQ"])
        self.assertEqual(workflow_asset_groups["short"]["symbol"].tolist(), ["SQQQ"])
        self.assertEqual(workflow_asset_groups["short"]["rsi_symbol"].tolist(), ["QQQ"])
        self.assertEqual(
            workflow_asset_groups["short"].attrs["universe_counts"]["RSI mappings needing review"],
            1,
        )
        self.assertEqual(
            workflow_asset_groups["short"].attrs["universe_counts"]["RSI mappings excluded pending review"],
            1,
        )

        review_call = next(
            call
            for call in mock_save_table.call_args_list
            if call.args[2] == "universe_rsi_mapping_review"
        )
        review_df = review_call.args[0]
        self.assertEqual(review_df["workflow"].tolist(), ["Short"])
        self.assertEqual(review_df["symbol"].tolist(), ["FOOS"])
        self.assertEqual(review_df["confidence"].tolist(), ["needs_review"])
        self.assertEqual(review_df["mapping_source"].tolist(), ["unresolved_single_stock"])

    def test_explicit_self_fallback_basket_mapping_remains_executable(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [
                {
                    "symbol": "TQQQ",
                    "name": "ProShares UltraPro QQQ",
                    "fund_type": "ETF",
                },
                {
                    "symbol": "BEGS",
                    "name": "Rareview 2X Bull Cryptocurrency & Precious Metals ETF",
                    "fund_type": "ETF",
                },
            ]
        )
        empty_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        empty_rows.attrs["workflow_source_status"] = []

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value={"BEGS", "TQQQ", "QQQ"}),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
        ):
            workflow_assets = determine_workflow_assets(UniverseConfig(sqlite_db_path="state.sqlite"))

        self.assertEqual(workflow_assets["symbol"].tolist(), ["BEGS", "TQQQ"])
        self.assertEqual(workflow_assets["rsi_symbol"].tolist(), ["BEGS", "QQQ"])
        self.assertEqual(workflow_assets.attrs["universe_counts"]["RSI mappings needing review"], 0)
        self.assertEqual(workflow_assets.attrs["universe_counts"]["RSI mappings excluded pending review"], 0)
        self.assertEqual(workflow_assets.attrs["rsi_mapping_review"], [])

        begs = workflow_assets.loc[workflow_assets["symbol"] == "BEGS"].iloc[0]
        self.assertEqual(begs["confidence"], "fallback_to_self")
        self.assertEqual(begs["mapping_source"], "self_fallback_override")

        review_call = next(
            call
            for call in mock_save_table.call_args_list
            if call.args[2] == "universe_rsi_mapping_review"
        )
        review_df = review_call.args[0]
        self.assertTrue(review_df.empty)

    def test_spacex_underlying_maps_to_spcx_even_when_active_listing_is_partial(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [
                {"symbol": "QQQ", "name": "Invesco QQQ Trust", "fund_type": "ETF"},
                {"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"},
                {
                    "symbol": "SPAL",
                    "name": "GraniteShares 2x Long SpaceX Daily ETF",
                    "fund_type": "ETF (GraniteShares)",
                },
            ]
        )
        empty_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        empty_rows.attrs["workflow_source_status"] = []
        partial_listing = ActiveListedSymbols(
            {"TQQQ", "SPAL"},
            [
                {"source": "nasdaq_listed", "status": "loaded"},
                {"source": "other_listed", "status": "error"},
            ],
        )

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value=partial_listing),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
        ):
            workflow_assets = determine_workflow_assets(UniverseConfig(sqlite_db_path="state.sqlite"))

        self.assertEqual(workflow_assets["symbol"].tolist(), ["SPAL", "TQQQ"])
        self.assertEqual(workflow_assets["rsi_symbol"].tolist(), ["SPCX", "QQQ"])
        self.assertNotIn("SPACEX", workflow_assets["rsi_symbol"].tolist())
        self.assertEqual(workflow_assets.attrs["universe_counts"]["RSI mappings needing review"], 0)
        self.assertEqual(workflow_assets.attrs["rsi_mapping_review"], [])

        nasdaq_call = next(
            call
            for call in mock_save_table.call_args_list
            if call.args[2] == "nasdaq_etf_universe"
        )
        nasdaq_universe = nasdaq_call.args[0]
        self.assertNotIn("SPACEX", nasdaq_universe["rsi_symbol"].tolist())
        self.assertEqual(
            nasdaq_universe.loc[nasdaq_universe["symbol"] == "SPAL", "rsi_symbol"].item(),
            "SPCX",
        )
        self.assertEqual(
            nasdaq_universe.loc[nasdaq_universe["symbol"] == "SPAL", "confidence"].item(),
            "curated",
        )

    def test_partial_active_listing_still_rejects_unknown_inferred_rsi_symbols(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [
                {"symbol": "ENGY", "name": "Ultra Energy", "fund_type": "ETF"},
                {"symbol": "QQQ", "name": "Invesco QQQ Trust", "fund_type": "ETF"},
                {"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"},
            ]
        )
        empty_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        empty_rows.attrs["workflow_source_status"] = []
        partial_listing = ActiveListedSymbols(
            {"TQQQ"},
            [
                {"source": "nasdaq_listed", "status": "loaded"},
                {"source": "other_listed", "status": "error", "error": "offline"},
            ],
        )

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value=partial_listing),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
        ):
            workflow_assets = determine_workflow_assets(UniverseConfig(sqlite_db_path="state.sqlite"))

        dig = workflow_assets.loc[workflow_assets["symbol"] == "ENGY"].iloc[0]
        tqqq = workflow_assets.loc[workflow_assets["symbol"] == "TQQQ"].iloc[0]
        self.assertEqual(dig["rsi_symbol"], "ENGY")
        self.assertEqual(dig["confidence"], "fallback_to_self")
        self.assertEqual(tqqq["rsi_symbol"], "QQQ")
        self.assertNotIn("ENERGY", workflow_assets["rsi_symbol"].tolist())
        self.assertTrue(workflow_assets.attrs["universe_degraded"])
        self.assertEqual(workflow_assets.attrs["universe_counts"]["Active listing sources failed"], 1)

        nasdaq_call = next(
            call
            for call in mock_save_table.call_args_list
            if call.args[2] == "nasdaq_etf_universe"
        )
        nasdaq_universe = nasdaq_call.args[0]
        self.assertNotIn("ENERGY", nasdaq_universe["rsi_symbol"].tolist())

    def test_inactive_issuer_only_product_is_excluded_from_workflow(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [{"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"}]
        )
        issuer_rows = pd.DataFrame(
            [
                {
                    "symbol": "STALE",
                    "name": "Example 2X Long STALE Daily ETF",
                    "fund_type": "ETF (Example)",
                    "source": "Example issuer table",
                }
            ]
        )
        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=issuer_rows),
            patch(
                "leveraged_trader.universe.load_etn_universe",
                return_value=pd.DataFrame(columns=issuer_rows.columns),
            ),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value={"TQQQ", "QQQ"}),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
        ):
            workflow_assets = determine_workflow_assets(UniverseConfig(sqlite_db_path="state.sqlite"))

        self.assertEqual(workflow_assets["symbol"].tolist(), ["TQQQ"])
        inactive_rows = next(
            call.args[0]
            for call in mock_save_table.call_args_list
            if call.args[2] == "universe_inactive_discovered_products"
        )
        self.assertEqual(inactive_rows["symbol"].tolist(), ["STALE"])
        self.assertEqual(
            workflow_assets.attrs["universe_counts"]["Inactive issuer-discovered ETFs/ETNs skipped"],
            1,
        )

    def test_partial_active_listing_snapshot_does_not_filter_issuer_products(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [{"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"}]
        )
        issuer_rows = pd.DataFrame(
            [{
                "symbol": "ISSUER", "name": "Example 2X Leveraged Broad Market ETF",
                "fund_type": "ETF (Example)", "source": "Example issuer table",
            }]
        )
        partial_listing = ActiveListedSymbols(
            {"TQQQ"},
            [
                {"source": "nasdaq_listed", "status": "loaded"},
                {"source": "other_listed", "status": "error"},
            ],
        )
        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=issuer_rows),
            patch(
                "leveraged_trader.universe.load_etn_universe",
                return_value=pd.DataFrame(columns=issuer_rows.columns),
            ),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value=partial_listing),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite"),
        ):
            workflow_assets = determine_workflow_assets(UniverseConfig(sqlite_db_path="state.sqlite"))

        self.assertEqual(workflow_assets["symbol"].tolist(), ["ISSUER", "TQQQ"])
        self.assertFalse(workflow_assets.attrs["universe_counts"]["Active listing snapshot complete"])

    def test_issuer_source_failure_is_recorded_on_the_returned_universe(self) -> None:
        with (
            patch("leveraged_trader.universe.ISSUER_UNIVERSE_SOURCES", [("Test Issuer", "https://issuer.test")]),
            patch("leveraged_trader.universe.requests.get", side_effect=RuntimeError("offline")),
        ):
            issuer_universe = load_issuer_etf_universe()

        self.assertTrue(issuer_universe.empty)
        status = issuer_universe.attrs["workflow_source_status"]
        self.assertEqual(status[0]["status"], "source_error")
        self.assertIn("offline", str(status[0]["error"]))

    @patch("leveraged_trader.universe.requests.get")
    def test_issuer_source_with_unparseable_success_response_is_a_parse_error(self, mock_get: Mock) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.text = "<html><body>temporary maintenance page</body></html>"
        mock_get.return_value = response

        with patch(
            "leveraged_trader.universe.ISSUER_UNIVERSE_SOURCES",
            [("Test Issuer", "https://issuer.test")],
        ):
            issuer_universe = load_issuer_etf_universe()

        self.assertTrue(issuer_universe.empty)
        status = issuer_universe.attrs["workflow_source_status"]
        self.assertTrue(status)
        self.assertTrue(all(row["status"] == "parse_error" for row in status))
        self.assertTrue(all("No product rows" in str(row["error"]) for row in status))

    def test_registered_only_workflow_source_is_healthy(self) -> None:
        with patch(
            "leveraged_trader.universe.ISSUER_UNIVERSE_SOURCES",
            [
                UniverseSource(
                    "Blocked Issuer",
                    "https://issuer.test",
                    "issuer_etf",
                    enabled=False,
                    notes="blocked",
                )
            ],
        ):
            issuer_universe = load_issuer_etf_universe()

        status = issuer_universe.attrs["workflow_source_status"]
        self.assertEqual(status[0]["status"], "registered_only")
        self.assertIn("blocked", status[0]["error"])

    @patch("leveraged_trader.universe.requests.get")
    def test_issuer_source_with_valid_zero_leveraged_matches_is_healthy(self, mock_get: Mock) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.text = """
            <table>
                <tr><th>Ticker</th><th>Fund Name</th></tr>
                <tr><td>SAFE</td><td>Acme Income ETF</td></tr>
            </table>
        """
        mock_get.return_value = response

        with patch(
            "leveraged_trader.universe.ISSUER_UNIVERSE_SOURCES",
            [("Test Issuer", "https://issuer.test")],
        ):
            issuer_universe = load_issuer_etf_universe()

        self.assertTrue(issuer_universe.empty)
        status = issuer_universe.attrs["workflow_source_status"]
        self.assertEqual(status[0]["status"], "loaded_zero_matches")
        self.assertEqual(status[0]["parsed_row_count"], 1)
        self.assertEqual(status[0]["row_count"], 0)

    def test_strict_workflow_source_mode_aborts_after_recording_source_health(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [{"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"}]
        )
        issuer_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        issuer_rows.attrs["workflow_source_status"] = [
            {
                "source": "Test Issuer",
                "source_type": "issuer_etf",
                "url": "https://issuer.test",
                "status": "error",
                "row_count": 0,
                "error": "offline",
            }
        ]
        etn_rows = pd.DataFrame(columns=issuer_rows.columns)
        etn_rows.attrs["workflow_source_status"] = []
        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=issuer_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=etn_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value={"TQQQ", "QQQ"}),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
            self.assertRaisesRegex(RuntimeError, "Test Issuer"),
        ):
            determine_workflow_assets(
                UniverseConfig(sqlite_db_path="state.sqlite", require_workflow_source_success=True)
            )

        self.assertIn(
            "universe_workflow_source_status",
            [call.args[2] for call in mock_save_table.call_args_list],
        )

    def test_strict_workflow_source_mode_aborts_on_active_listing_failure(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [
                {"symbol": "QQQ", "name": "Invesco QQQ Trust", "fund_type": "ETF"},
                {"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"},
            ]
        )
        empty_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        empty_rows.attrs["workflow_source_status"] = []
        partial_listing = ActiveListedSymbols(
            {"TQQQ"},
            [
                {"source": "nasdaq_listed", "status": "loaded"},
                {"source": "other_listed", "status": "error", "error": "offline"},
            ],
        )

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value=partial_listing),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
            self.assertRaisesRegex(RuntimeError, "active listing sources were unusable: other_listed"),
        ):
            determine_workflow_assets(
                UniverseConfig(sqlite_db_path="state.sqlite", require_workflow_source_success=True)
            )

        self.assertIn(
            "universe_active_listing_source_status",
            [call.args[2] for call in mock_save_table.call_args_list],
        )

    def test_parse_error_workflow_source_marks_the_universe_degraded(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [{"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"}]
        )
        issuer_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        issuer_rows.attrs["workflow_source_status"] = [
            {
                "source": "Test Issuer",
                "source_type": "issuer_etf",
                "url": "https://issuer.test",
                "status": "parse_error",
                "row_count": 0,
                "error": "No product rows could be parsed from a successful source response.",
            }
        ]
        etn_rows = pd.DataFrame(columns=issuer_rows.columns)
        etn_rows.attrs["workflow_source_status"] = []

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=issuer_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=etn_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value={"TQQQ", "QQQ"}),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite"),
        ):
            workflow_assets = determine_workflow_assets(UniverseConfig(sqlite_db_path="state.sqlite"))

        self.assertTrue(workflow_assets.attrs["universe_degraded"])
        self.assertEqual(workflow_assets.attrs["universe_counts"]["Workflow universe sources failed"], 1)

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=issuer_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=etn_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value={"TQQQ", "QQQ"}),
            patch("leveraged_trader.universe.save_table_to_sqlite"),
            self.assertRaisesRegex(RuntimeError, "Test Issuer"),
        ):
            determine_workflow_assets(
                UniverseConfig(sqlite_db_path="state.sqlite", require_workflow_source_success=True)
            )

    def test_zero_match_workflow_source_is_healthy_in_strict_mode(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [{"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"}]
        )
        issuer_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        issuer_rows.attrs["workflow_source_status"] = [
            {
                "source": "Test Issuer",
                "source_type": "issuer_etf",
                "url": "https://issuer.test",
                "status": "loaded_zero_matches",
                "parsed_row_count": 1,
                "row_count": 0,
                "error": "",
            }
        ]
        etn_rows = pd.DataFrame(columns=issuer_rows.columns)
        etn_rows.attrs["workflow_source_status"] = []

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=issuer_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=etn_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value={"TQQQ", "QQQ"}),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite"),
        ):
            workflow_assets = determine_workflow_assets(
                UniverseConfig(sqlite_db_path="state.sqlite", require_workflow_source_success=True)
            )

        self.assertFalse(workflow_assets.attrs["universe_degraded"])
        self.assertEqual(workflow_assets.attrs["universe_counts"]["Workflow universe sources failed"], 0)


if __name__ == "__main__":
    unittest.main()
