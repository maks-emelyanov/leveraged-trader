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
    _etracs_leverage_table_to_universe,
    _html_cards_to_universe,
    _issuer_table_to_universe,
    _merge_universe_sources,
    _microsectors_html_to_universe,
    _sec_company_tickers_to_universe,
    _sec_exchange_tickers_to_universe,
    _sec_mutual_fund_tickers_to_universe,
    _with_audit_metadata,
    build_universe_audit_report,
    determine_workflow_assets,
    infer_leverage_and_direction,
    infer_rsi_symbol,
    is_long_leveraged_name,
    load_active_listed_symbols,
    load_current_etf_universe,
    load_issuer_etf_universe,
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

    def test_inferred_underlying_must_be_known_when_known_symbols_are_provided(self) -> None:
        self.assertEqual(
            infer_rsi_symbol("SKYU", "ProShares Ultra Cloud Computing", known_symbols={"SKYU", "QQQ"}),
            "SKYU",
        )
        self.assertEqual(
            infer_rsi_symbol("AAPU", "Direxion Daily AAPL Bull 2X ETF", known_symbols={"AAPU", "AAPL"}),
            "AAPL",
        )

    def test_normalizes_brkb_to_yahoo_symbol(self) -> None:
        self.assertEqual(infer_rsi_symbol("BRKU", "2X Long BRKB Daily ETF"), "BRK-B")
        self.assertEqual(infer_rsi_symbol("BRKU", "2X Long BRK.B Daily ETF"), "BRK-B")

    def test_long_duration_names_are_not_leveraged(self) -> None:
        self.assertFalse(is_long_leveraged_name("Vanguard Long-Term Corporate Bond ETF"))
        self.assertFalse(is_long_leveraged_name("Baillie Gifford Long Term Global Growth ETF"))
        self.assertFalse(is_long_leveraged_name("Innovator U.S. Equity Ultra Buffer ETF"))
        self.assertFalse(is_long_leveraged_name("MicroSectors FANG+ 1X Long Exposure ETN"))
        self.assertTrue(is_long_leveraged_name("GraniteShares 2x Long NVDA Daily ETF"))

    def test_embedded_company_name_multipliers_are_not_leverage(self) -> None:
        for name in [
            "8X8 INC /DE/",
            "10x Genomics, Inc.",
            "V2X, Inc.",
            "Ultragenyx Pharmaceutical Inc.",
        ]:
            self.assertEqual(infer_leverage_and_direction(name), (None, None))

    def test_extra_issuer_sources_are_registered(self) -> None:
        issuer_names = {issuer for issuer, _url in ISSUER_UNIVERSE_SOURCES}

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
                    <td><div>/product/detail/index/ussymbol/PLAIN</div><span>PLAIN</span></td>
                    <td>ETRACS Plain Index ETN</td>
                    <td>--</td>
                </tr>
            </table>
        """

        out = _etracs_leverage_table_to_universe(html, source)

        self.assertEqual(out["symbol"].tolist(), ["BDCX"])
        self.assertEqual(out["fund_type"].tolist(), ["ETN (UBS ETRACS)"])

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

        self.assertEqual(report["symbol"].tolist(), ["MISSING"])

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
                    "name": "Example 2X Long EXTRA Daily ETF",
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

        self.assertEqual(workflow_assets.attrs["universe_title"], "All Long Leveraged ETFs/ETNs From Merged Universe")
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
        self.assertEqual(workflow_assets["rsi_symbol"].tolist(), ["BDCX", "BRK-B", "EXTRA", "QQQ"])
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
            ],
        )
        mock_print.assert_not_called()

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
                "symbol": "ISSUER", "name": "Example 2X Long ISSUER Daily ETF",
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

        issuer_universe = load_issuer_etf_universe()

        self.assertTrue(issuer_universe.empty)
        status = issuer_universe.attrs["workflow_source_status"]
        self.assertTrue(status)
        self.assertTrue(all(row["status"] == "parse_error" for row in status))
        self.assertTrue(all("No product rows" in str(row["error"]) for row in status))

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
