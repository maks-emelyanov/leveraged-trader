from datetime import date, datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from leveraged_trader import universe
from leveraged_trader.config import UniverseConfig
from leveraged_trader.universe import ActiveListedSymbols, determine_workflow_asset_groups
from leveraged_trader.universe_lifecycle import PRODUCT_CLOSURES, split_closed_products


@pytest.mark.parametrize("symbol", PRODUCT_CLOSURES)
def test_closure_applies_only_after_last_trading_day(symbol: str) -> None:
    closure = PRODUCT_CLOSURES[symbol]
    products = pd.DataFrame([{"symbol": symbol, "name": closure.name, "fund_type": "ETF"}])

    current, inactive = split_closed_products(products, as_of=closure.last_trading_date)
    pd.testing.assert_frame_equal(current, products)
    assert inactive.empty

    current, inactive = split_closed_products(products, as_of=date(2026, 10, 2))
    assert current.empty
    assert inactive["symbol"].tolist() == [symbol]
    assert inactive["inactive_source"].tolist() == [closure.source_url]
    assert closure.last_trading_date.isoformat() in inactive.iloc[0]["inactive_reason"]


def test_closure_requires_product_identity_and_keeps_unknown_issuer_products() -> None:
    products = pd.DataFrame(
        [
            {"symbol": "USML", "name": "New Issuer 2X Long USM Daily ETF"},
            {"symbol": "NEW", "name": PRODUCT_CLOSURES["USML"].name},
            {"symbol": "FRESH", "name": "New Issuer 2X Long QQQ Daily ETF"},
            {"symbol": "usml", "name": PRODUCT_CLOSURES["USML"].name.upper().replace(" ", "\u00a0 ")},
        ],
        index=[10, 20, 30, 40],
    )

    current, inactive = split_closed_products(products, as_of=date(2026, 10, 2))

    assert current.index.tolist() == [10, 20, 30]
    assert inactive.index.tolist() == [40]


@pytest.mark.parametrize("directory_complete", [True, False])
def test_closed_issuer_products_are_audited_before_workflow_selection(directory_complete: bool) -> None:
    primary = pd.DataFrame(
        [
            {"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"},
            {"symbol": "PAAU", "name": PRODUCT_CLOSURES["PAAU"].name, "fund_type": "ETF"},
        ]
    )
    issuer = pd.DataFrame([{"symbol": "FRESH", "name": "Example 2X Long QQQ Daily ETF", "fund_type": "ETF (Example)"}])
    etns = pd.DataFrame(
        [
            {"symbol": symbol, "name": closure.name, "fund_type": "ETN (UBS ETRACS)"}
            for symbol, closure in PRODUCT_CLOSURES.items()
            if symbol != "PAAU"
        ]
    )
    directory = ActiveListedSymbols(
        {"TQQQ", "QQQ", "PAAU", "USML"},
        [] if directory_complete else [{"status": "error"}],
    )
    with (
        patch("leveraged_trader.universe.load_current_etf_universe", return_value=primary),
        patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=issuer),
        patch("leveraged_trader.universe.load_etn_universe", return_value=etns),
        patch("leveraged_trader.universe.load_active_listed_symbols", return_value=directory),
        patch("leveraged_trader.universe.load_audit_universe_sources", return_value=(pd.DataFrame(), pd.DataFrame())),
        patch("leveraged_trader.universe.save_table_to_sqlite") as save,
        patch(
            "leveraged_trader.universe._nasdaq_directory_now",
            return_value=datetime(2026, 10, 2, tzinfo=ZoneInfo("America/New_York")),
        ),
    ):
        groups = determine_workflow_asset_groups(UniverseConfig(sqlite_db_path="unused.sqlite"))

    assert set(groups["long"]["symbol"]) == {"TQQQ", "FRESH"}
    tables = {call.args[2]: call.args[0] for call in save.call_args_list}
    assert set(tables["universe_inactive_discovered_products"]["symbol"]) == set(PRODUCT_CLOSURES)
    assert set(tables["nasdaq_etf_universe"]["symbol"]) == {"TQQQ", "FRESH"}
    counts = groups["long"].attrs["universe_counts"]
    assert counts["Inactive primary Nasdaq ETFs skipped"] == 1
    assert counts["Inactive issuer-discovered ETFs/ETNs skipped"] == 7


@pytest.mark.parametrize("stored_as", ["visible", "canonical", "closed_audit"])
def test_stale_closure_does_not_conflict_with_a_reused_primary_ticker(stored_as: str) -> None:
    current_name = "New Issuer 2X MSCI US Minimum Volatility ETF"
    primary = pd.DataFrame(
        [
            {"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"},
            {"symbol": "USML", "name": current_name, "fund_type": "ETF"},
        ]
    )
    empty = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
    closed_row = {
        "symbol": "USML",
        "name": PRODUCT_CLOSURES["USML"].name,
        "fund_type": "ETN (UBS ETRACS)",
        "source": "UBS ETRACS issuer table",
    }
    etns = pd.DataFrame([closed_row]) if stored_as == "visible" else empty.copy()
    etns.attrs["workflow_source_status"] = [
        {
            "source": "UBS ETRACS",
            "source_type": "etn_issuer",
            "url": "https://etn.test",
            "status": "loaded",
            "parsed_row_count": 1,
            "row_count": 1,
            "error": "",
        }
    ]
    etns.attrs["workflow_symbol_sources"] = {"USML": (("UBS ETRACS", "https://etn.test"),)}
    if stored_as == "canonical":
        etns.attrs["workflow_canonical_product_rows"] = [closed_row, closed_row]
    elif stored_as == "closed_audit":
        _, closed = split_closed_products(pd.DataFrame([closed_row]), as_of=date(2026, 10, 2))
        etns.attrs["workflow_canonical_product_rows"] = []
        etns.attrs["workflow_closed_product_rows"] = closed.to_dict("records") * 2

    with (
        patch.object(universe, "load_current_etf_universe", return_value=primary),
        patch.object(universe, "load_issuer_etf_universe", return_value=empty),
        patch.object(universe, "load_etn_universe", return_value=etns),
        patch.object(universe, "load_active_listed_symbols", return_value={"TQQQ", "USML", "QQQ"}),
        patch.object(universe, "load_audit_universe_sources", return_value=(pd.DataFrame(), pd.DataFrame())),
        patch.object(universe, "save_table_to_sqlite") as save,
        patch.object(
            universe,
            "_nasdaq_directory_now",
            return_value=datetime(2026, 10, 2, tzinfo=ZoneInfo("America/New_York")),
        ),
    ):
        groups = determine_workflow_asset_groups(
            UniverseConfig(sqlite_db_path="unused.sqlite", require_workflow_source_success=True)
        )

    assert set(groups["long"]["symbol"]) == {"TQQQ", "USML"}
    assert groups["long"].set_index("symbol").loc["USML", "name"] == current_name
    assert not groups["long"].attrs["universe_degraded"]
    assert groups["long"].attrs["universe_counts"]["Inactive issuer-discovered ETFs/ETNs skipped"] == 1
    tables = {call.args[2]: call.args[0] for call in save.call_args_list}
    inactive = tables["universe_inactive_discovered_products"]
    assert inactive["symbol"].tolist() == ["USML"]
    assert inactive["name"].tolist() == [closed_row["name"]]
    assert inactive["source"].tolist() == [closed_row["source"]]
    assert inactive["inactive_source"].tolist() == [PRODUCT_CLOSURES["USML"].source_url]
    assert tables["universe_workflow_source_status"]["status"].tolist() == ["loaded", "loaded"]


@pytest.mark.parametrize("source_kind", ["issuer", "etn"])
@pytest.mark.parametrize("include_reused_ticker", [False, True])
def test_source_loaders_filter_closures_before_classification_and_preserve_audit(
    source_kind: str, include_reused_ticker: bool
) -> None:
    source = universe.UniverseSource("Example", "https://example.test", "issuer_etf")
    closed_row = {
        "symbol": "PAAU",
        "name": PRODUCT_CLOSURES["PAAU"].name,
        "fund_type": "ETF (Example)",
        "source": "Example issuer table",
    }
    records = [closed_row]
    if include_reused_ticker:
        records.append({**closed_row, "name": "New Issuer 2X Long QQQ Daily ETF"})
    parser_name = (
        "_workflow_issuer_source_to_universe" if source_kind == "issuer" else "_workflow_etn_source_to_universe"
    )
    sources_name = "ISSUER_UNIVERSE_SOURCES" if source_kind == "issuer" else "WORKFLOW_ETN_SOURCES"
    loader = universe.load_issuer_etf_universe if source_kind == "issuer" else universe.load_etn_universe
    with (
        patch.object(universe, sources_name, [source]),
        patch.object(universe, "_fetch_enabled_sources", return_value=[universe._SourceFetchResult(text="inventory")]),
        patch.object(universe, parser_name, return_value=pd.DataFrame(records)),
        patch.object(
            universe,
            "_nasdaq_directory_now",
            return_value=datetime(2026, 10, 2, tzinfo=ZoneInfo("America/New_York")),
        ),
    ):
        loaded = loader()

    assert loaded["symbol"].tolist() == (["PAAU"] if include_reused_ticker else [])
    assert loaded.attrs["workflow_canonical_product_rows"] == (records[1:] if include_reused_ticker else [])
    status = loaded.attrs["workflow_source_status"][0]
    assert status["status"] == ("loaded" if include_reused_ticker else "loaded_zero_matches")
    assert status["parsed_row_count"] == len(records)
    assert status["row_count"] == int(include_reused_ticker)
    assert status["error"] == ""
    inactive = loaded.attrs["workflow_closed_product_rows"]
    assert len(inactive) == 1
    assert inactive[0]["symbol"] == "PAAU"
    assert inactive[0]["name"] == closed_row["name"]
    assert inactive[0]["inactive_source"] == PRODUCT_CLOSURES["PAAU"].source_url
