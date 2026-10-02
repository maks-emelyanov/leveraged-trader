from __future__ import annotations

import pytest

from leveraged_trader.universe import ISSUER_UNIVERSE_SOURCES, UniverseSource, _tradr_html_to_universe

SOURCE = UniverseSource("Tradr", "https://www.tradretfs.com/etfs", "issuer_etf", parser="tradr_html")
HEADER = (
    "<tr><th>Ticker</th><th>Fund Name</th><th>Reference Security</th>"
    "<th>Target</th><th>Exposure</th><th>Reset Period</th></tr>"
)


def _table(symbols: tuple[str, ...]) -> str:
    rows = "".join(
        f"<tr><td>{symbol}</td><td>Tradr 2X Long TEST Daily ETF</td><td>TEST</td>"
        "<td>2X</td><td>Long</td><td>Daily</td></tr>"
        for symbol in symbols
    )
    return f"<table>{HEADER}{rows}</table>"


def _current_inventory(content: str) -> str:
    return f'<div class="table_with_filter_global_section table_widget_123">{content}</div>'


def _legacy_inventory() -> str:
    # These retired funds were still present in the homepage's old module,
    # with the same complete schema as its current inventory, on 2026-10-02.
    symbols = (
        "AURU",
        "BLSX",
        "CELT",
        "DASX",
        "DOGD",
        "ENPX",
        "GSX",
        "LYFX",
        "MDBX",
        "NETX",
        "NWMX",
        "OKTX",
        "QSX",
        "SRPU",
        "VOYX",
    )
    return f'<div class="set_you_trade_section">{_table(symbols)}</div>'


def test_tradr_source_uses_the_dedicated_current_product_page() -> None:
    source = next(source for source in ISSUER_UNIVERSE_SOURCES if source.name == "Tradr")
    assert source.url == SOURCE.url


@pytest.mark.parametrize("legacy_first", [False, True])
def test_tradr_current_inventory_excludes_same_schema_retired_module(legacy_first: bool) -> None:
    current = _current_inventory(_table(("CURRENT",)))
    legacy = _legacy_inventory()
    # The real homepage hides both modules. CSS classes and table order cannot
    # determine which inventory is current.
    modules = (legacy, current) if legacy_first else (current, legacy)
    html = "<style>.layout-hidden {display:none}</style>" + "".join(
        f'<div class="layout-hidden">{module}</div>' for module in modules
    )

    out = _tradr_html_to_universe(html, SOURCE, require_leveraged=False)

    assert out["symbol"].tolist() == ["CURRENT"]
    assert out["reference_security"].tolist() == ["TEST"]


@pytest.mark.parametrize(
    ("current", "error"),
    [
        ("", "did not contain any authoritative product rows"),
        (f"<table>{HEADER}</table>", "did not contain any authoritative product rows"),
        (
            "<table><tr><th>Ticker</th><th>Fund Name</th><th>Target</th></tr>"
            "<tr><td>BAD</td><td>Tradr 2X Long TEST Daily ETF</td><td>2X</td></tr></table>",
            "omitted current authoritative columns",
        ),
        (_table(("BAD",)).replace("<td>Daily</td>", ""), "declared columns"),
        (_table(("BAD",)).replace("<td>2X</td>", "<td>3X</td>"), "leverage contradicted"),
    ],
)
def test_tradr_broken_current_inventory_never_falls_back_to_legacy(current: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        _tradr_html_to_universe(_current_inventory(current) + _legacy_inventory(), SOURCE)


def test_tradr_missing_current_inventory_never_accepts_legacy_module() -> None:
    with pytest.raises(ValueError, match="omitted its current product inventory"):
        _tradr_html_to_universe(_legacy_inventory(), SOURCE)


def test_tradr_multiple_current_inventories_are_ambiguous() -> None:
    html = _current_inventory(_table(("FIRST",))) + _current_inventory(_table(("SECOND",)))
    with pytest.raises(ValueError, match="ambiguous current product inventories"):
        _tradr_html_to_universe(html, SOURCE)
