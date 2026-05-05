from __future__ import annotations

import pandas as pd

from crossalpha.data.adapters.wikipedia import _flatten_columns, _is_changes_table, WikipediaSp500Adapter


def test_wikipedia_changes_table_detection_with_effective_date_headers():
    table = pd.DataFrame(
        [["April 9, 2026", "CASY", "HOLX", "Acquisition"]],
        columns=pd.MultiIndex.from_tuples([
            ("Effective Date", "Effective Date"),
            ("Added", "Ticker"),
            ("Removed", "Ticker"),
            ("Reason", "Reason"),
        ]),
    )
    flattened = _flatten_columns(table)
    assert _is_changes_table(flattened)


def test_wikipedia_normalize_changes_accepts_effective_date_column():
    changes = pd.DataFrame({
        "Effective Date Effective Date": ["April 9, 2026"],
        "Added Ticker": ["CASY"],
        "Removed Ticker": ["HOLX"],
    })
    normalized = WikipediaSp500Adapter._normalize_changes(changes)
    assert list(normalized.columns) == ["date", "added_ticker", "removed_ticker"]
    assert normalized.loc[0, "added_ticker"] == "CASY"
    assert normalized.loc[0, "removed_ticker"] == "HOLX"
