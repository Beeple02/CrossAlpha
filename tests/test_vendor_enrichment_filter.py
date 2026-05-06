from __future__ import annotations

import pandas as pd

from crossalpha.data.pipeline import _vendor_enrichment_tickers


def test_vendor_enrichment_tickers_intersects_recent_and_current_constituents():
    recent_tickers = ["AAA", "BBB", "CCC"]
    wiki_metadata = pd.DataFrame({
        "ticker": ["AAA", "CCC", "DDD"],
        "sector": ["Tech", "Health", "Energy"],
    })
    assert _vendor_enrichment_tickers(recent_tickers, wiki_metadata) == ["AAA", "CCC"]
