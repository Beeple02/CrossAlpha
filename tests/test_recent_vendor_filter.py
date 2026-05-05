from __future__ import annotations

import pandas as pd

from crossalpha.data.pipeline import _tickers_with_recent_price_history


def test_recent_price_history_filter_uses_end_date_cutoff():
    price_history = pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB", "CCC"],
        "date": pd.to_datetime(["2026-04-29", "2026-04-30", "2025-12-01", "2026-01-15"]),
        "open": [10.0, 10.1, 20.0, 30.0],
        "high": [10.2, 10.3, 20.2, 30.2],
        "low": [9.8, 9.9, 19.8, 29.8],
        "close": [10.1, 10.2, 20.1, 30.1],
        "adj_close": [10.1, 10.2, 20.1, 30.1],
    })
    recent = _tickers_with_recent_price_history(price_history, end_date="2026-04-30", max_staleness_days=120)
    assert recent == ["AAA", "CCC"]
