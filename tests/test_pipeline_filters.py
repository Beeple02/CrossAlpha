from __future__ import annotations

import pandas as pd

from crossalpha.data.pipeline import _tickers_with_price_history


def test_tickers_with_price_history_skips_all_nan_rows():
    price_history = pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB", "CCC"],
        "open": [10.0, None, None, None],
        "high": [10.5, None, None, None],
        "low": [9.5, None, None, None],
        "close": [10.2, None, None, 20.0],
        "adj_close": [10.2, None, None, 20.0],
    })
    assert _tickers_with_price_history(price_history) == ["AAA", "CCC"]
