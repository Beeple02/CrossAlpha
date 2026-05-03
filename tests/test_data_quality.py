from __future__ import annotations

import pandas as pd

from crossalpha.config import ProjectConfig
from crossalpha.data.quality import clean_prices_and_flags


def test_quality_flags_detect_stale_and_short_history(tmp_path):
    cfg = ProjectConfig(root_dir=tmp_path)
    cfg.ensure_directories()
    dates = pd.bdate_range("2024-01-01", periods=8)
    prices = pd.DataFrame({
        "date": [dates[0], dates[1], dates[2], dates[6]],
        "ticker": ["AAA"] * 4,
        "open": [10, 11, 12, 13],
        "high": [10, 11, 12, 13],
        "low": [10, 11, 12, 13],
        "close": [10, 11, 12, 13],
        "adj_close": [10, 11, 12, 13],
        "volume": [1000, 1000, 1000, 1000],
        "source": ["synthetic"] * 4,
    })
    cleaned, quality = clean_prices_and_flags(prices, dates, cfg)
    assert not cleaned.empty
    last_row = quality.sort_values("date").iloc[-1]
    assert bool(last_row["insufficient_price_history"])
    assert last_row["max_missing_gap_21d"] >= 3
