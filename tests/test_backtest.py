from __future__ import annotations

import pandas as pd

from crossalpha.backtest.simulator import _simulate_horizon
from crossalpha.config import ProjectConfig


def test_backtest_turnover_and_costs(tmp_path):
    cfg = ProjectConfig(root_dir=tmp_path)
    cfg.ensure_directories()
    recommendations = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
        "ticker": ["AAA", "BBB", "AAA", "BBB"],
        "decision": ["BUY", "NO_BUY", "NO_BUY", "BUY"],
        "rank": [1, 2, 2, 1],
        "regime_state": ["OPEN", "OPEN", "OPEN", "OPEN"],
    })
    price_returns = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
        "ticker": ["AAA", "BBB", "AAA", "BBB"],
        "next_day_return": [0.01, 0.0, 0.0, 0.02],
    })
    benchmark = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "benchmark_return": [0.0, 0.0],
    })
    daily = _simulate_horizon(recommendations, price_returns, benchmark, cfg)
    assert len(daily) == 2
    assert daily["turnover"].iloc[0] > 0
    assert daily["cost"].iloc[1] > 0
