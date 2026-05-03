from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crossalpha.config import ProjectConfig, ValidationConfig, ValidationSplitConfig
from crossalpha.utils.io import write_parquet


@pytest.fixture()
def synthetic_project(tmp_path):
    cfg = ProjectConfig(root_dir=tmp_path)
    cfg.validation = ValidationConfig(
        embargo_days=5,
        holdout_start="2019-09-01",
        holdout_end="2019-12-31",
        splits=[
            ValidationSplitConfig(train_end="2019-03-29", val_start="2019-04-08", val_end="2019-06-28"),
        ],
    )
    cfg.ensure_directories()

    dates = pd.bdate_range("2018-01-01", periods=380)
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    sectors = {
        "AAA": "Technology",
        "BBB": "Technology",
        "CCC": "Healthcare",
        "DDD": "Finance",
        "EEE": "Energy",
    }

    rng = np.random.default_rng(7)
    price_rows = []
    quality_rows = []
    universe_rows = []
    benchmark_rows = []
    vix_rows = []
    riskfree_rows = []
    earnings_rows = []
    fundamentals_rows = []

    benchmark_level = 2500 + np.linspace(0, 250, len(dates)) + np.sin(np.arange(len(dates)) / 20.0) * 20
    vix_level = 18 + np.sin(np.arange(len(dates)) / 15.0) * 4

    for date, bench_close, vix_close in zip(dates, benchmark_level, vix_level, strict=True):
        benchmark_rows.append({
            "date": date,
            "ticker": "^GSPC",
            "open": bench_close * 0.995,
            "high": bench_close * 1.01,
            "low": bench_close * 0.99,
            "close": bench_close,
            "adj_close": bench_close,
            "volume": 1_000_000,
            "source": "synthetic",
        })
        vix_rows.append({
            "date": date,
            "ticker": "^VIX",
            "open": vix_close,
            "high": vix_close * 1.02,
            "low": vix_close * 0.98,
            "close": vix_close,
            "adj_close": vix_close,
            "volume": 0,
            "source": "synthetic",
        })
        riskfree_rows.append({
            "date": date,
            "series_id": "DGS3MO",
            "value": 2.0,
            "source": "synthetic",
        })

    for idx, ticker in enumerate(tickers):
        base = 40 + idx * 15
        trend = 0.05 + idx * 0.01
        noise = rng.normal(scale=0.6 + idx * 0.1, size=len(dates)).cumsum() * 0.05
        closes = base + trend * np.arange(len(dates)) + noise
        volumes = 1_000_000 + idx * 100_000 + (np.sin(np.arange(len(dates)) / 10.0) * 50_000)
        for row_num, date in enumerate(dates):
            close = max(5.0, closes[row_num])
            open_price = close * (1 - 0.002)
            high = close * 1.01
            low = close * 0.99
            volume = max(1000, volumes[row_num])
            price_rows.append({
                "date": date,
                "ticker": ticker,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "adj_close": close,
                "volume": volume,
                "source": "synthetic",
                "was_observed": True,
            })
            quality_rows.append({
                "date": date,
                "ticker": ticker,
                "observed_price": True,
                "history_days": row_num + 1,
                "listing_age_days": row_num + 1,
                "stale_price_days": 0,
                "max_missing_gap_21d": 0,
                "insufficient_price_history": row_num + 1 < 252,
                "insufficient_post_listing_history": row_num + 1 < 63,
                "stale_price": False,
                "missing_gap_fail": False,
            })
            universe_rows.append({
                "date": date,
                "ticker": ticker,
                "is_member": True,
                "membership_reliability": 1.0,
                "source": "synthetic",
                "sector": sectors[ticker],
                "industry": f"{sectors[ticker]} Industry",
                "universe_reliable": True,
                "first_membership_date": dates[0],
                "membership_days": row_num + 1,
            })

        earnings_dates = dates[20::63][:5]
        for event_num, event_date in enumerate(earnings_dates):
            earnings_rows.append({
                "ticker": ticker,
                "earnings_date": event_date,
                "eps_estimate": 1.0 + event_num * 0.02,
                "eps_actual": 1.05 + event_num * 0.03,
                "surprise_pct": 5.0 + event_num,
                "source": "synthetic",
            })

        filing_dates = dates[40::63][:5]
        for report_num, filed_at in enumerate(filing_dates):
            fundamentals_rows.append({
                "ticker": ticker,
                "filed_at": filed_at,
                "period_end": filed_at - pd.Timedelta(days=45),
                "form": "10-Q",
                "revenue": 1_000_000 + report_num * 25_000 + idx * 50_000,
                "net_income": 100_000 + report_num * 5_000 + idx * 7_500,
                "total_assets": 5_000_000 + report_num * 50_000 + idx * 100_000,
                "total_liabilities": 2_000_000 + report_num * 30_000 + idx * 80_000,
                "stockholders_equity": 3_000_000 + report_num * 20_000 + idx * 50_000,
                "cash_and_equivalents": 500_000 + report_num * 5_000,
                "common_shares_outstanding": 100_000_000,
                "eps_diluted": 1.0 + report_num * 0.02,
                "operating_income": 130_000 + report_num * 4_000,
                "depreciation_amortization": 10_000 + report_num * 250,
                "source": "synthetic",
            })

    write_parquet(pd.DataFrame(price_rows), cfg.processed_dir / "prices_cleaned.parquet")
    write_parquet(pd.DataFrame(quality_rows), cfg.processed_dir / "price_quality.parquet")
    write_parquet(pd.DataFrame(universe_rows), cfg.processed_dir / "daily_universe.parquet")
    write_parquet(pd.DataFrame(benchmark_rows), cfg.processed_dir / "benchmark.parquet")
    write_parquet(pd.DataFrame(vix_rows), cfg.processed_dir / "vix.parquet")
    write_parquet(pd.DataFrame(riskfree_rows), cfg.processed_dir / "risk_free.parquet")
    write_parquet(pd.DataFrame(earnings_rows), cfg.processed_dir / "earnings.parquet")
    write_parquet(pd.DataFrame(fundamentals_rows), cfg.processed_dir / "fundamentals.parquet")
    write_parquet(pd.DataFrame([{"ticker": t, "sector": sectors[t], "industry": f"{sectors[t]} Industry", "source": "synthetic"} for t in tickers]), cfg.processed_dir / "metadata.parquet")
    return cfg
