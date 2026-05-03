from __future__ import annotations

import pandas as pd

from crossalpha.features.engine import _add_fundamental_features, _compute_security_features
from crossalpha.validation.splits import WalkForwardSplit, split_frame_for_horizon


def test_return_features_only_use_past_data():
    dates = pd.bdate_range("2024-01-01", periods=10)
    frame = pd.DataFrame({
        "date": dates,
        "ticker": "AAA",
        "adj_close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        "open": [100] * 10,
        "high": [101] * 10,
        "low": [99] * 10,
        "close": [100] * 10,
        "volume": [1_000_000] * 10,
        "spx_ret_1d": [0.0] * 10,
        "spx_ret_5d": [0.0] * 10,
        "spx_ret_21d": [0.0] * 10,
        "spx_ret_63d": [0.0] * 10,
        "sector": ["Technology"] * 10,
    })
    result = _compute_security_features(frame)
    target_row = result.loc[result["date"] == dates[5]].iloc[0]
    expected = 105 / 100 - 1
    assert abs(target_row["ret_5d"] - expected) < 1e-12


def test_fundamental_join_uses_filing_date_not_period_end(synthetic_project):
    cfg = synthetic_project
    feature_frame = pd.DataFrame({
        "date": pd.to_datetime(["2018-03-01", "2018-03-20", "2018-06-20"]),
        "ticker": ["AAA", "AAA", "AAA"],
        "adj_close": [50.0, 51.0, 52.0],
    })
    fundamentals = pd.DataFrame({
        "ticker": ["AAA", "AAA"],
        "filed_at": pd.to_datetime(["2018-02-15", "2018-06-15"]),
        "period_end": pd.to_datetime(["2017-12-31", "2018-03-31"]),
        "form": ["10-Q", "10-Q"],
        "revenue": [1000.0, 1100.0],
        "net_income": [100.0, 120.0],
        "total_assets": [5000.0, 5100.0],
        "total_liabilities": [2000.0, 2050.0],
        "stockholders_equity": [3000.0, 3050.0],
        "cash_and_equivalents": [500.0, 525.0],
        "common_shares_outstanding": [100.0, 100.0],
        "eps_diluted": [1.0, 1.1],
        "operating_income": [130.0, 140.0],
        "depreciation_amortization": [10.0, 11.0],
        "source": ["synthetic", "synthetic"],
    })
    merged = _add_fundamental_features(feature_frame, fundamentals, cfg)
    assert merged.loc[merged["date"] == pd.Timestamp("2018-03-20"), "filed_at"].iloc[0] == pd.Timestamp("2018-02-15")
    assert merged.loc[merged["date"] == pd.Timestamp("2018-06-20"), "filed_at"].iloc[0] == pd.Timestamp("2018-06-15")


def test_walkforward_split_purges_overlapping_labels():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-10"]),
        "label_end_date": pd.to_datetime(["2020-01-09", "2020-01-10", "2020-01-11", "2020-01-20"]),
    })
    split = WalkForwardSplit(
        name="test",
        train_end=pd.Timestamp("2020-01-03"),
        val_start=pd.Timestamp("2020-01-10"),
        val_end=pd.Timestamp("2020-01-31"),
    )
    train_df, val_df = split_frame_for_horizon(frame, split, embargo_days=5)
    assert train_df["label_end_date"].max() < split.val_start
    assert set(val_df["date"]) == {pd.Timestamp("2020-01-10")}
