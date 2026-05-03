"""Math helpers used across features, validation, and backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def safe_divide(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return numerator / np.where(np.asarray(denominator) == 0, np.nan, denominator)


def zscore_series(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_demeaned = x - x_mean
    denom = np.sum(x_demeaned ** 2)

    def _calc(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        y = values - values.mean()
        return float(np.dot(x_demeaned, y) / denom)

    return series.rolling(window).apply(_calc, raw=True)


def population_stability_index(train: pd.Series, test: pd.Series, bins: int = 10) -> float:
    train = train.dropna()
    test = test.dropna()
    if train.empty or test.empty:
        return np.nan

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(train.quantile(quantiles).to_numpy())
    if len(edges) <= 2:
        return 0.0

    train_bins = pd.cut(train, bins=edges, include_lowest=True)
    test_bins = pd.cut(test, bins=edges, include_lowest=True)

    train_dist = train_bins.value_counts(normalize=True).sort_index()
    test_dist = test_bins.value_counts(normalize=True).reindex(train_dist.index).fillna(1e-6)
    train_dist = train_dist.replace(0, 1e-6)
    test_dist = test_dist.replace(0, 1e-6)
    return float(((test_dist - train_dist) * np.log(test_dist / train_dist)).sum())


def max_drawdown(cumulative_returns: pd.Series) -> float:
    peak = cumulative_returns.cummax()
    drawdown = cumulative_returns / peak - 1.0
    return float(drawdown.min())
