"""Price cleaning and data-quality rules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crossalpha.config import ProjectConfig


PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close"]


def clean_prices_and_flags(
    prices: pd.DataFrame,
    benchmark_dates: pd.DatetimeIndex,
    cfg: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned_frames: list[pd.DataFrame] = []
    quality_frames: list[pd.DataFrame] = []

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])

    for ticker, frame in prices.groupby("ticker", sort=False):
        aligned = frame.sort_values("date").set_index("date").reindex(benchmark_dates)
        aligned["ticker"] = ticker
        observed = aligned["adj_close"].notna()

        last_seen_date = pd.Series(aligned.index.where(observed), index=aligned.index).ffill()
        aligned[PRICE_COLUMNS] = aligned[PRICE_COLUMNS].ffill(limit=cfg.quality.max_forward_fill_days)
        aligned["volume"] = aligned["volume"].ffill(limit=cfg.quality.max_forward_fill_days)
        aligned["source"] = aligned["source"].ffill()

        missing_streak = (~observed).astype(int).groupby(observed.cumsum()).cumsum()
        history_days = observed.cumsum()
        first_valid_date = pd.Series(aligned.index.where(observed), index=aligned.index).bfill()
        listing_age_days = np.where(
            first_valid_date.notna(),
            ((aligned.index - pd.to_datetime(first_valid_date)).days // 1) + 1,
            0,
        )
        stale_days = np.where(last_seen_date.notna(), (aligned.index - pd.to_datetime(last_seen_date)).days, np.inf)
        rolling_missing_gap = pd.Series(missing_streak, index=aligned.index).rolling(21, min_periods=1).max()

        quality = pd.DataFrame({
            "date": aligned.index,
            "ticker": ticker,
            "observed_price": observed.to_numpy(),
            "history_days": history_days.to_numpy(),
            "listing_age_days": listing_age_days,
            "stale_price_days": stale_days,
            "max_missing_gap_21d": rolling_missing_gap.to_numpy(),
            "insufficient_price_history": history_days.to_numpy() < cfg.quality.min_price_history_days,
            "insufficient_post_listing_history": listing_age_days < cfg.quality.ipo_buffer_days,
            "stale_price": stale_days > cfg.quality.stale_price_days,
            "missing_gap_fail": rolling_missing_gap.to_numpy() > cfg.quality.max_forward_fill_days,
        })
        quality_frames.append(quality)

        aligned = aligned.reset_index().rename(columns={"index": "date"})
        aligned["was_observed"] = observed.to_numpy()
        cleaned_frames.append(aligned[["date", "ticker", *PRICE_COLUMNS, "volume", "source", "was_observed"]])

    cleaned = pd.concat(cleaned_frames, ignore_index=True)
    quality = pd.concat(quality_frames, ignore_index=True)
    return cleaned, quality
