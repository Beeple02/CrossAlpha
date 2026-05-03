"""Forward-return label generation for all horizons."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from crossalpha.config import ProjectConfig
from crossalpha.data.storage import DataCatalog
from crossalpha.utils.io import read_parquet, write_parquet


LOGGER = logging.getLogger(__name__)


def build_label_store(cfg: ProjectConfig) -> pd.DataFrame:
    catalog = DataCatalog(cfg)
    LOGGER.info("Building label store.")

    prices = read_parquet(catalog.processed("prices_cleaned"))
    universe = read_parquet(catalog.processed("daily_universe"))
    earnings = read_parquet(catalog.processed("earnings"))

    labels = _build_labels(prices, universe, earnings, cfg)
    write_parquet(labels, catalog.processed("label_store"))
    LOGGER.info("Label store complete with %s rows.", len(labels))
    return labels


def _build_labels(
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    earnings: pd.DataFrame,
    cfg: ProjectConfig,
) -> pd.DataFrame:
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    universe = universe[universe["is_member"]].copy()
    universe["date"] = pd.to_datetime(universe["date"])

    price_panel = prices[["date", "ticker", "adj_close"]].copy()
    label_frames: list[pd.DataFrame] = []

    for horizon_name, horizon_days in cfg.horizons.items():
        horizon_panel = _forward_returns_by_horizon(price_panel, horizon_name, horizon_days)
        horizon_panel = horizon_panel.merge(
            universe[["date", "ticker", "membership_reliability", "universe_reliable"]],
            on=["date", "ticker"],
            how="inner",
        )
        horizon_panel["horizon"] = horizon_name
        horizon_panel["horizon_days"] = horizon_days
        horizon_panel["earnings_in_window"] = _flag_earnings_in_window(horizon_panel, earnings)

        valid_mask = horizon_panel["forward_return"].notna()
        horizon_panel.loc[valid_mask, "target_rank"] = horizon_panel.loc[valid_mask].groupby("date")["forward_return"].rank(
            pct=True,
            method="average",
        )
        horizon_panel.loc[valid_mask, "daily_median_forward_return"] = horizon_panel.loc[valid_mask].groupby("date")["forward_return"].transform("median")
        horizon_panel["forward_excess_return"] = horizon_panel["forward_return"] - horizon_panel["daily_median_forward_return"]
        horizon_panel["target_class"] = pd.Series(pd.NA, index=horizon_panel.index, dtype="Int64")
        horizon_panel.loc[valid_mask, "target_class"] = (
            horizon_panel.loc[valid_mask, "target_rank"] >= cfg.model.baseline.positive_class_quantile
        ).astype("Int64")
        label_frames.append(horizon_panel)

    labels = pd.concat(label_frames, ignore_index=True)
    keep = [
        "date",
        "ticker",
        "horizon",
        "horizon_days",
        "label_end_date",
        "forward_return",
        "forward_excess_return",
        "daily_median_forward_return",
        "target_rank",
        "target_class",
        "earnings_in_window",
        "label_truncated",
        "membership_reliability",
        "universe_reliable",
    ]
    return labels[keep].sort_values(["date", "ticker", "horizon"]).reset_index(drop=True)


def _forward_returns_by_horizon(price_panel: pd.DataFrame, horizon_name: str, horizon_days: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ticker, frame in price_panel.groupby("ticker", sort=False):
        frame = frame.sort_values("date").copy()
        frame["label_price"] = frame["adj_close"].ffill()
        frame["future_price"] = frame["label_price"].shift(-horizon_days)
        frame["label_end_date"] = frame["date"].shift(-horizon_days)
        frame["forward_return"] = frame["future_price"] / frame["label_price"] - 1.0
        frame["label_truncated"] = frame["future_price"].isna()
        frame["ticker"] = ticker
        frames.append(frame[["date", "ticker", "label_end_date", "forward_return", "label_truncated"]])
    return pd.concat(frames, ignore_index=True)


def _flag_earnings_in_window(horizon_panel: pd.DataFrame, earnings: pd.DataFrame) -> pd.Series:
    if earnings.empty:
        return pd.Series(False, index=horizon_panel.index)

    earnings = earnings.copy()
    earnings["earnings_date"] = pd.to_datetime(earnings["earnings_date"]).dt.normalize()

    flags = np.zeros(len(horizon_panel), dtype=bool)
    for ticker, indices in horizon_panel.groupby("ticker").groups.items():
        window = horizon_panel.loc[indices, ["date", "label_end_date"]]
        event_dates = earnings.loc[earnings["ticker"] == ticker, "earnings_date"].sort_values().to_numpy(dtype="datetime64[ns]")
        if len(event_dates) == 0:
            continue

        start = window["date"].to_numpy(dtype="datetime64[ns]")
        end = window["label_end_date"].to_numpy(dtype="datetime64[ns]")
        left = np.searchsorted(event_dates, start, side="right")
        right = np.searchsorted(event_dates, end, side="right")
        flags[np.array(list(indices))] = right > left
    return pd.Series(flags, index=horizon_panel.index)
