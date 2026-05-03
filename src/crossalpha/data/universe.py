"""Universe construction utilities."""

from __future__ import annotations

import pandas as pd

from crossalpha.config import ProjectConfig


def build_daily_universe(
    membership_history: pd.DataFrame,
    metadata: pd.DataFrame,
    cfg: ProjectConfig,
) -> pd.DataFrame:
    universe = membership_history.copy()
    universe["date"] = pd.to_datetime(universe["date"])
    universe["ticker"] = universe["ticker"].astype(str)

    metadata = metadata.copy()
    if "ticker" in metadata.columns:
        metadata["ticker"] = metadata["ticker"].astype(str)

    universe = universe.merge(
        metadata[["ticker", "sector", "industry"]].drop_duplicates(subset=["ticker"]),
        on="ticker",
        how="left",
    )
    universe["sector"] = universe["sector"].fillna("Unknown")
    universe["industry"] = universe["industry"].fillna("Unknown")
    universe["universe_reliable"] = universe["membership_reliability"] >= cfg.universe.membership_min_reliability
    universe["first_membership_date"] = universe.groupby("ticker")["date"].transform("min")
    universe["membership_days"] = universe.groupby("ticker").cumcount() + 1
    return universe.sort_values(["date", "ticker"]).reset_index(drop=True)
