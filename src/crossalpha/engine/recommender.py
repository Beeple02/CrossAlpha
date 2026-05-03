"""Recommendation scoring and decision logic."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from crossalpha.config import ProjectConfig
from crossalpha.data.storage import DataCatalog
from crossalpha.models.base import load_pickle
from crossalpha.utils.io import read_parquet, write_parquet


LOGGER = logging.getLogger(__name__)


def generate_recommendations(
    cfg: ProjectConfig,
    start_date: str | None = None,
    end_date: str | None = None,
    latest_only: bool = False,
) -> pd.DataFrame:
    catalog = DataCatalog(cfg)
    features = read_parquet(catalog.processed("feature_store"))
    features["date"] = pd.to_datetime(features["date"])

    selected = _slice_feature_dates(features, start_date=start_date, end_date=end_date, latest_only=latest_only)
    scored_frames: list[pd.DataFrame] = []

    for horizon in cfg.horizons:
        payload = load_pickle(catalog.model(f"{horizon}_ranker.pkl"))
        model = payload["model"]
        metadata = payload["metadata"]
        feature_columns = metadata["feature_columns"]

        horizon_frame = selected.copy()
        horizon_frame["horizon"] = horizon
        horizon_frame["score"] = model.predict_scores(horizon_frame[feature_columns]).to_numpy()
        scored_frames.append(horizon_frame)

    scored = pd.concat(scored_frames, ignore_index=True)
    decisions = apply_recommendation_logic(scored, cfg)
    output_name = "latest_recommendations.parquet" if latest_only else "recommendations.parquet"
    write_parquet(decisions, catalog.report(output_name))
    LOGGER.info("Generated %s recommendation rows.", len(decisions))
    return decisions


def apply_recommendation_logic(scored_frame: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (date, horizon), frame in scored_frame.groupby(["date", "horizon"], sort=False):
        frame = frame.copy()
        regime = _regime_state(frame, cfg)
        frame["regime_state"] = regime
        frame["not_enough_data_reason"] = frame.apply(_not_enough_data_reason, axis=1)
        invalid = frame["not_enough_data_reason"].ne("")
        frame["decision"] = np.where(invalid, "NOT_ENOUGH_DATA", "NO_BUY")
        frame["confidence"] = None
        frame["rank"] = np.nan

        valid = frame.loc[~invalid].sort_values("score", ascending=False).copy()
        if not valid.empty:
            valid["rank"] = np.arange(1, len(valid) + 1)
            score_std = valid["score"].std(ddof=0)
            score_mean = valid["score"].mean()
            zscore = (valid["score"] - score_mean) / (score_std if score_std not in (0, np.nan) else 1.0)
            valid["confidence"] = np.select(
                [zscore >= 2.0, zscore >= 1.0],
                ["HIGH", "MEDIUM"],
                default="LOW",
            )

            if regime == "CLOSED":
                pass
            else:
                buy_count = max(cfg.recommendation.min_buy_count, math.floor(len(valid) * cfg.recommendation.buy_fraction))
                if regime == "CAUTION":
                    buy_count = min(buy_count, cfg.recommendation.caution_max_buys)
                buy_index = valid.head(buy_count).index
                frame.loc[buy_index, "decision"] = "BUY"
            frame.loc[valid.index, "rank"] = valid["rank"]
            frame.loc[valid.index, "confidence"] = valid["confidence"]

        rows.append(frame)

    decisions = pd.concat(rows, ignore_index=True)
    consensus = decisions.assign(is_buy=decisions["decision"].eq("BUY")).groupby(["date", "ticker"], observed=True)["is_buy"].sum().rename("buy_horizon_count")
    decisions = decisions.merge(consensus, on=["date", "ticker"], how="left")
    decisions["consensus_buy"] = decisions["buy_horizon_count"].fillna(0) >= 3

    keep = [
        "date",
        "ticker",
        "horizon",
        "score",
        "rank",
        "decision",
        "confidence",
        "regime_state",
        "consensus_buy",
        "buy_horizon_count",
        "not_enough_data_reason",
        "target_class",
        "forward_return",
        "forward_excess_return",
        "daily_median_forward_return",
        "sector",
        "industry",
    ]
    available = [column for column in keep if column in decisions.columns]
    return decisions[available].sort_values(["date", "horizon", "rank", "ticker"], na_position="last").reset_index(drop=True)


def _slice_feature_dates(
    features: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
    latest_only: bool,
) -> pd.DataFrame:
    if latest_only or (start_date is None and end_date is None):
        latest_date = features["date"].max()
        return features[features["date"] == latest_date].copy()

    frame = features.copy()
    if start_date is not None:
        frame = frame[frame["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        frame = frame[frame["date"] <= pd.Timestamp(end_date)]
    return frame


def _regime_state(frame: pd.DataFrame, cfg: ProjectConfig) -> str:
    vix_close = float(frame["vix_close"].iloc[0]) if "vix_close" in frame.columns else np.nan
    spx_above_200d = float(frame["spx_above_200d"].iloc[0]) if "spx_above_200d" in frame.columns else 0.0

    if pd.notna(vix_close) and vix_close >= cfg.recommendation.closed_vix:
        return "CLOSED"
    if pd.notna(vix_close) and vix_close >= cfg.recommendation.caution_vix and spx_above_200d < 1:
        return "CAUTION"
    return "OPEN"


def _not_enough_data_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if bool(row.get("insufficient_price_history", False)):
        reasons.append("short_price_history")
    if bool(row.get("missing_gap_fail", False)):
        reasons.append("recent_missing_price_gap")
    if bool(row.get("feature_missing_fail", False)):
        reasons.append("incomplete_feature_set")
    if bool(row.get("stale_price", False)):
        reasons.append("stale_price")
    if bool(row.get("insufficient_post_listing_history", False)):
        reasons.append("post_listing_buffer")
    if bool(row.get("fundamentals_missing_fail", False)):
        reasons.append("stale_or_missing_fundamentals")
    if bool(row.get("universe_history_fail", False)):
        reasons.append("unreliable_universe_history")
    return "|".join(reasons)
