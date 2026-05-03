"""Final model training and persistence."""

from __future__ import annotations

import logging

import pandas as pd

from crossalpha.config import ProjectConfig
from crossalpha.data.storage import DataCatalog
from crossalpha.models.base import ModelMetadata, save_pickle
from crossalpha.models.baseline import LogisticBaselineModel
from crossalpha.models.ranker import MainRankerModel
from crossalpha.utils.io import read_json, read_parquet, write_json, write_parquet


LOGGER = logging.getLogger(__name__)

NON_FEATURE_COLUMNS = {
    "date",
    "ticker",
    "sector",
    "industry",
    "source",
    "was_observed",
    "observed_price",
    "history_days",
    "listing_age_days",
    "stale_price_days",
    "max_missing_gap_21d",
    "insufficient_price_history",
    "insufficient_post_listing_history",
    "stale_price",
    "missing_gap_fail",
    "is_member",
    "membership_reliability",
    "universe_reliable",
    "first_membership_date",
    "membership_days",
    "universe_history_fail",
    "feature_missing_fail",
    "fundamentals_missing_fail",
    "trainable",
    "missing_feature_ratio",
    "filed_at",
    "period_end",
    "market_cap",
    "enterprise_value",
    "revenue_ttm",
    "net_income_ttm",
    "operating_income_ttm",
    "depreciation_ttm",
    "ebitda_ttm",
    "total_assets",
    "total_liabilities",
    "stockholders_equity",
    "cash_and_equivalents",
    "common_shares_outstanding",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "spx_close",
    "spx_ret_1d",
    "spx_ret_5d",
    "spx_ret_63d",
    "label_end_date",
    "forward_return",
    "forward_excess_return",
    "daily_median_forward_return",
    "target_rank",
    "target_class",
    "earnings_in_window",
    "label_truncated",
    "horizon",
    "horizon_days",
}


def train_final_models(cfg: ProjectConfig) -> dict[str, dict[str, str]]:
    catalog = DataCatalog(cfg)
    LOGGER.info("Training final models up to holdout start %s.", cfg.validation.holdout_start)

    model_frame = build_model_frame(cfg)
    train_cutoff = pd.Timestamp(cfg.validation.holdout_start)
    outputs: dict[str, dict[str, str]] = {}

    for horizon in cfg.horizons:
        LOGGER.info("Starting final training for horizon %s.", horizon)
        horizon_frame = model_frame[
            (model_frame["horizon"] == horizon)
            & (model_frame["date"] < train_cutoff)
            & model_frame["trainable"]
            & model_frame["target_rank"].notna()
            & model_frame["target_class"].notna()
        ].copy()

        if horizon_frame.empty:
            LOGGER.warning("Skipping %s final training because no rows were available.", horizon)
            continue

        feature_columns = infer_feature_columns(horizon_frame)
        x_train = horizon_frame[feature_columns]
        y_train_rank = horizon_frame["target_rank"]
        y_train_class = horizon_frame["target_class"].astype(int)
        group_train = horizon_frame.groupby("date").size().tolist()

        baseline = LogisticBaselineModel(cfg.model.baseline).fit(x_train, y_train_class)
        ranker = MainRankerModel(cfg.model.ranker).fit(x_train, y_train_rank, y_train_class, group_train)

        baseline_path = catalog.model(f"{horizon}_baseline.pkl")
        ranker_path = catalog.model(f"{horizon}_ranker.pkl")
        baseline_metadata = ModelMetadata.create(
            horizon=horizon,
            model_name="baseline",
            backend=baseline.backend,
            feature_columns=feature_columns,
            trained_rows=len(horizon_frame),
            train_start=horizon_frame["date"].min(),
            train_end=horizon_frame["date"].max(),
        )
        ranker_metadata = ModelMetadata.create(
            horizon=horizon,
            model_name="ranker",
            backend=ranker.backend,
            feature_columns=feature_columns,
            trained_rows=len(horizon_frame),
            train_start=horizon_frame["date"].min(),
            train_end=horizon_frame["date"].max(),
        )

        save_pickle({"model": baseline, "metadata": baseline_metadata.to_dict()}, baseline_path)
        save_pickle({"model": ranker, "metadata": ranker_metadata.to_dict()}, ranker_path)
        outputs[horizon] = {
            "baseline": str(baseline_path),
            "ranker": str(ranker_path),
        }

        write_json(baseline_metadata.to_dict(), catalog.model(f"{horizon}_baseline_metadata.json"))
        write_json(ranker_metadata.to_dict(), catalog.model(f"{horizon}_ranker_metadata.json"))
        LOGGER.info(
            "Finished horizon %s final training. Baseline -> %s | Ranker -> %s",
            horizon,
            baseline_path,
            ranker_path,
        )

    write_json(outputs, catalog.report("final_model_registry.json"))
    LOGGER.info("Completed final model training for %s horizons.", len(outputs))
    return outputs


def build_model_frame(cfg: ProjectConfig) -> pd.DataFrame:
    catalog = DataCatalog(cfg)
    artifact = catalog.processed("model_frame")
    if artifact.exists():
        return read_parquet(artifact)

    features = read_parquet(catalog.processed("feature_store"))
    labels = read_parquet(catalog.processed("label_store"))
    model_frame = features.merge(labels, on=["date", "ticker"], how="inner")
    write_parquet(model_frame, artifact)
    return model_frame


def infer_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column for column in frame.columns
        if column not in NON_FEATURE_COLUMNS
    ]
