"""Purged walk-forward model validation."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from crossalpha.config import ProjectConfig
from crossalpha.data.storage import DataCatalog
from crossalpha.models.baseline import LogisticBaselineModel
from crossalpha.models.ranker import MainRankerModel
from crossalpha.models.training import build_model_frame, infer_feature_columns
from crossalpha.utils.io import write_json, write_parquet
from crossalpha.validation.metrics import calibration_table, class_imbalance_diagnostics, compute_ranking_metrics, feature_drift_table
from crossalpha.validation.splits import build_walkforward_splits, split_frame_for_horizon


LOGGER = logging.getLogger(__name__)


def run_validation(cfg: ProjectConfig) -> dict[str, object]:
    catalog = DataCatalog(cfg)
    LOGGER.info("Running walk-forward validation.")

    model_frame = build_model_frame(cfg)
    splits = build_walkforward_splits(cfg)

    split_metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    drift_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    for horizon in cfg.horizons:
        LOGGER.info("Starting validation for horizon %s.", horizon)
        horizon_frame = model_frame[
            (model_frame["horizon"] == horizon)
            & model_frame["trainable"]
            & model_frame["target_rank"].notna()
            & model_frame["target_class"].notna()
            & ~model_frame["label_truncated"].fillna(True)
        ].copy()
        if horizon_frame.empty:
            LOGGER.warning("Skipping validation for %s because no eligible rows were found.", horizon)
            continue

        feature_columns = infer_feature_columns(horizon_frame)
        for split in splits:
            LOGGER.info("Validating horizon %s on %s.", horizon, split.name)
            train_df, val_df = split_frame_for_horizon(horizon_frame, split, cfg.validation.embargo_days)
            if train_df.empty or val_df.empty:
                LOGGER.warning("Skipping %s %s because the train or validation frame was empty.", horizon, split.name)
                continue

            train_df, val_df, eval_feature_columns = _attach_noise_features(train_df, val_df, feature_columns, cfg.features.random_noise_feature_count)
            drift = feature_drift_table(train_df, val_df, eval_feature_columns)
            drift["horizon"] = horizon
            drift["split"] = split.name
            drift_frames.append(drift)

            split_metrics_rows.extend(_evaluate_one_model(
                model_name="baseline",
                model=LogisticBaselineModel(cfg.model.baseline),
                feature_columns=eval_feature_columns,
                train_df=train_df,
                val_df=val_df,
                horizon=horizon,
                split_name=split.name,
                cfg=cfg,
                prediction_frames=prediction_frames,
                calibration_frames=calibration_frames,
                importance_frames=importance_frames,
            ))
            split_metrics_rows.extend(_evaluate_one_model(
                model_name="ranker",
                model=MainRankerModel(cfg.model.ranker),
                feature_columns=eval_feature_columns,
                train_df=train_df,
                val_df=val_df,
                horizon=horizon,
                split_name=split.name,
                cfg=cfg,
                prediction_frames=prediction_frames,
                calibration_frames=calibration_frames,
                importance_frames=importance_frames,
            ))

    split_metrics = pd.DataFrame(split_metrics_rows)
    if prediction_frames:
        predictions = pd.concat(prediction_frames, ignore_index=True)
        write_parquet(predictions, catalog.processed("validation_predictions"))
    if drift_frames:
        write_parquet(pd.concat(drift_frames, ignore_index=True), catalog.report("feature_drift.parquet"))
    if calibration_frames:
        write_parquet(pd.concat(calibration_frames, ignore_index=True), catalog.report("calibration.parquet"))
    if importance_frames:
        write_parquet(pd.concat(importance_frames, ignore_index=True), catalog.report("feature_importance.parquet"))
    write_parquet(split_metrics, catalog.report("validation_split_metrics.parquet"))

    summary = _summarize_metrics(split_metrics)
    write_json(summary, catalog.report("validation_summary.json"))
    LOGGER.info("Validation complete.")
    return summary


def _evaluate_one_model(
    model_name: str,
    model: LogisticBaselineModel | MainRankerModel,
    feature_columns: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    horizon: str,
    split_name: str,
    cfg: ProjectConfig,
    prediction_frames: list[pd.DataFrame],
    calibration_frames: list[pd.DataFrame],
    importance_frames: list[pd.DataFrame],
) -> list[dict[str, object]]:
    x_train = train_df[feature_columns]
    x_val = val_df[feature_columns]
    y_train_rank = train_df["target_rank"]
    y_train_class = train_df["target_class"].astype(int)
    y_val_rank = val_df["target_rank"]
    y_val_class = val_df["target_class"].astype(int)
    group_train = train_df.groupby("date").size().tolist()
    group_val = val_df.groupby("date").size().tolist()

    if model_name == "baseline":
        model.fit(x_train, y_train_class)
    else:
        model.fit(x_train, y_train_rank, y_train_class, group_train, x_val, y_val_rank, y_val_class, group_val)

    scores = model.predict_scores(x_val)
    predictions = val_df[[
        "date",
        "ticker",
        "horizon",
        "forward_return",
        "forward_excess_return",
        "daily_median_forward_return",
        "target_rank",
        "target_class",
        "earnings_in_window",
    ]].copy()
    predictions["score"] = scores.to_numpy()
    predictions["model_name"] = model_name
    predictions["split"] = split_name
    prediction_frames.append(predictions)

    metrics = compute_ranking_metrics(predictions, cfg.recommendation.buy_fraction)
    metrics.update(class_imbalance_diagnostics(val_df))
    metrics.update({
        "horizon": horizon,
        "split": split_name,
        "model_name": model_name,
        "rows": len(val_df),
        "train_rows": len(train_df),
    })

    calibration = calibration_table(predictions)
    calibration["horizon"] = horizon
    calibration["split"] = split_name
    calibration["model_name"] = model_name
    calibration_frames.append(calibration)

    importance = model.feature_importances(feature_columns)
    importance["horizon"] = horizon
    importance["split"] = split_name
    importance["model_name"] = model_name
    importance_frames.append(importance)
    LOGGER.info(
        "Finished %s %s %s | NDCG@20=%.4f | Precision@10=%.4f | HitRate=%.4f",
        horizon,
        split_name,
        model_name,
        metrics["ndcg_at_20"],
        metrics["precision_at_10"],
        metrics["hit_rate"],
    )
    return [metrics]


def _attach_noise_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list[str],
    count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if count <= 0:
        return train_df, val_df, feature_columns

    train_df = train_df.copy()
    val_df = val_df.copy()
    rng = np.random.default_rng(7)
    noise_columns: list[str] = []
    for index in range(count):
        column = f"noise_feature_{index + 1}"
        train_df[column] = rng.normal(size=len(train_df))
        val_df[column] = rng.normal(size=len(val_df))
        noise_columns.append(column)
    return train_df, val_df, feature_columns + noise_columns


def _summarize_metrics(split_metrics: pd.DataFrame) -> dict[str, object]:
    if split_metrics.empty:
        return {"status": "no_results"}

    grouped = split_metrics.groupby(["horizon", "model_name"], observed=True).agg({
        "ndcg_at_10": "mean",
        "ndcg_at_20": "mean",
        "precision_at_10": "mean",
        "hit_rate": "mean",
        "mean_excess_return": "mean",
        "positive_rate": "mean",
        "rows": "sum",
    }).reset_index()

    return {
        "status": "ok",
        "rows": grouped.to_dict(orient="records"),
    }
