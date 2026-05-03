"""Validation metrics and diagnostics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from crossalpha.utils.math import population_stability_index


def compute_ranking_metrics(predictions: pd.DataFrame, buy_fraction: float) -> dict[str, float]:
    ndcg10: list[float] = []
    ndcg20: list[float] = []
    precision10: list[float] = []
    hit_rate: list[float] = []
    mean_excess_return: list[float] = []

    for _, frame in predictions.groupby("date", sort=False):
        frame = frame.sort_values("score", ascending=False)
        if frame.empty:
            continue

        ndcg10.append(_ndcg(frame["target_rank"].to_numpy(), frame["score"].to_numpy(), k=min(10, len(frame))))
        ndcg20.append(_ndcg(frame["target_rank"].to_numpy(), frame["score"].to_numpy(), k=min(20, len(frame))))

        top10 = frame.head(min(10, len(frame)))
        precision10.append(float(top10["target_class"].mean()))

        buy_count = max(1, math.floor(len(frame) * buy_fraction))
        buys = frame.head(buy_count)
        hit_rate.append(float((buys["forward_return"] > buys["daily_median_forward_return"]).mean()))
        mean_excess_return.append(float(buys["forward_excess_return"].mean()))

    return {
        "ndcg_at_10": float(np.nanmean(ndcg10)) if ndcg10 else np.nan,
        "ndcg_at_20": float(np.nanmean(ndcg20)) if ndcg20 else np.nan,
        "precision_at_10": float(np.nanmean(precision10)) if precision10 else np.nan,
        "hit_rate": float(np.nanmean(hit_rate)) if hit_rate else np.nan,
        "mean_excess_return": float(np.nanmean(mean_excess_return)) if mean_excess_return else np.nan,
    }


def calibration_table(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    frame = predictions[["score", "target_class"]].dropna().copy()
    if frame.empty:
        return pd.DataFrame(columns=["score_bin", "mean_score", "positive_rate", "count"])

    frame["score_rank"] = frame["score"].rank(method="first", pct=True)
    frame["score_bin"] = pd.qcut(frame["score_rank"], q=min(bins, frame["score_rank"].nunique()), duplicates="drop")
    output = frame.groupby("score_bin", observed=True).agg(
        mean_score=("score", "mean"),
        positive_rate=("target_class", "mean"),
        count=("target_class", "size"),
    ).reset_index()
    return output


def class_imbalance_diagnostics(frame: pd.DataFrame) -> dict[str, float]:
    positive_rate = frame["target_class"].mean()
    by_date = frame.groupby("date")["target_class"].mean()
    return {
        "positive_rate": float(positive_rate),
        "positive_rate_std_by_date": float(by_date.std(ddof=0)),
        "cross_sections": float(frame["date"].nunique()),
    }


def feature_drift_table(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for column in feature_columns:
        rows.append({
            "feature": column,
            "train_mean": float(train_df[column].mean()),
            "val_mean": float(val_df[column].mean()),
            "mean_shift": float(val_df[column].mean() - train_df[column].mean()),
            "psi": float(population_stability_index(train_df[column], val_df[column])),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def _ndcg(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if k <= 0:
        return np.nan
    order = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[order][:k]
    ideal_sorted = np.sort(y_true)[::-1][:k]

    discounts = 1.0 / np.log2(np.arange(2, len(y_true_sorted) + 2))
    dcg = np.sum((2 ** y_true_sorted - 1) * discounts)
    idcg = np.sum((2 ** ideal_sorted - 1) * discounts)
    if idcg == 0:
        return np.nan
    return float(dcg / idcg)
