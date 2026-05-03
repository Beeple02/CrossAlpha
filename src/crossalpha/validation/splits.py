"""Purged walk-forward split helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from crossalpha.config import ProjectConfig


@dataclass(slots=True)
class WalkForwardSplit:
    name: str
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def build_walkforward_splits(cfg: ProjectConfig) -> list[WalkForwardSplit]:
    splits: list[WalkForwardSplit] = []
    for index, raw_split in enumerate(cfg.validation.splits, start=1):
        splits.append(WalkForwardSplit(
            name=f"split_{index}",
            train_end=pd.Timestamp(raw_split.train_end),
            val_start=pd.Timestamp(raw_split.val_start),
            val_end=pd.Timestamp(raw_split.val_end),
        ))
    return splits


def validate_embargo(split: WalkForwardSplit, embargo_days: int) -> None:
    gap = len(pd.bdate_range(split.train_end, split.val_start)) - 1
    if gap < embargo_days:
        raise ValueError(
            f"Split {split.name} violates the embargo. Gap={gap} business days, required={embargo_days}."
        )


def split_frame_for_horizon(
    frame: pd.DataFrame,
    split: WalkForwardSplit,
    embargo_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_embargo(split, embargo_days)
    train_mask = (
        (frame["date"] <= split.train_end)
        & (frame["label_end_date"] < split.val_start)
    )
    val_mask = (frame["date"] >= split.val_start) & (frame["date"] <= split.val_end)
    return frame.loc[train_mask].copy(), frame.loc[val_mask].copy()
