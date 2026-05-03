"""Small helpers for report writing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from crossalpha.utils.io import write_json, write_parquet


def write_table(df: pd.DataFrame, path: Path) -> None:
    write_parquet(df, path)


def write_summary(summary: dict[str, object], path: Path) -> None:
    write_json(summary, path)
