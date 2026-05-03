"""Artifact path helpers and shared storage utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from crossalpha.config import ProjectConfig


@dataclass(slots=True)
class DataCatalog:
    cfg: ProjectConfig

    def raw(self, name: str) -> Path:
        return self.cfg.raw_dir / f"{name}.parquet"

    def processed(self, name: str) -> Path:
        return self.cfg.processed_dir / f"{name}.parquet"

    def model(self, name: str) -> Path:
        return self.cfg.models_dir / name

    def report(self, name: str) -> Path:
        return self.cfg.reports_dir / name


def dedupe_sort(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    return df.drop_duplicates(subset=subset, keep="last").sort_values(subset).reset_index(drop=True)
