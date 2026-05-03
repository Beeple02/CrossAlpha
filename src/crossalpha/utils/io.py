"""I/O helpers for Parquet and JSON artifacts."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import pandas as pd


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    df.to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_json(payload: dict[str, Any] | list[Any], path: Path) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
