"""Shared model utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, UTC
from pathlib import Path
import pickle

import pandas as pd


class OptionalDependencyError(RuntimeError):
    """Raised when an optional ML dependency is missing."""


@dataclass(slots=True)
class ModelMetadata:
    horizon: str
    model_name: str
    backend: str
    feature_columns: list[str]
    trained_rows: int
    train_start: str
    train_end: str
    trained_at_utc: str

    @classmethod
    def create(
        cls,
        horizon: str,
        model_name: str,
        backend: str,
        feature_columns: list[str],
        trained_rows: int,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
    ) -> "ModelMetadata":
        return cls(
            horizon=horizon,
            model_name=model_name,
            backend=backend,
            feature_columns=feature_columns,
            trained_rows=trained_rows,
            train_start=str(train_start.date()),
            train_end=str(train_end.date()),
            trained_at_utc=datetime.now(UTC).isoformat(),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def save_pickle(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(payload))


def load_pickle(path: Path) -> object:
    return pickle.loads(path.read_bytes())
