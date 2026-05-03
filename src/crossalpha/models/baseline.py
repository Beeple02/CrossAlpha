"""Baseline logistic regression model."""

from __future__ import annotations

import pandas as pd

from crossalpha.config import BaselineModelConfig
from crossalpha.models.base import OptionalDependencyError


class LogisticBaselineModel:
    def __init__(self, cfg: BaselineModelConfig) -> None:
        self.cfg = cfg
        self.backend = "sklearn_logistic_regression"
        self.model = None

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> "LogisticBaselineModel":
        try:
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:  # pragma: no cover - dependency boundary
            raise OptionalDependencyError("scikit-learn is required for the baseline model.") from exc

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=self.cfg.C,
                max_iter=self.cfg.max_iter,
                class_weight="balanced",
                random_state=7,
            )),
        ])
        pipeline.fit(x_train, y_train)
        self.model = pipeline
        return self

    def predict_scores(self, x: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("The baseline model has not been fit.")
        probabilities = self.model.predict_proba(x)[:, 1]
        return pd.Series(probabilities, index=x.index)

    def feature_importances(self, feature_columns: list[str]) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("The baseline model has not been fit.")
        coefficients = self.model.named_steps["model"].coef_[0]
        return pd.DataFrame({
            "feature": feature_columns,
            "importance": coefficients,
        }).sort_values("importance", key=lambda series: series.abs(), ascending=False).reset_index(drop=True)
