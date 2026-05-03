"""Primary ranking model with LightGBM and a classifier fallback."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crossalpha.config import RankerModelConfig
from crossalpha.models.base import OptionalDependencyError


class MainRankerModel:
    def __init__(self, cfg: RankerModelConfig) -> None:
        self.cfg = cfg
        self.backend = "uninitialized"
        self.model = None

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train_rank: pd.Series,
        y_train_class: pd.Series,
        group_train: list[int],
        x_val: pd.DataFrame | None = None,
        y_val_rank: pd.Series | None = None,
        y_val_class: pd.Series | None = None,
        group_val: list[int] | None = None,
    ) -> "MainRankerModel":
        lightgbm_error: Exception | None = None
        if self.cfg.backend == "lightgbm":
            try:
                from lightgbm import LGBMRanker

                self.backend = "lightgbm_lambdarank"
                model = LGBMRanker(
                    objective="lambdarank",
                    metric="ndcg",
                    ndcg_eval_at=[10, 20, 50],
                    num_leaves=self.cfg.num_leaves,
                    learning_rate=self.cfg.learning_rate,
                    feature_fraction=self.cfg.feature_fraction,
                    bagging_fraction=self.cfg.bagging_fraction,
                    bagging_freq=self.cfg.bagging_freq,
                    min_child_samples=self.cfg.min_child_samples,
                    max_depth=self.cfg.max_depth,
                    n_estimators=self.cfg.n_estimators,
                    random_state=7,
                    verbose=-1,
                )
                y_train = np.clip((y_train_rank.fillna(0) * 100).round().astype(int), 0, 100)
                fit_kwargs: dict[str, object] = {"group": group_train}
                if x_val is not None and y_val_rank is not None and group_val is not None:
                    y_validation = np.clip((y_val_rank.fillna(0) * 100).round().astype(int), 0, 100)
                    fit_kwargs["eval_set"] = [(x_val, y_validation)]
                    fit_kwargs["eval_group"] = [group_val]
                    fit_kwargs["callbacks"] = []
                model.fit(x_train, y_train, **fit_kwargs)
                self.model = model
                return self
            except ImportError as exc:  # pragma: no cover - dependency boundary
                lightgbm_error = exc

        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
        except ImportError as exc:  # pragma: no cover - dependency boundary
            if lightgbm_error is not None:
                raise OptionalDependencyError(
                    "Install lightgbm or scikit-learn to use the main CrossAlpha model."
                ) from lightgbm_error
            raise OptionalDependencyError("scikit-learn is required for the main model fallback.") from exc

        self.backend = "sklearn_hist_gradient_boosting"
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                learning_rate=self.cfg.learning_rate,
                max_depth=self.cfg.max_depth if self.cfg.max_depth > 0 else None,
                max_iter=self.cfg.n_estimators,
                random_state=7,
            )),
        ])
        model.fit(x_train, y_train_class)
        self.model = model
        return self

    def predict_scores(self, x: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("The main model has not been fit.")
        if self.backend == "lightgbm_lambdarank":
            scores = self.model.predict(x)
        else:
            scores = self.model.predict_proba(x)[:, 1]
        return pd.Series(scores, index=x.index)

    def feature_importances(self, feature_columns: list[str]) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("The main model has not been fit.")

        if hasattr(self.model, "feature_importances_"):
            values = self.model.feature_importances_
        elif hasattr(self.model, "named_steps") and hasattr(self.model.named_steps["model"], "feature_importances_"):
            values = self.model.named_steps["model"].feature_importances_
        else:
            values = np.zeros(len(feature_columns))

        return pd.DataFrame({
            "feature": feature_columns,
            "importance": values,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
