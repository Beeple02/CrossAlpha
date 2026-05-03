from __future__ import annotations

import pytest

from crossalpha.backtest.simulator import run_backtest
from crossalpha.engine.recommender import generate_recommendations
from crossalpha.features.engine import build_feature_store
from crossalpha.labels.engine import build_label_store
from crossalpha.models.training import train_final_models
from crossalpha.validation.walkforward import run_validation


def test_end_to_end_pipeline_runs_on_synthetic_data(synthetic_project):
    pytest.importorskip("sklearn")
    cfg = synthetic_project
    build_feature_store(cfg)
    build_label_store(cfg)
    validation_summary = run_validation(cfg)
    registry = train_final_models(cfg)
    recommendations = generate_recommendations(cfg, latest_only=True)
    backtest_summary = run_backtest(cfg)

    assert validation_summary["status"] == "ok"
    assert registry
    assert set(recommendations["decision"].unique()).issubset({"BUY", "NO_BUY", "NOT_ENOUGH_DATA"})
    assert backtest_summary["rows"]
