from __future__ import annotations

import pytest

from crossalpha.features.engine import build_feature_store
from crossalpha.labels.engine import build_label_store
from crossalpha.validation.walkforward import run_validation


def test_validation_generates_summary(synthetic_project):
    pytest.importorskip("sklearn")
    cfg = synthetic_project
    build_feature_store(cfg)
    build_label_store(cfg)
    summary = run_validation(cfg)
    assert summary["status"] == "ok"
    assert summary["rows"]
