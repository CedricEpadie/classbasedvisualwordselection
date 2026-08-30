"""Unit tests for the metrics registry / computation, decoupled from any
actual model training (per spec section 5): we build a fake predictions
DataFrame directly.
"""
import numpy as np
import pandas as pd
import pytest

from src.metrics_engine import (
    _REGISTRY,
    available_metrics,
    compute_metrics_from_predictions,
    register_metric,
)


@pytest.fixture
def predictions_path(tmp_path):
    df = pd.DataFrame(
        {
            "image_id": ["a", "b", "c", "d"],
            "true_label": ["cat", "dog", "cat", "dog"],
            "predicted_label": ["cat", "dog", "dog", "dog"],
            "predicted_proba_cat": [0.9, 0.2, 0.4, 0.1],
            "predicted_proba_dog": [0.1, 0.8, 0.6, 0.9],
        }
    )
    path = tmp_path / "preds.parquet"
    df.to_parquet(path, index=False)
    return path


def test_accuracy(predictions_path):
    metrics = compute_metrics_from_predictions(predictions_path, ["accuracy"])
    assert metrics["accuracy"] == pytest.approx(3 / 4)


def test_all_default_metrics_present(predictions_path):
    metrics = compute_metrics_from_predictions(predictions_path)
    for name in available_metrics():
        assert name in metrics


def test_auc_uses_proba_columns(predictions_path):
    metrics = compute_metrics_from_predictions(predictions_path, ["auc_macro_ovr"])
    assert 0.0 <= metrics["auc_macro_ovr"] <= 1.0


def test_unknown_metric_raises(predictions_path):
    with pytest.raises(KeyError):
        compute_metrics_from_predictions(predictions_path, ["not_a_real_metric"])


def test_new_metric_registers_without_touching_training_code(predictions_path):
    """Demonstrates the Registry pattern requirement from spec section 5:
    adding a metric is a pure addition, no training code involved."""

    @register_metric("always_one")
    def _always_one(y_true, y_pred, y_proba=None):
        return 1.0

    try:
        metrics = compute_metrics_from_predictions(predictions_path, ["always_one"])
        assert metrics["always_one"] == 1.0
        assert "always_one" in available_metrics()
    finally:
        _REGISTRY.pop("always_one", None)


def test_metric_gracefully_handles_missing_proba(tmp_path):
    df = pd.DataFrame(
        {
            "image_id": ["a", "b"],
            "true_label": ["cat", "dog"],
            "predicted_label": ["cat", "dog"],
        }
    )
    path = tmp_path / "no_proba.parquet"
    df.to_parquet(path, index=False)
    metrics = compute_metrics_from_predictions(path, ["accuracy", "auc_macro_ovr"])
    assert metrics["accuracy"] == 1.0
    assert np.isnan(metrics["auc_macro_ovr"])
