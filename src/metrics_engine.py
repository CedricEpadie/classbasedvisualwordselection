"""Metrics engine — fully decoupled from training.

Training only ever writes raw predictions (`image_id, true_label,
predicted_label, predicted_proba_<class>...`) to Parquet. This module reads
those files and computes whatever metrics are registered, so adding a new
metric never touches the training code — only this registry.

Usage as a script:
    python -m src.metrics_engine --recompute-all --output-dir outputs
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

from src.utils.io_utils import atomic_write_json

MetricFn = Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]
_REGISTRY: Dict[str, MetricFn] = {}


def register_metric(name: str):
    """Decorator: `@register_metric("f1_macro")` adds a metric function to
    the registry under that name. Signature: (y_true, y_pred, y_proba) -> float.
    `y_proba` may be None if the metric doesn't need it."""

    def _decorator(fn: MetricFn) -> MetricFn:
        _REGISTRY[name] = fn
        return fn

    return _decorator


@register_metric("accuracy")
def _accuracy(y_true, y_pred, y_proba=None) -> float:
    return float(accuracy_score(y_true, y_pred))


@register_metric("f1_macro")
def _f1_macro(y_true, y_pred, y_proba=None) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


@register_metric("recall_macro")
def _recall_macro(y_true, y_pred, y_proba=None) -> float:
    return float(recall_score(y_true, y_pred, average="macro", zero_division=0))


@register_metric("auc_macro_ovr")
def _auc_macro_ovr(y_true, y_pred, y_proba=None) -> float:
    if y_proba is None:
        raise ValueError("auc_macro_ovr requires predicted probabilities")
    classes = sorted(set(y_true) | set(y_pred))
    if len(classes) < 2:
        return float("nan")
    if y_proba.shape[1] == 2:
        return float(roc_auc_score(y_true, y_proba[:, 1]))
    return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=classes))


def available_metrics() -> List[str]:
    return sorted(_REGISTRY.keys())


def _extract_proba_columns(df: pd.DataFrame) -> tuple[Optional[np.ndarray], Optional[List[str]]]:
    proba_cols = sorted(c for c in df.columns if c.startswith("predicted_proba_"))
    if not proba_cols:
        return None, None
    class_names = [c[len("predicted_proba_") :] for c in proba_cols]
    return df[proba_cols].to_numpy(), class_names


def compute_metrics_from_predictions(
    predictions_path: str | Path, metric_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """Load one predictions Parquet file and compute the requested metrics
    (defaults to every registered metric)."""
    df = pd.read_parquet(predictions_path)
    y_true = df["true_label"].to_numpy()
    y_pred = df["predicted_label"].to_numpy()
    y_proba, _classes = _extract_proba_columns(df)

    names = metric_names or available_metrics()
    results: Dict[str, float] = {}
    for name in names:
        fn = _REGISTRY.get(name)
        if fn is None:
            raise KeyError(f"Unknown metric '{name}'. Available: {available_metrics()}")
        try:
            results[name] = fn(y_true, y_pred, y_proba)
        except Exception as exc:  # metric may legitimately fail (e.g. no proba)
            results[name] = float("nan")
            logging.getLogger(__name__).warning("Metric %s failed on %s: %s", name, predictions_path, exc)
    return results


def recompute_all(output_dir: str | Path, metric_names: Optional[List[str]] = None) -> None:
    """Recompute metrics for every predictions file already on disk,
    writing/overwriting the matching `metrics/<...>.json`, without
    retraining anything."""
    output_dir = Path(output_dir)
    predictions_dir = output_dir / "predictions"
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(glob.glob(str(predictions_dir / "*.parquet")))
    for pred_file in pred_files:
        stem = Path(pred_file).stem
        metrics = compute_metrics_from_predictions(pred_file, metric_names)
        out_path = metrics_dir / f"{stem}.json"
        atomic_write_json(out_path, metrics)
        print(f"[metrics_engine] {stem}: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recompute metrics from stored predictions.")
    parser.add_argument("--recompute-all", action="store_true")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--metrics", nargs="*", default=None, help="Subset of metrics to compute")
    args = parser.parse_args()

    if args.recompute_all:
        recompute_all(args.output_dir, args.metrics)
    else:
        parser.print_help()
