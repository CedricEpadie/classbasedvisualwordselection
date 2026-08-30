"""Builds the final comparative table (Dataset x Approach x Classifier x
Metrics) from the list of result rows produced by `PipelineRunner.run` /
`run_all` / `pipeline.run_for_all_datasets`.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import List

import pandas as pd

from src.config import PipelineConfig
from src.utils.io_utils import ensure_dir


def build_comparison_table(rows: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # "dataset" is only present when rows come from `run_for_all_datasets`
    # (multi-dataset mode); keep it first when it exists instead of letting
    # it fall alphabetically among the metric columns.
    preferred_cols = [c for c in ["dataset", "approach", "feature_variant", "classifier"] if c in df.columns]
    metric_cols = [c for c in df.columns if c not in preferred_cols]
    return df[preferred_cols + sorted(metric_cols)].sort_values(preferred_cols).reset_index(drop=True)


def save_comparison_table(df: pd.DataFrame, cfg: PipelineConfig) -> Path:
    out_dir = ensure_dir(Path(cfg.paths.output_dir) / "comparison")
    out_path = out_dir / f"comparison_{cfg.dataset_name}_{cfg.runtime.run_id}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def save_multi_dataset_comparison_table(df: pd.DataFrame, output_dir: str) -> Path:
    """Save the aggregated table from a `--datasets-root` run. Unlike
    `save_comparison_table`, there's no single dataset/run_id to key the
    filename off, so a timestamp is used instead — this only affects the
    name of this one summary artifact, not any cached pipeline step, which
    stays keyed per-dataset as usual."""
    out_dir = ensure_dir(Path(output_dir) / "comparison")
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"comparison_all_datasets_{ts}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def print_comparison_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("No results to display.")
        return
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 160):
        print(df.to_string(index=False))
