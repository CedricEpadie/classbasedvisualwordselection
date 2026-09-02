#!/usr/bin/env python
"""CLI entry point: run one/several/all approaches, on one dataset or every
dataset under a directory, and produce the final comparison table (CSV +
formatted console output).

Runtime configurability (no config.yaml edits needed):
    # Just run MLP, no other classifier, on a given dataset
    python run_comparison.py --config config.yaml --dataset-dir data/dtd --classifiers mlp

    # Only the baseline and CNN+BoVW+selection approaches
    python run_comparison.py --config config.yaml --approaches bovw_baseline cnn_bovw_cvws

    # One specific approach only
    python run_comparison.py --config config.yaml --approach bovw_baseline

    # Any other field via a generic dotted-path override
    python run_comparison.py --config config.yaml --set vocabulary.k=128 --set selection.n_candidates_per_class=10

Re-running any of the above unchanged is cheap: every already-computed
artifact (preprocessing, features, vocabulary, histograms, selection,
per-classifier predictions/metrics) is cached under a deterministic run_id
derived from the dataset name and the exact config content (see
`src/config.ensure_run_id`), so nothing is recomputed unless something
that would actually change the result changed too.

Multi-dataset mode: run the whole comparison on every dataset found under
a directory, in one invocation:
    python run_comparison.py --config config.yaml --datasets-root data/all_datasets
expects `data/all_datasets/<dataset_name>/<class_name>/<image>` for every
dataset; each gets its own cache/output namespace (see
`src/pipeline.run_for_all_datasets`), and results are combined into one
CSV with a `dataset` column.
"""
from __future__ import annotations

import argparse
import sys

from src.classifiers import CLASSIFIER_REGISTRY
from src.comparison import (
    build_comparison_table,
    print_comparison_table,
    save_comparison_table,
    save_multi_dataset_comparison_table,
)
from src.config import apply_overrides, ensure_run_id, load_config
from src.pipeline import APPROACHES, PipelineRunner, run_for_all_datasets


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the BoVW/CNN comparison pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")

    # --- Dataset selection ------------------------------------------------
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--dataset-dir", default=None, help="Override paths.dataset_dir (single-dataset mode)."
    )
    dataset_group.add_argument(
        "--datasets-root",
        default=None,
        help=(
            "Run on every dataset found under this directory "
            "(<datasets_root>/<dataset_name>/<class_name>/<image>), in one invocation."
        ),
    )
    parser.add_argument(
        "--dataset-name", default=None, help="Override dataset_name (single-dataset mode only)."
    )

    # --- What to run --------------------------------------------------------
    parser.add_argument(
        "--approach",
        default=None,
        choices=APPROACHES,
        help="Run only this one approach (overrides --approaches / config approaches.enabled).",
    )
    parser.add_argument(
        "--approaches",
        nargs="+",
        default=None,
        choices=APPROACHES,
        metavar="APPROACH",
        help="Override approaches.enabled: which approaches to run when --approach isn't given.",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=None,
        choices=sorted(CLASSIFIER_REGISTRY),
        metavar="CLASSIFIER",
        help="Override classifiers.enabled: e.g. `--classifiers mlp` to run only MLP.",
    )
    parser.add_argument(
        "--selection-strategies",
        nargs="+",
        default=None,
        metavar="STRATEGY",
        help="Override selection.strategies (GCF / ICC / CED).",
    )
    parser.add_argument("--vocabulary-k", type=int, default=None, help="Override vocabulary.k.")
    parser.add_argument("--output-dir", default=None, help="Override paths.output_dir.")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Override runtime.run_id explicitly (single-dataset mode only). "
        "Default: deterministic id derived from the dataset name + config content, "
        "which is what makes re-running the framework on an unchanged config reuse the cache.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY.PATH=VALUE",
        help="Generic config override, repeatable, e.g. --set vocabulary.seed=7 --set runtime.test_size=0.3",
    )

    # --- Introspection ------------------------------------------------------
    parser.add_argument(
        "--list-approaches", action="store_true", help="Print the available approaches and exit."
    )
    parser.add_argument(
        "--list-classifiers", action="store_true", help="Print the available classifiers and exit."
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_approaches:
        print("\n".join(APPROACHES))
        return
    if args.list_classifiers:
        for name, estimator in sorted(CLASSIFIER_REGISTRY.items()):
            print(f"{name}: {estimator}")
        return

    cfg = load_config(args.config)

    # Generic overrides first, then the friendlier named shortcuts (which
    # take precedence if both happen to touch the same field).
    if args.overrides:
        apply_overrides(cfg, args.overrides)

    if args.dataset_name:
        cfg.dataset_name = args.dataset_name
    if args.dataset_dir:
        cfg.paths.dataset_dir = args.dataset_dir
        cfg.paths.datasets_root = None
    if args.datasets_root:
        cfg.paths.datasets_root = args.datasets_root
    if args.approaches:
        cfg.approaches.enabled = args.approaches
    if args.classifiers:
        cfg.classifiers.enabled = args.classifiers
    if args.selection_strategies:
        cfg.selection.strategies = args.selection_strategies
    if args.vocabulary_k:
        cfg.vocabulary.k = args.vocabulary_k
    if args.output_dir:
        cfg.paths.output_dir = args.output_dir
        cfg.paths.logs_dir = f"{args.output_dir}/logs"
    if args.run_id:
        cfg.runtime.run_id = args.run_id

    # --- Multi-dataset mode --------------------------------------------------
    if cfg.paths.datasets_root:
        rows = run_for_all_datasets(cfg, cfg.paths.datasets_root, approach=args.approach)
        df = build_comparison_table(rows)
        out_path = save_multi_dataset_comparison_table(df, cfg.paths.output_dir)
        print(f"\nCombined comparison table ({df['dataset'].nunique() if not df.empty else 0} datasets) saved to: {out_path}\n")
        print_comparison_table(df)
        return

    # --- Single-dataset mode --------------------------------------------------
    if not cfg.paths.dataset_dir:
        parser.error(
            "No dataset configured: set paths.dataset_dir in the config, or pass "
            "--dataset-dir, or use --datasets-root to run on multiple datasets."
        )

    ensure_run_id(cfg)
    runner = PipelineRunner(cfg)
    rows = runner.run(args.approach) if args.approach else runner.run_all()

    df = build_comparison_table(rows)
    out_path = save_comparison_table(df, cfg)
    print(f"\nComparison table saved to: {out_path}\n")
    print_comparison_table(df)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # top-level safety net: never a raw traceback for the user
        print(f"\n[FATAL] Pipeline run failed: {exc}", file=sys.stderr)
        print("Check the run's log file under outputs/logs/ for the full stack trace.", file=sys.stderr)
        sys.exit(1)
