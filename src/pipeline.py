"""Main orchestrator: runs one of the seven approaches end-to-end (or all of
them), driving every step through `PipelineState` for crash-safe,
idempotent execution. Also supports running on every dataset found under a
`datasets_root` directory in one CLI invocation (see `discover_datasets_root`
/ `run_for_all_datasets`).

Approaches:
    bovw_baseline        preprocessing -> SIFT -> KMeans (direct on D) -> histograms -> classifiers
    cnn_bovw             preprocessing -> CNN local features -> KMeans (direct on D) -> histograms -> classifiers
    bovw_cvws       preprocessing -> SIFT -> 11-step CVWS pipeline -> classifiers
    cnn_bovw_cvws   preprocessing -> CNN local features -> 11-step CVWS pipeline -> classifiers
    vit_cvws        preprocessing -> ViT patch-token local features -> 11-step CVWS pipeline -> classifiers
                    === ViT (cvws), see run_vit_cvws ===
    cnn_end_to_end        preprocessing -> CNN global pooled vector -> classifiers (no BoVW step)
    vit_end_to_end        preprocessing -> ViT [CLS] embedding -> classifiers (no BoVW step)
                         === ViT (option B), see run_vit_end_to_end ===

The *_cvws approaches' 11-step CVWS ("Class-based Visual Word Selection")
pipeline (see `PipelineRunner._run_cvws_variants`, `cvws_clustering.py`'s
module docstring, and `selection_strategies.py`) differs structurally from
bovw_baseline/cnn_bovw's direct K-means: local descriptors (with
image/descriptor provenance kept) are UMAP-reduced (d=10) then HDBSCAN-
clustered (cosine); noise (label -1) is dropped; each training image is
symbolically re-encoded as per-cluster occurrence counts; for each class,
clusters are scored (GCF/ICC/CED) and the top
`selection.n_candidates_per_class` kept; a cluster selected by
`selection_count` distinct classes contributes its `selection_count`
highest-probability member descriptors (mapped back to their ORIGINAL,
pre-UMAP space) to a pooled candidate set; K-means (cosine, K=vocabulary.k)
on that pool produces the final vocabulary. Selection shapes which
descriptors feed vocabulary construction, rather than post-hoc masking a
vocabulary already built on all of D. One full run of the
selection-through-final-vocabulary steps happens per entry in
`selection.strategies`. See "Pipeline de construction d'un vocabulaire
visuel par sélection de mots visuels basée sur la classe" (Epadie, Aug 2026).
`vit_cvws` feeds this same pipeline with ViT patch-token descriptors
(`ViTExtractor`'s "local" mode) instead of SIFT/CNN local features.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.classifiers import build_classifier
from src.config import PipelineConfig, ensure_run_id
from src.cvws_clustering import (
    ClusteringResult,
    compute_cluster_stats,
    reduce_and_cluster,
    select_clusters_and_build_candidates,
    select_clusters_and_build_weighted_candidates,
)
from src.position_reconstruction import build_position_representatives, build_reconstructed_images, select_all_position_words
from src.tail_encoders import CnnTailEncoder, ViTTailEncoder
from src.feature_extraction import CnnExtractor, ViTExtractor, extract_features_dataset  # === ViT (option B): ViTExtractor added ===
from src.pipeline_state import PipelineState
from src.preprocessing import preprocess_dataset
from src.selection_strategies import build_strategy
from src.utils.io_utils import atomic_write_json, atomic_write_npy, atomic_write_pickle, ensure_dir, read_pickle
from src.utils.logging_utils import log_step, setup_logger
from src.vocabulary import Vocabulary, build_vocabulary, build_vocabulary_from_descriptors, encode_histograms

# Every approach the framework knows how to run, in a stable display order.
# Exposed here (rather than only inside ApproachesConfig) so the CLI's
# --list-approaches can introspect it without constructing a config first.
# === ViT (option B): "vit_end_to_end" added to the list of runnable
# approaches (see PipelineRunner.run_vit_end_to_end below). ===
APPROACHES: List[str] = ["bovw_baseline", "bovw_cvws", "cnn_bovw", "cnn_bovw_cvws", "vit_bovw", "vit_cvws", "cnn_end_to_end", "vit_end_to_end"]


# --------------------------------------------------------------------------- #
# Dataset discovery
# --------------------------------------------------------------------------- #
def discover_dataset(dataset_dir: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Expects `dataset_dir/<class_name>/<image_file>` layout. Returns
    ({image_id: path}, {image_id: class_label})."""
    image_paths: Dict[str, str] = {}
    labels: Dict[str, str] = {}
    root = Path(dataset_dir)
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        class_name = class_dir.name
        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            image_id = f"{class_name}__{img_path.stem}"
            image_paths[image_id] = str(img_path)
            labels[image_id] = class_name
    return image_paths, labels


def discover_datasets_root(datasets_root: str) -> Dict[str, str]:
    """Expects `datasets_root/<dataset_name>/<class_name>/<image_file>`
    layout: every subdirectory of `datasets_root` that itself contains at
    least one subdirectory (a class folder) is treated as one dataset.
    Returns {dataset_name: dataset_dir}, used to fan a single CLI
    invocation out over every dataset (see `run_for_all_datasets`)."""
    root = Path(datasets_root)
    if not root.is_dir():
        raise FileNotFoundError(f"datasets_root '{datasets_root}' is not a directory")
    datasets: Dict[str, str] = {}
    for candidate in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if any(child.is_dir() for child in candidate.iterdir()):
            datasets[candidate.name] = str(candidate)
    if not datasets:
        raise ValueError(
            f"No datasets found under '{datasets_root}': expected "
            f"'{datasets_root}/<dataset_name>/<class_name>/<image>' layout."
        )
    return datasets


def run_for_all_datasets(
    base_cfg: PipelineConfig, datasets_root: str, approach: Optional[str] = None
) -> List[dict]:
    """Run the full comparison (or a single `approach`) independently on
    every dataset discovered under `datasets_root`, reusing the exact same
    classifier/approach/vocabulary/selection settings from `base_cfg` for
    all of them. Every dataset gets its own deterministic run_id (derived
    from its own name + config, see `ensure_run_id`) so their cached
    artifacts and output files never collide, whether run in this single
    invocation or resumed later one dataset at a time. Returns the
    combined list of result rows, each tagged with a "dataset" key.
    """
    datasets = discover_datasets_root(datasets_root)
    all_rows: List[dict] = []
    for name, dataset_dir in datasets.items():
        cfg = copy.deepcopy(base_cfg)
        cfg.dataset_name = name
        cfg.paths.dataset_dir = dataset_dir
        cfg.paths.datasets_root = None  # avoid re-triggering multi-dataset mode downstream
        cfg.runtime.run_id = None  # force a fresh, dataset-specific deterministic id
        ensure_run_id(cfg)

        runner = PipelineRunner(cfg)
        rows = runner.run(approach) if approach else runner.run_all()
        for row in rows:
            row["dataset"] = name
        all_rows.extend(rows)
    return all_rows


class PipelineRunner:
    """Owns config, state, and logger for a single run; exposes one method
    per approach plus a `run` dispatcher."""

    def __init__(self, cfg: PipelineConfig):
        ensure_run_id(cfg)
        self.cfg = cfg
        self.logger = setup_logger("bovw_pipeline", cfg.paths.logs_dir, cfg.runtime.run_id)
        # Namespaced by dataset (not run_id): see pipeline_state.py's module
        # docstring for why this is what makes an unrelated config change
        # (e.g. adding a classifier) not spuriously invalidate already-computed
        # preprocessing/feature/vocabulary/histogram steps for this dataset.
        self.state = PipelineState(Path(cfg.paths.output_dir) / "state", cfg.dataset_name)
        self._cnn_extractor: Optional[CnnExtractor] = None
        self._vit_extractor: Optional[ViTExtractor] = None  # === ViT (option B) ===

    # ------------------------------------------------------------------ #
    def _step(self, step_name: str, config_hash: str, fn, *args, **kwargs):
        """Run `fn(*args, **kwargs)` under checkpointing: skip if a valid
        cached result already exists for this exact config hash, else run,
        record success/failure, and always log via `log_step`."""
        if not self.state.should_run(step_name, config_hash):
            self.logger.info("SKIP (cache hit) | %s", step_name)
            return None

        self.state.mark_running(step_name, config_hash)
        start = time.time()
        try:
            with log_step(self.logger, step_name):
                result = fn(*args, **kwargs)
        except Exception as exc:
            self.state.mark_failed(step_name, exc)
            raise
        self.state.mark_done(step_name, time.time() - start)
        return result

    def _get_cnn_extractor(self) -> CnnExtractor:
        if self._cnn_extractor is None:
            self._cnn_extractor = CnnExtractor(
                layer_name=self.cfg.feature_extraction.cnn.layer_name,
                global_pool_layer=self.cfg.feature_extraction.cnn.global_pool_layer,
                pretrained=self.cfg.feature_extraction.cnn.pretrained,
                device=self.cfg.feature_extraction.cnn.device,
            )
        return self._cnn_extractor

    # === ViT (option B): lazy-init helper, mirrors _get_cnn_extractor ===
    def _get_vit_extractor(self) -> ViTExtractor:
        if self._vit_extractor is None:
            self._vit_extractor = ViTExtractor(
                pretrained=self.cfg.feature_extraction.vit.pretrained,
                device=self.cfg.feature_extraction.vit.device,
            )
        return self._vit_extractor

    # ------------------------------------------------------------------ #
    def _preprocess(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        image_paths, labels = discover_dataset(self.cfg.paths.dataset_dir)
        h = self.cfg.section_hash("preprocessing")
        preprocessed = self._step("preprocess_dataset", h, preprocess_dataset, image_paths, self.cfg, self.logger)
        if preprocessed is None:
            out_dir = Path(self.cfg.paths.output_dir) / "preprocessed" / self.cfg.dataset_name
            preprocessed = {img_id: str(out_dir / f"{img_id}.npy") for img_id in image_paths}
        return preprocessed, labels

    def _train_test_split(self, image_ids: List[str], labels: Dict[str, str]) -> Tuple[List[str], List[str]]:
        y = [labels[i] for i in image_ids]
        train_ids, test_ids = train_test_split(
            image_ids,
            test_size=self.cfg.runtime.test_size,
            random_state=self.cfg.runtime.seed,
            stratify=y,
        )
        return train_ids, test_ids

    def _extract_sift(self, preprocessed: Dict[str, str]) -> Dict[str, str]:
        h = self.cfg.section_hash("feature_extraction")
        result = self._step("extract_sift", h, extract_features_dataset, preprocessed, self.cfg, self.logger, "sift")
        if result is None:
            out_dir = Path(self.cfg.paths.output_dir) / "features" / f"{self.cfg.dataset_name}_sift"
            result = {img_id: str(out_dir / f"{img_id}.npy") for img_id in preprocessed}
        return result

    def _extract_cnn_local(self, preprocessed: Dict[str, str]) -> Dict[str, str]:
        """CNN local descriptors (intermediate conv layer activations,
        reshaped to one row per spatial position) for BoVW-style approaches
        (cnn_bovw, cnn_bovw_cvws)."""
        layer = self.cfg.feature_extraction.cnn.layer_name
        h = self.cfg.section_hash("feature_extraction")
        result = self._step(
            "extract_cnn_local",
            h,
            extract_features_dataset,
            preprocessed,
            self.cfg,
            self.logger,
            "cnn_local",
            self._get_cnn_extractor(),
        )
        if result is None:
            out_dir = Path(self.cfg.paths.output_dir) / "features" / f"{self.cfg.dataset_name}_googlenet_{layer}"
            result = {img_id: str(out_dir / f"{img_id}.npy") for img_id in preprocessed}
        return result

    def _extract_cnn_global(self, preprocessed: Dict[str, str]) -> Dict[str, str]:
        """CNN global pooled feature vector (one vector per image) for the
        raw end-to-end CNN approach (cnn_end_to_end) — no BoVW step."""
        layer = self.cfg.feature_extraction.cnn.global_pool_layer
        h = self.cfg.section_hash("feature_extraction")
        result = self._step(
            "extract_cnn_global",
            h,
            extract_features_dataset,
            preprocessed,
            self.cfg,
            self.logger,
            "cnn_global",
            self._get_cnn_extractor(),
        )
        if result is None:
            out_dir = Path(self.cfg.paths.output_dir) / "features" / f"{self.cfg.dataset_name}_googlenet_{layer}"
            result = {img_id: str(out_dir / f"{img_id}.npy") for img_id in preprocessed}
        return result

    # === ViT (option B) ===================================================
    # Mirrors `_extract_sift` / `_extract_cnn_local`, but for the single
    # pooled ViT [CLS] embedding used by `run_vit_end_to_end`. There is no
    # equivalent "build_vocab_and_histograms" step afterwards for this
    # approach: the pooled vector goes straight to the classifiers (see
    # `_load_feature_matrix` and `run_vit_end_to_end` below).
    def _extract_vit_global(self, preprocessed: Dict[str, str]) -> Dict[str, str]:
        backbone = self.cfg.feature_extraction.vit.backbone
        h = self.cfg.section_hash("feature_extraction")
        result = self._step(
            "extract_vit_global",
            h,
            extract_features_dataset,
            preprocessed,
            self.cfg,
            self.logger,
            "vit_global",
            None,  # cnn_extractor (unused here)
            self._get_vit_extractor(),
        )
        if result is None:
            out_dir = Path(self.cfg.paths.output_dir) / "features" / f"{self.cfg.dataset_name}_vit_{backbone}"
            result = {img_id: str(out_dir / f"{img_id}.npy") for img_id in preprocessed}
        return result
    # === fin ViT (option B) ================================================

    # === ViT (cvws) =========================================================
    # Mirrors `_extract_cnn_local`, but for ViT patch-token descriptors
    # (used by `run_vit_cvws`'s 11-step CVWS pipeline instead of the
    # pooled [CLS] embedding above).
    def _extract_vit_local(self, preprocessed: Dict[str, str]) -> Dict[str, str]:
        backbone = self.cfg.feature_extraction.vit.backbone
        h = self.cfg.section_hash("feature_extraction")
        result = self._step(
            "extract_vit_local",
            h,
            extract_features_dataset,
            preprocessed,
            self.cfg,
            self.logger,
            "vit_local",
            None,  # cnn_extractor (unused here)
            self._get_vit_extractor(),
        )
        if result is None:
            out_dir = Path(self.cfg.paths.output_dir) / "features" / f"{self.cfg.dataset_name}_vit_{backbone}_local"
            result = {img_id: str(out_dir / f"{img_id}.npy") for img_id in preprocessed}
        return result
    # === fin ViT (cvws) ======================================================

    def _build_vocab_and_histograms(
        self, descriptor_paths: Dict[str, str], train_ids: List[str], tag: str
    ) -> Tuple[Vocabulary, Dict[str, str]]:
        vocab_dir = ensure_dir(Path(self.cfg.paths.output_dir) / "vocabulary")
        vocab_path = vocab_dir / f"{self.cfg.dataset_name}_{tag}_k{self.cfg.vocabulary.k}_seed{self.cfg.vocabulary.seed}.pkl"
        # Both the clustering params AND the descriptors feeding them
        # (feature_extraction section) must invalidate this cache.
        h = self.cfg.section_hash("vocabulary", "feature_extraction")
        train_descriptors = {i: descriptor_paths[i] for i in train_ids}
        vocab = self._step("build_vocabulary_" + tag, h, build_vocabulary, train_descriptors, self.cfg, self.logger, vocab_path)
        if vocab is None:
            vocab = read_pickle(vocab_path)

        hist_dir = Path(self.cfg.paths.output_dir) / "histograms" / f"{self.cfg.dataset_name}_{tag}_k{self.cfg.vocabulary.k}_seed{self.cfg.vocabulary.seed}"
        histograms = self._step(
            "encode_histograms_" + tag, h, encode_histograms, descriptor_paths, vocab, self.cfg, self.logger, hist_dir
        )
        if histograms is None:
            histograms = {img_id: str(hist_dir / f"{img_id}.npy") for img_id in descriptor_paths}
        return vocab, histograms

    # ------------------------------------------------------------------ #
    # Sections/fields that actually feed X_train/X_test/y_train/y_test and
    # the classifier itself. Used to scope the training step's cache key +
    # output filenames so that changing something irrelevant to a given
    # classifier's result (metrics.enabled, paths.output_dir, or even
    # *which other classifiers* are enabled) does not force a spurious
    # retrain. "selection" is added on top for the selection approaches
    # only (see `_run_cvws_variants`). Deliberately excludes
    # `classifiers.enabled` itself (the list of which classifiers to loop
    # over doesn't change any individual classifier's training result) and
    # `runtime.n_jobs` (parallelism only, not a result-affecting parameter).
    def _training_hash(self, *extra_sections: str, include_vocabulary: bool = True) -> str:
        import hashlib
        import json

        payload = {
            "preprocessing": self.cfg.preprocessing.model_dump(),
            "feature_extraction": self.cfg.feature_extraction.model_dump(),
            "classifiers": {k: v for k, v in self.cfg.classifiers.model_dump().items() if k != "enabled"},
            "runtime": {"seed": self.cfg.runtime.seed, "test_size": self.cfg.runtime.test_size},
        }
        # === ViT (option B): `vit_end_to_end` has no vocabulary/histogram
        # step at all, so including `vocabulary` in its hash would cause
        # spurious cache invalidation whenever someone tweaks
        # `vocabulary.k` for the *other* approaches. Every other approach
        # keeps the previous, unconditional behaviour (include_vocabulary
        # defaults to True). ===
        if include_vocabulary:
            payload["vocabulary"] = self.cfg.vocabulary.model_dump()
        for section in extra_sections:
            payload[section] = getattr(self.cfg, section).model_dump()
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:10]

    def _train_and_evaluate(
        self,
        approach: str,
        feature_variant: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        test_ids: List[str],
        nb_vw,
        training_hash: Optional[str] = None,
    ) -> List[dict]:
        """Train every enabled classifier, save raw predictions, and return
        a list of {approach, classifier, ...metric} rows via metrics_engine
        (called by comparison.py, not here — this only persists predictions).

        `training_hash` scopes both the checkpoint cache key AND the output
        filename: whenever it is unchanged (i.e. nothing that actually
        feeds this classifier's training changed), the exact same
        predictions/metrics files are targeted, so a rerun cleanly resumes
        from cache regardless of unrelated config changes or a different
        run_id.
        """
        from src.metrics_engine import compute_metrics_from_predictions

        h_base = training_hash if training_hash is not None else self._training_hash()
        pred_dir = ensure_dir(Path(self.cfg.paths.output_dir) / "predictions")
        rows = []
        classes = sorted(set(y_train) | set(y_test))

        for clf_name in self.cfg.classifiers.enabled:
            step_name = f"train_{approach}_{feature_variant}_{clf_name}"
            # Per-classifier: the classifier's own hyperparameters aren't
            # otherwise reflected in h_base beyond "which classifiers are
            # enabled", so fold the classifier name itself into the file
            # tag for readability; correctness comes from h_base above.
            file_tag = f"{approach}_{feature_variant}_{clf_name}_{self.cfg.dataset_name}_{h_base}"
            pred_path = pred_dir / f"{file_tag}.parquet"

            def _train_one(clf_name=clf_name, pred_path=pred_path, classes=classes):
                clf = build_classifier(
                    clf_name,
                    seed=self.cfg.runtime.seed,
                    logistic_regression_max_iter=self.cfg.classifiers.logistic_regression_max_iter,
                    naive_bayes_variant=self.cfg.classifiers.naive_bayes_variant,
                )
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                try:
                    y_proba = clf.predict_proba(X_test)
                    proba_classes = list(clf.classes_)
                except NotImplementedError:
                    y_proba, proba_classes = None, []

                df = pd.DataFrame(
                    {
                        "image_id": test_ids,
                        "true_label": y_test,
                        "predicted_label": y_pred,
                    }
                )
                if y_proba is not None:
                    proba_data = {
                        f"predicted_proba_{cls}": y_proba[:, i]
                        for i, cls in enumerate(proba_classes)
                    }
                    proba_df = pd.DataFrame(proba_data, index=df.index)
                    df = pd.concat([df, proba_df], axis=1)
                ensure_dir(pred_path.parent)
                # Parquet write is not "atomic" at the OS level via os.replace in
                # pandas directly, so we write to a temp file then replace.
                tmp_path = pred_path.with_suffix(".tmp.parquet")
                df.to_parquet(tmp_path, index=False)
                import os

                os.replace(tmp_path, pred_path)
                return str(pred_path)

            result_path = self._step(step_name, h_base, _train_one)
            if result_path is None:
                result_path = str(pred_path)

            metrics = compute_metrics_from_predictions(result_path, self.cfg.metrics.enabled)
            metrics_path = Path(self.cfg.paths.output_dir) / "metrics" / f"{file_tag}.json"
            atomic_write_json(metrics_path, metrics)

            row = {
                "dataset": self.cfg.dataset_name,
                "nb_vw": nb_vw,
                "approach": approach,
                "feature_variant": feature_variant,
                "classifier": clf_name,
            }
            row.update(metrics)
            rows.append(row)

        return rows

    # ------------------------------------------------------------------ #
    # Approaches
    # ------------------------------------------------------------------ #
    def run_bovw_baseline(self) -> List[dict]:
        preprocessed, labels = self._preprocess()
        ids = list(preprocessed.keys())
        train_ids, test_ids = self._train_test_split(ids, labels)
        sift_paths = self._extract_sift(preprocessed)
        vocab, histograms = self._build_vocab_and_histograms(sift_paths, train_ids, tag="sift")
        X_train, y_train = self._load_histogram_matrix(histograms, train_ids, labels)
        X_test, y_test = self._load_histogram_matrix(histograms, test_ids, labels)
        return self._train_and_evaluate(
            "bovw_baseline", "full", X_train, y_train, X_test, y_test, test_ids, nb_vw=vocab.k
        )

    def run_bovw_cvws(self) -> List[dict]:
        """CVWS ('Ma Méthode' for SIFT): preprocessing -> SIFT -> 11-step
        CVWS pipeline (UMAP + HDBSCAN + per-class selection + final
        cosine K-means) -> classifiers. See `_run_cvws_variants`."""
        preprocessed, labels = self._preprocess()
        ids = list(preprocessed.keys())
        train_ids, test_ids = self._train_test_split(ids, labels)
        sift_paths = self._extract_sift(preprocessed)
        return self._run_cvws_variants("bovw_cvws", sift_paths, train_ids, test_ids, labels, tag="sift")

    def run_cnn_end_to_end(self) -> List[dict]:
        """Raw CNN classification: no BoVW step, no visual vocabulary at all."""
        preprocessed, labels = self._preprocess()
        ids = list(preprocessed.keys())
        train_ids, test_ids = self._train_test_split(ids, labels)
        global_paths = self._extract_cnn_global(preprocessed)
        X_train = np.stack([np.load(global_paths[i]) for i in train_ids])
        y_train = np.array([labels[i] for i in train_ids])
        X_test = np.stack([np.load(global_paths[i]) for i in test_ids])
        y_test = np.array([labels[i] for i in test_ids])
        return self._train_and_evaluate(
            "cnn_end_to_end", "global", X_train, y_train, X_test, y_test, test_ids, nb_vw="RAS"
        )

    def run_cnn_bovw(self) -> List[dict]:
        preprocessed, labels = self._preprocess()
        ids = list(preprocessed.keys())
        train_ids, test_ids = self._train_test_split(ids, labels)
        local_paths = self._extract_cnn_local(preprocessed)
        vocab, histograms = self._build_vocab_and_histograms(local_paths, train_ids, tag="cnn")
        X_train, y_train = self._load_histogram_matrix(histograms, train_ids, labels)
        X_test, y_test = self._load_histogram_matrix(histograms, test_ids, labels)
        return self._train_and_evaluate(
            "cnn_bovw", "full", X_train, y_train, X_test, y_test, test_ids, nb_vw=vocab.k
        )

    def run_cnn_bovw_cvws(self) -> List[dict]:
        """CVWS for CNN (2026-09 methodology update): preprocessing -> CNN
        local features -> UMAP/HDBSCAN + per-position, per-class word
        selection -> synthetic per-class image reconstruction -> CNN-tail
        encoding -> classifiers. See `_run_position_cvws_variant` and
        `position_reconstruction.py`'s module docstring (no more K-means/
        histogram step for this approach -- superseded by that update)."""
        preprocessed, labels = self._preprocess()
        ids = list(preprocessed.keys())
        train_ids, test_ids = self._train_test_split(ids, labels)
        local_paths = self._extract_cnn_local(preprocessed)
        return self._run_position_cvws_variant(
            "cnn_bovw_cvws", "cnn", local_paths, train_ids, test_ids, labels, tag="cnn"
        )

    # === ViT (cvws) =========================================================
    
    def run_vit_cvws(self) -> List[dict]:
        """CVWS for ViT (2026-09 methodology update): preprocessing -> ViT
        patch-token local features -> UMAP/HDBSCAN + per-position,
        per-class word selection -> synthetic per-class "image"
        (patch-token sequence) reconstruction -> ViT-encoder tail encoding
        ([CLS] + reconstructed patches + positional embedding) ->
        classifiers. See `_run_position_cvws_variant` and
        `position_reconstruction.py`'s module docstring (no more K-means/
        histogram step for this approach -- superseded by that update)."""
        preprocessed, labels = self._preprocess()
        ids = list(preprocessed.keys())
        train_ids, test_ids = self._train_test_split(ids, labels)
        local_paths = self._extract_vit_local(preprocessed)
        return self._run_position_cvws_variant(
            "vit_cvws", "vit", local_paths, train_ids, test_ids, labels, tag="vit"
        )
    # === fin ViT (cvws) ======================================================

    def run_vit_end_to_end(self) -> List[dict]:
        """ViT end-to-end: preprocessing -> ViT [CLS] embedding -> classifiers.

        Deliberately skips SIFT/CNN-local-feature extraction, K-means
        vocabulary building, histogram encoding, and word selection
        entirely -- unlike the four BoVW-style approaches, the pooled ViT
        embedding is used directly as the classifier's feature vector.
        This mirrors the old (removed) "cnn_end_to_end" approach mentioned
        in the README, but with a Vision Transformer backbone.
        """
        preprocessed, labels = self._preprocess()
        ids = list(preprocessed.keys())
        train_ids, test_ids = self._train_test_split(ids, labels)
        vit_paths = self._extract_vit_global(preprocessed)
        X_train, y_train = self._load_feature_matrix(vit_paths, train_ids, labels)
        X_test, y_test = self._load_feature_matrix(vit_paths, test_ids, labels)
        return self._train_and_evaluate(
            "vit_end_to_end",
            "global",
            X_train,
            y_train,
            X_test,
            y_test,
            test_ids,
            nb_vw="RAS",
            # No vocabulary section involved for this approach -- see the
            # `include_vocabulary` note on `_training_hash` above.
            training_hash=self._training_hash(include_vocabulary=False),
        )
        
    def run_vit_bovw(self) -> List[dict]:
        preprocessed, labels = self._preprocess()
        ids = list(preprocessed.keys())
        train_ids, test_ids = self._train_test_split(ids, labels)
        local_paths = self._extract_vit_local(preprocessed)
        vocab, histograms = self._build_vocab_and_histograms(local_paths, train_ids, tag="vit")
        X_train, y_train = self._load_histogram_matrix(histograms, train_ids, labels)
        X_test, y_test = self._load_histogram_matrix(histograms, test_ids, labels)
        return self._train_and_evaluate(
            "vit_bovw", "full", X_train, y_train, X_test, y_test, test_ids, nb_vw=vocab.k
        )
    

    def _load_feature_matrix(
        self, feature_paths: Dict[str, str], ids: List[str], labels: Dict[str, str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Like `_load_histogram_matrix`, but for a raw pooled feature
        vector (ViT [CLS] embedding): no sum-normalization, since this
        isn't a count histogram -- just stack the vectors as-is."""
        X = np.stack([np.load(feature_paths[i]) for i in ids])
        y = np.array([labels[i] for i in ids])
        return X, y
    # === fin ViT (option B) ================================================

    # ------------------------------------------------------------------ #
    # CVWS pipeline (bovw_cvws / cnn_bovw_cvws / vit_cvws only) — steps 3-10
    # (steps 1-2, extraction+provenance, already done by the caller):
    #   3-5. UMAP -> HDBSCAN (cosine) -> drop noise           -> ClusteringResult
    #   6.   symbolic re-encoding + per-class cluster stats    -> VocabularyStats
    #   7-8. per-class Top-N cluster selection + probability-
    #        weighted candidate reconstruction (original space) -> candidates
    #   9.   final K-means (cosine, K = vocabulary.k)          -> final vocabulary V
    #   10.  encode every image (train+test) on V              -> histograms -> classifiers
    # Steps 3-6 run once per approach/tag (shared across strategies); steps
    # 7-10 (and therefore the final vocabulary itself) run once PER
    # strategy, since different strategies select different clusters.
    # ------------------------------------------------------------------ #
    def _umap_hdbscan_and_stats(
        self,
        descriptor_paths: Dict[str, str],
        train_ids: List[str],
        labels: Dict[str, str],
        tag: str,
    ) -> Tuple[ClusteringResult, "VocabularyStats"]:
        """Steps 3-6, shared verbatim by `_run_cvws_variants` (bovw_cvws)
        and `_run_position_cvws_variant` (cnn_bovw_cvws/vit_cvws): UMAP ->
        HDBSCAN (cosine) -> drop noise -> symbolic re-encoding + per-class
        cluster stats. Cached under the SAME path for a given `tag`
        regardless of which of the two callers runs first, since the
        clustering itself doesn't depend on how its output is later
        turned into a final vocabulary (bovw_cvws) or per-position
        synthetic images (cnn_bovw_cvws/vit_cvws)."""
        h_clustering = self.cfg.section_hash("vocabulary", "feature_extraction")

        clustering_dir = ensure_dir(Path(self.cfg.paths.output_dir) / "vocabulary")
        clustering_path = clustering_dir / f"{self.cfg.dataset_name}_{tag}_umap_hdbscan.pkl"
        train_descriptors = {i: descriptor_paths[i] for i in train_ids}
        clustering_result = self._step(
            f"reduce_and_cluster_{tag}", h_clustering, reduce_and_cluster, train_descriptors, self.cfg, self.logger, clustering_path
        )
        if clustering_result is None:
            clustering_result = read_pickle(clustering_path)

        stats_dir = ensure_dir(Path(self.cfg.paths.output_dir) / "vocabulary_stats")
        stats_path = stats_dir / f"{self.cfg.dataset_name}_{tag}_umap_hdbscan.pkl"

        def _compute_and_persist_stats():
            stats = compute_cluster_stats(clustering_result, labels, self.logger)
            atomic_write_pickle(stats_path, stats)
            return stats

        stats = self._step(f"compute_cluster_stats_{tag}", h_clustering, _compute_and_persist_stats)
        if stats is None:
            stats = read_pickle(stats_path)
        return clustering_result, stats

    # ------------------------------------------------------------------ #
    # Position-based reconstruction pipeline (cnn_bovw_cvws / vit_cvws only,
    # 2026-09 methodology update) — steps 3-6 shared with bovw_cvws (see
    # `_umap_hdbscan_and_stats`), then:
    #   7-8. per-position, per-class word ranking + one representative
    #        vector per selected (position, cluster)     -> position_reconstruction.py
    #   9.   y synthetic images per class, encoded via the CNN/ViT tail
    #        (same backbone, downstream of the local-descriptor hook)
    #        -> synthetic TRAINING vectors
    #   10.  REAL test images encoded via the standard global pooled
    #        feature (same tail, applied to a real forward pass instead of
    #        a reconstructed one)                          -> TEST vectors
    #   11.  classification (unchanged)
    # No K-means/histogram step at all for this pipeline -- see
    # `position_reconstruction.py`'s module docstring for why.
    # ------------------------------------------------------------------ #
    def _run_position_cvws_variant(
        self,
        approach: str,
        backend: str,  # "cnn" or "vit"
        descriptor_paths: Dict[str, str],
        train_ids: List[str],
        test_ids: List[str],
        labels: Dict[str, str],
        tag: str,
    ) -> List[dict]:
        clustering_result, stats = self._umap_hdbscan_and_stats(descriptor_paths, train_ids, labels, tag)

        # n_positions/descriptor_dim: every image contributes the SAME
        # fixed-size, fixed-order local descriptor set (CNN spatial conv
        # activations / ViT patch tokens), unlike SIFT's variable-length
        # keypoint sets -- see `position_reconstruction.py`'s module
        # docstring. Read from one real descriptor file rather than
        # hardcoding 196, since it depends on input resolution/backbone.
        sample = np.load(next(iter(descriptor_paths.values())))
        n_positions, descriptor_dim = sample.shape
        classes = sorted({labels[i] for i in train_ids})
        images_per_class = self.cfg.selection.images_per_class
        if images_per_class is None:
            images_per_class = max(1, len(train_ids) // len(classes))

        # --- Real TEST vectors: standard pooled global feature (same tail
        # the synthetic vectors below go through, applied to a real
        # forward pass instead of a reconstructed one) ---
        if backend == "cnn":
            global_paths = self._extract_cnn_global(self._preprocessed_paths_for(test_ids))
            tail_encoder = CnnTailEncoder(
                self._get_cnn_extractor(),
                self.cfg.feature_extraction.cnn.layer_name,
                self.cfg.feature_extraction.cnn.global_pool_layer,
            )
        else:
            global_paths = self._extract_vit_global(self._preprocessed_paths_for(test_ids))
            tail_encoder = ViTTailEncoder(self._get_vit_extractor())
        X_test = np.stack([np.load(global_paths[i]) for i in test_ids])
        y_test = np.array([labels[i] for i in test_ids])

        all_rows: List[dict] = []
        for strategy_name in self.cfg.selection.strategies:
            strategy = build_strategy(strategy_name)
            top_n = self.cfg.selection.position_top_n
            h_selection = self.cfg.section_hash("selection", "vocabulary", "feature_extraction")

            selection_dir = ensure_dir(Path(self.cfg.paths.output_dir) / "selection")
            audit_path = (
                selection_dir / f"{self.cfg.dataset_name}_{tag}_strategy-{strategy_name}_positional_topn{top_n}.json"
            )
            synthetic_dir = ensure_dir(Path(self.cfg.paths.output_dir) / "reconstructed_vectors")
            synthetic_path = (
                synthetic_dir
                / f"{self.cfg.dataset_name}_{tag}_strategy-{strategy_name}_topn{top_n}_y{images_per_class}.npz"
            )

            def _compute_and_persist_synthetic(
                strategy=strategy, top_n=top_n, audit_path=audit_path, synthetic_path=synthetic_path
            ):
                position_words = select_all_position_words(
                    strategy, clustering_result, labels, n_positions, top_n, self.logger
                )
                representatives = build_position_representatives(clustering_result, descriptor_paths, position_words)
                reconstructed = build_reconstructed_images(
                    position_words, representatives, classes, n_positions, descriptor_dim, images_per_class, self.logger
                )
                X_synth_parts, y_synth_parts = [], []
                for c in classes:
                    vectors = tail_encoder.encode(reconstructed[c])
                    X_synth_parts.append(vectors)
                    y_synth_parts.extend([c] * vectors.shape[0])
                X_synth = np.concatenate(X_synth_parts, axis=0)
                y_synth = np.array(y_synth_parts)

                audit_info = {
                    "strategy": strategy.name,
                    "position_top_n": top_n,
                    "images_per_class": images_per_class,
                    "n_positions": n_positions,
                    "n_words_selected_per_position_class": {
                        str(p): {c: len(ids) for c, ids in by_class.items()} for p, by_class in position_words.items()
                    },
                }
                atomic_write_json(audit_path, audit_info)
                ensure_dir(synthetic_path.parent)
                tmp_path = synthetic_path.with_suffix(".tmp.npz")
                np.savez(tmp_path, X=X_synth, y=y_synth)
                import os

                os.replace(tmp_path, synthetic_path)
                return X_synth, y_synth

            synthetic = self._step(
                f"reconstruct_and_encode_{approach}_{strategy_name}", h_selection, _compute_and_persist_synthetic
            )
            if synthetic is None:
                with np.load(synthetic_path) as npz:
                    synthetic = (npz["X"], npz["y"])
            X_train, y_train = synthetic

            rows = self._train_and_evaluate(
                approach,
                strategy_name,
                X_train,
                y_train,
                X_test,
                y_test,
                test_ids,
                nb_vw="RAS",
                training_hash=self._training_hash("selection"),
            )
            all_rows.extend(rows)
        return all_rows

    def _preprocessed_paths_for(self, ids: List[str]) -> Dict[str, str]:
        """Restrict an already-preprocessed id->path mapping to `ids`
        (used to extract global features for the TEST set only, in
        `_run_position_cvws_variant`, without re-extracting for the
        training images the synthetic-image path doesn't need it for)."""
        preprocessed_dir = Path(self.cfg.paths.output_dir) / "preprocessed" / self.cfg.dataset_name
        return {i: str(preprocessed_dir / f"{i}.npy") for i in ids}

    def _run_cvws_variants(
        self,
        approach: str,
        descriptor_paths: Dict[str, str],
        train_ids: List[str],
        test_ids: List[str],
        labels: Dict[str, str],
        tag: str,
    ) -> List[dict]:
        # Steps 3-6 (UMAP -> HDBSCAN -> drop noise -> symbolic re-encoding +
        # per-class cluster stats), shared verbatim with
        # `_run_position_cvws_variant` -- see `_umap_hdbscan_and_stats`.
        clustering_result, stats = self._umap_hdbscan_and_stats(descriptor_paths, train_ids, labels, tag)

        all_rows: List[dict] = []
        for strategy_name in self.cfg.selection.strategies:
            strategy = build_strategy(strategy_name)
            n = self.cfg.selection.n_candidates_per_class
            # Selection depends on the selection params AND everything the
            # cluster set itself depends on.
            h_selection = self.cfg.section_hash("selection", "vocabulary", "feature_extraction")

            # --- Steps 7-8: per-class cluster selection + weighted candidate reconstruction ---
            selection_dir = ensure_dir(Path(self.cfg.paths.output_dir) / "selection")
            selection_path = selection_dir / f"{self.cfg.dataset_name}_{tag}_strategy-{strategy_name}_ncand{n}.json"
            candidates_path = (
                selection_dir / f"{self.cfg.dataset_name}_{tag}_strategy-{strategy_name}_ncand{n}_candidates.npy"
            )

            def _compute_and_persist_candidates(
                strategy=strategy, n=n, selection_path=selection_path, candidates_path=candidates_path
            ):
                # audit_info (per-class selected cluster ids + per-cluster
                # selection counts/allocations) is persisted for
                # interpretability, next to the actual candidate descriptor
                # matrix fed to step 9. `selection.reconstruction_mode`
                # (2026-09 update, bovw_cvws only -- see
                # `cvws_clustering.py`'s module docstring) picks between the
                # original selection_count-based step 8 ("legacy") and the
                # score-weighted N/|C| allocation ("weighted", default).
                if self.cfg.selection.reconstruction_mode == "weighted":
                    pool_size = self.cfg.selection.candidate_pool_size
                    if pool_size is None:
                        pool_size = len(clustering_result.image_ids)  # N = surviving training descriptors
                    candidate_descriptors, audit_info = select_clusters_and_build_weighted_candidates(
                        clustering_result, stats, strategy, n, pool_size, descriptor_paths, self.logger
                    )
                else:
                    candidate_descriptors, audit_info = select_clusters_and_build_candidates(
                        clustering_result, stats, strategy, n, descriptor_paths, self.logger
                    )
                atomic_write_json(selection_path, audit_info)
                atomic_write_npy(candidates_path, candidate_descriptors)
                return candidate_descriptors

            candidate_descriptors = self._step(
                f"select_and_build_candidates_{approach}_{strategy_name}", h_selection, _compute_and_persist_candidates
            )
            if candidate_descriptors is None:
                candidate_descriptors = np.load(candidates_path)

            # --- Step 9: final K-means (cosine, K = vocabulary.k) on the candidates ---
            final_vocab_dir = ensure_dir(Path(self.cfg.paths.output_dir) / "vocabulary")
            final_vocab_path = (
                final_vocab_dir
                / f"{self.cfg.dataset_name}_{tag}_strategy-{strategy_name}_k{self.cfg.vocabulary.k}_seed{self.cfg.vocabulary.seed}.pkl"
            )
            final_vocab = self._step(
                f"build_final_vocab_{approach}_{strategy_name}",
                h_selection,
                build_vocabulary_from_descriptors,
                candidate_descriptors,
                self.cfg,
                self.logger,
                final_vocab_path,
                None,  # k=None -> defaults to cfg.vocabulary.k
                "cosine",  # metric -- see build_vocabulary_from_descriptors's docstring
            )
            if final_vocab is None:
                final_vocab = read_pickle(final_vocab_path)

            # --- Step 10: encode EVERY image (train+test) on the final vocabulary ---
            final_hist_dir = (
                Path(self.cfg.paths.output_dir)
                / "histograms"
                / f"{self.cfg.dataset_name}_{tag}_strategy-{strategy_name}_k{self.cfg.vocabulary.k}_seed{self.cfg.vocabulary.seed}"
            )
            histograms = self._step(
                f"encode_final_histograms_{approach}_{strategy_name}",
                h_selection,
                encode_histograms,
                descriptor_paths,
                final_vocab,
                self.cfg,
                self.logger,
                final_hist_dir,
            )
            if histograms is None:
                histograms = {img_id: str(final_hist_dir / f"{img_id}.npy") for img_id in descriptor_paths}

            X_train, y_train = self._load_histogram_matrix(histograms, train_ids, labels)
            X_test, y_test = self._load_histogram_matrix(histograms, test_ids, labels)
            rows = self._train_and_evaluate(
                approach,
                strategy_name,
                X_train,
                y_train,
                X_test,
                y_test,
                test_ids,
                nb_vw=final_vocab.k,
                training_hash=self._training_hash("selection"),
            )
            all_rows.extend(rows)
        return all_rows

    @staticmethod
    def _normalize(hist: np.ndarray) -> np.ndarray:
        total = hist.sum()
        return hist / total if total > 0 else hist

    def _load_histogram_matrix(
        self, histograms: Dict[str, str], ids: List[str], labels: Dict[str, str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = np.stack([self._normalize(np.load(histograms[i])) for i in ids])
        y = np.array([labels[i] for i in ids])
        return X, y

    # ------------------------------------------------------------------ #
    def run(self, approach: str) -> List[dict]:
        dispatch = {
            "bovw_baseline": self.run_bovw_baseline,
            "bovw_cvws": self.run_bovw_cvws,
            "cnn_bovw": self.run_cnn_bovw,
            "cnn_bovw_cvws": self.run_cnn_bovw_cvws,
            "vit_bovw": self.run_vit_bovw,
            "vit_cvws": self.run_vit_cvws,
            "cnn_end_to_end": self.run_cnn_end_to_end,
            "vit_end_to_end": self.run_vit_end_to_end,
        }
        if approach not in dispatch:
            raise ValueError(f"Unknown approach: {approach}")
        self.logger.info("=== Running approach: %s ===", approach)
        return dispatch[approach]()

    def run_all(self) -> List[dict]:
        rows: List[dict] = []
        for approach in self.cfg.approaches.enabled:
            rows.extend(self.run(approach))
        return rows
