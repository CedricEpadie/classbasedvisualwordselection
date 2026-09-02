"""Central configuration module.

Loads a single YAML file and validates it with pydantic models. Every other
module in the framework receives a fully-typed `PipelineConfig` instance
instead of reading files or environment variables itself, so the whole run
is reproducible from one artifact (the config) plus its content-hash.

Also provides the runtime-override machinery (`apply_overrides`) used by
`run_comparison.py`'s CLI flags, and `ensure_run_id`, which derives a
deterministic (non-timestamped) run id from the config's own content hash
so that re-launching the framework with an unchanged configuration reuses
every cached artifact instead of recomputing it (see `pipeline.py` /
`pipeline_state.py` for how that cache is consulted).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    """Base class for every config section: allows attribute assignment
    (needed so CLI overrides can mutate an already-loaded config) while
    still re-validating the field on each assignment, so a bad `--set`
    override fails fast with a clear pydantic error rather than silently
    corrupting the config."""

    model_config = ConfigDict(validate_assignment=True)


# --------------------------------------------------------------------------- #
# Sub-sections
# --------------------------------------------------------------------------- #
class PathsConfig(StrictModel):
    # Single-dataset mode.
    dataset_dir: str = ""
    # Multi-dataset mode: a directory containing one subfolder per dataset,
    # each itself laid out as <dataset>/<class_name>/<image>. When set, it
    # takes precedence over `dataset_dir` / the top-level `dataset_name`
    # (see `pipeline.discover_datasets_root` and `run_comparison.py`).
    datasets_root: Optional[str] = None
    output_dir: str = "outputs"
    logs_dir: str = "outputs/logs"


class PreprocessingConfig(StrictModel):
    target_size: List[int] = Field(default=[128, 128], min_length=2, max_length=2)
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: List[int] = Field(default=[8, 8], min_length=2, max_length=2)
    batch_size: int = 64


class SiftConfig(StrictModel):
    n_features: int = 0  # 0 = unlimited (OpenCV default behaviour)
    contrast_threshold: float = 0.04
    edge_threshold: float = 10.0


class CnnFeatureConfig(StrictModel):
    backbone: Literal["googlenet"] = "googlenet"
    pretrained: bool = True
    layer_name: str = "inception4e"  # intermediate conv layer -> local descriptors for BoVW
    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = 32
    global_pool_layer: str = "avgpool"  # pooled output -> global vector for cnn_end_to_end


# === ViT (option B) : config de l'extracteur ViT end-to-end =================
# Un seul champ nouveau ("vit") ; pas de nouvelle dépendance, torchvision
# fournit déjà vit_b_16 (voir src/feature_extraction.py: ViTExtractor).
class ViTFeatureConfig(StrictModel):
    backbone: Literal["vit_b_16"] = "vit_b_16"
    pretrained: bool = True
    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = 32
# === fin ViT (option B) ======================================================


class FeatureExtractionConfig(StrictModel):
    sift: SiftConfig = SiftConfig()
    cnn: CnnFeatureConfig = CnnFeatureConfig()
    vit: ViTFeatureConfig = ViTFeatureConfig()  # === ViT (option B) ===


class UMAPConfig(StrictModel):
    """Step 3 of the CVWS pipeline ('Réduction de dimensionnalité') — only
    used by the *_cvws approaches (bovw_cvws, cnn_bovw_cvws, vit_cvws);
    bovw_baseline/cnn_bovw skip this entirely and K-means directly on the
    full descriptor space D.
    """

    n_components: int = 10  # d = 10, per the spec
    n_neighbors: int = 15  # umap-learn default
    min_dist: float = 0.1  # umap-learn default


class HDBSCANConfig(StrictModel):
    """Step 4 ('Clustering non supervisé'). Hyperparameters are expressed
    as PERCENTAGES of the training descriptor count / of min_cluster_size,
    per the spec, so they scale automatically with corpus size:
        min_cluster_size = 3
        min_samples      = 1
    Both are clamped to sklearn's minimums (min_cluster_size >= 2,
    min_samples >= 1) — see `cvws_clustering.reduce_and_cluster`.
    """

    min_cluster_size: int = 3  # 3 descriptors
    min_samples: int = 1  # 1 descriptor


class VocabularyConfig(StrictModel):
    k: int = 256
    seed: int = 42
    minibatch: bool = True
    minibatch_batch_size: int = 1000
    max_iter: int = 100
    umap: UMAPConfig = UMAPConfig()
    hdbscan: HDBSCANConfig = HDBSCANConfig()


class SelectionConfig(StrictModel):
    """Configures the CVWS candidate-selection step (steps 7-8 of the
    11-step CVWS pipeline used by bovw_cvws/cnn_bovw_cvws/vit_cvws — see
    `pipeline.PipelineRunner._run_cvws_variants` and
    `cvws_clustering.py`'s module docstring):

      1. Preprocessing (unchanged)
      2. Local descriptor extraction, provenance (image <-> descriptor) kept
      3. UMAP -> d=umap.n_components
      4. HDBSCAN (cosine) on the UMAP embedding -> cluster ids
      5. Drop noise (HDBSCAN label -1)
      6. Symbolic re-encoding: each image -> per-cluster occurrence counts
      7. THIS STEP: each class independently ranks cluster ids by one of
         the three strategies (GCF/ICC/CED, "Stratégie de sélection des
         mots visuels", Epadie 2026) over their step-6 occurrence counts,
         and keeps its own `n_candidates_per_class` highest-scoring
         cluster ids -- no competition against other classes, so a
         cluster id can be kept by several classes at once.
      8. THIS STEP: a cluster selected by `selection_count` distinct
         classes contributes its `selection_count` highest-HDBSCAN-
         probability member descriptors (nearest its medoid) to the final
         candidate pool.
      9. Candidates mapped back to the original (pre-UMAP) descriptor
         space -> K-means (K = vocabulary.k, cosine) -> final vocabulary
      10. Every image re-encoded (BoVW histogram) on that final vocabulary
      11. Classification (unchanged)

    One full run of steps 7-10 happens per entry in `strategies`, each
    producing its own final vocabulary/results row (steps 1-6 are shared
    across strategies).
    """

    strategies: List[
        Literal["GCF", "ICC", "CED"]
    ] = Field(
        default_factory=lambda: [
            "GCF",
            "ICC",
            "CED",
        ]
    )
    # n in "les n meilleurs [identifiants de cluster] de chaque classe"
    # (step 7). Explicit and required, exactly as before -- except it now
    # caps the number of CLUSTER IDS kept per class (typically a much
    # smaller space than the old Mean-Shift candidate count), not raw
    # descriptor candidates directly.
    n_candidates_per_class: int = 50


class ClassifiersConfig(StrictModel):
    enabled: List[
        Literal["mlp", "svc", "decision_tree", "knn", "logistic_regression", "xgboost", "naive_bayes"]
    ] = Field(
        default_factory=lambda: [
            "mlp",
            "svc",
            "decision_tree",
            "knn",
            "logistic_regression",
            "xgboost",
            "naive_bayes",
        ]
    )
    logistic_regression_max_iter: int = 1000
    naive_bayes_variant: Literal["multinomial", "gaussian"] = "multinomial"


class MetricsConfig(StrictModel):
    enabled: List[str] = Field(
        default_factory=lambda: ["accuracy", "f1_macro", "recall_macro", "auc_macro_ovr"]
    )


class ApproachesConfig(StrictModel):
    # "cnn_end_to_end" (raw CNN global-vector classification) had been
    # removed: this framework only compared BoVW-style pipelines (SIFT or
    # CNN-local-feature-fed vocabularies, with/without word selection).
    #
    # === ViT (option B) ===================================================
    # "vit_end_to_end" reintroduces that same *kind* of approach (a single
    # pooled global feature vector -> sklearn classifiers, no BoVW /
    # vocabulary / histogram step at all) but using a Vision Transformer
    # ([CLS] token embedding) instead of a CNN. See src/pipeline.py's
    # `run_vit_end_to_end`. Not added to the default `enabled` list below,
    # so existing configs/behaviour are unaffected until you opt in
    # explicitly in config.yaml or via `--approaches`.
    # =======================================================================
    # === ViT (cvws) =========================================================
    # "vit_cvws" feeds ViT patch-token local descriptors into the same
    # 11-step CVWS pipeline as bovw_cvws/cnn_bovw_cvws (see
    # pipeline.run_vit_cvws / _run_cvws_variants). Also opt-in, same
    # rationale as vit_end_to_end above.
    # =======================================================================
    enabled: List[
        Literal["bovw_baseline", "bovw_cvws", "cnn_bovw", "cnn_bovw_cvws", "vit_cvws", "cnn_end_to_end", "vit_end_to_end"]
    ] = Field(
        default_factory=lambda: ["bovw_baseline", "bovw_cvws", "cnn_bovw", "cnn_bovw_cvws"]
    )


class RuntimeConfig(StrictModel):
    seed: int = 42
    test_size: float = 0.2
    n_jobs: int = -1
    run_id: Optional[str] = None  # deterministic default: see `ensure_run_id`


class PipelineConfig(StrictModel):
    paths: PathsConfig
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    feature_extraction: FeatureExtractionConfig = FeatureExtractionConfig()
    vocabulary: VocabularyConfig = VocabularyConfig()
    selection: SelectionConfig = SelectionConfig()
    classifiers: ClassifiersConfig = ClassifiersConfig()
    metrics: MetricsConfig = MetricsConfig()
    approaches: ApproachesConfig = ApproachesConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    dataset_name: str = "dataset"

    @field_validator("dataset_name")
    @classmethod
    def _no_spaces(cls, v: str) -> str:
        if " " in v:
            raise ValueError("dataset_name must not contain spaces (used in file paths)")
        return v

    # ------------------------------------------------------------------ #
    def content_hash(self) -> str:
        """Stable short hash of the whole config, used to invalidate caches
        and to build reproducible ``run_id`` values."""
        payload = json.dumps(self.model_dump(), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:10]

    def section_hash(self, *sections: str) -> str:
        """Hash of a subset of the config (e.g. only `vocabulary` + `feature_extraction`),
        used to invalidate a single pipeline step without invalidating everything."""
        payload = {s: self.model_dump()[s] for s in sections}
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:10]


def load_config(path: str | Path) -> PipelineConfig:
    """Load, parse and validate a YAML config file into a `PipelineConfig`.

    Does NOT assign a default `run_id` — call `ensure_run_id` once every
    CLI override (dataset, classifiers, ...) has been applied, since the
    deterministic run id depends on the *final* config content.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return PipelineConfig(**raw)


def ensure_run_id(cfg: PipelineConfig) -> PipelineConfig:
    """Assign a deterministic `run_id` if none is set: `<dataset_name>_<content_hash>`.

    Two separate launches of the framework with the exact same effective
    configuration (same dataset, same classifiers/approaches/vocabulary/...)
    always get the same run_id and therefore the same predictions/metrics
    file paths, so already-computed results are reused instead of being
    retrained — this is what makes re-running the framework on a dataset
    you've already processed cheap. Changing anything in the config
    (adding a classifier, a different K, ...) changes the hash and so
    starts a fresh run_id, avoiding any collision with the previous
    experiment's cached results.
    """
    if cfg.runtime.run_id is None:
        # Hashed while run_id is still None, so the value is stable and
        # doesn't depend on itself.
        cfg.runtime.run_id = f"{cfg.dataset_name}_{cfg.content_hash()}"
    return cfg


def apply_overrides(cfg: PipelineConfig, overrides: List[str]) -> PipelineConfig:
    """Apply `key.path=value` CLI overrides onto an already-loaded config,
    in place, returning it for chaining. `value` is parsed with
    `yaml.safe_load` so plain scalars (`42`, `true`, `mlp`) as well as
    inline lists (`[mlp,svc]`) and dicts work without extra quoting.
    Raises `ValueError` with a clear message on an unknown path or an
    invalid value for that field (thanks to `StrictModel`'s
    `validate_assignment=True`).
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --set override '{item}': expected 'key.path=value'")
        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        _set_nested(cfg, key.strip(), value)
    return cfg


def _set_nested(cfg: PipelineConfig, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    target: Any = cfg
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise ValueError(f"Unknown config section '{part}' in override '{dotted_key}'")
        target = getattr(target, part)
    last = parts[-1]
    if not hasattr(target, last):
        raise ValueError(f"Unknown config field '{last}' in override '{dotted_key}'")
    setattr(target, last, value)
