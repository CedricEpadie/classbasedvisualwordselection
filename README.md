# BoVW / CNN Image Classification Comparison Framework

A modular, crash-resilient Python framework for comparing seven image
classification approaches on one or many datasets:

| # | Approach key          | Pipeline |
|---|------------------------|----------|
| 1 | `bovw_baseline`        | preprocessing → SIFT → K-means (direct on D) → histograms → classifiers |
| 2 | `bovw_cvws`            | preprocessing → SIFT → 11-step CVWS pipeline (§7) → classifiers |
| 3 | `cnn_bovw`             | preprocessing → GoogleNet local features → K-means (direct on D) → histograms → classifiers |
| 4 | `cnn_bovw_cvws`        | ("Ma Méthode") preprocessing → GoogleNet local features → 11-step CVWS pipeline (§7) → classifiers |
| 5 | `vit_cvws`             | preprocessing → ViT-B/16 patch-token local features → 11-step CVWS pipeline (§7) → classifiers |
| 6 | `cnn_end_to_end`       | preprocessing → GoogleNet global pooled vector → classifiers — no BoVW/vocabulary step |
| 7 | `vit_end_to_end`       | preprocessing → Vision Transformer (ViT-B/16) `[CLS]` embedding → classifiers — no BoVW/vocabulary step |

Designed for long-running (hours/days), interruptible experiments: every
step is checkpointed and idempotent, so re-launching the same command after
a crash, manual stop, or power cut — or simply relaunching on a dataset
you've already processed, e.g. with a different classifier selection —
resumes exactly where it left off instead of recomputing anything already
done (see section 5).

---

## 1. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+. GPU is optional — device is auto-detected
(`cuda` if available, else `cpu`).

---

## 2. Project structure

```
src/
├── config.py                # pydantic-validated YAML config, content hashing, CLI overrides
├── preprocessing.py         # resize + CLAHE contrast enhancement
├── feature_extraction.py    # SIFT + GoogleNet local/global + ViT (local patch-token/global CLS) extractors
├── vocabulary.py            # K-means vocabulary (+ cosine metric), histogram encoding, VocabularyStats
├── cvws_clustering.py       # CVWS pipeline steps 3-8: UMAP, HDBSCAN, symbolic re-encoding, candidate reconstruction
├── selection_strategies.py  # Strategy pattern: Methods 1, 2, 3 (GCF/ICC/CED, independent per-class Top-N ranking)
├── classifiers.py           # ClassifierWrapper around MLP/SVC/DT/LogReg/XGBoost/NB
├── metrics_engine.py        # Registry pattern, computed from stored predictions only
├── pipeline_state.py        # Checkpointing / crash recovery (StepStatus, state.json)
├── pipeline.py               # Orchestrator: wires every step together per approach/dataset
├── comparison.py            # Final Dataset × Approach × Classifier × Metrics table
└── utils/
    ├── logging_utils.py     # per-run file + console logger
    └── io_utils.py           # atomic writes (temp file + os.replace)

tests/
├── test_selection_strategies.py
├── test_vocabulary.py
├── test_cvws_clustering.py
└── test_metrics_engine.py

config.yaml            # example, fully commented config
run_comparison.py      # CLI entry point
```

---

## 3. Dataset layout

**Single dataset:** one folder per class:

```
data/<dataset_name>/
├── class_a/
│   ├── img001.jpg
│   └── img002.jpg
└── class_b/
    ├── img003.jpg
    └── img004.jpg
```

Set `paths.dataset_dir` in `config.yaml` (or pass `--dataset-dir`) accordingly.

**Multiple datasets in one run:** a folder containing several datasets, each
laid out as above:

```
data/all_datasets/
├── flowers17/
│   ├── daffodil/...
│   └── tulip/...
└── dtd/
    ├── banded/...
    └── striped/...
```

Set `paths.datasets_root` in `config.yaml` (or pass `--datasets-root`) — see
section 4.4.

---

## 4. Running the pipeline

### 4.1 Full comparison (all approaches/classifiers enabled in `config.yaml`)

```bash
python run_comparison.py --config config.yaml
```

This produces `outputs/comparison/comparison_<dataset>_<run_id>.csv` and
prints a formatted table to the console.

### 4.2 Runtime overrides — no config.yaml editing needed

Every field that's likely to change between experiments can be overridden
straight from the command line:

```bash
# Just MLP, nothing else, on a specific dataset
python run_comparison.py --config config.yaml --dataset-dir data/dtd --classifiers mlp

# Only two of the seven approaches
python run_comparison.py --config config.yaml --approaches bovw_baseline cnn_bovw_cvws

# Different final vocabulary size, different selection strategies
python run_comparison.py --config config.yaml --vocabulary-k 128 --selection-strategies GCF

# One specific approach only (shortcut for --approaches with a single value)
python run_comparison.py --config config.yaml --approach bovw_baseline

# Anything else, via a generic dotted-path override (repeatable)
python run_comparison.py --config config.yaml --set runtime.test_size=0.3 --set selection.n_candidates_per_class=10

# What's available:
python run_comparison.py --list-approaches
python run_comparison.py --list-classifiers
```

Run `python run_comparison.py --help` for the full flag list (`--dataset-name`,
`--output-dir`, `--run-id`, ...).

### 4.3 Recomputing metrics only (no retraining)

Useful after adding a new metric, or to re-audit an old run:

```bash
python -m src.metrics_engine --recompute-all --output-dir outputs
```

### 4.4 Multiple datasets in one invocation

```bash
python run_comparison.py --config config.yaml --datasets-root data/all_datasets
```

Every subdirectory of `data/all_datasets` is treated as its own dataset
(`<name>/<class>/<image>` layout) and run through the exact same
approaches/classifiers/vocabulary/selection settings from `config.yaml`
(overridable the same way as above — e.g. combine with `--classifiers mlp`).
Each dataset gets its own cache/output namespace, so datasets never
interfere with each other and can equally well be run one at a time later
without recomputing anything already done for that dataset. Results are
combined into one CSV under `outputs/comparison/comparison_all_datasets_<timestamp>.csv`
with an extra `dataset` column, alongside each dataset's own regular
per-dataset comparison outputs.

---

## 5. Caching: avoiding repeated work on a dataset you've already processed

**Just re-run the same command — or a variation of it (different
classifiers, a new approach, ...) — and only what actually needs
recomputing gets recomputed.** This works on two levels:

1. **Deterministic run id.** `runtime.run_id` defaults to
   `<dataset_name>_<config_content_hash>` (see `src/config.ensure_run_id`)
   instead of a timestamp: the exact same configuration always resolves to
   the exact same id, so its predictions/metrics/comparison files are
   reused rather than duplicated under a fresh name.
2. **Per-step, per-dataset granular cache.** `outputs/state/pipeline_state_<dataset_name>.json`
   tracks each step (preprocessing, feature extraction, K-means, UMAP/HDBSCAN
   clustering, candidate selection, per-classifier training, ...)
   independently, each keyed by a hash of *only the config section(s)
   that step actually depends on* — not the whole config. Concretely:
   * Preprocessing only reruns if `preprocessing` changed.
   * Feature extraction only reruns if `feature_extraction` changed.
   * `bovw_baseline`/`cnn_bovw`'s direct K-means vocabulary + histograms
     only rerun if `vocabulary` or `feature_extraction` changed.
   * `bovw_cvws`/`cnn_bovw_cvws`/`vit_cvws`'s UMAP+HDBSCAN clustering
     (steps 3-5) and per-class cluster stats (step 6) only rerun if
     `vocabulary` (its `umap.*`/`hdbscan.*` fields) or `feature_extraction`
     changed; per-strategy candidate selection/reconstruction (steps 7-8)
     and the resulting final K-means/histograms (steps 9-10) only rerun if
     `selection`, `vocabulary`, or `feature_extraction` changed.
   * A given classifier's training only reruns if something that actually
     feeds its `X_train`/`X_test`/hyperparameters changed — critically,
     **not** when you simply add or remove a different classifier from
     `classifiers.enabled`, or change an unrelated field like
     `metrics.enabled` or `paths.output_dir`.

   So going from `--classifiers mlp` to `--classifiers mlp svc` on the same
   dataset trains only `svc`; `mlp`'s already-computed predictions are
   reused as-is.

If a step is missing, `pending`, or `failed` for its current hash, it
(re)runs. If a step failed, its full stack trace is stored in the state
file and in `outputs/logs/run_<run_id>.log`.

Because every intermediate artifact (preprocessed image, descriptor,
histogram, vocabulary, selection, prediction) is written **atomically**
(temp file + `os.replace`), a crash mid-write can never leave a corrupted
cache file that would be silently treated as valid on restart.

Pass an explicit `--run-id` (or set `runtime.run_id` in the config) only if
you need to pin/target one specific historical run by name; leave it `null`
(the default) for the caching behavior described above.

---

## 6. Adding a new metric

Metrics are fully decoupled from training (spec section 5): training only
ever writes raw predictions (`image_id, true_label, predicted_label,
predicted_proba_<class>...`) to Parquet. To add a metric, add a function to
`src/metrics_engine.py` and register it — no other file changes:

```python
from sklearn.metrics import matthews_corrcoef
from src.metrics_engine import register_metric

@register_metric("mcc")
def matthews(y_true, y_pred, y_proba=None):
    return matthews_corrcoef(y_true, y_pred)
```

Then either re-run the pipeline (new metric is computed automatically for
every classifier) or recompute for existing runs with
`python -m src.metrics_engine --recompute-all`.

---

## 7. CVWS: Class-based Visual Word Selection (`bovw_cvws` / `cnn_bovw_cvws` / `vit_cvws`)

Unlike `bovw_baseline`/`cnn_bovw` (K-means run directly on the full
descriptor set), `bovw_cvws`/`cnn_bovw_cvws`/`vit_cvws` build their
vocabulary through an 11-step pipeline ("Pipeline de construction d'un
vocabulaire visuel par sélection de mots visuels basée sur la classe",
C. Epadie, Aug. 2026) that lets per-class discriminability shape *which
descriptors feed the final quantization*, rather than filtering an
already-built vocabulary after the fact:

1. **Preprocessing.** Every image is resized to 224×224 and contrast-enhanced.
2. **Local descriptor extraction.** Each image is represented by a set of
   local descriptors: SIFT (`bovw_cvws`), GoogleNet intermediate-layer
   activations (`cnn_bovw_cvws`), or ViT-B/16 patch tokens, i.e. every
   patch embedding excluding `[CLS]` (`vit_cvws`, `ViTExtractor`'s
   `"local"` mode). Their union over the whole corpus is the descriptor
   space `D`. Each descriptor's image of origin is kept (provenance),
   needed to map candidates back to their original vectors in step 9.
3. **Dimensionality reduction (UMAP).** Every TRAINING descriptor in `D`
   is projected to a `vocabulary.umap.n_components`-dimensional space
   (default 10). The correspondence between each reduced vector and its
   original descriptor is preserved.
4. **Unsupervised clustering (HDBSCAN, cosine).** HDBSCAN clusters the
   UMAP-reduced descriptors, with `min_cluster_size = 3`/`min_samples = 1`
   .Neither HDBSCAN nor the final K-means (step 9)
   support cosine distance natively in scikit-learn (HDBSCAN's cosine mode
   forces an O(N²) brute-force search; K-means has no cosine variant at
   all), so both are implemented via the standard, mathematically
   equivalent trick of L2-normalizing vectors and running the plain
   Euclidean algorithm (`vocabulary.l2_normalize`) — see
   `cvws_clustering.py`'s module docstring for the full rationale.
5. **Noise removal.** Descriptors HDBSCAN labels as noise (`-1`) are dropped.
6. **Symbolic re-encoding.** Each training image is re-encoded as its
   per-cluster occurrence counts (each surviving descriptor is replaced by
   its cluster id) — this reuses `vocabulary.VocabularyStats` directly, no
   nearest-centroid search needed, since HDBSCAN already gives every
   surviving descriptor a hard cluster assignment.
7. **Per-class discriminative selection.** For each class `C`,
   *independently*, its clusters are ranked by one of the three strategies
   below (`score(v, C)` for `C` alone — no comparison against any other
   class) and the `n_candidates_per_class` best-scoring ones are kept. A
   cluster id can therefore legitimately be selected by several classes at
   once — the retained sets are not necessarily disjoint, and no longer an
   argmax-based partition.
8. **Weighted candidate reconstruction.** A cluster selected by
   `selection_count` distinct classes (i.e. it appears in that many
   classes' Top-N from step 7) contributes its `selection_count`
   highest-HDBSCAN-probability member descriptors (nearest its medoid) to
   the final candidate pool — so a cluster several classes value
   contributes proportionally more representative descriptors. If
   `selection_count` exceeds a cluster's own member count, every member is
   kept (clamped).
9. **Return to the original space + final quantization.** The selected
   candidates, originally identified in UMAP space, are mapped back to
   their ORIGINAL (pre-UMAP) descriptor vectors via the provenance kept in
   step 2, then K-means (cosine, `K = vocabulary.k` — the *same* K used by
   `bovw_baseline`/`cnn_bovw`'s direct K-means, so approaches stay
   comparable at equal feature dimension) is run on them. The resulting
   centroids are the final vocabulary `V = {v1, ..., vK}`.
10. **Image encoding.** Every image (train and test) is finally encoded as
    its BoVW frequency histogram over `V`.
11. **Classification.** The histograms feed the same classifiers as every
    other approach.

`src/selection_strategies.py` implements the three step-7 scoring methods
from *"Stratégie de sélection des mots visuels"* (C. Epadie, Aug. 2026).
Selection is **independent per class**: `select_top_n(stats, class_label,
n)` ranks `score_matrix(stats)[class_label]` on its own and keeps the top
`n`, with no comparison against any other class:

| Method | Config name | Score `score(v, C)` |
|---|---|---|
| 1 — Global Class Frequency (GCF) | `GCF` | `S_GCF(v,C) = Σ_{i∈C} f(v,i)` |
| 2 — Intra-Class Coverage (ICC) | `ICC` | `S_ICC(v,C) = S_GCF(v,C) · H_intra(v,C)` |
| 3 — Class Exclusivity Discriminative (CED) | `CED` | `S_CED(v,C) = S_ICC(v,C) · (1 − H_inter(v))` |

where `f(v,i)` is the occurrence count of candidate `v` in image `i`,
`H_intra(v,C)` is the normalized Shannon entropy of `v`'s distribution
across the images of `C` (→ 1 if spread uniformly over every image of
`C`, → 0 if confined to a single image), and `H_inter(v)` is the
normalized Shannon entropy of `v`'s distribution **across classes** (→ 1
if spread evenly over every class — generic, non-discriminant — → 0 if
`v` is exclusive to one class). `H_inter`, and therefore the
discriminability factor `1 − H_inter(v)`, is a single global value per
candidate (not per class) — but since it multiplies each class's
`S_ICC(v, ·)` by a WORD-specific (not class-specific) constant, CED *can*
reorder a class's own ranking of different candidates relative to ICC
(unlike a class-invariant rescaling): a candidate with high `S_ICC` but
that's also generic across classes (`D(v)` near 0) can rank below one with
lower `S_ICC` but strong class exclusivity (`D(v)` near 1).

**`n_candidates_per_class` (`selection.n_candidates_per_class`, explicit,
required).** This is a mandatory config field with no auto-derived
default: the HDBSCAN cluster count is data-dependent, unlike the old
(removed) `top_n = vocabulary_size // n_classes` heuristic which operated
on the final vocabulary's own, known-in-advance size. If the reconstructed
candidate pool (step 8) ends up smaller than `vocabulary.k`, the final
K-means (step 9) raises a clear error asking you to raise
`n_candidates_per_class` or lower `vocabulary.k`. The per-class selected
cluster ids and per-cluster `selection_count`/`cluster_size`/`n_taken` are
persisted in `outputs/selection/*.json` for interpretability/auditing.

### Adding a new selection strategy

Subclass `VisualWordSelectionStrategy` and implement `score_matrix` (the
shared `select_top_n` ranking logic is inherited, no need to touch it). If
your strategy needs per-image data like ICC/CED do, use
`vocabulary_stats.per_class_matrix[c]` (shape `(N_c, K)`, raw per-image
candidate counts); if aggregate counts suffice, like GCF, use
`vocabulary_stats.F[c]` (shape `(K,)`):

```python
class MyStrategy(VisualWordSelectionStrategy):
    name = "my_strategy"

    def score_matrix(self, vocabulary_stats):
        # return {class_label: score_vector} with score_vector shape (K,)
        ...
```

Register it in `_STRATEGIES` (same file), then add `"my_strategy"` to
`selection.strategies` in `config.yaml`.

## 8. Output layout

```
outputs/
├── preprocessed/<dataset>/<image_id>.npy
├── features/<dataset>_sift/ | <dataset>_googlenet_<layer>/ | <dataset>_vit_<backbone>/ | <dataset>_vit_<backbone>_local/
├── vocabulary/<dataset>_<tag>_k<K>_seed<seed>.pkl                        # bovw_baseline/cnn_bovw: direct K-means
├── vocabulary/<dataset>_<tag>_umap_hdbscan.pkl                           # *_cvws steps 3-5: ClusteringResult
├── vocabulary/<dataset>_<tag>_strategy-<name>_k<K>_seed<seed>.pkl        # *_cvws step 9: final vocabulary (per strategy, cosine)
├── vocabulary_stats/<dataset>_<tag>_k<K>_seed<seed>.pkl                  # bovw_baseline/cnn_bovw: per-class F(w,c)/DF(w,c) cache
├── vocabulary_stats/<dataset>_<tag>_umap_hdbscan.pkl                     # *_cvws step 6: per-class cluster stats (F/DF/per_class_matrix)
├── histograms/<dataset>_<tag>_k<K>_seed<seed>/<image_id>.npy             # bovw_baseline/cnn_bovw
├── histograms/<dataset>_<tag>_strategy-<name>_k<K>_seed<seed>/<image_id>.npy  # *_cvws step 10: final histograms (per strategy)
├── selection/<dataset>_<tag>_strategy-<name>_ncand<N>.json               # *_cvws step 7-8: per-class selected cluster ids + per-cluster selection_count/cluster_size/n_taken
├── selection/<dataset>_<tag>_strategy-<name>_ncand<N>_candidates.npy     # *_cvws step 8: reconstructed candidate descriptors (original space)
├── predictions/<approach>_<variant>_<classifier>_<dataset>_<training_hash>.parquet
├── metrics/<approach>_<variant>_<classifier>_<dataset>_<training_hash>.json
├── comparison/comparison_<dataset>_<run_id>.csv                  # single-dataset mode
├── comparison/comparison_all_datasets_<timestamp>.csv            # --datasets-root mode
├── state/pipeline_state_<dataset>.json     # persists across run_ids for that dataset
└── logs/run_<run_id>.log
```

`<training_hash>` (see section 5) — not `run_id` — is what actually
determines whether predictions/metrics filenames match across separate
invocations; `run_id` is only used for the comparison CSV and log file
names.

---

## 9. Running the tests

```bash
pytest tests/ -v
```

`test_selection_strategies.py` verifies the entropy-based ICC/CED formulas
against hand-computed expected values (using `math.log2` directly, mirroring
the spec), and demonstrates independent-per-class ranking: the same
candidate can legitimately appear in more than one class's top-N (the
"candidate can be selected by several classes" premise of step 8), and CED
can reorder ICC's within-class ranking since its discriminability factor
varies per candidate, not per class. `test_vocabulary.py` and
`test_cvws_clustering.py` cover the final cosine K-means (`Vocabulary`'s
`metric="cosine"` mode, direction-invariant to descriptor magnitude) and
the CVWS pipeline's symbolic re-encoding / weighted candidate
reconstruction (steps 6-8), including the multi-class-overlap and
cluster-size-clamping edge cases. `test_metrics_engine.py` verifies the
metrics registry and demonstrates that adding a metric requires no
training-code changes.

---

## 10. Notes and known limitations

* `LogisticRegression.max_iter` is raised above the sklearn default — this
  is the single documented exception to "default hyperparameters
  everywhere" (see `config.yaml: classifiers.logistic_regression_max_iter`),
  needed for convergence on BoVW histogram inputs.
* `XGBClassifier` requires integer-encoded labels; `ClassifierWrapper`
  transparently label-encodes/decodes for it only, so every other module
  keeps working in string-label space.
* `Vocabulary.assign` uses brute-force nearest-center search, which is
  simple and exact but scales as O(N·K); for very large `K` (thousands+)
  or very large batches, swapping in `sklearn.cluster.KMeans.predict` or a
  KD-tree would be a drop-in optimization inside `vocabulary.py`.
* See section 7 above regarding `selection.n_candidates_per_class`: it is
  mandatory (no auto-derived default) since the HDBSCAN cluster count is
  data-dependent, unlike the final vocabulary's known-in-advance size.
* UMAP/HDBSCAN and the final cosine K-means are fit on TRAINING descriptors
  only, consistent with every other vocabulary-building step in this
  framework — test images are only ever encoded later, against the
  already-built final vocabulary (step 10).
* `umap-learn` is only imported lazily, inside `cvws_clustering.reduce_and_cluster`
  (like `torch`/`torchvision` inside `ViTExtractor`/`CnnExtractor`), so it's
  only required if a `*_cvws` approach actually runs.
* `cnn_end_to_end` (raw CNN global pooled-vector classification, no BoVW
  step) and `vit_end_to_end` (raw ViT `[CLS]`-embedding classification,
  also no BoVW step) both skip vocabulary/histogram construction entirely
  — see `pipeline.py`'s `run_cnn_end_to_end` / `run_vit_end_to_end`.
* `vit_cvws` reuses the exact same `_run_cvws_variants` orchestration as
  `bovw_cvws`/`cnn_bovw_cvws` — only the local descriptors fed in differ
  (ViT patch tokens vs. SIFT/CNN local features), via `ViTExtractor`'s
  `"local"` mode (`_extract_vit_local`), separate from the `"global"`
  mode used by `vit_end_to_end`.
