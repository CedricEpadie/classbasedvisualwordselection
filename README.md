# BoVW / CNN Image Classification Comparison Framework

A modular, crash-resilient Python framework for comparing seven image
classification approaches on one or many datasets:

| # | Approach key          | Pipeline |
|---|------------------------|----------|
| 1 | `bovw_baseline`        | preprocessing → SIFT → K-means (direct on D) → histograms → classifiers |
| 2 | `bovw_cvws`            | preprocessing → SIFT → 5-step CVWS candidate pipeline (§7) → classifiers |
| 3 | `cnn_bovw`             | preprocessing → GoogleNet local features → K-means (direct on D) → histograms → classifiers |
| 4 | `cnn_bovw_cvws`        | ("Ma Méthode") preprocessing → GoogleNet local features → 5-step CVWS candidate pipeline (§7) → classifiers |
| 5 | `vit_cvws`             | preprocessing → ViT-B/16 patch-token local features → 5-step CVWS candidate pipeline (§7) → classifiers |
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
├── vocabulary.py            # K-means vocabulary, Mean Shift candidate reduction, histogram encoding, VocabularyStats
├── selection_strategies.py  # Strategy pattern: Methods 1, 2, 3 (GCF/ICC/CED, argmax candidate→class assignment)
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
python run_comparison.py --config config.yaml --vocabulary-k 128 --selection-strategies global_class_frequency

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
   tracks each step (preprocessing, feature extraction, K-means, Mean
   Shift reduction, candidate selection, per-classifier training, ...)
   independently, each keyed by a hash of *only the config section(s)
   that step actually depends on* — not the whole config. Concretely:
   * Preprocessing only reruns if `preprocessing` changed.
   * Feature extraction only reruns if `feature_extraction` changed.
   * `bovw_baseline`/`cnn_bovw`'s direct K-means vocabulary + histograms
     only rerun if `vocabulary` or `feature_extraction` changed.
   * `bovw_cvws`/`cnn_bovw_cvws`'s Mean Shift reduction only reruns if
     `vocabulary` (its `mean_shift.*` fields) or `feature_extraction`
     changed; per-strategy candidate selection and the resulting final
     K-means/histograms only rerun if `selection`, `vocabulary`, or
     `feature_extraction` changed.
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
vocabulary through a 5-step pipeline that lets per-class discriminability
shape *which descriptors feed the final quantization*, rather than
filtering an already-built vocabulary after the fact:

1. **Local descriptor extraction.** Each image is represented by a set of
   local descriptors: SIFT (`bovw_cvws`), GoogleNet intermediate-layer
   activations (`cnn_bovw_cvws`), or ViT-B/16 patch tokens, i.e. every
   patch embedding excluding `[CLS]` (`vit_cvws`, `ViTExtractor`'s
   `"local"` mode). Their union over the whole corpus is the descriptor
   space `D`.
2. **Redundancy reduction (Mean Shift).** Mean Shift clusters every
   training descriptor in `D` (bandwidth auto-estimated via scikit-learn's
   `estimate_bandwidth`, no subsampling — see `vocabulary.mean_shift.*` in
   `config.yaml`). The resulting centroids are the reduced, representative
   *candidate* descriptors for step 3.
3. **Per-class discriminative selection.** For each class `C`, its
   candidates are scored with one of the three strategies below and the
   `n_candidates_per_class` best-scoring ones are kept. Selection is
   independent per class, so a candidate can be kept by more than one
   class — the retained sets are not necessarily disjoint. One full run
   of steps 3-5 happens per entry in `selection.strategies`.
4. **Final quantization (K-means).** All retained candidates, across
   every class, are pooled and clustered with K-means (`K = vocabulary.k`
   — the *same* K used by `bovw_baseline`/`cnn_bovw`'s direct K-means, so
   approaches stay comparable at equal feature dimension). The resulting
   centroids are the final vocabulary `V = {v1, ..., vK}`.
5. **Image encoding.** Every image (train and test) is finally encoded as
   its BoVW frequency histogram over `V`.

`src/selection_strategies.py` implements the three step-3 scoring methods
from *"Stratégie de sélection des mots visuels"* (C. Epadie, Aug. 2026).
Each is an **assignment**: every candidate `v` is assigned to the single
class `c*(v)` maximizing a score, `c*(v) = argmax_c score(v,c)`:

| Method | Config name | Score `score(v, C)` |
|---|---|---|
| 1 — Global Class Frequency (GCF) | `global_class_frequency` | `S_GCF(v,C) = Σ_{i∈C} f(v,i)` |
| 2 — Intra-Class Coverage (ICC) | `intra_class_corverage` | `S_ICC(v,C) = S_GCF(v,C) · H_intra(v,C)` |
| 3 — Class Exclusivity Discriminative (CED) | `class_exclusivity_discriminatve` | `S_CED(v,C) = S_ICC(v,C) · (1 − H_inter(v))` |

where `f(v,i)` is the occurrence count of candidate `v` in image `i`,
`H_intra(v,C)` is the normalized Shannon entropy of `v`'s distribution
across the images of `C` (→ 1 if spread uniformly over every image of
`C`, → 0 if confined to a single image), and `H_inter(v)` is the
normalized Shannon entropy of `v`'s distribution **across classes** (→ 1
if spread evenly over every class — generic, non-discriminant — → 0 if
`v` is exclusive to one class). Note `H_inter`, and therefore the
discriminability factor `1 − H_inter(v)`, is a single global value per
candidate, not per class: it can only rescale `S_ICC`'s per-candidate
class ranking, never flip it.

**`n_candidates_per_class` (`selection.n_candidates_per_class`, explicit,
required).** Selection alone (uncapped) is a *partition* of the candidate
set (every candidate assigned to exactly one class), so `S(c1) ∪ S(c2) ∪
... = ` the full candidate set. Capping each class's assigned set to its
`n_candidates_per_class` highest-scoring members before taking the union
is what actually reduces the pool of descriptors handed to the final
K-means (step 4) below the full candidate count — this is a mandatory
config field with no auto-derived default: the Mean-Shift candidate
count is data-dependent, unlike the old (removed) `top_n = vocabulary_size
// n_classes` heuristic which operated on the final vocabulary's own,
known-in-advance size. If the selected union ends up smaller than
`vocabulary.k`, the final K-means (step 4) raises a clear error asking you
to raise `n_candidates_per_class` or lower `vocabulary.k`. The full
(uncapped) per-class assignment is still persisted in
`outputs/selection/*.json`'s `per_class_assignment` for
interpretability/auditing, alongside the capped `selected_indices` union
actually fed to step 4.

### Adding a new selection strategy

Subclass `VisualWordSelectionStrategy` and implement `score_matrix`
(the shared `assign`/`select`/`select_top_n` argmax logic is inherited, no
need to touch it). If your strategy needs per-image data like ICC/CED do,
use `vocabulary_stats.per_class_matrix[c]` (shape `(N_c, K)`, raw
per-image candidate counts); if aggregate counts suffice, like GCF, use
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

---

## 8. Output layout

```
outputs/
├── preprocessed/<dataset>/<image_id>.npy
├── features/<dataset>_sift/ | <dataset>_googlenet_<layer>/ | <dataset>_vit_<backbone>/
├── vocabulary/<dataset>_<tag>_k<K>_seed<seed>.pkl                        # bovw_baseline/cnn_bovw: direct K-means
├── vocabulary/<dataset>_<tag>_meanshift_candidates.pkl                   # *_cvws step 2: Mean Shift candidates
├── vocabulary/<dataset>_<tag>_strategy-<name>_k<K>_seed<seed>.pkl        # *_cvws step 4: final vocabulary (per strategy)
├── vocabulary_stats/<dataset>_<tag>_k<K>_seed<seed>.pkl                  # bovw_baseline/cnn_bovw: per-class F(w,c)/DF(w,c) cache
├── vocabulary_stats/<dataset>_<tag>_meanshift_candidates.pkl             # *_cvws step 3: per-class candidate stats (F/DF/per_class_matrix)
├── histograms/<dataset>_<tag>_k<K>_seed<seed>/<image_id>.npy             # bovw_baseline/cnn_bovw
├── histograms/<dataset>_<tag>_meanshift_candidates/<image_id>.npy        # *_cvws step 3: candidate histograms (train only)
├── histograms/<dataset>_<tag>_strategy-<name>_k<K>_seed<seed>/<image_id>.npy  # *_cvws step 5: final histograms (per strategy)
├── selection/<dataset>_<tag>_strategy-<name>_ncand<N>.json   # per_class_assignment + union of selected candidates
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
the spec), checks the partition property (disjoint, union = full candidate
set) and the `n_candidates_per_class` cap's dimensionality reduction, and
confirms CED never flips ICC's per-candidate class ranking.
`test_metrics_engine.py` verifies the metrics registry and demonstrates
that adding a metric requires no training-code changes.

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
  mandatory (no auto-derived default) since the Mean-Shift candidate count
  is data-dependent, unlike the final vocabulary's known-in-advance size.
* `cnn_end_to_end` (raw CNN global pooled-vector classification, no BoVW
  step) and `vit_end_to_end` (raw ViT `[CLS]`-embedding classification,
  also no BoVW step) both skip vocabulary/histogram construction entirely
  — see `pipeline.py`'s `run_cnn_end_to_end` / `run_vit_end_to_end`.
* `vit_cvws` reuses the exact same `_run_cvws_variants` orchestration as
  `bovw_cvws`/`cnn_bovw_cvws` — only the local descriptors fed in differ
  (ViT patch tokens vs. SIFT/CNN local features), via `ViTExtractor`'s
  `"local"` mode (`_extract_vit_local`), separate from the `"global"`
  mode used by `vit_end_to_end`.
