"""Feature extraction module.

Two families of descriptors are needed by the five approaches:

* SIFT local descriptors (classic BoVW, approaches 1 & 2) via
  `SiftExtractor`.
* GoogleNet/Inception-v1 features (approaches 3, 4, 5) via `CnnExtractor`,
  which can return either:
    - "local" mode: a set of local descriptors per image, built by
      reshaping the spatial activations of an intermediate conv layer to
      `(H*W, C)`, so they can feed BoVW exactly like SIFT descriptors do.
    - "global" mode: a single pooled feature vector per image, for the
      end-to-end CNN approach (3).

Both extractors expose the same minimal interface:
    extract(image: np.ndarray) -> np.ndarray   # (n_descriptors, dim) or (dim,)
so `pipeline.py` can treat them interchangeably where it only cares about
"give me descriptors for this image".
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Literal, Optional

import cv2
import numpy as np
from tqdm import tqdm

from src.config import PipelineConfig
from src.utils.io_utils import atomic_write_npy, ensure_dir, path_exists_and_valid


# --------------------------------------------------------------------------- #
# SIFT
# --------------------------------------------------------------------------- #
class SiftExtractor:
    """Wraps `cv2.SIFT_create` to produce local descriptors for BoVW."""

    def __init__(self, n_features: int = 0, contrast_threshold: float = 0.04, edge_threshold: float = 10.0):
        self._sift = cv2.SIFT_create(
            nfeatures=n_features,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
        )

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Returns an (n_keypoints, 128) float32 array. May be empty
        (shape (0, 128)) for degenerate/near-uniform images."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        _keypoints, descriptors = self._sift.detectAndCompute(gray, None)
        if descriptors is None:
            return np.zeros((0, 128), dtype=np.float32)
        return descriptors.astype(np.float32)


# --------------------------------------------------------------------------- #
# CNN (GoogleNet / Inception v1)
# --------------------------------------------------------------------------- #
class CnnExtractor:
    """Feature extractor built on `torchvision.models.googlenet`.

    Registers a forward hook on `layer_name` to capture intermediate
    spatial activations for the "local" (BoVW-feeding) mode, and uses the
    network's own average-pool output for the "global" mode.
    """

    def __init__(
        self,
        layer_name: str = "inception4e",
        global_pool_layer: str = "avgpool",
        pretrained: bool = True,
        device: str = "auto",
    ):
        import torch
        import torchvision

        self._torch = torch
        self.device = self._resolve_device(device)

        weights = torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = torchvision.models.googlenet(weights=weights, aux_logits=True)
        self.model.eval()
        self.model.to(self.device)

        self._activation: Dict[str, "torch.Tensor"] = {}
        self._register_hook(layer_name, "local")
        self._register_hook(global_pool_layer, "global")

        # ImageNet normalization, as expected by torchvision pretrained weights.
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if self._torch.cuda.is_available() else "cpu"
        return device

    def _register_hook(self, layer_name: str, key: str) -> None:
        module = dict(self.model.named_modules()).get(layer_name)
        if module is None:
            raise ValueError(
                f"Layer '{layer_name}' not found in googlenet. "
                f"Available: {list(dict(self.model.named_modules()).keys())}"
            )

        def _hook(_module, _input, output):
            self._activation[key] = output.detach()

        module.register_forward_hook(_hook)

    def _preprocess_batch(self, images: list[np.ndarray]) -> "torch.Tensor":
        torch = self._torch
        tensors = []
        for img in images:
            # BGR (OpenCV) -> RGB, resize to 224x224 as expected by GoogleNet.
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
            arr = rgb.astype(np.float32) / 255.0
            arr = (arr - self._mean) / self._std
            arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
            tensors.append(arr)
        batch = np.stack(tensors, axis=0)
        return torch.from_numpy(batch).to(self.device)

    def extract_batch(
        self, images: list[np.ndarray], mode: Literal["local", "global"]
    ) -> list[np.ndarray]:
        """Run a batch of preprocessed images through GoogleNet and return
        one array per image: (n_descriptors, C) for "local", (C,) for
        "global"."""
        torch = self._torch
        with torch.no_grad():
            batch_tensor = self._preprocess_batch(images)
            self.model(batch_tensor)

        if mode == "local":
            act = self._activation["local"]  # (B, C, H, W)
            b, c, h, w = act.shape
            act = act.permute(0, 2, 3, 1).reshape(b, h * w, c)  # (B, H*W, C)
            return [act[i].cpu().numpy().astype(np.float32) for i in range(b)]
        else:
            act = self._activation["global"]  # (B, C, 1, 1) typically
            act = act.reshape(act.shape[0], -1)
            return [act[i].cpu().numpy().astype(np.float32) for i in range(act.shape[0])]


# --------------------------------------------------------------------------- #
# === ViT (option B) : Vision Transformer, end-to-end ======================== #
# --------------------------------------------------------------------------- #
class ViTExtractor:
    """Feature extractor built on `torchvision.models.vit_b_16`.

    Supports two modes:
      - "global": one pooled embedding per image (the Transformer's own
        [CLS] token, after the encoder's final LayerNorm), fed *directly*
        to the sklearn classifiers with no BoVW step. Used by
        `vit_end_to_end` (see `pipeline.run_vit_end_to_end`).
      - "local": one descriptor PER PATCH TOKEN (excluding [CLS]), i.e.
        (N_patches, D) per image -- the ViT analogue of `CnnExtractor`'s
        "local" mode (spatial conv activations reshaped to one row per
        position). Used by `vit_cvws` (see `pipeline.run_vit_cvws`), which
        feeds these into the same 11-step CVWS pipeline
        (UMAP -> HDBSCAN -> per-class selection -> cosine K-means ->
        histograms) as `bovw_cvws`/`cnn_bovw_cvws`.

    Implementation note: torchvision's `VisionTransformer.forward` does
    roughly `x = encoder(x); x = x[:, 0]; x = heads(x)`. We hook
    `encoder.ln` (the final LayerNorm, applied to the full token sequence
    of shape (B, N_patches+1, D)) and slice it ourselves -- index 0 for
    the pooled [CLS] embedding ("global"), the remaining N_patches rows
    for the per-patch descriptors ("local") -- both taken *before* the
    ImageNet classification head, exactly the way `CnnExtractor` hooks an
    intermediate GoogleNet layer.
    """

    def __init__(self, pretrained: bool = True, device: str = "auto"):
        import torch
        import torchvision

        self._torch = torch
        self.device = self._resolve_device(device)

        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = torchvision.models.vit_b_16(weights=weights)
        self.model.eval()
        self.model.to(self.device)

        self._activation: Dict[str, "torch.Tensor"] = {}
        self._register_hook()

        # Same ImageNet normalization as CnnExtractor; ViT-B/16 also
        # expects 224x224 inputs.
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if self._torch.cuda.is_available() else "cpu"
        return device

    def _register_hook(self) -> None:
        def _hook(_module, _input, output):
            # output: (B, N_patches + 1, D) -- full post-LayerNorm token
            # sequence, [CLS] token first, then every patch token.
            self._activation["sequence"] = output.detach()

        self.model.encoder.ln.register_forward_hook(_hook)

    def _preprocess_batch(self, images: list[np.ndarray]) -> "torch.Tensor":
        # Identical preprocessing recipe to CnnExtractor (BGR->RGB, resize
        # to 224x224, ImageNet mean/std normalize, HWC->CHW).
        torch = self._torch
        tensors = []
        for img in images:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
            arr = rgb.astype(np.float32) / 255.0
            arr = (arr - self._mean) / self._std
            arr = np.transpose(arr, (2, 0, 1))
            tensors.append(arr)
        batch = np.stack(tensors, axis=0)
        return torch.from_numpy(batch).to(self.device)

    def extract_batch(self, images: list[np.ndarray], mode: Literal["global", "local"] = "global") -> list[np.ndarray]:
        """Run a batch of preprocessed images through ViT-B/16.

        mode="global": returns one pooled (D,) [CLS]-token embedding per image.
        mode="local":  returns one (N_patches, D) array of per-patch-token
                       descriptors per image (CVWS candidate pipeline input).
        """
        if mode not in ("global", "local"):
            raise ValueError("ViTExtractor only supports mode='global' or mode='local'.")
        torch = self._torch
        with torch.no_grad():
            batch_tensor = self._preprocess_batch(images)
            self.model(batch_tensor)

        seq = self._activation["sequence"]  # (B, N_patches + 1, D)
        if mode == "global":
            cls_tokens = seq[:, 0, :]  # (B, D) -- the pooled [CLS] embedding
            return [cls_tokens[i].cpu().numpy().astype(np.float32) for i in range(cls_tokens.shape[0])]
        patch_tokens = seq[:, 1:, :]  # (B, N_patches, D) -- drop [CLS]
        return [patch_tokens[i].cpu().numpy().astype(np.float32) for i in range(patch_tokens.shape[0])]
# === fin ViT (option B) ====================================================== #


# --------------------------------------------------------------------------- #
# Batch driver
# --------------------------------------------------------------------------- #
def extract_features_dataset(
    preprocessed_paths: Dict[str, str],
    cfg: PipelineConfig,
    logger: logging.Logger,
    method: Literal["sift", "cnn_local", "cnn_global", "vit_global", "vit_local"],  # === ViT (option B/cvws): "vit_global"/"vit_local" added ===
    cnn_extractor: Optional[CnnExtractor] = None,
    vit_extractor: Optional["ViTExtractor"] = None,  # === ViT (option B) ===
) -> Dict[str, str]:
    """Extract features for every preprocessed image and persist them as
    `.npy` files. Returns {image_id: output_path}. Idempotent: existing,
    valid outputs are skipped.
    """
    # === ViT (option B/cvws): which extract_batch() mode to use below.
    # "local" covers both cnn_local's per-position descriptors AND
    # vit_local's per-patch-token descriptors (both feed the CVWS
    # candidate pipeline); "global" is the single pooled vector used by
    # cnn_global and vit_global (both end-to-end approaches).
    batch_mode: Literal["local", "global"] = "local"

    if method == "sift":
        subdir = f"{cfg.dataset_name}_sift"
        extractor = SiftExtractor(
            n_features=cfg.feature_extraction.sift.n_features,
            contrast_threshold=cfg.feature_extraction.sift.contrast_threshold,
            edge_threshold=cfg.feature_extraction.sift.edge_threshold,
        )
    elif method == "cnn_local":
        layer = cfg.feature_extraction.cnn.layer_name
        subdir = f"{cfg.dataset_name}_googlenet_{layer}"
        extractor = cnn_extractor or CnnExtractor(
            layer_name=cfg.feature_extraction.cnn.layer_name,
            pretrained=cfg.feature_extraction.cnn.pretrained,
            device=cfg.feature_extraction.cnn.device,
        )
    elif method == "cnn_global":
        subdir = f"{cfg.dataset_name}_googlenet_{cfg.feature_extraction.cnn.global_pool_layer}"
        extractor = cnn_extractor or CnnExtractor(
            layer_name=cfg.feature_extraction.cnn.layer_name,
            pretrained=cfg.feature_extraction.cnn.pretrained,
            device=cfg.feature_extraction.cnn.device,
        )
        batch_mode = "global"
    elif method == "vit_global":  # === ViT (option B) ===
        subdir = f"{cfg.dataset_name}_vit_{cfg.feature_extraction.vit.backbone}"
        extractor = vit_extractor or ViTExtractor(
            pretrained=cfg.feature_extraction.vit.pretrained,
            device=cfg.feature_extraction.vit.device,
        )
        batch_mode = "global"
    else:  # === ViT (cvws): method == "vit_local" ===
        subdir = f"{cfg.dataset_name}_vit_{cfg.feature_extraction.vit.backbone}_local"
        extractor = vit_extractor or ViTExtractor(
            pretrained=cfg.feature_extraction.vit.pretrained,
            device=cfg.feature_extraction.vit.device,
        )
        batch_mode = "local"

    out_dir = ensure_dir(Path(cfg.paths.output_dir) / "features" / subdir)
    outputs: Dict[str, str] = {}
    ids = list(preprocessed_paths.keys())

    pending_ids, pending_arrays = [], []
    n_skipped = 0

    def _flush_cnn_batch():
        nonlocal pending_ids, pending_arrays
        if not pending_ids:
            return
        results = extractor.extract_batch(pending_arrays, mode=batch_mode)
        for img_id, feat in zip(pending_ids, results):
            out_path = out_dir / f"{img_id}.npy"
            atomic_write_npy(out_path, feat)
            outputs[img_id] = str(out_path)
        pending_ids, pending_arrays = [], []

    # === ViT (option B/cvws): both vit_global and vit_local use the ViT
    # config's own batch_size; sift/cnn_local/cnn_global use the CNN one
    # (sift's own batching doesn't use this at all -- see below). ===
    batch_size = (
        cfg.feature_extraction.vit.batch_size
        if method in ("vit_global", "vit_local")
        else cfg.feature_extraction.cnn.batch_size
    )
    for image_id in tqdm(ids, desc=f"feature extraction ({method})"):
        out_path = out_dir / f"{image_id}.npy"
        outputs[image_id] = str(out_path)
        if path_exists_and_valid(out_path):
            n_skipped += 1
            continue

        image = np.load(preprocessed_paths[image_id])

        if method == "sift":
            descriptors = extractor.extract(image)
            atomic_write_npy(out_path, descriptors)
        else:
            pending_ids.append(image_id)
            pending_arrays.append(image)
            if len(pending_ids) >= batch_size:
                _flush_cnn_batch()

    if method != "sift":
        _flush_cnn_batch()

    logger.info(
        "Feature extraction (%s) complete: %d computed, %d skipped (cache hit)",
        method,
        len(ids) - n_skipped,
        n_skipped,
    )
    return outputs
