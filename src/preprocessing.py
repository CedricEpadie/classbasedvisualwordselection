"""Image preprocessing: resizing + contrast enhancement (CLAHE).

Exposes a pure function `preprocess_image` (no I/O, no global state) so it
is trivially unit-testable, plus a batch driver `preprocess_dataset` that
handles the disk I/O, idempotent skipping, and atomic writes.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.config import PipelineConfig
from src.utils.io_utils import atomic_write_npy, ensure_dir, path_exists_and_valid


def preprocess_image(image: np.ndarray, config: dict) -> np.ndarray:
    """Resize an image and apply CLAHE contrast enhancement.

    Args:
        image: HxWx3 (BGR, as read by OpenCV) or HxW grayscale uint8 array.
        config: dict with keys `target_size` ([H, W]), `clahe_clip_limit`,
            `clahe_tile_grid_size` ([gx, gy]).

    Returns:
        Preprocessed HxWx3 uint8 array, contrast-enhanced on the luminance
        channel, resized to `target_size`.
    """
    target_h, target_w = config["target_size"]
    clip_limit = config["clahe_clip_limit"]
    tile_grid = tuple(config["clahe_tile_grid_size"])

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_enhanced = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_bgr


def _preprocessing_config_dict(cfg: PipelineConfig) -> dict:
    return {
        "target_size": cfg.preprocessing.target_size,
        "clahe_clip_limit": cfg.preprocessing.clahe_clip_limit,
        "clahe_tile_grid_size": cfg.preprocessing.clahe_tile_grid_size,
    }


def preprocess_dataset(
    image_paths: Dict[str, str],
    cfg: PipelineConfig,
    logger: logging.Logger,
) -> Dict[str, str]:
    """Preprocess every image in `image_paths` ({image_id: path_on_disk}),
    save results as `.npy` under `outputs/preprocessed/<dataset_name>/`, and
    return {image_id: output_npy_path}. Already-preprocessed images (given
    unchanged config) are skipped -> idempotent, resumable batch processing.
    """
    out_dir = ensure_dir(Path(cfg.paths.output_dir) / "preprocessed" / cfg.dataset_name)
    p_cfg = _preprocessing_config_dict(cfg)
    outputs: Dict[str, str] = {}

    batch_size = cfg.preprocessing.batch_size
    ids = list(image_paths.keys())
    n_done, n_skipped = 0, 0

    for start in tqdm(range(0, len(ids), batch_size), desc="preprocessing batches"):
        batch_ids = ids[start : start + batch_size]
        for image_id in batch_ids:
            out_path = out_dir / f"{image_id}.npy"
            outputs[image_id] = str(out_path)
            if path_exists_and_valid(out_path):
                n_skipped += 1
                continue
            src_path = image_paths[image_id]
            image = cv2.imread(src_path, cv2.IMREAD_COLOR)
            if image is None:
                logger.error("Could not read image %s at %s, skipping", image_id, src_path)
                continue
            processed = preprocess_image(image, p_cfg)
            atomic_write_npy(out_path, processed)
            n_done += 1

    logger.info("Preprocessing complete: %d processed, %d skipped (cache hit)", n_done, n_skipped)
    return outputs
