"""Encodes the reconstructed per-position feature tensors from
`position_reconstruction.py` into vector representations, by running them
through the TAIL of the same pretrained CNN/ViT backbone already used for
local-descriptor extraction (`feature_extraction.py`) -- i.e. everything
downstream of the layer local descriptors were hooked from, ending at the
same pooled representation `cnn_global`/`vit_global` extract into. This is
what makes synthetic (reconstructed) training vectors and real test-image
vectors comparable: both pass through the exact same downstream computation,
only the input to it differs (a real spatial activation / patch-token
sequence vs. a reconstructed one) -- see
`pipeline.PipelineRunner._run_position_cvws_variant`.

Lazy `torch`/`torchvision` dependency, same pattern as
`feature_extraction.CnnExtractor`/`ViTExtractor` (whose already-loaded
`.model`/`.device`/`._torch` are reused here directly -- no separate model
load, no separate weights).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.feature_extraction import CnnExtractor, ViTExtractor


class CnnTailEncoder:
    """Runs a (B, n_positions, C) reconstructed local-descriptor tensor
    through the part of GoogLeNet that comes AFTER the local-descriptor hook
    layer (`cfg.feature_extraction.cnn.layer_name`, "inception4e" by
    default) up to and including `global_pool_layer` ("avgpool" by
    default) -- exactly the computation real images already go through
    between `CnnExtractor.extract_batch(mode="local")` and
    `extract_batch(mode="global")` (torchvision `GoogLeNet._forward`'s own
    `inception4e -> maxpool4 -> inception5a -> inception5b -> avgpool`
    sequence), so synthetic and real vectors land in the same space.

    Only supports the DEFAULT layer_name="inception4e" / global_pool_
    layer="avgpool" pairing -- raises clearly at construction time
    otherwise rather than silently building the wrong tail for a custom
    hook point (extend `_TAIL_MODULE_NAMES` with the matching submodule
    sequence before using a different one).
    """

    _TAIL_MODULE_NAMES = ("maxpool4", "inception5a", "inception5b", "avgpool")

    def __init__(self, cnn_extractor: "CnnExtractor", layer_name: str, global_pool_layer: str):
        if layer_name != "inception4e" or global_pool_layer != "avgpool":
            raise NotImplementedError(
                "CnnTailEncoder only knows the default GoogLeNet tail "
                "(layer_name='inception4e' -> global_pool_layer='avgpool': "
                f"{' -> '.join(self._TAIL_MODULE_NAMES)}). Got layer_name={layer_name!r}, "
                f"global_pool_layer={global_pool_layer!r} -- extend _TAIL_MODULE_NAMES with the "
                "matching submodule sequence for this layer pair before using a non-default one."
            )
        self._extractor = cnn_extractor
        self._torch = cnn_extractor._torch
        self._model = cnn_extractor.model
        self._tail_modules = [dict(self._model.named_modules())[name] for name in self._TAIL_MODULE_NAMES]

    def encode(self, reconstructed: np.ndarray) -> np.ndarray:
        """`reconstructed`: (B, n_positions, C), n_positions == H*W of the
        hooked layer's own spatial output (the caller is responsible for
        that match -- `_run_position_cvws_variant` derives n_positions
        from the real local-descriptor extraction's own output shape).
        Assumes a square H==W spatial layout (true of every square input
        resolution this framework preprocesses to). Returns (B, C_out)
        pooled vectors, the same space as `CnnExtractor.extract_batch(mode
        ="global")`."""
        torch = self._torch
        b, n_positions, c = reconstructed.shape
        h = w = int(round(n_positions**0.5))
        if h * w != n_positions:
            raise ValueError(
                f"CnnTailEncoder expects a square spatial layout (H*W == n_positions); "
                f"got n_positions={n_positions}, not a perfect square."
            )
        with torch.no_grad():
            x = torch.from_numpy(reconstructed.astype(np.float32)).to(self._extractor.device)
            x = x.permute(0, 2, 1).reshape(b, c, h, w)  # (B, n_positions, C) -> (B, C, H, W)
            for module in self._tail_modules:
                x = module(x)
            x = torch.flatten(x, 1)
        return x.cpu().numpy().astype(np.float32)


class ViTTailEncoder:
    """Runs a (B, n_patches, D) reconstructed patch-token tensor through
    the SAME "[CLS] + patches + positional embedding -> transformer
    encoder" path real images go through, per the spec: "les patchs des
    images sont remplacés par nos nouveaux vecteurs [...] (token_cls, nos
    patchs reconstruits, positional embedding)". `model.encoder` already
    applies the positional embedding, dropout, transformer layers and
    final LayerNorm internally (see torchvision's
    `VisionTransformer.forward`/`Encoder.forward`) -- reused UNCHANGED
    here, exactly as it runs for real images.
    """

    def __init__(self, vit_extractor: "ViTExtractor"):
        self._extractor = vit_extractor
        self._torch = vit_extractor._torch
        self._model = vit_extractor.model

    def encode(self, reconstructed: np.ndarray) -> np.ndarray:
        """`reconstructed`: (B, n_patches, D) -- n_patches must match the
        model's own patch count (e.g. 196 for ViT-B/16 at 224x224; the
        caller is responsible for that match, same as `CnnTailEncoder`).
        Returns (B, D) pooled [CLS] embeddings, the same space as
        `ViTExtractor.extract_batch(mode="global")`."""
        torch = self._torch
        b, n_patches, _d = reconstructed.shape
        expected_tokens = self._model.encoder.pos_embedding.shape[1]
        if n_patches + 1 != expected_tokens:
            raise ValueError(
                f"ViTTailEncoder: reconstructed tensor has {n_patches} patches (+1 for [CLS] = "
                f"{n_patches + 1} tokens), but the model's positional embedding expects "
                f"{expected_tokens} tokens. n_patches must match the model's own patch grid."
            )
        with torch.no_grad():
            tokens = torch.from_numpy(reconstructed.astype(np.float32)).to(self._extractor.device)
            class_token = self._model.class_token.expand(b, -1, -1)
            x = torch.cat([class_token, tokens], dim=1)
            x = self._model.encoder(x)  # adds pos_embedding + dropout + transformer layers + final ln
            cls_out = x[:, 0]
        return cls_out.cpu().numpy().astype(np.float32)
