"""Learned monocular depth estimation via HuggingFace transformers.

This is the "AI" half of the pipeline. It runs any transformers depth model;
Depth Anything V2 Small is the default because it is small enough to download
quickly and fast enough to run on CPU.

Requires the ``ai`` extra::

    pip install 'image-to-model[ai]'

and network access to huggingface.co the first time a model is used, after
which weights are cached locally by ``huggingface_hub``.
"""

from __future__ import annotations

import numpy as np

from ..errors import DependencyMissing, DepthEstimationError
from ..logging import get_logger
from ..types import DepthMap
from .base import DepthEstimator

__all__ = ["LearnedDepthEstimator", "MODEL_ALIASES", "resolve_device"]

log = get_logger("depth.learned")


#: Short names mapped to HuggingFace model ids. Any id can also be passed
#: directly. The boolean records whether the model emits *disparity* (larger
#: means nearer) and therefore needs inverting into our depth convention.
MODEL_ALIASES: dict[str, tuple[str, bool]] = {
    "depth-anything": ("depth-anything/Depth-Anything-V2-Small-hf", True),
    "depth-anything-small": ("depth-anything/Depth-Anything-V2-Small-hf", True),
    "depth-anything-base": ("depth-anything/Depth-Anything-V2-Base-hf", True),
    "depth-anything-large": ("depth-anything/Depth-Anything-V2-Large-hf", True),
    "dpt": ("Intel/dpt-hybrid-midas", True),
    "dpt-large": ("Intel/dpt-large", True),
    "midas": ("Intel/dpt-hybrid-midas", True),
    "zoedepth": ("Intel/zoedepth-nyu-kitti", False),  # metric depth, already far-is-large
}

#: Preference order when the caller asks for "auto".
AUTO_ORDER = ("depth-anything", "dpt")

# Loading weights costs seconds, so keep models alive per (id, device).
_MODEL_CACHE: dict[tuple[str, str], tuple[object, object]] = {}


def resolve_device(device: str = "auto") -> str:
    """Pick a torch device string."""
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _require_torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise DependencyMissing("Learned depth estimation", "torch", extra="ai") from exc


def _require_transformers():
    try:
        import transformers

        return transformers
    except ImportError as exc:
        raise DependencyMissing("Learned depth estimation", "transformers", extra="ai") from exc


def resolve_model(name: str) -> tuple[str, bool]:
    """Map an alias or raw HuggingFace id to ``(model_id, is_disparity)``."""
    if name in MODEL_ALIASES:
        return MODEL_ALIASES[name]
    # An unrecognised name is assumed to be a HuggingFace id. Most published
    # relative-depth models emit disparity, so that is the safer default.
    return name, True


class LearnedDepthEstimator(DepthEstimator):
    """Monocular depth from a transformers ``AutoModelForDepthEstimation``."""

    requires_weights = True

    def __init__(self, model: str = "depth-anything", device: str = "auto") -> None:
        self.model_id, self.is_disparity = resolve_model(model)
        self.device = resolve_device(device)
        self.name = f"learned:{self.model_id}"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def _load(self):
        key = (self.model_id, self.device)
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]

        _require_torch()
        transformers = _require_transformers()

        log.info("Loading depth model %s on %s", self.model_id, self.device)
        try:
            processor = transformers.AutoImageProcessor.from_pretrained(self.model_id)
            model = transformers.AutoModelForDepthEstimation.from_pretrained(self.model_id)
        except Exception as exc:
            raise DepthEstimationError(
                f"Could not load depth model {self.model_id!r}: {exc}. "
                "Check the model id and that huggingface.co is reachable, or pass "
                "--depth-model heuristic to run without downloaded weights."
            ) from exc

        model = model.to(self.device).eval()
        _MODEL_CACHE[key] = (processor, model)
        return processor, model

    def estimate(self, rgb: np.ndarray, mask: np.ndarray | None = None) -> DepthMap:
        torch = _require_torch()
        from PIL import Image

        processor, model = self._load()
        rgb = np.asarray(rgb, dtype=np.uint8)
        height, width = rgb.shape[:2]

        inputs = processor(images=Image.fromarray(rgb, mode="RGB"), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            predicted = model(**inputs).predicted_depth

        if predicted.ndim == 3:
            predicted = predicted.unsqueeze(1)  # (B, 1, H, W)
        # Models predict at their own working resolution, so resample back.
        resized = torch.nn.functional.interpolate(
            predicted.float(), size=(height, width), mode="bicubic", align_corners=False
        )
        values = resized.squeeze().detach().cpu().numpy().astype(np.float32)

        if values.shape != (height, width):
            raise DepthEstimationError(
                f"Depth model returned shape {values.shape}, expected {(height, width)}"
            )
        if not np.isfinite(values).any():
            raise DepthEstimationError(f"Depth model {self.model_id!r} returned no finite values")
        values = np.nan_to_num(values, nan=float(np.nanmedian(values)))

        if self.is_disparity:
            # Disparity is large when near; flip it so large means far.
            values = float(values.max()) - values

        return DepthMap(
            depth=values,
            mask=None if mask is None else np.asarray(mask, dtype=np.float32),
            source=self.name,
            metadata={
                "model_id": self.model_id,
                "device": self.device,
                "was_disparity": self.is_disparity,
                "relative": True,
            },
        )
