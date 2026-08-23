"""Monocular depth estimation with graceful degradation.

``create_estimator("auto")`` prefers a learned model and falls back to the
dependency-free heuristic when torch/transformers are missing. ``estimate_depth``
goes further and also catches runtime failures (no network on first run, a bad
model id) so a reconstruction still completes rather than aborting.
"""

from __future__ import annotations

import numpy as np

from ..errors import DepthEstimationError
from ..logging import get_logger
from ..types import DepthMap
from .base import DepthEstimator
from .heuristic import HeuristicDepthEstimator
from .learned import AUTO_ORDER, MODEL_ALIASES, LearnedDepthEstimator, resolve_device

__all__ = [
    "DepthEstimator",
    "HeuristicDepthEstimator",
    "LearnedDepthEstimator",
    "create_estimator",
    "estimate_depth",
    "available_models",
    "MODEL_ALIASES",
    "resolve_device",
]

log = get_logger("depth")


def available_models() -> list[str]:
    """Depth model names usable right now."""
    names = ["heuristic"]
    if LearnedDepthEstimator.is_available():
        names.extend(sorted(MODEL_ALIASES))
    return names


def create_estimator(model: str = "auto", device: str = "auto") -> DepthEstimator:
    """Build a depth estimator by name.

    ``auto`` picks a learned model when the optional dependencies are present
    and the heuristic otherwise.
    """
    if model == "heuristic":
        return HeuristicDepthEstimator()

    if model == "auto":
        if LearnedDepthEstimator.is_available():
            return LearnedDepthEstimator(AUTO_ORDER[0], device=device)
        log.info(
            "torch/transformers not installed; using heuristic depth. "
            "Install 'image-to-model[ai]' for learned depth estimation."
        )
        return HeuristicDepthEstimator()

    return LearnedDepthEstimator(model, device=device)


def estimate_depth(
    rgb: np.ndarray,
    mask: np.ndarray | None = None,
    model: str = "auto",
    device: str = "auto",
    allow_fallback: bool = True,
) -> DepthMap:
    """Estimate depth, optionally falling back to the heuristic on failure.

    Weight downloads fail for ordinary reasons (offline machine, blocked host),
    and producing a usable model beats aborting the run, so the fallback is on
    by default. Set ``allow_fallback=False`` to make those failures fatal.
    """
    estimator = create_estimator(model, device=device)
    try:
        return estimator.estimate(rgb, mask)
    except Exception as exc:
        if not allow_fallback or isinstance(estimator, HeuristicDepthEstimator):
            raise
        log.warning(
            "Depth model %r failed (%s); falling back to heuristic depth.",
            getattr(estimator, "model_id", model),
            exc,
        )
        try:
            return HeuristicDepthEstimator().estimate(rgb, mask)
        except Exception as inner:  # pragma: no cover - the fallback is very simple
            raise DepthEstimationError(f"All depth estimators failed: {inner}") from inner
