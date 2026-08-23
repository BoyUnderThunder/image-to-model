"""Silhouette-inflation depth, with no learned weights.

This is the fallback that keeps the pipeline usable when no model can be
downloaded, and it is a reasonable first choice in its own right for flat art:
logos, sprites, clipart and decals, where a learned depth model has little real
depth to find anyway.

The approach is the classic "inflation" idea from sketch-based modelling: push
each pixel out by an amount that grows with its distance from the silhouette
edge, which turns a flat shape into a rounded solid. Image shading is then
mixed in at high frequency so surface detail survives.
"""

from __future__ import annotations

import numpy as np

from ..imaging import gaussian_blur
from ..logging import get_logger
from ..types import DepthMap
from .base import DepthEstimator

__all__ = ["HeuristicDepthEstimator", "distance_transform"]

log = get_logger("depth.heuristic")


def _iterative_distance(mask: np.ndarray, max_iterations: int = 512) -> np.ndarray:
    """Approximate distance-to-background by repeated erosion counting.

    Each pass peels one pixel off the silhouette, so a pixel's value ends up
    being how many passes it survived. That is a chessboard-ish metric rather
    than a true Euclidean one, which is smoothed out afterwards.
    """
    from ..morphology import binary_erode

    distance = np.zeros(mask.shape, dtype=np.float32)
    current = mask.copy()
    for _ in range(max_iterations):
        if not current.any():
            break
        distance += current
        current = binary_erode(current, 1)
    return distance


def distance_transform(mask: np.ndarray) -> np.ndarray:
    """Distance from each foreground pixel to the nearest background pixel."""
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.float32)
    try:
        from scipy.ndimage import distance_transform_edt

        return distance_transform_edt(binary).astype(np.float32)
    except ImportError:
        return _iterative_distance(binary)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return (rgb.astype(np.float32) / 255.0) @ weights


class HeuristicDepthEstimator(DepthEstimator):
    """Depth from silhouette inflation plus a shading-derived detail term."""

    name = "heuristic"
    requires_weights = False

    def __init__(
        self,
        roundness: float = 0.5,
        shading_weight: float = 0.15,
        shading_sigma: float = 4.0,
    ) -> None:
        #: Exponent on normalised distance. 0.5 gives a hemispherical bulge,
        #: 1.0 a cone, and values above that an increasingly pointed profile.
        self.roundness = float(roundness)
        self.shading_weight = float(np.clip(shading_weight, 0.0, 1.0))
        self.shading_sigma = float(shading_sigma)

    def estimate(self, rgb: np.ndarray, mask: np.ndarray | None = None) -> DepthMap:
        rgb = np.asarray(rgb, dtype=np.uint8)
        height, width = rgb.shape[:2]

        if mask is None:
            binary = np.ones((height, width), dtype=bool)
        else:
            binary = np.asarray(mask, dtype=np.float32) > 0.5
        if not binary.any():
            binary = np.ones((height, width), dtype=bool)

        distance = distance_transform(binary)
        peak = float(distance.max())
        if peak <= 1e-6:
            height_field = np.zeros_like(distance)
        else:
            height_field = np.power(distance / peak, self.roundness, dtype=np.float32)

        if self.shading_weight > 0:
            # Only the high-frequency part of the image is used: broad
            # brightness changes are usually albedo or lighting rather than
            # shape, and folding them in warps the whole silhouette.
            luminance = _luminance(rgb)
            detail = luminance - gaussian_blur(luminance, self.shading_sigma)
            spread = float(np.abs(detail).max())
            if spread > 1e-6:
                height_field = height_field + self.shading_weight * (detail / spread)

        height_field = gaussian_blur(height_field.astype(np.float32), 1.0)
        height_field = np.where(binary, height_field, 0.0).astype(np.float32)

        # Height is "towards the camera"; depth is the opposite.
        depth = float(height_field.max()) - height_field

        return DepthMap(
            depth=depth.astype(np.float32),
            mask=None if mask is None else np.asarray(mask, dtype=np.float32),
            source=self.name,
            metadata={
                "roundness": self.roundness,
                "shading_weight": self.shading_weight,
                "relative": True,
            },
        )
