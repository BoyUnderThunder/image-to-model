"""Depth estimator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..types import DepthMap

__all__ = ["DepthEstimator"]


class DepthEstimator(ABC):
    """Predicts a relative depth map from a single RGB image.

    Implementations return depth where **larger means further from the
    camera**. Models that emit disparity (larger means nearer) must invert
    before returning, so downstream geometry can rely on one convention.
    """

    name: str = "base"
    requires_weights: bool = False

    @classmethod
    def is_available(cls) -> bool:
        """Whether this estimator can run in the current environment."""
        return True

    @abstractmethod
    def estimate(self, rgb: np.ndarray, mask: np.ndarray | None = None) -> DepthMap:
        """Predict depth for an RGB uint8 image.

        ``mask`` is the subject alpha when one is known. Estimators may use it
        to focus normalisation on the subject, or ignore it entirely.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
