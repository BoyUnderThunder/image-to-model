"""Reconstruction backend interface.

A backend turns a segmented :class:`~image_to_model.types.Subject` into a mesh.
The built-in :mod:`~image_to_model.backends.depth_backend` does it by predicting
depth and meshing the result; the interface exists so a feed-forward
image-to-3D model can be dropped in without touching the surrounding pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import ReconstructionConfig
from ..types import DepthMap, Mesh, Subject

__all__ = ["ReconstructionBackend", "BackendOutput"]


@dataclass
class BackendOutput:
    """What a backend produces.

    The mesh is expected in the pipeline's canonical convention: Y up, +Z
    towards the camera, and not yet scaled to the target's units — the pipeline
    handles unit scaling and axis conversion so every backend behaves alike.
    """

    mesh: Mesh
    texture: np.ndarray | None = None
    depth: DepthMap | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ReconstructionBackend(ABC):
    """Base class for reconstruction strategies."""

    name: str = "base"
    description: str = ""

    @classmethod
    def is_available(cls) -> bool:
        """Whether this backend can run in the current environment."""
        return True

    @abstractmethod
    def reconstruct(self, subject: Subject, config: ReconstructionConfig) -> BackendOutput:
        """Build a mesh from a segmented subject."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
