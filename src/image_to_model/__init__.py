"""image-to-model: turn a single photograph into a textured 3D model."""

from __future__ import annotations

from .config import PRESETS, ReconstructionConfig, preset
from .errors import (
    BackendNotFound,
    DependencyMissing,
    DepthEstimationError,
    ExportError,
    ImageToModelError,
    InputError,
    ReconstructionError,
    SegmentationError,
)
from .pipeline import ReconstructionResult, reconstruct, reconstruct_file
from .targets import TargetProfile, ValidationReport, available_targets, get_target
from .types import CameraIntrinsics, DepthMap, Mesh, Subject

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "reconstruct",
    "reconstruct_file",
    "ReconstructionResult",
    "ReconstructionConfig",
    "PRESETS",
    "preset",
    "Mesh",
    "DepthMap",
    "Subject",
    "CameraIntrinsics",
    "TargetProfile",
    "ValidationReport",
    "get_target",
    "available_targets",
    "ImageToModelError",
    "InputError",
    "DependencyMissing",
    "SegmentationError",
    "DepthEstimationError",
    "ReconstructionError",
    "ExportError",
    "BackendNotFound",
]
