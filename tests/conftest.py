"""Shared fixtures.

Tests run at small resolutions and tiny polygon budgets on purpose: the
pipeline's cost is dominated by decimation, and the properties worth asserting
(watertightness, budgets, UV ranges, file validity) hold at any size.
"""

from __future__ import annotations

import numpy as np
import pytest

from image_to_model.config import ReconstructionConfig
from image_to_model.samples import make_sample
from image_to_model.types import Mesh


@pytest.fixture
def unit_cube() -> Mesh:
    """A closed unit cube with outward-facing winding."""
    vertices = (
        np.array(
            [
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            ],
            dtype=np.float32,
        )
        - 0.5
    )
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [2, 3, 7], [2, 7, 6],
            [1, 2, 6], [1, 6, 5],
            [0, 4, 7], [0, 7, 3],
        ],
        dtype=np.int32,
    )
    colors = ((vertices + 0.5) * 255).astype(np.uint8)
    return Mesh(vertices=vertices, faces=faces, vertex_colors=colors)


@pytest.fixture
def ball_image() -> np.ndarray:
    return make_sample("ball", 128)


@pytest.fixture
def star_image() -> np.ndarray:
    """A sample with a real alpha channel."""
    return make_sample("star", 128)


@pytest.fixture
def fast_config() -> ReconstructionConfig:
    """A configuration small enough to run many times in a test suite."""
    return ReconstructionConfig(
        working_resolution=128,
        target_faces=600,
        smooth_iterations=2,
        texture_size=128,
    )
