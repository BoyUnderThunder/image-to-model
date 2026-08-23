"""Building the solid as a volume rather than two stitched sheets.

The sheet mesher builds a front height field and a back one and sews their
boundaries together. That works, but it bakes in three problems: a seam right
around the silhouette where the two sheets join, a minimum wall thickness needed
to stop front and back vertices coinciding (which would make the join
non-manifold), and a hard split of every vertex into "front" or "back" that
shows up as a ragged line in the texture.

Sampling the solid into a scalar field and extracting one isosurface removes all
three. The surface wraps around the silhouette because that is simply where the
field changes sign, so there is no join to make and nothing to classify.

The field is

    f(x, y, z) = H(x, y) - |z - M(x, y)|

where ``H`` is the local half-thickness and ``M`` the midplane. Both fall to
zero outside the subject, so the solid closes on its own. Because the inflation
profile approaches the silhouette with a vertical tangent, that closure is a
rounded edge rather than a knife edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..errors import ReconstructionError
from ..imaging import resize_array
from ..logging import get_logger
from ..types import CameraIntrinsics, Mesh
from .isosurface import surface_nets
from .lift import camera_distance
from .surface import _RIM_WIDTH_FRACTION, _WORLD_WIDTH, RIM_PROFILES

__all__ = ["VolumeResult", "build_volume_solid", "choose_resolution"]

log = get_logger("geometry.volume")

#: Cells of empty space kept around the subject so the field is negative at the
#: border and the surface closes instead of being clipped by the grid.
_PAD_CELLS = 3


@dataclass
class VolumeResult:
    """A solid extracted from a scalar field."""

    mesh: Mesh
    intrinsics: CameraIntrinsics
    distance: float
    image_size: tuple[int, int]
    resolution: int
    """Cells across the subject's longest side."""


#: Triangles produced per unit of resolution squared, measured on compact
#: subjects (38k at N=64, 85k at N=96 -- both 9.3 * N^2). Thin subjects come out
#: below this, which is the safe direction to be wrong in.
_TRIANGLES_PER_CELL_SQUARED = 9.3


def choose_resolution(target_faces: int, oversample: float = 3.0) -> int:
    """Pick a grid resolution that extracts comfortably above the face budget.

    Surface area, and so triangle count, grows with the square of the
    resolution. This inverts that to land a few times over the budget, leaving
    decimation room to place detail well without extracting far more geometry
    than it is about to collapse -- decimation is the expensive stage, so
    overshooting here costs much more time than it buys quality.
    """
    if target_faces <= 0:
        return 128
    desired = oversample * target_faces
    return int(np.clip(round(math.sqrt(desired / _TRIANGLES_PER_CELL_SQUARED)), 40, 192))


def build_volume_solid(
    depth: np.ndarray,
    mask: np.ndarray,
    relief_scale: float = 1.0,
    relief_mode: str = "inradius",
    thickness: float = 1.0,
    fov_degrees: float = 55.0,
    mask_threshold: float = 0.5,
    resolution: int = 128,
    rim_profile: str = "fillet",
    rim_width: float = 0.0,
) -> VolumeResult:
    """Build a closed solid from a normalised depth map and subject mask.

    ``depth`` must be in [0, 1] with larger meaning further from the camera.
    """
    from ..depth.heuristic import distance_transform

    depth = np.asarray(depth, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    if depth.shape != mask.shape:
        raise ReconstructionError(
            f"Depth shape {depth.shape} does not match mask shape {mask.shape}"
        )

    height, width = depth.shape
    intrinsics = CameraIntrinsics.from_fov(width, height, fov_degrees)

    binary = mask > mask_threshold
    if not binary.any():
        raise ReconstructionError(
            "The subject mask is empty. Try a lower --mask-threshold or a "
            "different --segmentation method."
        )

    subject_extent = max(
        int(np.any(binary, axis=1).sum()), int(np.any(binary, axis=0).sum()), 1
    )
    distance = camera_distance(intrinsics, subject_extent, _WORLD_WIDTH)

    # Relief depth, in world units where the subject's longest side is 1.
    distance_map = distance_transform(binary)
    if relief_mode == "inradius":
        relief_depth = float(relief_scale) * float(distance_map.max()) / subject_extent
    elif relief_mode == "fraction":
        relief_depth = float(relief_scale) * _WORLD_WIDTH
    else:
        raise ValueError(f"relief_mode must be 'inradius' or 'fraction', got {relief_mode!r}")

    # Normalised relief from the depth map: 1 at the nearest point, 0 furthest.
    subject_depth = depth[binary]
    low, high = float(subject_depth.min()), float(subject_depth.max())
    if high - low < 1e-6:
        unit = np.zeros_like(depth)
    else:
        unit = np.clip((high - depth) / (high - low), 0.0, 1.0)
    unit = np.where(binary, unit, 0.0).astype(np.float32)

    # Taper towards the silhouette. A learned depth map does not fall to zero
    # at the outline on its own, and without this the solid would be sheared
    # off flat there instead of rounding over.
    if rim_width <= 0:
        rim_width = max(1.0, _RIM_WIDTH_FRACTION * math.sqrt(max(1, int(binary.sum()))))
    if rim_profile not in RIM_PROFILES:
        raise ValueError(f"Unknown rim profile {rim_profile!r}")
    rim = np.where(binary, RIM_PROFILES[rim_profile](distance_map / rim_width), 0.0)

    # Cubic voxels: one world unit spans `resolution` cells on every axis.
    cell = _WORLD_WIDTH / float(resolution)

    relief = (rim * unit * relief_depth).astype(np.float32)
    half_thickness = relief * (1.0 + float(thickness)) * 0.5
    midplane = relief * (1.0 - float(thickness)) * 0.5

    peak = float(half_thickness.max())
    if peak <= 1e-6:
        raise ReconstructionError(
            "The predicted relief is flat, so there is no solid to build. Try "
            "raising --relief-scale."
        )

    rows = np.flatnonzero(np.any(binary, axis=1))
    cols = np.flatnonzero(np.any(binary, axis=0))
    row0, row1 = int(rows[0]), int(rows[-1]) + 1
    col0, col1 = int(cols[0]), int(cols[-1]) + 1

    # World bounds of the subject, matching the orthographic convention used
    # elsewhere: X = (col - cx) / subject_extent, Y = -(row - cy) / extent.
    x_min = (col0 - intrinsics.cx) / subject_extent - (_PAD_CELLS + 0.5) * cell
    x_max = (col1 - intrinsics.cx) / subject_extent + (_PAD_CELLS + 0.5) * cell
    y_min = -(row1 - intrinsics.cy) / subject_extent - (_PAD_CELLS + 0.5) * cell
    y_max = -(row0 - intrinsics.cy) / subject_extent + (_PAD_CELLS + 0.5) * cell
    z_limit = peak + max(abs(float(midplane.min())), abs(float(midplane.max())))
    # Half-cell offset, so no sample plane lands exactly on the solid's extreme.
    # At such a plane the surface is tangent to the grid and the field there is
    # within interpolation noise of zero, so the sign speckles between
    # neighbouring cells and the extracted surface comes out non-manifold. A
    # half-cell shift puts the nearest samples a clear distance off the surface.
    z_min = -z_limit - (_PAD_CELLS + 0.5) * cell
    z_max = z_limit + (_PAD_CELLS + 0.5) * cell

    nx = max(4, int(math.ceil((x_max - x_min) / cell)) + 1)
    ny = max(4, int(math.ceil((y_max - y_min) / cell)) + 1)
    nz = max(4, int(math.ceil((z_max - z_min) / cell)) + 1)

    # Resample the two 2D fields onto the grid's XY footprint. The grid is
    # built in world space, so this is a straight resize of the pixel window.
    window = (slice(row0, row1), slice(col0, col1))
    inner_x = nx - 2 * _PAD_CELLS
    inner_y = ny - 2 * _PAD_CELLS
    if inner_x < 2 or inner_y < 2:
        raise ReconstructionError(
            "The subject is too small to voxelise at this resolution. Raise "
            "--volume-resolution or --working-resolution."
        )

    # Rows run top-to-bottom in the image but Y increases upwards, so flip.
    half_grid = np.zeros((nx, ny), dtype=np.float32)
    mid_grid = np.zeros((nx, ny), dtype=np.float32)
    # Area averaging, not Lanczos: Lanczos rings, and an overshoot of a few
    # parts in a thousand near where the field crosses zero flips the sign in
    # isolated cells. That speckle is what makes the extracted surface
    # non-manifold, so the filter choice here is a correctness matter.
    resized_half = resize_array(half_thickness[window], (inner_x, inner_y), mode="area").T[:, ::-1]
    resized_mid = resize_array(midplane[window], (inner_x, inner_y), mode="area").T[:, ::-1]
    interior = (slice(_PAD_CELLS, _PAD_CELLS + inner_x), slice(_PAD_CELLS, _PAD_CELLS + inner_y))
    half_grid[interior] = resized_half
    mid_grid[interior] = resized_mid

    z_axis = z_min + np.arange(nz, dtype=np.float32) * cell
    field = half_grid[:, :, None] - np.abs(z_axis[None, None, :] - mid_grid[:, :, None])

    log.debug(
        "volume: grid %dx%dx%d (%.1f MB), relief_depth=%.4f",
        nx,
        ny,
        nz,
        field.nbytes / 1e6,
        relief_depth,
    )

    mesh = surface_nets(field, spacing=(cell, cell, cell), origin=(x_min, y_min, z_min))
    if mesh.is_empty:
        raise ReconstructionError("Isosurface extraction produced no geometry.")

    return VolumeResult(
        mesh=mesh,
        intrinsics=intrinsics,
        distance=distance,
        image_size=(width, height),
        resolution=resolution,
    )
