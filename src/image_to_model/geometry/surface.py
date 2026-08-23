"""Building a closed surface from a depth map.

The subject is meshed as two height fields over the same grid: a front sheet
driven by predicted depth, and a back sheet that mirrors it inward. Both are
tapered to zero at the silhouette by a rim factor, so their boundary rings land
on identical positions and welding turns the pair into one watertight solid.

That taper is what makes the result a closed model rather than a floating
sheet. It costs a little accuracy right at the silhouette, where the true
surface is turning away from the camera and depth is least reliable anyway.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..errors import ReconstructionError
from ..logging import get_logger
from ..types import CameraIntrinsics, Mesh
from .lift import camera_distance, project_to_world

__all__ = ["SurfaceResult", "build_surface", "choose_stride", "RIM_PROFILES", "FRONT", "BACK"]

log = get_logger("geometry.surface")

FRONT = 0
BACK = 1

#: World-space width the subject is normalised to before scaling to target units.
_WORLD_WIDTH = 1.0

#: Default rim width, as a fraction of the subject's square-root area. Wider
#: spreads the rounding over more triangles, which measurably lowers surface
#: roughness; past roughly 0.12 the gain flattens out and the shape just
#: gets softer, so this sits at the knee.
_RIM_WIDTH_FRACTION = 0.12


@dataclass
class SurfaceResult:
    """A freshly built surface plus what is needed to texture it later."""

    mesh: Mesh
    pixel_coords: np.ndarray
    """``(N, 2)`` source pixel ``(col, row)`` for each vertex, in the
    resolution of the depth map it was built from."""

    side: np.ndarray
    """``(N,)`` uint8, :data:`FRONT` or :data:`BACK` per vertex."""

    intrinsics: CameraIntrinsics
    distance: float
    """Camera-to-subject distance used for the projection."""

    image_size: tuple[int, int]
    """``(width, height)`` of the depth map."""


def choose_stride(
    mask: np.ndarray, target_faces: int, oversample: float = 6.0, max_stride: int = 24
) -> int:
    """Pick a grid step that lands comfortably above the final face budget.

    Meshing at exactly the budget would leave decimation nothing to work with,
    and meshing at full pixel resolution wastes a lot of time producing
    triangles that are about to be collapsed. Aiming a few times over the
    budget gives the decimator enough freedom to place detail where it matters.
    """
    subject_pixels = int(np.count_nonzero(mask))
    if subject_pixels == 0:
        return 1
    if target_faces <= 0:
        return 1

    # A grid cell inside the mask yields ~2 front triangles and ~2 back ones.
    desired_cells = max(1.0, (oversample * target_faces) / 4.0)
    stride = math.sqrt(subject_pixels / desired_cells)
    return int(np.clip(round(stride), 1, max_stride))


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _fillet(t: np.ndarray) -> np.ndarray:
    """Quarter-circle profile: 0 at t=0, 1 at t=1, vertical tangent at t=0.

    This is the shape a rounded edge actually has. Where a smoothstep leaves
    the silhouette flat and lets the side wall drop away at a right angle,
    this leaves it tangent to the view direction, which is the defining
    property of a silhouette on a smooth object.
    """
    t = np.clip(t, 0.0, 1.0)
    return np.sqrt(np.clip(1.0 - (1.0 - t) ** 2, 0.0, 1.0))


#: Named rim profiles. ``fillet`` rounds the silhouette; ``smoothstep`` keeps
#: the flatter, harder-edged profile; ``linear`` gives a plain chamfer.
RIM_PROFILES = {
    "fillet": _fillet,
    "smoothstep": _smoothstep,
    "linear": lambda t: np.clip(t, 0.0, 1.0),
}


def _rim_factor(distance: np.ndarray, rim_width: float, profile: str = "fillet") -> np.ndarray:
    """Taper the relief towards the silhouette so the model closes smoothly.

    ``distance`` is distance-to-background in source pixels, sampled at the
    grid points. Measuring it at full resolution rather than on the subsampled
    grid matters for the ``fillet`` profile: a grid-quantised distance steps in
    whole cells, and a profile with a vertical tangent turns that first step
    into a large jump in height, which shows up as a sawtooth along the
    silhouette. Fractional distances remove the aliasing at its source.

    The taper stops just short of zero at the outermost ring. A rim reaching
    exactly zero would put front and back vertices in the same place, and
    merging those turns any boundary edge with both endpoints on the rim into a
    four-way non-manifold edge.
    """
    if rim_width <= 0:
        return np.ones_like(distance, dtype=np.float32)
    if profile not in RIM_PROFILES:
        raise ValueError(
            f"Unknown rim profile {profile!r}. Choose from: {', '.join(sorted(RIM_PROFILES))}"
        )
    return RIM_PROFILES[profile](distance / float(rim_width)).astype(np.float32)


def _boundary_directed_edges(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    """Directed edges that belong to exactly one face.

    On a consistently wound surface each interior edge appears once in each
    direction, so an edge whose reverse is absent lies on the boundary. Keeping
    the direction lets the stitch band inherit the front sheet's orientation.
    """
    if len(faces) == 0:
        return np.zeros((0, 2), dtype=np.int64)

    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    ).astype(np.int64)

    keys = edges[:, 0] * n_vertices + edges[:, 1]
    reversed_keys = edges[:, 1] * n_vertices + edges[:, 0]

    order = np.argsort(keys)
    sorted_keys = keys[order]
    positions = np.searchsorted(sorted_keys, reversed_keys)
    positions = np.clip(positions, 0, len(sorted_keys) - 1)
    has_reverse = sorted_keys[positions] == reversed_keys

    return edges[~has_reverse]


def _stitch_band(boundary: np.ndarray, offset: int) -> np.ndarray:
    """Triangles joining a front boundary loop to the matching back loop.

    Front and back share connectivity, so back vertex ``v + offset`` is the
    counterpart of front vertex ``v`` and the two boundary loops correspond
    edge for edge.
    """
    if len(boundary) == 0:
        return np.zeros((0, 3), dtype=np.int32)
    a, b = boundary[:, 0], boundary[:, 1]
    a_back, b_back = a + offset, b + offset
    return np.concatenate(
        [np.stack([a, b, b_back], axis=1), np.stack([a, b_back, a_back], axis=1)], axis=0
    ).astype(np.int32)


def _grid_triangles(index_map: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Triangulate a masked regular grid, front-facing (+Z) winding.

    Quads with all four corners present give two triangles; quads with exactly
    three give one, which keeps the silhouette from looking stepped.
    """
    a, b = inside[:-1, :-1], inside[:-1, 1:]
    c, d = inside[1:, :-1], inside[1:, 1:]
    ia, ib = index_map[:-1, :-1], index_map[:-1, 1:]
    ic, id_ = index_map[1:, :-1], index_map[1:, 1:]

    groups: list[np.ndarray] = []

    full = a & b & c & d
    if full.any():
        groups.append(np.stack([ia[full], ic[full], ib[full]], axis=1))
        groups.append(np.stack([ib[full], ic[full], id_[full]], axis=1))

    # Exactly three corners present: one triangle, wound to face +Z.
    for present, corners in (
        (a & b & c & ~d, (ia, ic, ib)),
        (a & b & ~c & d, (ia, id_, ib)),
        (a & ~b & c & d, (ia, ic, id_)),
        (~a & b & c & d, (ib, ic, id_)),
    ):
        if present.any():
            groups.append(np.stack([corner[present] for corner in corners], axis=1))

    if not groups:
        return np.zeros((0, 3), dtype=np.int32)
    return np.concatenate(groups, axis=0).astype(np.int32)


def build_surface(
    depth: np.ndarray,
    mask: np.ndarray,
    relief_scale: float = 0.35,
    thickness: float = 0.55,
    close_back: bool = True,
    fov_degrees: float = 55.0,
    mask_threshold: float = 0.5,
    stride: int = 1,
    rim_width: float = 0.0,
    min_thickness: float = 0.012,
    rim_profile: str = "fillet",
) -> SurfaceResult:
    """Build a surface mesh from a normalised depth map and subject mask.

    ``depth`` must be in [0, 1] with larger meaning further from the camera.

    ``min_thickness`` is the smallest separation between the front and back
    sheets, as a fraction of the subject's width. It keeps the silhouette from
    closing to a knife edge — which would put front and back vertices in the
    same place and make the surface non-manifold — and gives the model an edge
    that physics and 3D printing can actually handle.
    """
    depth = np.asarray(depth, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    if depth.shape != mask.shape:
        raise ReconstructionError(
            f"Depth shape {depth.shape} does not match mask shape {mask.shape}"
        )

    height, width = depth.shape
    intrinsics = CameraIntrinsics.from_fov(width, height, fov_degrees)

    # Sample the grid. Slicing keeps the sampled coordinates exact, and the
    # final row/column is appended so the subject is never clipped.
    stride = max(1, int(stride))
    rows = np.arange(0, height, stride, dtype=np.int32)
    cols = np.arange(0, width, stride, dtype=np.int32)
    if rows[-1] != height - 1:
        rows = np.append(rows, height - 1)
    if cols[-1] != width - 1:
        cols = np.append(cols, width - 1)

    depth_grid = depth[np.ix_(rows, cols)]
    mask_grid = mask[np.ix_(rows, cols)]
    inside = mask_grid > mask_threshold
    if not inside.any():
        raise ReconstructionError(
            "The subject mask is empty at the meshing resolution. Try a lower "
            "--mask-threshold, a different --segmentation method, or a larger "
            "--working-resolution."
        )

    # Height towards the camera, normalised across the subject only.
    subject_depth = depth_grid[inside]
    low, high = float(subject_depth.min()), float(subject_depth.max())
    if high - low < 1e-6:
        relief_unit = np.zeros_like(depth_grid)
    else:
        relief_unit = np.clip((high - depth_grid) / (high - low), 0.0, 1.0)

    full_binary = mask > mask_threshold
    subject_pixels = max(1, int(full_binary.sum()))
    from ..depth.heuristic import distance_transform

    distance_grid = distance_transform(full_binary)[np.ix_(rows, cols)]

    if rim_width <= 0:
        # Scale the taper with the subject so it stays proportionally narrow.
        rim_width = max(1.0, _RIM_WIDTH_FRACTION * math.sqrt(subject_pixels))
    rim = _rim_factor(distance_grid, rim_width, rim_profile)
    rim = np.where(inside, rim, 0.0).astype(np.float32)

    rows_grid, cols_grid = np.meshgrid(rows, cols, indexing="ij")
    relief_depth = float(relief_scale) * _WORLD_WIDTH

    # Distance is set so the subject spans one world unit across its larger side.
    subject_rows = np.any(mask > mask_threshold, axis=1)
    subject_cols = np.any(mask > mask_threshold, axis=0)
    subject_extent = max(int(subject_rows.sum()), int(subject_cols.sum()), 1)
    distance = camera_distance(intrinsics, subject_extent, _WORLD_WIDTH)

    half_gap = 0.5 * float(min_thickness) * _WORLD_WIDTH
    front_relief = rim * relief_unit * relief_depth + half_gap
    front_points = project_to_world(cols_grid, rows_grid, front_relief, intrinsics, distance)

    index_map = np.full(inside.shape, -1, dtype=np.int32)
    index_map[inside] = np.arange(int(inside.sum()), dtype=np.int32)

    front_faces = _grid_triangles(index_map, inside)
    if len(front_faces) == 0:
        raise ReconstructionError(
            "The subject is too small to mesh at this resolution. Increase "
            "--working-resolution or lower --target-faces."
        )

    vertices = [front_points[inside]]
    pixel_coords = [np.stack([cols_grid[inside], rows_grid[inside]], axis=1).astype(np.float32)]
    sides = [np.full(int(inside.sum()), FRONT, dtype=np.uint8)]
    faces = [front_faces]

    if close_back:
        # The back mirrors the front inward. A thickness of 0 still leaves a
        # flat sheet at Z=0 rather than nothing, which is the bas-relief case.
        back_relief = -(rim * relief_unit * relief_depth * float(thickness)) - half_gap
        back_points = project_to_world(cols_grid, rows_grid, back_relief, intrinsics, distance)
        offset = int(inside.sum())

        vertices.append(back_points[inside])
        pixel_coords.append(pixel_coords[0].copy())
        sides.append(np.full(offset, BACK, dtype=np.uint8))
        # Reversed winding so the back faces away from the camera.
        faces.append((front_faces + offset)[:, ::-1])
        # Join the two open loops into a closed solid.
        boundary = _boundary_directed_edges(front_faces, offset)
        faces.append(_stitch_band(boundary, offset))
        log.debug("stitched %d boundary edges", len(boundary))

    mesh = Mesh(
        vertices=np.concatenate(vertices, axis=0),
        faces=np.concatenate(faces, axis=0),
    )
    log.debug(
        "surface: stride=%d grid=%dx%d subject_cells=%d vertices=%d faces=%d",
        stride,
        len(rows),
        len(cols),
        int(inside.sum()),
        mesh.n_vertices,
        mesh.n_faces,
    )

    return SurfaceResult(
        mesh=mesh,
        pixel_coords=np.concatenate(pixel_coords, axis=0),
        side=np.concatenate(sides, axis=0),
        intrinsics=intrinsics,
        distance=distance,
        image_size=(width, height),
    )
