"""Lifting pixels into 3D.

A monocular depth map has no absolute scale, so rather than pretending to
recover metric geometry the pipeline works in a normalised object space: the
subject is one unit wide, and the predicted depth is mapped onto a relief whose
extent is a chosen fraction of that width. Perspective divergence is still
modelled, so a wide-angle photo produces the mild fan-out it should.

Output convention is the one the exporters expect: X right, **Y up**, +Z
towards the camera.
"""

from __future__ import annotations

import numpy as np

from ..types import CameraIntrinsics

__all__ = ["camera_distance", "project_to_world", "world_to_pixel"]


def camera_distance(intrinsics: CameraIntrinsics, subject_pixels: float, world_width: float) -> float:
    """Distance from the camera at which ``subject_pixels`` spans ``world_width``.

    Follows straight from the pinhole relation ``pixels = fx * world / distance``.
    """
    if subject_pixels <= 0:
        raise ValueError("subject_pixels must be positive")
    return float(intrinsics.fx * world_width / subject_pixels)


def project_to_world(
    cols: np.ndarray,
    rows: np.ndarray,
    relief: np.ndarray,
    intrinsics: CameraIntrinsics,
    distance: float,
    orthographic: bool = True,
) -> np.ndarray:
    """Turn pixel coordinates plus a relief height into world points.

    ``relief`` is displacement towards the camera, in world units, measured
    from the plane sitting ``distance`` away. The returned points are centred
    on that plane, so a flat relief lands on Z = 0.

    Orthographic projection is the default, and for this pipeline it is the
    more correct choice. Perspective divergence shrinks a point's lateral
    offset as it comes towards the camera, so a surface bulging forward also
    narrows -- turning a reconstructed sphere into a teardrop. That narrowing
    is only right if the relief is true metric depth. Monocular depth is
    relative and the relief height is a prior, so the silhouette is the widest
    cross-section and lateral position should not move with height.
    """
    cols = np.asarray(cols, dtype=np.float32)
    rows = np.asarray(rows, dtype=np.float32)
    relief = np.asarray(relief, dtype=np.float32)

    # Normalised image-plane directions.
    u = (cols - intrinsics.cx) / intrinsics.fx
    v = (rows - intrinsics.cy) / intrinsics.fy

    # Under perspective, points nearer the camera subtend a wider angle, so
    # their lateral offset shrinks for the same pixel position. Holding the
    # ray distance constant is exactly what makes the projection orthographic.
    ray_distance = distance if orthographic else distance - relief

    points = np.empty(relief.shape + (3,), dtype=np.float32)
    points[..., 0] = u * ray_distance
    points[..., 1] = -v * ray_distance  # image rows increase downwards, Y increases up
    points[..., 2] = relief
    return points


def world_to_pixel(
    points: np.ndarray,
    intrinsics: CameraIntrinsics,
    distance: float,
    orthographic: bool = True,
) -> np.ndarray:
    """Inverse of :func:`project_to_world`, giving ``(col, row)`` per point.

    Used to re-derive texture coordinates after decimation has moved vertices,
    which is exact and avoids interpolating UVs through edge collapses.
    """
    points = np.asarray(points, dtype=np.float32)
    if orthographic:
        ray_distance = np.full(points.shape[:-1], float(distance), dtype=np.float32)
    else:
        ray_distance = np.maximum(distance - points[..., 2], 1e-6)

    cols = points[..., 0] / ray_distance * intrinsics.fx + intrinsics.cx
    rows = -points[..., 1] / ray_distance * intrinsics.fy + intrinsics.cy
    return np.stack([cols, rows], axis=-1).astype(np.float32)
