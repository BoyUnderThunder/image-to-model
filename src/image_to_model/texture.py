"""Texture baking and UV assignment.

Roblox drops per-vertex colour on import, so a model needs real UVs and a
baked image to arrive with its appearance intact. The atlas is a simple
two-tile layout: the subject as photographed on the top half for the front
surface, and a darkened copy on the bottom half for the back.

Darkening the back is a deliberate cheat. A single photograph says nothing
about the far side of the object, and mirroring the front at full brightness
reads as obviously wrong; a darker copy reads as the object turned away from
the light, which is what a viewer expects.

UVs are assigned by projecting each vertex back through the same camera that
created it. Doing it this way means decimation can move vertices around freely
and the texture still lands correctly, with no need to carry UVs through the
edge collapses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .imaging import bilinear_sample, resize_array
from .types import CameraIntrinsics

__all__ = ["AtlasLayout", "build_atlas", "compute_uvs", "sample_vertex_colors", "inpaint_background"]


def inpaint_background(rgb: np.ndarray, mask: np.ndarray, iterations: int = 12) -> np.ndarray:
    """Flood subject colour outwards into the background.

    Without this, a texel that straddles the silhouette picks up the original
    background and the model gets a bright halo along its edge. Growing the
    subject's own colours outwards means any bleeding stays in-palette.
    """
    image = rgb.astype(np.float32).copy()
    known = np.asarray(mask, dtype=np.float32) > 0.5
    image[~known] = 0.0

    for _ in range(int(iterations)):
        if known.all():
            break
        weight = known.astype(np.float32)
        total = np.zeros_like(image)
        count = np.zeros_like(weight)
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
            total += np.roll(image * weight[..., None], shift, axis=axis)
            count += np.roll(weight, shift, axis=axis)

        fillable = (~known) & (count > 0)
        if not fillable.any():
            break
        safe_count = np.where(count > 0, count, 1.0)[..., None]
        filled = total / safe_count
        image[fillable] = filled[fillable]
        known = known | fillable

    return np.clip(image, 0, 255).astype(np.uint8)


@dataclass
class AtlasLayout:
    """How subject pixels map into the baked atlas.

    The front occupies the top half of the image and the back the bottom half,
    with a small inset on each so bilinear filtering and mipmapping cannot pull
    one tile's colour across into the other.
    """

    x0: float
    y0: float
    width: float
    height: float
    inset: float = 0.004

    def uv(self, pixel_coords: np.ndarray, is_back: np.ndarray) -> np.ndarray:
        """Map source pixel ``(col, row)`` pairs to atlas UVs in [0, 1]."""
        cols = np.asarray(pixel_coords, dtype=np.float32)[:, 0]
        rows = np.asarray(pixel_coords, dtype=np.float32)[:, 1]

        u = np.clip((cols - self.x0) / max(self.width, 1e-6), 0.0, 1.0)
        v = np.clip((rows - self.y0) / max(self.height, 1e-6), 0.0, 1.0)

        # Squeeze each tile into its half, leaving the inset as a guard band.
        half = 0.5 - self.inset
        v_tile = self.inset + v * (half - self.inset)
        v_final = np.where(np.asarray(is_back, dtype=bool), 0.5 + v_tile, v_tile)
        return np.stack([u, v_final], axis=1).astype(np.float32)


def build_atlas(
    rgb: np.ndarray,
    mask: np.ndarray,
    texture_size: int = 1024,
    backface_darkening: float = 0.55,
    padding: float = 0.04,
) -> tuple[np.ndarray, AtlasLayout]:
    """Bake a two-tile texture and return it with its layout.

    ``padding`` widens the crop around the subject as a fraction of its size,
    which keeps the silhouette away from the tile edge.
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    mask = np.asarray(mask, dtype=np.float32)
    height, width = rgb.shape[:2]

    rows = np.any(mask > 0.5, axis=1)
    cols = np.any(mask > 0.5, axis=0)
    if rows.any() and cols.any():
        y0, y1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
        x0, x1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    else:
        x0, y0, x1, y1 = 0, 0, width, height

    pad_x = (x1 - x0) * padding
    pad_y = (y1 - y0) * padding
    x0 = max(0.0, x0 - pad_x)
    y0 = max(0.0, y0 - pad_y)
    x1 = min(float(width), x1 + pad_x)
    y1 = min(float(height), y1 + pad_y)
    crop_width = max(x1 - x0, 1.0)
    crop_height = max(y1 - y0, 1.0)

    filled = inpaint_background(rgb, mask)
    crop = filled[int(y0) : int(np.ceil(y1)), int(x0) : int(np.ceil(x1))]
    if crop.size == 0:
        crop = filled

    tile_width = int(texture_size)
    tile_height = max(1, int(texture_size) // 2)
    front_tile = resize_array(crop, (tile_width, tile_height))
    back_tile = np.clip(
        front_tile.astype(np.float32) * float(backface_darkening), 0, 255
    ).astype(np.uint8)

    atlas = np.concatenate([front_tile, back_tile], axis=0)
    layout = AtlasLayout(x0=x0, y0=y0, width=crop_width, height=crop_height)
    return atlas, layout


def compute_uvs(
    vertices: np.ndarray,
    is_back: np.ndarray,
    intrinsics: CameraIntrinsics,
    distance: float,
    layout: AtlasLayout,
) -> np.ndarray:
    """UVs for arbitrary vertices, derived by re-projecting them into the image."""
    from .geometry.lift import world_to_pixel

    pixel_coords = world_to_pixel(np.asarray(vertices, dtype=np.float32), intrinsics, distance)
    return layout.uv(pixel_coords, is_back)


def sample_vertex_colors(
    rgb: np.ndarray,
    pixel_coords: np.ndarray,
    is_back: np.ndarray | None = None,
    backface_darkening: float = 0.55,
) -> np.ndarray:
    """Sample per-vertex colours from the source image.

    Used for targets that support vertex colour, and as a fallback when texture
    baking is turned off.
    """
    coords = np.asarray(pixel_coords, dtype=np.float32)
    colors = bilinear_sample(rgb.astype(np.float32), coords[:, 0], coords[:, 1])
    if is_back is not None:
        back = np.asarray(is_back, dtype=bool)
        colors[back] *= float(backface_darkening)
    return np.clip(colors, 0, 255).astype(np.uint8)
