"""Procedurally generated sample images.

Used by the test suite and by ``image-to-model demo`` so the package can be
exercised end to end without shipping binary assets or needing a photo to hand.
"""

from __future__ import annotations

import numpy as np

__all__ = ["shaded_ball", "star_cutout", "toy_rocket", "SAMPLES", "make_sample"]


def _coords(size: int) -> tuple[np.ndarray, np.ndarray]:
    axis = (np.arange(size, dtype=np.float32) - (size - 1) / 2.0) / (size / 2.0)
    return axis[None, :].repeat(size, 0), axis[:, None].repeat(size, 1)


def shaded_ball(size: int = 256, radius: float = 0.62) -> np.ndarray:
    """A lit sphere on a soft studio background, as RGBA uint8.

    Shading follows a real Lambertian term against a fixed light, so a depth
    model has genuine cues to work with rather than a flat disc.
    """
    xs, ys = _coords(size)
    radius_sq = xs**2 + ys**2
    inside = radius_sq <= radius**2

    # Surface normal of the sphere at each pixel.
    z = np.sqrt(np.clip(radius**2 - radius_sq, 0.0, None))
    normals = np.stack([xs, ys, z], axis=-1)
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 1e-6)

    light = np.array([-0.45, -0.55, 0.70], dtype=np.float32)
    light /= np.linalg.norm(light)
    lambert = np.clip((normals * light).sum(axis=-1), 0.0, 1.0)
    specular = np.power(lambert, 48.0) * 0.85

    base = np.array([0.85, 0.28, 0.24], dtype=np.float32)
    shaded = base[None, None, :] * (0.18 + 0.82 * lambert)[..., None] + specular[..., None]

    backdrop = np.clip(0.80 - 0.16 * (ys + 1.0), 0.0, 1.0)
    background = np.repeat(backdrop[..., None], 3, axis=-1) * np.array(
        [0.93, 0.94, 0.97], dtype=np.float32
    )

    rgb = np.where(inside[..., None], shaded, background)
    rgb = np.clip(rgb, 0.0, 1.0)

    rgba = np.empty((size, size, 4), dtype=np.uint8)
    rgba[..., :3] = (rgb * 255).astype(np.uint8)
    rgba[..., 3] = 255  # opaque: the subject must be found by segmentation
    return rgba


def star_cutout(size: int = 256, points: int = 5) -> np.ndarray:
    """A flat star on a transparent background, as RGBA uint8.

    Stands in for the logo/sprite/decal case, where the alpha channel already
    carries the silhouette and there is no real depth to recover.
    """
    xs, ys = _coords(size)
    angle = np.arctan2(ys, xs)
    radius = np.sqrt(xs**2 + ys**2)

    # Radius oscillates between the inner and outer points of the star.
    wave = np.cos(points * angle)
    boundary = 0.42 + 0.32 * (wave * 0.5 + 0.5)
    inside = radius <= boundary

    tint = 0.55 + 0.45 * np.clip(1.0 - radius / np.maximum(boundary, 1e-6), 0.0, 1.0)
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., 0] = np.clip(255 * tint, 0, 255).astype(np.uint8)
    rgba[..., 1] = np.clip(190 * tint, 0, 255).astype(np.uint8)
    rgba[..., 2] = np.clip(40 * tint, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.where(inside, 255, 0).astype(np.uint8)
    return rgba


def toy_rocket(size: int = 256) -> np.ndarray:
    """A blocky rocket on a plain background, as RGBA uint8.

    Several distinct parts at different depths, which is closer to the kind of
    prop someone would actually want in a game.
    """
    xs, ys = _coords(size)
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., 3] = 255

    backdrop = np.clip(0.55 + 0.30 * (ys * 0.5 + 0.5), 0.0, 1.0)
    rgb = np.repeat(backdrop[..., None], 3, axis=-1) * np.array(
        [0.62, 0.70, 0.86], dtype=np.float32
    )

    body = (np.abs(xs) <= 0.20) & (ys >= -0.30) & (ys <= 0.62)
    nose = (ys < -0.30) & (ys >= -0.78) & (np.abs(xs) <= 0.20 * (ys + 0.78) / 0.48)
    fin_left = (xs < -0.16) & (xs >= -0.46) & (ys >= 0.28) & (ys <= 0.62) & (
        (ys - 0.28) / 0.34 >= (-xs - 0.16) / 0.30
    )
    fin_right = fin_left[:, ::-1]
    window = (xs**2 + (ys + 0.02) ** 2) <= 0.085**2

    shade = 0.72 + 0.28 * np.clip(1.0 - np.abs(xs) / 0.22, 0.0, 1.0)

    rgb = np.where((body | nose)[..., None], np.array([0.92, 0.92, 0.94]) * shade[..., None], rgb)
    rgb = np.where(nose[..., None], np.array([0.86, 0.24, 0.20]) * shade[..., None], rgb)
    rgb = np.where((fin_left | fin_right)[..., None], np.array([0.86, 0.24, 0.20]), rgb)
    rgb = np.where(window[..., None], np.array([0.24, 0.62, 0.86]), rgb)

    rgba[..., :3] = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return rgba


SAMPLES = {
    "ball": shaded_ball,
    "star": star_cutout,
    "rocket": toy_rocket,
}


def make_sample(name: str = "ball", size: int = 256) -> np.ndarray:
    """Generate a named sample image as RGBA uint8."""
    if name not in SAMPLES:
        raise ValueError(f"Unknown sample {name!r}. Choose from: {', '.join(sorted(SAMPLES))}")
    return SAMPLES[name](size)
