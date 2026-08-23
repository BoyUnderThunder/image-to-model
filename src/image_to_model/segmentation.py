"""Separating the subject from the background.

Three strategies, tried in this order by ``auto``:

``alpha``
    The image already carries a real alpha channel, so trust it. This is the
    common case for renders and cut-out PNGs.
``rembg``
    A learned matting model, if the optional ``rembg`` package is installed.
    Much better on cluttered backgrounds.
``border``
    A classical fallback with no dependencies: model the background from the
    image border and score every pixel against it. Works well for the studio-ish
    product photos this pipeline is usually pointed at, and degrades on busy
    scenes.
"""

from __future__ import annotations

import numpy as np

from .errors import SegmentationError
from .imaging import gaussian_blur
from .logging import get_logger
from .morphology import (
    binary_close,
    binary_open,
    fill_holes,
    largest_component,
    otsu_threshold,
)

__all__ = ["segment", "has_real_alpha", "available_methods", "SEGMENTATION_METHODS"]

log = get_logger("segmentation")

SEGMENTATION_METHODS = ("auto", "alpha", "rembg", "border", "none")

# Below/above these coverage fractions the mask is treated as a failed
# segmentation rather than a very small or very large subject.
_MIN_COVERAGE = 0.005
_MAX_COVERAGE = 0.98


def has_real_alpha(image_rgba: np.ndarray) -> bool:
    """True when the alpha channel actually carries transparency."""
    if image_rgba.ndim != 3 or image_rgba.shape[2] < 4:
        return False
    return bool((image_rgba[..., 3] < 250).any())


def _rembg_available() -> bool:
    try:
        import rembg  # noqa: F401

        return True
    except Exception:
        return False


def available_methods() -> list[str]:
    """Segmentation methods usable in this environment."""
    methods = ["border", "none", "alpha"]
    if _rembg_available():
        methods.append("rembg")
    return sorted(methods)


def _border_mask(shape: tuple[int, int], fraction: float = 0.04) -> np.ndarray:
    height, width = shape
    band = max(2, int(round(fraction * min(height, width))))
    mask = np.zeros(shape, dtype=bool)
    mask[:band, :] = True
    mask[-band:, :] = True
    mask[:, :band] = True
    mask[:, -band:] = True
    return mask


def _background_distance(rgb: np.ndarray, band: int) -> np.ndarray:
    """Colour distance from each pixel to the nearest background reference.

    Three references are compared and the smallest distance wins, because any
    one of them being a good match is enough to call a pixel background:

    * the median colour of that pixel's own row taken from the left/right
      border bands, which tracks a vertical gradient exactly;
    * the same for its column from the top/bottom bands, for a horizontal one;
    * the median of the whole border, which covers a flat backdrop.
    """
    height, width = rgb.shape[:2]
    values = rgb.astype(np.float32)

    row_ref = np.median(
        np.concatenate([values[:, :band, :], values[:, -band:, :]], axis=1), axis=1
    )  # (H, 3)
    col_ref = np.median(
        np.concatenate([values[:band, :, :], values[-band:, :, :]], axis=0), axis=0
    )  # (W, 3)
    global_ref = np.median(row_ref, axis=0)  # (3,)

    distance_row = np.linalg.norm(values - row_ref[:, None, :], axis=-1)
    distance_col = np.linalg.norm(values - col_ref[None, :, :], axis=-1)
    distance_global = np.linalg.norm(values - global_ref[None, None, :], axis=-1)

    return np.minimum(np.minimum(distance_row, distance_col), distance_global).astype(np.float32)


def _segment_border(rgb: np.ndarray, center_bias: float = 0.35) -> np.ndarray:
    """Score pixels by how far their colour sits from the background model.

    A centre prior breaks ties in favour of the thing the photographer was
    pointing at, and the border band itself is forced to background, which
    anchors the threshold when the subject fills most of the frame.
    """
    height, width = rgb.shape[:2]
    band = max(2, int(round(0.04 * min(height, width))))

    score = _background_distance(rgb, band)
    # Compress the tail so one saturated colour cannot dominate the threshold.
    high = float(np.percentile(score, 99.0))
    if high > 1e-6:
        score = np.clip(score / high, 0.0, 1.0)

    if center_bias > 0:
        ys = (np.arange(height, dtype=np.float32) - height / 2.0) / (height / 2.0)
        xs = (np.arange(width, dtype=np.float32) - width / 2.0) / (width / 2.0)
        radius_sq = ys[:, None] ** 2 + xs[None, :] ** 2
        prior = np.exp(-radius_sq / 0.75)
        score = score + center_bias * (prior - 0.5)

    score = gaussian_blur(score.astype(np.float32), 1.5)
    foreground = score >= otsu_threshold(score)

    # The frame edge is background in essentially every subject photo, and
    # saying so keeps a subject that runs to the edge from swallowing the frame.
    foreground[_border_mask((height, width), fraction=0.02)] = False
    return foreground


def _segment_rembg(image_rgba: np.ndarray) -> np.ndarray:
    from PIL import Image
    from rembg import remove

    source = Image.fromarray(image_rgba, mode="RGBA")
    result = remove(source)
    alpha = np.array(result.convert("RGBA"), dtype=np.uint8)[..., 3]
    return alpha.astype(np.float32) / 255.0


def _refine(binary: np.ndarray, feather: float, close_radius: int, open_radius: int) -> np.ndarray:
    """Tidy a raw binary mask into a soft, single-blob alpha."""
    refined = binary_close(binary, close_radius)
    refined = binary_open(refined, open_radius)
    refined = fill_holes(refined)
    refined = largest_component(refined)
    soft = refined.astype(np.float32)
    if feather > 0:
        soft = np.clip(gaussian_blur(soft, feather), 0.0, 1.0)
    return soft


def segment(
    image_rgba: np.ndarray,
    method: str = "auto",
    feather: float = 1.0,
    close_radius: int = 3,
    open_radius: int = 2,
    center_bias: float = 0.35,
) -> tuple[np.ndarray, str]:
    """Produce a soft foreground mask.

    Returns the mask in [0, 1] together with the name of the method that
    actually ran, which may differ from ``method`` when ``auto`` is used or a
    strategy falls back.
    """
    if method not in SEGMENTATION_METHODS:
        raise SegmentationError(
            f"Unknown segmentation method {method!r}. "
            f"Choose from: {', '.join(SEGMENTATION_METHODS)}"
        )

    image = np.asarray(image_rgba, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise SegmentationError(f"Expected an RGB or RGBA image, got shape {image.shape}")
    rgb = image[..., :3]
    height, width = rgb.shape[:2]

    if method == "none":
        return np.ones((height, width), dtype=np.float32), "none"

    resolved = method
    if method == "auto":
        if has_real_alpha(image):
            resolved = "alpha"
        elif _rembg_available():
            resolved = "rembg"
        else:
            resolved = "border"
        log.debug("auto-selected segmentation method: %s", resolved)

    if resolved == "alpha":
        if image.shape[2] < 4 or not has_real_alpha(image):
            log.warning("No usable alpha channel; falling back to border segmentation")
            resolved = "border"
        else:
            soft = image[..., 3].astype(np.float32) / 255.0
            return _refine(soft > 0.5, feather, close_radius, open_radius), "alpha"

    if resolved == "rembg":
        try:
            soft = _segment_rembg(image if image.shape[2] == 4 else np.dstack([rgb, np.full((height, width), 255, np.uint8)]))
            mask = _refine(soft > 0.5, feather, close_radius, open_radius)
            if _MIN_COVERAGE < float(mask.mean()) < _MAX_COVERAGE:
                return mask, "rembg"
            log.warning("rembg returned an implausible mask; falling back to border")
        except Exception as exc:
            log.warning("rembg failed (%s); falling back to border segmentation", exc)
        resolved = "border"

    binary = _segment_border(rgb, center_bias=center_bias)
    mask = _refine(binary, feather, close_radius, open_radius)

    coverage = float(mask.mean())
    if not _MIN_COVERAGE < coverage < _MAX_COVERAGE:
        # Nothing separated cleanly. Meshing the whole frame is a more useful
        # outcome than raising, and the caller is told what happened.
        log.warning(
            "Border segmentation found no clear subject (coverage %.1f%%); "
            "using the whole image. Pass an image with a plainer background, "
            "a cut-out PNG, or install rembg for learned matting.",
            coverage * 100.0,
        )
        return np.ones((height, width), dtype=np.float32), "none"

    return mask, "border"
