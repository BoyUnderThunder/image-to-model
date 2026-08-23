"""Binary morphology and connected components for 2D masks.

SciPy is used when installed; otherwise every operation falls back to a
vectorised NumPy implementation so the package keeps working with only numpy
and pillow available.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "binary_erode",
    "binary_dilate",
    "binary_open",
    "binary_close",
    "fill_holes",
    "largest_component",
    "label_components",
    "otsu_threshold",
]


def _has_scipy() -> bool:
    try:
        import scipy.ndimage  # noqa: F401

        return True
    except ImportError:
        return False


def _window_reduce(mask: np.ndarray, radius: int, mode: str) -> np.ndarray:
    """Min/max over a square window, as a separable two-pass reduction.

    Doing rows then columns keeps this O(n * k) instead of O(n * k^2).
    """
    if radius <= 0:
        return mask
    size = 2 * radius + 1
    reduce_fn = np.min if mode == "erode" else np.max
    pad_value = True if mode == "erode" else False

    out = mask
    for axis in (0, 1):
        pad_width = [(0, 0), (0, 0)]
        pad_width[axis] = (radius, radius)
        padded = np.pad(out, pad_width, mode="constant", constant_values=pad_value)
        windows = np.lib.stride_tricks.sliding_window_view(padded, size, axis=axis)
        out = reduce_fn(windows, axis=-1)
    return out


def binary_erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Shrink a boolean mask by ``radius`` pixels."""
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask
    if _has_scipy():
        from scipy.ndimage import binary_erosion

        size = 2 * radius + 1
        return binary_erosion(mask, structure=np.ones((size, size), dtype=bool), border_value=1)
    return _window_reduce(mask, radius, "erode")


def binary_dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Grow a boolean mask by ``radius`` pixels."""
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask
    if _has_scipy():
        from scipy.ndimage import binary_dilation

        size = 2 * radius + 1
        return binary_dilation(mask, structure=np.ones((size, size), dtype=bool))
    return _window_reduce(mask, radius, "dilate")


def binary_open(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Erode then dilate: removes speckle without shrinking the body."""
    return binary_dilate(binary_erode(mask, radius), radius)


def binary_close(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Dilate then erode: seals small gaps without growing the body."""
    return binary_erode(binary_dilate(mask, radius), radius)


def _propagate_labels(mask: np.ndarray, seeds: np.ndarray, max_iterations: int = 4096) -> np.ndarray:
    """Spread seed values to 4-connected neighbours until nothing changes.

    Each pass costs a handful of array shifts, and the number of passes is
    bounded by the longest path through the region, so blob-shaped masks
    converge in a few dozen iterations.
    """
    current = np.where(mask, seeds, 0)
    for _ in range(max_iterations):
        neighbour_max = current.copy()
        neighbour_max[1:, :] = np.maximum(neighbour_max[1:, :], current[:-1, :])
        neighbour_max[:-1, :] = np.maximum(neighbour_max[:-1, :], current[1:, :])
        neighbour_max[:, 1:] = np.maximum(neighbour_max[:, 1:], current[:, :-1])
        neighbour_max[:, :-1] = np.maximum(neighbour_max[:, :-1], current[:, 1:])
        neighbour_max = np.where(mask, neighbour_max, 0)
        if np.array_equal(neighbour_max, current):
            break
        current = neighbour_max
    return current


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label 4-connected regions of a boolean mask, background as 0."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32), 0

    if _has_scipy():
        from scipy.ndimage import label

        labels, count = label(mask)
        return labels.astype(np.int32), int(count)

    seeds = np.arange(1, mask.size + 1, dtype=np.int64).reshape(mask.shape)
    spread = _propagate_labels(mask, seeds)
    unique = np.unique(spread[spread > 0])
    remap = np.zeros(int(spread.max()) + 1, dtype=np.int32)
    remap[unique] = np.arange(1, len(unique) + 1, dtype=np.int32)
    return remap[spread], len(unique)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the biggest connected region."""
    mask = np.asarray(mask, dtype=bool)
    labels, count = label_components(mask)
    if count <= 1:
        return mask
    sizes = np.bincount(labels.reshape(-1))
    sizes[0] = 0  # ignore background
    return labels == int(np.argmax(sizes))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes, keeping regions that touch the image border open.

    Background is flooded inward from the border; whatever the flood cannot
    reach is enclosed by the subject and gets filled.
    """
    mask = np.asarray(mask, dtype=bool)
    if _has_scipy():
        from scipy.ndimage import binary_fill_holes

        return np.asarray(binary_fill_holes(mask), dtype=bool)

    background = ~mask
    border_seed = np.zeros(mask.shape, dtype=np.int64)
    border_seed[0, :] = 1
    border_seed[-1, :] = 1
    border_seed[:, 0] = 1
    border_seed[:, -1] = 1
    border_seed = np.where(background, border_seed, 0)
    outside = _propagate_labels(background, border_seed) > 0
    return ~outside


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's method: the threshold maximising between-class variance."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.5
    low, high = float(flat.min()), float(flat.max())
    if high - low < 1e-12:
        return low

    histogram, edges = np.histogram(flat, bins=bins, range=(low, high))
    histogram = histogram.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) * 0.5

    weight_bg = np.cumsum(histogram)
    weight_fg = weight_bg[-1] - weight_bg
    valid = (weight_bg > 0) & (weight_fg > 0)
    if not valid.any():
        return float(np.median(flat))

    cumulative_mean = np.cumsum(histogram * centers)
    mean_bg = np.divide(cumulative_mean, weight_bg, out=np.zeros_like(weight_bg), where=weight_bg > 0)
    mean_fg = np.divide(
        cumulative_mean[-1] - cumulative_mean,
        weight_fg,
        out=np.zeros_like(weight_fg),
        where=weight_fg > 0,
    )
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between[~valid] = -1.0
    return float(centers[int(np.argmax(between))])
