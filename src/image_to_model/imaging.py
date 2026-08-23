"""Image loading and small array utilities.

Only NumPy and Pillow are required. SciPy is used when present for a faster
Gaussian blur, but a pure-NumPy separable convolution stands in for it so the
package works with no optional dependencies at all.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .errors import InputError

__all__ = [
    "load_image",
    "save_image",
    "resize_longest",
    "resize_array",
    "gaussian_blur",
    "joint_bilateral_filter",
    "bilinear_sample",
    "to_float",
    "to_uint8",
    "SUPPORTED_INPUT_FORMATS",
]

SUPPORTED_INPUT_FORMATS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

# Guard against decompression-bomb style inputs while still allowing large photos.
Image.MAX_IMAGE_PIXELS = 200_000_000


def to_float(array: np.ndarray) -> np.ndarray:
    """uint8 [0,255] to float32 [0,1]; already-float input passes through."""
    arr = np.asarray(array)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


def to_uint8(array: np.ndarray) -> np.ndarray:
    """float [0,1] to uint8 [0,255] with clipping."""
    arr = np.asarray(array)
    if arr.dtype == np.uint8:
        return arr
    return np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)


def load_image(path: str | os.PathLike) -> np.ndarray:
    """Load an image as an RGBA uint8 array with EXIF rotation applied.

    Phones record orientation in EXIF rather than rotating pixels, so skipping
    the transpose silently produces sideways models.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise InputError(f"Image not found: {file_path}")
    if not file_path.is_file():
        raise InputError(f"Not a file: {file_path}")

    try:
        with Image.open(file_path) as img:
            img = ImageOps.exif_transpose(img)
            has_alpha = img.mode in ("RGBA", "LA") or (
                img.mode == "P" and "transparency" in img.info
            )
            rgba = img.convert("RGBA")
            array = np.array(rgba, dtype=np.uint8)
    except InputError:
        raise
    except Exception as exc:  # Pillow raises a wide variety of decode errors
        raise InputError(f"Could not read image {file_path}: {exc}") from exc

    if array.ndim != 3 or array.shape[2] != 4:
        raise InputError(f"Unexpected image shape {array.shape} from {file_path}")
    if not has_alpha:
        # convert() fabricates an opaque alpha channel; record that it carries
        # no real matte so segmentation does not trust it.
        array[..., 3] = 255
    return array


def save_image(array: np.ndarray, path: str | os.PathLike) -> Path:
    """Write an RGB/RGBA/greyscale array to disk, creating parent directories."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        arr = to_uint8(arr if arr.max() <= 1.0 else arr / max(float(arr.max()), 1e-8))
    if arr.ndim == 2:
        image = Image.fromarray(arr, mode="L")
    elif arr.shape[2] == 3:
        image = Image.fromarray(arr, mode="RGB")
    elif arr.shape[2] == 4:
        image = Image.fromarray(arr, mode="RGBA")
    else:
        raise InputError(f"Cannot save array with shape {arr.shape}")
    image.save(out_path)
    return out_path


def resize_longest(array: np.ndarray, max_size: int) -> np.ndarray:
    """Scale so the longest edge equals ``max_size``, preserving aspect ratio.

    Images smaller than the target are left alone: upsampling adds pixels but
    no detail, and only makes the mesh heavier.
    """
    height, width = array.shape[:2]
    longest = max(height, width)
    if longest <= max_size or longest == 0:
        return array
    scale = max_size / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return resize_array(array, new_size)


#: Resampling filters. Lanczos is sharpest for images, but it overshoots near
#: edges; that ringing is harmless in a photo and destructive in a scalar field,
#: where a sub-percent overshoot near a zero crossing flips the sign.
_RESAMPLE_FILTERS = {
    "lanczos": Image.Resampling.LANCZOS,
    "bilinear": Image.Resampling.BILINEAR,
    "area": Image.Resampling.BOX,
    "nearest": Image.Resampling.NEAREST,
}


def resize_array(
    array: np.ndarray,
    size: tuple[int, int],
    nearest: bool = False,
    mode: str = "lanczos",
) -> np.ndarray:
    """Resize an array to ``(width, height)`` using Pillow.

    ``mode`` picks the filter; use ``area`` when downsampling a field whose
    values are about to be compared against a threshold, since it cannot
    overshoot the input range.
    """
    arr = np.asarray(array)
    if nearest:
        mode = "nearest"
    if mode not in _RESAMPLE_FILTERS:
        raise ValueError(f"Unknown resize mode {mode!r}")
    resample = _RESAMPLE_FILTERS[mode]
    if arr.dtype == np.uint8:
        return np.array(Image.fromarray(arr).resize(size, resample), dtype=np.uint8)

    # Pillow handles float only in single-channel "F" mode, so resize per channel.
    float_arr = arr.astype(np.float32)
    if float_arr.ndim == 2:
        resized = Image.fromarray(float_arr, mode="F").resize(size, resample)
        return np.array(resized, dtype=np.float32)
    channels = [
        np.array(Image.fromarray(float_arr[..., c], mode="F").resize(size, resample))
        for c in range(float_arr.shape[2])
    ]
    return np.stack(channels, axis=-1).astype(np.float32)


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x**2) / (2.0 * sigma * sigma))
    return (kernel / kernel.sum()).astype(np.float32)


def _convolve1d(array: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = len(kernel) // 2
    moved = np.moveaxis(array, axis, -1)
    pad_width = [(0, 0)] * (moved.ndim - 1) + [(pad, pad)]
    padded = np.pad(moved, pad_width, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, len(kernel), axis=-1)
    result = np.einsum("...k,k->...", windows, kernel, optimize=True)
    return np.moveaxis(result, -1, axis)


def gaussian_blur(array: np.ndarray, sigma: float) -> np.ndarray:
    """Blur a 2D array, or each channel of a 3D array, with a Gaussian."""
    if sigma <= 0:
        return np.asarray(array, dtype=np.float32)
    arr = np.asarray(array, dtype=np.float32)

    try:
        from scipy.ndimage import gaussian_filter

        axes_sigma = (sigma, sigma) if arr.ndim == 2 else (sigma, sigma, 0.0)
        return gaussian_filter(arr, sigma=axes_sigma, mode="nearest").astype(np.float32)
    except ImportError:
        pass

    kernel = _gaussian_kernel(sigma)
    blurred = _convolve1d(arr, kernel, axis=0)
    return _convolve1d(blurred, kernel, axis=1).astype(np.float32)


def bilinear_sample(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Sample ``image`` at fractional pixel coordinates.

    ``xs``/``ys`` are in pixel units and are clamped to the image, which keeps
    edge samples valid instead of wrapping to the far side.
    """
    arr = np.asarray(image, dtype=np.float32)
    height, width = arr.shape[:2]
    xs = np.clip(np.asarray(xs, dtype=np.float32), 0.0, width - 1.0)
    ys = np.clip(np.asarray(ys, dtype=np.float32), 0.0, height - 1.0)

    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)

    wx = (xs - x0)[..., None] if arr.ndim == 3 else (xs - x0)
    wy = (ys - y0)[..., None] if arr.ndim == 3 else (ys - y0)

    top = arr[y0, x0] * (1.0 - wx) + arr[y0, x1] * wx
    bottom = arr[y1, x0] * (1.0 - wx) + arr[y1, x1] * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32)


def joint_bilateral_filter(
    values: np.ndarray,
    guide: np.ndarray,
    sigma_spatial: float,
    sigma_range: float = 0.1,
) -> np.ndarray:
    """Smooth ``values`` while keeping the edges present in ``guide``.

    A neighbour contributes in proportion to both how near it is and how
    similar the *guide* looks there, so smoothing stops at boundaries the guide
    shows. Filtering a depth map guided by the photograph therefore removes
    per-pixel depth noise without rounding off the object's real edges, which
    is exactly what a plain Gaussian cannot do: it blurs across a depth
    discontinuity just as happily as along a flat surface.

    ``guide`` is expected in roughly [0, 1]; ``sigma_range`` is measured in
    those same units.
    """
    values = np.asarray(values, dtype=np.float32)
    guide = np.asarray(guide, dtype=np.float32)
    if sigma_spatial <= 0:
        return values
    if guide.shape != values.shape:
        raise ValueError(f"guide shape {guide.shape} does not match values shape {values.shape}")

    radius = max(1, int(round(2.0 * sigma_spatial)))
    padded_values = np.pad(values, radius, mode="edge")
    padded_guide = np.pad(guide, radius, mode="edge")
    height, width = values.shape

    accumulator = np.zeros_like(values)
    weight_total = np.zeros_like(values)
    range_denominator = 2.0 * max(sigma_range, 1e-6) ** 2
    spatial_denominator = 2.0 * sigma_spatial**2

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            spatial = np.exp(-(dx * dx + dy * dy) / spatial_denominator)
            if spatial < 1e-4:
                continue
            y0, x0 = dy + radius, dx + radius
            shifted_values = padded_values[y0 : y0 + height, x0 : x0 + width]
            shifted_guide = padded_guide[y0 : y0 + height, x0 : x0 + width]

            difference = guide - shifted_guide
            weight = spatial * np.exp(-(difference * difference) / range_denominator)
            accumulator += weight * shifted_values
            weight_total += weight

    return (accumulator / np.maximum(weight_total, 1e-8)).astype(np.float32)
