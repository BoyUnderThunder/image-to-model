"""Core data types shared by every stage of the pipeline.

Everything is plain NumPy so the types stay cheap to construct, easy to test
and free of any dependency on a mesh library.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

__all__ = ["Mesh", "DepthMap", "Subject", "CameraIntrinsics"]


def _as_array(value: Any, dtype, ndim: int, width: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.size == 0:
        return arr.reshape((0, width))
    if arr.ndim != ndim or arr.shape[1] != width:
        raise ValueError(f"{name} must have shape (N, {width}), got {arr.shape}")
    return np.ascontiguousarray(arr)


@dataclass
class CameraIntrinsics:
    """Pinhole camera parameters used to lift a depth map into 3D.

    ``fx``/``fy`` are in pixels; ``cx``/``cy`` are the principal point. The
    default construction from a field of view is what the pipeline uses, since
    a single photo carries no reliable calibration.
    """

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov(cls, width: int, height: int, fov_degrees: float = 55.0) -> CameraIntrinsics:
        """Build intrinsics for an image of ``width``x``height`` with a horizontal FOV."""
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        fov = np.radians(float(np.clip(fov_degrees, 1.0, 179.0)))
        fx = (width * 0.5) / np.tan(fov * 0.5)
        return cls(fx=float(fx), fy=float(fx), cx=width * 0.5, cy=height * 0.5)

    def as_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )


@dataclass
class DepthMap:
    """A dense per-pixel depth prediction.

    ``depth`` holds *relative* depth in arbitrary units where larger means
    further from the camera. Monocular estimators only recover depth up to an
    unknown scale and offset, so the pipeline treats these values as ordinal
    and fixes the physical scale later.
    """

    depth: np.ndarray
    mask: np.ndarray | None = None
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.depth = np.ascontiguousarray(np.asarray(self.depth, dtype=np.float32))
        if self.depth.ndim != 2:
            raise ValueError(f"depth must be a 2D array, got shape {self.depth.shape}")
        if self.mask is not None:
            mask = np.asarray(self.mask, dtype=np.float32)
            if mask.shape != self.depth.shape:
                raise ValueError(
                    f"mask shape {mask.shape} does not match depth shape {self.depth.shape}"
                )
            self.mask = np.ascontiguousarray(np.clip(mask, 0.0, 1.0))

    @property
    def shape(self) -> tuple[int, int]:
        return self.depth.shape  # type: ignore[return-value]

    def valid_values(self) -> np.ndarray:
        """Depth samples inside the subject mask (all samples when unmasked)."""
        if self.mask is None:
            return self.depth.reshape(-1)
        selected = self.depth[self.mask > 0.5]
        return selected if selected.size else self.depth.reshape(-1)

    def normalized(self) -> DepthMap:
        """Rescale depth to [0, 1] using robust percentiles of the masked region.

        Percentiles rather than min/max keep a few speckled outliers from
        flattening the entire subject into a narrow band.
        """
        values = self.valid_values()
        lo, hi = np.percentile(values, [2.0, 98.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
            lo, hi = float(np.min(self.depth)), float(np.max(self.depth))
        if hi - lo < 1e-8:
            normalized = np.zeros_like(self.depth)
        else:
            normalized = np.clip((self.depth - lo) / (hi - lo), 0.0, 1.0)
        return replace(self, depth=normalized.astype(np.float32))

    def inverted(self) -> DepthMap:
        """Flip the near/far convention."""
        return replace(self, depth=(float(self.depth.max()) - self.depth).astype(np.float32))


@dataclass
class Subject:
    """A photograph with its foreground isolated.

    ``image`` is RGB uint8 and ``mask`` is a soft alpha in [0, 1] where 1 marks
    the object being reconstructed.
    """

    image: np.ndarray
    mask: np.ndarray
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.image = np.ascontiguousarray(np.asarray(self.image, dtype=np.uint8))
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError(f"image must have shape (H, W, 3), got {self.image.shape}")
        mask = np.asarray(self.mask, dtype=np.float32)
        if mask.shape != self.image.shape[:2]:
            raise ValueError(
                f"mask shape {mask.shape} does not match image shape {self.image.shape[:2]}"
            )
        self.mask = np.ascontiguousarray(np.clip(mask, 0.0, 1.0))

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def coverage(self) -> float:
        """Fraction of the frame occupied by the subject."""
        return float(self.mask.mean())

    def bounding_box(self) -> tuple[int, int, int, int]:
        """Pixel bounds of the subject as ``(x0, y0, x1, y1)``, end-exclusive."""
        rows = np.any(self.mask > 0.5, axis=1)
        cols = np.any(self.mask > 0.5, axis=0)
        if not rows.any() or not cols.any():
            return 0, 0, self.width, self.height
        y0, y1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
        x0, x1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
        return x0, y0, x1, y1


@dataclass
class Mesh:
    """An indexed triangle mesh with optional per-vertex colour, normals and UVs."""

    vertices: np.ndarray
    faces: np.ndarray
    vertex_colors: np.ndarray | None = None
    vertex_normals: np.ndarray | None = None
    uvs: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vertices = _as_array(self.vertices, np.float32, 2, 3, "vertices")
        self.faces = _as_array(self.faces, np.int32, 2, 3, "faces")
        if self.faces.size and self.vertices.size:
            max_index = int(self.faces.max())
            if max_index >= len(self.vertices):
                raise ValueError(
                    f"face index {max_index} is out of range for {len(self.vertices)} vertices"
                )
        if self.vertex_colors is not None:
            self.vertex_colors = _as_array(self.vertex_colors, np.uint8, 2, 3, "vertex_colors")
            if len(self.vertex_colors) != len(self.vertices):
                raise ValueError("vertex_colors must have one entry per vertex")
        if self.vertex_normals is not None:
            self.vertex_normals = _as_array(self.vertex_normals, np.float32, 2, 3, "vertex_normals")
            if len(self.vertex_normals) != len(self.vertices):
                raise ValueError("vertex_normals must have one entry per vertex")
        if self.uvs is not None:
            self.uvs = _as_array(self.uvs, np.float32, 2, 2, "uvs")
            if len(self.uvs) != len(self.vertices):
                raise ValueError("uvs must have one entry per vertex")

    # -- basic properties -------------------------------------------------

    @property
    def n_vertices(self) -> int:
        return int(len(self.vertices))

    @property
    def n_faces(self) -> int:
        return int(len(self.faces))

    @property
    def is_empty(self) -> bool:
        return self.n_vertices == 0 or self.n_faces == 0

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned ``(min, max)`` corners."""
        if self.is_empty:
            zeros = np.zeros(3, dtype=np.float32)
            return zeros, zeros.copy()
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def extent(self) -> np.ndarray:
        lo, hi = self.bounds()
        return hi - lo

    def centroid(self) -> np.ndarray:
        """Centre of the bounding box, which is more stable than the vertex mean
        when vertex density is uneven across the surface."""
        lo, hi = self.bounds()
        return (lo + hi) * 0.5

    def copy(self) -> Mesh:
        return Mesh(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            vertex_colors=None if self.vertex_colors is None else self.vertex_colors.copy(),
            vertex_normals=None if self.vertex_normals is None else self.vertex_normals.copy(),
            uvs=None if self.uvs is None else self.uvs.copy(),
            metadata=dict(self.metadata),
        )

    # -- geometry ---------------------------------------------------------

    def face_normals(self, normalize: bool = True) -> np.ndarray:
        """Per-face normals; unnormalized values have length 2x the triangle area."""
        if self.is_empty:
            return np.zeros((0, 3), dtype=np.float32)
        tri = self.vertices[self.faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        if normalize:
            lengths = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 1e-12)
        return normals.astype(np.float32)

    def face_areas(self) -> np.ndarray:
        return (np.linalg.norm(self.face_normals(normalize=False), axis=1) * 0.5).astype(np.float32)

    def surface_area(self) -> float:
        return float(self.face_areas().sum())

    def volume(self) -> float:
        """Signed volume via the divergence theorem.

        Only meaningful for a closed, consistently wound surface; the sign
        reveals inverted winding.
        """
        if self.is_empty:
            return 0.0
        tri = self.vertices.astype(np.float64)[self.faces]
        return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)

    def edges(self) -> np.ndarray:
        """Every directed edge of every face, shape ``(3F, 2)``."""
        f = self.faces
        return np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)

    def is_watertight(self) -> bool:
        """True when every undirected edge is shared by exactly two faces."""
        if self.is_empty:
            return False
        undirected = np.sort(self.edges(), axis=1)
        _, counts = np.unique(undirected, axis=0, return_counts=True)
        return bool(np.all(counts == 2))

    def is_closed_surface(self, tolerance: float = 1e-5) -> bool:
        """True when the surface is closed once coincident vertices are merged.

        :meth:`is_watertight` works on indices, so it reports False for a mesh
        carrying UV seams: splitting a vertex so each side can have its own
        texture coordinate breaks the index-level join even though the two
        copies sit in exactly the same place and the solid is still closed.
        Every textured asset has such seams, so this is the check that matches
        what "is it a solid" actually means.
        """
        if self.is_empty:
            return False
        from .geometry.mesh_ops import weld_vertices

        return weld_vertices(self, tolerance).is_watertight()

    def euler_characteristic(self) -> int:
        """V - E + F. A closed genus-0 surface gives 2."""
        if self.is_empty:
            return 0
        unique_edges = np.unique(np.sort(self.edges(), axis=1), axis=0)
        return self.n_vertices - len(unique_edges) + self.n_faces

    # -- transforms -------------------------------------------------------

    def translated(self, offset: np.ndarray) -> Mesh:
        out = self.copy()
        out.vertices = (out.vertices + np.asarray(offset, dtype=np.float32)).astype(np.float32)
        return out

    def scaled(self, factor: float) -> Mesh:
        out = self.copy()
        out.vertices = (out.vertices * float(factor)).astype(np.float32)
        return out

    def centered(self) -> Mesh:
        return self.translated(-self.centroid())

    def normalized(self, target_size: float = 1.0) -> Mesh:
        """Centre at the origin and scale so the longest axis spans ``target_size``."""
        out = self.centered()
        longest = float(out.extent().max())
        if longest > 1e-9:
            out = out.scaled(target_size / longest)
        return out

    def with_normals(self) -> Mesh:
        """Return a copy carrying area-weighted smooth vertex normals."""
        from .geometry.mesh_ops import compute_vertex_normals

        out = self.copy()
        out.vertex_normals = compute_vertex_normals(out.vertices, out.faces)
        return out

    def flipped_winding(self) -> Mesh:
        out = self.copy()
        out.faces = np.ascontiguousarray(out.faces[:, ::-1])
        if out.vertex_normals is not None:
            out.vertex_normals = -out.vertex_normals
        return out

    # -- reporting --------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """A compact summary used by the CLI and by tests."""
        lo, hi = self.bounds()
        return {
            "vertices": self.n_vertices,
            "faces": self.n_faces,
            "watertight": self.is_watertight(),
            "closed": self.is_closed_surface(),
            "euler_characteristic": self.euler_characteristic(),
            "surface_area": round(self.surface_area(), 6),
            "volume": round(self.volume(), 6),
            "bounds_min": [round(float(v), 6) for v in lo],
            "bounds_max": [round(float(v), 6) for v in hi],
            "has_vertex_colors": self.vertex_colors is not None,
            "has_normals": self.vertex_normals is not None,
            "has_uvs": self.uvs is not None,
        }

    def __repr__(self) -> str:
        return f"Mesh(vertices={self.n_vertices}, faces={self.n_faces})"
