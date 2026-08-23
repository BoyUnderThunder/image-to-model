"""Isosurface extraction by manifold Surface Nets.

Given a scalar field sampled on a regular 3D grid, this pulls out the surface
where the field crosses zero as a single closed triangle mesh.

Why this rather than the two-sheet mesher: building a front height field and a
back one and sewing their boundaries together forces a seam right around the
silhouette, needs a minimum wall thickness to keep the join manifold, and needs
the relief tapered to the outline so the two sheets can meet at all. Extracting
one surface from a volume has none of those problems -- the surface wraps around
the silhouette because that is simply where the field changes sign.

Surface Nets rather than Marching Cubes: it places vertices per cell rather than
triangles per case, so it needs no 256-entry triangle table, produces far fewer
and better-shaped triangles, and comes out noticeably smoother.

The one thing plain Surface Nets gets wrong is a cell the surface passes through
twice. Giving such a cell a single vertex makes the two sheets share it, and
every edge into it then carries four faces instead of two. That is not a corner
case: 88 of the 256 corner patterns have their inside corners split into more
than one group. So each cell emits one vertex *per connected group of inside
corners*, and a quad takes the vertex belonging to whichever group its own
corner falls in. The result is manifold by construction.
"""

from __future__ import annotations

import numpy as np

from ..types import Mesh

__all__ = ["surface_nets"]

# The eight cell corners, indexed so bit 2 is x, bit 1 is y and bit 0 is z.
_CORNERS = np.array(
    [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)], dtype=np.int64
)

# The twelve cell edges, as pairs of corner indices differing in one axis.
_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]


def _build_component_tables() -> tuple[np.ndarray, np.ndarray]:
    """Group each corner pattern's inside corners into connected components.

    Returns ``(component_of_corner, component_count)``, where
    ``component_of_corner[pattern, corner]`` is that corner's group index (-1
    when it is outside) and ``component_count[pattern]`` is how many vertices
    the cell needs.
    """
    component_of_corner = np.full((256, 8), -1, dtype=np.int8)
    component_count = np.zeros(256, dtype=np.int8)

    for pattern in range(256):
        inside = [bool(pattern >> i & 1) for i in range(8)]
        labels = [-1] * 8
        groups = 0
        for start in range(8):
            if not inside[start] or labels[start] != -1:
                continue
            stack = [start]
            labels[start] = groups
            while stack:
                current = stack.pop()
                for a, b in _EDGES:
                    neighbour = b if a == current else (a if b == current else None)
                    if neighbour is None or not inside[neighbour] or labels[neighbour] != -1:
                        continue
                    labels[neighbour] = groups
                    stack.append(neighbour)
            groups += 1
        component_of_corner[pattern] = labels
        component_count[pattern] = groups

    return component_of_corner, component_count


_COMPONENT_OF_CORNER, _COMPONENT_COUNT = _build_component_tables()


def _corner_views(field: np.ndarray) -> list[np.ndarray]:
    """The field sampled at each of the eight corners of every cell."""
    nx, ny, nz = field.shape
    return [
        field[dx : dx + nx - 1, dy : dy + ny - 1, dz : dz + nz - 1] for dx, dy, dz in _CORNERS
    ]


def surface_nets(
    field: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    level: float = 0.0,
) -> Mesh:
    """Extract the ``field == level`` surface as a closed triangle mesh.

    The field is treated as positive inside the solid. Faces are wound so their
    normals point outwards, away from the positive region.
    """
    values = np.asarray(field, dtype=np.float32) - float(level)
    if values.ndim != 3 or min(values.shape) < 2:
        raise ValueError(f"field must be a 3D grid of at least 2x2x2, got {values.shape}")

    # Guard against a sample sitting exactly on the surface, which would put an
    # edge crossing exactly on a grid corner shared between cells.
    exact_zero = values == 0.0
    if exact_zero.any():
        magnitude = float(np.abs(values).max())
        values = np.where(exact_zero, -1e-6 * max(magnitude, 1e-12), values)

    inside = values > 0.0
    corner_inside = _corner_views(inside)
    corner_values = _corner_views(values)

    pattern = np.zeros(corner_inside[0].shape, dtype=np.uint8)
    for index, corner in enumerate(corner_inside):
        pattern |= corner.astype(np.uint8) << index

    # Only cells the surface actually crosses get vertices. A fully-inside cell
    # has one "component" but no crossings, and allocating for it would leave a
    # vertex per interior cell dangling in the mesh.
    crossed_cell = (pattern != 0) & (pattern != 255)
    per_cell = np.where(crossed_cell, _COMPONENT_COUNT[pattern], 0).astype(np.int64)
    total_vertices = int(per_cell.sum())
    if total_vertices == 0:
        return Mesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3)))

    # Vertices are laid out cell by cell, several per cell where the surface
    # passes through more than once.
    flat_counts = per_cell.reshape(-1)
    offsets = np.zeros_like(flat_counts)
    np.cumsum(flat_counts[:-1], out=offsets[1:])
    offsets = offsets.reshape(per_cell.shape)

    position_sum = np.zeros((total_vertices, 3), dtype=np.float64)
    crossing_count = np.zeros(total_vertices, dtype=np.float64)

    for a, b in _EDGES:
        crossed = corner_inside[a] != corner_inside[b]
        if not crossed.any():
            continue

        # A crossing belongs to whichever of the two endpoints is inside.
        component = np.where(
            corner_inside[a], _COMPONENT_OF_CORNER[pattern, a], _COMPONENT_OF_CORNER[pattern, b]
        )
        vertex_id = (offsets + component)[crossed]

        value_a = corner_values[a][crossed].astype(np.float64)
        value_b = corner_values[b][crossed].astype(np.float64)
        t = np.clip(value_a / (value_a - value_b), 0.0, 1.0)

        cell = np.argwhere(crossed)
        offset_a = _CORNERS[a].astype(np.float64)
        offset_b = _CORNERS[b].astype(np.float64)
        point = cell + offset_a + t[:, None] * (offset_b - offset_a)

        for axis in range(3):
            position_sum[:, axis] += np.bincount(
                vertex_id, weights=point[:, axis], minlength=total_vertices
            )
        crossing_count += np.bincount(vertex_id, minlength=total_vertices)

    local = position_sum / np.maximum(crossing_count, 1.0)[:, None]
    vertices = local * np.asarray(spacing, dtype=np.float64) + np.asarray(
        origin, dtype=np.float64
    )

    faces = _connect(inside, pattern, offsets)
    return Mesh(vertices=vertices.astype(np.float32), faces=faces)


def _connect(inside: np.ndarray, pattern: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Join cell vertices into quads across every sign-changing grid edge.

    Each grid edge whose endpoints disagree is pierced by the surface once, and
    the four cells around it each contribute the vertex for the group their own
    copy of that corner belongs to. One quad per crossed edge, so the surface
    comes out closed and manifold.
    """
    quads: list[np.ndarray] = []

    for axis in range(3):
        length = inside.shape[axis]
        low = np.take(inside, np.arange(length - 1), axis=axis)
        high = np.take(inside, np.arange(1, length), axis=axis)
        changed = low != high

        other = [i for i in range(3) if i != axis]
        interior = [slice(None)] * 3
        interior[other[0]] = slice(1, -1)
        interior[other[1]] = slice(1, -1)

        selected = np.argwhere(changed[tuple(interior)])
        if not len(selected):
            continue

        # Undo the interior slicing to recover the edge's grid coordinates.
        point = np.zeros((len(selected), 3), dtype=np.int64)
        point[:, axis] = selected[:, axis]
        point[:, other[0]] = selected[:, other[0]] + 1
        point[:, other[1]] = selected[:, other[1]] + 1

        low_inside = low[tuple(interior)][selected[:, 0], selected[:, 1], selected[:, 2]]
        # Offset, within each cell, of whichever endpoint lies inside.
        inside_offset = np.where(low_inside, 0, 1)

        corner_ids = []
        for da, db in ((0, 0), (0, 1), (1, 1), (1, 0)):
            cell = point.copy()
            cell[:, other[0]] -= da
            cell[:, other[1]] -= db

            delta = np.zeros((len(selected), 3), dtype=np.int64)
            delta[:, axis] = inside_offset
            delta[:, other[0]] = da
            delta[:, other[1]] = db
            corner = delta[:, 0] * 4 + delta[:, 1] * 2 + delta[:, 2]

            cell_pattern = pattern[cell[:, 0], cell[:, 1], cell[:, 2]]
            component = _COMPONENT_OF_CORNER[cell_pattern, corner]
            corner_ids.append(offsets[cell[:, 0], cell[:, 1], cell[:, 2]] + component)
        corner_ids = np.stack(corner_ids, axis=1)

        # Orientation: the outward normal runs from the inside endpoint towards
        # the outside one. The corner order above traverses other[1] before
        # other[0], so its natural normal is -(e_other0 x e_other1) -- which
        # already points along +axis for y but against it for x and z. Ignoring
        # that makes the winding disagree between axes, and the surface then
        # encloses no consistent volume.
        natural_faces_positive = axis == 1
        keep_order = low_inside == natural_faces_positive
        oriented = np.where(keep_order[:, None], corner_ids, corner_ids[:, ::-1])
        quads.append(oriented)

    if not quads:
        return np.zeros((0, 3), dtype=np.int32)

    quad = np.concatenate(quads, axis=0)
    triangles = np.concatenate([quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]], axis=0)
    return triangles.astype(np.int32)
