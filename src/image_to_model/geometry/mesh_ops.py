"""Mesh cleanup and refinement operations.

Everything here is vectorised NumPy and works on the plain arrays inside
:class:`~image_to_model.types.Mesh`.
"""

from __future__ import annotations

import numpy as np

from ..types import Mesh

__all__ = [
    "compute_vertex_normals",
    "remove_degenerate_faces",
    "remove_unreferenced_vertices",
    "weld_vertices",
    "boundary_vertices",
    "vertex_adjacency",
    "connected_components",
    "keep_largest_components",
    "taubin_smooth",
    "clean_mesh",
]


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted smooth vertex normals.

    Face normals are accumulated unnormalized so their magnitude (twice the
    triangle area) weights each contribution, which keeps large triangles from
    being outvoted by clusters of slivers.
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    normals = np.zeros_like(vertices)
    if len(faces) == 0 or len(vertices) == 0:
        return normals

    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])

    for corner in range(3):
        for axis in range(3):
            normals[:, axis] += np.bincount(
                faces[:, corner], weights=face_normals[:, axis], minlength=len(vertices)
            )

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 1e-12)
    return normals.astype(np.float32)


def remove_degenerate_faces(mesh: Mesh, area_epsilon: float = 1e-12) -> Mesh:
    """Drop triangles with a repeated index or effectively zero area."""
    if mesh.is_empty:
        return mesh
    faces = mesh.faces
    distinct = (
        (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    )
    keep = distinct & (mesh.face_areas() > area_epsilon)
    if keep.all():
        return mesh
    out = mesh.copy()
    out.faces = np.ascontiguousarray(faces[keep])
    return out


def remove_duplicate_faces(mesh: Mesh) -> Mesh:
    """Drop triangles that occupy the same three vertices as another triangle.

    Where the front and back sheets both flatten onto the silhouette they emit
    the same triangle wound in opposite directions: a zero-thickness flap. The
    closed surface simply does not include those, and leaving them in makes
    their edges four-way non-manifold. Removing every face in a duplicated
    group (rather than keeping one) is what leaves the neighbours manifold.
    """
    if mesh.is_empty:
        return mesh
    key = np.sort(mesh.faces, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    keep = counts[inverse.reshape(-1)] == 1
    if keep.all():
        return mesh
    out = mesh.copy()
    out.faces = np.ascontiguousarray(mesh.faces[keep])
    return out


def compact_vertices(mesh: Mesh) -> tuple[Mesh, np.ndarray]:
    """Drop unreferenced vertices, returning the mesh and the kept indices.

    The index array lets a caller carry per-vertex data it tracks separately
    (which surface a vertex belongs to, for instance) through the reindexing.
    """
    if mesh.is_empty:
        return mesh, np.arange(mesh.n_vertices, dtype=np.int64)

    used_mask = np.zeros(mesh.n_vertices, dtype=bool)
    used_mask[mesh.faces.reshape(-1)] = True
    used = np.flatnonzero(used_mask).astype(np.int64)
    if used_mask.all():
        return mesh, used

    remap = np.full(mesh.n_vertices, -1, dtype=np.int32)
    remap[used_mask] = np.arange(len(used), dtype=np.int32)

    out = mesh.copy()
    out.vertices = np.ascontiguousarray(mesh.vertices[used_mask])
    out.faces = np.ascontiguousarray(remap[mesh.faces])
    if mesh.vertex_colors is not None:
        out.vertex_colors = np.ascontiguousarray(mesh.vertex_colors[used_mask])
    if mesh.vertex_normals is not None:
        out.vertex_normals = np.ascontiguousarray(mesh.vertex_normals[used_mask])
    if mesh.uvs is not None:
        out.uvs = np.ascontiguousarray(mesh.uvs[used_mask])
    return out, used


def remove_unreferenced_vertices(mesh: Mesh) -> Mesh:
    """Drop vertices no face refers to and reindex what remains."""
    return compact_vertices(mesh)[0]


def weld_vertices(mesh: Mesh, tolerance: float = 1e-6) -> Mesh:
    """Merge vertices that share a position, so the surface becomes connected.

    Grid meshing emits the front and back sheets separately; welding is what
    turns them into one manifold that smoothing and decimation can traverse.
    """
    if mesh.is_empty:
        return mesh
    if tolerance <= 0:
        return mesh

    quantized = np.round(mesh.vertices / tolerance).astype(np.int64)
    _, first_index, inverse = np.unique(quantized, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    if len(first_index) == mesh.n_vertices:
        return mesh

    # np.unique sorts its output, so remap through the sorted order.
    order = np.argsort(first_index)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    keep = first_index[order]

    out = mesh.copy()
    out.vertices = np.ascontiguousarray(mesh.vertices[keep])
    out.faces = np.ascontiguousarray(rank[inverse][mesh.faces].astype(np.int32))
    if mesh.vertex_colors is not None:
        out.vertex_colors = np.ascontiguousarray(mesh.vertex_colors[keep])
    if mesh.vertex_normals is not None:
        out.vertex_normals = np.ascontiguousarray(mesh.vertex_normals[keep])
    if mesh.uvs is not None:
        out.uvs = np.ascontiguousarray(mesh.uvs[keep])
    return remove_degenerate_faces(out)


def boundary_vertices(mesh: Mesh) -> np.ndarray:
    """Boolean mask of vertices lying on an open edge of the surface."""
    mask = np.zeros(mesh.n_vertices, dtype=bool)
    if mesh.is_empty:
        return mask
    undirected = np.sort(mesh.edges(), axis=1)
    unique, counts = np.unique(undirected, axis=0, return_counts=True)
    open_edges = unique[counts == 1]
    if len(open_edges):
        mask[open_edges.reshape(-1)] = True
    return mask


def vertex_adjacency(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric neighbour lists as a ``(source, target)`` index pair."""
    edges = mesh.edges()
    both = np.concatenate([edges, edges[:, ::-1]], axis=0)
    both = np.unique(both, axis=0)
    return both[:, 0].astype(np.int64), both[:, 1].astype(np.int64)


def connected_components(mesh: Mesh) -> tuple[np.ndarray, int]:
    """Label each vertex with its connected component index.

    Uses a union-find with path halving, which handles the few hundred thousand
    vertices a reconstruction produces without needing SciPy.
    """
    n = mesh.n_vertices
    if n == 0:
        return np.zeros(0, dtype=np.int32), 0

    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    for a, b in mesh.edges():
        root_a, root_b = find(int(a)), find(int(b))
        if root_a != root_b:
            parent[root_b] = root_a

    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    unique_roots, labels = np.unique(roots, return_inverse=True)
    return labels.reshape(-1).astype(np.int32), len(unique_roots)


def keep_largest_components(mesh: Mesh, min_ratio: float = 0.05) -> Mesh:
    """Discard components with far fewer vertices than the biggest one.

    Depth noise around the mask edge tends to break off into small floating
    islands; this removes them without touching the main body. Components are
    ranked by vertex count, which for grid-built meshes is proportional to the
    image area they came from — so this measures how much of the photo a piece
    accounts for, not how physically large it ended up.
    """
    if mesh.is_empty or min_ratio <= 0:
        return mesh
    labels, count = connected_components(mesh)
    if count <= 1:
        return mesh

    sizes = np.bincount(labels, minlength=count)
    threshold = sizes.max() * float(min_ratio)
    keep_labels = np.flatnonzero(sizes >= threshold)
    if len(keep_labels) == count:
        return mesh

    keep_vertex = np.isin(labels, keep_labels)
    keep_face = keep_vertex[mesh.faces].all(axis=1)

    out = mesh.copy()
    out.faces = np.ascontiguousarray(mesh.faces[keep_face])
    return remove_unreferenced_vertices(out)


def _neighbor_average(vertices: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    n = len(vertices)
    counts = np.bincount(src, minlength=n).astype(np.float32)
    counts[counts == 0] = 1.0
    averaged = np.empty_like(vertices)
    for axis in range(3):
        averaged[:, axis] = np.bincount(src, weights=vertices[:, axis][dst], minlength=n) / counts
    return averaged


def taubin_smooth(
    mesh: Mesh,
    iterations: int = 10,
    lam: float = 0.5,
    mu: float = -0.53,
    pin_boundary: bool = True,
) -> Mesh:
    """Taubin lambda/mu smoothing.

    A plain Laplacian pass shrinks the model a little every iteration. Taubin
    follows each shrinking pass with a slightly larger inflating pass, which
    removes high-frequency stair-stepping while holding the volume roughly
    constant.
    """
    if mesh.is_empty or iterations <= 0:
        return mesh

    src, dst = vertex_adjacency(mesh)
    if len(src) == 0:
        return mesh

    vertices = mesh.vertices.astype(np.float32).copy()
    movable = np.ones(len(vertices), dtype=np.float32)
    if pin_boundary:
        movable[boundary_vertices(mesh)] = 0.0
    movable = movable[:, None]

    for _ in range(int(iterations)):
        for factor in (lam, mu):
            laplacian = _neighbor_average(vertices, src, dst) - vertices
            vertices += movable * float(factor) * laplacian

    out = mesh.copy()
    out.vertices = np.ascontiguousarray(vertices.astype(np.float32))
    if out.vertex_normals is not None:
        out.vertex_normals = compute_vertex_normals(out.vertices, out.faces)
    return out


def clean_mesh(mesh: Mesh, weld_tolerance: float = 1e-6, min_component_ratio: float = 0.0) -> Mesh:
    """Weld, drop degenerate and duplicated faces, remove islands and reindex.

    The order matters: welding is what makes coincident triangles share
    indices, so duplicate removal has to follow it.
    """
    out = weld_vertices(mesh, weld_tolerance)
    out = remove_degenerate_faces(out)
    out = remove_duplicate_faces(out)
    if min_component_ratio > 0:
        out = keep_largest_components(out, min_component_ratio)
    return remove_unreferenced_vertices(out)
