"""Mesh simplification by quadric error metric edge collapse.

This is the Garland-Heckbert algorithm. Every vertex carries a quadric
summarising the squared distance to the planes of its incident faces; collapsing
an edge adds the two quadrics, and the cost of a collapse is the error at the
resulting position. Collapses are taken cheapest-first, so flat regions are
thinned aggressively while creases and silhouettes keep their detail.

It matters here because a platform polygon budget is a hard constraint: Roblox
will not import a mesh over its limit, and uniform grid subsampling to reach
that budget throws away exactly the sharp features that make a prop readable.

Two implementation notes:

* The inner loop is deliberately plain Python floats rather than NumPy. Each
  collapse only touches a dozen triangles, and at that size NumPy's per-call
  overhead dominates completely — scalar arithmetic runs the whole loop well
  over an order of magnitude faster.
* Each quadric is stored as the 10 unique entries of a symmetric 4x4 matrix,
  laid out as ``[a00 a01 a02 a03 a11 a12 a13 a22 a23 a33]``.

The collapse returns a map from each surviving vertex back to an original
vertex, which lets the caller re-derive UVs and colours by projection rather
than interpolating them through the collapse sequence.
"""

from __future__ import annotations

import heapq
import itertools

import numpy as np

from ..logging import get_logger
from ..types import Mesh
from .mesh_ops import boundary_vertices, remove_degenerate_faces, remove_duplicate_faces

__all__ = ["decimate", "DecimationResult"]

log = get_logger("geometry.decimate")


class DecimationResult:
    """A simplified mesh plus the mapping back to the original vertices."""

    __slots__ = ("mesh", "vertex_map")

    def __init__(self, mesh: Mesh, vertex_map: np.ndarray) -> None:
        self.mesh = mesh
        #: ``vertex_map[i]`` is the index in the *original* mesh of the vertex
        #: that survived to become vertex ``i`` here.
        self.vertex_map = vertex_map


def _accumulate_quadrics(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Sum each face's fundamental error quadric onto its three vertices.

    Done once up front, so this part stays vectorised.
    """
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 1e-18)
    offsets = -np.einsum("ij,ij->i", normals, tri[:, 0])
    planes = np.concatenate([normals, offsets[:, None]], axis=1)

    # Weighting by area keeps large flat faces from being outvoted by slivers.
    areas = (lengths.reshape(-1) * 0.5).astype(np.float64)

    pairs = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    products = np.stack([planes[:, i] * planes[:, j] * areas for i, j in pairs], axis=1)

    quadrics = np.zeros((len(vertices), 10), dtype=np.float64)
    for corner in range(3):
        index = faces[:, corner]
        for slot in range(10):
            quadrics[:, slot] += np.bincount(
                index, weights=products[:, slot], minlength=len(vertices)
            )
    return quadrics


def _quadric_error(q: list[float], x: float, y: float, z: float) -> float:
    """Evaluate ``p^T Q p`` for the symmetric quadric ``q`` at ``(x, y, z)``."""
    return (
        q[0] * x * x
        + 2.0 * q[1] * x * y
        + 2.0 * q[2] * x * z
        + 2.0 * q[3] * x
        + q[4] * y * y
        + 2.0 * q[5] * y * z
        + 2.0 * q[6] * y
        + q[7] * z * z
        + 2.0 * q[8] * z
        + q[9]
    )


def _solve_optimal(q: list[float]) -> tuple[float, float, float] | None:
    """Point where the quadric is minimised, or None if the system is singular.

    Solves the 3x3 system by Cramer's rule. Singularity is the normal case in
    flat or symmetric neighbourhoods, where a whole line of points is equally
    good and the caller should fall back to the edge endpoints.
    """
    a00, a01, a02 = q[0], q[1], q[2]
    a11, a12 = q[4], q[5]
    a22 = q[7]
    b0, b1, b2 = -q[3], -q[6], -q[8]

    c00 = a11 * a22 - a12 * a12
    c01 = a02 * a12 - a01 * a22
    c02 = a01 * a12 - a02 * a11
    determinant = a00 * c00 + a01 * c01 + a02 * c02
    if abs(determinant) < 1e-12:
        return None

    c11 = a00 * a22 - a02 * a02
    c12 = a02 * a01 - a00 * a12
    c22 = a00 * a11 - a01 * a01
    inverse_determinant = 1.0 / determinant
    return (
        (c00 * b0 + c01 * b1 + c02 * b2) * inverse_determinant,
        (c01 * b0 + c11 * b1 + c12 * b2) * inverse_determinant,
        (c02 * b0 + c12 * b1 + c22 * b2) * inverse_determinant,
    )


def decimate(
    mesh: Mesh,
    target_faces: int,
    preserve_boundary: bool = True,
    max_normal_flip_degrees: float = 100.0,
) -> DecimationResult:
    """Simplify ``mesh`` down to roughly ``target_faces`` triangles.

    ``target_faces <= 0``, or a mesh already under budget, returns the input
    unchanged with an identity vertex map.
    """
    identity = np.arange(mesh.n_vertices, dtype=np.int64)
    if mesh.is_empty or target_faces <= 0 or mesh.n_faces <= target_faces:
        return DecimationResult(mesh, identity)

    n_vertices = mesh.n_vertices
    quadrics = _accumulate_quadrics(mesh.vertices.astype(np.float64), mesh.faces.astype(np.int64))

    # Everything below runs on plain Python containers; see the module docstring.
    px = [float(v) for v in mesh.vertices[:, 0]]
    py = [float(v) for v in mesh.vertices[:, 1]]
    pz = [float(v) for v in mesh.vertices[:, 2]]
    quadric_list: list[list[float]] = quadrics.tolist()
    face_list: list[list[int]] = mesh.faces.astype(np.int64).tolist()

    face_alive = bytearray(b"\x01") * len(face_list)
    vertex_alive = bytearray(b"\x01") * n_vertices
    version = [0] * n_vertices

    locked = [False] * n_vertices
    if preserve_boundary:
        # Open borders define the silhouette of an unclosed surface; moving them
        # is far more visible than moving anything inside.
        for index in np.flatnonzero(boundary_vertices(mesh)):
            locked[int(index)] = True

    vertex_faces: list[set[int]] = [set() for _ in range(n_vertices)]
    for face_index, (a, b, c) in enumerate(face_list):
        vertex_faces[a].add(face_index)
        vertex_faces[b].add(face_index)
        vertex_faces[c].add(face_index)

    cos_limit = float(np.cos(np.radians(max_normal_flip_degrees)))
    counter = itertools.count()
    heap: list[tuple] = []

    def best_position(u: int, v: int) -> tuple[float, float, float, float]:
        """Cheapest placement for the merged vertex, as ``(cost, x, y, z)``."""
        qu, qv = quadric_list[u], quadric_list[v]
        combined = [qu[i] + qv[i] for i in range(10)]

        solved = _solve_optimal(combined)
        if solved is not None:
            x, y, z = solved
            return max(_quadric_error(combined, x, y, z), 0.0), x, y, z

        best = None
        for x, y, z in (
            (px[u], py[u], pz[u]),
            (px[v], py[v], pz[v]),
            ((px[u] + px[v]) * 0.5, (py[u] + py[v]) * 0.5, (pz[u] + pz[v]) * 0.5),
        ):
            error = _quadric_error(combined, x, y, z)
            if best is None or error < best[0]:
                best = (error, x, y, z)
        return max(best[0], 0.0), best[1], best[2], best[3]

    def push(u: int, v: int) -> None:
        if u == v or not (vertex_alive[u] and vertex_alive[v]):
            return
        if locked[u] and locked[v]:
            return
        cost, x, y, z = best_position(u, v)
        heapq.heappush(heap, (cost, next(counter), u, v, version[u], version[v], x, y, z))

    for a, b in np.unique(np.sort(mesh.edges(), axis=1), axis=0).tolist():
        push(int(a), int(b))

    def ring(u: int) -> set[int]:
        result: set[int] = set()
        for face_index in vertex_faces[u]:
            a, b, c = face_list[face_index]
            result.add(a)
            result.add(b)
            result.add(c)
        result.discard(u)
        return result

    def violates_link_condition(u: int, v: int) -> bool:
        """Reject collapses that would change the surface topology.

        The link condition: the vertices adjacent to both ``u`` and ``v`` must
        be exactly the vertices opposite the faces on edge ``(u, v)``. Any extra
        shared neighbour means the collapse would fuse two parts of the surface
        that only meet through it, producing a non-manifold edge or pinching a
        hole shut.
        """
        shared = ring(u) & ring(v)
        opposite = set()
        face_count = 0
        for face_index in vertex_faces[u] & vertex_faces[v]:
            a, b, c = face_list[face_index]
            face_count += 1
            for corner in (a, b, c):
                if corner != u and corner != v:
                    opposite.add(corner)
        if face_count not in (1, 2):
            return True  # already non-manifold here; leave it alone
        return shared != opposite

    def would_flip(u: int, v: int, x: float, y: float, z: float) -> bool:
        """True if moving ``u``/``v`` to the new point inverts an incident face."""
        for source in (u, v):
            other = v if source == u else u
            for face_index in vertex_faces[source]:
                a, b, c = face_list[face_index]
                if a == other or b == other or c == other:
                    continue  # this face disappears in the collapse

                ax, ay, az = (x, y, z) if a == source else (px[a], py[a], pz[a])
                bx, by, bz = (x, y, z) if b == source else (px[b], py[b], pz[b])
                cx, cy, cz = (x, y, z) if c == source else (px[c], py[c], pz[c])

                o1x, o1y, o1z = px[b] - px[a], py[b] - py[a], pz[b] - pz[a]
                o2x, o2y, o2z = px[c] - px[a], py[c] - py[a], pz[c] - pz[a]
                bnx = o1y * o2z - o1z * o2y
                bny = o1z * o2x - o1x * o2z
                bnz = o1x * o2y - o1y * o2x
                before_sq = bnx * bnx + bny * bny + bnz * bnz
                if before_sq < 1e-30:
                    continue

                n1x, n1y, n1z = bx - ax, by - ay, bz - az
                n2x, n2y, n2z = cx - ax, cy - ay, cz - az
                anx = n1y * n2z - n1z * n2y
                any_ = n1z * n2x - n1x * n2z
                anz = n1x * n2y - n1y * n2x
                after_sq = anx * anx + any_ * any_ + anz * anz
                if after_sq < 1e-30:
                    return True

                dot = bnx * anx + bny * any_ + bnz * anz
                if dot < cos_limit * (before_sq * after_sq) ** 0.5:
                    return True
        return False

    live_faces = len(face_list)
    collapses = 0

    while live_faces > target_faces and heap:
        _, _, u, v, version_u, version_v, x, y, z = heapq.heappop(heap)
        if not (vertex_alive[u] and vertex_alive[v]):
            continue
        if version[u] != version_u or version[v] != version_v:
            continue  # a neighbouring collapse invalidated this candidate

        # Collapse into whichever endpoint is free to move.
        if locked[u] and not locked[v]:
            u, v = v, u
        if locked[v]:
            continue
        if locked[u]:
            x, y, z = px[u], py[u], pz[u]

        if violates_link_condition(u, v) or would_flip(u, v, x, y, z):
            continue

        px[u], py[u], pz[u] = x, y, z
        qu, qv = quadric_list[u], quadric_list[v]
        quadric_list[u] = [qu[i] + qv[i] for i in range(10)]

        for face_index in list(vertex_faces[v]):
            a, b, c = face_list[face_index]
            if a == u or b == u or c == u:
                # Face spans the collapsed edge, so it degenerates away.
                if face_alive[face_index]:
                    face_alive[face_index] = 0
                    live_faces -= 1
                vertex_faces[a].discard(face_index)
                vertex_faces[b].discard(face_index)
                vertex_faces[c].discard(face_index)
                continue
            face = face_list[face_index]
            for slot in range(3):
                if face[slot] == v:
                    face[slot] = u
            vertex_faces[u].add(face_index)

        vertex_faces[v].clear()
        vertex_alive[v] = 0
        version[u] += 1
        collapses += 1

        for neighbor in ring(u):
            push(u, neighbor)

    kept = np.array([face_list[i] for i in range(len(face_list)) if face_alive[i]], dtype=np.int64)
    if kept.size == 0:
        return DecimationResult(mesh, identity)

    used = np.unique(kept.reshape(-1))
    remap = np.full(n_vertices, -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)

    positions = np.stack(
        [np.asarray(px, dtype=np.float32), np.asarray(py, dtype=np.float32), np.asarray(pz, dtype=np.float32)],
        axis=1,
    )
    simplified = Mesh(vertices=positions[used], faces=remap[kept])
    simplified = remove_duplicate_faces(remove_degenerate_faces(simplified))

    log.debug(
        "decimate: %d -> %d faces in %d collapses", mesh.n_faces, simplified.n_faces, collapses
    )
    return DecimationResult(simplified, used.astype(np.int64))
