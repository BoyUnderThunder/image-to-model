"""Binary PLY writer with vertex colours and normals."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..errors import ExportError
from ..types import Mesh

__all__ = ["write_ply"]


def write_ply(mesh: Mesh, path: str | os.PathLike, **_: object) -> list[Path]:
    """Write ``mesh`` as a little-endian binary PLY."""
    if mesh.is_empty:
        raise ExportError("Cannot export an empty mesh to PLY")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    has_normals = mesh.vertex_normals is not None
    has_colors = mesh.vertex_colors is not None

    fields: list[tuple[str, str]] = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment created by image-to-model",
        f"element vertex {mesh.n_vertices}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_normals:
        fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
        header_lines += ["property float nx", "property float ny", "property float nz"]
    if has_colors:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        header_lines += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header_lines += [
        f"element face {mesh.n_faces}",
        "property list uchar int vertex_indices",
        "end_header",
    ]

    vertex_data = np.zeros(mesh.n_vertices, dtype=np.dtype(fields))
    vertex_data["x"], vertex_data["y"], vertex_data["z"] = mesh.vertices.T
    if has_normals:
        vertex_data["nx"], vertex_data["ny"], vertex_data["nz"] = mesh.vertex_normals.T
    if has_colors:
        vertex_data["red"], vertex_data["green"], vertex_data["blue"] = mesh.vertex_colors.T

    face_dtype = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
    face_data = np.zeros(mesh.n_faces, dtype=face_dtype)
    face_data["count"] = 3
    face_data["indices"] = mesh.faces

    with open(out_path, "wb") as handle:
        handle.write(("\n".join(header_lines) + "\n").encode("ascii"))
        handle.write(vertex_data.tobytes())
        handle.write(face_data.tobytes())

    return [out_path]
