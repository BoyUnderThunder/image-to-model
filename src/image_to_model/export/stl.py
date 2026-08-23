"""Binary STL writer.

STL stores unindexed triangles with no colour or UVs, so it is only used for
3D-printing targets.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

from ..errors import ExportError
from ..types import Mesh

__all__ = ["write_stl"]


def write_stl(mesh: Mesh, path: str | os.PathLike, **_: object) -> list[Path]:
    """Write ``mesh`` as a binary STL file."""
    if mesh.is_empty:
        raise ExportError("Cannot export an empty mesh to STL")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tri = mesh.vertices[mesh.faces].astype(np.float32)  # (F, 3, 3)
    normals = mesh.face_normals().astype(np.float32)

    # Each STL record is 50 bytes: 3 floats of normal, 9 of vertices, then a
    # 2-byte attribute word. Build it as one array and dump it in a single write.
    records = np.zeros((len(tri), 50), dtype=np.uint8)
    payload = np.concatenate([normals[:, None, :], tri], axis=1).reshape(len(tri), 12)
    records[:, :48] = payload.astype("<f4").view(np.uint8).reshape(len(tri), 48)

    header = b"image-to-model binary STL".ljust(80, b"\0")
    with open(out_path, "wb") as handle:
        handle.write(header)
        handle.write(struct.pack("<I", len(tri)))
        handle.write(records.tobytes())

    return [out_path]
