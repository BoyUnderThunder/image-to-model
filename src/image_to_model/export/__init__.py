"""Mesh export in several formats.

All writers take a mesh in the pipeline's canonical convention (Y up, +Z
towards the camera, already scaled to the target's units) and handle any
per-format conversion themselves.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..errors import ExportError
from ..types import Mesh
from .axes import convert_up_axis, prepare_for_target, scale_to_units
from .gltf import write_glb
from .obj import write_obj
from .ply import write_ply
from .stl import write_stl

__all__ = [
    "export_mesh",
    "export_many",
    "write_obj",
    "write_glb",
    "write_ply",
    "write_stl",
    "convert_up_axis",
    "scale_to_units",
    "prepare_for_target",
    "SUPPORTED_FORMATS",
    "format_for_path",
]

_WRITERS = {
    "obj": write_obj,
    "glb": write_glb,
    "gltf": write_glb,  # always emitted as binary glTF
    "ply": write_ply,
    "stl": write_stl,
}

SUPPORTED_FORMATS = tuple(sorted(_WRITERS))

# Formats that can carry a texture map; the rest silently drop it.
_TEXTURED_FORMATS = {"obj", "glb", "gltf"}


def format_for_path(path: str | os.PathLike) -> str:
    """Infer the export format from a file extension."""
    suffix = Path(path).suffix.lower().lstrip(".")
    if not suffix:
        raise ExportError(
            f"No file extension on {path!r}; expected one of: {', '.join(SUPPORTED_FORMATS)}"
        )
    if suffix not in _WRITERS:
        raise ExportError(
            f"Unsupported export format {suffix!r}. Supported: {', '.join(SUPPORTED_FORMATS)}"
        )
    return suffix


def export_mesh(
    mesh: Mesh,
    path: str | os.PathLike,
    texture: np.ndarray | None = None,
    up_axis: str = "y",
    include_vertex_colors: bool = True,
    name: str = "model",
) -> list[Path]:
    """Write ``mesh`` to ``path``, choosing the writer from the extension.

    Returns every file produced. OBJ writes a sidecar ``.mtl`` and ``.png``
    when textured, so the list can hold more than one entry.
    """
    fmt = format_for_path(path)
    oriented = convert_up_axis(mesh, up_axis)
    writer = _WRITERS[fmt]
    return writer(
        oriented,
        path,
        texture=texture if fmt in _TEXTURED_FORMATS else None,
        include_vertex_colors=include_vertex_colors,
        name=name,
    )


def export_many(
    mesh: Mesh,
    out_dir: str | os.PathLike,
    stem: str,
    formats: list[str] | tuple[str, ...],
    texture: np.ndarray | None = None,
    up_axis: str = "y",
    include_vertex_colors: bool = True,
) -> list[Path]:
    """Write the same mesh in several formats into ``out_dir``."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        target_path = directory / f"{stem}.{fmt.lower().lstrip('.')}"
        written.extend(
            export_mesh(
                mesh,
                target_path,
                texture=texture,
                up_axis=up_axis,
                include_vertex_colors=include_vertex_colors,
                name=stem,
            )
        )
    return written
