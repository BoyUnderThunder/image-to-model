"""Binary glTF (.glb) writer.

Produces a single self-contained file with geometry, normals, UVs, optional
vertex colours and an embedded PNG texture. Written by hand against the glTF
2.0 spec so the package needs no glTF library.
"""

from __future__ import annotations

import io
import json
import os
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from ..errors import ExportError
from ..types import Mesh

__all__ = ["write_glb"]

# glTF component and buffer-view type constants.
_FLOAT = 5126
_UNSIGNED_BYTE = 5121
_UNSIGNED_SHORT = 5123
_UNSIGNED_INT = 5125
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963

_GLB_MAGIC = 0x46546C67  # "glTF"
_CHUNK_JSON = 0x4E4F534A  # "JSON"
_CHUNK_BIN = 0x004E4942  # "BIN\0"


def _encode_png(array: np.ndarray) -> bytes:
    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    mode = "RGB" if arr.ndim == 3 and arr.shape[2] == 3 else "RGBA"
    buffer = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


class _BufferBuilder:
    """Accumulates binary data and the bufferViews/accessors that describe it."""

    def __init__(self) -> None:
        self.data = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def add_view(self, payload: bytes, target: int | None = None) -> int:
        # Every view starts on a 4-byte boundary, which satisfies the alignment
        # requirement for all component types used here.
        while len(self.data) % 4:
            self.data.append(0)
        view = {
            "buffer": 0,
            "byteOffset": len(self.data),
            "byteLength": len(payload),
        }
        if target is not None:
            view["target"] = target
        self.data.extend(payload)
        self.views.append(view)
        return len(self.views) - 1

    def add_accessor(
        self,
        array: np.ndarray,
        component_type: int,
        type_name: str,
        target: int | None,
        normalized: bool = False,
        with_bounds: bool = False,
    ) -> int:
        view_index = self.add_view(array.tobytes(), target)
        accessor: dict = {
            "bufferView": view_index,
            "componentType": component_type,
            "count": int(array.shape[0]),
            "type": type_name,
        }
        if normalized:
            accessor["normalized"] = True
        if with_bounds:
            accessor["min"] = [float(v) for v in np.asarray(array).min(axis=0)]
            accessor["max"] = [float(v) for v in np.asarray(array).max(axis=0)]
        self.accessors.append(accessor)
        return len(self.accessors) - 1


def write_glb(
    mesh: Mesh,
    path: str | os.PathLike,
    texture: np.ndarray | None = None,
    include_vertex_colors: bool = True,
    name: str = "model",
    **_: object,
) -> list[Path]:
    """Write ``mesh`` as a single binary glTF file."""
    if mesh.is_empty:
        raise ExportError("Cannot export an empty mesh to GLB")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    builder = _BufferBuilder()
    attributes: dict[str, int] = {}

    positions = np.ascontiguousarray(mesh.vertices, dtype="<f4")
    attributes["POSITION"] = builder.add_accessor(
        positions, _FLOAT, "VEC3", _ARRAY_BUFFER, with_bounds=True
    )

    if mesh.vertex_normals is not None:
        normals = np.ascontiguousarray(mesh.vertex_normals, dtype="<f4")
        attributes["NORMAL"] = builder.add_accessor(normals, _FLOAT, "VEC3", _ARRAY_BUFFER)

    if mesh.uvs is not None:
        uvs = np.ascontiguousarray(mesh.uvs, dtype="<f4")
        attributes["TEXCOORD_0"] = builder.add_accessor(uvs, _FLOAT, "VEC2", _ARRAY_BUFFER)

    if include_vertex_colors and mesh.vertex_colors is not None:
        rgba = np.concatenate(
            [mesh.vertex_colors, np.full((mesh.n_vertices, 1), 255, dtype=np.uint8)], axis=1
        )
        attributes["COLOR_0"] = builder.add_accessor(
            np.ascontiguousarray(rgba, dtype=np.uint8),
            _UNSIGNED_BYTE,
            "VEC4",
            _ARRAY_BUFFER,
            normalized=True,
        )

    # Indices are flattened to a scalar accessor; 16-bit where it fits.
    if mesh.n_vertices <= 65_535:
        index_array = np.ascontiguousarray(mesh.faces.reshape(-1), dtype="<u2")
        index_component = _UNSIGNED_SHORT
    else:
        index_array = np.ascontiguousarray(mesh.faces.reshape(-1), dtype="<u4")
        index_component = _UNSIGNED_INT
    index_view = builder.add_view(index_array.tobytes(), _ELEMENT_ARRAY_BUFFER)
    builder.accessors.append(
        {
            "bufferView": index_view,
            "componentType": index_component,
            "count": int(index_array.size),
            "type": "SCALAR",
        }
    )
    index_accessor = len(builder.accessors) - 1

    material: dict = {
        "name": f"{name}_material",
        "doubleSided": True,
        "pbrMetallicRoughness": {
            "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.9,
        },
    }

    gltf: dict = {
        "asset": {"version": "2.0", "generator": "image-to-model"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [
            {
                "name": name,
                "primitives": [
                    {"attributes": attributes, "indices": index_accessor, "material": 0, "mode": 4}
                ],
            }
        ],
        "materials": [material],
    }

    if texture is not None and mesh.uvs is not None:
        png_bytes = _encode_png(texture)
        image_view = builder.add_view(png_bytes)
        gltf["images"] = [{"bufferView": image_view, "mimeType": "image/png", "name": f"{name}_tex"}]
        gltf["samplers"] = [
            {
                "magFilter": 9729,  # LINEAR
                "minFilter": 9987,  # LINEAR_MIPMAP_LINEAR
                "wrapS": 33071,  # CLAMP_TO_EDGE, avoids bleeding across atlas seams
                "wrapT": 33071,
            }
        ]
        gltf["textures"] = [{"sampler": 0, "source": 0}]
        material["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0, "texCoord": 0}

    binary = bytes(builder.data)
    gltf["bufferViews"] = builder.views
    gltf["accessors"] = builder.accessors
    gltf["buffers"] = [{"byteLength": len(binary)}]

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)  # JSON pads with spaces
    binary += b"\0" * ((4 - len(binary) % 4) % 4)  # BIN pads with zeros

    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    with open(out_path, "wb") as handle:
        handle.write(struct.pack("<III", _GLB_MAGIC, 2, total))
        handle.write(struct.pack("<II", len(json_bytes), _CHUNK_JSON))
        handle.write(json_bytes)
        handle.write(struct.pack("<II", len(binary), _CHUNK_BIN))
        handle.write(binary)

    return [out_path]
