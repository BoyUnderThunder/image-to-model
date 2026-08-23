"""End-to-end reconstruction: photograph in, 3D model out."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .backends import get_backend
from .config import ReconstructionConfig
from .errors import InputError
from .export import export_mesh
from .export.axes import scale_to_units
from .imaging import load_image, resize_longest, save_image
from .logging import get_logger, stage
from .segmentation import segment
from .targets import ValidationReport
from .types import DepthMap, Mesh, Subject

__all__ = ["ReconstructionResult", "reconstruct", "reconstruct_file"]

log = get_logger("pipeline")


@dataclass
class ReconstructionResult:
    """A finished model plus everything worth knowing about how it was made."""

    mesh: Mesh
    config: ReconstructionConfig
    subject: Subject
    texture: np.ndarray | None = None
    depth: DepthMap | None = None
    validation: ValidationReport = field(default_factory=ValidationReport)
    metadata: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def target_name(self) -> str:
        return self.config.profile.name

    def save(
        self,
        path: str | os.PathLike,
        formats: list[str] | tuple[str, ...] | None = None,
        write_guide: bool = True,
    ) -> list[Path]:
        """Write the model to disk.

        ``path`` may be a file (the extension chooses the format) or a
        directory, in which case the target's preferred formats are written
        using the source image's name. Returns every file created.
        """
        profile = self.config.profile
        destination = Path(path)
        written: list[Path] = []

        stem = Path(self.subject.source_path).stem if self.subject.source_path else "model"

        if formats:
            targets = [(destination / f"{stem}.{fmt.lstrip('.')}") for fmt in formats]
        elif destination.suffix:
            targets = [destination]
        else:
            targets = [destination / f"{stem}.{fmt}" for fmt in profile.preferred_formats]

        for target_path in targets:
            written.extend(
                export_mesh(
                    self.mesh,
                    target_path,
                    texture=self.texture,
                    up_axis=self.config.resolve_up_axis(),
                    include_vertex_colors=profile.supports_vertex_colors,
                    name=stem,
                )
            )

        if write_guide and self.config.target.startswith("roblox"):
            written.extend(self._write_roblox_guide(targets[0], stem))

        return written

    def _write_roblox_guide(self, mesh_path: Path, stem: str) -> list[Path]:
        """Drop an import guide and a Luau snippet next to the exported mesh."""
        from .targets.roblox import import_instructions, luau_snippet

        directory = mesh_path.parent
        directory.mkdir(parents=True, exist_ok=True)

        texture_name = f"{mesh_path.stem}_texture.png" if self.texture is not None else None
        guide_path = directory / "ROBLOX_IMPORT.md"
        guide_path.write_text(
            import_instructions(mesh_path.name, texture_name), encoding="utf-8"
        )

        script_path = directory / f"{stem}_spawn.luau"
        script_path.write_text(luau_snippet(), encoding="utf-8")
        return [guide_path, script_path]

    def save_debug(self, directory: str | os.PathLike) -> list[Path]:
        """Write the intermediate mask, depth map and texture for inspection."""
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = [save_image(self.subject.mask, out_dir / "mask.png")]
        if self.depth is not None:
            normalized = self.depth.normalized().depth
            written.append(save_image(1.0 - normalized, out_dir / "depth.png"))
        if self.texture is not None:
            written.append(save_image(self.texture, out_dir / "texture.png"))
        return written

    def summary(self) -> str:
        """A human-readable report of the run."""
        stats = self.mesh.stats()
        profile = self.config.profile
        unit = profile.unit_name
        extent = self.mesh.extent()

        lines = [
            f"Target        : {profile.name} ({profile.description})",
            f"Backend       : {self.metadata.get('backend', self.config.backend)}",
            f"Depth source  : {self.metadata.get('depth_source', 'n/a')}",
            f"Segmentation  : {self.metadata.get('segmentation', 'n/a')}",
            f"Triangles     : {stats['faces']:,}  (budget {profile.triangle_budget:,}"
            + (f", limit {profile.triangle_hard_limit:,}" if profile.triangle_hard_limit else "")
            + ")",
            f"Vertices      : {stats['vertices']:,}",
            f"Watertight    : {stats['watertight']}",
            f"Size          : {extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f} {unit}s",
            "Texture       : "
            + (
                f"{self.texture.shape[1]}x{self.texture.shape[0]}"
                if self.texture is not None
                else "none (vertex colours)"
            ),
        ]
        if self.timings:
            total = sum(self.timings.values())
            lines.append(f"Time          : {total:.2f}s")

        if self.validation.issues:
            lines.append("")
            lines.append("Target checks:")
            lines.extend(f"  {issue}" for issue in self.validation.issues)
        else:
            lines.append("Target checks : all passed")

        return "\n".join(lines)


def _load_subject(
    image: str | os.PathLike | np.ndarray, config: ReconstructionConfig
) -> tuple[Subject, str]:
    """Load, downscale and segment the input."""
    source_path = None
    if isinstance(image, (str, os.PathLike)):
        source_path = str(image)
        rgba = load_image(image)
    else:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise InputError(f"Expected an RGB or RGBA array, got shape {array.shape}")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.shape[2] == 3:
            opaque = np.full(array.shape[:2] + (1,), 255, dtype=np.uint8)
            array = np.concatenate([array, opaque], axis=2)
        rgba = array

    rgba = resize_longest(rgba, config.working_resolution)
    mask, method = segment(rgba, method=config.segmentation)

    subject = Subject(image=rgba[..., :3], mask=mask, source_path=source_path)
    return subject, method


def reconstruct(
    image: str | os.PathLike | np.ndarray,
    config: ReconstructionConfig | None = None,
    **overrides: Any,
) -> ReconstructionResult:
    """Turn a single image into a 3D model.

    ``image`` is a path or an RGB/RGBA array. Keyword overrides are applied on
    top of ``config``, so ``reconstruct("photo.jpg", target="roblox")`` works
    without building a config first.
    """
    config = (config or ReconstructionConfig()).merged(**overrides)
    config.validate()

    timings: dict[str, float] = {}

    start = time.perf_counter()
    with stage("Loading and segmenting", log):
        subject, segmentation_method = _load_subject(image, config)
    timings["load_and_segment"] = time.perf_counter() - start

    log.info(
        "Subject covers %.1f%% of a %dx%d image (segmentation: %s)",
        subject.coverage * 100.0,
        subject.width,
        subject.height,
        segmentation_method,
    )

    start = time.perf_counter()
    backend = get_backend(config.backend)
    output = backend.reconstruct(subject, config)
    timings["reconstruct"] = time.perf_counter() - start

    mesh = output.mesh
    if mesh.is_empty:
        raise InputError(
            "Reconstruction produced an empty mesh. The subject was probably not "
            "found; try --segmentation none to mesh the whole frame."
        )

    # Unit scaling happens here rather than in the backend so every backend
    # lands in the target's units the same way. Axis conversion is left to
    # export, since it differs per format.
    mesh = scale_to_units(mesh, config.resolve_size_units())
    mesh = mesh.with_normals()

    texture_size = None if output.texture is None else (
        int(output.texture.shape[1]),
        int(output.texture.shape[0]),
    )
    validation = config.profile.validate(mesh, texture_size)
    for issue in validation.issues:
        (log.error if issue.severity == "error" else log.warning)("%s", issue)

    metadata = dict(output.metadata)
    metadata["segmentation"] = segmentation_method
    metadata["working_resolution"] = config.working_resolution

    return ReconstructionResult(
        mesh=mesh,
        config=config,
        subject=subject,
        texture=output.texture,
        depth=output.depth,
        validation=validation,
        metadata=metadata,
        timings=timings,
    )


def reconstruct_file(
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
    config: ReconstructionConfig | None = None,
    **overrides: Any,
) -> tuple[ReconstructionResult, list[Path]]:
    """Reconstruct ``input_path`` and write the model to ``output_path``."""
    result = reconstruct(input_path, config=config, **overrides)
    written = result.save(output_path)
    return result, written
