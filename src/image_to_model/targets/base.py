"""Platform target profiles.

A target profile captures the constraints of wherever the model is going to be
used: polygon budget, texture ceiling, axis convention and unit scale. The
pipeline reads the profile to pick defaults, and validates the finished mesh
against it so a model that a platform would reject is caught here rather than
at import time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["TargetProfile", "ValidationIssue", "ValidationReport"]


@dataclass(frozen=True)
class ValidationIssue:
    """A single way the model does not meet a target's requirements."""

    severity: str  # "error" blocks import; "warning" is worth knowing about
    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.code}: {self.message}"


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        """True when nothing would block import on the target platform."""
        return not self.errors

    def add(self, severity: str, code: str, message: str) -> None:
        self.issues.append(ValidationIssue(severity, code, message))

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        if not self.issues:
            return "All target checks passed."
        return "\n".join(str(i) for i in self.issues)


@dataclass(frozen=True)
class TargetProfile:
    """Constraints and conventions for one destination platform."""

    name: str
    description: str = ""

    # Geometry budget.
    triangle_budget: int = 100_000
    """Default decimation target: what the pipeline aims for."""

    triangle_hard_limit: int = 0
    """Import-blocking ceiling. 0 means the platform imposes none."""

    # Texturing.
    max_texture_size: int = 4096
    supports_vertex_colors: bool = True
    """When False the mesh needs real UVs and a baked texture image, because the
    platform silently drops per-vertex colour on import."""

    # Conventions.
    up_axis: str = "y"
    """``y`` or ``z``. Source geometry is produced Y-up and converted on export."""

    unit_name: str = "meter"
    meters_per_unit: float = 1.0
    """Size of one exported unit in metres."""

    default_size_units: float = 1.0
    """Longest axis of the exported model, in ``unit_name`` units."""

    preferred_formats: tuple[str, ...] = ("glb", "obj")
    """Export formats the platform imports, best first."""

    def size_in_meters(self) -> float:
        return self.default_size_units * self.meters_per_unit

    def validate(self, mesh, texture_size: tuple[int, int] | None = None) -> ValidationReport:
        """Check a finished mesh against this profile."""
        report = ValidationReport()

        if mesh.is_empty:
            report.add("error", "empty_mesh", "The mesh has no geometry.")
            return report

        if self.triangle_hard_limit and mesh.n_faces > self.triangle_hard_limit:
            report.add(
                "error",
                "triangle_limit",
                f"{mesh.n_faces:,} triangles exceeds the {self.name} limit of "
                f"{self.triangle_hard_limit:,}. Lower --target-faces.",
            )
        elif mesh.n_faces > self.triangle_budget:
            report.add(
                "warning",
                "triangle_budget",
                f"{mesh.n_faces:,} triangles is above the recommended budget of "
                f"{self.triangle_budget:,} for {self.name}.",
            )

        if not self.supports_vertex_colors and mesh.uvs is None:
            report.add(
                "error",
                "missing_uvs",
                f"{self.name} ignores per-vertex colour, so the mesh needs UVs and a "
                "baked texture. Enable texture baking.",
            )

        if texture_size is not None:
            width, height = texture_size
            if max(width, height) > self.max_texture_size:
                report.add(
                    "error",
                    "texture_too_large",
                    f"Texture is {width}x{height}; {self.name} allows at most "
                    f"{self.max_texture_size}x{self.max_texture_size}.",
                )

        if mesh.uvs is not None:
            uvs = np.asarray(mesh.uvs)
            if uvs.size and (uvs.min() < -1e-4 or uvs.max() > 1.0 + 1e-4):
                report.add(
                    "warning",
                    "uv_out_of_range",
                    "Some UV coordinates fall outside [0, 1]; they will wrap on import.",
                )

        # Judged after merging coincident vertices, so UV seams -- which split
        # vertices by design -- are not mistaken for holes.
        if not mesh.is_closed_surface():
            report.add(
                "warning",
                "not_watertight",
                "The mesh is not closed. It will still import, but physics and "
                "CSG behave better on solid geometry.",
            )

        degenerate = int((mesh.face_areas() <= 1e-12).sum())
        if degenerate:
            report.add(
                "warning",
                "degenerate_faces",
                f"{degenerate:,} zero-area triangles remain in the mesh.",
            )

        return report
