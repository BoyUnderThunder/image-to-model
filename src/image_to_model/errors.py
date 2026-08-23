"""Exception hierarchy for image-to-model."""

from __future__ import annotations


class ImageToModelError(Exception):
    """Base class for every error raised by this package."""


class InputError(ImageToModelError):
    """The supplied image could not be read or is unusable."""


class DependencyMissing(ImageToModelError):
    """An optional dependency is required for the requested feature.

    Carries the extra that installs it so the message can tell the caller
    exactly what to run.
    """

    def __init__(self, feature: str, package: str, extra: str | None = None) -> None:
        hint = f"pip install 'image-to-model[{extra}]'" if extra else f"pip install {package}"
        super().__init__(f"{feature} requires the '{package}' package. Install it with: {hint}")
        self.feature = feature
        self.package = package
        self.extra = extra


class SegmentationError(ImageToModelError):
    """The subject could not be separated from the background."""


class DepthEstimationError(ImageToModelError):
    """Depth prediction failed or produced a degenerate result."""


class ReconstructionError(ImageToModelError):
    """Geometry could not be built from the predicted depth."""


class ExportError(ImageToModelError):
    """The mesh could not be written in the requested format."""


class BackendNotFound(ImageToModelError):
    """No reconstruction backend is registered under the requested name."""

    def __init__(self, name: str, available: list[str]) -> None:
        super().__init__(
            f"Unknown backend {name!r}. Available backends: {', '.join(sorted(available)) or 'none'}"
        )
        self.name = name
        self.available = available
