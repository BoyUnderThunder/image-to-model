"""Logging helpers.

The package logs through the standard :mod:`logging` module and installs no
handlers of its own when imported as a library. :func:`configure` is called by
the CLI so command-line runs get readable, timestamped progress output.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

LOGGER_NAME = "image_to_model"

_logger = logging.getLogger(LOGGER_NAME)
_logger.addHandler(logging.NullHandler())


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the package logger, or a child of it."""
    if name is None:
        return _logger
    return _logger.getChild(name)


def configure(verbosity: int = 0, stream=None) -> None:
    """Attach a stream handler to the package logger.

    ``verbosity`` of 0 shows warnings, 1 shows progress, 2 or more shows debug
    detail. Calling this repeatedly replaces the previous handler rather than
    stacking duplicates.
    """
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG

    for handler in list(_logger.handlers):
        if not isinstance(handler, logging.NullHandler):
            _logger.removeHandler(handler)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S"))
    _logger.addHandler(handler)
    _logger.setLevel(level)
    _logger.propagate = False


@contextmanager
def stage(name: str, logger: logging.Logger | None = None) -> Iterator[None]:
    """Log the start and wall-clock duration of a pipeline stage."""
    log = logger or _logger
    log.info("%s ...", name)
    start = time.perf_counter()
    try:
        yield
    except Exception:
        log.error("%s failed after %.2fs", name, time.perf_counter() - start)
        raise
    log.info("%s done in %.2fs", name, time.perf_counter() - start)
