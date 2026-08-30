"""Progress module."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

ProgressCallback = Callable[[str], None]

_progress_callback: ContextVar[ProgressCallback | None] = ContextVar(
    "starwars_autocalls_progress_callback",
    default=None,
)


def report_progress(message: str) -> None:
    """Report an interactive progress event when a callback is active."""
    callback = _progress_callback.get()
    if callback is not None:
        callback(message)


@contextmanager
def progress_reporting(callback: ProgressCallback) -> Iterator[None]:
    """Install a progress callback for the current command context."""
    token = _progress_callback.set(callback)
    try:
        yield
    finally:
        _progress_callback.reset(token)
