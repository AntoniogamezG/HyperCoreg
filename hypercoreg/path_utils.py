"""Path helpers for report labels that may originate on another OS."""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from typing import Any


def _portable_name(path: Any, fallback: str = "") -> str:
    """Return the final path component for either POSIX or Windows separators."""
    if path is None:
        return fallback
    text = str(path)
    if not text:
        return fallback

    candidates = [PureWindowsPath(text).name, PurePosixPath(text).name]
    candidates = [candidate for candidate in candidates if candidate]
    if not candidates:
        return fallback
    return min(candidates, key=len)


def _portable_stem(path: Any, fallback: str = "") -> str:
    name = _portable_name(path, fallback=fallback)
    return PurePosixPath(name).stem if name else fallback


def _portable_suffix(path: Any) -> str:
    name = _portable_name(path)
    return PurePosixPath(name).suffix


__all__ = ["_portable_name", "_portable_stem", "_portable_suffix"]
