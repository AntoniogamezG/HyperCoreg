"""Sentinel-2 CDSE API helpers.

This module is the migration target for query/ranking/download runtime logic
now owned by the modular pipeline runtime.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CATALOGUE_URL",
    "HTTP_DOWNLOAD_RETRY_ATTEMPTS",
    "HTTP_QUERY_RETRY_ATTEMPTS",
    "HTTP_TIMEOUT_CONNECT_READ",
    "HTTP_TIMEOUT_S",
    "_bbox_to_wkt",
    "_download_s2_product",
    "_emit_progress",
    "_get_attr",
    "_query_s2",
    "_query_s2_with_retry",
    "_rank_s2_candidates",
]


def _runtime():
    # Imported lazily: runtime pulls in AROSICS/geoarray and is edited independently,
    # so names are looked up on every access instead of being copied at import time.
    return import_module("hypercoreg.pipeline.runtime")


def __getattr__(name: str) -> Any:
    if name in __all__:
        return getattr(_runtime(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
