"""Shared helpers for interpreting pipeline result payloads."""

from __future__ import annotations

from typing import Any, Mapping, Optional


def _count_items(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, Mapping):
        return len(value)
    try:
        return len(value)
    except TypeError:
        return 1


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _failure_status_reason(result: Mapping[str, Any]) -> Optional[str]:
    status = str(result.get("status", "") or "").strip()
    status_upper = status.upper()
    if bool(result.get("aborted")) or status_upper.startswith("ABORTED_"):
        reason = result.get("abort_reason") or result.get("reason") or result.get("error")
        if reason:
            return f"Batch aborted with status {status or 'ABORTED'}: {reason}"
        return f"Batch aborted with status {status or 'ABORTED'}"
    if not status_upper or status_upper.startswith("SKIPPED_"):
        return None
    if status_upper in {"FAIL", "FAILED", "ERROR"} or status_upper.startswith("FAIL"):
        reason = (
            result.get("reason")
            or result.get("error")
            or result.get("message")
            or result.get("notes")
        )
        if reason:
            return f"Pipeline returned status {status}: {reason}"
        return f"Pipeline returned status {status}"
    return None


def describe_result_failure(result: Any, mode: str) -> Optional[str]:
    """Return a human-readable failure reason for a pipeline result payload."""
    if not isinstance(result, Mapping):
        return None

    status_failure = _failure_status_reason(result)
    if status_failure:
        return status_failure

    if str(mode).lower() != "batch":
        return None

    total = _as_int(result.get("total"))
    if total is not None and total <= 0:
        return "Batch found no valid hyperspectral scenes."

    failed_count = max(
        _count_items(result.get("failed")),
        _count_items(result.get("failed_details")),
        _count_items(result.get("errors")),
    )
    if failed_count > 0:
        return f"Batch completed with {failed_count} failed scene(s)."

    return None
