"""Datetime helpers shared by pipeline modules."""

from __future__ import annotations

from datetime import datetime, timezone


def _to_utc_datetime(value: datetime) -> datetime:
    """Return a timezone-aware UTC datetime, treating naive values as UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
