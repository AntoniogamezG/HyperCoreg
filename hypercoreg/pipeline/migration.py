from __future__ import annotations

import os
from typing import Any, Dict

_PARITY_KEYS = (
    "status",
    "notes",
    "tie_points_count",
    "accuracy_pct",
    "residual_mean_m",
    "residual_median_m",
    "residual_rmse_m",
    "residual_p90_m",
    "polynomial_warp_used",
    "quality_score",
)


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled"}:
        return False
    return bool(default)


def use_pipeline_native(config: Dict[str, Any]) -> bool:
    if "use_pipeline_native" in config:
        return _parse_bool(config.get("use_pipeline_native"), default=False)
    return _parse_bool(os.environ.get("HYPERCOREG_USE_PIPELINE_NATIVE"), default=False)


def enable_legacy_fallback(config: Dict[str, Any]) -> bool:
    if "enable_legacy_fallback" in config:
        return _parse_bool(config.get("enable_legacy_fallback"), default=True)
    return _parse_bool(os.environ.get("HYPERCOREG_ENABLE_LEGACY_FALLBACK"), default=True)


def assert_legacy_parity(config: Dict[str, Any]) -> bool:
    if "assert_legacy_parity" in config:
        return _parse_bool(config.get("assert_legacy_parity"), default=False)
    return _parse_bool(os.environ.get("HYPERCOREG_ASSERT_LEGACY_PARITY"), default=False)


def compare_result_subset(native_result: Dict[str, Any], legacy_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return a key->value mismatch map for selected parity fields."""
    mismatches: Dict[str, Dict[str, Any]] = {}
    for key in _PARITY_KEYS:
        if native_result.get(key) != legacy_result.get(key):
            mismatches[key] = {
                "native": native_result.get(key),
                "legacy": legacy_result.get(key),
            }
    return mismatches
