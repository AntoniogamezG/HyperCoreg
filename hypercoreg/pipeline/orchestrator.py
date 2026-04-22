"""High-level pipeline orchestration wrappers."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from hypercoreg import _legacy_coreg as _legacy
from hypercoreg.pipeline import migration
from hypercoreg.pipeline._types import BatchSceneRequest, ResolvedRunConfig, SingleSceneRequest

__all__ = [
    "_ProgressHeartbeat",
    "_collect_multiband_tiepoints",
    "_emit_progress",
    "_process_detector_branch_candidate",
    "run_batch_coregistration",
    "run_coregistration",
]

logger = logging.getLogger("COREG_PROCESSING")

_LEGACY_RUN_COREGISTRATION = _legacy.run_coregistration
_LEGACY_RUN_BATCH_COREGISTRATION = _legacy.run_batch_coregistration
_LEGACY_PROCESS_DETECTOR_BRANCH_CANDIDATE = _legacy._process_detector_branch_candidate
_NATIVE_RUN_COREGISTRATION = None
_NATIVE_RUN_BATCH_COREGISTRATION = None
_ProgressHeartbeat = _legacy._ProgressHeartbeat
_emit_progress = _legacy._emit_progress
_collect_multiband_tiepoints = _legacy._collect_multiband_tiepoints
_process_detector_branch_candidate = _LEGACY_PROCESS_DETECTOR_BRANCH_CANDIDATE

_NATIVE_BACKEND_UNAVAILABLE_MESSAGE = (
    "Pipeline-native backend is not available yet. "
    "Disable use_pipeline_native/assert_legacy_parity or keep them unset."
)


def _raise_native_backend_unavailable() -> None:
    raise RuntimeError(_NATIVE_BACKEND_UNAVAILABLE_MESSAGE)


def _sync_cdse_session_back(source_config: Dict[str, Any], target_config: Dict[str, Any]) -> None:
    """Propagate mutable CDSE session state back to the caller config."""
    if "_cdse_session" in source_config:
        target_config["_cdse_session"] = source_config["_cdse_session"]


def run_coregistration(
    hs_file: str,
    hyp_type: str,
    output_dir: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    scene_idx: int = 1,
    scene_total: int = 1,
) -> Dict[str, Any]:
    request = SingleSceneRequest(
        hs_file=hs_file,
        hyp_type=hyp_type,
        output_dir=output_dir,
        config=ResolvedRunConfig.from_mapping(config),
    )
    resolved_config = request.config.to_dict()
    use_native = migration.use_pipeline_native(resolved_config)
    parity_check = migration.assert_legacy_parity(resolved_config)

    def _run_native() -> Dict[str, Any]:
        _raise_native_backend_unavailable()
        return {}

    def _run_legacy() -> Dict[str, Any]:
        return _LEGACY_RUN_COREGISTRATION(
            request.hs_file,
            request.hyp_type,
            request.output_dir,
            resolved_config,
            progress_callback=progress_callback,
            scene_idx=scene_idx,
            scene_total=scene_total,
        )

    if use_native or parity_check:
        _run_native()

    result = _run_legacy()
    _sync_cdse_session_back(resolved_config, config)
    return result


def run_batch_coregistration(
    input_dir: str,
    output_dir: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    request = BatchSceneRequest(
        input_dir=input_dir,
        output_dir=output_dir,
        config=ResolvedRunConfig.from_mapping(config),
    )
    resolved_config = request.config.to_dict()
    use_native = migration.use_pipeline_native(resolved_config)
    parity_check = migration.assert_legacy_parity(resolved_config)

    def _run_native() -> Dict[str, Any]:
        _raise_native_backend_unavailable()
        return {}

    def _run_legacy() -> Dict[str, Any]:
        return _LEGACY_RUN_BATCH_COREGISTRATION(
            request.input_dir,
            request.output_dir,
            resolved_config,
            progress_callback=progress_callback,
        )

    if use_native or parity_check:
        _run_native()

    result = _run_legacy()
    _sync_cdse_session_back(resolved_config, config)
    return result
