"""High-level native pipeline orchestration wrappers."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from hypercoreg.pipeline import runtime
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

_ProgressHeartbeat = runtime._ProgressHeartbeat
_emit_progress = runtime._emit_progress
_collect_multiband_tiepoints = runtime._collect_multiband_tiepoints
_process_detector_branch_candidate = runtime._process_detector_branch_candidate


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

    result = runtime.run_coregistration(
        request.hs_file,
        request.hyp_type,
        request.output_dir,
        resolved_config,
        progress_callback=progress_callback,
        scene_idx=scene_idx,
        scene_total=scene_total,
    )
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

    result = runtime.run_batch_coregistration(
        request.input_dir,
        request.output_dir,
        resolved_config,
        progress_callback=progress_callback,
    )
    _sync_cdse_session_back(resolved_config, config)
    return result
