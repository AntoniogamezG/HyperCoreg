"""Public coregistration facade for the modular HyperCoreg pipeline."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from hypercoreg.pipeline import auth, orchestrator

__all__ = [
    "_prompt_cdse_userpass_gui",
    "run_batch_coregistration",
    "run_coregistration",
]


def run_coregistration(
    hs_file: str,
    hyp_type: str,
    output_dir: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    scene_idx: int = 1,
    scene_total: int = 1,
) -> Dict[str, Any]:
    return orchestrator.run_coregistration(
        hs_file,
        hyp_type,
        output_dir,
        config,
        progress_callback=progress_callback,
        scene_idx=scene_idx,
        scene_total=scene_total,
    )


def run_batch_coregistration(
    input_dir: str,
    output_dir: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    return orchestrator.run_batch_coregistration(
        input_dir,
        output_dir,
        config,
        progress_callback=progress_callback,
    )


def _prompt_cdse_userpass_gui() -> Any:
    return auth._prompt_cdse_userpass_gui()
