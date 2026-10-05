"""Manifest and summary reporting helpers.

This module is the migration target for reporting/runtime metadata helpers
now owned by the modular pipeline runtime.
"""

from __future__ import annotations

import os
from importlib import import_module
from typing import Any, Dict, Optional, Tuple

__all__ = [
    "BATCH_SUMMARY_CONTEXT_COLUMNS",
    "BATCH_SUMMARY_XLSX_COLUMNS",
    "DATASET_MULTIBAND_COLUMNS",
    "DATASET_XLSX_COLUMNS",
    "METADATA_SCHEMA_VERSION",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "_atomic_write_text",
    "_build_batch_summary_row",
    "_build_dataset_row",
    "_build_dataset_xlsx_filename",
    "_build_failed_scene_metrics",
    "_build_scene_output_name",
    "_build_scene_output_root",
    "_build_skip_result",
    "_collect_batch_summary_rows",
    "_create_scene_root_with_collision_suffix",
    "_fmt_issue",
    "_format_scene_datetime_tags",
    "_infer_sensor_from_identifiers",
    "_resolve_dataset_scene_name",
    "_row_identity_key",
    "_safe_dataset_file_stem",
    "_sanitize_config_for_manifest",
    "_sanitize_manifest_value",
    "_scene_collision_suffix_candidates_from_source",
    "_scene_collision_suffix_from_source",
    "_scene_root_matches_source",
    "_write_batch_summary_txt",
    "_write_batch_summary_xlsx",
    "_write_per_scene_metrics_json",
    "_write_scene_run_manifest",
    "_write_shift_report",
    "_write_single_scene_dataset_xlsx",
]

# Defined only in this module (no runtime.py equivalent); everything else
# in __all__ resolves to hypercoreg.pipeline.runtime on attribute access.
_LOCAL_NAMES = frozenset({
    "_create_scene_root_with_collision_suffix",
    "_row_identity_key",
})


def _runtime():
    # Imported lazily: runtime pulls in AROSICS/geoarray and is edited independently,
    # so names are looked up on every access instead of being copied at import time.
    return import_module("hypercoreg.pipeline.runtime")


def _create_scene_root_with_collision_suffix(
    output_dir: str,
    hyp_type: Optional[str],
    acquisition_time: Any,
    source_path: Optional[str],
) -> Tuple[str, Optional[str]]:
    runtime = _runtime()
    _scene_root_matches_source = runtime._scene_root_matches_source
    _scene_collision_suffix_candidates_from_source = runtime._scene_collision_suffix_candidates_from_source
    # runtime._build_scene_output_dir is the runtime equivalent of this module's former
    # 3-argument _build_scene_output_root(output_dir, hyp_type, acquisition_time).
    scene_root = runtime._build_scene_output_dir(output_dir, hyp_type, acquisition_time)
    if scene_root == output_dir:
        return scene_root, None
    try:
        os.makedirs(scene_root, exist_ok=False)
        return scene_root, None
    except FileExistsError as exc:
        if _scene_root_matches_source(scene_root, source_path):
            raise FileExistsError(
                "Scene output folder already exists for this acquisition time and source identity: "
                f"{scene_root}. Remove or rename the existing folder before rerunning."
            ) from exc
        suffixes = _scene_collision_suffix_candidates_from_source(source_path)
        for suffix in suffixes:
            suffixed_root = f"{scene_root}_{suffix}"
            try:
                os.makedirs(suffixed_root, exist_ok=False)
                return suffixed_root, suffix
            except FileExistsError as suffix_exc:
                if _scene_root_matches_source(suffixed_root, source_path):
                    raise FileExistsError(
                        "Scene output folder already exists for this acquisition time and source identity: "
                        f"{suffixed_root}. Remove or rename the existing folder before rerunning."
                    ) from suffix_exc
                continue
        if suffixes:
            last_root = f"{scene_root}_{suffixes[-1]}"
            raise FileExistsError(
                "Scene output folder already exists for this acquisition time and all derived source suffixes: "
                f"{last_root}. Remove or rename the existing folder before rerunning."
            ) from exc
        raise FileExistsError(
            "Scene output folder already exists for this acquisition time: "
            f"{scene_root}. Remove or rename the existing folder before rerunning."
        ) from exc


def _row_identity_key(row: Dict[str, Any]) -> str:
    for field in ("source_path", "run_manifest_path", "dataset_xlsx_path"):
        value = row.get(field)
        if value:
            return str(os.path.normcase(os.path.abspath(str(value))))
    for field in ("scene_name", "filename"):
        value = row.get(field)
        if value:
            return f"{field}:{str(value)}"
    return ""


def __getattr__(name: str) -> Any:
    if name in __all__:
        return getattr(_runtime(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
