from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

_METRIC_KEYS = (
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


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def collect_artifact_inventory(root_dir: str) -> List[Dict[str, Any]]:
    root = Path(root_dir)
    if not root.exists():
        return []
    items: List[Dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        items.append(
            {
                "path": rel,
                "size": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    return items


def _subset_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {k: payload.get(k) for k in _METRIC_KEYS if k in payload}


def collect_metric_snapshots(root_dir: str) -> List[Dict[str, Any]]:
    root = Path(root_dir)
    if not root.exists():
        return []
    snapshots: List[Dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*.json") if p.is_file()):
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(parsed, dict):
            subset = _subset_metrics(parsed)
            if subset:
                snapshots.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "metrics": subset,
                    }
                )
    return snapshots


def build_baseline_record(
    fixture_id: str,
    backend: str,
    command: Iterable[str],
    return_code: int,
    output_dir: str,
) -> Dict[str, Any]:
    return {
        "fixture_id": str(fixture_id),
        "backend": str(backend),
        "command": [str(x) for x in command],
        "return_code": int(return_code),
        "output_dir": str(output_dir),
        "artifacts": collect_artifact_inventory(output_dir),
        "metric_snapshots": collect_metric_snapshots(output_dir),
    }


def index_artifacts(record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in record.get("artifacts", []) or []:
        key = str(item.get("path", ""))
        if key:
            out[key] = item
    return out


def index_metrics(record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in record.get("metric_snapshots", []) or []:
        key = str(item.get("path", ""))
        metrics = item.get("metrics")
        if key and isinstance(metrics, dict):
            out[key] = metrics
    return out


def compare_baseline_records(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    left_art = index_artifacts(left)
    right_art = index_artifacts(right)
    left_paths = set(left_art.keys())
    right_paths = set(right_art.keys())

    only_left = sorted(left_paths - right_paths)
    only_right = sorted(right_paths - left_paths)
    common = sorted(left_paths & right_paths)
    changed_artifacts = []
    for path in common:
        l = left_art[path]
        r = right_art[path]
        if l.get("sha256") != r.get("sha256") or l.get("size") != r.get("size"):
            changed_artifacts.append(
                {
                    "path": path,
                    "left_size": l.get("size"),
                    "right_size": r.get("size"),
                    "left_sha256": l.get("sha256"),
                    "right_sha256": r.get("sha256"),
                }
            )

    left_metrics = index_metrics(left)
    right_metrics = index_metrics(right)
    metric_paths = sorted(set(left_metrics.keys()) | set(right_metrics.keys()))
    metric_differences: List[Dict[str, Any]] = []
    for path in metric_paths:
        lm = left_metrics.get(path)
        rm = right_metrics.get(path)
        if lm is None or rm is None:
            metric_differences.append({"path": path, "left": lm, "right": rm})
            continue
        diffs: Dict[str, Dict[str, Any]] = {}
        for key in sorted(set(lm.keys()) | set(rm.keys())):
            if lm.get(key) != rm.get(key):
                diffs[key] = {"left": lm.get(key), "right": rm.get(key)}
        if diffs:
            metric_differences.append({"path": path, "differences": diffs})

    return {
        "left_fixture_id": left.get("fixture_id"),
        "right_fixture_id": right.get("fixture_id"),
        "left_backend": left.get("backend"),
        "right_backend": right.get("backend"),
        "only_left_artifacts": only_left,
        "only_right_artifacts": only_right,
        "changed_artifacts": changed_artifacts,
        "metric_differences": metric_differences,
        "has_differences": bool(
            only_left or only_right or changed_artifacts or metric_differences
        ),
    }

