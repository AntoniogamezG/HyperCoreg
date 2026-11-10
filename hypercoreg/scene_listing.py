"""
Discover hyperspectral input scenes and describe them before processing.

Used by the CLI, the batch runner and the GUI to show which scenes will be
processed, with sensor, file, acquisition date and CRS.
"""

import glob
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger("COREG_PROCESSING")

HYPERSPECTRAL_INPUT_PATTERNS = (
    "*.he5",
    "*.HE5",
    "*-SPECTRAL_IMAGE.tif",
    "*-SPECTRAL_IMAGE.TIF",
    "*-SPECTRAL_IMAGE.tiff",
    "*-SPECTRAL_IMAGE.TIFF",
    "*-SPECTRAL_IMAGE.bsq",
    "*-SPECTRAL_IMAGE.BSQ",
)


def find_hyperspectral_inputs(input_dir: str) -> List[str]:
    """Return PRISMA .he5 and EnMAP SPECTRAL_IMAGE files below ``input_dir`` (recursive)."""
    found: List[str] = []
    for pattern in HYPERSPECTRAL_INPUT_PATTERNS:
        found.extend(glob.glob(os.path.join(input_dir, "**", pattern), recursive=True))
    # Normalize and deduplicate
    return list(dict.fromkeys(os.path.normcase(os.path.normpath(f)) for f in found))


def _utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    zone = int(np.floor((float(lon) + 180.0) / 6.0)) % 60 + 1
    return (32600 if float(lat) >= 0 else 32700) + zone


def _describe_prisma(path: str) -> Dict[str, Any]:
    import h5py

    from hypercoreg.readers.prisma import _parse_product_start_time

    with h5py.File(path, "r") as f:
        date = _parse_product_start_time(f.attrs["Product_StartTime"])
        lat = f["HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields/Latitude"][:]
        lon = f["HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields/Longitude"][:]
    epsg = _utm_epsg_for_lonlat(float(np.nanmean(lon)), float(np.nanmean(lat)))
    return {"date": date, "crs": f"EPSG:{epsg}", "crs_derived": True}


def _describe_enmap(path: str) -> Dict[str, Any]:
    import rasterio

    from hypercoreg.readers.enmap import find_enmap_metadata_for_spectral_image, read_enmap_metadata

    info: Dict[str, Any] = {"date": None, "crs": None, "crs_derived": False}
    with rasterio.open(path) as src:
        if src.crs is not None:
            epsg = src.crs.to_epsg()
            info["crs"] = f"EPSG:{epsg}" if epsg else src.crs.to_string()
    meta_path = find_enmap_metadata_for_spectral_image(path)
    if meta_path:
        meta = read_enmap_metadata(meta_path, spectral_image_path=path)
        info["date"] = meta.get("acquisition_time")
        if info["crs"] is None and meta.get("crs"):
            info["crs"] = str(meta.get("crs"))
    if info["date"] is None:
        match = re.search(r"_(\d{8}T\d{6})Z_", os.path.basename(path))
        if match:
            info["date"] = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
    return info


def describe_input_scenes(paths: Sequence[str]) -> List[Dict[str, Any]]:
    """Return one row per scene: sensor, file, path, date, crs, crs_derived, error."""
    from hypercoreg.utils import detect_hyp_type

    rows: List[Dict[str, Any]] = []
    for path in paths:
        row: Dict[str, Any] = {
            "sensor": None,
            "file": os.path.basename(path),
            "path": path,
            "date": None,
            "crs": None,
            "crs_derived": False,
            "error": None,
        }
        try:
            row["sensor"] = detect_hyp_type(path)
            row.update(_describe_prisma(path) if row["sensor"] == "PRISMA" else _describe_enmap(path))
        except Exception as exc:
            row["error"] = str(exc)
        rows.append(row)
    rows.sort(key=lambda r: (str(r["date"] or ""), r["file"]))
    return rows


def format_scene_crs(row: Dict[str, Any]) -> str:
    if not row.get("crs"):
        return "unknown"
    return str(row["crs"])


def format_scene_date(row: Dict[str, Any]) -> str:
    date = row.get("date")
    if date is None:
        return "unknown"
    if hasattr(date, "strftime"):
        return date.strftime("%Y-%m-%d %H:%M:%S")
    return str(date)


def format_scene_table(rows: Sequence[Dict[str, Any]]) -> str:
    """Plain-text table of the scenes to process."""
    header = ("#", "Sensor", "Date (UTC)", "CRS", "File")
    lines = [
        (str(i), str(r.get("sensor") or "?"), format_scene_date(r), format_scene_crs(r), r["file"])
        for i, r in enumerate(rows, start=1)
    ]
    widths = [max(len(h), *(len(line[c]) for line in lines)) if lines else len(h) for c, h in enumerate(header)]
    out = ["  ".join(h.ljust(w) for h, w in zip(header, widths))]
    out.append("  ".join("-" * w for w in widths))
    for r, line in zip(rows, lines):
        text = "  ".join(v.ljust(w) for v, w in zip(line, widths))
        if r.get("error"):
            text += f"  [not readable: {r['error']}]"
        out.append(text.rstrip())
    return "\n".join(out)


def log_scene_table(paths: Sequence[str], rows: Optional[Sequence[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Describe and log the scenes that will be processed; returns the rows."""
    rows = list(rows) if rows is not None else describe_input_scenes(paths)
    logger.info("Scenes to process (%d):\n%s", len(rows), format_scene_table(rows))
    return rows
