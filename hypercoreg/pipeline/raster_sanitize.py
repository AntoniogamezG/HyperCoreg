"""Raster sanitization helpers with no legacy coregistration imports."""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np

from hypercoreg.config import PROCESSING_NODATA


def _sanitize_raster_nonfinite_inplace(
    path: str,
    nodata: float = PROCESSING_NODATA,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "ok": False,
        "error": None,
        "nonfinite_replaced": 0,
        "bands": 0,
    }
    try:
        import rasterio

        if not os.path.exists(path):
            result["error"] = f"Raster not found: {path}"
            return result
        with rasterio.open(path, "r+") as dst:
            result["bands"] = int(dst.count)
            if dst.count < 1:
                result["error"] = "Raster has no bands."
                return result

            src_nodata = dst.nodata
            write_nodata = (
                float(src_nodata)
                if src_nodata is not None and np.isfinite(float(src_nodata))
                else float(nodata)
            )
            if src_nodata is None or not np.isfinite(float(src_nodata)):
                dst.nodata = float(write_nodata)

            replaced_total = 0
            for bidx in range(1, int(dst.count) + 1):
                for _, window in dst.block_windows(bidx):
                    band = dst.read(bidx, window=window)
                    bad = ~np.isfinite(band)
                    if not np.any(bad):
                        continue
                    band = band.astype(np.float32, copy=False)
                    band[bad] = write_nodata
                    dst.write(band, bidx, window=window)
                    replaced_total += int(np.count_nonzero(bad))

        result["ok"] = True
        result["nonfinite_replaced"] = replaced_total
        return result
    except Exception as exc:
        result["error"] = str(exc)
        return result
