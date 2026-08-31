"""Regression coverage for radiometric metadata across grid harmonization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


def test_shared_grid_resample_preserves_active_scale_offset_and_tags(
    tmp_path: Path,
) -> None:
    pytest.importorskip("arosics")
    pytest.importorskip("geoarray")
    from hypercoreg.pipeline import runtime

    source_path = tmp_path / "native_dn_source.tif"
    output_path = tmp_path / "native_dn_harmonized.tif"
    transform = from_origin(500000.0, 1000.0, 30.0, 30.0)
    values = np.stack(
        [
            np.arange(256, dtype=np.float32).reshape(16, 16),
            np.arange(256, dtype=np.float32).reshape(16, 16) + 1000.0,
        ]
    )

    with rasterio.open(
        source_path,
        "w",
        driver="GTiff",
        width=16,
        height=16,
        count=2,
        dtype="float32",
        crs="EPSG:32633",
        transform=transform,
        nodata=-9999.0,
        tiled=True,
        blockxsize=16,
        blockysize=16,
    ) as source:
        source.write(values)
        source.scales = (1.156710192278065e-5, 1.3255558216190724e-5)
        source.offsets = (0.0, 0.01)
        source.update_tags(RADIOMETRIC_QUANTITY="native_encoded_dn")
        source.set_band_description(1, "VNIR test band")
        source.set_band_description(2, "SWIR test band")
        source.update_tags(1, DETECTOR="VNIR", SOURCE_DN_GAIN=str(source.scales[0]))
        source.update_tags(2, DETECTOR="SWIR", SOURCE_DN_GAIN=str(source.scales[1]))

    result = runtime._resample_raster_to_shared_grid(
        source_path=str(source_path),
        output_path=str(output_path),
        target_transform=transform,
        target_width=16,
        target_height=16,
        target_crs="EPSG:32633",
        nodata=-9999.0,
        out_dtype=np.float32,
    )

    assert result["ok"] is True, result.get("error")
    with rasterio.open(output_path) as output:
        assert output.scales == pytest.approx(
            (1.156710192278065e-5, 1.3255558216190724e-5)
        )
        assert output.offsets == pytest.approx((0.0, 0.01))
        assert output.tags()["RADIOMETRIC_QUANTITY"] == "native_encoded_dn"
        assert output.descriptions == ("VNIR test band", "SWIR test band")
        assert output.tags(1)["DETECTOR"] == "VNIR"
        assert output.tags(2)["DETECTOR"] == "SWIR"
        np.testing.assert_allclose(output.read(), values, rtol=0.0, atol=0.0)
