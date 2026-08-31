"""Focused regression tests for the PRISMA PCO quality matrix contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import h5py
import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from hypercoreg.readers import read_prisma_pan_quality_mask


PAN_QUALITY_PATH = "HDFEOS/SWATHS/PRS_L2D_PCO/Data Fields/PIXEL_L2_ERR_MATRIX"


def _write_pan_quality(path: Path, data: np.ndarray, *, dataset_path: str = PAN_QUALITY_PATH) -> Path:
    with h5py.File(path, "w") as product:
        parent, name = dataset_path.rsplit("/", 1)
        product.require_group(parent).create_dataset(name, data=data)
    return path


def test_pan_quality_reader_reads_exact_pco_matrix_as_uint8(tmp_path: Path) -> None:
    expected = np.array([[0, 1, 2], [3, 4, 255]], dtype=np.int16)
    path = _write_pan_quality(tmp_path / "scene.he5", expected)

    actual = read_prisma_pan_quality_mask(str(path))

    assert actual is not None
    assert actual.dtype == np.uint8
    assert actual.shape == (2, 3)
    np.testing.assert_array_equal(actual, expected.astype(np.uint8))


@pytest.mark.parametrize(
    ("data", "warning_fragment"),
    [
        (np.zeros((1, 2, 3), dtype=np.uint8), "exactly 2-D"),
        (np.array([[0, 256]], dtype=np.int16), "uint8 range"),
        (np.array([[0.0, 1.0]], dtype=np.float32), "integer uint8-compatible"),
    ],
)
def test_pan_quality_reader_rejects_noncompatible_matrix(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    data: np.ndarray,
    warning_fragment: str,
) -> None:
    path = _write_pan_quality(tmp_path / "invalid.he5", data)

    assert read_prisma_pan_quality_mask(str(path)) is None
    assert warning_fragment in caplog.text


def test_pan_quality_reader_does_not_probe_nonstandard_dataset_names(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = _write_pan_quality(
        tmp_path / "wrong_name.he5",
        np.zeros((2, 3), dtype=np.uint8),
        dataset_path="HDFEOS/SWATHS/PRS_L2D_PCO/Data Fields/PIXEL_ERR_MATRIX",
    )

    assert read_prisma_pan_quality_mask(str(path)) is None
    assert PAN_QUALITY_PATH in caplog.text


def _run_mocked_pan_ancillary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pan_quality_data: np.ndarray | None,
    *,
    fail_quality_warp: bool = False,
):
    # The runtime imports these optional processing packages eagerly.  Their
    # actual algorithms are replaced below, so lightweight modules are enough
    # for this focused orchestration test on environments without AROSICS.
    using_runtime_stubs = False
    if importlib.util.find_spec("geoarray") is None:
        using_runtime_stubs = True
        geoarray_stub = types.ModuleType("geoarray")
        geoarray_stub.GeoArray = object
        monkeypatch.setitem(sys.modules, "geoarray", geoarray_stub)
    if importlib.util.find_spec("arosics") is None:
        using_runtime_stubs = True
        arosics_stub = types.ModuleType("arosics")
        arosics_stub.COREG = object
        arosics_stub.COREG_LOCAL = object
        monkeypatch.setitem(sys.modules, "arosics", arosics_stub)
    from hypercoreg.pipeline import runtime

    temp_dir = tmp_path / "temp"
    coreg_dir = tmp_path / "coreg"
    temp_dir.mkdir(parents=True)
    coreg_dir.mkdir(parents=True)
    folder_struct = {"temp": str(temp_dir), "coreg": str(coreg_dir)}
    target_crs = None
    transform = Affine.translation(500000.0, 1000.0) * Affine.scale(5.0, -5.0)

    pan = np.array(
        [
            [0.0, 11.0, 22.0, 33.0],
            [44.0, 55.0, 66.0, 77.0],
            [88.0, 99.0, 111.0, 122.0],
        ],
        dtype=np.float32,
    )
    pan_geo = {
        "rows": 3,
        "cols": 4,
        "ul_lon": 0.0,
        "ul_lat": 1.0,
        "ur_lon": 1.0,
        "ur_lat": 1.0,
        "ll_lon": 0.0,
        "ll_lat": 0.0,
        "lr_lon": 1.0,
        "lr_lat": 0.0,
        "pixel_size_m": 5.0,
    }
    accepted_gcps = [object() for _ in range(6)]
    warp_calls = []

    monkeypatch.setattr(runtime, "_estimate_transform_from_corner_coords", lambda *_: transform)
    monkeypatch.setattr(runtime, "_infer_raster_native_resolution", lambda *_args, **_kwargs: (5.0, 5.0))
    monkeypatch.setattr(
        runtime,
        "_merge_tiepoints",
        lambda frames, **_kwargs: {"merged_df": frames[0], "fallback_notes": []},
    )
    monkeypatch.setattr(
        runtime,
        "_build_tps_gcps_for_source_raster",
        lambda *_args, **_kwargs: {"success": True, "gcps": accepted_gcps},
    )
    monkeypatch.setattr(
        runtime,
        "_decide_polynomial_order",
        lambda **_kwargs: {"order_used": 1},
    )

    def _fake_polynomial_warp(**kwargs):
        warp_calls.append(dict(kwargs))
        if fail_quality_warp and "PAN_quality_src" in str(kwargs["input_raster"]):
            raise RuntimeError("forced categorical quality warp failure")
        with rasterio.open(kwargs["input_raster"]) as src:
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                nodata=kwargs["nodata"],
                compress="lzw",
                tiled=True,
            )
            with rasterio.open(kwargs["output_raster"], "w", **profile) as dst:
                dst.write(src.read())
        return {"success": True, "output_path": kwargs["output_raster"]}

    monkeypatch.setattr(runtime, "_apply_polynomial_warp", _fake_polynomial_warp)
    monkeypatch.setattr(
        runtime,
        "_validate_ancillary_raster",
        lambda *_args, **_kwargs: {"ok": True},
    )

    result = runtime._coregister_prisma_ancillary_outputs(
        scene_name="PRISMA_TEST",
        folder_struct=folder_struct,
        s2_crs=target_crs,
        save_pan=True,
        save_quality_mask=False,
        pan_data=pan,
        pan_geo_info=pan_geo,
        vnir_quality_data=None,
        swir_quality_data=None,
        lat_qm=None,
        lon_qm=None,
        best_candidate={"local_tiepoints_df": [{"id": 1}]},
        pan_use_synthetic_reference=False,
        pan_min_points_for_poly2=20,
        pan_quality_data=pan_quality_data,
        radiometric_contract=None,
    )
    if using_runtime_stubs:
        # Do not make a stub-backed runtime import visible to unrelated tests
        # that correctly skip when the optional production stack is absent.
        sys.modules.pop("hypercoreg.pipeline.runtime", None)
        import hypercoreg.pipeline as pipeline_package

        pipeline_package.__dict__.pop("runtime", None)
    return result, pan, accepted_gcps, warp_calls, temp_dir


def test_runtime_masks_only_flag4_and_exports_full_categorical_pan_quality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality = np.array(
        [
            [1, 2, 3, 4],
            [0, 1, 2, 255],
            [4, 0, 3, 2],
        ],
        dtype=np.uint8,
    )
    result, pan, accepted_gcps, warp_calls, temp_dir = _run_mocked_pan_ancillary(
        tmp_path,
        monkeypatch,
        quality,
    )

    assert result["status"] == "ok"
    assert result["pan_quality_evidence"]["status"] == "exported"
    assert result["pan_quality_evidence"]["numeric_pixels_masked"] == 3
    assert result["pan_quality_evidence"]["source_nodata_pixels"] == 1
    assert result["pan_quality_evidence"]["prewarp_hole_fill"] == (
        "inverse_distance_from_valid_pan_pixels"
    )
    assert result["pan_quality_evidence"]["prewarp_hole_fill_unfilled_pixels"] == 0
    assert result["pan_quality_evidence"]["postwarp_numeric_mask_applied"] is True
    assert result["pan_quality_evidence"]["postwarp_flag4_pixels"] == 2
    assert result["pan_quality_evidence"]["grid_matches_pan"] is True
    assert result["pan_quality_evidence"]["observed_output_categories"] == [0, 1, 2, 3, 4, 255]

    pan_path = Path(result["outputs"]["pan"])
    quality_path = Path(result["outputs"]["quality_pan"])
    assert pan_path.exists()
    assert quality_path.exists()
    assert quality_path.with_suffix(".hdr").exists()

    with rasterio.open(pan_path) as pan_ds, rasterio.open(quality_path) as quality_ds:
        pan_out = pan_ds.read(1)
        np.testing.assert_array_equal(quality_ds.read(1), quality)
        assert quality_ds.dtypes == ("uint8",)
        assert quality_ds.nodata == 255
        assert quality_ds.transform == pan_ds.transform
        assert quality_ds.crs == pan_ds.crs
        assert quality_ds.width == pan_ds.width
        assert quality_ds.height == pan_ds.height
        assert quality_ds.tags()["QUALITY_RESAMPLING"] == "nearest"
        assert quality_ds.tags()["PAN_NUMERIC_NODATA_FLAG"] == "4"
        assert quality_ds.tags()["PAN_NUMERIC_OUTSIDE_FOOTPRINT_NODATA_FLAG"] == "255"

    # Flags 1-3 do not mask values, including native DN zero; flag 4 does.
    assert pan_out[0, 0] == 0.0
    assert pan_out[0, 1] == pan[0, 1]
    assert pan_out[0, 2] == pan[0, 2]
    assert pan_out[0, 3] == -9999.0
    assert pan_out[1, 3] == -9999.0
    assert pan_out[2, 0] == -9999.0

    assert len(warp_calls) == 2
    pan_call, quality_call = warp_calls
    assert pan_call["gcps"] is accepted_gcps
    assert quality_call["gcps"] is accepted_gcps
    assert quality_call["polynomial_order"] == pan_call["polynomial_order"] == 1
    assert quality_call["output_resolution"] == pan_call["output_resolution"] == 5.0
    assert quality_call["resampling"] == "near"
    assert quality_call["nodata"] == 255
    assert quality_call["s2_bounds"] is not None

    assert not list(temp_dir.glob("*PAN*"))


def test_runtime_refuses_to_warp_unfilled_pan_quality_holes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality = np.full((3, 4), 4, dtype=np.uint8)

    result, _pan, _accepted_gcps, warp_calls, temp_dir = _run_mocked_pan_ancillary(
        tmp_path,
        monkeypatch,
        quality,
    )

    assert result["status"] == "degraded"
    assert result["outputs"]["pan"] is None
    assert result["pan_quality_evidence"]["status"] == "prefill_failed"
    assert result["pan_quality_evidence"]["prewarp_hole_fill_unfilled_pixels"] == 12
    assert any("refusing to warp source-invalid" in warning for warning in result["warnings"])
    assert warp_calls == []
    assert not list(temp_dir.glob("*PAN*"))


def test_runtime_withholds_numeric_pan_when_available_quality_cannot_be_applied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality = np.zeros((3, 4), dtype=np.uint8)
    quality[0, 3] = 4

    result, _pan, _accepted_gcps, warp_calls, temp_dir = _run_mocked_pan_ancillary(
        tmp_path,
        monkeypatch,
        quality,
        fail_quality_warp=True,
    )

    assert result["status"] == "degraded"
    assert result["outputs"]["pan"] is None
    assert result["outputs"]["quality_pan"] is None
    assert result["pan_quality_evidence"]["status"] == "export_failed"
    assert result["pan_quality_evidence"]["postwarp_numeric_mask_applied"] is False
    assert result["pan_quality_evidence"]["numeric_output_withheld"] is True
    assert any("numeric PAN output was withheld" in warning for warning in result["warnings"])
    assert len(warp_calls) == 2
    assert not list((tmp_path / "coreg").rglob("*pan_coreg.tif"))
    assert not list(temp_dir.glob("*PAN*"))


@pytest.mark.parametrize(
    ("quality", "status", "warning_fragment"),
    [
        (None, "unavailable", "is unavailable"),
        (np.zeros((2, 4), dtype=np.uint8), "shape_mismatch", "shape mismatch"),
    ],
)
def test_runtime_missing_or_mismatched_quality_is_explicitly_degraded_without_zero_masking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quality: np.ndarray | None,
    status: str,
    warning_fragment: str,
) -> None:
    result, _pan, _gcps, warp_calls, _temp_dir = _run_mocked_pan_ancillary(
        tmp_path,
        monkeypatch,
        quality,
    )

    assert result["status"] == "degraded"
    assert result["outputs"]["pan"] is not None
    assert result["outputs"]["quality_pan"] is None
    assert result["pan_quality_evidence"]["status"] == status
    assert result["pan_quality_evidence"]["numeric_mask_applied"] is False
    assert any(warning_fragment in warning for warning in result["warnings"])
    assert len(warp_calls) == 1

    with rasterio.open(result["outputs"]["pan"]) as pan_ds:
        assert pan_ds.read(1)[0, 0] == 0.0
