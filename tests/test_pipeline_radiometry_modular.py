import inspect
import json

import pytest

from hypercoreg.pipeline import reporting


RADIOMETRIC_CONTRACT = {
    "mode": "reflectance",
    "quantity": "surface_reflectance",
    "l2_scaling_applied": True,
    "formula": "MIN + DN * (MAX - MIN) / 65535",
    "scales": {
        "VNIR": {"minimum": 0.1, "maximum": 0.5},
        "SWIR": {"minimum": -0.2, "maximum": 0.8},
        "PAN": {"minimum": 0.0, "maximum": 1.0},
    },
}

EXPECTED_SCALE_FIELDS = {
    "prisma_l2_scale_vnir_min": 0.1,
    "prisma_l2_scale_vnir_max": 0.5,
    "prisma_l2_scale_swir_min": -0.2,
    "prisma_l2_scale_swir_max": 0.8,
    "prisma_l2_scale_pan_min": 0.0,
    "prisma_l2_scale_pan_max": 1.0,
}


def test_reporting_schema_carries_prisma_radiometry_columns():
    expected = {
        "prisma_radiometric_mode",
        "radiometric_quantity",
        "prisma_l2_scaling_applied",
        "prisma_l2_scale_vnir_min",
        "prisma_l2_scale_vnir_max",
        "prisma_l2_scale_swir_min",
        "prisma_l2_scale_swir_max",
        "prisma_l2_scale_pan_min",
        "prisma_l2_scale_pan_max",
    }
    assert expected.issubset(reporting.DATASET_XLSX_COLUMNS)
    assert expected.issubset(reporting.BATCH_SUMMARY_XLSX_COLUMNS)


def test_failed_metrics_preserve_contract_and_old_call_shape():
    legacy = reporting._build_failed_scene_metrics("scene.he5", "PRISMA", "failed")
    assert legacy["prisma_radiometric_mode"] is None
    assert legacy["radiometric"] == {}

    metrics = reporting._build_failed_scene_metrics(
        "scene.he5",
        "PRISMA",
        "failed",
        radiometric_contract=RADIOMETRIC_CONTRACT,
    )
    assert metrics["prisma_radiometric_mode"] == "reflectance"
    assert metrics["radiometric_quantity"] == "surface_reflectance"
    assert metrics["prisma_l2_scaling_applied"] is True
    assert metrics["radiometric"] == RADIOMETRIC_CONTRACT
    assert {key: metrics[key] for key in EXPECTED_SCALE_FIELDS} == EXPECTED_SCALE_FIELDS
    assert {
        key: reporting._build_dataset_row(metrics)[key] for key in EXPECTED_SCALE_FIELDS
    } == EXPECTED_SCALE_FIELDS


def test_skip_metrics_and_manifest_preserve_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, "_write_single_scene_dataset_xlsx", lambda *_: True)

    metrics = reporting._build_skip_result(
        hs_file="scene.he5",
        hyp_type="PRISMA",
        output_dir=str(tmp_path),
        scene_idx=1,
        scene_total=1,
        hs_time=None,
        bbox=None,
        config={"prisma_radiometric_mode": "reflectance"},
        status="SKIP",
        reason="cloud threshold",
        normalization_mode="none",
        normalization_params={},
        build_overviews=False,
        remove_detector_overlap_bands=False,
        strict_metadata=True,
        metadata_extension_level="basic",
        metadata_stats_mode="none",
        metadata_stats_sample_windows=1,
        metadata_stats_seed=0,
        metadata_histogram_buckets=16,
        metadata_label_precision=2,
        validation_max_windows=1,
        radiometric_contract=RADIOMETRIC_CONTRACT,
    )

    assert metrics["prisma_radiometric_mode"] == "reflectance"
    assert metrics["radiometric_quantity"] == "surface_reflectance"
    assert metrics["prisma_l2_scaling_applied"] is True
    assert metrics["radiometric"] == RADIOMETRIC_CONTRACT
    assert {key: metrics[key] for key in EXPECTED_SCALE_FIELDS} == EXPECTED_SCALE_FIELDS

    with open(metrics["run_manifest_path"], "r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    assert manifest["processing"]["prisma_radiometric_mode"] == "reflectance"
    assert manifest["processing"]["radiometric"] == RADIOMETRIC_CONTRACT


def test_raster_io_exports_canonical_radiometric_paths():
    pytest.importorskip("arosics")
    from hypercoreg.pipeline import raster_io, runtime

    assert raster_io._coregister_prisma_ancillary_outputs is runtime._coregister_prisma_ancillary_outputs
    assert raster_io._save_precoreg_output is runtime._save_precoreg_output
    assert raster_io._finalize_coreg_output is runtime._finalize_coreg_output
    assert raster_io._stream_copy_raster_with_band_order is runtime._stream_copy_raster_with_band_order
    assert raster_io._write_envi_header is runtime._write_envi_header
    assert "radiometric_contract" in inspect.signature(
        raster_io._coregister_prisma_ancillary_outputs
    ).parameters


def test_runtime_skip_result_keeps_legacy_optional_positional_order():
    pytest.importorskip("arosics")
    from hypercoreg.pipeline import runtime

    parameters = list(inspect.signature(runtime._build_skip_result).parameters)
    assert parameters[-3:] == ["extra_summary", "extra_metrics", "radiometric_contract"]
