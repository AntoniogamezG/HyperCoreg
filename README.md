# HyperCoreg

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Automated hyperspectral PRISMA and EnMAP to Sentinel-2 coregistration pipeline.**

HyperCoreg is a Python tool that automatically aligns hyperspectral satellite imagery (PRISMA, EnMAP) to geometrically accurate Sentinel-2 reference data. It addresses significant geolocation errors in hyperspectral products (80-250m for PRISMA, 14-17m for EnMAP) to enable sub-pixel alignment for data fusion and multitemporal analysis.

## Features

- **Multi-sensor support**: PRISMA (.he5) and EnMAP (SPECTRAL_IMAGE GeoTIFF)
- **Automated Sentinel-2 reference selection**: Queries Copernicus Data Space Ecosystem (CDSE) for optimal reference scenes
- **Robust tie point detection**: Multi-band feature matching using AROSICS
- **Quality-driven processing**: MAD-based outlier filtering, spatial distribution analysis
- **2nd-order polynomial warping**: Handles non-linear geometric distortions
- **Auxiliary data processing**: Coregisters panchromatic bands and quality masks
- **Comprehensive outputs**: GeoTIFF with spectral metadata, ENVI headers, quality reports
- **Dual interface**: GUI for interactive use, CLI for automation/scripting

## Installation

### Recommended: Conda (handles GDAL dependencies)

```bash
# Open this folder
cd Hypercoreg_StandardUser

# Create conda environment
conda env create -f environment.yml
conda activate hypercoreg

# Install package
pip install .
```

### Alternative: pip (requires GDAL pre-installed)

```bash
# Ensure GDAL is installed system-wide or via conda first
pip install -r requirements.txt
pip install .
```

## Quick Start

### GUI Mode

```bash
python runGUI.py
```

The GUI will guide you through:
1. Selecting single file or batch mode
2. Choosing input file/folder and output directory
3. Configuring processing parameters
4. Running the coregistration

### CLI Mode

```bash
# Single file processing
python runCLI.py single -i /path/to/image.he5 -o /path/to/output

# Batch processing
python runCLI.py batch -i /path/to/input_folder -o /path/to/output

# Accuracy-first processing
python runCLI.py single -i image.he5 -o ./output --preset accuracy

# Faster production processing
python runCLI.py batch -i /path/to/input_folder -o ./output --preset fast

# With custom parameters
python runCLI.py single -i image.he5 -o ./output \
    --days-window 60 \
    --max-cloud 30 \
    --min-tie-points 15
```

### As a Python Library

```python
from hypercoreg import run_coregistration
from hypercoreg.utils import detect_hyp_type

# Detect sensor type
hyp_type = detect_hyp_type("/path/to/image.he5")

# Run coregistration
config = {
    'days_window': 30,
    'max_cloud': 20,
    'min_overlap': 0.5,
    'max_input_cloud_cover': 70.0,
}
metrics = run_coregistration("/path/to/image.he5", hyp_type, "/path/to/output", config)
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `days_window` | 30 | Temporal window for Sentinel-2 search (days) |
| `min_overlap` | 0.5 | Minimum spatial overlap with S2 (0-1) |
| `max_cloud` | 20 | Maximum cloud cover for S2 reference (%) |
| `max_input_cloud_cover` | 70 | Skip input if cloud cover exceeds (%) |
| `local_s2_stack_path` | `None` | Optional local Sentinel-2 L2A stack in canonical output order |
| `residual_threshold` | 25 | Maximum acceptable tie point residual (m) |
| `min_tie_points` | 10 | Minimum required tie points |
| `max_s2_candidates` | 3 | Maximum S2 candidates to evaluate |
| `residual_mad_factor` | 3.0 | MAD factor for outlier removal |
| `s2_ref_band` | 8 | Sentinel-2 reference band (B08 NIR) |
| `min_accuracy` | 70 | Minimum coregistration accuracy (%) |
| `max_displacement` | 350 | Maximum allowed displacement (m) |
| `normalization_mode` | `none` | Output normalization mode: `none`, `minmax`, `percentile` |
| `metadata_stats_mode` | `approx` | PAM stats strategy: `exact`, `approx`, or `none` |
| `enmap_metadata_stats_mode` | `none` | EnMAP-only PAM stats strategy override (`none` by default for faster EnMAP output) |
| `validation_max_windows` | `64` | Final validation scan cap (`0` = full scan) |
| `preset` | `default` | Named runtime preset: `default`, `fast`, or `accuracy`; explicit CLI flags override preset values |
| `pan_gcp_mode` | `map_inverse` | PAN TPS GCP mode: `map_inverse` or `scaled_image` |
| `pan_map_dxdy_source` | `auto` | Map-shift source for PAN scaled GCPs: `auto`, `xy_shift_m`, `zero` |
| `pan_target_aligned_pixels` | `False` | Use GDAL `-tap` in PAN warp |
| `pan_residual_check` | `False` | Optional post-warp PAN residual translation estimate |

By default, ancillary outputs such as PRISMA PAN and quality auxiliaries are disabled. Enable them explicitly with `save_pan`, `save_quality_mask`, `--save-pan`, or `--save-quality-mask`.

Sentinel-2 reference outputs are written as 12-band L2A spectral GeoTIFF stacks in this order: `B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12`. `B10` is not included because Sentinel-2 Level-2A products do not provide it as surface reflectance. SCL is used internally for masking and valid-pixel scoring, but it is not persisted as a band in the reference stack.

## CDSE Credentials

HyperCoreg requires Copernicus Data Space Ecosystem (CDSE) credentials to download Sentinel-2 data.

1. Register at [https://dataspace.copernicus.eu/](https://dataspace.copernicus.eu/)
2. Create OAuth client credentials in your account settings

Provide credentials via:
- **Environment variables** (recommended):
  ```bash
  export CDSE_CLIENT_ID="your-client-id"
  export CDSE_CLIENT_SECRET="your-client-secret"
  ```
- **Config file**: `~/.hypercoreg/credentials.json`
- **GUI prompt**: Enter when prompted for the first time

## Output Structure

```text
output_dir/
|- PRISMA_YYMMDD_HASH/
|  |- 00_inputs/       # Optional: pre-coreg raster when --save-pre is enabled
|  |- 01_reference/
|  |- 02_temp/
|  |- 03_coreg/        # Coregistered outputs
|  |  |- *_coreg.tif   # Coregistered hyperspectral data
|  |  |- pan/          # Optional: coregistered PAN (PRISMA)
|  |  `- quality/      # Optional: coregistered quality masks (PRISMA VNIR/SWIR)
|  |- 04_reports/      # Metrics JSON, shift reports, per-scene *_DATASET.xlsx
|  `- 05_quicklooks/   # PNG quicklooks (scene quicklook always, tiepoints optional)
|- *_DATASET.xlsx      # Fallback per-scene dataset for early skip/fail before scene folder creation
`- batch_summary.xlsx
```

## PAN Alignment Notes

If PRISMA PAN output still shows residual offset/distortion after HS coregistration, enable:

- `pan_gcp_mode="scaled_image"`: builds PAN GCP pixel/line from tiepoint image coordinates (`X_IM`,`Y_IM`) scaled by HS/PAN resolution ratio instead of inverse map transform.
- `pan_target_aligned_pixels=True`: enables GDAL `-tap` to stabilize grid alignment.

Optional diagnostics:

- `pan_residual_check=True` (with `pan_residual_threshold_px`) runs a lightweight translation estimate and warns when residual shift is still high.

## Requirements

- Python 3.9+
- GDAL 3.4+
- See `environment.yml` for full dependency list

## Citation

If you use HyperCoreg in your research, please cite (TO CHECK):

```bibtex
@software{hypercoreg,
  title = {HyperCoreg: Automated hyperspectral to Sentinel-2 coregistration},
  author = {HyperCoreg Contributors},
  year = {2024},
  url = {https://github.com/AntoniogamezG/HyperCoreg-An-optimized-hyperspectral-to-Sentinel-2-co-registration-pipeline-for-PRISMA-and-EnMAP}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [AROSICS](https://github.com/GFZ/arosics) - Automated and Robust Open-Source Image Co-Registration Software
- [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) - Sentinel-2 data access
- ASI (Italian Space Agency) - PRISMA mission
- DLR (German Aerospace Center) - EnMAP mission

