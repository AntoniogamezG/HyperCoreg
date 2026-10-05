# HyperCoreg

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.6.0-green.svg)](https://github.com/AntoniogamezG/HyperCoreg-An-Automated-Optimized-Pipeline-for-Co-Registering-PRISMA-and-EnMAP-Hyperspectral-Imagery)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-beta-orange.svg)](#project-status)

Automated co-registration pipeline for aligning PRISMA and EnMAP hyperspectral
imagery to Sentinel-2 L2A reference data.

HyperCoreg is designed for remote-sensing workflows where hyperspectral products
need reliable geometric alignment before fusion, comparison, validation, or
multitemporal analysis. It can download Sentinel-2 references from the
Copernicus Data Space Ecosystem (CDSE), or use a local Sentinel-2 stack when one
is already available.

## Contents

- [Features](#features)
- [Supported Inputs](#supported-inputs)
- [PRISMA Radiometry](#prisma-radiometry)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [CDSE Credentials](#cdse-credentials)
- [Outputs](#outputs)
- [Repository Layout](#repository-layout)
- [Python API](#python-api)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Project Status](#project-status)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Features

- PRISMA L2D `.he5` processing.
- Product-specific PRISMA L2D decoding to unitless surface reflectance, with an
  explicit native-DN compatibility mode.
- EnMAP `*-SPECTRAL_IMAGE.TIF`, `.TIFF`, and `.BSQ` processing with matching
  `*-METADATA.XML`.
- Automatic Sentinel-2 L2A search and download through CDSE.
- Optional local Sentinel-2 reference stack support.
- AROSICS-based tie point detection, outlier filtering, and polynomial warping.
- Coregistered GeoTIFF output with spectral metadata and ENVI headers.
- Quicklooks, processing metrics, scene reports, and batch summaries.
- Command-line and graphical user interfaces.

## Supported Inputs

| Sensor | Input file | Required companion files |
| --- | --- | --- |
| PRISMA | L2D `.he5` | None |
| EnMAP | `*-SPECTRAL_IMAGE.TIF`, `.TIFF`, or `.BSQ` | Matching `*-METADATA.XML` in the same product folder |
| Sentinel-2 reference | Auto-downloaded L2A scene or local stack | CDSE credentials are only needed for auto-download |

Batch mode recursively searches for PRISMA `.he5` files and EnMAP
`*-SPECTRAL_IMAGE.*` products.

## PRISMA Radiometry

PRISMA L2D samples are encoded `uint16` values. HyperCoreg now defaults to
`reflectance` mode, which applies the VNIR, SWIR, and (when exported) PAN scale
minimum and maximum stored in that exact source HE5 product. Outputs are
`float32`, unitless surface reflectance when normalization is `none`.

Use `--prisma-radiometric-mode native-dn` only when the previous encoded-value
representation is required. This compatibility mode records the source gain
and offset; it does not turn encoded samples into calibrated reflectance.

PRISMA L2 decoding is independent of `--normalization-mode`. Min-max or
percentile normalization is scene-dependent, so an output using either is
labelled `normalized_unitless`, even when decoding occurred first. See
[PRISMA L2D radiometry and migration](docs/PRISMA_RADIOMETRY.md) for the exact
formula, output metadata contract, and guidance for legacy products.

## Installation

Conda or Mamba is recommended because GDAL and geospatial dependencies are much
more reliable.

```bash
git clone https://github.com/AntoniogamezG/HyperCoreg-An-Automated-Optimized-Pipeline-for-Co-Registering-PRISMA-and-EnMAP-Hyperspectral-Imagery.git
cd HyperCoreg-An-Automated-Optimized-Pipeline-for-Co-Registering-PRISMA-and-EnMAP-Hyperspectral-Imagery

conda env create -f environment.yml
conda activate hypercoreg

python -m pip install -e .
```

Verify the installation:

```bash
hypercoreg --help
hypercoreg --version
gdalwarp --version
python -c "import hypercoreg; print(hypercoreg.__version__)"
```

Pip-only installation is possible only when GDAL is already installed and
available in the active environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Quick Start

Process one PRISMA or EnMAP scene:

```bash
hypercoreg single -i /path/to/input.he5 -o /path/to/output
```

Process a folder of scenes:

```bash
hypercoreg batch -i /path/to/input_folder -o /path/to/output
```

Open the graphical interface:

```bash
hypercoreg-gui
```

You can also run directly from the source checkout:

```bash
python runCLI.py single -i /path/to/input.he5 -o /path/to/output
python runGUI.py
```

## Usage

### Command Line

Single PRISMA scene:

```bash
hypercoreg single -i ./data/PRS_L2D_STD_20230615.he5 -o ./output
```

Single EnMAP scene:

```bash
hypercoreg single -i ./data/ENMAP01-SPECTRAL_IMAGE.TIF -o ./output
```

Batch processing:

```bash
hypercoreg batch -i ./input_scenes -o ./output --batch-workers 4
```

Use a runtime preset:

```bash
hypercoreg single -i ./data/image.he5 -o ./output --preset fast
hypercoreg single -i ./data/image.he5 -o ./output --preset accuracy
```

Use an existing local Sentinel-2 stack instead of CDSE search/download:

```bash
hypercoreg single \
  -i ./data/image.he5 \
  -o ./output \
  --local-s2-stack ./references/sentinel2_l2a_stack.tif
```

Common options:

| Option | Default | Description |
| --- | --- | --- |
| `--preset` | `default` | Runtime preset: `default`, `fast`, or `accuracy`. |
| `--days-window` | `30` | Sentinel-2 temporal search window in days. |
| `--max-cloud` | `20` | Maximum Sentinel-2 cloud cover percentage. |
| `--max-input-cloud` | `70` | Skip input scenes above this cloud cover percentage. |
| `--min-overlap` | `0.5` | Minimum required spatial overlap with Sentinel-2. |
| `--local-s2-stack` | none | Use a local Sentinel-2 stack instead of CDSE. |
| `--s2-cache-dir` | none | Persistent Sentinel-2 reference stack cache directory. |
| `--min-tie-points` | `10` | Minimum tie points required for processing. |
| `--residual-threshold` | `25` | Maximum accepted tie point residual in meters. |
| `--max-s2-candidates` | `3` | Number of Sentinel-2 candidates to evaluate. |
| `--s2-band` | `8` | Sentinel-2 reference band: `2`, `3`, `4`, or `8`. |
| `--prisma-radiometric-mode` | `reflectance` | PRISMA L2D radiometry: product-decoded `reflectance` or legacy `native-dn`. |
| `--normalization-mode` | `none` | Output normalization: `none`, `minmax`, or `percentile`. |
| `--metadata-stats-mode` | `approx` | Metadata statistics mode: `exact`, `approx`, or `none`. |
| `--validation-max-windows` | `64` | Final validation scan cap. Use `0` for a full scan. |
| `--batch-workers` | `1` | Number of worker processes in batch mode. |

See the full CLI reference with:

```bash
hypercoreg single --help
hypercoreg batch --help
```

### Quality and speed options

These are on by default. Each has a switch to restore the previous behaviour.

| Behaviour | Default | Switch / config key |
| --- | --- | --- |
| Download only the Sentinel-2 band files the stack uses (OData Nodes), falling back to the full ZIP | on | `--no-band-only-download` / `s2_band_only_download` |
| Keep downloaded S2 products in a shared, size-capped cache (`~/.cache/hypercoreg/s2_products`) | on, 20 GB | `--no-s2-product-cache`, `--s2-product-cache-dir`, `--s2-product-cache-max-gb` |
| Re-rank the best S2 candidates by cloud cover over the scene footprint (SCL band) | on | `--no-footprint-cloud-screen` / `s2_footprint_cloud_screen` |
| Candidate ranking weighs temporal distance in days (1.5 per day vs 1 per % cloud) | on | `s2_rank_days_weight`, `s2_rank_cloud_weight`, `s2_rank_overlap_weight` |
| Blur the S2 matching reference to the HS sensor's PSF and mask clouds, shadows, water (SCL) and HS-detected water (NDWI) | on | `--psf-fwhm-factor 0`, `matching_exclude_scl_classes`, `matching_mask_hs_water` |
| Override AROSICS' internal resampling (e.g. `average`); AROSICS' recommended cubic is kept by default | off | `arosics_resamp_alg_calc` |
| Per-band tie-point weighting; drop S2 bands with median reliability below 30 % | on | `band_min_median_reliability`, `band_weight_score` |
| Choose affine / order-2 / TPS by held-out tie-point error | `cv` | `--transform-model rule_based` |
| Report independent check-point accuracy (`checkpoint_rmse_m`, `checkpoint_p90_m` in metrics and batch summary) | 20 % held out | `checkpoint_holdout_fraction` |
| Run the per-band local matches in parallel processes | on | `--no-parallel-bands` / `local_band_parallel` |
| Match PRISMA PAN against a synthetic PAN band from the coregistered HS cube | `hs` | `--pan-reference s2` |
| ZSTD (or DEFLATE) + predictor compression for intermediate GeoTIFFs | on | `HYPERCOREG_GTIFF_COMPRESS=LZW` |

### Graphical Interface

Start the GUI with:

```bash
hypercoreg-gui
```

The GUI supports single-scene and batch workflows, output directory selection,
processing presets, CDSE access, optional local Sentinel-2 stacks, and quality
control outputs. For PRISMA input, choose `reflectance` or `native-dn` under
**PRISMA radiometry**; normalization remains a separate control.

## CDSE Credentials

HyperCoreg uses CDSE credentials only when it needs to search and download
Sentinel-2 L2A reference data. Create an account at
[dataspace.copernicus.eu](https://dataspace.copernicus.eu/) before using remote
Sentinel-2 processing.

## Outputs

HyperCoreg writes one folder per processed scene:

```text
output_dir/
|- PRISMA_YYMMDD_HASH/
|  |- 00_inputs/       # Optional pre-coregistration raster
|  |- 01_reference/    # Sentinel-2 reference products
|  |- 02_temp/         # Temporary processing files
|  |- 03_coreg/        # Coregistered GeoTIFF outputs
|  |- 04_reports/      # Metrics, logs, reports, and spreadsheets
|  `- 05_quicklooks/   # PNG quicklooks and optional tie point plots
|- *_DATASET.xlsx      # Fallback per-scene report for early skip/fail cases
`- batch_summary.xlsx  # Batch-level status and metrics
```

The main outputs are:

- `03_coreg/*_coreg.tif`: coregistered hyperspectral image.
- `04_reports/`: processing metrics and quality reports.
- `05_quicklooks/`: visual quality checks.
- `batch_summary.xlsx`: batch-level summary.

PRISMA GeoTIFFs declare the radiometric quantity, units, decoding state, source
coefficients, and detector-specific gain/offset in dataset and per-band
metadata. The same radiometric contract is also recorded in each scene's
metrics JSON and run manifest. Exported PAN quality is an aligned categorical
mask: source flag `4` and pixels outside the warped footprint remain PAN
NoData, while valid zero-DN pixels remain valid.

## Repository Layout

```text
.
|- hypercoreg/
|  |- cli.py                 # Command-line interface
|  |- gui.py                 # Graphical interface
|  |- coregistration.py      # Public processing entry points
|  |- readers/               # PRISMA and EnMAP readers
|  |- pipeline/              # Pipeline orchestration and processing modules
|  `- data/                  # Bundled spectral response/band tables
|- docs/                     # Additional project documentation
|- environment.yml           # Recommended Conda environment
|- requirements.txt          # Pip dependency fallback
|- pyproject.toml            # Package metadata and console scripts
|- runCLI.py                 # Source checkout CLI launcher
|- runGUI.py                 # Source checkout GUI launcher
`- README.md
```

## Python API

```python
from hypercoreg import run_coregistration
from hypercoreg.utils import detect_hyp_type

input_path = "/path/to/image.he5"
output_dir = "/path/to/output"

hyp_type = detect_hyp_type(input_path)

config = {
    "days_window": 30,
    "max_cloud": 20,
    "min_overlap": 0.5,
    "max_input_cloud_cover": 70.0,
    "prisma_radiometric_mode": "reflectance",
    "normalization_mode": "none",
}

metrics = run_coregistration(input_path, hyp_type, output_dir, config)
print(metrics)
```

## Troubleshooting

### `gdalwarp` is not found

Activate the Conda environment and verify GDAL:

```bash
conda activate hypercoreg
gdalwarp --version
```

If this fails, recreate or update the environment from `environment.yml`.

### CDSE authentication fails

Check that credentials are available in the active shell:

```bash
echo $CDSE_USERNAME
echo $CDSE_PASSWORD
```

PowerShell:

```powershell
echo $env:CDSE_USERNAME
echo $env:CDSE_PASSWORD
```

You can avoid remote CDSE access by providing `--local-s2-stack`.

### Batch mode finds no files

Confirm that the input folder contains supported files. For EnMAP, the spectral
image filename must contain `SPECTRAL_IMAGE`, and the matching metadata XML must
be available in the same product folder.

### Processing is slow

Use the fast preset for initial checks:

```bash
hypercoreg batch -i ./input_scenes -o ./output --preset fast
```

## Development

Install the package in editable mode and run the test suite from the repository
root:

```bash
python -m pip install -e .
python -m pip install pytest
python -m pytest
```

## Project Status

HyperCoreg is an experimental research pipeline. The public interfaces are usable, but
processing behavior and output metadata may still evolve as the pipeline is
validated across more PRISMA and EnMAP scenes.

## Citation

If you use HyperCoreg in research, please cite:

```bibtex
@software{hypercoreg,
  title = {HyperCoreg: Automated hyperspectral to Sentinel-2 coregistration},
  author = {HyperCoreg Contributors},
  year = {2024},
  url = {https://github.com/AntoniogamezG/HyperCoreg-An-Automated-Optimized-Pipeline-for-Co-Registering-PRISMA-and-EnMAP-Hyperspectral-Imagery}
}
```

## Acknowledgments

- ASI, Italian Space Agency, for the PRISMA mission.
- DLR, German Aerospace Center, for the EnMAP mission.
- [AROSICS](https://github.com/GFZ/arosics) for automated image co-registration.
- [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) for
  Sentinel-2 data access.


## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE)
for details.
