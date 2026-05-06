# HyperCoreg

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.5.0-green.svg)](https://github.com/AntoniogamezG/HyperCoreg-An-Automated-Optimized-Pipeline-for-Co-Registering-PRISMA-and-EnMAP-Hyperspectral-Imagery)
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
| `--normalization-mode` | `none` | Output normalization: `none`, `minmax`, or `percentile`. |
| `--metadata-stats-mode` | `approx` | Metadata statistics mode: `exact`, `approx`, or `none`. |
| `--validation-max-windows` | `64` | Final validation scan cap. Use `0` for a full scan. |
| `--batch-workers` | `1` | Number of worker processes in batch mode. |

See the full CLI reference with:

```bash
hypercoreg single --help
hypercoreg batch --help
```

### Graphical Interface

Start the GUI with:

```bash
hypercoreg-gui
```

The GUI supports single-scene and batch workflows, output directory selection,
processing presets, CDSE access, optional local Sentinel-2 stacks, and quality
control outputs.

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
