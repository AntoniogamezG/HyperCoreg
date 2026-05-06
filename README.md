# HyperCoreg

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.5.0-green.svg)](https://github.com/AntoniogamezG/HyperCoreg-An-Automated-Optimized-Pipeline-for-Co-Registering-PRISMA-and-EnMAP-Hyperspectral-Imagery)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Automated hyperspectral PRISMA and EnMAP to Sentinel-2 coregistration pipeline.**

HyperCoreg aligns hyperspectral satellite imagery from PRISMA and EnMAP to geometrically accurate Sentinel-2 reference data. It is designed for scenes where hyperspectral products have meaningful geolocation errors and need sub-pixel alignment before data fusion, comparison, or multitemporal analysis.

## What HyperCoreg Does

- Supports PRISMA L2D `.he5` products.
- Supports EnMAP `*-SPECTRAL_IMAGE.TIF`, `.TIFF`, and `.BSQ` products when the matching `*-METADATA.XML` file is available.
- Searches Copernicus Data Space Ecosystem (CDSE) for Sentinel-2 L2A reference scenes, or uses a local Sentinel-2 stack when provided.
- Detects tie points with AROSICS, filters outliers, and applies polynomial warping.
- Writes coregistered GeoTIFF outputs, spectral metadata, ENVI headers, quicklooks, and processing reports.
- Provides both a graphical interface and a command-line interface.

## Before You Start

You need:

- A working Conda or Mamba installation. This is strongly recommended because GDAL is difficult to install reliably with plain pip.
- Python 3.9, 3.10, or 3.11.
- PRISMA or EnMAP input data.
- CDSE credentials if you want HyperCoreg to download Sentinel-2 references automatically.
- Enough disk space for temporary Sentinel-2 downloads, intermediate rasters, and output products.

If you already have a local Sentinel-2 L2A reference stack, you can skip CDSE credentials by using `--local-s2-stack`.

## Installation

### 1. Clone And Enter The Repository

```bash
git clone https://github.com/AntoniogamezG/HyperCoreg-An-Automated-Optimized-Pipeline-for-Co-Registering-PRISMA-and-EnMAP-Hyperspectral-Imagery.git
cd HyperCoreg-An-Automated-Optimized-Pipeline-for-Co-Registering-PRISMA-and-EnMAP-Hyperspectral-Imagery
```

If you downloaded the repository as a ZIP file, extract it and open a terminal in the extracted folder.

### 2. Create The Conda Environment

```bash
conda env create -f environment.yml
conda activate hypercoreg
```

If the environment already exists and you want to refresh it:

```bash
conda env update -f environment.yml --prune
conda activate hypercoreg
```

### 3. Install HyperCoreg

For normal use from this source checkout, install it in editable mode:

```bash
python -m pip install -e .
```

Editable mode keeps the installed command linked to this folder. If the code changes later, you do not need to reinstall after every edit.

### 4. Verify The Installation

Run these commands from the activated `hypercoreg` environment:

```bash
hypercoreg --help
gdalwarp --version
python -c "import hypercoreg; print(hypercoreg.__version__)"
```

You should see the HyperCoreg command help, a GDAL version, and the installed package version.

### Pip-Only Installation

Use this only if GDAL is already installed and available in your environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

If `rasterio`, `geopandas`, `arosics`, or `gdalwarp` fail to install or import, use the Conda installation above.

## CDSE Credentials

HyperCoreg can automatically search and download Sentinel-2 L2A reference scenes from the Copernicus Data Space Ecosystem. Create an account at [https://dataspace.copernicus.eu/](https://dataspace.copernicus.eu/) before running remote Sentinel-2 processing.

The easiest option for most users is to set credentials as environment variables before running HyperCoreg.

PowerShell:

```powershell
$env:CDSE_USERNAME="your-cdse-username"
$env:CDSE_PASSWORD="your-cdse-password"
```

Bash or macOS/Linux terminal:

```bash
export CDSE_USERNAME="your-cdse-username"
export CDSE_PASSWORD="your-cdse-password"
```

If your account uses a time-based one-time password, also set:

```bash
export CDSE_TOTP="123456"
```

HyperCoreg also supports CDSE OAuth client credentials:

```bash
export CDSE_CLIENT_ID="your-client-id"
export CDSE_CLIENT_SECRET="your-client-secret"
```

For a persistent local setup, create `~/.hypercoreg/credentials.json`:

```json
{
  "cdse": {
    "username": "your-cdse-username",
    "password": "your-cdse-password"
  }
}
```

or:

```json
{
  "cdse": {
    "client_id": "your-client-id",
    "client_secret": "your-client-secret"
  }
}
```

You can store the file somewhere else by setting `HYPERCOREG_CREDENTIALS_FILE` to the full path.

## Input Data

### PRISMA

Use the PRISMA L2D `.he5` file directly:

```text
PRS_L2D_STD_20230615_....he5
```

### EnMAP

Use the EnMAP spectral image file. The matching metadata XML must be in the same product folder so HyperCoreg can detect and read the scene:

```text
ENMAP01-....-SPECTRAL_IMAGE.TIF
ENMAP01-....-METADATA.XML
```

Supported spectral image extensions are `.TIF`, `.TIFF`, and `.BSQ`.

### Batch Folders

Batch mode searches recursively for:

```text
*.he5
*.HE5
*-SPECTRAL_IMAGE.tif
*-SPECTRAL_IMAGE.TIF
*-SPECTRAL_IMAGE.tiff
*-SPECTRAL_IMAGE.TIFF
*-SPECTRAL_IMAGE.bsq
*-SPECTRAL_IMAGE.BSQ
```

## Running HyperCoreg

### Option A: Graphical Interface

Start the GUI:

```bash
hypercoreg-gui
```

If you are running directly from the source folder without using the installed command:

```bash
python runGUI.py
```

The GUI workflow is:

1. Select single-file mode or batch mode.
2. Choose the PRISMA/EnMAP input file or input folder.
3. Choose the output directory.
4. Adjust processing options if needed.
5. Run the coregistration and inspect the output folder.

### Option B: Command Line

Single PRISMA scene:

```bash
hypercoreg single -i /path/to/PRS_L2D_STD_20230615.he5 -o /path/to/output
```

Single EnMAP scene:

```bash
hypercoreg single -i /path/to/ENMAP01-SPECTRAL_IMAGE.TIF -o /path/to/output
```

Batch processing:

```bash
hypercoreg batch -i /path/to/input_folder -o /path/to/output
```

Use `python runCLI.py` instead of `hypercoreg` if you prefer running from the repository folder:

```bash
python runCLI.py single -i /path/to/image.he5 -o /path/to/output
```

## Recommended Execution Recipes

### Default Run

Use this first when you are processing a normal scene and want balanced defaults:

```bash
hypercoreg single -i image.he5 -o ./output
```

### Faster Run

Use this for quicker batch checks or production runs where you want fewer diagnostics:

```bash
hypercoreg batch -i ./input_scenes -o ./output --preset fast
```

### Accuracy-First Run

Use this when quality is more important than runtime:

```bash
hypercoreg single -i image.he5 -o ./output --preset accuracy
```

### Run With A Local Sentinel-2 Stack

Use this when you already have a Sentinel-2 L2A reference stack and do not want HyperCoreg to search or download from CDSE:

```bash
hypercoreg single -i image.he5 -o ./output --local-s2-stack /path/to/s2_l2a_stack.tif
```

The expected Sentinel-2 reflectance band order is:

```text
B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
```

Optional trailing ancillary bands are accepted in this order:

```text
SCL, AOT, WVP
```

### Batch Run With Multiple Workers

```bash
hypercoreg batch -i ./input_scenes -o ./output --batch-workers 4
```

For multi-worker batch runs, set CDSE credentials with environment variables or `~/.hypercoreg/credentials.json` before starting. Interactive credential prompts are not suitable for parallel batch processing.

### Useful Output Options

```bash
hypercoreg single -i image.he5 -o ./output \
  --save-pre \
  --save-pan \
  --save-quality-mask \
  --gen-tiepoint-pngs \
  --build-overviews
```

Notes:

- `--save-pan` applies to PRISMA panchromatic output.
- `--save-quality-mask` saves PRISMA masks and EnMAP quicklook auxiliaries when available.
- `--build-overviews` can make large GeoTIFFs easier to open in GIS software.
- `--gen-tiepoint-pngs` is useful for quality control but adds runtime.

## Common Parameters

| Option | Default | What It Controls |
|--------|---------|------------------|
| `--preset` | `default` | Runtime preset: `default`, `fast`, or `accuracy`. |
| `--days-window` | `30` | Sentinel-2 temporal search window in days. |
| `--max-cloud` | `20` | Maximum Sentinel-2 cloud cover percentage. |
| `--max-input-cloud` | `70` | Skip hyperspectral input scenes above this cloud percentage. |
| `--min-overlap` | `0.5` | Minimum required spatial overlap with Sentinel-2. |
| `--local-s2-stack` | none | Use a local Sentinel-2 stack instead of CDSE download. |
| `--s2-cache-dir` | none | Directory for persistent Sentinel-2 reference stack cache. |
| `--min-tie-points` | `10` | Minimum tie points required for processing. |
| `--residual-threshold` | `25` | Maximum accepted tie point residual in meters. |
| `--max-s2-candidates` | `3` | Number of Sentinel-2 candidates to evaluate. |
| `--s2-band` | `8` | Sentinel-2 reference band: `2`, `3`, `4`, or `8`. |
| `--normalization-mode` | `none` | Output normalization: `none`, `minmax`, or `percentile`. |
| `--metadata-stats-mode` | `approx` | Metadata statistics mode: `exact`, `approx`, or `none`. |
| `--validation-max-windows` | `64` | Final validation scan cap. Use `0` for full scan. |
| `--batch-workers` | `1` | Number of worker processes in batch mode. |

See all available options with:

```bash
hypercoreg single --help
hypercoreg batch --help
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

## Output Structure

HyperCoreg creates one scene folder per processed scene:

```text
output_dir/
|- PRISMA_YYMMDD_HASH/
|  |- 00_inputs/       # Optional pre-coreg raster when --save-pre is enabled
|  |- 01_reference/    # Sentinel-2 reference products
|  |- 02_temp/         # Temporary processing files
|  |- 03_coreg/        # Coregistered outputs
|  |  |- *_coreg.tif
|  |  |- pan/          # Optional PRISMA PAN output
|  |  `- quality/      # Optional quality auxiliaries
|  |- 04_reports/      # Metrics JSON, reports, and per-scene *_DATASET.xlsx
|  `- 05_quicklooks/   # PNG quicklooks and optional tie point plots
|- *_DATASET.xlsx      # Fallback per-scene dataset for early skip/fail cases
`- batch_summary.xlsx  # Batch summary when batch mode is used
```

The most important outputs for users are usually:

- `03_coreg/*_coreg.tif`: the coregistered hyperspectral image.
- `04_reports/`: processing metrics and quality reports.
- `05_quicklooks/`: visual checks of the scene and optional tie points.
- `batch_summary.xlsx`: batch-level status and metrics.

## Troubleshooting

### `gdalwarp` Is Not Found

Activate the Conda environment and check GDAL:

```bash
conda activate hypercoreg
gdalwarp --version
```

If that fails, recreate or update the environment from `environment.yml`.

### CDSE Authentication Fails

Check that both parts of the credential pair are set:

```bash
echo $CDSE_USERNAME
echo $CDSE_PASSWORD
```

In PowerShell:

```powershell
echo $env:CDSE_USERNAME
echo $env:CDSE_PASSWORD
```

You can also avoid remote CDSE access by providing `--local-s2-stack`.

### Batch Mode Finds No Files

Confirm that the input folder contains supported files. For EnMAP, the spectral image filename must contain `SPECTRAL_IMAGE`, and the matching metadata XML must be available.

### EnMAP Detection Fails

Make sure the EnMAP spectral image and metadata file are from the same product and are kept together. HyperCoreg expects `SPECTRAL_IMAGE.TIF`, `.TIFF`, or `.BSQ` with a matching `-METADATA.XML`.

### Processing Is Slow

Try:

```bash
hypercoreg batch -i ./input_scenes -o ./output --preset fast
```

For batch processing, increase `--batch-workers` carefully. Large scenes and multiple workers can use substantial CPU, memory, and disk I/O.

### Outputs Are Large Or Slow To Open

Add internal overviews:

```bash
hypercoreg single -i image.he5 -o ./output --build-overviews
```

## Requirements

The primary dependency list is in `environment.yml`. Key dependencies include:

- Python 3.9 to 3.11
- GDAL 3.4+
- rasterio
- geopandas
- h5py
- AROSICS
- geoarray
- numpy, scipy, pandas, matplotlib, openpyxl

## Citation

If you use HyperCoreg in your research, please cite:

```bibtex
@software{hypercoreg,
  title = {HyperCoreg: Automated hyperspectral to Sentinel-2 coregistration},
  author = {HyperCoreg Contributors},
  year = {2024},
  url = {https://github.com/AntoniogamezG/HyperCoreg-An-Automated-Optimized-Pipeline-for-Co-Registering-PRISMA-and-EnMAP-Hyperspectral-Imagery}
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [AROSICS](https://github.com/GFZ/arosics): Automated and Robust Open-Source Image Co-Registration Software.
- [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/): Sentinel-2 data access.
- ASI, Italian Space Agency: PRISMA mission.
- DLR, German Aerospace Center: EnMAP mission.
