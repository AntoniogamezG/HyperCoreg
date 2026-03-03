# HyperCoreg

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Automated hyperspectral coregistration pipeline for PRISMA and EnMAP scenes against Sentinel-2 reference data.

HyperCoreg is a Python package plus CLI/GUI wrappers that aligns hyperspectral imagery to Sentinel-2, writes coregistered outputs, and produces per-scene QA reports/manifests.

## Repository Layout

- `hypercoreg/`: package source (pipeline, readers, CLI/GUI entry points)
- `runCLI.py`: source-tree launcher for CLI
- `runGUI.py`: source-tree launcher for GUI
- `environment.yml`: conda environment (recommended)
- `requirements.txt`: pip requirements (GDAL must already be available)
- `docs/`: project notes and audit docs

## Current Capabilities

- PRISMA L2D input (`.he5`)
- EnMAP L2A spectral input (`*-SPECTRAL_IMAGE.tif/.tiff/.bsq`)
- Single-scene and batch processing
- CLI (`hypercoreg`) and GUI (`hypercoreg-gui`) workflows
- Automatic Sentinel-2 search/download from CDSE
- Optional local Sentinel-2 stack override (`--local-s2-stack`)
- Multi-band tie-point collection with spatial stratification and consensus filtering
- Polynomial warp safety controls (preferred order with auto-downgrade safeguards)
- Optional post-warp phase-correlation QA checks
- PRISMA ancillary products: PAN and quality masks
- Scene quicklooks, metrics JSON, run manifest JSON, and per-scene `*_DATASET.xlsx`

## Installation

### Recommended (Conda)

```bash
git clone https://github.com/AntoniogamezG/HyperCoreg-An-optimized-hyperspectral-to-Sentinel-2-co-registration-pipeline-for-PRISMA-and-EnMAP.git Hypercoreg
cd Hypercoreg

conda env create -f environment.yml
conda activate hypercoreg
pip install -e .
```

### Alternative (pip, GDAL pre-installed)

```bash
pip install -r requirements.txt
pip install -e .
```

## Run HyperCoreg

### CLI

After installation:

```bash
hypercoreg --help
```

From a source checkout (without script entry points):

```bash
python runCLI.py --help
```

Examples:

```bash
# Single scene
hypercoreg single -i /path/to/PRS_scene.he5 -o /path/to/output

# Batch mode (recursive scan)
hypercoreg batch -i /path/to/input_folder -o /path/to/output

# Use a local Sentinel-2 stack instead of CDSE download
hypercoreg single -i /path/to/ENMAP-SPECTRAL_IMAGE.TIF -o /path/to/output \
  --local-s2-stack /path/to/S2_stack_6bands.tif

# Enable optional QA diagnostics and PAN alignment safeguards
hypercoreg single -i /path/to/PRS_scene.he5 -o /path/to/output \
  --postwarp-phasecorr-check \
  --pan-gcp-mode scaled_image \
  --pan-target-aligned-pixels \
  --pan-residual-check
```

### GUI

After installation:

```bash
hypercoreg-gui
```

From source:

```bash
python runGUI.py
```

## Input Requirements

- PRISMA: `.he5`
- EnMAP: spectral raster with `SPECTRAL_IMAGE` in filename (`.tif/.tiff/.bsq`)
- EnMAP metadata XML (`*-METADATA.XML`) must be available in the scene folder
- For EnMAP `.bsq`, an ENVI header sidecar (`.hdr`) is required

## CDSE Authentication

If you do not use `--local-s2-stack`, HyperCoreg needs CDSE access to retrieve Sentinel-2 scenes.

Supported credential methods (checked in this order):

1. `CDSE_ACCESS_TOKEN`
2. `CDSE_CLIENT_ID` + `CDSE_CLIENT_SECRET`
3. `CDSE_USERNAME` + `CDSE_PASSWORD` (+ optional `CDSE_TOTP`)

Notes:

- In GUI mode, missing credentials can be requested interactively via dialog.
- In CLI mode, missing credentials can be requested in terminal prompt.

## Common CLI Parameters

`hypercoreg --help` shows the full list. Commonly tuned options:

| Option | Default | Description |
|---|---|---|
| `--days-window` | `30` | Temporal search window for Sentinel-2 candidates (days) |
| `--min-overlap` | `0.5` | Minimum overlap with Sentinel-2 scene |
| `--max-cloud` | `20` | Max Sentinel-2 cloud cover (%) |
| `--max-input-cloud` | `70` | Skip input scene when cloud cover exceeds threshold (%) |
| `--local-s2-stack` | `None` | Use prebuilt local Sentinel-2 stack (skip CDSE) |
| `--min-tie-points` | `10` | Minimum tie points required |
| `--mad-factor` | `3.0` | MAD factor for tie-point outlier filtering |
| `--prefer-fixed-band-pairs` | `True` | Use curated PRISMA to Sentinel-2 band pairs first |
| `--min-band-support` | `2` | Minimum distinct band matches for consensus |
| `--preferred-polynomial-order` | `2` | Preferred warp polynomial order (`1` or `2`) |
| `--auto-downgrade-polynomial-order` | `True` | Downgrade polynomial order when geometry support is weak |
| `--postwarp-phasecorr-check` | `False` | Optional post-warp QA via phase correlation |
| `--normalization-mode` | `none` | Output normalization: `none`, `minmax`, `percentile` |
| `--metadata-extension` | `stats` | Metadata sidecar level: `none`, `stats`, `full` |
| `--metadata-stats-mode` | `exact` | Metadata stats mode: `exact`, `approx`, `none` |
| `--enmap-metadata-stats-mode` | `none` | EnMAP-only metadata stats override |
| `--validation-max-windows` | `0` | Validation scan cap (`0` means full scan) |
| `--pan-gcp-mode` | `map_inverse` | PRISMA PAN GCP strategy: `map_inverse` or `scaled_image` |
| `--pan-target-aligned-pixels` | `False` | Enable GDAL `-tap` for PAN warp |
| `--pan-residual-check` | `False` | Optional PAN post-warp residual check |

## Output Structure

```text
output_dir/
|- coreg_processing_YYYYMMDD_HHMMSS_xxxx.log
|- PRISMA_YYMMDD_HASH/ or ENMAP_YYMMDD_HASH/
|  |- 00_inputs/                 # Optional pre-coreg rasters
|  |- 01_reference/              # Sentinel-2 stack used for matching
|  |- 02_temp/                   # Intermediate products
|  |- 03_coreg/
|  |  |- *_coreg.tif             # Core hyperspectral output
|  |  |- pan/                    # PRISMA PAN output (optional)
|  |  `- quality/                # PRISMA quality masks (optional)
|  |- 04_reports/
|  |  |- *_metrics.json
|  |  |- *_shift_report.txt
|  |  |- *_run_manifest.json
|  |  |- *_DATASET.xlsx
|  |  `- *_displacement_vectors.*  # Optional shapefile set
|  `- 05_quicklooks/
|     |- *_quicklook.png
|     `- *_tiepoints.png         # Only when enabled
|- *_run_manifest.json           # Fallback manifest for early skipped scenes
|- *_DATASET.xlsx                # Fallback for scenes that fail early
|- batch_summary.txt
`- batch_summary.xlsx
```

## Python API

```python
from hypercoreg import run_coregistration, run_batch_coregistration
from hypercoreg.utils import detect_hyp_type

config = {
    "days_window": 30,
    "max_cloud": 20,
    "min_overlap": 0.5,
    "local_s2_stack_path": None,
}

single_input = "/path/to/PRS_scene.he5"
hyp_type = detect_hyp_type(single_input)
metrics = run_coregistration(single_input, hyp_type, "/path/to/output", config)

batch_results = run_batch_coregistration("/path/to/input_folder", "/path/to/output", config)
```

## Requirements

- Python `>=3.9` (`environment.yml` targets `<3.12`)
- GDAL `>=3.4`
- Dependencies in `environment.yml` / `requirements.txt`

## Citation

If you use HyperCoreg in academic work, please cite the software repository:

```bibtex
@software{hypercoreg,
  title = {HyperCoreg: Automated hyperspectral to Sentinel-2 coregistration},
  author = {HyperCoreg Contributors},
  url = {https://github.com/AntoniogamezG/HyperCoreg-An-optimized-hyperspectral-to-Sentinel-2-co-registration-pipeline-for-PRISMA-and-EnMAP}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Acknowledgments

- [AROSICS](https://github.com/GFZ/arosics)
- [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/)
- ASI (PRISMA mission)
- DLR (EnMAP mission)

