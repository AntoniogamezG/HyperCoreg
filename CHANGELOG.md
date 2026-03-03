# Changelog

All notable changes to HyperCoreg will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2024-XX-XX

### Added
- Initial public release
- Support for PRISMA L2D hyperspectral data (.he5 files)
- Support for EnMAP L2A hyperspectral data (SPECTRAL_IMAGE GeoTIFF)
- Automated Sentinel-2 reference selection via CDSE API
- Multi-band tie point detection using AROSICS
- 2nd-order polynomial geometric correction
- Panchromatic band coregistration (PRISMA)
- Quality mask and cloud mask coregistration
- GUI interface for interactive processing
- CLI interface for scripting and automation
- Batch processing mode
- Comprehensive quality metrics and reports

### Changed
- Refactored from single-file script to Python package structure
- Improved credential handling (environment variables, config file)
- Cross-platform GDAL configuration

### Fixed
- N/A (initial release)

## [Unreleased]

### Added
- PRISMA PAN ancillary option `pan_gcp_mode` with `scaled_image` strategy that builds GCP pixel/line from tiepoint image-space coordinates (`X_IM`,`Y_IM`) scaled by HS/PAN resolution ratio.
- PAN ancillary controls:
  - `pan_map_dxdy_source` (`auto|xy_shift_m|zero`)
  - `pan_target_aligned_pixels` (GDAL `-tap`)
  - `pan_residual_check`, `pan_residual_threshold_px`, `pan_residual_max_dim`
- Lightweight post-warp PAN residual translation diagnostic (phase-correlation based, optional).

### Changed
- PAN TPS warp command builder now supports target-aligned pixels (`-tap`) when requested.
- CLI now exposes PAN ancillary alignment options for non-GUI workflows.

### Planned
- Unit tests
- GitHub Actions CI/CD
- Documentation website
- Example Jupyter notebooks
