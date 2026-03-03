# Legacy Audit Report (2026-03-03)

## Scope
- In scope: `hypercoreg/`, `runCLI.py`, `runGUI.py`, `README.md`, `docs/`
- Out of scope for final findings: `tests/` (except behavior notes)
- Marker definition used: explicit `legacy`, `deprecated`, `backward`, `backwards`, `compat mode`

## Method
- Primary marker scan:
  - `rg -n --hidden -S "legacy|deprecated|backward|backwards|compat mode" hypercoreg runCLI.py runGUI.py README.md docs`
- Legacy wrapper usage scan:
  - `rg -n "\b_merge_tiepoints_max_reliability\b" hypercoreg runCLI.py runGUI.py README.md docs`
  - `rg -n "\b_apply_polynomial_warp_order2\b" hypercoreg runCLI.py runGUI.py README.md docs`
- Alias behavior scan:
  - `rg -n "spectral_normalization_01" hypercoreg runCLI.py runGUI.py README.md docs`
  - `rg -n "pan_gcp_mode|map_inverse" hypercoreg runCLI.py runGUI.py README.md docs`

## A. Executive Summary
- Explicit-marker hits: `22` lines across `5` files.
- Bucket counts:
  - `Legacy Function`: `2`
  - `Legacy Alias/Behavior`: `2`
  - `Legacy Text`: `7`
- Highest-severity finding:
  - Deprecated public config key `spectral_normalization_01` is still accepted and propagated through runtime outputs.

## B. File-by-File Inventory

### High
1. Deprecated config key still active in runtime behavior and outputs.
- Evidence:
  - `hypercoreg/coregistration.py:7701` reads `spectral_normalization_01`.
  - `hypercoreg/coregistration.py:7710` emits explicit deprecation warning.
  - `hypercoreg/coregistration.py:7721` warns on conflict with `normalization_mode`.
  - `hypercoreg/config.py:219` default config still includes `spectral_normalization_01`.
  - `hypercoreg/config.py:333` dataclass still exposes `spectral_normalization_01`.
  - `hypercoreg/config.py:440` `to_dict()` still serializes the key.
  - `hypercoreg/normalization.py:589` and `hypercoreg/normalization.py:617` still emit compatibility field.
- Impact:
  - Two normalization control paths are still present (`normalization_mode` and deprecated boolean), increasing ambiguity in config surfaces and downstream metadata contracts.

### Medium
1. Legacy wrappers are present and appear unused.
- Evidence:
  - `hypercoreg/coregistration.py:2360` `_merge_tiepoints_max_reliability` (docstring marks backward compatibility).
  - `hypercoreg/coregistration.py:2604` `_apply_polynomial_warp_order2` (docstring marks legacy call sites).
- Usage check:
  - Repository scan finds only the function definitions, no call sites.
- Impact:
  - Dead compatibility wrappers increase maintenance surface and can mislead readers about active pathways.

2. User-visible legacy mode wording remains in CLI/docs, and legacy-labeled mode is default.
- Evidence:
  - `hypercoreg/cli.py:559` help text says `map_inverse` is legacy.
  - `README.md:113` documents `map_inverse` as legacy.
  - `hypercoreg/config.py:238` default `pan_gcp_mode` is still `"map_inverse"`.
  - `hypercoreg/config.py:352` dataclass default remains `"map_inverse"`.
- Impact:
  - Product messaging marks a mode as legacy while keeping it as default, which can confuse user intent and migration expectations.

### Low
1. Internal legacy/backward text markers (comments/docstrings/logs) still exist.
- Evidence:
  - `hypercoreg/coregistration.py:1921` log text includes `compat mode`.
  - `hypercoreg/spectral.py:73`, `hypercoreg/spectral.py:79`, `hypercoreg/spectral.py:88` include backwards-compatibility docstrings.
  - `hypercoreg/readers/enmap.py:1177` comment notes backwards compatibility fallback.
- Impact:
  - Low runtime risk; mostly documentation clarity/consistency debt.

## C. Likely Dead Legacy Wrappers
1. `_merge_tiepoints_max_reliability` at `hypercoreg/coregistration.py:2360`
- Call-site count in scope: `0` (definition only).
- Recommended status: candidate for removal after one release-note cycle.

2. `_apply_polynomial_warp_order2` at `hypercoreg/coregistration.py:2604`
- Call-site count in scope: `0` (definition only).
- Recommended status: candidate for removal after one release-note cycle.

## D. User-Visible Legacy Wording
1. CLI help text:
- `hypercoreg/cli.py:559` labels `map_inverse` as legacy.

2. README parameter table:
- `README.md:113` labels `pan_gcp_mode=map_inverse` as legacy.

3. Operational migration hint:
- `hypercoreg/coregistration.py:7476` recommends trying `pan_gcp_mode='scaled_image'`, reinforcing de facto migration direction.

## E. Exclusions and False Positives
- Not counted as legacy findings:
  - Generic compatibility wording without legacy/deprecation meaning, such as:
    - `hypercoreg/coregistration.py:7992` (`maximum compatibility` for ENVI header standardization).
    - `hypercoreg/utils.py:304` (`cross-platform compatibility` path normalization).
  - Function names that include `compatibility` but describe current behavior rather than legacy retention:
    - `hypercoreg/readers/enmap.py:1136` `check_enmap_crs_compatibility`.
- No explicit legacy-marker findings in:
  - `runCLI.py`
  - `runGUI.py`
  - `docs/` markdown files in current scope

