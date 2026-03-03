# Phase 1 Feature Audit Guide

## Purpose
Use the audit helper to validate Phase-1 feature claims against an exact repository snapshot.

This avoids false negatives caused by stale branches, old line numbers, or different local directories.

## Run Commands

```powershell
# Markdown report for UAT/verification notes
C:\Users\josea\miniforge3\envs\hypbox\python.exe scripts/phase1_feature_audit.py --format markdown

# JSON report for automation or tooling
C:\Users\josea\miniforge3\envs\hypbox\python.exe scripts/phase1_feature_audit.py --format json

# Fail CI/check scripts if any capability token is missing
C:\Users\josea\miniforge3\envs\hypbox\python.exe scripts/phase1_feature_audit.py --format markdown --strict
```

## Mandatory Provenance Fields
Every review that claims a feature is missing or implemented must capture:
- `repo_root`
- `branch`
- `commit_sha`
- `generated_at_utc`

If these fields are absent, treat the review as unverified.

## What The Script Checks
- `STRATIFICATION`: staged merge path includes spatial stratification helpers.
- `FIXED_BAND_PAIRS`: fixed PRISMA-S2 mapping entrypoints and config key exist.
- `CONSENSUS_COMPOSITE`: consensus grouping and `QUALITY_SCORE` path exist.
- `POLYNOMIAL_SAFETY`: polynomial order decision + downgrade-capable warp helper exist.
- `SENSOR_PROFILES`: sensor-aware global/local profile controls exist.
- `POSTWARP_QA`: optional post-warp phase-correlation QA controls exist.
- `GEO_MESH_AFFINE`: optional geolocation-mesh affine controls exist.

## Interpreting Results
- `PASS`: all required tokens for a capability were found with file+line evidence.
- `FAIL`: one or more required tokens were not found; inspect the `missing` entries.

## Verification Workflow Integration
1. Run the audit script.
2. Paste report into `01-VERIFICATION.md` or UAT evidence.
3. If `FAIL`, open a gap with the report attached.
4. If `PASS`, include commit SHA in the claim to make the review reproducible.
