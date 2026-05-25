# TAHMO Incoming Solar Radiation Prediction Challenge

This repository contains a GitHub-ready v28.0 reproduction package for the
Zindi TAHMO Incoming Solar Radiation Prediction Challenge.

The v28.0 solution is a deterministic final-stage correction pipeline. It
starts from strong prior submissions and applies a physically constrained,
mean-neutral clear-sky-index edge-month correction. It does not train a new
tree model.

## Public Result

| version | public score | Abs MBE | RMSE |
|---|---:|---:|---:|
| v28.0 | 0.449493955 | 2.611275853 | 60.52415722 |

## Repository Layout

```text
runs/v28.0_sharing/
  v28_0_reproduce.py          # final-stage reproduction script
  requirements.txt            # minimal package versions
  README.md                   # package-level guide
  docs/                       # method, data, environment, lineage, reproduction notes
  data/ARTIFACT_MANIFEST.csv  # required external artifacts, sizes, SHA256 hashes
  preprocessing/              # external-data download and preprocessing source
  reference_outputs/          # reference metrics and diagnostics
  upstream_scripts/           # upstream generator scripts for lineage review
```

Large CSV artifacts are intentionally not committed to normal Git history.
The prepared train/test tables are over 800 MB each and exceed GitHub's
regular file limit. Put the files listed in
`runs/v28.0_sharing/data/ARTIFACT_MANIFEST.csv` next to
`runs/v28.0_sharing/v28_0_reproduce.py` before running the package.

## Quick Start

From the repository root:

```powershell
.\.venv\Scripts\python.exe .\runs\v28.0_sharing\v28_0_reproduce.py --mode validate --smoke-test
```

Full final-stage reproduction:

```powershell
.\.venv\Scripts\python.exe .\runs\v28.0_sharing\v28_0_reproduce.py --mode submit
```

Expected generated output:

```text
runs/v28.0_sharing/submission.csv
```

## Submission Rule

Zindi requires the final submission columns to be exactly:

```text
ID, TargetMBE, TargetRMSE
```

`TargetMBE` and `TargetRMSE` must be copied from the same final prediction
vector. The reproduction script enforces that rule for the final submission
and all emitted candidate submissions.

## Main Documentation

- `runs/v28.0_sharing/docs/METHOD_SUMMARY.md`: scientific method and algorithm
- `runs/v28.0_sharing/docs/REPRODUCTION.md`: commands and verification
- `runs/v28.0_sharing/docs/DATA_USED.md`: required input artifacts and columns
- `runs/v28.0_sharing/docs/DATA_SOURCES.md`: original data sources and download lineage
- `runs/v28.0_sharing/docs/GENERATED_INPUT_LINEAGE.md`: generated input lineage
- `runs/v28.0_sharing/docs/UPSTREAM_V166_V177.md`: upstream v16.6/v17.7 artifact notes
- `runs/v28.0_sharing/docs/ENVIRONMENT.md`: tested runtime environment
