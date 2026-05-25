# v28.0 Sharing Package

This folder contains the GitHub-facing v28.0 reproduction package for the
TAHMO solar radiation challenge. The package is designed for code review:
the method, input lineage, reference diagnostics, and submission-format guard
are visible without requiring reviewers to inspect the full local workspace.

## Result

| leaderboard | rank | score | Abs MBE | RMSE |
|---|---:|---:|---:|---:|
| Public | - | 0.449493955 | 2.611275853 | 60.52415722 |
| Private | 38 | 0.456169433 | 2.515170202 | 60.78379502 |

For the broader project story, feature families, and version-by-version
development history, see the repository root `README.md`.

## Artifact Policy

The final-stage script needs prepared CSV artifacts that are too large for a
normal GitHub repository. Those files are not committed here. Before running,
place the files listed in `data/ARTIFACT_MANIFEST.csv` in this folder, next to
`v28_0_reproduce.py`.

Required runtime artifacts include:

```text
train_physical_preprocessed.csv
test_physical_preprocessed.csv
SampleSubmission.csv
v166_val_predictions.csv
v166_submission.csv
v177_val_predictions.csv
v177_submission.csv
jieun20.csv
ds15.csv
```

Optional verification artifact:

```text
submission_reference_v28_0.csv
```

## Quick Run

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

If `submission_reference_v28_0.csv` is available and matches the manifest,
the regenerated `submission.csv` should be value-identical to it.

## Folder Layout

```text
v28.0_sharing/
  v28_0_reproduce.py
  requirements.txt
  README.md

  docs/
    METHOD_SUMMARY.md
    REPRODUCTION.md
    DATA_USED.md
    DATA_SOURCES.md
    ENVIRONMENT.md
    GENERATED_INPUT_LINEAGE.md
    UPSTREAM_V166_V177.md

  data/
    ARTIFACT_MANIFEST.csv

  preprocessing/
    README.md
    requirements-external.txt
    tahmo_tropomi_cloud_downloader_v3_daily.py
    merge_tropomi_cloud_features.py
    v14_1_nasa_power_validation.py
    v14_2_cams_radiation_validation.py
    v20_0_physical_preprocessing.py
    v25_1_sarah3_access_audit.py
    v25_6_sarah3_multi_day_pilot.py

  reference_outputs/
    metrics_reference_v28_0.json
    metrics.json
    metrics_smoke.json
    V28_0_METHOD_SUMMARY.md
    diagnostics/

  upstream_scripts/
    v20_0_physical_preprocessing.py
    v16_6_aggressive_pv_residual_stack.py
    v17_7_dual_blend_station_filter.py
```

Generated after running `--mode submit`:

```text
submission.csv
metrics.json
train.log
diagnostics/
submission_candidates/
V28_0_METHOD_SUMMARY.md
```

## Method Summary

v28.0 is not a fresh model-training script. It is a deterministic final-stage
physics/meta correction:

1. Use `jieun20.csv` as the main test-time anchor shape.
2. Use prepared physical features: daylight flag, solar elevation, clear-sky
   reference, LSASAF/CAMS/POWER fusion, AOD/opacity, and reliability signals.
3. Aggregate daylight energy by `station`, `year`, and `month`.
4. Estimate target even-month clear-sky-index energy from adjacent odd-month
   station history and external satellite/weather anomaly.
5. Apply the active correction only to edge test months: February and December.
6. Keep the correction globally mean-neutral.
7. Write `ID`, `TargetMBE`, and `TargetRMSE`, with both target columns copied
   from the exact same final prediction vector.

## Main Files To Read

- `docs/METHOD_SUMMARY.md`: scientific explanation and algorithm details
- `docs/REPRODUCTION.md`: exact commands and verification
- `docs/DATA_USED.md`: required input files and their roles
- `docs/DATA_SOURCES.md`: external data sources, access methods, and join policy
- `docs/GENERATED_INPUT_LINEAGE.md`: generated input lineage
- `docs/UPSTREAM_V166_V177.md`: upstream v16.6/v17.7 artifact generation notes
- `docs/ENVIRONMENT.md`: Python/package context
- `data/ARTIFACT_MANIFEST.csv`: file sizes and SHA256 checksums for external artifacts
- `preprocessing/`: external-data download and preprocessing scripts
