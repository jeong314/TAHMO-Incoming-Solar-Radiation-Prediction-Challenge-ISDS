# Reproduction Guide

## Prerequisites

Run from the repository root. Use the project virtual environment:

```powershell
.\.venv\Scripts\python.exe
```

Required packages are listed in `requirements.txt`. The script only needs `numpy` and `pandas`.

Before running, place the external artifacts listed in
`../data/ARTIFACT_MANIFEST.csv` next to `v28_0_reproduce.py`.

## Exact Command

```powershell
.\.venv\Scripts\python.exe .\runs\v28.0_sharing\v28_0_reproduce.py --mode submit
```

Expected runtime on the current machine: about 4-5 minutes.

## Smoke Test

Before full reproduction:

```powershell
.\.venv\Scripts\python.exe .\runs\v28.0_sharing\v28_0_reproduce.py --mode validate --smoke-test
```

This reads the first 120,000 rows and validates the code path quickly.

## Expected Outputs

After `--mode submit`, the folder should contain:

```text
submission.csv
metrics.json
train.log
diagnostics/candidate_metrics.csv
diagnostics/submission_decision.csv
diagnostics/test_candidate_diagnostics.csv
submission_candidates/*.csv
V28_0_METHOD_SUMMARY.md
```

## Submission Equality Check

The regenerated `submission.csv` should exactly match `submission_reference_v28_0.csv`.

Validation run in this package produced:

```text
rows: 683353
columns: ['ID', 'TargetMBE', 'TargetRMSE']
ID order matches SampleSubmission: True
TargetMBE == TargetRMSE: True
finite predictions: True
same as submission_reference_v28_0.csv: True
max_abs_diff_ref: 0.0
mean: 185.47287532737013
p90: 657.477335512712
p99: 877.2299867540268
```

The reference submission SHA256 is:

```text
D3D8F0FD72F24ECB1D1DD451C3672573F35361660422AA972729B2F67FB28CD6
```

## Verification Command

```powershell
.\.venv\Scripts\python.exe -c "import pandas as pd, numpy as np; sub=pd.read_csv('runs/v28.0_sharing/submission.csv'); ref=pd.read_csv('runs/v28.0_sharing/submission_reference_v28_0.csv'); sample=pd.read_csv('runs/v28.0_sharing/SampleSubmission.csv', usecols=['ID']); y=sub['TargetMBE'].to_numpy(float); yr=ref['TargetMBE'].to_numpy(float); print(len(sub)); print(list(sub.columns)); print(np.array_equal(sub['ID'].to_numpy(), sample['ID'].to_numpy())); print(np.array_equal(y, sub['TargetRMSE'].to_numpy(float))); print(np.isfinite(y).all()); print(np.array_equal(y, yr)); print(np.max(np.abs(y-yr)))"
```

Expected final line:

```text
0.0
```

## Notes

This package is final-stage reproducible when the artifacts in
`../data/ARTIFACT_MANIFEST.csv` are present. It does not rebuild the upstream v20
physical preprocessing, v16.6 model, or v17.7 model from raw competition data.
