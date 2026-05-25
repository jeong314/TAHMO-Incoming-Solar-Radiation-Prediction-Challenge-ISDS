# Data Artifacts

Large CSV artifacts required by `v28_0_reproduce.py` are not committed to this
repository. They must be placed in `runs/v28.0_sharing/` before running the
script.

Use `ARTIFACT_MANIFEST.csv` to verify file size and SHA256 for each artifact.

## Required For Full Submit Mode

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

## Optional Reference Check

```text
submission_reference_v28_0.csv
```

When this file is present, regenerated `submission.csv` should match it exactly
by value. Its expected SHA256 is recorded in `ARTIFACT_MANIFEST.csv`.

## PowerShell Hash Check

From the repository root, after placing the artifacts:

```powershell
Import-Csv .\runs\v28.0_sharing\data\ARTIFACT_MANIFEST.csv | ForEach-Object {
  $path = Join-Path .\runs\v28.0_sharing $_.file
  if (Test-Path -LiteralPath $path) {
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
    [PSCustomObject]@{ file = $_.file; size_ok = ((Get-Item -LiteralPath $path).Length -eq [int64]$_.size_bytes); sha256_ok = ($hash -eq $_.sha256) }
  }
}
```
