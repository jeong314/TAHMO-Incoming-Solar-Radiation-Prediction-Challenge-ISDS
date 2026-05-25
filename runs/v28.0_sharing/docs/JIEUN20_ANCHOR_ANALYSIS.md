# jieun20 Anchor Analysis

This note documents the team-provided `jieun20` artifacts because v28.0 uses
`jieun20.csv` as its final 15-minute prediction anchor.

The most important distinction is that the v28.0 anchor is not the same file as
the `candidate01_v1592_ta00338_m4m6m8m10_soft025.csv` file found in the
team-provided sharing bundle. The `candidate01` file matches the `ds15.csv`
reference artifact by size, row count, prediction statistics, and SHA256 hash.
The actual v28.0 anchor is `jieun20.csv`.

## Files Audited

| artifact | role in v28.0 review | rows | mean prediction | zero ratio | SHA256 |
|---|---|---:|---:|---:|---|
| `jieun20.csv` | final v28.0 anchor submission | 683,353 | 185.476478363 | 0.501995 | `6672039d548cd5d927e0aa9a056f31701f9b66015413d00541d89491d289b7f2` |
| `ds15.csv` | diagnostic/reference submission | 683,353 | 186.880194602 | 0.488468 | `74a328b7dcc1e0eefec2ea402c3049cc207e7243d88eeb9990d62b556483483a` |
| `candidate01_v1592_ta00338_m4m6m8m10_soft025.csv` | team-shared champion candidate, same content as `ds15.csv` | 683,353 | 186.880194602 | 0.488468 | `74a328b7dcc1e0eefec2ea402c3049cc207e7243d88eeb9990d62b556483483a` |

All three submission files use the competition format:

```text
ID,TargetMBE,TargetRMSE
```

`TargetMBE` and `TargetRMSE` are identical in the audited files.

## What jieun20 Contributed

`jieun20.csv` gave v28.0 a strong full-test prediction shape. The final script
does not retrain this anchor. It scales the anchor only in high-confidence edge
months after estimating monthly clear-sky-index energy from adjacent observed
months and external physical features.

In the submitted v28.0 output:

```text
mean_delta_vs_jieun20 = -0.0036030358435313876
MAE_vs_jieun20        = 0.4684304980392266
RMSE_vs_jieun20       = 2.09467889158946
```

This means v28.0 stayed very close to `jieun20` globally. The change was a
sparse physical edge-month correction, not a replacement model.

## Strategy Notes From jieun20

The team strategy document described a champion-bias-correction track:

- start from a strong champion submission with good RMSE but positive MBE;
- apply post-processing shifts to reduce MBE;
- protect night rows by zeroing rows with low solar elevation;
- test global, day-only, station-month, and known/auto sensor-state variants;
- use model outputs as candidates or hedges rather than blindly averaging them.

The important modeling lesson for v28.0 was not the exact leaderboard shift
constant. It was that a strong shape anchor should be preserved, while MBE or
monthly energy bias should be corrected only through explicit post-processing
guards. v28.0 follows that principle by applying a mean-neutral edge-month
clear-sky-index correction on top of `jieun20`.

## Shared Model-Ready Inputs

The local team bundle also included model-ready train/test tables with external
physical data already joined:

| file | rows | columns | target included | station count | months |
|---|---:|---:|---:|---:|---|
| `Train_with_LSASAF_MDSSFTD_albedo_AOD_OPACITY_THREDDS_CAMS_NASA_modelready.csv` | 642,175 | 84 | yes | 40 | odd months, 2016-01 to 2020-11 |
| `Test_with_LSASAF_MDSSFTD_albedo_AOD_OPACITY_THREDDS_CAMS_NASA_modelready.csv` | 683,353 | 83 | no | 40 | even months, 2016-02 to 2020-12 |

Key feature groups in those tables:

- competition columns: `ID`, `timestamp`, weather variables, station metadata;
- solar geometry: `solar_elevation`;
- LSA-SAF / MDSSFTD: `DSSF`, `FRACTION_DIFFUSE`, quality and missing flags;
- aerosol/cloud opacity: `AOD`, `OPACITY_INDEX`, status flags and interactions;
- albedo: broadband/near-infrared/visible albedo and age/quality fields;
- CAMS radiation: all-sky/clear-sky GHI, BHI, DHI, BNI, clearness, diffuse
  fraction, reliability;
- NASA POWER: all-sky/clear-sky radiation, cloud, temperature, humidity,
  precipitation, pressure, wind, TOA radiation and clear-sky ratios.

The sharing repository does not commit these large model-ready CSVs. Their role
is represented through the smaller v28.0 contract:

- `train_physical_preprocessed.csv`
- `test_physical_preprocessed.csv`
- `data/ARTIFACT_MANIFEST.csv`
- `docs/DATA_SOURCES.md`
- `preprocessing/`

## LGBM Candidate Table

The bundle's `test_prediction_table.csv` contains 16 prediction columns:

```text
lgbm_full_raw
lgbm_foldavg_raw
lgbm_full_bias
lgbm_foldavg_bias
lgbm_full_bias_day
lgbm_foldavg_bias_day
lgbm_full_sm_shift
lgbm_foldavg_sm_shift
lgbm_full_known_state
lgbm_foldavg_known_state
lgbm_full_sm_known_state
lgbm_foldavg_sm_known_state
lgbm_full_auto_state
lgbm_foldavg_auto_state
lgbm_full_sm_auto_state
lgbm_foldavg_sm_auto_state
```

These columns show the same late-stage pattern used elsewhere in this project:
raw model predictions were less important than controlled variants with MBE
shift, day-only shift, station-month correction, and sensor-state correction.
v28.0 did not directly use this table at runtime, but the audit supports why
the final package treats `jieun20` as a trusted anchor and keeps further changes
small, explicit, and guarded.

## GitHub Policy

Do not commit the team-shared model-ready CSVs to normal Git history. They are
large derived artifacts and include competition/test rows. GitHub should expose:

- the runtime artifact manifest and checksums;
- the source/preprocessing scripts used to build physical features;
- the anchor analysis in this document;
- the final deterministic correction code.
