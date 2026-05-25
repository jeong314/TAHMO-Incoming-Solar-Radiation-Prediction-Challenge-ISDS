# TAHMO Incoming Solar Radiation Prediction Challenge

This repository documents and reproduces our solution for the Zindi TAHMO
Incoming Solar Radiation Prediction Challenge. The final package is centered on
`runs/v28.0_sharing`, which contains the v28.0 final-stage reproduction script,
method notes, external-data lineage, preprocessing source code, and reference
diagnostics.

The challenge is to reconstruct missing even-month 15-minute incoming shortwave
radiation for TAHMO stations. The solution evolved from direct regression into
a physics-guided stack built around solar geometry, clear-sky index modeling,
satellite/reanalysis residuals, station sensor reliability, analog clear-sky
index retrieval, and a final edge-month energy correction.

## Final Result

| leaderboard | rank | score | Abs MBE | RMSE |
|---|---:|---:|---:|---:|
| Public | - | 0.449493955 | 2.611275853 | 60.52415722 |
| Private | 38 | 0.456169433 | 2.515170202 | 60.78379502 |

The final submitted version was `v28.0`. It is not a fresh training pipeline.
It is a deterministic final-stage correction that starts from strong prior/team
submission shapes and applies a mean-neutral, physically constrained
clear-sky-index correction to February and December.

## Core Modeling View

Solar radiation is mostly constrained by geometry. At night it is zero; under
clear sky it follows a predictable envelope; clouds, aerosols, humidity,
surface state, and defective sensors mostly modulate the clear-sky index:

```text
kt = observed_or_predicted_radiation / clear_sky_reference
```

That view drove most of the project:

- model ratios or residuals around clear-sky radiation instead of raw radiation
  whenever possible;
- treat LSA-SAF, CAMS, and NASA POWER as noisy physical sensors, not ground
  truth;
- separate true cloudy low-radiation events from suspicious daytime zeroes;
- keep station identity and station-month reliability visible throughout the
  pipeline;
- isolate leaderboard/MBE calibration as post-processing, never as hidden
  training leakage;
- always write one prediction vector to both `TargetMBE` and `TargetRMSE`.

## Feature Families

| feature family | examples | preprocessing | impact in this project |
|---|---|---|---|
| Solar geometry and time | solar elevation, SZA/cosz, local solar time, day-of-year, daylight/night flags, extraterrestrial horizontal radiation | Computed from timestamp and station coordinates; clipped to physical ranges; cyclic encodings for month/day/hour; final night clipping by solar elevation. | The first-order signal. It defines when radiation can exist, the clear-sky envelope, validation regimes, and the final zero-at-night guard. |
| Clear-sky index features | `target / clear_sky`, `DSSF / clear_sky`, `CAMS_GHI / clear_sky`, monthly daylight energy kt | Denominators guarded with eps/min-clear thresholds; ratios clipped to plausible ranges; aggregated by station/year/month for energy correction. | The most important representation. It made station/month energy transfer and edge-month correction stable. |
| LSA-SAF / MDSSFTD | DSSF, fraction diffuse, quality/missing flags, DSSF clear-sky ratio, diffuse/direct proxy | Clipped to physical W/m2 ranges; night handling separated from raw values; quality bits exposed; residual and ratio features built against clear sky and other sources. | The strongest target-like external signal early in the project. Useful but biased, especially under clouds and defective station periods. |
| CAMS Solar Radiation | all-sky GHI, clear-sky GHI, BHI/DHI/BNI, clearness, diffuse fraction, reliability | Downloaded as 15-minute time series; Wh/m2 values converted to W/m2 averages; merged by station/timestamp; source gaps and reliability retained. | The best additional external radiation source after LSA-SAF. In v14.4 it reduced OOF RMSE materially when learned inside the model stack. |
| NASA POWER | hourly all-sky and clear-sky SW radiation, cloud amount, temperature, dew point, RH, pressure, wind | Downloaded per station; hourly values joined to each quarter-hour; clear-sky ratios and gaps versus DSSF/CAMS built. | Helpful background atmosphere/cloud signal. By itself it was not a replacement for satellite radiation, but it improved fusion and residual context. |
| AOD, opacity, albedo, MLST/LST | aerosol/opacity values, effective albedo, MLST quality, land-air temperature deltas | Non-negative clipping; station/global median imputation; log transforms; status one-hot flags; interactions with daylight and clear-sky radiation. | Secondary but useful for reliability and residual regimes. Aggregated importance showed albedo and AOD features among the better physical residual signals. |
| Station and sensor reliability | station ID, station-month zero rate, suspicious daytime zero flag, source disagreement, fusion reliability | Train-only target diagnostics kept separate from test features; suspicious daytime zeroes flagged/downweighted; station-month summaries emitted. | Critical for avoiding wrong corrections. Defective stations such as `TA00338` showed physically normal satellite radiation but near-zero daytime target readings. |
| Analog kt and station profiles | station quarter-hour kt medians, solar-bin kt medians, day counts, source-month kt | Built fold-safely from observed months; target months estimated from adjacent odd months; distance-weighted monthly aggregation. | Useful late-stage signal. It provided sparse, interpretable corrections where broad residual models were too noisy. |
| External fusion | weighted all-sky blend, external clear-index blend, sensor disagreement, source CV | LSA-SAF, CAMS, and POWER clipped and blended with reliability/disagreement diagnostics. | More robust than trusting a single source. `anchor_vs_fusion_gap` was the strongest v21.1 residual-gate feature by summed importance. |

## What Mattered Most

The largest structural gain came from respecting solar geometry and clear-sky
normalization. A raw radiation model has to relearn the sun every time; a kt or
residual model only needs to learn how clouds, aerosols, and station behavior
modify a physically constrained envelope.

Among external datasets, LSA-SAF DSSF was the closest physical match to the
target, but CAMS Solar Radiation was the most valuable new independent source.
The clearest quantified jump was v14.4: adding CAMS 15-minute radiation and
NASA POWER hourly features inside a LightGBM/Ridge component stack improved
local OOF from the v12.14 base weighted score `36.7739` / RMSE `73.5411` to
`35.6274` / RMSE `71.1519`. Public RMSE reached `60.41089492`.

Later, broad from-scratch physical models were not enough. v21.0 showed that
the v20 feature table was more useful as diagnostics and gated correction
signals than as a full replacement for a strong anchor. The final solution
therefore uses strong anchor shapes, then applies small, auditable physical
energy corrections only where the validation story was credible.

## Version Evolution

| stage | version | main idea | evidence / result | lesson |
|---|---|---|---|---|
| Baseline anchor | v12.14 | Strong prior model/submission used as the early anchor. | v14.4 OOF reference: weighted `36.7739`, RMSE `73.5411`. | A strong shape anchor was hard to beat with external data alone. |
| External source validation | v14.1 | Download NASA POWER hourly station data and test residual value. | Validation-only; POWER alone did not beat the main anchor. | Useful context, not a primary radiation source. |
| External source validation | v14.2 | Download CAMS Solar Radiation 15-minute time series and test residual value. | CAMS residual signal was useful, but naive direct use could be biased. | CAMS should be learned as a noisy sensor/residual source. |
| External model stack | v14.4 | Add CAMS + NASA POWER to LSASAF/solar features in LightGBM residual/direct/kt components, blended by Ridge. | Public score `0.380668753`, Abs MBE `3.435864683`, RMSE `60.41089492`; OOF RMSE improved by about `2.39`. | External data improved RMSE most in high-sun and satellite-disagreement regimes. |
| Sensor-state direction | v15.x | Move from generic external stacking toward clear-sky residuals, dead/low-gain station handling, and fold-safe analog features. | Diagnostics highlighted station-month failures such as `TA00338` daytime zero behavior. | The task is partly sensor-state reconstruction, not only solar radiation estimation. |
| PV/energy residual stack | v16.6 | Aggressive pvlib/daily-energy residual stack while preserving one-vector submission compliance. | Public reference in later configs: score `0.431366812`, Abs MBE `2.853991111`, RMSE `60.10061752`; local weighted `34.9621`. | Controlled MBE reduction could beat pure RMSE improvements. |
| Station reliability filter | v17.7 | Revert harmful stations from v17.6-style correction back to v16.6 base. | Local weighted `34.9279`; filtered 8 stations including `TA00109`, `TA00118`, `TA00355`. | Station-level harm detection matters more than uniform correction. |
| Physical preprocessing table | v20.0 | Rebuild deterministic physical features: solar geometry, LSASAF/CAMS/POWER fusion, AOD/opacity/albedo/MLST, reliability diagnostics. | Contract: 642,175 train rows, 683,353 test rows, 40 stations, no train/test feature mismatch after train-only diagnostics removed. | This became the diagnostic and correction substrate for late-stage work. |
| Physical model benchmark | v21.0 / v21.1 | Test from-scratch physical feature models and anchor residual gates. | v21.0 from-scratch models were worse; v21.1 residual gate gain was too small for submission. | Do not replace the anchor; use physical features for sparse, high-confidence corrections. |
| Energy residual shaping | v27.1 / v27.2 | Clear-sky-index residual assimilation over strong base predictions and team references. | Public references used for risk comparison: v16.6, ds15, jieun20. | Energy shape adjustments can improve MBE but must be mean/risk constrained. |
| Edge-month correction | v27.7 | Adjacent-month edge-season kt anomaly and mean-neutral shape correction. | Local best candidates improved weighted/risk slightly but were not strict submission guards. | Edge months were the remaining interpretable bias pattern. |
| Final package | v28.0 | Asymmetric February/December kt energy correction anchored to `jieun20`, globally mean-neutral. | Public `0.449493955`; private `0.456169433`; private rank 38. | Small edge-only physical correction gave the best final leaderboard balance. |

## v28.0 Final Algorithm

The final script `runs/v28.0_sharing/v28_0_reproduce.py` is deterministic:

1. Read prepared physical train/test tables and frozen prior prediction
   artifacts.
2. Build daylight monthly energy summaries by `station`, `year`, and `month`.
3. Estimate missing even-month clear-sky-index energy from adjacent observed odd
   months plus external all-sky/clear-sky-index anomaly.
4. Apply correction only to edge months:
   - validation: March and November;
   - test: February and December.
5. Redistribute monthly energy by scaling the trusted 15-minute anchor shape.
6. Subtract the global active-row mean shift so the correction is shape/energy
   constrained rather than a pure mean trick.
7. Clip to physical bounds and write a Zindi-compliant one-vector submission.

The selected public candidate used:

```text
anchor = jieun20
beta = 1.0
shrink = 0.30
dist_power = 0.75
ratio_clip = 0.16
alpha_early = 0.18
alpha_late = 0.30
gate = edge
neutral = global
```

## Repository Layout

```text
runs/v28.0_sharing/
  v28_0_reproduce.py          # final-stage reproduction script
  requirements.txt            # minimal final-stage package versions
  README.md                   # package-level guide
  docs/                       # method, data, environment, lineage, reproduction notes
  data/ARTIFACT_MANIFEST.csv  # required external artifacts, sizes, SHA256 hashes
  preprocessing/              # external-data download and preprocessing source
  reference_outputs/          # reference metrics and diagnostics
  upstream_scripts/           # upstream generator scripts for lineage review
```

Large CSV artifacts are intentionally not committed to normal Git history. The
prepared train/test tables are over 800 MB each and exceed GitHub's regular
file limit. Put the files listed in
`runs/v28.0_sharing/data/ARTIFACT_MANIFEST.csv` next to
`runs/v28.0_sharing/v28_0_reproduce.py` before running the package.

## Reproduction

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
and all emitted candidate submissions. No metric-specific column shift is
applied.

## Main Documentation

- `runs/v28.0_sharing/docs/METHOD_SUMMARY.md`: final v28.0 algorithm details
- `runs/v28.0_sharing/docs/REPRODUCTION.md`: commands and verification
- `runs/v28.0_sharing/docs/DATA_USED.md`: required input artifacts and columns
- `runs/v28.0_sharing/docs/DATA_SOURCES.md`: original data sources and download lineage
- `runs/v28.0_sharing/docs/GENERATED_INPUT_LINEAGE.md`: generated input lineage
- `runs/v28.0_sharing/docs/UPSTREAM_V166_V177.md`: upstream v16.6/v17.7 artifact notes
- `runs/v28.0_sharing/docs/ENVIRONMENT.md`: tested runtime environment
- `runs/v28.0_sharing/preprocessing/README.md`: external-data preprocessing code map

## Limitations

The repository supports exact final-stage reproduction when the manifest-listed
artifacts are present. It also includes the external-data download and
preprocessing source used for lineage review. It is not yet a single-command
clean-room rebuild from only raw competition CSVs and public APIs, because the
historical research pipeline produced very large intermediate model-ready
tables and frozen prediction artifacts.
