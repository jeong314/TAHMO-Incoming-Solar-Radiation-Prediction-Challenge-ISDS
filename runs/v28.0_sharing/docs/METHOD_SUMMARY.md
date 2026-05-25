# Method Summary

## Problem View

The challenge asks us to predict missing even-month 15-minute incoming shortwave radiation for TAHMO stations. Solar radiation is strongly constrained by geometry: at night it is zero, under clear sky it follows a predictable envelope, and cloud/aerosol/sensor effects mostly modulate the clear-sky index:

```text
kt = observed_radiation / clear_sky_reference
```

v28.0 is built around that view. It does not try to relearn all 15-minute radiation from scratch. Instead, it starts from the strong team submission `jieun20.csv` and applies a physically constrained energy correction where validation and public feedback showed systematic bias: edge months.

## Core Hypothesis

The strongest remaining error after v27/v28 work was seasonal edge bias:

- early edge months need a smaller positive/negative adjustment;
- late edge months need a stronger correction;
- raw 15-minute external cloud timing is noisy, but monthly clear-sky-index energy residuals are useful.

v28.0 therefore applies a distance-weighted adjacent-month `kt` anomaly as a monthly energy correction, then redistributes that energy using the trusted anchor submission shape.

## Inputs Used By The Algorithm

The script reads the following kinds of signals:

- `phys_clear_sky_ref`: clear-sky radiation envelope
- `phys_solar_elevation_clipped`: night/day and final night clipping
- `phys_daylight_flag`: daylight aggregation mask
- `fusion_weighted_allsky` and `fusion_external_clear_index_weighted`: external all-sky radiation and clear-sky-index proxy
- `lsa_dssf_phys`, `cams_ghi_phys`, `power_allsky_phys`: satellite/weather radiation references used in the physical preprocessing table
- `fusion_reliability_score`, `fusion_sensor_disagreement`, `fusion_allsky_cv`: reliability diagnostics
- prior-stage validation/test predictions: v16.6 and v17.7
- strong team submissions: `jieun20.csv` as the final anchor, `ds15.csv` for diagnostics

See `JIEUN20_ANCHOR_ANALYSIS.md` for the audit of the `jieun20` anchor and the
team-provided model-ready input bundle.

## Monthly Energy Aggregation

For daylight rows only, the code aggregates by:

```text
station, dt_year, dt_month
```

It computes sums for clear-sky reference, external all-sky radiation, and target radiation where available. Monthly `kt` values are then:

```text
target_kt = target_sum / clear_sum
fusion_kt = fusion_sum / clear_sum
```

This makes the correction operate on physically normalized energy instead of raw W/m2.

## Adjacent-Month kt Prediction

For a target month, source months are adjacent observed odd months.

Validation examples:

```text
March -> January and May
November -> July and September
```

Test examples:

```text
February -> January and March
December -> September and November
```

Source months are distance-weighted:

```text
weight = clear_sum / distance(month, target_month) ** dist_power
```

The monthly predicted clear-sky index is:

```text
pred_kt = source_target_kt + beta * (target_external_kt - source_external_kt)
pred_kt = (1 - shrink) * pred_kt + shrink * station_kt
pred_kt clipped to [0, 1.45]
```

For the submitted v28.0 top candidate:

```text
beta = 1.0
shrink = 0.30
dist_power = 0.75
ratio_clip = 0.16
alpha_early = 0.18
alpha_late = 0.30
gate = edge
neutral = global
anchor = jieun20
```

## Reconstructing 15-Minute Predictions

The monthly kt prediction produces a monthly daylight energy target. The 15-minute shape still comes from the anchor submission.

For each `station, year, month` daylight group:

```text
base_sum = sum(anchor_prediction)
model_sum = predicted_monthly_energy
target_sum = (1 - alpha) * base_sum + alpha * model_sum
ratio = clip(target_sum / base_sum, 1 - ratio_clip, 1 + ratio_clip)
prediction_15min = anchor_prediction_15min * ratio
```

Month-specific alpha:

```text
February/March: alpha_early = 0.18
November/December: alpha_late = 0.30
Other months: base alpha = 0.15
```

Only edge months are active in the final submitted candidate:

```text
validation edge months: March, November
test edge months: February, December
```

All non-edge test months remain equal to the anchor. This is deliberate:
`jieun20.csv` supplies the trusted 15-minute shape, while v28.0 only changes
February and December when the monthly clear-sky-index energy evidence passes
the configured edge-month correction path.

## Mean-Neutral Constraint

After applying the edge-month delta, v28.0 subtracts the global active-row mean shift from active daylight correction rows. This preserves the global prediction mean relative to the anchor and avoids a pure leaderboard mean-shift trick.

The submitted v28.0 output has:

```text
mean = 185.47287532737013
mean_delta_vs_jieun20 = -0.0036030358435313876
MAE_vs_jieun20 = 0.4684304980392266
RMSE_vs_jieun20 = 2.09467889158946
```

The small global mean delta is important: v28.0 is best understood as a sparse
physical correction to `jieun20`, not as a new full-test prediction model.

## Submission Guard

The code writes submissions through `make_submission()`, which enforces:

- exact column order: `ID`, `TargetMBE`, `TargetRMSE`
- ID order matches `SampleSubmission.csv`
- `TargetMBE` and `TargetRMSE` are exactly identical
- all predictions are finite
- final clipping to `[0, 1500]`
- night rows with solar elevation `<= -1` are zeroed

## Why v28.0 Beat Later Attempts

v28.4/v28.5 pushed the same edge-correction idea more aggressively. They improved local edge MBE diagnostics but over-perturbed the test distribution, and public RMSE/hidden distribution did not reward that extra aggressiveness. v28.0 was the better balance:

- enough asymmetric edge correction to reduce Abs MBE;
- not so much correction that RMSE degradation dominated;
- still anchored tightly to the strong `jieun20` submission shape.
