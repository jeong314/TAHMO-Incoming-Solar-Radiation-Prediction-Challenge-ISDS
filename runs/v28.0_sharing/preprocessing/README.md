# External Data and Preprocessing Code

This folder contains the source scripts used to download, audit, merge, and
physically preprocess the external data that feed the v28.0 final-stage package.

The scripts are included for code review and lineage transparency. They do not
commit raw downloads, API keys, local caches, or the large joined CSV outputs.

## Scripts

| script | role |
|---|---|
| `tahmo_tropomi_cloud_downloader_v3_daily.py` | Extract daily Sentinel-5P/TROPOMI cloud features for every station through Google Earth Engine. |
| `merge_tropomi_cloud_features.py` | Merge daily TROPOMI station features onto TAHMO train/test or model-ready CSVs. |
| `v14_1_nasa_power_validation.py` | Download NASA POWER hourly station data and build station-hour features. |
| `v14_2_cams_radiation_validation.py` | Download CAMS Solar Radiation Time Series at each station and convert 15-minute Wh/m2 values to W/m2. |
| `v25_1_sarah3_access_audit.py` | Audit EUMETSAT SARAH-3 access and build a station/date request plan. |
| `v25_6_sarah3_multi_day_pilot.py` | Download pilot SARAH-3 SIS/CAL NetCDF files through `eumdac` and extract nearest station grid values. |
| `v20_0_physical_preprocessing.py` | Build the final physical train/test tables used by `v28_0_reproduce.py`. |
| `CAMS_POWER_column_dictionary.csv` | Column descriptions for CAMS/POWER-derived features. |

## Expected Flow

```text
Competition Train/Test/SampleSubmission
        |
        |-- LSA-SAF MDSSFTD / CAMS / NASA POWER / AOD / opacity / albedo joins
        |-- optional TROPOMI and SARAH-3 probes
        v
Train_with_LSASAF_MDSSFTD_albedo_AOD_OPACITY_THREDDS_CAMS_NASA_modelready.csv
Test_with_LSASAF_MDSSFTD_albedo_AOD_OPACITY_THREDDS_CAMS_NASA_modelready.csv
        |
        v
v20_0_physical_preprocessing.py
        |
        v
train_physical_preprocessed.csv
test_physical_preprocessed.csv
        |
        v
v28_0_reproduce.py
```

## Credentials

Some sources require user-level authentication:

- CAMS ADS: local `cams_credentials.local.json`, or set `CAMS_CREDENTIALS_PATH`
- EUMETSAT Data Store / SARAH-3: local `eumdac_credentials.local.json`, or set `EUMDAC_CREDENTIALS_PATH`
- Google Earth Engine: authenticated Earth Engine project/account

Do not commit credential files. The repository `.gitignore` excludes
`*.local.json` and common raw/cache folders.

## Scope

The scripts are preserved as the historical source used in this workspace.
They are not guaranteed to be one-command clean-room rebuilds from only public
URLs, because some intermediate joins were produced during iterative research.
For exact v28.0 reproduction, use the frozen artifacts listed in
`../data/ARTIFACT_MANIFEST.csv`.
