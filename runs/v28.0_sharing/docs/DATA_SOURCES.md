# Data Sources and Download Lineage

This package uses the competition-provided TAHMO files plus public or
competition-allowed external radiation and atmospheric data. Large downloaded
or joined data files are not committed; their final-stage artifact checksums
are listed in `../data/ARTIFACT_MANIFEST.csv`.

## Competition Data

| source | files | role |
|---|---|---|
| Zindi TAHMO challenge data | `Train.csv`, `Test.csv`, `SampleSubmission.csv`, `dataset_data_dictionary.csv` | Base station metadata, weather observations, target radiation, test ID order. |

The raw competition files are treated as immutable. All joins and feature
generation write new derived files.

## External Sources

| source | access | scripts in this repo | variables used |
|---|---|---|---|
| LSA-SAF / EUMETSAT MSG MDSSFTD | LSA-SAF data service / downloaded product files | upstream lineage plus `preprocessing/v20_0_physical_preprocessing.py` | `DSSF`, `FRACTION_DIFFUSE`, quality/missing flags, AOD/opacity-style auxiliary signals where available. |
| NASA POWER Hourly API | public HTTPS API | `preprocessing/v14_1_nasa_power_validation.py` | all-sky SW down, clear-sky SW down, clearness index, cloud amount, temperature, dew point, RH, precipitation, pressure, wind. |
| CAMS Solar Radiation Time Series | Copernicus Atmosphere Data Store API, user credentials required | `preprocessing/v14_2_cams_radiation_validation.py` | 15-minute all-sky/clear-sky GHI, beam/diffuse components, reliability flag. |
| Sentinel-5P/TROPOMI CLOUD | Google Earth Engine collection `COPERNICUS/S5P/OFFL/L3_CLOUD`, Earth Engine authentication required | `preprocessing/tahmo_tropomi_cloud_downloader_v3_daily.py`, `preprocessing/merge_tropomi_cloud_features.py` | daily station cloud fraction, cloud top/base pressure/height, optical depth, surface albedo, viewing/solar angles. |
| EUMETSAT SARAH-3 | EUMETSAT Data Store collection `EO:EUM:DAT:0863`, `eumdac` credentials required | `preprocessing/v25_1_sarah3_access_audit.py`, `preprocessing/v25_6_sarah3_multi_day_pilot.py` | pilot SIS and effective cloud albedo station series. |

Official source URLs:

- NASA POWER Hourly API: `https://power.larc.nasa.gov/docs/services/api/temporal/hourly/`
- CAMS ADS catalogue: `https://ads.atmosphere.copernicus.eu/datasets`
- Google Earth Engine Sentinel-5P CLOUD catalogue: `https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_CLOUD`
- EUMETSAT SARAH-3 collection: `https://api.eumetsat.int/data/browse/collections/EO%3AEUM%3ADAT%3A0863?format=html`
- LSA-SAF radiation products: `https://lsa-saf.eumetsat.int/en/data/products/radiation/`

## Example Download and Preprocessing Commands

Run commands from the repository root after placing the competition CSVs in the
root folder. Network/API steps are intentionally separated from final-stage
reproduction.

NASA POWER station-hour table:

```powershell
.\.venv\Scripts\python.exe .\runs\v28.0_sharing\preprocessing\v14_1_nasa_power_validation.py --smoke
```

CAMS Solar Radiation Time Series, using a local ADS credential file expected by
the script:

```powershell
.\.venv\Scripts\python.exe .\runs\v28.0_sharing\preprocessing\v14_2_cams_radiation_validation.py --smoke
```

TROPOMI CLOUD daily station features through Google Earth Engine:

```powershell
.\.venv\Scripts\python.exe .\runs\v28.0_sharing\preprocessing\tahmo_tropomi_cloud_downloader_v3_daily.py --train Train.csv --test Test.csv --out_dir data_external\tropomi_cloud_v3 --project YOUR_EE_PROJECT --buffer_m 10000 --sleep 0.5
```

Merge TROPOMI daily features into model-ready train/test files:

```powershell
.\.venv\Scripts\python.exe .\runs\v28.0_sharing\preprocessing\merge_tropomi_cloud_features.py --train Train_with_LSASAF_MDSSFTD_scaled_modelready.csv --test Test_with_LSASAF_MDSSFTD_scaled_modelready.csv --tropomi data_external\tropomi_cloud_v3\tropomi_cloud_station_daily_full_grid.csv --out_dir data_external\tropomi_cloud_merged
```

SARAH-3 access audit and pilot extraction:

```powershell
.\.venv\Scripts\python.exe .\runs\v28.0_sharing\preprocessing\v25_1_sarah3_access_audit.py --smoke-test
.\.venv\Scripts\python.exe .\runs\v28.0_sharing\preprocessing\v25_6_sarah3_multi_day_pilot.py --dates 2016-01-12 --types SIS,CAL --download
```

Final physical preprocessing once the large model-ready CSVs exist:

```powershell
.\.venv\Scripts\python.exe .\runs\v28.0_sharing\preprocessing\v20_0_physical_preprocessing.py --smoke-test
```

## Join and Resampling Policy

All sources are aligned to the TAHMO station-time grid:

- spatial key: station latitude/longitude;
- temporal key: UTC timestamp unless the source explicitly returns a station
  point time series already aligned to UTC;
- target grid: 15-minute rows from the competition `Train.csv` and `Test.csv`;
- hourly products are merged by station and hour, then reused by the four
  quarter-hour rows in that hour;
- daily products are merged by station and date with explicit missing flags;
- satellite grid products are extracted by nearest station grid cell or by
  station-buffer aggregation, depending on the source script.

Derived radiation features are built in physically normalized forms whenever
possible:

```text
clear_sky_index = all_sky_radiation / clear_sky_radiation
satellite_residual = station_or_anchor_prediction - satellite_radiation
cloudiness_proxy = 1 - all_sky_radiation / clear_sky_radiation
source_disagreement = std(LSA-SAF, CAMS, NASA POWER)
```

## Rebuild Levels

This repository supports two different levels of reproducibility:

| level | supported by | notes |
|---|---|---|
| Exact v28.0 final-stage reproduction | `v28_0_reproduce.py` plus files in `ARTIFACT_MANIFEST.csv` | Preferred code-review path. No network calls. |
| External data lineage review | scripts under `preprocessing/` | Shows how external sources were downloaded and joined. Requires source accounts/credentials for CAMS, Earth Engine, or EUMETSAT. |
| Full historical clean rebuild | not a single-command pipeline | Earlier research joins produced large intermediate model-ready CSVs. The source scripts and lineage are included, but frozen artifacts remain the authoritative final-stage inputs. |

## Credential and Cache Policy

Credential files must stay local and are intentionally ignored:

```text
*.local.json
```

Raw downloads, API caches, NetCDF files, and large joined CSVs are also kept out
of Git. This prevents accidental publication of credentials or very large data
files while keeping the source code and exact artifact manifest reviewable.
