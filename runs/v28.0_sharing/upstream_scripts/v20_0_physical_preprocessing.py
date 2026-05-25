from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


RUN_DIR = Path("runs/v20.0_physical_preprocessing")
DIAG_DIR = RUN_DIR / "diagnostics"

TRAIN_MAIN = Path("Train_with_LSASAF_MDSSFTD_albedo_AOD_OPACITY_THREDDS_CAMS_NASA_modelready.csv")
TEST_MAIN = Path("Test_with_LSASAF_MDSSFTD_albedo_AOD_OPACITY_THREDDS_CAMS_NASA_modelready.csv")
TRAIN_MLST = Path("Train_with_LSASAF_MDSSFTD_MLST_SOLARm3_scaled_modelready_0515.csv")
TEST_MLST = Path("Test_with_LSASAF_MDSSFTD_MLST_SOLARm3_scaled_modelready_0515.csv")
SAMPLE_SUBMISSION = Path("SampleSubmission.csv")

TARGET_COL = "radiation (W/m2)"
TEMP_COL = "temperature (degrees Celsius)"
RH_COL = "relativehumidity (-)"
PRECIP_COL = "precipitation (mm)"

EPS = 1e-6
SOLAR_CONSTANT = 1367.0


def log(message: str) -> None:
    elapsed = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{elapsed}] {message}"
    print(line, flush=True)
    with (RUN_DIR / "train.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def require_paths(paths: Sequence[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files inside PDS: " + ", ".join(missing))


def slug(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "blank"


def as_num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def clip_series(values: pd.Series | np.ndarray, low: float, high: float) -> pd.Series:
    return pd.Series(values, copy=False).clip(lower=low, upper=high)


def safe_ratio(
    numerator: pd.Series | np.ndarray,
    denominator: pd.Series | np.ndarray,
    cap: Optional[float] = None,
    floor: Optional[float] = None,
) -> pd.Series:
    num = pd.Series(numerator, copy=False).astype("float64")
    den = pd.Series(denominator, copy=False).astype("float64")
    out = num / den.where(den.abs() > EPS)
    out = out.replace([np.inf, -np.inf], np.nan)
    if floor is not None or cap is not None:
        out = out.clip(lower=floor, upper=cap)
    return out


def nanmedian_frame(columns: Sequence[pd.Series]) -> pd.Series:
    if not columns:
        raise ValueError("nanmedian_frame requires at least one column")
    arr = np.column_stack([pd.to_numeric(c, errors="coerce").to_numpy(dtype="float64") for c in columns])
    with np.errstate(all="ignore"):
        med = np.nanmedian(arr, axis=1)
    return pd.Series(med)


def nanmean_weighted(values: Sequence[pd.Series], weights: Sequence[pd.Series]) -> pd.Series:
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")
    val_arr = np.column_stack([pd.to_numeric(v, errors="coerce").to_numpy(dtype="float64") for v in values])
    weight_arr = np.column_stack([pd.to_numeric(w, errors="coerce").to_numpy(dtype="float64") for w in weights])
    valid = np.isfinite(val_arr) & np.isfinite(weight_arr) & (weight_arr > 0)
    numerator = np.where(valid, val_arr * weight_arr, 0.0).sum(axis=1)
    denominator = np.where(valid, weight_arr, 0.0).sum(axis=1)
    out = np.full(len(numerator), np.nan, dtype="float64")
    good = denominator > EPS
    out[good] = numerator[good] / denominator[good]
    return pd.Series(out)


def add_cyclic(out: pd.DataFrame, base: pd.Series, period: float, prefix: str) -> None:
    angle = 2.0 * math.pi * pd.to_numeric(base, errors="coerce") / period
    out[f"{prefix}_sin"] = np.sin(angle)
    out[f"{prefix}_cos"] = np.cos(angle)


def read_input(path: Path, smoke_rows: Optional[int]) -> pd.DataFrame:
    kwargs = {"low_memory": False}
    if smoke_rows is not None:
        kwargs["nrows"] = smoke_rows
    return pd.read_csv(path, **kwargs)


def read_mlst(path: Path, smoke_rows: Optional[int]) -> pd.DataFrame:
    mlst_cols = [
        "ID",
        "needs_mlst",
        "max_solar_elevation",
        "max_DSSF",
        "lsa_mlst_not_requested",
        "lsa_mlst_c",
        "lsa_mlst_k",
        "lsa_mlst_missing",
        "lsa_mlst_quality_flag",
        "lsa_mlst_standard_error",
        "lsa_mlst_file_missing",
        "lsa_mlst_merge_missing",
        "lsa_mlst_minus_temperature",
        "lsa_mlst_x_DSSF",
        "lsa_mlst_x_solar_elevation",
        "lsa_mlst_daytime_anomaly",
    ]
    kwargs = {"usecols": lambda c: c in mlst_cols, "low_memory": False}
    if smoke_rows is not None:
        kwargs["nrows"] = smoke_rows
    return pd.read_csv(path, **kwargs)


def merge_mlst(main: pd.DataFrame, mlst_path: Path, smoke_rows: Optional[int]) -> pd.DataFrame:
    mlst = read_mlst(mlst_path, smoke_rows)
    if "ID" not in mlst.columns:
        return main
    mlst = mlst.drop_duplicates("ID")
    return main.merge(mlst, on="ID", how="left", validate="one_to_one")


def make_status_maps(train: pd.DataFrame, test: pd.DataFrame, cols: Sequence[str]) -> Dict[str, List[str]]:
    maps: Dict[str, List[str]] = {}
    for col in cols:
        if col not in train.columns and col not in test.columns:
            continue
        values: List[str] = []
        for df in (train, test):
            if col in df.columns:
                s = df[col].astype("string").fillna("missing").map(lambda x: slug(x))
                values.extend(s.dropna().unique().tolist())
        maps[col] = sorted(set(values))
    return maps


def make_station_impute_maps(train: pd.DataFrame, test: pd.DataFrame, cols: Sequence[str]) -> Dict[str, Dict[str, object]]:
    both = pd.concat(
        [train[["station", *[c for c in cols if c in train.columns]]], test[["station", *[c for c in cols if c in test.columns]]]],
        axis=0,
        ignore_index=True,
    )
    maps: Dict[str, Dict[str, object]] = {}
    for col in cols:
        if col not in both.columns:
            continue
        numeric = pd.to_numeric(both[col], errors="coerce")
        temp = pd.DataFrame({"station": both["station"], col: numeric})
        station_median = temp.groupby("station")[col].median().to_dict()
        global_median = float(numeric.median()) if np.isfinite(numeric.median()) else np.nan
        maps[col] = {"station_median": station_median, "global_median": global_median}
    return maps


def station_fill(df: pd.DataFrame, col: str, maps: Mapping[str, Mapping[str, object]]) -> pd.Series:
    raw = as_num(df, col)
    if col not in maps:
        return raw
    station_map = maps[col].get("station_median", {})
    global_median = maps[col].get("global_median", np.nan)
    fill = df["station"].map(station_map).astype("float64")
    fill = fill.fillna(global_median)
    return raw.fillna(fill)


def add_time_and_geometry(df: pd.DataFrame, out: pd.DataFrame) -> Dict[str, pd.Series]:
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    out["dt_year"] = ts.dt.year
    out["dt_month"] = ts.dt.month
    out["dt_dayofyear"] = ts.dt.dayofyear
    out["dt_day"] = ts.dt.day
    out["dt_hour"] = ts.dt.hour
    out["dt_minute"] = ts.dt.minute
    out["dt_decimal_hour"] = ts.dt.hour + ts.dt.minute / 60.0 + ts.dt.second / 3600.0

    lon = as_num(df, "X")
    lat = as_num(df, "Y")
    local_solar_time = (out["dt_decimal_hour"] + lon / 15.0) % 24.0
    out["phys_local_solar_time"] = local_solar_time

    doy = out["dt_dayofyear"].astype("float64")
    doy_angle = 2.0 * math.pi * (doy - 1.0) / 365.2425
    decl_rad = np.deg2rad(23.44) * np.sin(2.0 * math.pi * (doy - 81.0) / 365.2425)
    lat_rad = np.deg2rad(lat)
    hour_angle_rad = np.deg2rad(15.0 * (local_solar_time - 12.0))
    cosz_formula = np.sin(lat_rad) * np.sin(decl_rad) + np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_angle_rad)
    cosz_formula = pd.Series(cosz_formula).clip(lower=0.0, upper=1.0)

    solar_elev = as_num(df, "solar_elevation")
    cosz_from_elev = pd.Series(np.sin(np.deg2rad(solar_elev))).clip(lower=0.0, upper=1.0)
    cosz_primary = cosz_from_elev.where(cosz_from_elev.notna(), cosz_formula)
    out["phys_solar_declination_rad"] = decl_rad
    out["phys_hour_angle_rad"] = hour_angle_rad
    out["phys_cosz_formula"] = cosz_formula
    out["phys_cosz_from_elevation"] = cosz_from_elev
    out["phys_cosz"] = cosz_primary.clip(lower=0.0, upper=1.0)
    out["phys_sin_elevation"] = np.sin(np.deg2rad(solar_elev))
    out["phys_solar_elevation_clipped"] = solar_elev.clip(lower=-90.0, upper=90.0)

    etr_normal = SOLAR_CONSTANT * (1.0 + 0.033 * np.cos(2.0 * math.pi * doy / 365.2425))
    out["phys_etr_normal"] = etr_normal
    out["phys_etr_horizontal"] = (etr_normal * out["phys_cosz"]).clip(lower=0.0, upper=1500.0)

    out["phys_daylight_flag"] = ((solar_elev > 0.0) & (out["phys_cosz"] > 0.0)).astype("int8")
    out["phys_sun_above_5deg_flag"] = (solar_elev > 5.0).astype("int8")
    out["phys_sun_above_20deg_flag"] = (solar_elev > 20.0).astype("int8")
    out["phys_low_sun_flag"] = ((solar_elev > 0.0) & (solar_elev <= 10.0)).astype("int8")
    out["phys_night_flag"] = (solar_elev <= 0.0).astype("int8")

    add_cyclic(out, out["dt_month"], 12.0, "cyc_month")
    add_cyclic(out, out["dt_dayofyear"], 365.2425, "cyc_doy")
    add_cyclic(out, out["dt_decimal_hour"], 24.0, "cyc_clock_hour")
    add_cyclic(out, out["phys_local_solar_time"], 24.0, "cyc_local_solar_time")

    return {
        "timestamp": ts,
        "solar_elev": solar_elev,
        "cosz": out["phys_cosz"],
        "daylight": out["phys_daylight_flag"],
    }


def add_status_one_hot(
    df: pd.DataFrame,
    out: pd.DataFrame,
    col: str,
    values: Sequence[str],
    prefix: str,
) -> None:
    if col in df.columns:
        s = df[col].astype("string").fillna("missing").map(lambda x: slug(x))
    else:
        s = pd.Series("missing", index=df.index)
    for value in values:
        out[f"{prefix}_status_{value}"] = (s == value).astype("int8")
    out[f"{prefix}_status_missing_like"] = s.str.contains("missing|nan|none|blank", regex=True).fillna(True).astype("int8")
    out[f"{prefix}_status_ok_like"] = s.str.contains("ok|valid|nominal|good", regex=True).fillna(False).astype("int8")


def build_features(
    df: pd.DataFrame,
    split: str,
    status_maps: Mapping[str, Sequence[str]],
    impute_maps: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["ID"] = df["ID"].astype(str)
    out["timestamp"] = df["timestamp"].astype(str)
    out["station"] = df["station"].astype(str)
    out["station_name"] = df.get("station_name", pd.Series("", index=df.index)).astype(str)
    out["country"] = df.get("country", pd.Series("", index=df.index)).astype(str)
    out["installation_height"] = as_num(df, "installation_height")
    out["elevation"] = as_num(df, "elevation")
    out["latitude"] = as_num(df, "Y")
    out["longitude"] = as_num(df, "X")

    geom = add_time_and_geometry(df, out)
    solar_elev = geom["solar_elev"]
    cosz = geom["cosz"]
    daylight = geom["daylight"].astype(bool)

    temp = as_num(df, TEMP_COL)
    rh = as_num(df, RH_COL)
    precip = as_num(df, PRECIP_COL)
    out["weather_temperature_c_phys"] = temp.clip(lower=-15.0, upper=55.0)
    out["weather_temperature_outlier_flag"] = ((temp < -15.0) | (temp > 55.0)).fillna(False).astype("int8")
    out["weather_rh_phys"] = rh.clip(lower=0.0, upper=1.0)
    out["weather_rh_outlier_flag"] = ((rh < 0.0) | (rh > 1.0)).fillna(False).astype("int8")
    out["weather_precip_mm_phys"] = precip.clip(lower=0.0, upper=300.0)
    out["weather_precip_log1p"] = np.log1p(out["weather_precip_mm_phys"].fillna(0.0))
    out["weather_rain_flag"] = (out["weather_precip_mm_phys"] > 0.0).astype("int8")
    out["weather_heavy_rain_flag"] = (out["weather_precip_mm_phys"] >= 10.0).astype("int8")
    sat_vapor_kpa = 0.6108 * np.exp((17.27 * out["weather_temperature_c_phys"]) / (out["weather_temperature_c_phys"] + 237.3))
    out["weather_vpd_kpa"] = (sat_vapor_kpa * (1.0 - out["weather_rh_phys"])).clip(lower=0.0, upper=10.0)
    out["weather_temp_x_rh"] = out["weather_temperature_c_phys"] * out["weather_rh_phys"]

    dssf = as_num(df, "DSSF").clip(lower=0.0, upper=1600.0)
    dssf_raw_scaled = (as_num(df, "DSSF_raw") / 10.0).clip(lower=0.0, upper=1600.0)
    fd = as_num(df, "FRACTION_DIFFUSE").clip(lower=0.0, upper=1.0)
    qflag = as_num(df, "quality_flag")
    qflag_int = qflag.fillna(0).astype("int64")
    qflag_int_arr = qflag_int.to_numpy(dtype=np.int64)
    dssf_missing = as_num(df, "DSSF_missing").fillna(as_num(df, "DSSF_missing_before_fill")).fillna(dssf.isna()).astype("float64")
    fd_missing = as_num(df, "FRACTION_DIFFUSE_missing").fillna(as_num(df, "FRACTION_DIFFUSE_daytime_missing_before_fill")).fillna(fd.isna()).astype("float64")
    out["lsa_dssf_phys"] = dssf
    out["lsa_dssf_night_zeroed"] = dssf.where(daylight, 0.0)
    out["lsa_dssf_raw_scaled"] = dssf_raw_scaled
    out["lsa_dssf_raw_scaled_absdiff"] = (dssf - dssf_raw_scaled).abs()
    out["lsa_fraction_diffuse_phys"] = fd
    out["lsa_dssf_missing_flag"] = (dssf_missing > 0.5).astype("int8")
    out["lsa_fraction_diffuse_missing_flag"] = (fd_missing > 0.5).astype("int8")
    out["lsa_quality_flag"] = qflag
    out["lsa_quality_nonzero_flag"] = (qflag_int != 0).astype("int8")
    out["lsa_satellite_skipped_night_flag"] = as_num(df, "satellite_skipped_night").fillna(0).clip(lower=0, upper=1).astype("int8")
    for bit in range(8):
        out[f"lsa_quality_bit_{bit}"] = ((qflag_int_arr >> bit) & 1).astype("int8")

    cams_toa = as_num(df, "cams_toa_wm2").clip(lower=0.0, upper=1600.0)
    cams_clear = as_num(df, "cams_clear_ghi_wm2").clip(lower=0.0, upper=1600.0)
    cams_ghi = as_num(df, "cams_ghi_wm2").clip(lower=0.0, upper=1600.0)
    cams_bhi = as_num(df, "cams_bhi_wm2").clip(lower=0.0, upper=1600.0)
    cams_dhi = as_num(df, "cams_dhi_wm2").clip(lower=0.0, upper=1600.0)
    cams_bni = as_num(df, "cams_bni_wm2").clip(lower=0.0, upper=1600.0)
    cams_rel = as_num(df, "cams_reliability").clip(lower=0.0, upper=1.0)
    out["cams_toa_phys"] = cams_toa
    out["cams_clear_ghi_phys"] = cams_clear
    out["cams_ghi_phys"] = cams_ghi
    out["cams_bhi_phys"] = cams_bhi
    out["cams_dhi_phys"] = cams_dhi
    out["cams_bni_phys"] = cams_bni
    out["cams_reliability_phys"] = cams_rel
    out["cams_component_gap"] = cams_ghi - (cams_bhi + cams_dhi)
    out["cams_component_gap_rel"] = safe_ratio(out["cams_component_gap"], cams_ghi + 20.0, cap=2.0, floor=-2.0)
    out["cams_clearness_phys"] = safe_ratio(cams_ghi, cams_clear, cap=2.0, floor=0.0)
    out["cams_clear_to_toa"] = safe_ratio(cams_clear, cams_toa, cap=1.5, floor=0.0)
    out["cams_diffuse_fraction_phys"] = as_num(df, "cams_diffuse_fraction").clip(lower=0.0, upper=1.0)
    out["cams_beam_fraction_phys"] = as_num(df, "cams_beam_fraction").clip(lower=0.0, upper=1.2)

    power_allsky = as_num(df, "power_allsky_sfc_sw_dwn").fillna(as_num(df, "power_allsky")).clip(lower=0.0, upper=1600.0)
    power_clear = as_num(df, "power_clrsky_sfc_sw_dwn").fillna(as_num(df, "power_clear")).clip(lower=0.0, upper=1600.0)
    power_toa = as_num(df, "power_toa_sw_dwn").fillna(as_num(df, "power_toa")).clip(lower=0.0, upper=1600.0)
    power_cloud = as_num(df, "power_cloud_amt").fillna(as_num(df, "power_cloud")).clip(lower=0.0, upper=100.0)
    out["power_allsky_phys"] = power_allsky
    out["power_clear_phys"] = power_clear
    out["power_toa_phys"] = power_toa
    out["power_cloud_amt_phys"] = power_cloud
    out["power_hourly_context_flag"] = 1
    out["power_allsky_over_clear_phys"] = safe_ratio(power_allsky, power_clear, cap=2.0, floor=0.0)
    out["power_clear_over_toa_phys"] = safe_ratio(power_clear, power_toa, cap=1.5, floor=0.0)
    out["power_cloud_fraction_phys"] = (power_cloud / 100.0).clip(lower=0.0, upper=1.0)
    out["power_temp_delta_vs_station"] = as_num(df, "power_t2m") - out["weather_temperature_c_phys"]
    out["power_rh_delta_vs_station"] = as_num(df, "power_rh2m") / 100.0 - out["weather_rh_phys"]
    out["power_precip_log1p"] = np.log1p(as_num(df, "power_prectotcorr").clip(lower=0.0, upper=300.0).fillna(0.0))

    clear_ref = nanmedian_frame([cams_clear, power_clear])
    clear_ref = clear_ref.fillna(out["phys_etr_horizontal"] * 0.75)
    clear_ref = clear_ref.clip(lower=0.0, upper=1600.0)
    clear_ref = clear_ref.where(daylight, 0.0)
    toa_ref = nanmedian_frame([cams_toa, power_toa, out["phys_etr_horizontal"]]).fillna(out["phys_etr_horizontal"])
    toa_ref = toa_ref.clip(lower=0.0, upper=1600.0).where(daylight, 0.0)
    out["phys_clear_sky_ref"] = clear_ref
    out["phys_toa_ref"] = toa_ref
    out["phys_clear_to_toa_ref"] = safe_ratio(clear_ref, toa_ref, cap=1.5, floor=0.0)

    out["lsa_dssf_over_clear_ref"] = safe_ratio(out["lsa_dssf_night_zeroed"], clear_ref, cap=2.5, floor=0.0)
    out["lsa_dssf_over_toa_ref"] = safe_ratio(out["lsa_dssf_night_zeroed"], toa_ref, cap=2.0, floor=0.0)
    out["lsa_diffuse_energy_proxy"] = out["lsa_dssf_night_zeroed"] * out["lsa_fraction_diffuse_phys"]
    out["lsa_direct_energy_proxy"] = out["lsa_dssf_night_zeroed"] * (1.0 - out["lsa_fraction_diffuse_phys"])
    out["cams_ghi_over_clear_ref"] = safe_ratio(cams_ghi.where(daylight, 0.0), clear_ref, cap=2.5, floor=0.0)
    out["power_allsky_over_clear_ref"] = safe_ratio(power_allsky.where(daylight, 0.0), clear_ref, cap=2.5, floor=0.0)
    out["clear_ref_minus_lsa_dssf"] = clear_ref - out["lsa_dssf_night_zeroed"]
    out["clear_ref_minus_cams_ghi"] = clear_ref - cams_ghi.where(daylight, 0.0)
    out["clear_ref_minus_power_allsky"] = clear_ref - power_allsky.where(daylight, 0.0)

    albedo = station_fill(df, "albedo_effective", impute_maps).clip(lower=0.03, upper=0.60)
    al_ni = station_fill(df, "al_ni_dh", impute_maps).clip(lower=0.03, upper=0.80)
    al_vi = station_fill(df, "al_vi_dh", impute_maps).clip(lower=0.03, upper=0.80)
    out["albedo_effective_phys"] = albedo
    out["albedo_missing_flag"] = as_num(df, "albedo_effective").isna().astype("int8")
    out["albedo_is_valid_flag"] = as_num(df, "albedo_is_valid").fillna(0).clip(lower=0, upper=1).astype("int8")
    out["albedo_qflag"] = as_num(df, "albedo_qflag")
    out["albedo_age_log_phys"] = as_num(df, "albedo_age_log").fillna(np.log1p(as_num(df, "albedo_z_age").clip(lower=0.0))).clip(lower=0.0, upper=10.0)
    out["albedo_ni_vi_ratio_phys"] = safe_ratio(al_ni, al_vi, cap=5.0, floor=0.0)
    out["albedo_surface_reflection_potential"] = albedo * clear_ref * cosz

    aod = as_num(df, "AOD").fillna(as_num(df, "AOD_raw")).clip(lower=0.0, upper=3.0)
    opacity = as_num(df, "OPACITY_INDEX").fillna(as_num(df, "OPACITY_INDEX_raw")).clip(lower=0.0, upper=1.0)
    out["aod_phys"] = aod
    out["aod_log1p"] = np.log1p(aod)
    out["aod_missing_flag"] = as_num(df, "AOD_missing_before_fill").fillna(as_num(df, "AOD_raw").isna()).astype("float64").clip(0, 1).astype("int8")
    out["aod_high_flag"] = (aod >= 0.70).astype("int8")
    out["aod_extreme_flag"] = (aod >= 1.20).astype("int8")
    out["opacity_phys"] = opacity
    out["opacity_missing_flag"] = as_num(df, "OPACITY_INDEX_missing_before_fill").fillna(as_num(df, "OPACITY_INDEX_raw").isna()).astype("float64").clip(0, 1).astype("int8")
    out["opacity_cloudy_flag"] = (opacity >= 0.60).astype("int8")
    out["opacity_opaque_flag"] = (opacity >= 0.90).astype("int8")
    out["aod_x_day_cosz"] = aod * cosz * out["phys_daylight_flag"]
    out["opacity_x_day_cosz"] = opacity * cosz * out["phys_daylight_flag"]
    out["aod_x_clear_ref"] = aod * clear_ref
    out["opacity_x_clear_ref"] = opacity * clear_ref
    add_status_one_hot(df, out, "AOD_status", status_maps.get("AOD_status", []), "aod")
    add_status_one_hot(df, out, "OPACITY_INDEX_status", status_maps.get("OPACITY_INDEX_status", []), "opacity")
    add_status_one_hot(df, out, "Q_FLAG_HDF5_status", status_maps.get("Q_FLAG_HDF5_status", []), "qflag_hdf5")

    mlst_c_raw = as_num(df, "lsa_mlst_c")
    mlst_k_c = as_num(df, "lsa_mlst_k") - 273.15
    mlst_c = mlst_c_raw.where(mlst_c_raw.notna(), mlst_k_c).clip(lower=-25.0, upper=85.0)
    mlst_se = as_num(df, "lsa_mlst_standard_error").clip(lower=0.0, upper=30.0)
    mlst_missing = as_num(df, "lsa_mlst_missing").fillna(1.0)
    mlst_qflag = as_num(df, "lsa_mlst_quality_flag")
    mlst_valid = (mlst_c.notna() & (mlst_missing < 0.5) & (mlst_se <= 10.0)).astype("int8")
    out["mlst_c_phys"] = mlst_c
    out["mlst_missing_flag"] = (mlst_missing >= 0.5).astype("int8")
    out["mlst_valid_flag"] = mlst_valid
    out["mlst_quality_flag"] = mlst_qflag
    out["mlst_standard_error_phys"] = mlst_se
    out["mlst_air_temperature_delta"] = (mlst_c - out["weather_temperature_c_phys"]).clip(lower=-40.0, upper=70.0)
    out["mlst_daytime_delta"] = out["mlst_air_temperature_delta"].where(daylight, np.nan)
    out["mlst_x_clear_ref"] = mlst_c * clear_ref
    out["mlst_x_day_cosz"] = mlst_c * cosz * out["phys_daylight_flag"]
    out["mlst_not_requested_flag"] = as_num(df, "lsa_mlst_not_requested").fillna(1).clip(0, 1).astype("int8")
    out["mlst_file_missing_flag"] = as_num(df, "lsa_mlst_file_missing").fillna(1).clip(0, 1).astype("int8")
    out["mlst_merge_missing_flag"] = as_num(df, "lsa_mlst_merge_missing").fillna(1).clip(0, 1).astype("int8")

    lsa_valid_weight = (
        0.45
        * out["phys_daylight_flag"]
        * (1.0 - out["lsa_dssf_missing_flag"])
        * (1.0 - out["lsa_quality_nonzero_flag"] * 0.25)
    ).clip(lower=0.0, upper=0.45)
    cams_weight = (0.35 * out["phys_daylight_flag"] * cams_rel.fillna(0.75)).clip(lower=0.0, upper=0.35)
    power_weight = (0.20 * out["phys_daylight_flag"] * (1.0 - 0.35 * out["power_cloud_fraction_phys"].fillna(0.0))).clip(lower=0.03, upper=0.20)
    allsky_median = nanmedian_frame([
        out["lsa_dssf_night_zeroed"],
        cams_ghi.where(daylight, 0.0),
        power_allsky.where(daylight, 0.0),
    ])
    allsky_mean = pd.concat(
        [out["lsa_dssf_night_zeroed"], cams_ghi.where(daylight, 0.0), power_allsky.where(daylight, 0.0)], axis=1
    ).mean(axis=1, skipna=True)
    allsky_std = pd.concat(
        [out["lsa_dssf_night_zeroed"], cams_ghi.where(daylight, 0.0), power_allsky.where(daylight, 0.0)], axis=1
    ).std(axis=1, skipna=True)
    weighted = nanmean_weighted(
        [out["lsa_dssf_night_zeroed"], cams_ghi.where(daylight, 0.0), power_allsky.where(daylight, 0.0)],
        [lsa_valid_weight, cams_weight, power_weight],
    )
    out["fusion_allsky_median"] = allsky_median.where(daylight, 0.0)
    out["fusion_allsky_mean"] = allsky_mean.where(daylight, 0.0)
    out["fusion_allsky_std"] = allsky_std.where(daylight, 0.0)
    out["fusion_allsky_cv"] = safe_ratio(out["fusion_allsky_std"], out["fusion_allsky_median"] + 30.0, cap=5.0, floor=0.0)
    out["fusion_weighted_allsky"] = weighted.fillna(allsky_median).where(daylight, 0.0)
    out["fusion_weight_lsa"] = lsa_valid_weight
    out["fusion_weight_cams"] = cams_weight
    out["fusion_weight_power"] = power_weight
    out["fusion_external_clear_index_median"] = safe_ratio(out["fusion_allsky_median"], clear_ref, cap=2.5, floor=0.0)
    out["fusion_external_clear_index_weighted"] = safe_ratio(out["fusion_weighted_allsky"], clear_ref, cap=2.5, floor=0.0)
    out["fusion_cloud_loss_median"] = clear_ref - out["fusion_allsky_median"]
    out["fusion_sensor_disagreement"] = (
        pd.concat([out["lsa_dssf_night_zeroed"], cams_ghi.where(daylight, 0.0), power_allsky.where(daylight, 0.0)], axis=1).max(axis=1, skipna=True)
        - pd.concat([out["lsa_dssf_night_zeroed"], cams_ghi.where(daylight, 0.0), power_allsky.where(daylight, 0.0)], axis=1).min(axis=1, skipna=True)
    ).where(daylight, 0.0)
    out["fusion_reliability_score"] = (
        (1.0 - out["fusion_allsky_cv"].fillna(1.0).clip(0.0, 1.0))
        * (0.4 + 0.6 * cams_rel.fillna(0.5))
        * (1.0 - 0.2 * out["lsa_dssf_missing_flag"])
    ).clip(lower=0.0, upper=1.0)

    out["regime_clear_high_sun_flag"] = (
        (out["phys_sun_above_20deg_flag"] == 1)
        & (out["fusion_external_clear_index_median"] >= 0.75)
        & (out["aod_phys"] < 0.7)
        & (out["opacity_phys"] < 0.5)
    ).astype("int8")
    out["regime_hazy_flag"] = ((out["aod_phys"] >= 0.7) & (out["phys_daylight_flag"] == 1)).astype("int8")
    out["regime_cloudy_flag"] = (
        ((out["opacity_phys"] >= 0.6) | (out["fusion_external_clear_index_median"] <= 0.45))
        & (out["phys_daylight_flag"] == 1)
    ).astype("int8")
    out["regime_low_sun_cloud_sensitive_flag"] = (
        (out["phys_low_sun_flag"] == 1) & ((out["opacity_phys"] >= 0.4) | (out["aod_phys"] >= 0.5))
    ).astype("int8")

    if TARGET_COL in df.columns:
        target = as_num(df, TARGET_COL)
        out[TARGET_COL] = target
        out["target_clean_phys"] = target.clip(lower=0.0, upper=1600.0)
        out["target_clear_index"] = safe_ratio(out["target_clean_phys"].where(daylight, 0.0), clear_ref, cap=3.0, floor=0.0)
        out["target_over_fusion_median"] = safe_ratio(out["target_clean_phys"].where(daylight, 0.0), out["fusion_allsky_median"], cap=3.0, floor=0.0)
        out["target_over_fusion_weighted"] = safe_ratio(out["target_clean_phys"].where(daylight, 0.0), out["fusion_weighted_allsky"], cap=3.0, floor=0.0)
        out["target_suspicious_daytime_zero_flag"] = (
            (target <= 1.0) & (solar_elev > 5.0) & (clear_ref > 150.0)
        ).fillna(False).astype("int8")

    return out.copy()


def data_contract(train_raw: pd.DataFrame, test_raw: pd.DataFrame, train_out: pd.DataFrame, test_out: pd.DataFrame) -> Dict[str, object]:
    def split_contract(raw: pd.DataFrame, out: pd.DataFrame, split: str) -> Dict[str, object]:
        ts = pd.to_datetime(raw["timestamp"], errors="coerce")
        info: Dict[str, object] = {
            "split": split,
            "raw_rows": int(len(raw)),
            "raw_columns": int(raw.shape[1]),
            "output_rows": int(len(out)),
            "output_columns": int(out.shape[1]),
            "id_unique": bool(raw["ID"].is_unique),
            "duplicate_id_count": int(raw["ID"].duplicated().sum()),
            "station_count": int(raw["station"].nunique()),
            "timestamp_min": str(ts.min()),
            "timestamp_max": str(ts.max()),
            "month_counts": {str(k): int(v) for k, v in ts.dt.month.value_counts().sort_index().items()},
            "missing_timestamp_count": int(ts.isna().sum()),
        }
        if TARGET_COL in raw.columns:
            target = pd.to_numeric(raw[TARGET_COL], errors="coerce")
            info["target_stats"] = {
                "missing": int(target.isna().sum()),
                "min": float(target.min()),
                "mean": float(target.mean()),
                "median": float(target.median()),
                "max": float(target.max()),
            }
            if "target_suspicious_daytime_zero_flag" in out.columns:
                info["target_suspicious_daytime_zero_rate"] = float(out["target_suspicious_daytime_zero_flag"].mean())
        return info

    train_feature_cols = set(train_out.columns) - {TARGET_COL, "target_clean_phys", "target_clear_index", "target_over_fusion_median", "target_over_fusion_weighted", "target_suspicious_daytime_zero_flag"}
    test_feature_cols = set(test_out.columns)
    return {
        "train": split_contract(train_raw, train_out, "train"),
        "test": split_contract(test_raw, test_out, "test"),
        "feature_column_mismatch_train_minus_test": sorted(train_feature_cols - test_feature_cols),
        "feature_column_mismatch_test_minus_train": sorted(test_feature_cols - train_feature_cols),
    }


def write_feature_summary(df: pd.DataFrame, path: Path) -> None:
    numeric = df.select_dtypes(include=[np.number])
    rows = []
    for col in numeric.columns:
        s = numeric[col]
        rows.append(
            {
                "column": col,
                "missing_rate": float(s.isna().mean()),
                "min": float(s.min()) if s.notna().any() else np.nan,
                "p01": float(s.quantile(0.01)) if s.notna().any() else np.nan,
                "p50": float(s.quantile(0.50)) if s.notna().any() else np.nan,
                "p99": float(s.quantile(0.99)) if s.notna().any() else np.nan,
                "max": float(s.max()) if s.notna().any() else np.nan,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_station_month_diagnostics(train_out: pd.DataFrame, path: Path) -> None:
    if TARGET_COL not in train_out.columns:
        return
    day = train_out[train_out["phys_daylight_flag"] == 1].copy()
    if day.empty:
        return
    grouped = (
        day.groupby(["station", "dt_month"], dropna=False)
        .agg(
            n=("ID", "size"),
            target_mean=(TARGET_COL, "mean"),
            target_median=(TARGET_COL, "median"),
            target_clear_index_median=("target_clear_index", "median"),
            target_suspicious_zero_rate=("target_suspicious_daytime_zero_flag", "mean"),
            fusion_weighted_mean=("fusion_weighted_allsky", "mean"),
            fusion_clear_index_median=("fusion_external_clear_index_weighted", "median"),
            fusion_cv_mean=("fusion_allsky_cv", "mean"),
            lsa_missing_rate=("lsa_dssf_missing_flag", "mean"),
            clear_ref_mean=("phys_clear_sky_ref", "mean"),
            aod_mean=("aod_phys", "mean"),
            opacity_mean=("opacity_phys", "mean"),
        )
        .reset_index()
    )
    state = np.full(len(grouped), "normal", dtype=object)
    state[(grouped["target_suspicious_zero_rate"] >= 0.40) & (grouped["clear_ref_mean"] > 150.0)] = "daytime_zero_suspicious"
    state[(grouped["target_clear_index_median"] < 0.35) & (grouped["fusion_clear_index_median"] > 0.55)] = "low_gain_suspicious"
    state[(grouped["target_clear_index_median"] > 1.25) & (grouped["n"] >= 50)] = "high_gain_suspicious"
    state[(grouped["fusion_cv_mean"] > 0.45) & (grouped["n"] >= 50)] = "external_disagreement"
    grouped["sensor_state_label"] = state
    grouped.to_csv(path, index=False)


def write_json(obj: Mapping[str, object], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_summary_md(paths: Mapping[str, str], contract: Mapping[str, object], smoke: bool) -> None:
    suffix = " smoke" if smoke else ""
    md = f"""# v20.0 Physical Preprocessing{suffix}

## Purpose
This run rebuilds the feature table from physical assumptions only. It does not train a model and does not create a submission. Raw competition files are not modified.

## Input Tables
- Main train: `{TRAIN_MAIN.name}`
- Main test: `{TEST_MAIN.name}`
- MLST/LST train: `{TRAIN_MLST.name}`
- MLST/LST test: `{TEST_MLST.name}`

## Physical Design
- Solar geometry: timestamp, latitude, longitude, solar elevation, cosine SZA, local solar time, extraterrestrial horizontal radiation, day/night and low-sun regimes.
- LSASAF: DSSF and fraction diffuse are clipped to physical ranges, raw scaled DSSF consistency is tracked, night values are zeroed only in derived columns, and quality flag bits are exposed.
- CAMS and NASA POWER: all-sky, clear-sky, TOA, cloud and component radiation are clipped and converted to clear-sky index style ratios.
- AOD/OPACITY: non-negative clipping, log-AOD, hazy/opaque flags, status one-hot columns, and interactions with daylight/clear-sky radiation.
- Albedo: station/global median imputation, physical clipping, albedo ratio features, and surface reflection potential.
- MLST/LST: Kelvin/Celsius consistency, quality/missing/error flags, land-air temperature deltas, and daylight interactions.
- Sensor fusion: LSASAF, CAMS and POWER are blended as a reliability-weighted external all-sky reference; disagreement and reliability scores are kept as diagnostics.
- Target diagnostics: train-only target clear-sky index and suspicious daytime zero flags are added for sensor-state analysis, not as test features.

## Outputs
- Train features: `{paths["train_features"]}`
- Test features: `{paths["test_features"]}`
- Data contract: `{paths["data_contract"]}`
- Feature summary: `{paths["feature_summary"]}`
- Station/month diagnostics: `{paths["station_month"]}`

## Contract Snapshot
- Train rows: `{contract.get("train", {}).get("raw_rows")}`
- Test rows: `{contract.get("test", {}).get("raw_rows")}`
- Train stations: `{contract.get("train", {}).get("station_count")}`
- Test stations: `{contract.get("test", {}).get("station_count")}`
- Feature mismatches train-test after excluding train-only target diagnostics: `{len(contract.get("feature_column_mismatch_train_minus_test", [])) + len(contract.get("feature_column_mismatch_test_minus_train", []))}`

## How To Run
Smoke:
```powershell
.\\.venv\\Scripts\\python.exe .\\runs\\v20.0_physical_preprocessing\\v20_0_physical_preprocessing.py --smoke-test
```

Full:
```powershell
.\\.venv\\Scripts\\python.exe .\\runs\\v20.0_physical_preprocessing\\v20_0_physical_preprocessing.py --mode full
```
"""
    (RUN_DIR / ("V20_0_PHYSICAL_PREPROCESSING_SMOKE_SUMMARY.md" if smoke else "V20_0_PHYSICAL_PREPROCESSING_SUMMARY.md")).write_text(md, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="v20.0 physical preprocessing for TAHMO solar radiation challenge")
    parser.add_argument("--mode", choices=["full"], default="full")
    parser.add_argument("--smoke-test", action="store_true", help="process a small head sample and write *_smoke outputs")
    parser.add_argument("--smoke-rows", type=int, default=50000)
    args = parser.parse_args(argv)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "train.log").write_text("", encoding="utf-8")

    require_paths([TRAIN_MAIN, TEST_MAIN, TRAIN_MLST, TEST_MLST, SAMPLE_SUBMISSION])
    smoke_rows = args.smoke_rows if args.smoke_test else None
    suffix = "_smoke" if args.smoke_test else ""

    config = {
        "version": "v20.0_physical_preprocessing",
        "mode": args.mode,
        "smoke_test": bool(args.smoke_test),
        "smoke_rows": smoke_rows,
        "inputs": {
            "train_main": str(TRAIN_MAIN),
            "test_main": str(TEST_MAIN),
            "train_mlst": str(TRAIN_MLST),
            "test_mlst": str(TEST_MLST),
            "sample_submission": str(SAMPLE_SUBMISSION),
        },
        "target": TARGET_COL,
        "notes": "Physical preprocessing only. No model training and no submission generation.",
    }
    write_json(config, RUN_DIR / f"config{suffix}.json")

    log(f"Reading main inputs smoke_rows={smoke_rows}")
    train_raw = read_input(TRAIN_MAIN, smoke_rows)
    test_raw = read_input(TEST_MAIN, smoke_rows)

    log("Merging MLST/LST inputs by ID")
    train_raw = merge_mlst(train_raw, TRAIN_MLST, smoke_rows)
    test_raw = merge_mlst(test_raw, TEST_MLST, smoke_rows)

    status_maps = make_status_maps(train_raw, test_raw, ["AOD_status", "OPACITY_INDEX_status", "Q_FLAG_HDF5_status"])
    impute_maps = make_station_impute_maps(
        train_raw,
        test_raw,
        ["albedo_effective", "al_ni_dh", "al_vi_dh"],
    )

    log("Building train physical features")
    train_out = build_features(train_raw, "train", status_maps, impute_maps)
    log("Building test physical features")
    test_out = build_features(test_raw, "test", status_maps, impute_maps)

    contract = data_contract(train_raw, test_raw, train_out, test_out)
    write_json(contract, DIAG_DIR / f"data_contract{suffix}.json")

    train_path = RUN_DIR / f"train_physical_preprocessed{suffix}.csv"
    test_path = RUN_DIR / f"test_physical_preprocessed{suffix}.csv"
    log(f"Writing features: {train_path}")
    train_out.to_csv(train_path, index=False)
    log(f"Writing features: {test_path}")
    test_out.to_csv(test_path, index=False)

    log("Writing diagnostics")
    write_feature_summary(train_out, DIAG_DIR / f"feature_summary_train{suffix}.csv")
    write_feature_summary(test_out, DIAG_DIR / f"feature_summary_test{suffix}.csv")
    write_station_month_diagnostics(train_out, DIAG_DIR / f"station_month_sensor_state{suffix}.csv")

    metrics = {
        "mode": "preprocessing_only",
        "train_rows": int(len(train_out)),
        "test_rows": int(len(test_out)),
        "train_columns": int(train_out.shape[1]),
        "test_columns": int(test_out.shape[1]),
        "train_feature_output": str(train_path),
        "test_feature_output": str(test_path),
        "data_contract": str(DIAG_DIR / f"data_contract{suffix}.json"),
        "feature_mismatch_train_minus_test": contract["feature_column_mismatch_train_minus_test"],
        "feature_mismatch_test_minus_train": contract["feature_column_mismatch_test_minus_train"],
    }
    write_json(metrics, RUN_DIR / f"metrics{suffix}.json")

    write_summary_md(
        {
            "train_features": str(train_path),
            "test_features": str(test_path),
            "data_contract": str(DIAG_DIR / f"data_contract{suffix}.json"),
            "feature_summary": str(DIAG_DIR / f"feature_summary_train{suffix}.csv"),
            "station_month": str(DIAG_DIR / f"station_month_sensor_state{suffix}.csv"),
        },
        contract,
        args.smoke_test,
    )

    log("Completed v20.0 physical preprocessing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
