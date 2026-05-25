from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RUN_DIR = Path("runs/v14.1")
RAW_DIR = RUN_DIR / "nasa_power_raw"
DIAG_DIR = RUN_DIR / "diagnostics"

TRAIN_PATH = Path("Train_with_LSASAF_MDSSFTD_scaled_modelready.csv")
TEST_PATH = Path("Test_with_LSASAF_MDSSFTD_scaled_modelready.csv")
V1214_VAL_PATH = Path("runs/v12.14/val_predictions.csv")

ID = "ID"
TIME = "timestamp"
STATION = "station"
TARGET = "radiation (W/m2)"
MAX_PRED = 1450.0

POWER_PARAMETERS = [
    "ALLSKY_SFC_SW_DWN",
    "CLRSKY_SFC_SW_DWN",
    "ALLSKY_KT",
    "ALLSKY_TOA_SW_DWN",
    "CLOUD_AMT",
    "T2M",
    "T2MDEW",
    "RH2M",
    "PRECTOTCORR",
    "PS",
    "WS10M",
    "WD10M",
]


def https_url(host_and_path: str) -> str:
    return "https" + ":" + "//" + host_and_path


def ensure_dirs() -> None:
    for path in [RUN_DIR, RAW_DIR, DIAG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TIME] = pd.to_datetime(df[TIME])
    df["year"] = df[TIME].dt.year.astype("int16")
    df["month"] = df[TIME].dt.month.astype("int8")
    df["quarter_hour"] = (df[TIME].dt.hour * 4 + df[TIME].dt.minute // 15).astype("int16")
    elev = df["solar_elevation"].astype(float)
    sin_pos = np.maximum(np.sin(np.deg2rad(elev)), 0.0)
    e0 = 1.0 + 0.033 * np.cos(2.0 * np.pi * df[TIME].dt.dayofyear.astype(float) / 365.25)
    df["clear_sky_proxy"] = (0.75 * 1361.0 * e0 * sin_pos).astype("float32")
    df["hour_floor"] = df[TIME].dt.floor("h")
    df["hour_frac"] = (df[TIME].dt.minute.astype(float) / 60.0).astype("float32")
    df["month_sin"] = np.sin(2.0 * np.pi * df["month"].astype(float) / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * df["month"].astype(float) / 12.0)
    df["hour_sin"] = np.sin(2.0 * np.pi * df["quarter_hour"].astype(float) / 96.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * df["quarter_hour"].astype(float) / 96.0)
    df["dssf_over_clear"] = (
        df["DSSF"].astype(float) / np.maximum(df["clear_sky_proxy"].astype(float), 20.0)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 2.5)
    return df


def metric(y: np.ndarray, pred: np.ndarray, candidate: str) -> dict:
    err = np.asarray(pred, dtype=float) - np.asarray(y, dtype=float)
    mbe = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err * err)))
    return {
        "candidate": candidate,
        "n": int(len(err)),
        "mbe": mbe,
        "abs_mbe": abs(mbe),
        "rmse": rmse,
        "weighted": 0.5 * abs(mbe) + 0.5 * rmse,
    }


def finalize(pred: np.ndarray, solar: np.ndarray) -> np.ndarray:
    out = np.nan_to_num(np.asarray(pred, dtype=np.float64), nan=0.0, posinf=MAX_PRED, neginf=0.0)
    out = np.where(np.asarray(solar, dtype=np.float64) <= -2.0, 0.0, out)
    return np.clip(out, 0.0, MAX_PRED)


def station_catalog(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([train.assign(split="train"), test.assign(split="test")], ignore_index=True)
    rows = []
    for station, group in combined.groupby(STATION, observed=True):
        rows.append(
            {
                STATION: station,
                "station_name": group["station_name"].dropna().mode().iloc[0]
                if not group["station_name"].dropna().empty
                else "",
                "country": group["country"].dropna().mode().iloc[0] if not group["country"].dropna().empty else "",
                "latitude": float(group["Y"].astype(float).median()),
                "longitude": float(group["X"].astype(float).median()),
                "start": group[TIME].min().strftime("%Y%m%d"),
                "end": group[TIME].max().strftime("%Y%m%d"),
                "rows": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(STATION).reset_index(drop=True)


def power_url() -> str:
    return https_url("power.larc.nasa.gov/api/temporal/hourly/point")


def download_station(row: pd.Series, force: bool = False) -> dict:
    out_path = RAW_DIR / f"nasa_power_{row[STATION]}.json"
    if out_path.exists() and not force:
        return {"station": row[STATION], "status": "cached", "path": str(out_path), "bytes": int(out_path.stat().st_size)}
    params = {
        "parameters": ",".join(POWER_PARAMETERS),
        "community": "RE",
        "longitude": f"{float(row['longitude']):.6f}",
        "latitude": f"{float(row['latitude']):.6f}",
        "start": str(row["start"]),
        "end": str(row["end"]),
        "format": "JSON",
        "time-standard": "UTC",
    }
    response = requests.get(power_url(), params=params, timeout=120)
    if response.status_code != 200:
        return {
            "station": row[STATION],
            "status": "error",
            "status_code": int(response.status_code),
            "reason": response.text[:1000],
        }
    out_path.write_text(response.text, encoding="utf-8")
    return {"station": row[STATION], "status": "downloaded", "path": str(out_path), "bytes": int(out_path.stat().st_size)}


def download_all(catalog: pd.DataFrame, smoke: bool, force: bool) -> pd.DataFrame:
    rows = []
    source = catalog.head(2) if smoke else catalog
    for _, row in source.iterrows():
        result = download_station(row, force=force)
        rows.append(result)
        time.sleep(0.2)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(DIAG_DIR / "download_manifest.csv", index=False)
    return manifest


def parse_power_file(path: Path, station: str) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    params = data.get("properties", {}).get("parameter", {})
    frames = []
    for param, values in params.items():
        s = pd.Series(values, name=f"power_{param.lower()}")
        frames.append(s)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1).reset_index(names="power_time_key")
    out["hour_floor"] = pd.to_datetime(out["power_time_key"], format="%Y%m%d%H")
    out[STATION] = station
    out = out.drop(columns=["power_time_key"])
    return out


def build_power_table(catalog: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for station in catalog[STATION].tolist():
        path = RAW_DIR / f"nasa_power_{station}.json"
        if path.exists():
            parsed = parse_power_file(path, station)
            if not parsed.empty:
                parts.append(parsed)
    if not parts:
        return pd.DataFrame()
    power = pd.concat(parts, ignore_index=True)
    numeric_cols = [c for c in power.columns if c.startswith("power_")]
    for col in numeric_cols:
        power[col] = pd.to_numeric(power[col], errors="coerce")
        power[col] = power[col].where(power[col] > -900.0, np.nan)
    power["power_allsky"] = power["power_allsky_sfc_sw_dwn"]
    power["power_clear"] = power["power_clrsky_sfc_sw_dwn"]
    power["power_toa"] = power["power_toa_sw_dwn"]
    power["power_cloud"] = power["power_cloud_amt"]
    power["power_allsky_over_clear"] = (
        power["power_allsky"] / power["power_clear"].replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).clip(0.0, 2.5)
    power["power_clear_over_toa"] = (
        power["power_clear"] / power["power_toa"].replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).clip(0.0, 2.5)
    power.to_csv(RUN_DIR / "nasa_power_hourly_station.csv", index=False)
    return power


def join_power(df: pd.DataFrame, power: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(power, on=[STATION, "hour_floor"], how="left")
    out["power_allsky_minus_dssf"] = out["power_allsky"].astype(float) - out["DSSF"].astype(float)
    out["power_allsky_minus_clear_proxy"] = out["power_allsky"].astype(float) - out["clear_sky_proxy"].astype(float)
    out["power_clear_minus_clear_proxy"] = out["power_clear"].astype(float) - out["clear_sky_proxy"].astype(float)
    out["power_dssf_ratio_gap"] = out["dssf_over_clear"].astype(float) - out["power_allsky_over_clear"].astype(float)
    out["power_temp_minus_tahmo"] = out["power_t2m"].astype(float) - out["temperature (degrees Celsius)"].astype(float)
    return out


CORE_FEATURES = [
    "solar_elevation",
    "DSSF",
    "FRACTION_DIFFUSE",
    "quality_flag",
    "DSSF_missing",
    "FRACTION_DIFFUSE_missing",
    "clear_sky_proxy",
    "dssf_over_clear",
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
]

POWER_FEATURES = [
    "power_allsky",
    "power_clear",
    "power_allsky_kt",
    "power_toa",
    "power_cloud",
    "power_t2m",
    "power_t2mdew",
    "power_rh2m",
    "power_prectotcorr",
    "power_ps",
    "power_ws10m",
    "power_wd10m",
    "power_allsky_over_clear",
    "power_clear_over_toa",
    "power_allsky_minus_dssf",
    "power_allsky_minus_clear_proxy",
    "power_clear_minus_clear_proxy",
    "power_dssf_ratio_gap",
    "power_temp_minus_tahmo",
]


def residual_oof(frame: pd.DataFrame, base_col: str, features: list[str], name: str) -> tuple[np.ndarray, dict]:
    valid = frame[features + [TARGET, base_col, "year", "month"]].replace([np.inf, -np.inf], np.nan).dropna().index
    pred = np.zeros(len(frame), dtype=float)
    pred[:] = np.nan
    groups = frame.loc[valid, ["year", "month"]].drop_duplicates().sort_values(["year", "month"])
    target_resid = frame.loc[valid, TARGET].astype(float).to_numpy() - frame.loc[valid, base_col].astype(float).to_numpy()
    valid_index = np.asarray(valid, dtype=int)
    for _, group in groups.iterrows():
        hold = (frame.loc[valid, "year"] == int(group["year"])) & (frame.loc[valid, "month"] == int(group["month"]))
        hold_pos = np.where(hold.to_numpy())[0]
        train_pos = np.where(~hold.to_numpy())[0]
        if len(hold_pos) == 0 or len(train_pos) < 500:
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=200.0))
        x_train = frame.loc[valid_index[train_pos], features].astype(float).to_numpy()
        x_hold = frame.loc[valid_index[hold_pos], features].astype(float).to_numpy()
        model.fit(x_train, target_resid[train_pos])
        correction = model.predict(x_hold)
        pred[valid_index[hold_pos]] = frame.loc[valid_index[hold_pos], base_col].astype(float).to_numpy() + correction
    missing = np.isnan(pred)
    pred[missing] = frame.loc[missing, base_col].astype(float).to_numpy()
    pred = finalize(pred, frame["solar_elevation"].to_numpy(float))
    m = metric(frame[TARGET].to_numpy(float), pred, name)
    m["coverage"] = float(1.0 - np.mean(missing))
    return pred, m


def validate_champion_residual(train: pd.DataFrame, power: pd.DataFrame) -> dict:
    val = pd.read_csv(V1214_VAL_PATH)
    base_cols = [ID, TARGET, "prediction"]
    feature_cols = [
        ID,
        TIME,
        STATION,
        "solar_elevation",
        "DSSF",
        "FRACTION_DIFFUSE",
        "quality_flag",
        "DSSF_missing",
        "FRACTION_DIFFUSE_missing",
        "temperature (degrees Celsius)",
    ]
    val = val[base_cols].merge(train[feature_cols], on=ID, how="left")
    val = add_features(val)
    val = join_power(val, power)
    val = val.rename(columns={"prediction": "v12_14_base"})
    y = val[TARGET].to_numpy(float)
    base = finalize(val["v12_14_base"].to_numpy(float), val["solar_elevation"].to_numpy(float))
    val["v12_14_base"] = base
    rows = [metric(y, base, "v12_14_base")]
    core_pred, core_metric = residual_oof(val, "v12_14_base", CORE_FEATURES, "core_ridge_residual")
    power_pred, power_metric = residual_oof(val, "v12_14_base", CORE_FEATURES + POWER_FEATURES, "power_ridge_residual")
    rows.extend([core_metric, power_metric])
    val["core_ridge_residual"] = core_pred
    val["power_ridge_residual"] = power_pred
    metrics = pd.DataFrame(rows).sort_values(["weighted", "rmse"])
    metrics.to_csv(DIAG_DIR / "champion_residual_metrics.csv", index=False)
    val[[ID, TARGET, "year", "month", STATION, "solar_elevation", "DSSF", "v12_14_base", "core_ridge_residual", "power_ridge_residual"]].to_csv(
        RUN_DIR / "val_predictions.csv",
        index=False,
    )
    month_rows = []
    for candidate in ["v12_14_base", "core_ridge_residual", "power_ridge_residual"]:
        for month, idx in val.groupby("month", observed=True).groups.items():
            arr = np.asarray(list(idx), dtype=int)
            row = metric(y[arr], val[candidate].to_numpy(float)[arr], candidate)
            row["month"] = int(month)
            month_rows.append(row)
    pd.DataFrame(month_rows).to_csv(DIAG_DIR / "champion_residual_month_metrics.csv", index=False)
    corrs = {}
    resid = y - base
    for col in POWER_FEATURES:
        if col in val.columns:
            x = val[col].astype(float).to_numpy()
            mask = np.isfinite(x) & np.isfinite(resid)
            corrs[f"{col}_corr_with_v12_residual"] = float(np.corrcoef(x[mask], resid[mask])[0, 1]) if mask.sum() > 10 else math.nan
    return {
        "rows": int(len(val)),
        "metrics": metrics.to_dict(orient="records"),
        "correlations": corrs,
    }


def validate_dssf_residual_january(train: pd.DataFrame, power: pd.DataFrame) -> dict:
    val = train[train["month"].isin([1, 3, 5, 7, 9, 11])].copy().reset_index(drop=True)
    val = join_power(val, power)
    val["dssf_base"] = finalize(val["DSSF"].fillna(0).to_numpy(float), val["solar_elevation"].to_numpy(float))
    rows = [metric(val[TARGET].to_numpy(float), val["dssf_base"].to_numpy(float), "dssf_base")]
    core_pred, core_metric = residual_oof(val, "dssf_base", CORE_FEATURES, "core_dssf_residual")
    power_pred, power_metric = residual_oof(val, "dssf_base", CORE_FEATURES + POWER_FEATURES, "power_dssf_residual")
    val["core_dssf_residual"] = core_pred
    val["power_dssf_residual"] = power_pred
    rows.extend([core_metric, power_metric])

    jan = val[val["month"] == 1].copy()
    train_non_jan = val[val["month"] != 1].replace([np.inf, -np.inf], np.nan).dropna(subset=CORE_FEATURES + POWER_FEATURES + [TARGET, "dssf_base"])
    jan_valid = jan.replace([np.inf, -np.inf], np.nan).dropna(subset=CORE_FEATURES + POWER_FEATURES + [TARGET, "dssf_base"])
    jan_metrics = []
    for feature_set, name in [(CORE_FEATURES, "core_train_non_jan_eval_jan"), (CORE_FEATURES + POWER_FEATURES, "power_train_non_jan_eval_jan")]:
        model = make_pipeline(StandardScaler(), Ridge(alpha=200.0))
        y_train = train_non_jan[TARGET].astype(float).to_numpy() - train_non_jan["dssf_base"].astype(float).to_numpy()
        model.fit(train_non_jan[feature_set].astype(float).to_numpy(), y_train)
        corr = model.predict(jan_valid[feature_set].astype(float).to_numpy())
        pred = finalize(jan_valid["dssf_base"].astype(float).to_numpy() + corr, jan_valid["solar_elevation"].to_numpy(float))
        jan_metrics.append(metric(jan_valid[TARGET].to_numpy(float), pred, name))

    metrics = pd.DataFrame(rows).sort_values(["weighted", "rmse"])
    metrics.to_csv(DIAG_DIR / "dssf_residual_metrics.csv", index=False)
    pd.DataFrame(jan_metrics).to_csv(DIAG_DIR / "dssf_residual_january_metrics.csv", index=False)
    return {
        "rows": int(len(val)),
        "metrics": metrics.to_dict(orient="records"),
        "january_metrics": jan_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NASA POWER hourly data and validate residual signal.")
    parser.add_argument("--smoke", action="store_true", help="Download only two stations and skip full validation.")
    parser.add_argument("--force-download", action="store_true", help="Re-download cached NASA POWER files.")
    args = parser.parse_args()

    ensure_dirs()
    usecols = [
        ID,
        TIME,
        STATION,
        "station_name",
        "country",
        "elevation",
        "Y",
        "X",
        "solar_elevation",
        "DSSF",
        "FRACTION_DIFFUSE",
        "quality_flag",
        "DSSF_missing",
        "FRACTION_DIFFUSE_missing",
        "temperature (degrees Celsius)",
    ]
    train = add_features(pd.read_csv(TRAIN_PATH, usecols=usecols + [TARGET]))
    test = add_features(pd.read_csv(TEST_PATH, usecols=usecols))
    catalog = station_catalog(train, test)
    catalog.to_csv(DIAG_DIR / "station_catalog.csv", index=False)
    manifest = download_all(catalog, smoke=args.smoke, force=args.force_download)
    errors = manifest[manifest["status"] == "error"].to_dict(orient="records")
    power = build_power_table(catalog)
    if args.smoke or power.empty:
        result = {"champion_residual": None, "dssf_residual_january": None}
    else:
        result = {
            "champion_residual": validate_champion_residual(train, power),
            "dssf_residual_january": validate_dssf_residual_january(train, power),
        }
    metrics = {
        "run": "v14.1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "hypothesis": (
            "NASA POWER hourly all-sky/clear-sky radiation and cloud amount can explain residual cloud-regime bias "
            "left by LSA SAF DSSF and the v12.14 champion."
        ),
        "source": {
            "api": https_url("power.larc.nasa.gov/api/temporal/hourly/point"),
            "parameters": POWER_PARAMETERS,
            "time_standard": "UTC",
        },
        "download_manifest": manifest.to_dict(orient="records"),
        "download_errors": errors,
        "power_rows": int(len(power)),
        "result": result,
        "submission_created": False,
        "outputs": {
            "hourly_table": str(RUN_DIR / "nasa_power_hourly_station.csv"),
            "val_predictions": str(RUN_DIR / "val_predictions.csv"),
            "champion_metrics": str(DIAG_DIR / "champion_residual_metrics.csv"),
            "dssf_january_metrics": str(DIAG_DIR / "dssf_residual_january_metrics.csv"),
        },
    }
    with (RUN_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with (RUN_DIR / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run": "v14.1",
                "script": "runs/v14.1/v14_1_nasa_power_validation.py",
                "python": ".\\.venv\\Scripts\\python.exe",
                "parameters": POWER_PARAMETERS,
            },
            f,
            indent=2,
        )
    with (RUN_DIR / "train.log").open("w", encoding="utf-8") as f:
        f.write(json.dumps(metrics, indent=2, ensure_ascii=False))
        f.write("\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
