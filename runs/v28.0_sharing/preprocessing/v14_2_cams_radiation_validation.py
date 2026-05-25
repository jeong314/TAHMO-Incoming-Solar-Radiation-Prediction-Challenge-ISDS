from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RUN_DIR = Path("runs/v14.2")
RAW_DIR = RUN_DIR / "cams_raw"
DIAG_DIR = RUN_DIR / "diagnostics"

TRAIN_PATH = Path("Train_with_LSASAF_MDSSFTD_scaled_modelready.csv")
TEST_PATH = Path("Test_with_LSASAF_MDSSFTD_scaled_modelready.csv")
V1214_VAL_PATH = Path("runs/v12.14/val_predictions.csv")
CREDENTIALS_PATH = Path(
    os.environ.get(
        "CAMS_CREDENTIALS_PATH",
        Path(__file__).resolve().parent / "cams_credentials.local.json",
    )
)

ID = "ID"
TIME = "timestamp"
STATION = "station"
TARGET = "radiation (W/m2)"
MAX_PRED = 1450.0

CAMS_COLUMNS = [
    "observation_period",
    "cams_toa_whm2",
    "cams_clear_ghi_whm2",
    "cams_clear_bhi_whm2",
    "cams_clear_dhi_whm2",
    "cams_clear_bni_whm2",
    "cams_ghi_whm2",
    "cams_bhi_whm2",
    "cams_dhi_whm2",
    "cams_bni_whm2",
    "cams_reliability",
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
                "elevation_m": float(group["elevation"].astype(float).median()),
                "start": group[TIME].min().strftime("%Y-%m-%d"),
                "end": group[TIME].max().strftime("%Y-%m-%d"),
                "rows": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(STATION).reset_index(drop=True)


def build_request(row: pd.Series) -> dict:
    return {
        "sky_type": "observed_cloud",
        "location": {
            "latitude": round(float(row["latitude"]), 6),
            "longitude": round(float(row["longitude"]), 6),
        },
        "altitude": f"{float(row['elevation_m']):.1f}",
        "date": f"{row['start']}/{row['end']}",
        "time_step": "15minute",
        "time_reference": "universal_time",
        "format": "csv",
    }


def download_cams(catalog: pd.DataFrame, smoke: bool, force: bool) -> pd.DataFrame:
    import cdsapi  # type: ignore

    creds = json.loads(CREDENTIALS_PATH.read_text(encoding="utf-8"))
    key = creds.get("key")
    if not key:
        raise ValueError("CAMS credentials file does not contain key")
    url = creds.get("url", https_url("ads.atmosphere.copernicus.eu/api"))
    client = cdsapi.Client(url=url, key=key, quiet=True)
    rows = []
    source = catalog.head(2) if smoke else catalog
    for _, row in source.iterrows():
        target = RAW_DIR / f"camsrad_{row[STATION]}_{row['start']}_{row['end']}.csv"
        if target.exists() and not force:
            rows.append({"station": row[STATION], "status": "cached", "target": str(target), "bytes": int(target.stat().st_size)})
            continue
        try:
            client.retrieve("cams-solar-radiation-timeseries", build_request(row), str(target))
            rows.append({"station": row[STATION], "status": "downloaded", "target": str(target), "bytes": int(target.stat().st_size)})
        except Exception as exc:  # noqa: BLE001
            rows.append({"station": row[STATION], "status": "error", "reason": str(exc)[:1200]})
        time.sleep(0.2)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(DIAG_DIR / "download_manifest.csv", index=False)
    return manifest


def parse_cams_file(path: Path, station: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", comment="#", names=CAMS_COLUMNS, engine="python")
    if df.empty:
        return pd.DataFrame()
    df = df[df["observation_period"].astype(str).str.contains("/", regex=False)].copy()
    df[TIME] = pd.to_datetime(df["observation_period"].astype(str).str.split("/", n=1).str[0])
    df[STATION] = station
    value_cols = [c for c in df.columns if c.endswith("_whm2") or c == "cams_reliability"]
    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    wh_cols = [c for c in df.columns if c.endswith("_whm2")]
    for col in wh_cols:
        df[col.replace("_whm2", "_wm2")] = df[col].astype(float) * 4.0
    return df[[STATION, TIME] + [c.replace("_whm2", "_wm2") for c in wh_cols] + ["cams_reliability"]]


def build_cams_table(catalog: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, row in catalog.iterrows():
        matches = sorted(RAW_DIR.glob(f"camsrad_{row[STATION]}_*.csv"))
        for path in matches:
            parsed = parse_cams_file(path, str(row[STATION]))
            if not parsed.empty:
                parts.append(parsed)
    if not parts:
        return pd.DataFrame()
    cams = pd.concat(parts, ignore_index=True).drop_duplicates([STATION, TIME])
    cams["cams_clearness"] = (
        cams["cams_ghi_wm2"] / cams["cams_clear_ghi_wm2"].replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).clip(0.0, 2.5)
    cams["cams_diffuse_fraction"] = (
        cams["cams_dhi_wm2"] / cams["cams_ghi_wm2"].replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).clip(0.0, 1.5)
    cams["cams_beam_fraction"] = (
        cams["cams_bhi_wm2"] / cams["cams_ghi_wm2"].replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).clip(0.0, 1.5)
    cams.to_csv(RUN_DIR / "cams_radiation_station_15min.csv", index=False)
    return cams


def join_cams(df: pd.DataFrame, cams: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(cams, on=[STATION, TIME], how="left")
    out["cams_ghi_minus_dssf"] = out["cams_ghi_wm2"].astype(float) - out["DSSF"].astype(float)
    out["cams_ghi_minus_clear_proxy"] = out["cams_ghi_wm2"].astype(float) - out["clear_sky_proxy"].astype(float)
    out["cams_clear_minus_clear_proxy"] = out["cams_clear_ghi_wm2"].astype(float) - out["clear_sky_proxy"].astype(float)
    out["dssf_minus_cams_ghi"] = out["DSSF"].astype(float) - out["cams_ghi_wm2"].astype(float)
    out["dssf_cams_clearness_gap"] = out["dssf_over_clear"].astype(float) - out["cams_clearness"].astype(float)
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

CAMS_FEATURES = [
    "cams_toa_wm2",
    "cams_clear_ghi_wm2",
    "cams_clear_bhi_wm2",
    "cams_clear_dhi_wm2",
    "cams_clear_bni_wm2",
    "cams_ghi_wm2",
    "cams_bhi_wm2",
    "cams_dhi_wm2",
    "cams_bni_wm2",
    "cams_reliability",
    "cams_clearness",
    "cams_diffuse_fraction",
    "cams_beam_fraction",
    "cams_ghi_minus_dssf",
    "cams_ghi_minus_clear_proxy",
    "cams_clear_minus_clear_proxy",
    "dssf_minus_cams_ghi",
    "dssf_cams_clearness_gap",
]


def residual_oof(frame: pd.DataFrame, base_col: str, features: list[str], name: str) -> tuple[np.ndarray, dict]:
    valid = frame[features + [TARGET, base_col, "year", "month"]].replace([np.inf, -np.inf], np.nan).dropna().index
    pred = np.full(len(frame), np.nan, dtype=float)
    groups = frame.loc[valid, ["year", "month"]].drop_duplicates().sort_values(["year", "month"])
    valid_index = np.asarray(valid, dtype=int)
    target_resid = frame.loc[valid, TARGET].astype(float).to_numpy() - frame.loc[valid, base_col].astype(float).to_numpy()
    for _, group in groups.iterrows():
        hold = (frame.loc[valid, "year"] == int(group["year"])) & (frame.loc[valid, "month"] == int(group["month"]))
        hold_pos = np.where(hold.to_numpy())[0]
        train_pos = np.where(~hold.to_numpy())[0]
        if len(hold_pos) == 0 or len(train_pos) < 500:
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=200.0))
        model.fit(frame.loc[valid_index[train_pos], features].astype(float).to_numpy(), target_resid[train_pos])
        corr = model.predict(frame.loc[valid_index[hold_pos], features].astype(float).to_numpy())
        pred[valid_index[hold_pos]] = frame.loc[valid_index[hold_pos], base_col].astype(float).to_numpy() + corr
    missing = np.isnan(pred)
    pred[missing] = frame.loc[missing, base_col].astype(float).to_numpy()
    pred = finalize(pred, frame["solar_elevation"].to_numpy(float))
    m = metric(frame[TARGET].to_numpy(float), pred, name)
    m["coverage"] = float(1.0 - np.mean(missing))
    return pred, m


def validate_champion_residual(train: pd.DataFrame, cams: pd.DataFrame) -> dict:
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
    ]
    val = val[base_cols].merge(train[feature_cols], on=ID, how="left")
    val = add_features(val)
    val = join_cams(val, cams)
    val = val.rename(columns={"prediction": "v12_14_base"})
    y = val[TARGET].to_numpy(float)
    base = finalize(val["v12_14_base"].to_numpy(float), val["solar_elevation"].to_numpy(float))
    val["v12_14_base"] = base
    candidates = {"v12_14_base": base}
    for col in ["cams_ghi_wm2", "DSSF"]:
        candidates[col] = finalize(val[col].fillna(0).to_numpy(float), val["solar_elevation"].to_numpy(float))
    candidates["mean_dssf_cams"] = finalize(
        0.5 * val["DSSF"].fillna(0).to_numpy(float) + 0.5 * val["cams_ghi_wm2"].fillna(0).to_numpy(float),
        val["solar_elevation"].to_numpy(float),
    )
    rows = [metric(y, pred, name) for name, pred in candidates.items()]
    core_pred, core_metric = residual_oof(val, "v12_14_base", CORE_FEATURES, "core_ridge_residual")
    cams_pred, cams_metric = residual_oof(val, "v12_14_base", CORE_FEATURES + CAMS_FEATURES, "cams_ridge_residual")
    rows.extend([core_metric, cams_metric])
    val["core_ridge_residual"] = core_pred
    val["cams_ridge_residual"] = cams_pred
    metrics = pd.DataFrame(rows).sort_values(["weighted", "rmse"])
    metrics.to_csv(DIAG_DIR / "champion_residual_metrics.csv", index=False)
    val[[ID, TARGET, "year", "month", STATION, "solar_elevation", "DSSF", "cams_ghi_wm2", "cams_clear_ghi_wm2", "v12_14_base", "core_ridge_residual", "cams_ridge_residual"]].to_csv(
        RUN_DIR / "val_predictions.csv",
        index=False,
    )
    month_rows = []
    for candidate in ["v12_14_base", "core_ridge_residual", "cams_ridge_residual", "cams_ghi_wm2"]:
        pred = val[candidate].to_numpy(float) if candidate in val.columns else candidates[candidate]
        for month, idx in val.groupby("month", observed=True).groups.items():
            arr = np.asarray(list(idx), dtype=int)
            row = metric(y[arr], pred[arr], candidate)
            row["month"] = int(month)
            month_rows.append(row)
    pd.DataFrame(month_rows).to_csv(DIAG_DIR / "champion_residual_month_metrics.csv", index=False)
    resid = y - base
    corrs = {}
    for col in CAMS_FEATURES:
        x = val[col].astype(float).to_numpy()
        mask = np.isfinite(x) & np.isfinite(resid)
        corrs[f"{col}_corr_with_v12_residual"] = float(np.corrcoef(x[mask], resid[mask])[0, 1]) if mask.sum() > 10 else math.nan
    return {"rows": int(len(val)), "metrics": metrics.to_dict(orient="records"), "correlations": corrs}


def validate_dssf_residual_january(train: pd.DataFrame, cams: pd.DataFrame) -> dict:
    val = train[train["month"].isin([1, 3, 5, 7, 9, 11])].copy().reset_index(drop=True)
    val = join_cams(val, cams)
    val["dssf_base"] = finalize(val["DSSF"].fillna(0).to_numpy(float), val["solar_elevation"].to_numpy(float))
    y = val[TARGET].to_numpy(float)
    rows = [
        metric(y, val["dssf_base"].to_numpy(float), "dssf_base"),
        metric(y, finalize(val["cams_ghi_wm2"].fillna(0).to_numpy(float), val["solar_elevation"].to_numpy(float)), "cams_ghi"),
    ]
    core_pred, core_metric = residual_oof(val, "dssf_base", CORE_FEATURES, "core_dssf_residual")
    cams_pred, cams_metric = residual_oof(val, "dssf_base", CORE_FEATURES + CAMS_FEATURES, "cams_dssf_residual")
    rows.extend([core_metric, cams_metric])
    val["core_dssf_residual"] = core_pred
    val["cams_dssf_residual"] = cams_pred
    jan = val[val["month"] == 1].copy()
    train_non_jan = val[val["month"] != 1].replace([np.inf, -np.inf], np.nan).dropna(subset=CORE_FEATURES + CAMS_FEATURES + [TARGET, "dssf_base"])
    jan_valid = jan.replace([np.inf, -np.inf], np.nan).dropna(subset=CORE_FEATURES + CAMS_FEATURES + [TARGET, "dssf_base"])
    jan_metrics = []
    for feature_set, name in [(CORE_FEATURES, "core_train_non_jan_eval_jan"), (CORE_FEATURES + CAMS_FEATURES, "cams_train_non_jan_eval_jan")]:
        model = make_pipeline(StandardScaler(), Ridge(alpha=200.0))
        y_train = train_non_jan[TARGET].astype(float).to_numpy() - train_non_jan["dssf_base"].astype(float).to_numpy()
        model.fit(train_non_jan[feature_set].astype(float).to_numpy(), y_train)
        corr = model.predict(jan_valid[feature_set].astype(float).to_numpy())
        pred = finalize(jan_valid["dssf_base"].astype(float).to_numpy() + corr, jan_valid["solar_elevation"].to_numpy(float))
        jan_metrics.append(metric(jan_valid[TARGET].to_numpy(float), pred, name))
    metrics = pd.DataFrame(rows).sort_values(["weighted", "rmse"])
    metrics.to_csv(DIAG_DIR / "dssf_residual_metrics.csv", index=False)
    pd.DataFrame(jan_metrics).to_csv(DIAG_DIR / "dssf_residual_january_metrics.csv", index=False)
    return {"rows": int(len(val)), "metrics": metrics.to_dict(orient="records"), "january_metrics": jan_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CAMS solar radiation and validate residual signal.")
    parser.add_argument("--smoke", action="store_true", help="Download two stations only and skip validation.")
    parser.add_argument("--force-download", action="store_true", help="Re-download existing CAMS files.")
    parser.add_argument("--skip-download", action="store_true", help="Use existing CAMS files only.")
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
    ]
    train = add_features(pd.read_csv(TRAIN_PATH, usecols=usecols + [TARGET]))
    test = add_features(pd.read_csv(TEST_PATH, usecols=usecols))
    catalog = station_catalog(train, test)
    catalog.to_csv(DIAG_DIR / "station_catalog.csv", index=False)
    if args.skip_download:
        manifest = pd.DataFrame([{"status": "skipped"}])
    else:
        manifest = download_cams(catalog, smoke=args.smoke, force=args.force_download)
    cams = build_cams_table(catalog)
    if args.smoke or cams.empty:
        result = {"champion_residual": None, "dssf_residual_january": None}
    else:
        result = {
            "champion_residual": validate_champion_residual(train, cams),
            "dssf_residual_january": validate_dssf_residual_january(train, cams),
        }
    metrics = {
        "run": "v14.2",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "hypothesis": (
            "CAMS 15-minute all-sky and clear-sky radiation provide an independent satellite radiation backbone "
            "that may explain errors left by LSA SAF DSSF and v12.14."
        ),
        "download_manifest": manifest.to_dict(orient="records"),
        "cams_rows": int(len(cams)),
        "result": result,
        "submission_created": False,
        "outputs": {
            "cams_table": str(RUN_DIR / "cams_radiation_station_15min.csv"),
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
                "run": "v14.2",
                "script": "runs/v14.2/v14_2_cams_radiation_validation.py",
                "python": ".\\.venv\\Scripts\\python.exe",
                "source": "cams-solar-radiation-timeseries",
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
