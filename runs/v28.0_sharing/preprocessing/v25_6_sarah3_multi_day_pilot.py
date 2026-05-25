from __future__ import annotations

import argparse
import json
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import eumdac
import h5py
import numpy as np
import pandas as pd


RUN_DIR = Path("runs/v25.6_sarah3_multi_day_pilot")
DATA_DIR = RUN_DIR / "data"
DIAG_DIR = RUN_DIR / "diagnostics"
CREDENTIAL_PATH = Path(
    os.environ.get(
        "EUMDAC_CREDENTIALS_PATH",
        Path(__file__).resolve().parent / "eumdac_credentials.local.json",
    )
)
COLLECTION_ID = "EO:EUM:DAT:0863"
TRAIN_RAW = Path("Train.csv")
TRAIN_MODELREADY = Path("Train_with_LSASAF_MDSSFTD_albedo_AOD_OPACITY_THREDDS_CAMS_NASA_modelready.csv")
DEFAULT_DATES = ["2016-01-12", "2016-03-15", "2016-07-15", "2016-11-15"]
DEFAULT_TYPES = ["SIS", "CAL"]


def decode_attr(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [decode_attr(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def parse_time(values: np.ndarray, units: str) -> pd.DatetimeIndex:
    match = re.search(r"since\s+(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T](\d{2}:\d{2}:\d{2}))?", units)
    if not match:
        raise ValueError(f"Unsupported time units: {units}")
    base_time = match.group(4) or "00:00:00"
    base = datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)), tzinfo=timezone.utc)
    hh, mm, ss = [int(x) for x in base_time.split(":")]
    base = base.replace(hour=hh, minute=mm, second=ss)
    unit = units.split(" since ")[0].strip().lower()
    scale = 3600.0 if unit.startswith("hour") else 60.0 if unit.startswith("minute") else 86400.0 if unit.startswith("day") else 1.0
    return pd.DatetimeIndex([base + timedelta(seconds=float(v) * scale) for v in values]).tz_convert(None)


def scaled_values(dataset: h5py.Dataset, values: np.ndarray) -> np.ndarray:
    out = values.astype("float64")
    for attr in ["_FillValue", "missing_value"]:
        fill = dataset.attrs.get(attr)
        if fill is not None:
            out[values == fill] = np.nan
    def first_float(raw: object, default: float) -> float:
        if raw is None:
            return default
        arr = np.asarray(raw).astype("float64").ravel()
        return float(arr[0]) if len(arr) else default

    scale = first_float(dataset.attrs.get("scale_factor"), 1.0)
    offset = first_float(dataset.attrs.get("add_offset"), 0.0)
    out = out * scale + offset
    return out


def read_credentials() -> tuple[str, str]:
    data = json.loads(CREDENTIAL_PATH.read_text(encoding="utf-8"))
    return str(data["consumer_key"]).strip(), str(data["consumer_secret"]).strip()


def datastore() -> eumdac.DataStore:
    token = eumdac.AccessToken(read_credentials())
    _ = token.access_token
    return eumdac.DataStore(token)


def station_metadata() -> pd.DataFrame:
    cols = ["station", "station_name", "country", "latitude", "longitude", "elevation"]
    df = pd.read_csv(TRAIN_RAW, usecols=cols)
    return df.drop_duplicates("station").sort_values("station").reset_index(drop=True)


def search_first_product(collection: object, product_date: date, product_type: str) -> object | None:
    products = collection.search(
        dtstart=datetime(product_date.year, product_date.month, product_date.day, 0, 0, tzinfo=timezone.utc),
        dtend=datetime(product_date.year, product_date.month, product_date.day, 1, 0, tzinfo=timezone.utc),
        type=product_type,
        compositeType="PT30M",
        bbox=[-20.0, -12.0, 52.0, 18.0],
    )
    for product in products:
        return product
    return None


def product_id(product: object) -> str:
    try:
        return str(product._id)
    except Exception:
        pass
    try:
        return str(product)
    except Exception:
        return "unknown_product"


def download_product(product: object, product_type: str, product_date: date) -> Path:
    entries = list(product.entries)
    nc_entries = [e for e in entries if str(e).lower().endswith(".nc")]
    selected = nc_entries[0] if nc_entries else entries[0]
    dest = DATA_DIR / f"{product_date.isoformat()}_{product_type}_{Path(str(selected)).name}"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"[download] {product_type} {product_date} {selected}", flush=True)
    with product.open(entry=selected) as src, dest.open("wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
    return dest


def extract_station_series(nc_path: Path, product_type: str, stations: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    with h5py.File(nc_path, "r") as h5:
        lat = h5["lat"][:].astype("float64")
        lon = h5["lon"][:].astype("float64")
        variable = product_type if product_type in h5 else [k for k in h5.keys() if k.upper() == product_type][0]
        ds = h5[variable]
        times = parse_time(h5["time"][:], str(decode_attr(h5["time"].attrs.get("units", ""))))
        rows = []
        for row in stations.itertuples(index=False):
            lat_idx = int(np.abs(lat - float(row.latitude)).argmin())
            lon_idx = int(np.abs(lon - float(row.longitude)).argmin())
            vals = scaled_values(ds, ds[:, lat_idx, lon_idx])
            for timestamp, value in zip(times, vals):
                rows.append({"station": row.station, "timestamp": timestamp, f"sarah_{product_type.lower()}": value})
        meta = {
            "path": str(nc_path),
            "variable": variable,
            "shape": [int(x) for x in ds.shape],
            "attrs": {k: decode_attr(v) for k, v in ds.attrs.items()},
            "time_start": str(times.min()),
            "time_end": str(times.max()),
        }
        return pd.DataFrame(rows), meta


def compare_with_train(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    raw_cols = ["ID", "timestamp", "station", "radiation (W/m2)"]
    raw = pd.read_csv(TRAIN_RAW, usecols=raw_cols, parse_dates=["timestamp"])
    min_ts = feature_df["timestamp"].min() - pd.Timedelta("20min")
    max_ts = feature_df["timestamp"].max() + pd.Timedelta("20min")
    raw = raw[(raw["timestamp"] >= min_ts) & (raw["timestamp"] <= max_ts)].copy()
    model_cols = ["ID", "DSSF", "FRACTION_DIFFUSE", "cams_ghi_wm2", "power_allsky_sfc_sw_dwn"]
    model = pd.read_csv(TRAIN_MODELREADY, usecols=model_cols)
    sample = raw.merge(model, on="ID", how="left").sort_values(["station", "timestamp"])
    merged_parts = []
    feature_df = feature_df.sort_values(["station", "timestamp"])
    for station, sdf in sample.groupby("station", sort=False):
        right = feature_df[feature_df["station"] == station].sort_values("timestamp")
        if right.empty:
            continue
        part = pd.merge_asof(
            sdf.sort_values("timestamp"),
            right.drop(columns=["station"]),
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("16min"),
        )
        part["station"] = station
        merged_parts.append(part)
    compare = pd.concat(merged_parts, ignore_index=True) if merged_parts else pd.DataFrame()
    metrics: dict[str, object] = {}
    target = "radiation (W/m2)"
    comparable = compare["sarah_sis"].notna() if "sarah_sis" in compare else pd.Series(False, index=compare.index)
    for col in ["sarah_sis", "DSSF", "cams_ghi_wm2", "power_allsky_sfc_sw_dwn"]:
        if col not in compare:
            continue
        mask = comparable & compare[target].notna() & compare[col].notna()
        if int(mask.sum()) == 0:
            continue
        err = compare.loc[mask, col] - compare.loc[mask, target]
        metrics[col] = {
            "n": int(mask.sum()),
            "mbe": float(err.mean()),
            "mae": float(err.abs().mean()),
            "rmse": float(np.sqrt(np.mean(err.to_numpy() ** 2))),
            "corr": float(compare.loc[mask, [col, target]].corr().iloc[0, 1]),
            "mean": float(compare.loc[mask, col].mean()),
        }
    if "sarah_cal" in compare:
        mask = comparable & compare[target].notna() & compare["sarah_cal"].notna()
        if int(mask.sum()) > 0:
            metrics["sarah_cal_regime"] = {
                "n": int(mask.sum()),
                "mean": float(compare.loc[mask, "sarah_cal"].mean()),
                "corr_with_target": float(compare.loc[mask, ["sarah_cal", target]].corr().iloc[0, 1]),
                "corr_with_sarah_error": float(
                    pd.concat(
                        [
                            compare.loc[mask, "sarah_cal"],
                            (compare.loc[mask, "sarah_sis"] - compare.loc[mask, target]).rename("sarah_error"),
                        ],
                        axis=1,
                    ).corr().iloc[0, 1]
                ),
            }
    return compare, metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", default=",".join(DEFAULT_DATES))
    parser.add_argument("--types", default=",".join(DEFAULT_TYPES))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--delete-raw-after-extract", action="store_true")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    dates = [date.fromisoformat(x.strip()) for x in args.dates.split(",") if x.strip()]
    product_types = [x.strip().upper() for x in args.types.split(",") if x.strip()]
    stations = station_metadata()
    store = datastore()
    collection = store.get_collection(COLLECTION_ID)
    report: dict[str, object] = {"dates": [d.isoformat() for d in dates], "types": product_types, "files": []}

    all_features: pd.DataFrame | None = None
    for product_date in dates:
        day_features: pd.DataFrame | None = None
        for product_type in product_types:
            product = search_first_product(collection, product_date, product_type)
            item: dict[str, object] = {"date": product_date.isoformat(), "type": product_type, "found": product is not None}
            if product is None:
                report["files"].append(item)
                continue
            item["product"] = product_id(product)
            if not args.download:
                report["files"].append(item)
                continue
            path = download_product(product, product_type, product_date)
            item["path"] = str(path)
            item["size_bytes"] = path.stat().st_size
            series, meta = extract_station_series(path, product_type, stations)
            item["meta"] = meta
            day_features = series if day_features is None else day_features.merge(series, on=["station", "timestamp"], how="outer")
            if args.delete_raw_after_extract:
                path.unlink()
                item["deleted_after_extract"] = True
            report["files"].append(item)
        if day_features is not None:
            all_features = day_features if all_features is None else pd.concat([all_features, day_features], ignore_index=True)

    if all_features is not None and not all_features.empty:
        all_features.to_csv(DIAG_DIR / "sarah3_pilot_station_features.csv", index=False)
        compare, metrics = compare_with_train(all_features)
        compare.to_csv(DIAG_DIR / "sarah3_pilot_vs_train.csv", index=False)
        report["feature_rows"] = int(len(all_features))
        report["compare_rows"] = int(len(compare))
        report["metrics"] = metrics
        print(json.dumps(metrics, indent=2), flush=True)

    (RUN_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
