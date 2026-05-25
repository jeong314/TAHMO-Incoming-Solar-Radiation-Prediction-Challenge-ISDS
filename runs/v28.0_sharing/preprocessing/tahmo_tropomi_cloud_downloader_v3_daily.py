#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TAHMO Challenge: robust DAILY TROPOMI CLOUD station feature extractor via Google Earth Engine.

This v3 script fixes the day-chunk filename collision in earlier scripts and avoids
large concurrent aggregations by downloading exactly one day at a time.

Collection:
  COPERNICUS/S5P/OFFL/L3_CLOUD

Example:
  python tahmo_tropomi_cloud_downloader_v3_daily.py --train Train.csv --test Test.csv --out_dir data_external\tropomi_cloud_v3 --project YOUR_PROJECT_ID --buffer_m 10000 --timeout 600 --sleep 0.5
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import requests

try:
    import ee
except ImportError as exc:
    raise SystemExit("earthengine-api is not installed. Run: pip install earthengine-api pandas requests") from exc

COLLECTION_ID = "COPERNICUS/S5P/OFFL/L3_CLOUD"
GEE_L3_START = pd.Timestamp("2018-07-04")

BANDS = [
    "cloud_fraction",
    "cloud_top_pressure",
    "cloud_top_height",
    "cloud_base_pressure",
    "cloud_base_height",
    "cloud_optical_depth",
    "surface_albedo",
    "sensor_zenith_angle",
    "solar_zenith_angle",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robust daily TROPOMI CLOUD extractor for TAHMO via Earth Engine")
    p.add_argument("--train", default="Train.csv")
    p.add_argument("--test", default="Test.csv")
    p.add_argument("--out_dir", default="data_external/tropomi_cloud_v3")
    p.add_argument("--project", default=None, help="Google Cloud / Earth Engine project id")
    p.add_argument("--buffer_m", type=float, default=10000.0)
    p.add_argument("--scale", type=float, default=1113.2)
    p.add_argument("--start", default=None, help="YYYY-MM-DD. Default: max(data_min_date, 2018-07-04)")
    p.add_argument("--end", default=None, help="YYYY-MM-DD inclusive. Default: data_max_date")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--sleep", type=float, default=0.5)
    p.add_argument("--force", action="store_true", help="Overwrite existing daily raw files")
    p.add_argument("--no_full_grid", action="store_true")
    return p.parse_args()


def init_ee(project: Optional[str]) -> None:
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception:
        print("[INFO] Earth Engine is not initialized. Starting authentication...", file=sys.stderr)
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()


def read_station_and_dates(train_path: Path, test_path: Path) -> Tuple[pd.DataFrame, pd.DatetimeIndex, pd.Timestamp, pd.Timestamp]:
    meta_cols = [
        "station", "station_name", "country", "installation_height", "elevation", "latitude", "longitude"
    ]
    usecols = set(meta_cols + ["timestamp"])
    train = pd.read_csv(train_path, usecols=lambda c: c in usecols)
    test = pd.read_csv(test_path, usecols=lambda c: c in usecols)
    train["timestamp"] = pd.to_datetime(train["timestamp"])
    test["timestamp"] = pd.to_datetime(test["timestamp"])

    stations = (
        pd.concat([train[meta_cols], test[meta_cols]], ignore_index=True)
        .drop_duplicates("station")
        .sort_values("station")
        .reset_index(drop=True)
    )
    all_ts = pd.concat([train["timestamp"], test["timestamp"]], ignore_index=True)
    all_dates = pd.DatetimeIndex(all_ts.dt.floor("D").drop_duplicates().sort_values())
    return stations, all_dates, all_dates.min(), all_dates.max()


def make_station_fc(stations: pd.DataFrame, buffer_m: float) -> "ee.FeatureCollection":
    features = []
    for _, r in stations.iterrows():
        geom = ee.Geometry.Point([float(r["longitude"]), float(r["latitude"])]).buffer(buffer_m)
        props = {
            "station": str(r["station"]),
            "station_name": str(r.get("station_name", "")),
            "country": str(r.get("country", "")),
            "installation_height": float(r.get("installation_height", float("nan"))),
            "elevation": float(r.get("elevation", float("nan"))),
            "latitude": float(r["latitude"]),
            "longitude": float(r["longitude"]),
        }
        features.append(ee.Feature(geom, props))
    return ee.FeatureCollection(features)


def build_fc_for_one_day(stations_fc: "ee.FeatureCollection", day: pd.Timestamp, buffer_m: float, scale: float) -> "ee.FeatureCollection":
    start = day.strftime("%Y-%m-%d")
    end = (day + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    col = (
        ee.ImageCollection(COLLECTION_ID)
        .select(BANDS)
        .filterDate(start, end)
        .filterBounds(stations_fc.geometry())
    )
    n_images = col.size()
    reducer = ee.Reducer.mean().combine(reducer2=ee.Reducer.count(), sharedInputs=True)

    img = col.mean()
    reduced = img.reduceRegions(
        collection=stations_fc,
        reducer=reducer,
        scale=scale,
        tileScale=4,
    )

    def strip_geom(feat):
        return ee.Feature(None, feat.toDictionary()).set({
            "date": start,
            "n_images": n_images,
            "buffer_m": buffer_m,
        })

    # If there are no images, return an empty FeatureCollection instead of failing downstream.
    return ee.FeatureCollection(ee.Algorithms.If(n_images.gt(0), reduced.map(strip_geom), ee.FeatureCollection([])))


def selectors() -> List[str]:
    s = ["station", "station_name", "country", "installation_height", "elevation", "latitude", "longitude", "date", "n_images", "buffer_m"]
    for b in BANDS:
        s.extend([f"{b}_mean", f"{b}_count"])
    return s


def download_fc_to_csv(fc: "ee.FeatureCollection", path: Path, timeout: int) -> pd.DataFrame:
    sel = selectors()
    url = fc.getDownloadURL(filetype="CSV", selectors=sel)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    text = response.content.decode("utf-8")
    path.write_text(text, encoding="utf-8")
    if not text.strip():
        return pd.DataFrame(columns=sel)
    return pd.read_csv(io.StringIO(text))


def clean_and_prefix(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    keep = ["station", "station_name", "country", "installation_height", "elevation", "latitude", "longitude", "date", "n_images", "buffer_m"]
    for b in BANDS:
        keep.extend([f"{b}_mean", f"{b}_count"])
    keep = [c for c in keep if c in raw.columns]
    df = raw[keep].copy()
    rename = {}
    for b in BANDS:
        if f"{b}_mean" in df.columns:
            rename[f"{b}_mean"] = f"tropomi_{b}"
        if f"{b}_count" in df.columns:
            rename[f"{b}_count"] = f"tropomi_{b}_pixel_count"
    df = df.rename(columns=rename)
    df["date"] = pd.to_datetime(df["date"])
    count_col = "tropomi_cloud_fraction_pixel_count"
    if count_col in df.columns:
        df["tropomi_cloud_valid_pixel_count"] = df[count_col]
        df["tropomi_cloud_missing"] = df[count_col].isna() | (df[count_col] <= 0)
    else:
        df["tropomi_cloud_valid_pixel_count"] = pd.NA
        df["tropomi_cloud_missing"] = True
    return df


def build_full_grid(stations: pd.DataFrame, all_dates: pd.DatetimeIndex, features: pd.DataFrame) -> pd.DataFrame:
    station_key = stations[["station", "station_name", "country", "installation_height", "elevation", "latitude", "longitude"]].drop_duplicates("station")
    grid = station_key[["station"]].assign(_key=1).merge(
        pd.DataFrame({"date": all_dates, "_key": 1}), on="_key"
    ).drop(columns="_key")

    feature_cols = ["station", "date"] + [c for c in features.columns if c.startswith("tropomi_") or c in ["n_images", "buffer_m"]]
    feature_cols = [c for c in feature_cols if c in features.columns]
    feat_small = features[feature_cols].drop_duplicates(["station", "date"]) if not features.empty else pd.DataFrame(columns=["station", "date"])

    out = grid.merge(feat_small, on=["station", "date"], how="left")
    out = out.merge(station_key, on="station", how="left")
    if "tropomi_cloud_missing" not in out.columns:
        out["tropomi_cloud_missing"] = True
    else:
        out["tropomi_cloud_missing"] = out["tropomi_cloud_missing"].fillna(True)
    out["tropomi_cloud_before_gee_l3_start"] = out["date"] < GEE_L3_START
    out.loc[out["tropomi_cloud_before_gee_l3_start"], "tropomi_cloud_missing"] = True
    return out.sort_values(["station", "date"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    train_path = Path(args.train)
    test_path = Path(args.test)
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw_daily"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    stations, all_dates, data_min, data_max = read_station_and_dates(train_path, test_path)
    user_start = pd.Timestamp(args.start) if args.start else data_min
    user_end = pd.Timestamp(args.end) if args.end else data_max
    extract_start = max(pd.Timestamp(user_start.date()), GEE_L3_START)
    extract_end = pd.Timestamp(user_end.date())
    if extract_start > extract_end:
        raise SystemExit(f"No extractable date range: requested end={extract_end.date()}, GEE_L3_START={GEE_L3_START.date()}")

    stations_path = out_dir / "tahmo_station_metadata_for_tropomi.csv"
    stations.to_csv(stations_path, index=False)

    print(f"[INFO] Stations: {len(stations)} -> {stations_path}")
    print(f"[INFO] TAHMO date range: {data_min.date()} to {data_max.date()}")
    print(f"[INFO] TROPOMI GEE L3 extraction range: {extract_start.date()} to {extract_end.date()}")
    print(f"[INFO] Collection: {COLLECTION_ID}")
    print(f"[INFO] Buffer: {args.buffer_m:.0f} m, scale: {args.scale:.1f} m")

    init_ee(args.project)
    stations_fc = make_station_fc(stations, args.buffer_m)

    parts = []
    days = pd.date_range(extract_start, extract_end, freq="D")
    for i, day in enumerate(days, start=1):
        part_path = raw_dir / f"tropomi_cloud_raw_{day:%Y%m%d}.csv"
        if part_path.exists() and part_path.stat().st_size > 0 and not args.force:
            part = pd.read_csv(part_path)
            print(f"[INFO] {i}/{len(days)} existing {day:%Y-%m-%d}: rows={len(part):,}")
        else:
            print(f"[INFO] {i}/{len(days)} extracting {day:%Y-%m-%d} ...")
            fc = build_fc_for_one_day(stations_fc, day, args.buffer_m, args.scale)
            try:
                part = download_fc_to_csv(fc, part_path, args.timeout)
            except Exception as exc:
                print(f"[WARN] failed {day:%Y-%m-%d}: {exc}", file=sys.stderr)
                part = pd.DataFrame(columns=selectors())
                part.to_csv(part_path, index=False)
            print(f"[INFO] saved {part_path} rows={len(part):,}")
            if args.sleep > 0:
                time.sleep(args.sleep)
        parts.append(part)

    raw_all = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=selectors())
    raw_path = out_dir / "tropomi_cloud_station_daily_raw.csv"
    raw_all.to_csv(raw_path, index=False)
    print(f"[INFO] Saved combined raw: {raw_path} rows={len(raw_all):,}")

    features = clean_and_prefix(raw_all)
    features_path = out_dir / "tropomi_cloud_station_daily_features_available_dates.csv"
    features.to_csv(features_path, index=False)
    print(f"[INFO] Saved cleaned available-date features: {features_path} rows={len(features):,}")

    if not args.no_full_grid:
        full_grid = build_full_grid(stations, all_dates, features)
        full_path = out_dir / "tropomi_cloud_station_daily_full_grid.csv"
        full_grid.to_csv(full_path, index=False)
        print(f"[INFO] Saved full station-date grid: {full_path} rows={len(full_grid):,}")

        report = (
            full_grid.assign(year=full_grid["date"].dt.year)
            .groupby("year")
            .agg(
                rows=("station", "size"),
                missing_rate=("tropomi_cloud_missing", "mean"),
                valid_pixel_count_mean=("tropomi_cloud_valid_pixel_count", "mean"),
            )
            .reset_index()
        )
        report_path = out_dir / "tropomi_cloud_coverage_report_by_year.csv"
        report.to_csv(report_path, index=False)
        print(f"[INFO] Saved coverage report: {report_path}")
        print(report.to_string(index=False))

    print("[DONE] TROPOMI CLOUD daily extraction finished.")


if __name__ == "__main__":
    main()
