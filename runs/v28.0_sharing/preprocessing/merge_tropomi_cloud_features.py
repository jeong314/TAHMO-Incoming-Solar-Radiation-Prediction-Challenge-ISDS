#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge daily TROPOMI CLOUD station features into TAHMO Train/Test or LSASAF model-ready files.

Example:
python merge_tropomi_cloud_features.py ^
  --train Train_with_LSASAF_MDSSFTD_scaled_modelready.csv ^
  --test Test_with_LSASAF_MDSSFTD_scaled_modelready.csv ^
  --tropomi data_external/tropomi_cloud/tropomi_cloud_station_daily_full_grid.csv ^
  --out_dir data_external/tropomi_cloud_merged
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="Train CSV to merge into")
    p.add_argument("--test", required=True, help="Test CSV to merge into")
    p.add_argument("--tropomi", required=True, help="tropomi_cloud_station_daily_full_grid.csv")
    p.add_argument("--out_dir", default="data_external/tropomi_cloud_merged")
    p.add_argument("--prefix", default="with_tropomi_cloud")
    return p.parse_args()


def load_tropomi(path: Path) -> pd.DataFrame:
    trop = pd.read_csv(path)
    trop["date"] = pd.to_datetime(trop["date"]).dt.date

    # Keep only feature columns and keys. Avoid duplicating station metadata already in Train/Test.
    key_cols = ["station", "date"]
    feature_cols = [
        c for c in trop.columns
        if c.startswith("tropomi_") or c in ["n_images", "buffer_m"]
    ]
    cols = key_cols + [c for c in feature_cols if c not in key_cols]
    trop = trop[cols].drop_duplicates(["station", "date"])

    # Make booleans model-friendly.
    for c in trop.columns:
        if trop[c].dtype == "bool":
            trop[c] = trop[c].astype("int8")
    return trop


def merge_one(input_path: Path, trop: pd.DataFrame, output_path: Path) -> None:
    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    out = df.merge(trop, on=["station", "date"], how="left")

    # Ensure missing flag is present and numeric.
    if "tropomi_cloud_missing" in out.columns:
        out["tropomi_cloud_missing"] = out["tropomi_cloud_missing"].fillna(True).astype("int8")
    else:
        out["tropomi_cloud_missing"] = 1

    if "tropomi_cloud_before_gee_l3_start" in out.columns:
        out["tropomi_cloud_before_gee_l3_start"] = out["tropomi_cloud_before_gee_l3_start"].fillna(True).astype("int8")

    out = out.drop(columns=["date"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"[INFO] Saved: {output_path} shape={out.shape}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    trop = load_tropomi(Path(args.tropomi))
    print(f"[INFO] TROPOMI features: {trop.shape}")

    train_out = out_dir / f"Train_{args.prefix}.csv"
    test_out = out_dir / f"Test_{args.prefix}.csv"

    merge_one(Path(args.train), trop, train_out)
    merge_one(Path(args.test), trop, test_out)


if __name__ == "__main__":
    main()
