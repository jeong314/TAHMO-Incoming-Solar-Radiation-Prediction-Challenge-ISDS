"""Reproduce the v28.0 public-best submission.

This script is a deterministic final-stage correction pipeline. It does not
train a new tree model. Instead, it reads prepared physical train/test tables,
prior-stage OOF/test predictions, and team anchor submissions from this same
folder, then applies a mean-neutral edge-month clear-sky-index correction.

Scientific idea:
- solar radiation is modeled through daylight clear-sky energy and kt
  (radiation / clear-sky reference);
- missing even-month edge periods, especially February and December, inherit
  a distance-weighted adjacent odd-month kt residual signal;
- the correction is applied to the strong jieun20 submission shape, with
  separate early/late edge strengths and a global mean-neutral constraint;
- TargetMBE and TargetRMSE are always written from the same final vector.

Run from the repository root:
    .\\.venv\\Scripts\\python.exe .\\runs\\v28.0_sharing\\v28_0_reproduce.py --mode submit
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


RUN_NAME = "v28.0_asymmetric_edge_kt_tilt_reproduction_pack"
SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR
DIAG_DIR = RUN_DIR / "diagnostics"
CAND_DIR = RUN_DIR / "submission_candidates"

# Reproduction-pack contract: all inputs must live next to this script.
TRAIN_PATH = SCRIPT_DIR / "train_physical_preprocessed.csv"
TEST_PATH = SCRIPT_DIR / "test_physical_preprocessed.csv"
SAMPLE_PATH = SCRIPT_DIR / "SampleSubmission.csv"
V166_VAL = SCRIPT_DIR / "v166_val_predictions.csv"
V166_SUB = SCRIPT_DIR / "v166_submission.csv"
V177_VAL = SCRIPT_DIR / "v177_val_predictions.csv"
V177_SUB = SCRIPT_DIR / "v177_submission.csv"
JIEUN20_SUB = SCRIPT_DIR / "jieun20.csv"
DS15_SUB = SCRIPT_DIR / "ds15.csv"

ID = "ID"
TARGET = "radiation (W/m2)"
STATION = "station"
MONTH = "dt_month"
VALID_MONTHS = [3, 5, 7, 9, 11]
FAST_MONTHS = [5, 9, 11]

VALIDATION_GATE_MONTHS = {
    "all": [3, 5, 7, 9, 11],
    "edge": [3, 11],
    "transition": [3, 9, 11],
    "early": [3],
    "late": [11],
}

TEST_GATE_MONTHS = {
    "all": [2, 4, 6, 8, 10, 12],
    "edge": [2, 12],
    "transition": [2, 10, 12],
    "early": [2],
    "late": [12],
}

READ_COLS = [
    ID,
    STATION,
    "dt_year",
    MONTH,
    "dt_dayofyear",
    "phys_solar_elevation_clipped",
    "phys_daylight_flag",
    "phys_clear_sky_ref",
    "fusion_weighted_allsky",
    "fusion_external_clear_index_weighted",
    "fusion_reliability_score",
    "fusion_sensor_disagreement",
    "fusion_allsky_cv",
    "lsa_dssf_phys",
    "cams_ghi_phys",
    "power_allsky_phys",
    "aod_phys",
    "opacity_phys",
    "regime_cloudy_flag",
    "regime_clear_high_sun_flag",
    TARGET,
    "target_suspicious_daytime_zero_flag",
]


def ensure_dirs() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    CAND_DIR.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (RUN_DIR / "train.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def require_inputs() -> None:
    required = [TRAIN_PATH, TEST_PATH, SAMPLE_PATH, V166_VAL, V166_SUB, V177_VAL, V177_SUB, JIEUN20_SUB, DS15_SUB]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required reproduction artifacts: " + ", ".join(missing))


def existing_usecols(path: Path, include_target: bool) -> List[str]:
    available = set(pd.read_csv(path, nrows=0).columns)
    cols = [col for col in READ_COLS if col in available]
    if not include_target and TARGET in cols:
        cols.remove(TARGET)
    return cols


def read_frames(smoke_rows: Optional[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    kwargs = {"low_memory": False}
    if smoke_rows is not None:
        kwargs["nrows"] = smoke_rows
    train = pd.read_csv(TRAIN_PATH, usecols=existing_usecols(TRAIN_PATH, True), **kwargs)
    test = pd.read_csv(TEST_PATH, usecols=existing_usecols(TEST_PATH, False), **kwargs)
    sample = pd.read_csv(SAMPLE_PATH, usecols=[ID])
    if smoke_rows is not None:
        sample = sample.iloc[: len(test)].copy()
    if smoke_rows is None and not np.array_equal(test[ID].to_numpy(), sample[ID].to_numpy()):
        raise ValueError("test ID order mismatch")
    return train, test, sample


def read_val_base(path: Path, pred_col: str, name: str) -> pd.DataFrame:
    val = pd.read_csv(path, usecols=[ID, pred_col])
    return val.rename(columns={pred_col: name})


def read_submission(path: Path, sample: pd.DataFrame, name: str) -> np.ndarray:
    sub = pd.read_csv(path)
    if len(sub) != len(sample):
        sub = sub.iloc[: len(sample)].copy()
    if not np.array_equal(sub[ID].to_numpy(), sample[ID].to_numpy()):
        raise ValueError(f"{name}: ID order mismatch")
    if not np.array_equal(sub["TargetMBE"].to_numpy(float), sub["TargetRMSE"].to_numpy(float)):
        raise ValueError(f"{name}: target columns differ")
    return sub["TargetMBE"].to_numpy(float)


def safe_num(frame: pd.DataFrame, col: str, fill: float = np.nan) -> pd.Series:
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce")
    return pd.Series(fill, index=frame.index, dtype="float64")


def add_aux(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["row_pos"] = np.arange(len(out), dtype=np.int64)
    clear = safe_num(out, "phys_clear_sky_ref", np.nan).clip(0.0, 1500.0)
    out["clear_ref_safe"] = clear.fillna(0.0)
    out["daylight_model_flag"] = (
        (safe_num(out, "phys_daylight_flag", 0).fillna(0) > 0)
        & (safe_num(out, "phys_solar_elevation_clipped", 0).fillna(0) > 1.0)
        & (out["clear_ref_safe"] > 35.0)
    ).astype("int8")
    fusion = safe_num(out, "fusion_weighted_allsky", np.nan)
    fusion_kt = safe_num(out, "fusion_external_clear_index_weighted", np.nan).clip(0.0, 1.8)
    out["external_fusion_rad"] = fusion.where(fusion.notna(), fusion_kt * out["clear_ref_safe"]).fillna(0.0).clip(0.0, 1500.0)
    for src_col, out_col in [("lsa_dssf_phys", "lsa_rad"), ("cams_ghi_phys", "cams_rad"), ("power_allsky_phys", "power_rad")]:
        out[out_col] = safe_num(out, src_col, np.nan).clip(0.0, 1500.0)
    return out


def final_clip(pred: np.ndarray, solar: np.ndarray) -> np.ndarray:
    out = np.nan_to_num(np.asarray(pred, dtype=float), nan=0.0, posinf=1500.0, neginf=0.0)
    out = np.where(np.asarray(solar, dtype=float) <= -1.0, 0.0, out)
    return np.clip(out, 0.0, 1500.0)


def aggregate_month(frame: pd.DataFrame, include_target: bool) -> pd.DataFrame:
    work = frame.loc[frame["daylight_model_flag"].astype(bool)].copy()
    group_cols = [STATION, "dt_year", MONTH]
    agg = {
        "row_count": (ID, "size"),
        "clear_sum": ("clear_ref_safe", "sum"),
        "fusion_sum": ("external_fusion_rad", "sum"),
        "lsa_sum": ("lsa_rad", "sum"),
        "cams_sum": ("cams_rad", "sum"),
        "power_sum": ("power_rad", "sum"),
        "reliability_mean": ("fusion_reliability_score", "mean"),
        "disagreement_mean": ("fusion_sensor_disagreement", "mean"),
        "cv_mean": ("fusion_allsky_cv", "mean"),
        "aod_mean": ("aod_phys", "mean"),
        "opacity_mean": ("opacity_phys", "mean"),
        "cloudy_rate": ("regime_cloudy_flag", "mean"),
        "clear_rate": ("regime_clear_high_sun_flag", "mean"),
    }
    if include_target:
        agg.update(
            {
                "target_sum": (TARGET, "sum"),
                "suspicious_zero_rate": ("target_suspicious_daytime_zero_flag", "mean"),
            }
        )
    monthly = work.groupby(group_cols, observed=True, sort=False).agg(**agg).reset_index()
    eps = 1e-6
    monthly["fusion_kt"] = (monthly["fusion_sum"] / np.maximum(monthly["clear_sum"], eps)).clip(0.0, 1.8)
    monthly["lsa_kt"] = (monthly["lsa_sum"] / np.maximum(monthly["clear_sum"], eps)).clip(0.0, 1.8)
    monthly["cams_kt"] = (monthly["cams_sum"] / np.maximum(monthly["clear_sum"], eps)).clip(0.0, 1.8)
    monthly["power_kt"] = (monthly["power_sum"] / np.maximum(monthly["clear_sum"], eps)).clip(0.0, 1.8)
    monthly["external_cv"] = np.nanstd(
        np.vstack([monthly["lsa_kt"].to_numpy(float), monthly["cams_kt"].to_numpy(float), monthly["power_kt"].to_numpy(float)]),
        axis=0,
    ) / np.maximum(monthly["fusion_kt"].to_numpy(float), 0.05)
    if include_target:
        monthly["target_kt"] = (monthly["target_sum"] / np.maximum(monthly["clear_sum"], eps)).clip(0.0, 1.8)
    return monthly


def source_months(month: int, validation: bool) -> List[int]:
    if validation:
        if month == 3:
            return [1, 5]
        if month == 5:
            return [3, 7]
        if month == 7:
            return [5, 9]
        if month == 9:
            return [7, 11]
        if month == 11:
            return [7, 9]
    else:
        if month == 2:
            return [1, 3]
        if month == 4:
            return [3, 5]
        if month == 6:
            return [5, 7]
        if month == 8:
            return [7, 9]
        if month == 10:
            return [9, 11]
        if month == 12:
            return [9, 11]
    return []


def weighted_source_stats(src: pd.DataFrame, target_month: int, dist_power: float) -> Tuple[float, float, float]:
    clear = src["clear_sum"].to_numpy(float)
    if len(src) == 0 or np.nansum(clear) <= 1e-6:
        return np.nan, np.nan, 0.0
    src_month = src[MONTH].to_numpy(int)
    dist = np.maximum(np.abs(src_month - int(target_month)).astype(float), 1.0)
    dist_weight = 1.0 / np.power(dist, float(dist_power))
    denom = float(np.nansum(clear * dist_weight))
    target_kt = float(np.nansum(src["target_sum"].to_numpy(float) * dist_weight) / max(denom, 1e-6))
    fusion_kt = float(np.nansum(src["fusion_sum"].to_numpy(float) * dist_weight) / max(denom, 1e-6))
    count = float(np.nansum(src["row_count"].to_numpy(float)))
    return target_kt, fusion_kt, count


def predict_monthly_kt(train_monthly: pd.DataFrame, target_monthly: pd.DataFrame, cfg: Mapping[str, float], validation: bool) -> pd.DataFrame:
    rows = []
    station_global = (
        train_monthly.groupby(STATION, observed=True)
        .apply(lambda g: pd.Series({"station_kt": np.nansum(g["target_sum"]) / max(np.nansum(g["clear_sum"]), 1e-6)}))
        .reset_index()
    )
    global_kt = float(np.nansum(train_monthly["target_sum"]) / max(np.nansum(train_monthly["clear_sum"]), 1e-6))
    station_map = dict(zip(station_global[STATION], station_global["station_kt"]))
    for _, tgt in target_monthly.iterrows():
        month = int(tgt[MONTH])
        src_months = source_months(month, validation=validation)
        src = train_monthly.loc[
            (train_monthly[STATION] == tgt[STATION])
            & (train_monthly["dt_year"].astype(int) == int(tgt["dt_year"]))
            & (train_monthly[MONTH].astype(int).isin(src_months))
        ]
        if src.empty:
            src = train_monthly.loc[
                (train_monthly[STATION] == tgt[STATION])
                & (train_monthly[MONTH].astype(int).isin(src_months))
            ]
        if src.empty:
            src = train_monthly.loc[train_monthly[STATION] == tgt[STATION]]
        src_kt, src_ext_kt, source_count = weighted_source_stats(src, target_month=month, dist_power=float(cfg.get("dist_power", 0.0)))
        if not np.isfinite(src_kt):
            src_kt = station_map.get(tgt[STATION], global_kt)
        if not np.isfinite(src_ext_kt) or src_ext_kt <= 1e-6:
            src_ext_kt = float(tgt["fusion_kt"])
        station_kt = station_map.get(tgt[STATION], global_kt)
        tgt_ext_kt = float(tgt["fusion_kt"])
        beta = float(cfg["beta"])
        if cfg["mode"] == "add":
            pred_kt = src_kt + beta * (tgt_ext_kt - src_ext_kt)
        elif cfg["mode"] == "mult":
            pred_kt = src_kt * float(np.clip(tgt_ext_kt / max(src_ext_kt, 0.05), 0.55, 1.65) ** beta)
        else:
            raise ValueError(str(cfg["mode"]))
        shrink = float(cfg["shrink"])
        pred_kt = (1.0 - shrink) * pred_kt + shrink * station_kt
        pred_kt = float(np.clip(pred_kt, 0.0, 1.45))
        rows.append(
            {
                STATION: tgt[STATION],
                "dt_year": int(tgt["dt_year"]),
                MONTH: month,
                "pred_kt": pred_kt,
                "pred_energy": pred_kt * float(tgt["clear_sum"]),
                "src_kt": src_kt,
                "src_ext_kt": src_ext_kt,
                "tgt_ext_kt": tgt_ext_kt,
                "source_count": source_count,
            }
        )
    return pd.DataFrame(rows)


def month_alpha(month: int, cfg: Mapping[str, float]) -> float:
    base = float(cfg.get("alpha", 0.15))
    if int(month) in (2, 3):
        return float(cfg.get("alpha_early", base))
    if int(month) in (11, 12):
        return float(cfg.get("alpha_late", base))
    if int(month) in (9, 10):
        return float(cfg.get("alpha_transition", base))
    return base


def reconstruct_monthly(frame: pd.DataFrame, shape: np.ndarray, pred_month: pd.DataFrame, cfg: Mapping[str, float]) -> np.ndarray:
    out = np.asarray(shape, dtype=float).copy()
    day = frame["daylight_model_flag"].to_numpy(dtype=bool)
    pred_key = pred_month.set_index([STATION, "dt_year", MONTH])["pred_energy"].to_dict()
    base = np.asarray(shape, dtype=float)
    for key, idx in frame.loc[day].groupby([STATION, "dt_year", MONTH], observed=True, sort=False).groups.items():
        loc = np.asarray(list(idx), dtype=int)
        base_sum = float(np.sum(base[loc]))
        model_sum = float(pred_key.get(key, base_sum))
        if base_sum <= 1e-6 or model_sum <= 1e-6:
            continue
        alpha = month_alpha(int(key[2]), cfg)
        target_sum = (1.0 - alpha) * base_sum + alpha * model_sum
        ratio = np.clip(target_sum / base_sum, 1.0 - float(cfg["ratio_clip"]), 1.0 + float(cfg["ratio_clip"]))
        out[loc] = base[loc] * ratio
    return final_clip(out, frame["phys_solar_elevation_clipped"].to_numpy(float))


def gate_mask(frame: pd.DataFrame, gate: str, validation: bool) -> np.ndarray:
    month_map = VALIDATION_GATE_MONTHS if validation else TEST_GATE_MONTHS
    if gate not in month_map:
        raise ValueError(f"unknown gate: {gate}")
    months = month_map[gate]
    return frame[MONTH].astype(int).isin(months).to_numpy(dtype=bool)


def neutralize_delta(frame: pd.DataFrame, delta: np.ndarray, gate: str, neutral: str, validation: bool) -> np.ndarray:
    out = np.asarray(delta, dtype=float).copy()
    use = gate_mask(frame, gate, validation=validation)
    day = frame["daylight_model_flag"].to_numpy(dtype=bool)
    active = use & day
    if neutral == "none" or not active.any():
        return out
    if neutral == "global":
        shift = float(np.mean(out))
        frac = float(np.mean(active))
        if frac > 1e-9:
            out[active] -= shift / frac
        return out
    group_cols = [MONTH]
    if neutral == "station_month":
        group_cols = [STATION, "dt_year", MONTH]
    elif neutral != "month":
        raise ValueError(f"unknown neutral mode: {neutral}")
    active_pos = np.flatnonzero(active)
    active_df = frame.iloc[active_pos][group_cols].copy()
    active_df["row_pos_local"] = active_pos
    for _, idx in active_df.groupby(group_cols, observed=True, sort=False)["row_pos_local"]:
        loc = idx.to_numpy(dtype=int)
        if len(loc) > 0:
            out[loc] -= float(np.mean(out[loc]))
    return out


def apply_gate_and_neutral(
    frame: pd.DataFrame,
    base: np.ndarray,
    raw_pred: np.ndarray,
    gate: str,
    neutral: str,
    validation: bool,
) -> np.ndarray:
    base_arr = np.asarray(base, dtype=float)
    raw_arr = np.asarray(raw_pred, dtype=float)
    delta = raw_arr - base_arr
    use = gate_mask(frame, gate, validation=validation)
    delta = np.where(use, delta, 0.0)
    delta = neutralize_delta(frame, delta, gate=gate, neutral=neutral, validation=validation)
    return final_clip(base_arr + delta, frame["phys_solar_elevation_clipped"].to_numpy(float))


def metric(y: np.ndarray, pred: np.ndarray, name: str) -> dict:
    err = np.asarray(pred, dtype=float) - np.asarray(y, dtype=float)
    mbe = float(np.mean(err))
    rmse = float(math.sqrt(np.mean(err * err)))
    return {"candidate": name, "n": int(len(err)), "mbe": mbe, "abs_mbe": abs(mbe), "rmse": rmse, "weighted": 0.5 * abs(mbe) + 0.5 * rmse}


def add_risk(row: dict, frame: pd.DataFrame, pred: np.ndarray) -> dict:
    y = frame[TARGET].to_numpy(float)
    month_rows = []
    for month, idx in frame.groupby(MONTH, observed=True).groups.items():
        loc = np.asarray(list(idx), dtype=int)
        m = metric(y[loc], pred[loc], f"m{int(month)}")
        row[f"m{int(month)}_mbe"] = m["mbe"]
        row[f"m{int(month)}_rmse"] = m["rmse"]
        month_rows.append(m)
    sm_abs = (
        pd.DataFrame({STATION: frame[STATION].to_numpy(), MONTH: frame[MONTH].to_numpy(), "err": pred - y})
        .groupby([STATION, MONTH], observed=True)["err"]
        .mean()
        .abs()
        .to_numpy(float)
    )
    month_df = pd.DataFrame(month_rows)
    row["max_abs_month_mbe"] = float(month_df["mbe"].abs().max()) if len(month_df) else 0.0
    row["p95_abs_station_month_mbe"] = float(np.quantile(sm_abs, 0.95)) if len(sm_abs) else 0.0
    row["risk_score"] = float(row["weighted"] + 0.03 * row["max_abs_month_mbe"] + 0.005 * row["p95_abs_station_month_mbe"])
    return row


def candidate_grid(fast: bool) -> List[dict]:
    # Validation showed opposite edge-season energy errors: March is high, November is low.
    # Search early/late tilt separately while keeping global mean neutralized downstream.
    if fast:
        alpha_pairs = [(0.08, 0.18), (0.10, 0.20), (0.12, 0.24), (0.15, 0.15)]
        powers = [1.25, 2.0]
        ratio_clips = [0.12]
    else:
        alpha_pairs = [
            (0.05, 0.15),
            (0.08, 0.18),
            (0.08, 0.24),
            (0.10, 0.15),
            (0.10, 0.20),
            (0.10, 0.25),
            (0.12, 0.20),
            (0.12, 0.24),
            (0.15, 0.15),
            (0.15, 0.25),
            (0.18, 0.30),
        ]
        powers = [0.75, 1.25, 2.0]
        ratio_clips = [0.10, 0.12, 0.16]
    base = []
    for alpha_early, alpha_late in alpha_pairs:
        for ratio_clip in ratio_clips:
            base.append(
                {
                    "mode": "add",
                    "beta": 1.0,
                    "shrink": 0.30,
                    "alpha": 0.15,
                    "alpha_early": alpha_early,
                    "alpha_late": alpha_late,
                    "ratio_clip": ratio_clip,
                }
            )
    rows = []
    for cfg in base:
        for power in powers:
            row = dict(cfg)
            row["dist_power"] = power
            rows.append(row)
    return rows


def gate_options(fast: bool) -> List[str]:
    return ["edge", "transition"] if fast else ["edge", "transition"]


def neutral_options(fast: bool) -> List[str]:
    return ["global"]


def run_validation(frame: pd.DataFrame, monthly: pd.DataFrame, valid_months: Sequence[int], fast: bool) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    val = frame.loc[frame[MONTH].astype(int).isin(valid_months)].copy().reset_index(drop=True)
    val_monthly = aggregate_month(val, include_target=True)
    source_monthly = monthly.copy()
    y = val[TARGET].to_numpy(float)
    shape_cols = {"shape_v166": "base_v166", "shape_v177": "base_v177"}
    rows = []
    pred_map = {}
    for shape_name, col in shape_cols.items():
        base = final_clip(val[col].to_numpy(float), val["phys_solar_elevation_clipped"].to_numpy(float))
        rows.append(add_risk(metric(y, base, f"{shape_name}_anchor"), val, base))
        pred_map[f"{shape_name}_anchor"] = base
    for cfg in candidate_grid(fast):
        pred_m = predict_monthly_kt(source_monthly, val_monthly, cfg, validation=True)
        for shape_name, col in shape_cols.items():
            base = pred_map[f"{shape_name}_anchor"]
            raw_pred = reconstruct_monthly(val, base, pred_m, cfg)
            anchor = rows[1] if shape_name == "shape_v177" else rows[0]
            for gate in gate_options(fast):
                for neutral in neutral_options(fast):
                    pred = apply_gate_and_neutral(val, base, raw_pred, gate=gate, neutral=neutral, validation=True)
                    name = (
                        f"{shape_name}_{cfg['mode']}_b{cfg['beta']:.2f}_s{cfg['shrink']:.2f}_"
                        f"ae{cfg.get('alpha_early', cfg['alpha']):.2f}_al{cfg.get('alpha_late', cfg['alpha']):.2f}_"
                        f"c{cfg['ratio_clip']:.2f}_dp{cfg.get('dist_power', 0.0):.2f}_{gate}_{neutral}"
                    )
                    row = add_risk(metric(y, pred, name), val, pred)
                    row.update(cfg)
                    row["shape"] = shape_name
                    row["gate"] = gate
                    row["neutral"] = neutral
                    row["mean_abs_delta_vs_anchor"] = float(np.mean(np.abs(pred - base)))
                    row["mean_delta_vs_anchor"] = float(np.mean(pred - base))
                    row["changed_row_ratio"] = float(np.mean(np.abs(pred - base) > 1e-9))
                    row["weighted_delta_vs_anchor"] = float(row["weighted"] - anchor["weighted"])
                    row["rmse_delta_vs_anchor"] = float(row["rmse"] - anchor["rmse"])
                    row["risk_delta_vs_anchor"] = float(row["risk_score"] - anchor["risk_score"])
                    row["selection_score"] = float(
                        row["risk_score"]
                        + 0.30 * abs(row["mean_delta_vs_anchor"])
                        + 0.03 * max(0.0, row["mean_abs_delta_vs_anchor"] - 6.0)
                    )
                    rows.append(row)
    return pd.DataFrame(rows), pred_map


def make_submission(sample: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    sub = pd.DataFrame({ID: sample[ID].to_numpy(), "TargetMBE": np.asarray(pred, dtype=float)})
    sub["TargetRMSE"] = sub["TargetMBE"].to_numpy(float)
    if not np.array_equal(sub[ID].to_numpy(), sample[ID].to_numpy()):
        raise ValueError("submission ID order mismatch")
    if not np.array_equal(sub["TargetMBE"].to_numpy(float), sub["TargetRMSE"].to_numpy(float)):
        raise ValueError("submission target columns differ")
    if not np.isfinite(sub["TargetMBE"].to_numpy(float)).all():
        raise ValueError("submission non-finite")
    sub.to_csv(path, index=False)


def safe_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)[:180]


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(RUN_DIR.resolve()).as_posix()
    except ValueError:
        return path.name


def build_test_candidates(train_monthly: pd.DataFrame, test: pd.DataFrame, test_monthly: pd.DataFrame, sample: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    refs = {
        "v166": read_submission(V166_SUB, sample, "v166"),
        "v177": read_submission(V177_SUB, sample, "v177"),
        "jieun20": read_submission(JIEUN20_SUB, sample, "jieun20"),
        "ds15": read_submission(DS15_SUB, sample, "ds15"),
    }
    out = {}
    for _, row in selected.iterrows():
        cfg = {
            k: row[k]
            for k in ["mode", "beta", "shrink", "alpha", "alpha_early", "alpha_late", "ratio_clip", "dist_power"]
            if k in row.index and pd.notna(row[k])
        }
        pred_m = predict_monthly_kt(train_monthly, test_monthly, cfg, validation=False)
        validated_ref_name = "v177" if row["shape"] == "shape_v177" else "v166"
        for anchor_name in ["jieun20", "ds15", validated_ref_name]:
            shape_ref = refs[anchor_name]
            raw_pred = reconstruct_monthly(test, shape_ref, pred_m, cfg)
            pred = apply_gate_and_neutral(
                test,
                shape_ref,
                raw_pred,
                gate=str(row.get("gate", "all")),
                neutral=str(row.get("neutral", "none")),
                validation=False,
            )
            out[f"{row['candidate']}__anchor_{anchor_name}"] = pred
    diag = test_diagnostics(test, out, refs)
    diag.to_csv(DIAG_DIR / "test_candidate_diagnostics.csv", index=False)
    for i, (name, pred) in enumerate(out.items()):
        make_submission(sample, pred, CAND_DIR / f"{safe_filename(name)}.csv")
        if i == 0:
            make_submission(sample, pred, RUN_DIR / "submission.csv")
    return diag


def test_diagnostics(frame: pd.DataFrame, candidates: Mapping[str, np.ndarray], refs: Mapping[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    day = frame["daylight_model_flag"].to_numpy(dtype=bool)
    clear = np.maximum(frame["clear_ref_safe"].to_numpy(float), 35.0)
    for name, pred in candidates.items():
        pred = np.asarray(pred, dtype=float)
        row = {
            "candidate": name,
            "mean": float(np.mean(pred)),
            "day_mean": float(np.mean(pred[day])) if day.any() else 0.0,
            "zero_ratio": float(np.mean(pred <= 1e-9)),
            "p50": float(np.quantile(pred, 0.50)),
            "p90": float(np.quantile(pred, 0.90)),
            "p99": float(np.quantile(pred, 0.99)),
            "kt_gt_1p15_day": float(np.mean((pred[day] / clear[day]) > 1.15)) if day.any() else 0.0,
        }
        for ref_name, ref in refs.items():
            row[f"mae_vs_{ref_name}"] = float(np.mean(np.abs(pred - ref)))
            row[f"rmse_vs_{ref_name}"] = float(math.sqrt(np.mean(np.square(pred - ref))))
            row[f"mean_delta_vs_{ref_name}"] = float(np.mean(pred - ref))
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary(selected: pd.DataFrame, valid_months: Sequence[int], fast: bool) -> None:
    cols = [
        c
        for c in [
            "candidate",
            "gate",
            "neutral",
            "dist_power",
            "alpha_early",
            "alpha_late",
            "weighted",
            "rmse",
            "abs_mbe",
            "risk_score",
            "weighted_delta_vs_best_anchor",
            "risk_delta_vs_best_anchor",
            "submission_guard",
        ]
        if c in selected.columns
    ]
    lines = [
        f"# {RUN_NAME}",
        "",
        "## Hypothesis",
        "Distance-weighted adjacent-month kt anomaly is applied as a mean-neutral shape correction to strong public/team anchors, especially jieun20. v28.0 separates early-edge and late-edge correction strength because validation and public feedback indicate an asymmetric seasonal energy tilt: early months need smaller correction than late months.",
        "",
        "## Leakage guard",
        f"Validation months: {list(valid_months)}. Held-out target month energy is predicted from adjacent non-held-out odd months. Mean-neutral modes use predictions only, not held-out targets. Test candidates are emitted for jieun20/ds15 plus the validation anchor.",
        "",
        "## Selected candidates",
        "```csv",
        selected.head(10)[cols].to_csv(index=False).strip(),
        "```",
    ]
    (RUN_DIR / "V28_0_METHOD_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "submit"], default="validate")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-rows", type=int, default=120000)
    args = parser.parse_args(argv)

    ensure_dirs()
    (RUN_DIR / "train.log").write_text("", encoding="utf-8")
    require_inputs()

    smoke_rows = args.smoke_rows if args.smoke_test else None
    valid_months = [3] if args.smoke_test else (FAST_MONTHS if args.fast else VALID_MONTHS)
    log(f"Reading frames smoke_rows={smoke_rows}, valid_months={valid_months}")
    train, test, sample = read_frames(smoke_rows)
    train = add_aux(train)
    test = add_aux(test)
    train = train.merge(read_val_base(V166_VAL, "selected_pred_v166", "base_v166"), on=ID, how="inner")
    train = train.merge(read_val_base(V177_VAL, "selected_pred", "base_v177"), on=ID, how="left")
    train["base_v177"] = train["base_v177"].fillna(train["base_v166"])
    monthly = aggregate_month(train, include_target=True)
    test_monthly = aggregate_month(test, include_target=False)
    metrics, pred_map = run_validation(train, monthly, valid_months, fast=args.fast or args.smoke_test)
    anchors = metrics[metrics["candidate"].str.endswith("_anchor")].copy()
    challengers = metrics[~metrics["candidate"].str.endswith("_anchor")].copy()
    best_anchor = anchors.sort_values("risk_score").iloc[0]
    challengers["risk_delta_vs_best_anchor"] = challengers["risk_score"] - float(best_anchor["risk_score"])
    challengers["weighted_delta_vs_best_anchor"] = challengers["weighted"] - float(best_anchor["weighted"])
    challengers["submission_guard"] = (
        (challengers["weighted_delta_vs_best_anchor"] <= -0.08)
        & (challengers["risk_delta_vs_best_anchor"] <= -0.08)
        & (challengers["rmse_delta_vs_anchor"] <= 0.00)
        & (challengers["abs_mbe"] <= float(best_anchor["abs_mbe"]) + 0.03)
        & (challengers["mean_abs_delta_vs_anchor"] <= 8.0)
        & (challengers["max_abs_month_mbe"] <= float(best_anchor["max_abs_month_mbe"]) + 0.30)
    )
    challengers = challengers.sort_values(["submission_guard", "selection_score", "weighted"], ascending=[False, True, True])
    metrics.to_csv(DIAG_DIR / ("candidate_metrics_smoke.csv" if args.smoke_test else "candidate_metrics.csv"), index=False)
    challengers.to_csv(DIAG_DIR / ("submission_decision_smoke.csv" if args.smoke_test else "submission_decision.csv"), index=False)
    selected = challengers.head(8).copy()
    test_diag = None
    if args.mode == "submit" and not args.smoke_test:
        if not bool(selected.iloc[0]["submission_guard"]):
            log("No candidate passed strict guard; writing diagnostic candidates only.")
        test_diag = build_test_candidates(monthly, test, test_monthly, sample, selected.head(5))
    write_summary(selected, valid_months, fast=args.fast)
    summary = {
        "run": RUN_NAME,
        "mode": args.mode,
        "fast": bool(args.fast),
        "smoke_test": bool(args.smoke_test),
        "valid_months": valid_months,
        "best_anchor": best_anchor.to_dict(),
        "top_candidates": selected.to_dict("records"),
        "test_top_candidates": [] if test_diag is None else test_diag.head(5).to_dict("records"),
        "outputs": {
            "candidate_metrics": display_path(DIAG_DIR / ("candidate_metrics_smoke.csv" if args.smoke_test else "candidate_metrics.csv")),
            "submission_decision": display_path(DIAG_DIR / ("submission_decision_smoke.csv" if args.smoke_test else "submission_decision.csv")),
            "submission": display_path(RUN_DIR / "submission.csv") if (RUN_DIR / "submission.csv").exists() else None,
        },
    }
    with (RUN_DIR / ("metrics_smoke.json" if args.smoke_test else "metrics.json")).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log("Completed v28.0 asymmetric edge kt tilt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
