from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


RUN_DIR = Path("runs/v17.7_dual_blend_station_filter")
DIAG_DIR = RUN_DIR / "diagnostics"
V176_DIR = Path("runs/v17.6_dual_regime_correction_blend")
SAMPLE_PATH = Path("SampleSubmission.csv")

ID = "ID"
TARGET = "radiation (W/m2)"
STATION = "station"


def ensure_dirs() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def load_submission_guard():
    sys.path.insert(0, str(Path("scripts").resolve()))
    from submission_guard import write_submission

    return write_submission


def metric(y: np.ndarray, pred: np.ndarray, name: str) -> dict:
    err = pred - y
    mbe = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err * err)))
    return {"candidate": name, "n": int(len(err)), "mbe": mbe, "abs_mbe": abs(mbe), "rmse": rmse, "weighted": 0.5 * abs(mbe) + 0.5 * rmse}


def add_risk(row: dict, frame: pd.DataFrame, pred: np.ndarray) -> dict:
    frame = frame.reset_index(drop=True)
    pred = np.asarray(pred, dtype=float)
    y = frame[TARGET].to_numpy(float)
    month_rows = []
    for month, idx in frame.groupby("month", observed=True).groups.items():
        loc = np.asarray(list(idx), dtype=int)
        m = metric(y[loc], pred[loc], f"m{int(month)}")
        row[f"m{int(month)}_mbe"] = m["mbe"]
        row[f"m{int(month)}_rmse"] = m["rmse"]
        month_rows.append(m)
    station_month_abs = frame.assign(error=pred - y).groupby([STATION, "month"], observed=True)["error"].mean().abs().to_numpy(float)
    month_df = pd.DataFrame(month_rows)
    row["max_abs_month_mbe"] = float(month_df["mbe"].abs().max())
    row["p95_abs_station_month_mbe"] = float(np.quantile(station_month_abs, 0.95)) if len(station_month_abs) else 0.0
    row["risk_score"] = float(row["weighted"] + 0.03 * row["max_abs_month_mbe"] + 0.005 * row["p95_abs_station_month_mbe"])
    return row


def station_effects(val: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station, group in val.groupby(STATION, observed=True):
        y = group[TARGET].to_numpy(float)
        base = group["v166_base"].to_numpy(float)
        pred = group["selected_pred"].to_numpy(float)
        base_err = base - y
        pred_err = pred - y
        base_rmse = float(np.sqrt(np.mean(base_err * base_err)))
        pred_rmse = float(np.sqrt(np.mean(pred_err * pred_err)))
        base_weighted = 0.5 * abs(float(np.mean(base_err))) + 0.5 * base_rmse
        pred_weighted = 0.5 * abs(float(np.mean(pred_err))) + 0.5 * pred_rmse
        rows.append(
            {
                "station": station,
                "n": int(len(group)),
                "mean_abs_delta": float(np.mean(np.abs(pred - base))),
                "mean_delta": float(np.mean(pred - base)),
                "weighted_delta": pred_weighted - base_weighted,
                "rmse_delta": pred_rmse - base_rmse,
            }
        )
    return pd.DataFrame(rows)


def evaluate(val: pd.DataFrame, pred: np.ndarray, test_pred: np.ndarray, name: str, base_test_mean: float) -> dict:
    row = add_risk(metric(val[TARGET].to_numpy(float), pred, name), val, pred)
    row["test_mean"] = float(np.mean(test_pred))
    row["test_mean_delta_vs_v166"] = float(np.mean(test_pred) - base_test_mean)
    row["test_zero_ratio"] = float(np.mean(test_pred <= 1.0e-9))
    row["val_mean_abs_delta"] = float(np.mean(np.abs(pred - val["v166_base"].to_numpy(float))))
    row["test_mean_abs_delta"] = float(np.mean(np.abs(test_pred - test["v166_base"].to_numpy(float)))) if "test" in globals() else np.nan
    return row


def apply_filter(val: pd.DataFrame, test_frame: pd.DataFrame, bad: set[str]) -> tuple[np.ndarray, np.ndarray]:
    pred = val["selected_pred"].to_numpy(float).copy()
    test_pred = test_frame["selected_pred"].to_numpy(float).copy()
    mask = val[STATION].isin(bad).to_numpy()
    test_mask = test_frame[STATION].isin(bad).to_numpy()
    pred[mask] = val.loc[mask, "v166_base"].to_numpy(float)
    test_pred[test_mask] = test_frame.loc[test_mask, "v166_base"].to_numpy(float)
    return pred, test_pred


def main() -> None:
    global test
    ensure_dirs()
    write_submission = load_submission_guard()
    log("Loading v17.6 dual correction blend")
    val = pd.read_csv(V176_DIR / "val_predictions.csv")
    test = pd.read_csv(V176_DIR / "diagnostics" / "test_component_predictions.csv")
    sample = pd.read_csv(SAMPLE_PATH, usecols=[ID])
    if not np.array_equal(test[ID].to_numpy(), sample[ID].to_numpy()):
        raise ValueError("v17.6 test diagnostics ID order mismatch.")
    base_val = val["v166_base"].to_numpy(float)
    base_test = test["v166_base"].to_numpy(float)
    base_test_mean = float(np.mean(base_test))
    effects = station_effects(val)
    effects.to_csv(DIAG_DIR / "station_effects.csv", index=False)

    rows = [evaluate(val, base_val, base_test, "v16_6_public_anchor", base_test_mean)]
    rows.append(evaluate(val, val["selected_pred"].to_numpy(float), test["selected_pred"].to_numpy(float), "v17_6_dual_anchor", base_test_mean))
    registry = {
        "v16_6_public_anchor": (base_val, base_test, {"mode": "anchor"}),
        "v17_6_dual_anchor": (val["selected_pred"].to_numpy(float), test["selected_pred"].to_numpy(float), {"mode": "v17.6"}),
    }

    log("Evaluating station filters after dual correction blend")
    for weighted_threshold in [0.0, 0.001, 0.003, 0.005, 0.010, 0.020, 0.040]:
        bad = set(effects[effects["weighted_delta"] > weighted_threshold][STATION].astype(str))
        pred, test_pred = apply_filter(val, test, bad)
        name = f"dual_station_filter_wthr{weighted_threshold:.3f}".replace(".", "p")
        row = evaluate(val, pred, test_pred, name, base_test_mean)
        row["filtered_station_n"] = int(len(bad))
        row["filtered_stations"] = ",".join(sorted(bad))
        rows.append(row)
        registry[name] = (pred, test_pred, {"weighted_threshold": weighted_threshold, "filtered_stations": sorted(bad)})

    metrics = pd.DataFrame(rows)
    anchor = metrics[metrics["candidate"] == "v16_6_public_anchor"].iloc[0]
    for col in ["rmse", "weighted", "risk_score", "max_abs_month_mbe", "p95_abs_station_month_mbe"]:
        metrics[f"{col}_delta_vs_v166"] = metrics[col] - float(anchor[col])
    metrics["delta_abs_gap"] = (metrics["test_mean_abs_delta"] - metrics["val_mean_abs_delta"]).abs()
    metrics["submission_candidate"] = (
        (metrics["candidate"] != "v16_6_public_anchor")
        & (metrics["weighted_delta_vs_v166"] <= -0.028)
        & (metrics["risk_score_delta_vs_v166"] <= -0.027)
        & (metrics["rmse_delta_vs_v166"] <= -0.055)
        & (metrics["max_abs_month_mbe_delta_vs_v166"] <= 0.12)
        & (metrics["p95_abs_station_month_mbe_delta_vs_v166"] <= 0.08)
        & (metrics["test_mean_delta_vs_v166"].abs() <= 0.100)
        & (metrics["delta_abs_gap"] <= 2.30)
    )
    metrics["selection_score"] = metrics["risk_score"] + 0.04 * metrics["test_mean_delta_vs_v166"].abs() + 0.005 * metrics["delta_abs_gap"]
    metrics = metrics.sort_values(["submission_candidate", "selection_score", "weighted", "rmse"], ascending=[False, True, True, True]).reset_index(drop=True)
    metrics.to_csv(DIAG_DIR / "candidate_metrics.csv", index=False)
    metrics[metrics["submission_candidate"]].to_csv(DIAG_DIR / "safe_candidates.csv", index=False)

    selected = str(metrics[metrics["submission_candidate"]].iloc[0]["candidate"]) if metrics["submission_candidate"].any() else str(metrics.iloc[0]["candidate"])
    selected_val, selected_test, selected_cfg = registry[selected]
    write_submission(RUN_DIR / "submission.csv", sample, selected_test)

    val[[ID, TARGET, STATION, "month", "solar_elevation", "clear_ref", "v166_base"]].assign(selected_pred=selected_val).to_csv(RUN_DIR / "val_predictions.csv", index=False)
    test[[ID, STATION, "month", "solar_elevation", "clear_ref", "v166_base"]].assign(selected_pred=selected_test).to_csv(DIAG_DIR / "test_component_predictions.csv", index=False)

    with (RUN_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run": "v17.7_dual_blend_station_filter",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "goal": "Remove station-level harm after the v17.6 dual correction blend.",
                "method": "Station reliability filter learned from validation station weighted deltas; affected stations revert to v16.6 base.",
                "selected_candidate": metrics[metrics["candidate"] == selected].iloc[0].to_dict(),
                "selected_config": selected_cfg,
                "safe_candidate_count": int(metrics["submission_candidate"].sum()),
                "outputs": {
                    "submission": str(RUN_DIR / "submission.csv"),
                    "val_predictions": str(RUN_DIR / "val_predictions.csv"),
                    "candidate_metrics": str(DIAG_DIR / "candidate_metrics.csv"),
                    "safe_candidates": str(DIAG_DIR / "safe_candidates.csv"),
                    "station_effects": str(DIAG_DIR / "station_effects.csv"),
                    "test_component_predictions": str(DIAG_DIR / "test_component_predictions.csv"),
                },
            },
            f,
            indent=2,
            default=str,
        )
    with (RUN_DIR / "config.json").open("w", encoding="utf-8") as f:
        json.dump({"run": "v17.7_dual_blend_station_filter", "python": ".\\.venv\\Scripts\\python.exe", "script": str(RUN_DIR / "v17_7_dual_blend_station_filter.py")}, f, indent=2)
    with (RUN_DIR / "train.log").open("w", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] Command:\n")
        f.write(f".\\.venv\\Scripts\\python.exe .\\{RUN_DIR}\\v17_7_dual_blend_station_filter.py\n")
        f.write(f"Selected: {selected}\n")
        f.write(f"Safe candidates: {int(metrics['submission_candidate'].sum())}\n")
        f.write(f"Submission: {RUN_DIR / 'submission.csv'}\n")
    log(f"Selected: {selected}")
    log(f"Safe candidates: {int(metrics['submission_candidate'].sum())}")
    log(f"Submission: {RUN_DIR / 'submission.csv'}")


if __name__ == "__main__":
    main()
