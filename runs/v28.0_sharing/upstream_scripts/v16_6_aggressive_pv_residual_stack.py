from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


RUN_DIR = Path("runs/v16.6_aggressive_pv_residual_stack")
DIAG_DIR = RUN_DIR / "diagnostics"
V165_SCRIPT = Path("runs/v16.5_daily_energy_pv_residual_stack/v16_5_daily_energy_pv_residual_stack.py")


def ensure_dirs() -> None:
    for path in [RUN_DIR, DIAG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def load_v165():
    spec = importlib.util.spec_from_file_location("v165", V165_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load v16.5 helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate_grid() -> list[dict]:
    rows = []
    for daily_weight in [0.70, 0.85, 1.00]:
        for pv_weight in [0.20, 0.24, 0.28, 0.32, 0.36, 0.42, 0.50]:
            for pv_cap_down, pv_cap_up in [(40.0, 30.0), (70.0, 45.0), (110.0, 55.0), (140.0, 70.0)]:
                for pv_min_abs in [4.0, 10.0, 18.0, 28.0]:
                    rows.append(
                        {
                            "daily_weight": daily_weight,
                            "pv_weight": pv_weight,
                            "pv_cap_down": pv_cap_down,
                            "pv_cap_up": pv_cap_up,
                            "pv_min_abs": pv_min_abs,
                            "min_solar": 3.0,
                            "min_clear": 40.0,
                            "name": f"aggr_dw{daily_weight:.2f}_pw{pv_weight:.2f}_dn{pv_cap_down:.0f}_up{pv_cap_up:.0f}_min{pv_min_abs:.0f}".replace(".", "p"),
                        }
                    )
    return rows


def main() -> None:
    ensure_dirs()
    v165 = load_v165()
    v163 = v165.load_v163()
    write_submission = v165.load_submission_guard()

    log("Loading stack inputs")
    sample = pd.read_csv(v165.SAMPLE_PATH, usecols=[v165.ID])
    base_sub = pd.read_csv(v165.BASE_SUB_PATH)
    v161_sub = pd.read_csv(v165.V161_SUB_PATH)
    if not np.array_equal(base_sub[v165.ID].to_numpy(), sample[v165.ID].to_numpy()):
        raise ValueError("v15.9.2 submission ID order mismatch")
    if not np.array_equal(v161_sub[v165.ID].to_numpy(), sample[v165.ID].to_numpy()):
        raise ValueError("v16.1 submission ID order mismatch")

    v161_val = pd.read_csv(v165.V161_VAL_PATH, usecols=[v165.ID, v165.TARGET, v165.STATION, "month", "solar_elevation", v165.BASE, "selected_pred", "selected_gate"])
    pv_val = pd.read_csv(v165.PV_VAL_PATH, usecols=[v165.ID, v165.PV_GHI, "oof_residual_correction"])
    pv_test = pd.read_csv(v165.PV_TEST_PATH, usecols=[v165.ID, v165.STATION, "month", "solar_elevation", v165.PV_GHI, "test_residual_correction"])
    val = v161_val.merge(pv_val, on=v165.ID, how="left")
    test = sample.merge(pv_test, on=v165.ID, how="left")
    test[v165.BASE] = base_sub["TargetMBE"].to_numpy(float)
    test["v16_1_pred"] = v161_sub["TargetMBE"].to_numpy(float)

    base_val = val[v165.BASE].to_numpy(float)
    base_test = test[v165.BASE].to_numpy(float)
    daily_val_delta = val["selected_pred"].to_numpy(float) - base_val
    daily_test_delta = test["v16_1_pred"].to_numpy(float) - base_test
    pv_val_corr = val["oof_residual_correction"].fillna(0.0).to_numpy(float)
    pv_test_corr = test["test_residual_correction"].fillna(0.0).to_numpy(float)
    base_test_mean = float(np.mean(base_test))

    rows = [
        v163.evaluate(val, test, base_val, base_test, np.zeros(len(val), dtype=bool), np.zeros(len(test), dtype=bool), "v15_9_2_public_anchor", base_test_mean),
        v163.evaluate(val, test, base_val + daily_val_delta, base_test + daily_test_delta, np.abs(daily_val_delta) > 1.0e-9, np.abs(daily_test_delta) > 1.0e-9, "v16_1_daily_energy_anchor", base_test_mean),
    ]
    registry = {
        "v15_9_2_public_anchor": (base_val, base_test, np.zeros(len(val), dtype=bool), np.zeros(len(test), dtype=bool), {"mode": "anchor"}),
        "v16_1_daily_energy_anchor": (base_val + daily_val_delta, base_test + daily_test_delta, np.abs(daily_val_delta) > 1.0e-9, np.abs(daily_test_delta) > 1.0e-9, {"mode": "v16_1"}),
    }

    log("Evaluating aggressive pv residual stack candidates")
    for cfg in candidate_grid():
        pred, gate = v165.apply_stack(val, daily_val_delta, pv_val_corr, cfg)
        test_pred, test_gate = v165.apply_stack(test, daily_test_delta, pv_test_corr, cfg)
        rows.append(v163.evaluate(val, test, pred, test_pred, gate, test_gate, cfg["name"], base_test_mean))
        registry[cfg["name"]] = (pred, test_pred, gate, test_gate, cfg)

    metrics = v165.add_deltas(pd.DataFrame(rows))
    metrics["submission_candidate"] = (
        (metrics["candidate"] != "v15_9_2_public_anchor")
        & (metrics["weighted_delta_vs_v1592"] <= -0.060)
        & (metrics["risk_score_delta_vs_v1592"] <= -0.060)
        & (metrics["rmse_delta_vs_v1592"] <= 0.080)
        & (metrics["max_abs_month_mbe_delta_vs_v1592"] <= 0.25)
        & (metrics["p95_abs_station_month_mbe_delta_vs_v1592"] <= 0.20)
        & (metrics["test_mean_delta_vs_v1592"].abs() <= 0.350)
        & (metrics["test_gate_n"] >= 100)
    )
    metrics = metrics.sort_values(["submission_candidate", "selection_score", "weighted", "rmse"], ascending=[False, True, True, True]).reset_index(drop=True)
    metrics.to_csv(DIAG_DIR / "candidate_metrics.csv", index=False)
    metrics[metrics["submission_candidate"]].to_csv(DIAG_DIR / "safe_candidates.csv", index=False)

    selected_name = str(metrics[metrics["submission_candidate"]].iloc[0]["candidate"]) if metrics["submission_candidate"].any() else str(metrics.iloc[0]["candidate"])
    selected_val, selected_test, selected_gate, selected_test_gate, selected_cfg = registry[selected_name]
    write_submission(RUN_DIR / "submission.csv", sample, selected_test)

    val[[v165.ID, v165.TARGET, v165.STATION, "month", "solar_elevation", v165.PV_GHI, v165.BASE, "selected_pred", "oof_residual_correction"]].assign(
        selected_pred_v166=selected_val,
        selected_gate=selected_gate.astype(np.int8),
    ).to_csv(RUN_DIR / "val_predictions.csv", index=False)
    v163.load_v160().load_helper().station_month_delta(val, base_val, selected_val, selected_gate).to_csv(DIAG_DIR / "selected_station_month_delta.csv", index=False)
    test[[v165.ID, v165.STATION, "month", "solar_elevation", v165.PV_GHI, v165.BASE, "v16_1_pred", "test_residual_correction"]].assign(
        selected_pred=selected_test,
        selected_gate=selected_test_gate.astype(np.int8),
    ).to_csv(DIAG_DIR / "test_selection_diagnostics.csv", index=False)

    with (RUN_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run": "v16.6",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "goal": "Stress-test stronger pvlib residual stacking while preserving one-vector submission compliance.",
                "hypothesis": "A slightly stronger fold-safe pvlib residual can materially reduce MBE enough to offset controlled RMSE movement.",
                "selected_candidate": metrics[metrics["candidate"] == selected_name].iloc[0].to_dict(),
                "selected_config": selected_cfg,
                "safe_candidate_count": int(metrics["submission_candidate"].sum()),
                "anchor_v15_9_2": metrics[metrics["candidate"] == "v15_9_2_public_anchor"].iloc[0].to_dict(),
                "anchor_v16_1": metrics[metrics["candidate"] == "v16_1_daily_energy_anchor"].iloc[0].to_dict(),
                "outputs": {
                    "submission": str(RUN_DIR / "submission.csv"),
                    "val_predictions": str(RUN_DIR / "val_predictions.csv"),
                    "candidate_metrics": str(DIAG_DIR / "candidate_metrics.csv"),
                    "safe_candidates": str(DIAG_DIR / "safe_candidates.csv"),
                    "selected_station_month_delta": str(DIAG_DIR / "selected_station_month_delta.csv"),
                    "test_selection_diagnostics": str(DIAG_DIR / "test_selection_diagnostics.csv"),
                },
            },
            f,
            indent=2,
            default=str,
        )
    with (RUN_DIR / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run": "v16.6_aggressive_pv_residual_stack",
                "python": ".\\.venv\\Scripts\\python.exe",
                "script": str(RUN_DIR / "v16_6_aggressive_pv_residual_stack.py"),
                "candidate_count": len(candidate_grid()),
                "submission_guard": "scripts/submission_guard.py",
            },
            f,
            indent=2,
        )
    with (RUN_DIR / "train.log").open("w", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] Command:\n")
        f.write(f".\\.venv\\Scripts\\python.exe .\\{RUN_DIR}\\v16_6_aggressive_pv_residual_stack.py\n\n")
        f.write(f"Selected: {selected_name}\n")
        f.write(f"Safe candidates: {int(metrics['submission_candidate'].sum())}\n")
        f.write(f"Submission: {RUN_DIR / 'submission.csv'}\n")
    log(f"Safe candidates: {int(metrics['submission_candidate'].sum())}")
    log(f"Selected: {selected_name}")
    log(f"Submission: {RUN_DIR / 'submission.csv'}")


if __name__ == "__main__":
    main()
