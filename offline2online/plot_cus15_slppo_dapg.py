from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = int(os.environ.get("SEED", "2005"))
EVRPTW_DB_ROOT = Path(os.environ.get("EVRPTW_DB_ROOT", "/data/Maojie/EVRPTW-DB")).resolve()
RUNS = [
    {
        "label": "SL-PPO u4 group+ref",
        "run_name": "O2O_CUS15_SL_PPO_GROUP_REF_R40_U4_E1000",
        "color": "#4c78a8",
        "linestyle": "--",
    },
    {
        "label": "SL-PPO u4 group+ref-gap + dyn + mem-gate + DDE-KV",
        "run_name": "O2O_CUS15_SL_PPO_GROUP_REFGAP_MEM_DYN_DDE_KV_R40_U4_E1000",
        "color": "#e45756",
        "linestyle": "-",
    },
    {
        "label": "FRRO u4 + dyn + mem + DDE-KV, alpha=0.10, lambda_E=2",
        "run_name": "O2O_CUS15_FRRO_A010_LE2_DYN_MEM_DDE_KV_R40_U4_E1000",
        "color": "#111111",
        "linestyle": "-",
    },
    {
        "label": "FRRO u4 + dyn + mem + DDE-KV, alpha=0.10, lambda_E=4",
        "run_name": "O2O_CUS15_FRRO_A010_LE4_DYN_MEM_DDE_KV_R40_U4_E1000",
        "color": "#e45756",
        "linestyle": "-",
    },
    {
        "label": "DAPG u4",
        "run_name": "O2O_CUS15_DAPG_R40_U4_E1000",
        "color": "#f58518",
        "linestyle": "--",
    },
    {
        "label": "DAPG u4 + dyn",
        "run_name": "O2O_CUS15_DAPG_DYN_R40_U4_E1000",
        "color": "#9467bd",
        "linestyle": "-.",
    },
]
DEFAULT_GUROBI_VAL_SUMMARY = EVRPTW_DB_ROOT / "results/offline_experts/dataset_v1/val/Cus15_Gurobi_2h/gurobi_summary.csv"
GUROBI_VAL_SUMMARY = Path(os.environ.get("CUS15_GUROBI_VAL_SUMMARY", str(DEFAULT_GUROBI_VAL_SUMMARY)))
OUTPUT_STEM = "cus15_offline_dynamic_update_ablation_r40_e1000"


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_eval_log(run_name: str) -> list[dict[str, float]]:
    path = REPO_ROOT / "results" / "logs" / "Cus_15_CS_3" / run_name / f"seed_{SEED}" / "eval_log.csv"
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("eval_status", "ok") != "ok":
                continue
            epoch = _float_or_nan(row.get("epoch"))
            obj = _float_or_nan(row.get("eval_avg_objective_distance_km"))
            fr = _float_or_nan(row.get("eval_feasible_rate"))
            if np.isfinite(epoch) and np.isfinite(obj):
                rows.append({"epoch": epoch, "objective": obj, "feasible_rate": fr})
    return rows


def gurobi_best_average() -> tuple[float, int]:
    if not GUROBI_VAL_SUMMARY.exists():
        return float("nan"), 0
    values: list[float] = []
    with GUROBI_VAL_SUMMARY.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            feasible = str(row.get("feasible", "")).lower()
            if feasible and feasible not in {"true", "1", "yes"}:
                continue
            objective = _float_or_nan(row.get("objective_distance_km"))
            if np.isfinite(objective):
                values.append(objective)
    if not values:
        return float("nan"), 0
    return float(np.mean(values)), len(values)


def write_summary(run_rows: dict[str, list[dict[str, float]]], gurobi_avg: float, gurobi_n: int) -> None:
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{OUTPUT_STEM}_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "latest_epoch", "latest_objective_km", "best_epoch", "best_objective_km", "gurobi_best_avg_km", "gurobi_n"],
        )
        writer.writeheader()
        for run in RUNS:
            rows = run_rows[run["label"]]
            if rows:
                latest = rows[-1]
                best = min(rows, key=lambda item: item["objective"])
                writer.writerow(
                    {
                        "method": run["label"],
                        "latest_epoch": int(latest["epoch"]),
                        "latest_objective_km": latest["objective"],
                        "best_epoch": int(best["epoch"]),
                        "best_objective_km": best["objective"],
                        "gurobi_best_avg_km": gurobi_avg,
                        "gurobi_n": gurobi_n,
                    }
                )
            else:
                writer.writerow(
                    {
                        "method": run["label"],
                        "latest_epoch": "",
                        "latest_objective_km": "",
                        "best_epoch": "",
                        "best_objective_km": "",
                        "gurobi_best_avg_km": gurobi_avg,
                        "gurobi_n": gurobi_n,
                    }
                )


def main() -> None:
    fig_dir = REPO_ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    run_rows = {run["label"]: read_eval_log(run["run_name"]) for run in RUNS}
    gurobi_avg, gurobi_n = gurobi_best_average()
    write_summary(run_rows, gurobi_avg, gurobi_n)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=180)
    has_curve = False
    for run in RUNS:
        rows = run_rows[run["label"]]
        if not rows:
            continue
        has_curve = True
        epochs = [row["epoch"] for row in rows]
        objectives = [row["objective"] for row in rows]
        best = min(rows, key=lambda item: item["objective"])
        ax.plot(
            epochs,
            objectives,
            marker="o",
            markersize=3.5,
            linewidth=2.0,
            color=run["color"],
            linestyle=run["linestyle"],
            label=run["label"],
        )
        ax.scatter([best["epoch"]], [best["objective"]], s=48, color=run["color"], edgecolor="black", linewidth=0.8, zorder=5)

    if np.isfinite(gurobi_avg):
        ax.axhline(
            gurobi_avg,
            color="#2ca02c",
            linestyle="--",
            linewidth=1.8,
            label=f"Gurobi best avg ({gurobi_avg:.2f} km, n={gurobi_n})",
        )

    ax.set_title("Cus15 Validation Objective: Offline-to-Online Dynamic/Update Ablation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg objective distance (km)")
    ax.set_xlim(0, 1000)
    if not has_curve and np.isfinite(gurobi_avg):
        ax.set_ylim(max(0.0, gurobi_avg - 20.0), gurobi_avg + 80.0)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best")
    fig.tight_layout()
    png_path = fig_dir / f"{OUTPUT_STEM}.png"
    pdf_path = fig_dir / f"{OUTPUT_STEM}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {png_path}")
    print(f"saved {pdf_path}")


if __name__ == "__main__":
    main()
