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
RUNS = [
    {
        "label": "DAPG u4",
        "run_name": "O2O_CUS50_DAPG_R70_U4_E1000",
        "color": "#f58518",
        "linestyle": "--",
    },
    {
        "label": "DAPG u4 + dyn",
        "run_name": "O2O_CUS50_DAPG_DYN_R70_U4_E1000",
        "color": "#9467bd",
        "linestyle": "-.",
    },
    {
        "label": "FRRO u4 + dyn + mem + DDE-KV, alpha=0.10, lambda_E=2",
        "run_name": "O2O_CUS50_FRRO_A010_LE2_DYN_MEM_DDE_KV_R70_U4_E1000",
        "color": "#111111",
        "linestyle": "-",
    },
    {
        "label": "FRRO u4 + dyn + mem + DDE-KV, alpha=0.10, lambda_E=4",
        "run_name": "O2O_CUS50_FRRO_A010_LE4_DYN_MEM_DDE_KV_R70_U4_E1000",
        "color": "#e45756",
        "linestyle": "-",
    },
]
GUROBI_VAL_SUMMARY = Path(os.environ.get("CUS50_GUROBI_VAL_SUMMARY", "/data/Maojie/gurobi_mul/results/val/Cus50/gurobi_summary.csv"))
OUTPUT_STEM = "cus50_offline_update_ablation_r70_e1000"
LEGACY_OUTPUT_STEM = "cus50_slppo_group_ref_vs_dapg_r70_u2_e1000"


def log_roots() -> list[Path]:
    roots = [REPO_ROOT / "results" / "logs" / "Cus_50_CS_10"]
    prev_root = REPO_ROOT.parent / "EVRPTW-OFFLINE2ONLINE_Prev" / "results" / "logs" / "Cus_50_CS_10"
    if prev_root.exists():
        roots.append(prev_root)
    extra = os.environ.get("CUS50_EXTRA_LOG_ROOTS")
    if extra:
        roots.extend(Path(item) for item in extra.split(os.pathsep) if item)
    return roots


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_eval_log(run_name: str) -> list[dict[str, float]]:
    path = next(
        (
            root / run_name / f"seed_{SEED}" / "eval_log.csv"
            for root in log_roots()
            if (root / run_name / f"seed_{SEED}" / "eval_log.csv").exists()
        ),
        None,
    )
    if path is None:
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
            if str(row.get("feasible", "")).lower() not in {"true", "1", "yes"}:
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

    ax.set_title("Cus50 Validation Objective: DAPG vs FRRO Lambda Sweep")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg objective distance (km)")
    ax.set_xlim(0, 1000)
    ax.set_ylim(150, 280)
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
    legacy_png_path = fig_dir / f"{LEGACY_OUTPUT_STEM}.png"
    legacy_pdf_path = fig_dir / f"{LEGACY_OUTPUT_STEM}.pdf"
    fig.savefig(legacy_png_path, bbox_inches="tight")
    fig.savefig(legacy_pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {png_path}")
    print(f"saved {pdf_path}")
    print(f"saved {legacy_png_path}")
    print(f"saved {legacy_pdf_path}")


if __name__ == "__main__":
    main()
