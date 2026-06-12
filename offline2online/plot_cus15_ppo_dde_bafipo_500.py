from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = REPO_ROOT / "results" / "logs" / "Cus_15_CS_3"
DEFAULT_OUTPUT_STEM = "cus15_ppo_dde_bafipo_e500_seed2005_val_obj"
DEFAULT_GUROBI_VAL = Path("/data/Maojie/Github2/Cus15/val/gurobi_summary.csv")

RUNS = [
    ("Vanilla PPO", "O2O_CUS15_PPO_VANILLA_NO_DYN_NO_DDE_R40_U3_NE480_EB1000_EN8_E500", 2005, "#4c78a8"),
    ("PPO + DDE", "O2O_CUS15_PPO_DDE_NO_DYN_R40_U3_NE480_EB1000_EN8_E500", 2005, "#59a14f"),
    ("BA-FIPO coef=0.05", "O2O_CUS15_BAFIPO_P005_MB8_NO_DYN_DDE_R40_U3_NE480_EB1000_EN8_E500", 2005, "#e15759"),
    ("BA-FIPO coef=0.10", "O2O_CUS15_BAFIPO_P010_MB8_NO_DYN_DDE_R40_U3_NE480_EB1000_EN8_E500", 2005, "#b07aa1"),
]


def _read_eval(path: Path) -> tuple[list[int], list[float]]:
    epochs: list[int] = []
    objs: list[float] = []
    if not path.exists():
        return epochs, objs
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                epoch = int(float(row["epoch"]))
                obj = float(row["eval_avg_objective_distance_km"])
            except Exception:
                continue
            if math.isfinite(obj):
                epochs.append(epoch)
                objs.append(obj)
    return epochs, objs


def _gurobi_avg(path: Path) -> tuple[float | None, int]:
    if not path.exists():
        return None, 0
    vals: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                obj = float(row.get("objective_distance_km", "nan"))
            except Exception:
                continue
            if math.isfinite(obj):
                vals.append(obj)
    if not vals:
        return None, 0
    return sum(vals) / len(vals), len(vals)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--gurobi-val", type=Path, default=DEFAULT_GUROBI_VAL)
    args = parser.parse_args()

    out_dir = REPO_ROOT / "results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / args.output_stem
    summary_rows: list[dict[str, object]] = []

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for label, run_name, seed, color in RUNS:
        path = LOG_ROOT / run_name / f"seed_{seed}" / "eval_log.csv"
        epochs, objs = _read_eval(path)
        if epochs:
            ax.plot(epochs, objs, marker="o", markersize=3, linewidth=1.8, label=label, color=color)
            best_idx = min(range(len(objs)), key=objs.__getitem__)
            summary_rows.append({
                "label": label,
                "run_name": run_name,
                "seed": seed,
                "points": len(objs),
                "latest_epoch": epochs[-1],
                "latest_obj": objs[-1],
                "best_epoch": epochs[best_idx],
                "best_obj": objs[best_idx],
                "status": "ok",
            })
        else:
            summary_rows.append({
                "label": label,
                "run_name": run_name,
                "seed": seed,
                "points": 0,
                "latest_epoch": "",
                "latest_obj": "",
                "best_epoch": "",
                "best_obj": "",
                "status": f"missing_or_empty:{path}",
            })

    gurobi, n = _gurobi_avg(args.gurobi_val)
    if gurobi is not None:
        ax.axhline(gurobi, color="black", linestyle="--", linewidth=1.4, label=f"Gurobi 2h avg ({gurobi:.3f})")
        summary_rows.append({
            "label": "Gurobi 2h avg",
            "run_name": str(args.gurobi_val),
            "seed": "",
            "points": n,
            "latest_epoch": "",
            "latest_obj": gurobi,
            "best_epoch": "",
            "best_obj": gurobi,
            "status": "baseline",
        })

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation objective distance (km)")
    ax.set_title("Cus15 Vanilla PPO / DDE / BA-FIPO comparison (seed 2005)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    plt.close(fig)

    summary_path = stem.with_name(stem.name + "_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "run_name", "seed", "points", "latest_epoch", "latest_obj", "best_epoch", "best_obj", "status"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(stem.with_suffix(".svg"))
    print(stem.with_suffix(".png"))
    print(summary_path)


if __name__ == "__main__":
    main()
