"""Plot Cus15 DAPG smoke test with online and solver references."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [
    {
        "label": "BC ActionKey-only s2005",
        "run_name": "O2O_CUS15_OFFLINE_SMOKE_BC_ACTIONKEY_R40_U3_NE64_MB4_E500",
        "seed": 2005,
        "color": "#2ca02c",
        "style": "-",
        "width": 2.0,
    },
    {
        "label": "AWBC ActionKey-only s2005",
        "run_name": "O2O_CUS15_OFFLINE_SMOKE_AWBC_ACTIONKEY_R40_U3_NE64_MB4_E500",
        "seed": 2005,
        "color": "#9467bd",
        "style": "-",
        "width": 2.0,
    },
    {
        "label": "DAPG ActionKey-only s2005",
        "run_name": "O2O_CUS15_OFFLINE_SMOKE_DAPG_ACTIONKEY_R40_U3_NE64_MB4_E500",
        "seed": 2005,
        "color": "#d62728",
        "style": "-",
        "width": 2.2,
    },
    {
        "label": "Gated DAPG ActionKey-only s2005",
        "run_name": "O2O_CUS15_OFFLINE_SMOKE_GADAPG_ACTIONKEY_R40_U3_NE64_MB4_E500",
        "seed": 2005,
        "color": "#ff7f0e",
        "style": "-",
        "width": 2.0,
    },
    {
        "label": "Online Static backbone s2005",
        "run_name": "O2O_CUS15_DDE0_STATIC_SINGLE_CRITIC_R40_U3_NE64_MB4_E500",
        "seed": 2005,
        "color": "#4c78a8",
        "style": "--",
        "width": 1.7,
    },
    {
        "label": "Online DDE ActionKey-only s2005",
        "run_name": "O2O_CUS15_SUPP_DDE2A_ACTION_KEY_ONLY_R40_U3_NE64_MB4_E1000",
        "seed": 2005,
        "color": "#b07aa1",
        "style": "-.",
        "width": 1.7,
    },
]


def _float(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_eval(path: Path):
    rows = []
    for row in read_csv(path):
        epoch = _float(row.get("epoch"))
        obj = _float(row.get("eval_avg_objective_distance_km"))
        if math.isnan(epoch) or math.isnan(obj):
            continue
        rows.append(
            {
                "epoch": int(epoch),
                "objective": obj,
                "vehicle_count": _float(row.get("eval_avg_vehicle_count")),
                "feasible_rate": _float(row.get("eval_feasible_rate")),
            }
        )
    return rows


def read_train(path: Path):
    out = {}
    for row in read_csv(path):
        epoch = _float(row.get("epoch"))
        if math.isnan(epoch):
            continue
        out[int(epoch)] = {
            "entropy": _float(row.get("entropy")),
            "bc_loss": _float(row.get("bc_loss")),
            "bc_coef": _float(row.get("bc_coef")),
            "bc_accuracy": _float(row.get("bc_accuracy")),
            "epoch_wall_time_s": _float(row.get("epoch_wall_time_s")),
        }
    return out


def gurobi_avg(path: Path) -> float:
    vals = []
    for row in read_csv(path):
        feasible = str(row.get("feasible", "")).lower()
        obj = _float(row.get("objective_distance_km"))
        if feasible in {"true", "1", "yes"} and not math.isnan(obj):
            vals.append(obj)
    return sum(vals) / len(vals) if vals else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, default=Path("results/logs/Cus_15_CS_3"))
    parser.add_argument("--gurobi-val", type=Path, default=Path("/data/Maojie/Github2/Cus15/val/gurobi_summary.csv"))
    parser.add_argument("--output-stem", default="cus15_offline_dapg_smoke_canvas")
    args = parser.parse_args()

    plot_dir = Path("results/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    loaded = []
    summary = []
    for spec in RUNS:
        seed_dir = args.log_root / spec["run_name"] / f"seed_{spec['seed']}"
        eval_rows = read_eval(seed_dir / "eval_log.csv")
        if not eval_rows:
            continue
        train_by_epoch = read_train(seed_dir / "train_log.csv")
        for row in eval_rows:
            train_row = train_by_epoch.get(row["epoch"], {})
            row.update(train_row)
        best = min(eval_rows, key=lambda r: r["objective"])
        last = eval_rows[-1]
        summary.append(
            {
                "label": spec["label"],
                "run_name": spec["run_name"],
                "seed": spec["seed"],
                "final_epoch": last["epoch"],
                "final_obj": last["objective"],
                "best_obj": best["objective"],
                "best_epoch": best["epoch"],
                "vehicle_count": last.get("vehicle_count", math.nan),
                "feasible_rate": last.get("feasible_rate", math.nan),
                "entropy": last.get("entropy", math.nan),
                "bc_loss": last.get("bc_loss", math.nan),
                "bc_coef": last.get("bc_coef", math.nan),
                "bc_accuracy": last.get("bc_accuracy", math.nan),
            "awbc_weight_mean": last.get("awbc_weight_mean", math.nan),
            "awbc_active_ratio": last.get("awbc_active_ratio", math.nan),
            }
        )
        loaded.append((spec, eval_rows))

    gavg = gurobi_avg(args.gurobi_val)

    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
    metrics = [
        ("objective", "Avg objective distance (km)"),
        ("vehicle_count", "Avg vehicle count"),
        ("entropy", "Policy entropy"),
        ("bc_loss", "DAPG demo/BC loss"),
    ]
    for ax, (metric, ylabel) in zip(axes, metrics):
        for spec, rows in loaded:
            xs = [r["epoch"] for r in rows if not math.isnan(r.get(metric, math.nan))]
            ys = [r[metric] for r in rows if not math.isnan(r.get(metric, math.nan))]
            if xs:
                ax.plot(
                    xs,
                    ys,
                    color=spec["color"],
                    linestyle=spec["style"],
                    linewidth=spec["width"],
                    marker="o",
                    markersize=2.2,
                    label=spec["label"],
                )
        if metric == "objective" and not math.isnan(gavg):
            ax.axhline(gavg, color="#111111", linestyle=":", linewidth=1.9, label=f"Gurobi 2h avg ({gavg:.3f})")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0].set_title("Cus15 OFFLINE DAPG smoke test with online references")
    axes[-1].set_xlabel("Epoch")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()

    svg = plot_dir / f"{args.output_stem}.svg"
    png = plot_dir / f"{args.output_stem}.png"
    fig.savefig(svg)
    fig.savefig(png, dpi=180)
    plt.close(fig)

    summary_path = plot_dir / f"{args.output_stem}_summary.csv"
    fields = ["label", "run_name", "seed", "final_epoch", "final_obj", "best_obj", "best_epoch", "vehicle_count", "feasible_rate", "entropy", "bc_loss", "bc_coef", "bc_accuracy", "awbc_weight_mean", "awbc_active_ratio"]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    print(svg)
    print(png)
    print(summary_path)
    print(f"gurobi_avg={gavg:.6f}" if not math.isnan(gavg) else "gurobi_avg=nan")
    for row in summary:
        print(f"{row['label']}: last e{row['final_epoch']}={row['final_obj']:.3f}, best e{row['best_epoch']}={row['best_obj']:.3f}")


if __name__ == "__main__":
    main()
