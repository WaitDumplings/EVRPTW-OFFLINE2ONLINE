"""Plot Cus15 DDE ablation validation curves."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_RUNS = [
    ("DDE-0 static", "O2O_CUS15_DDE0_STATIC_SINGLE_CRITIC_R40_U3_NE64_MB4_E500", "#4c78a8"),
    ("DDE-1 action bias", "O2O_CUS15_DDE1_ACTION_BIAS_R40_U3_NE64_MB4_E500", "#59a14f"),
    ("DDE-2 action key + bias", "O2O_CUS15_DDE2_ACTION_KEY_BIAS_R40_U3_NE64_MB4_E500", "#f28e2b"),
    ("DDE-3 full residual", "O2O_CUS15_DDE3_FULL_RESIDUAL_R40_U3_NE64_MB4_E500", "#e15759"),
]


def _to_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _read_eval(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = _to_float(row.get("epoch"))
            obj = _to_float(row.get("eval_avg_objective_distance_km"))
            if math.isnan(epoch) or math.isnan(obj):
                continue
            rows.append(
                {
                    "epoch": int(epoch),
                    "objective": obj,
                    "vehicle_count": _to_float(row.get("eval_avg_vehicle_count")),
                    "feasible_rate": _to_float(row.get("eval_feasible_rate")),
                    "entropy": _to_float(row.get("entropy")),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2005)
    parser.add_argument("--log-root", type=Path, default=Path("results/logs/Cus_15_CS_3"))
    parser.add_argument("--output-stem", default="cus15_dde_ablation_seed2005")
    args = parser.parse_args()

    plot_dir = Path("results/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    loaded = []
    summary = []
    for label, run_name, color in DEFAULT_RUNS:
        rows = _read_eval(args.log_root / run_name / f"seed_{args.seed}" / "eval_log.csv")
        if not rows:
            continue
        best = min(rows, key=lambda r: r["objective"])
        last = rows[-1]
        summary.append(
            {
                "label": label,
                "run_name": run_name,
                "last_epoch": last["epoch"],
                "last_val_obj": last["objective"],
                "best_epoch": best["epoch"],
                "best_val_obj": best["objective"],
                "last_vehicle_count": last["vehicle_count"],
                "last_feasible_rate": last["feasible_rate"],
                "num_eval_points": len(rows),
            }
        )
        loaded.append((label, color, rows))

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 10.0), sharex=True)
    metrics = [
        ("objective", "Avg objective distance (km)"),
        ("vehicle_count", "Avg vehicle count"),
        ("feasible_rate", "Feasible rate"),
    ]
    for ax, (metric, ylabel) in zip(axes, metrics):
        for label, color, rows in loaded:
            xs = [r["epoch"] for r in rows if not math.isnan(r[metric])]
            ys = [r[metric] for r in rows if not math.isnan(r[metric])]
            if xs:
                ax.plot(xs, ys, marker="o", markersize=2.8, linewidth=1.6, label=label, color=color)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0].set_title(f"Cus15 DDE ablation validation curves (seed {args.seed})")
    axes[-1].set_xlabel("Epoch")
    axes[0].legend(frameon=False, ncol=2)
    fig.tight_layout()

    svg = plot_dir / f"{args.output_stem}.svg"
    png = plot_dir / f"{args.output_stem}.png"
    fig.savefig(svg)
    fig.savefig(png, dpi=180)
    plt.close(fig)

    summary_path = plot_dir / f"{args.output_stem}_summary.csv"
    fields = [
        "label",
        "run_name",
        "last_epoch",
        "last_val_obj",
        "best_epoch",
        "best_val_obj",
        "last_vehicle_count",
        "last_feasible_rate",
        "num_eval_points",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    print(svg)
    print(png)
    print(summary_path)
    for row in summary:
        print(
            f"{row['label']}: last e{row['last_epoch']}={row['last_val_obj']:.3f}, "
            f"best e{row['best_epoch']}={row['best_val_obj']:.3f}"
        )


if __name__ == "__main__":
    main()
