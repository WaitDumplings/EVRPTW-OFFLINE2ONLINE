"""Plot Cus15 DDE supplement and robustness experiments."""

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
        "method": "Static",
        "label": "Static s2005",
        "run_name": "O2O_CUS15_DDE0_STATIC_SINGLE_CRITIC_R40_U3_NE64_MB4_E500",
        "seed": 2005,
        "color": "#4c78a8",
        "style": "-",
    },
    {
        "method": "Static",
        "label": "Static s2006",
        "run_name": "O2O_CUS15_SUPP_DDE0_STATIC_R40_U3_NE64_MB4_E1000",
        "seed": 2006,
        "color": "#4c78a8",
        "style": "--",
    },
    {
        "method": "Static",
        "label": "Static s3000",
        "run_name": "O2O_CUS15_SUPP_DDE0_STATIC_R40_U3_NE64_MB4_E1000",
        "seed": 3000,
        "color": "#4c78a8",
        "style": ":",
    },
    {
        "method": "Bias-only",
        "label": "Bias-only s2005",
        "run_name": "O2O_CUS15_DDE1_ACTION_BIAS_R40_U3_NE64_MB4_E500",
        "seed": 2005,
        "color": "#59a14f",
        "style": "-",
    },
    {
        "method": "Bias-only",
        "label": "Bias-only s2006",
        "run_name": "O2O_CUS15_SUPP_DDE1_ACTION_BIAS_R40_U3_NE64_MB4_E1000",
        "seed": 2006,
        "color": "#59a14f",
        "style": "--",
    },
    {
        "method": "ActionKey-only",
        "label": "ActionKey-only s2005",
        "run_name": "O2O_CUS15_SUPP_DDE2A_ACTION_KEY_ONLY_R40_U3_NE64_MB4_E1000",
        "seed": 2005,
        "color": "#b07aa1",
        "style": "-",
    },
    {
        "method": "ActionKey-only",
        "label": "ActionKey-only s2006",
        "run_name": "O2O_CUS15_SUPP_DDE2A_ACTION_KEY_ONLY_R40_U3_NE64_MB4_E1000",
        "seed": 2006,
        "color": "#b07aa1",
        "style": "--",
    },
    {
        "method": "ActionKey-only",
        "label": "ActionKey-only s3000",
        "run_name": "O2O_CUS15_SUPP_DDE2A_ACTION_KEY_ONLY_R40_U3_NE64_MB4_E1000",
        "seed": 3000,
        "color": "#b07aa1",
        "style": ":",
    },
    {
        "method": "ActionKey+bias",
        "label": "ActionKey+bias s2005",
        "run_name": "O2O_CUS15_DDE2_ACTION_KEY_BIAS_R40_U3_NE64_MB4_E500",
        "seed": 2005,
        "color": "#f28e2b",
        "style": "-",
    },
    {
        "method": "ActionKey+bias",
        "label": "ActionKey+bias s2006",
        "run_name": "O2O_CUS15_SUPP_DDE2_ACTION_KEY_BIAS_R40_U3_NE64_MB4_E1000",
        "seed": 2006,
        "color": "#f28e2b",
        "style": "--",
    },
    {
        "method": "ActionKey+bias",
        "label": "ActionKey+bias s3000",
        "run_name": "O2O_CUS15_SUPP_DDE2_ACTION_KEY_BIAS_R40_U3_NE64_MB4_E1000",
        "seed": 3000,
        "color": "#f28e2b",
        "style": ":",
    },
    {
        "method": "Full residual",
        "label": "Full residual s2005",
        "run_name": "O2O_CUS15_DDE3_FULL_RESIDUAL_R40_U3_NE64_MB4_E500",
        "seed": 2005,
        "color": "#e15759",
        "style": "-",
    },
]


def _float(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


def _read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_eval(path: Path):
    rows = []
    for row in _read_csv(path):
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
                "runtime": _float(row.get("eval_avg_runtime_s")),
            }
        )
    return rows


def read_train_by_epoch(path: Path):
    out = {}
    for row in _read_csv(path):
        epoch = _float(row.get("epoch"))
        if math.isnan(epoch):
            continue
        out[int(epoch)] = {
            "entropy": _float(row.get("entropy")),
            "epoch_wall_time_s": _float(row.get("epoch_wall_time_s")),
        }
    return out


def checkpoint_paths(seed_dir: Path, best_epoch: int, final_epoch: int):
    best = ""
    latest = ""
    candidates = list(seed_dir.rglob("*.pt")) + list(seed_dir.rglob("*.pth")) + list(seed_dir.rglob("*.ckpt"))
    if not candidates:
        return best, latest
    def has_epoch(path: Path, epoch: int) -> bool:
        return str(epoch) in path.name or f"{epoch:04d}" in path.name or f"{epoch:05d}" in path.name
    best_matches = [p for p in candidates if has_epoch(p, best_epoch)]
    final_matches = [p for p in candidates if has_epoch(p, final_epoch)]
    if best_matches:
        best = str(sorted(best_matches)[-1])
    if final_matches:
        latest = str(sorted(final_matches)[-1])
    else:
        latest = str(max(candidates, key=lambda p: p.stat().st_mtime))
    return best, latest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, default=Path("results/logs/Cus_15_CS_3"))
    parser.add_argument("--output-stem", default="cus15_dde_supplement_e1000")
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
        train_by_epoch = read_train_by_epoch(seed_dir / "train_log.csv")
        for row in eval_rows:
            train_row = train_by_epoch.get(row["epoch"], {})
            row["entropy"] = train_row.get("entropy", math.nan)
            row["epoch_wall_time_s"] = train_row.get("epoch_wall_time_s", math.nan)
        best = min(eval_rows, key=lambda r: r["objective"])
        last = eval_rows[-1]
        best_ckpt, latest_ckpt = checkpoint_paths(seed_dir, best["epoch"], last["epoch"])
        valid_times = [r["epoch_wall_time_s"] for r in eval_rows if not math.isnan(r["epoch_wall_time_s"])]
        epoch_time = sum(valid_times) / len(valid_times) if valid_times else math.nan
        summary.append(
            {
                "method": spec["method"],
                "seed": spec["seed"],
                "final_epoch": last["epoch"],
                "final_val_obj": last["objective"],
                "best_val_obj": best["objective"],
                "best_epoch": best["epoch"],
                "final_vehicle_count": last["vehicle_count"],
                "best_vehicle_count": best["vehicle_count"],
                "feasible_rate": last["feasible_rate"],
                "epoch_time": epoch_time,
                "best_checkpoint_path": best_ckpt,
                "latest_checkpoint_path": latest_ckpt,
            }
        )
        loaded.append((spec, eval_rows))

    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
    metrics = [
        ("objective", "Avg objective distance (km)"),
        ("vehicle_count", "Avg vehicle count"),
        ("feasible_rate", "Feasible rate"),
        ("entropy", "Policy entropy"),
    ]
    for ax, (metric, ylabel) in zip(axes, metrics):
        for spec, rows in loaded:
            xs = [r["epoch"] for r in rows if not math.isnan(r.get(metric, math.nan))]
            ys = [r[metric] for r in rows if not math.isnan(r.get(metric, math.nan))]
            if xs:
                ax.plot(
                    xs,
                    ys,
                    linestyle=spec["style"],
                    marker="o",
                    markersize=2.2,
                    linewidth=1.5,
                    color=spec["color"],
                    label=spec["label"],
                )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0].set_title("Cus15 DDE supplement and robustness")
    axes[-1].set_xlabel("Epoch")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()

    svg = plot_dir / f"{args.output_stem}.svg"
    png = plot_dir / f"{args.output_stem}.png"
    fig.savefig(svg)
    fig.savefig(png, dpi=180)
    plt.close(fig)

    summary_path = plot_dir / f"{args.output_stem}_summary.csv"
    fields = [
        "method",
        "seed",
        "final_epoch",
        "final_val_obj",
        "best_val_obj",
        "best_epoch",
        "final_vehicle_count",
        "best_vehicle_count",
        "feasible_rate",
        "epoch_time",
        "best_checkpoint_path",
        "latest_checkpoint_path",
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
            f"{row['method']} s{row['seed']}: "
            f"last e{row['final_epoch']}={row['final_val_obj']:.3f}, "
            f"best e{row['best_epoch']}={row['best_val_obj']:.3f}"
        )


if __name__ == "__main__":
    main()
