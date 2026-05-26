from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .integrations.evrptw_db import DEFAULT_EVRPTW_DB_ROOT


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_eval_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _summarize_solver(solver_name: str, seed: int, eval_log: Path) -> dict[str, Any]:
    rows = _read_eval_rows(eval_log)
    valid_rows = [row for row in rows if row.get("eval_status") == "ok"]
    best_obj = None
    best_epoch = None
    for row in valid_rows:
        obj = _to_float(row.get("eval_avg_objective_distance_km"))
        if obj is None:
            continue
        if best_obj is None or obj < best_obj:
            best_obj = obj
            best_epoch = int(row["epoch"])
    final = valid_rows[-1] if valid_rows else {}
    return {
        "solver_name": solver_name,
        "seed": seed,
        "eval_log": str(eval_log),
        "num_eval_rows": len(valid_rows),
        "best_epoch": best_epoch if best_epoch is not None else "",
        "best_eval_objective_distance_km": best_obj if best_obj is not None else "",
        "final_epoch": final.get("epoch", ""),
        "final_eval_objective_distance_km": final.get("eval_avg_objective_distance_km", ""),
        "final_eval_feasible_rate": final.get("eval_feasible_rate", ""),
        "final_eval_vehicle_count": final.get("eval_avg_vehicle_count", ""),
    }


def main() -> None:
    rows = []
    for seed in (2005, 2006):
        rows.append(
            _summarize_solver(
                "O2O_TERRAN_FULL",
                seed,
                REPO_ROOT
                / "results"
                / "logs"
                / "Cus_50_CS_10"
                / "O2O_TERRAN_FULL"
                / f"seed_{seed}"
                / "eval_log.csv",
            )
        )
        rows.append(
            _summarize_solver(
                "TERRAN",
                seed,
                DEFAULT_EVRPTW_DB_ROOT
                / "EVRPTW_Benchmark"
                / "Reinforcement_Learning"
                / "TERRAN"
                / "logs"
                / "Cus_50_CS_10"
                / "TERRAN"
                / f"seed_{seed}"
                / "eval_log.csv",
            )
        )
    out = REPO_ROOT / "results" / "cus50_model_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()

