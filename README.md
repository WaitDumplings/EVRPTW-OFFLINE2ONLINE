# EVRPTW-OFFLINE2ONLINE

Clean training workspace for the Cus15/Cus50 EVRP-TW offline-to-online experiments.

The active branch now keeps only:

- **True-road-metric PPO**: the current model/backbone with DDE and real road distance/time/energy features.
- **BC**: behavior-cloning baseline.
- **AWBC**: advantage-weighted behavior-cloning baseline.
- **DAPG**: demonstration-augmented policy-gradient baseline.

Older exploratory branches, diagnostic-only losses, and plotting/watch scripts have been removed from the maintained entry points.

## Local Dependency

The code expects `EVRPTW-DB` to be available locally. By default it reads:

```bash
/data/Maojie/EVRPTW-DB
```

Override it with:

```bash
export EVRPTW_DB_ROOT=/path/to/EVRPTW-DB
```

The canonical Cus15/Cus50 configs use the fixed dataset and Gurobi summaries under that database tree.

## Maintained Configs

Configs live in `configs/`:

- `cus15_true_metric.yaml`: Cus15 true-road-metric PPO run.
- `cus15_bc.yaml`: Cus15 BC baseline.
- `cus15_awbc.yaml`: Cus15 AWBC baseline.
- `cus15_dapg.yaml`: Cus15 DAPG baseline.
- `cus50_true_metric.yaml`: Cus50 true-road-metric PPO run.
- `cus50_bc.yaml`: Cus50 BC baseline.
- `cus50_awbc.yaml`: Cus50 AWBC baseline.
- `cus50_dapg.yaml`: Cus50 DAPG baseline.

All configs use the same model family. Cus15 uses `num_customers=15` and `num_charging_stations=3`; Cus50 uses `num_customers=50` and `num_charging_stations=10`. The baseline configs differ only in `offline.method` and the corresponding BC/AWBC/DAPG coefficients.

## Launch Scripts

Scripts live in `scripts/`:

```bash
bash scripts/run_cus50_true_metric.sh
bash scripts/run_cus50_bc.sh
bash scripts/run_cus50_awbc.sh
bash scripts/run_cus50_dapg.sh
bash scripts/run_cus15_true_metric.sh
bash scripts/run_cus15_bc.sh
bash scripts/run_cus15_awbc.sh
bash scripts/run_cus15_dapg.sh
```

Useful overrides:

```bash
CUDA_DEVICE=1 SEED=3009 bash scripts/run_cus50_dapg.sh --epochs 500
CUDA_DEVICE=0 SEED=3009 bash scripts/run_cus15_true_metric.sh --epochs 1500
```

Logs are written to `results/launch_logs/`; checkpoints and per-epoch metrics are written under `results/checkpoints/` using each config's `run_name`.

## Training Interface

Direct launcher:

```bash
python -m offline2online.train --config configs/cus50_true_metric.yaml --seed 3009 --device cuda
```

The EVRPTW observation must include `edge_distance`, `edge_time`, and `edge_energy`. Missing true road-network matrices now raise an error instead of silently falling back to Euclidean coordinate distance.

Supported `offline.method` values in this cleaned branch are only:

```text
ppo, bc_ppo, awbc, awbc_ppo, dapg
```

Old method names intentionally raise an error so stale YAML files cannot silently run deprecated code paths.

## Metric Logging

The main training CSV now records only the fields needed for current experiments:

- PPO losses, entropy, KL, clip fraction.
- BC/AWBC/DAPG baseline metrics.
- train/eval objective, feasibility, vehicle count, gap summaries.
- rollout/update/eval timing.

Deprecated experiment-specific fields are no longer added to new logs.
