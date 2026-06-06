# EVRPTW-OFFLINE2ONLINE

Offline-to-online PPO research code for EVRP-TW. The repository treats `EVRPTW-DB` as a local dependency and reads fixed training/validation/evaluation datasets from it instead of copying data into this repository.

## Research Scope

The main question is:

> Given a fixed routing dataset and slow solver-generated offline solution archives, how can an on-policy PPO routing policy use solution-level offline information without treating solver routes as on-policy PPO trajectories?

The main method is **SL-PPO**, a solution-level PPO auxiliary objective. Direct imitation methods such as BC, DAPG, and route-level BC are kept as ablation baselines under `ablation/`.

## Local Dependency

By default the code reads EVRPTW-DB from:

```bash
/data/Maojie/Github2/EVRPTW-DB
```

Override this location with:

```bash
export EVRPTW_DB_ROOT=/path/to/EVRPTW-DB
```

Expected dataset layout is the EVRPTW-DB fixed dataset format, for example:

```text
EVRPTW_Dataset/dataset_v1/dataset/train/Cus5
EVRPTW_Dataset/dataset_v1/dataset/val/Cus5
EVRPTW_Dataset/dataset_v1/dataset/train/Cus15
EVRPTW_Dataset/dataset_v1/dataset/val/Cus15
```

Expected offline archive format is a solver summary CSV with `instance_id`, `objective_distance_km`, and `routes_json` columns. The archive can be produced by Gurobi, ALNS, POMO, or another solver. In the current experiments the default paths point to Gurobi-style archives, for example:

```text
results/offline_experts/dataset_v1/train/Cus5_Gurobi/gurobi_summary.csv
results/offline_experts/dataset_v1/train/Cus15_Gurobi_2h/gurobi_summary.csv
```

## Main Configs

Main configs live in `configs/`:

- `cus5_o2o_full.yaml`: basic O2O backbone PPO baseline.
- `cus5_o2o_ppo.yaml`: explicit PPO baseline.
- `cus5_o2o_sl_ppo.yaml`: proposed solution-level SL-PPO, default reference advantage.
- `cus15_o2o_sl_ppo.yaml`: Cus15 SL-PPO scaffold using a Cus15 solver archive.

Dynamic embedding is disabled by default. Graph token remains enabled. The trainer forces `env_fast` and supports AMP through `training.mixed_precision: true`.

## SL-PPO Semantics

The proposed `offline.method: sl_ppo` does not imitate solver actions. It uses current on-policy rollout trajectories and adds a solution-level clipped PPO objective:

```text
r_route = exp(mean_t(logpi_new(a_t|s_t) - logpi_old(a_t|s_t)))
A_route = lambda_g * A_group + lambda_r * gate * A_ref
L_SL = -E[min(r_route * A_route, clip(r_route) * A_route)]
```

The solver archive supplies reference objectives, not PPO replay trajectories. The default SL-PPO config uses reference advantage with a soft gate and can optionally enable group advantage.

Useful overrides:

```bash
python -m offline2online.train --config cus5_o2o_sl_ppo.yaml --seed 2005 --epochs 2 --device cpu \
  --num-envs-per-gpu 2 --n-traj 2 --rollout-steps 4 --num-minibatches 1 \
  --eval-limit 2 --eval-batch-size 2 --eval-n-traj 2 --no-mixed-precision
```

To enable group advantage in SL-PPO:

```bash
python -m offline2online.train --config cus5_o2o_sl_ppo.yaml --seed 2005 --use-group-advantage
```

To run PPO with auxiliary reference advantage:

```bash
python -m offline2online.train --config cus5_o2o_ppo.yaml --seed 2005 --use-reference-advantage \
  --expert-solution-path results/offline_experts/dataset_v1/train/Cus5_Gurobi/gurobi_summary.csv
```

## Ablation Baselines

Ablation-only configs and scripts live in `ablation/`:

- `ablation/configs/cus5_o2o_bc_ppo.yaml`: BC warmup + PPO baseline.
- `ablation/configs/cus5_o2o_dapg.yaml`: DAPG-style demonstration gradient baseline.
- `ablation/configs/cus5_o2o_route_bc_ppo.yaml`: route-level imitation baseline, the old implementation previously called SL-PPO.
- `ablation/configs/cus5_o2o_ppo_group_adv.yaml`: PPO with step-level group advantage.
- `ablation/configs/cus5_o2o_ppo_ref_adv.yaml`: PPO with step-level reference advantage.
- `ablation/configs/cus5_o2o_sl_ppo_group_adv.yaml`: true SL-PPO with group-only route advantage.
- `ablation/configs/cus5_o2o_sl_ppo_ref_adv.yaml`: true SL-PPO with reference-only route advantage.

Example:

```bash
python -m offline2online.train --config ablation/configs/cus5_o2o_dapg.yaml --seed 2005 --epochs 1000
```

## Outputs

Training writes logs and checkpoints to:

```text
results/logs/Cus_<N>_CS_<M>/<run_name>/seed_<seed>/
results/checkpoints/Cus_<N>_CS_<M>/<run_name>/seed_<seed>/
```

`checkpoint_best.pt` is selected by validation objective. `results/` is ignored by git.

For a fuller experiment protocol, method map, and smoke-test commands, see
[`docs/usage.md`](docs/usage.md).
