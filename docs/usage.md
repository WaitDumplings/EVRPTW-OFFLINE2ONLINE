# EVRPTW-OFFLINE2ONLINE Usage Guide

This repository is the offline-to-online training layer for EVRP-TW policies. It assumes the fixed datasets and solver archives live in `EVRPTW-DB`; data and benchmark outputs are not copied into this repository.

## 1. Required Local Data

Set the EVRPTW-DB root if it is not in the default location:

```bash
export EVRPTW_DB_ROOT=/data/Maojie/Github2/EVRPTW-DB
```

The trainer expects fixed dataset splits such as:

```text
EVRPTW_Dataset/dataset_v1/dataset/train/Cus5
EVRPTW_Dataset/dataset_v1/dataset/val/Cus5
EVRPTW_Dataset/dataset_v1/dataset/train/Cus15
EVRPTW_Dataset/dataset_v1/dataset/val/Cus15
```

Offline archives are solver summary CSV files with at least:

```text
instance_id, objective_distance_km, routes_json
```

`route_sequence_json` is also accepted when `routes_json` is absent. If the archive contains multiple rows for the same `instance_id`, the loader keeps the feasible row with the smallest `objective_distance_km`; this supports Gurobi checkpoint traces as well as ALNS/POMO-style archives. Numeric Gurobi status `2` (`OPTIMAL`) and `9` (`TIME_LIMIT`), and textual statuses such as `OPTIMAL`, `TIME_LIMIT`, `RUNNING`, `SUBOPTIMAL`, and `FEASIBLE`, are accepted when a finite objective and route payload are present. The archive may come from Gurobi, ALNS, POMO, or another solver. For BC/DAPG/route-imitation baselines, expert routes are replayed through the RL environment and invalid routes are rejected. For SL-PPO and reference-advantage methods, the archive provides reference objective values; it is not treated as a PPO rollout buffer.

## 2. Main Methods

Main configs live in `configs/`.

| Config | Purpose |
| --- | --- |
| `cus5_o2o_ppo.yaml` | Vanilla PPO baseline with the O2O backbone. |
| `cus5_o2o_sl_ppo.yaml` | Proposed solution-level SL-PPO with reference advantage enabled by default. |
| `cus15_o2o_sl_ppo.yaml` | Cus15 SL-PPO scaffold using a Cus15 solver archive. |
| `cus50_o2o_sl_ppo_group_ref_1000.yaml` | Cus50 SL-PPO comparison config with route-level group + reference advantages. |
| `cus5_o2o_full.yaml`, `cus15_o2o_full.yaml`, `cus50_o2o_full.yaml` | Basic PPO configs for different scales. |

Default model/training choices:

- Dynamic embedding is off by default.
- Graph token remains on by default.
- The trainer and offline route replay force the fast EVRPTW environment.
- PPO uses GAE by default: `training.use_gae: true`, `training.gae_lambda: 0.95`.
- Mixed precision is enabled on CUDA when `training.mixed_precision: true`.

## 3. Ablation Methods

Ablation configs live in `ablation/configs/`.

| Config | Interpretation |
| --- | --- |
| `cus5_o2o_bc_ppo.yaml` | Behavior cloning warmup followed by PPO. |
| `cus5_o2o_dapg.yaml` | DAPG-style BC warmup followed by PPO with a demonstration gradient. |
| `cus50_o2o_dapg_1000.yaml` | Cus50 DAPG comparison config matching the current SL-PPO rollout/eval protocol. |
| `cus5_o2o_route_bc_ppo.yaml` | Route-level supervised imitation baseline; this is intentionally not the proposed SL-PPO. |
| `cus5_o2o_ppo_group_adv.yaml` | PPO with step-level group advantage augmentation. |
| `cus5_o2o_ppo_ref_adv.yaml` | PPO with step-level reference advantage augmentation. |
| `cus5_o2o_sl_ppo_group_adv.yaml` | True SL-PPO with route-level group advantage. |
| `cus5_o2o_sl_ppo_ref_adv.yaml` | True SL-PPO with route-level reference advantage. |

The proposed `sl_ppo` uses current-policy sampled trajectories and their stored old log probabilities:

```text
r_route = exp(mean_t(log pi_new(a_t|s_t) - log pi_old(a_t|s_t)))
L_SL = -E[min(r_route * A_route, clip(r_route) * A_route)]
```

Solver routes are not inserted into the PPO ratio. They only supply reference objective values, and only the ablation imitation baselines directly supervise solver actions.

DAPG uses two stages:

1. Behavior cloning warmup for `offline.bc_warmup_epochs` epochs, with `offline.bc_updates_per_epoch` supervised updates per epoch and coefficient `offline.bc_warmup_coef`.
2. PPO fine-tuning with an additional demonstration loss. The fine-tuning coefficient is:

```text
coef_demo = lambda0 * lambda1^k * max(A_on_policy)
```

`lambda0` defaults to `offline.bc_coef` unless `offline.dapg_lambda0` is set. `lambda1` defaults to `offline.bc_decay` unless `offline.dapg_lambda1` is set. The counter `k` starts at zero after BC warmup, so the first PPO/DAPG epoch after warmup uses the full `lambda0 * max(A_on_policy)` scale.

## 4. Smoke Tests

Run a minimal CPU check before launching long experiments:

```bash
conda run -n maojie python -m offline2online.train \
  --config cus5_o2o_sl_ppo.yaml \
  --seed 9907 \
  --epochs 1 \
  --device cpu \
  --num-envs-per-gpu 2 \
  --n-traj 2 \
  --rollout-steps 4 \
  --num-minibatches 1 \
  --eval-limit 2 \
  --eval-batch-size 2 \
  --eval-n-traj 2 \
  --expert-limit 4 \
  --max-replay-records 4 \
  --no-mixed-precision \
  --debug \
  --debug-log-every 1
```

Expected behavior:

- The offline archive loads and expert routes replay successfully.
- `use_dynamic_embedding=False` appears in the debug log.
- `mixed_precision=False` on CPU.
- Train, validation, logging, and checkpoint writing complete.

## 5. Full Experiment Pattern

For a main PPO baseline:

```bash
conda run -n maojie python -m offline2online.train \
  --config cus5_o2o_ppo.yaml \
  --seed 2005 \
  --epochs 1000
```

For proposed SL-PPO:

```bash
conda run -n maojie python -m offline2online.train \
  --config cus5_o2o_sl_ppo.yaml \
  --seed 2005 \
  --epochs 1000
```

For ablation baselines:

```bash
conda run -n maojie python -m offline2online.train \
  --config ablation/configs/cus5_o2o_dapg.yaml \
  --seed 2005 \
  --epochs 1000
```

Use the same seeds, dataset split, optimizer parameters, evaluation interval, and eval `n_traj` when comparing methods.

## 6. Cus50 SL-PPO vs DAPG Protocol

The current Cus50 comparison keeps all non-method variables aligned:

- Seed: `2005`.
- Customers/charging stations: `Cus50`, `10` charging stations.
- Rollout/eval horizon: `70` steps.
- PPO updates: `training.ppo_update_epochs: 2`.
- Environments: `training.num_envs_per_gpu: 640`.
- Minibatches: `training.num_minibatches: 16`.
- Gradient accumulation: `training.gradient_accumulation_steps: 2`.
- Model: graph token and attention bias enabled; dynamic embedding disabled.
- Validation: full Cus50 validation split, `eval_n_traj: 50`, `eval_batch_size: 250`.
- Offline source: Gurobi train summary at `/data/Maojie/gurobi_mul/results/train/Cus50/gurobi_summary.csv`.

Launch the two methods on separate GPUs:

```bash
# GPU0 by default
bash scripts/run_cus50_slppo_group_ref_1000.sh

# GPU1 by default
bash scripts/run_cus50_dapg_1000.sh
```

Override the seed or device when needed:

```bash
SEED=2005 CUDA_DEVICE=0 bash scripts/run_cus50_slppo_group_ref_1000.sh
SEED=2005 CUDA_DEVICE=1 bash scripts/run_cus50_dapg_1000.sh
```

Update the comparison figure while either training session is still active:

```bash
bash scripts/watch_cus50_slppo_dapg_plot.sh
```

The plot script reads both `eval_log.csv` files, overlays the Gurobi validation best average from `/data/Maojie/gurobi_mul/results/val/Cus50/gurobi_summary.csv`, and writes:

```text
results/figures/cus50_slppo_group_ref_vs_dapg_r70_u2_e1000.png
results/figures/cus50_slppo_group_ref_vs_dapg_r70_u2_e1000.pdf
results/cus50_slppo_group_ref_vs_dapg_r70_u2_e1000_summary.csv
```

## 7. Outputs

Outputs are written under `results/`, which is ignored by git:

```text
results/logs/Cus_<N>_CS_<M>/<run_name>/seed_<seed>/
results/checkpoints/Cus_<N>_CS_<M>/<run_name>/seed_<seed>/
```

Important files:

- `train_log.csv`: per-epoch training metrics, timing, advantage statistics, offline loss statistics.
- `eval_log.csv`: fixed validation metrics.
- `debug_log.txt`: readable progress and timing trace when debug is enabled.
- `checkpoint_best.pt`: best validation checkpoint.
- `checkpoint_final.pt`: final checkpoint.

## 8. Reproducibility Notes

- Training uses the train split. Validation uses the val split. Eval/test should be run separately and should not be used for early stopping or method selection.
- Offline archives used for training must correspond to the training split unless the experiment is explicitly studying transfer or leakage.
- The legacy config key `mother_board_pool_size` is retained for EVRPTW-DB compatibility; in paper text, refer to this object as the service territory pool.
- If a solver archive contains suboptimal solutions, report it as a suboptimal offline archive. This is valid for the offline-to-online setting and should not be described as global-optimal supervision.
