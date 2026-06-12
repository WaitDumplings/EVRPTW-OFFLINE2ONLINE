# EVRPTW-OFFLINE2ONLINE

Offline-to-online PPO research code for EVRP-TW. The repository treats `EVRPTW-DB` as a local dependency and reads fixed training/validation/evaluation datasets from it instead of copying data into this repository.

## Research Scope

The main question is:

> Given a fixed routing dataset and slow solver-generated offline solution archives, how can an on-policy PPO routing policy use solution-level offline information without treating solver routes as on-policy PPO trajectories?

The main method is **SL-PPO**, a solution-level PPO auxiliary objective. Direct imitation methods such as BC, DAPG, and route-level BC are kept as ablation baselines under `ablation/`.

## Local Dependency

By default the code reads EVRPTW-DB from:

```bash
/data/Maojie/EVRPTW-DB
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

Gurobi archives may contain incumbent solutions from interrupted runs. Numeric status `2` (`OPTIMAL`) and `9` (`TIME_LIMIT`), plus textual statuses such as `OPTIMAL`, `TIME_LIMIT`, `RUNNING`, `SUBOPTIMAL`, and `FEASIBLE`, are treated as usable when they include a finite objective and route payload.

## Main Configs

Main configs live in `configs/`:

- `cus5_o2o_full.yaml`: basic O2O backbone PPO baseline.
- `cus5_o2o_ppo.yaml`: explicit PPO baseline.
- `cus5_o2o_sl_ppo.yaml`: proposed solution-level SL-PPO, default reference advantage.
- `cus15_o2o_sl_ppo.yaml`: Cus15 SL-PPO scaffold using a Cus15 solver archive.
- `cus15_o2o_sl_ppo_group_ref_u4_1000.yaml`: Cus15 SL-PPO group + reference comparison run.
- `cus50_o2o_sl_ppo_group_ref_1000.yaml`: Cus50 SL-PPO comparison run with group + reference route advantages.
- `cus50_o2o_sl_ppo_group_ref_u4_1000.yaml`: Cus50 SL-PPO group + reference run with four PPO updates.
- `cus15_o2o_frro_u4_dyn_mem_dde_1000.yaml`: Cus15 FRRO run with dynamic embedding, memory falsification, and DDE-KV.
- `cus50_o2o_frro_u4_dyn_mem_dde_1000.yaml`: Cus50 FRRO run with dynamic embedding, memory falsification, and DDE-KV.

Dynamic embedding is disabled by default. Graph token remains enabled. The trainer forces `env_fast` and supports AMP through `training.mixed_precision: true`.

## Backbone Freeze Protocol

Before adding DDE or offline objectives, freeze the backbone with static decoder semantics and the distance-decomposed critic. Use the semantic checker first:

```bash
python -m offline2online.backbone_semantics --config configs/cus50_backbone_b2_decomp_total.yaml \
  --seed 2005 --device cpu --num-envs 2 --n-traj 4 --rollout-steps 8
```

Backbone configs for the B0-B4 sequence are:

- `cus50_backbone_b0_legacy_static_single_critic.yaml`: legacy-compatible static single-critic PPO path.
- `cus50_backbone_b1_static_qkv_single_critic.yaml`: static Q/K/V/ActionKey decoder with single critic.
- `cus50_backbone_b2_decomp_total.yaml`: three-head critic, actor advantage from normalized total GAE.
- `cus50_backbone_b3_decomp_exact.yaml`: three-head critic, actor advantage from exact boundary + internal GAE sum.
- `cus50_backbone_b4_decomp_balanced_0505.yaml`: separately normalized boundary/internal advantages, 0.5/0.5 weights.
- `cus50_backbone_b4_decomp_balanced_0307.yaml`: separately normalized boundary/internal advantages, 0.3/0.7 weights.

The decomposed reward semantics are: depot-related edges are `boundary`, all non-depot edges including charging-station edges are `internal`, and `reward_total = reward_boundary + reward_internal` at every transition.

The optional Dynamic Decision Encoder (`model.use_dynamic_decision_encoder: true`) is implemented as a key/value-only decoder adapter. Its input schema is split into a routing-generic core and an EVRPTW constraint supplement:

- Routing core: feasible frontier state, depot/customer/auxiliary node type, route membership, visit/order memory, current/previous candidate flags, current-to-candidate cost, return-to-depot cost, and depot-detour cost.
- EVRPTW supplement: load, battery, time-window, waiting, service-finish, and charging-station repeat margins.

This keeps the DDE reusable for other routing variants while allowing problem-specific constraints to be added as supplements.

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

## FRRO Semantics

`offline.method: frro` is a solution-level PPO method that treats each solver route as a falsifiable reference candidate, not as an action expert. Policy-sampled routes use a route-level clipped PPO ratio with a remaining-gap improvement advantage:

```text
J_base = mean successful objective of current policy rollouts
S = max(std_policy, kappa * [J_base - J_ref]_+, gap_floor_ratio * J_ref)
A_imp(tau) = clip((J_base - J(tau)) / S, -frro_clip, frro_clip)
```

The solver route is added as a bounded auxiliary reference candidate with the same clipped route-ratio form, but it is not described as an on-policy PPO sample. Its advantage is multiplied by a quality gate and a current/history falsification gate, then by `frro_expert_candidate_weight`. Once current sampled routes or historical policy memory beat the reference by `frro_falsification_margin`, the solver candidate is disabled for that instance.

In the checked-in FRRO configs, `offline.sl_coef` is the route objective weight `alpha_R` and `advantage.frro_coef` is kept at `1.0` so the improvement advantage is not double-scaled. Use `FRRO_ALPHA` to change `alpha_R` and `FRRO_EXPERT_WEIGHT` to sweep the expert candidate weight.

## Ablation Baselines

Ablation-only configs and scripts live in `ablation/`:

- `ablation/configs/cus5_o2o_bc_ppo.yaml`: BC warmup + PPO baseline.
- `ablation/configs/cus5_o2o_dapg.yaml`: DAPG-style demonstration gradient baseline.
- `ablation/configs/cus50_o2o_dapg_1000.yaml`: Cus50 DAPG comparison run with BC pretraining and DAPG fine-tuning.
- `ablation/configs/cus15_o2o_dapg_u4_1000.yaml`: Cus15 DAPG run with four PPO updates.
- `ablation/configs/cus15_o2o_dapg_u4_dyn_1000.yaml`: Cus15 DAPG run with candidate dynamic embedding enabled.
- `ablation/configs/cus50_o2o_dapg_u4_1000.yaml`: Cus50 DAPG run with four PPO updates.
- `ablation/configs/cus50_o2o_dapg_u4_dyn_1000.yaml`: Cus50 DAPG run with candidate dynamic embedding enabled.
- `ablation/configs/cus5_o2o_route_bc_ppo.yaml`: route-level imitation baseline, the old implementation previously called SL-PPO.
- `ablation/configs/cus5_o2o_ppo_group_adv.yaml`: PPO with step-level group advantage.
- `ablation/configs/cus5_o2o_ppo_ref_adv.yaml`: PPO with step-level reference advantage.
- `ablation/configs/cus5_o2o_sl_ppo_group_adv.yaml`: true SL-PPO with group-only route advantage.
- `ablation/configs/cus5_o2o_sl_ppo_ref_adv.yaml`: true SL-PPO with reference-only route advantage.

DAPG follows the original two-stage structure: optional behavior cloning warmup first, then PPO with an additional demonstration-gradient loss. During fine-tuning the demo loss coefficient is:

```text
coef_demo = lambda0 * lambda1^k * max(A_on_policy)
```

where `lambda0` defaults to `offline.bc_coef`, `lambda1` defaults to `offline.bc_decay`, and `k` starts at zero after the BC warmup phase.

Example:

```bash
python -m offline2online.train --config ablation/configs/cus5_o2o_dapg.yaml --seed 2005 --epochs 1000
```

## Portable Launch Scripts

The launch scripts derive `O2O_ROOT` from the checked-out repository and use the current `python` by default. Override paths or devices without editing the scripts:

```bash
export EVRPTW_DB_ROOT=/path/to/EVRPTW-DB
PYTHON_BIN=/path/to/python SEED=2005 CUDA_DEVICE=0 bash scripts/run_cus15_slppo_group_ref_u4_1000.sh
```

Additional trainer overrides can be appended to any launch script, for example:

```bash
bash scripts/run_cus15_dapg_u4_dyn_1000.sh --num-envs-per-gpu 320 --eval-limit 100
```

## Cus15 Comparison Helpers

The Cus15 bundle is intended for reproducing the current update/dynamic comparison on another server. It uses seed `2005`, rollout/eval horizon `40`, four PPO updates, graph token and attention bias enabled, and PBRS disabled.

```bash
bash scripts/run_cus15_slppo_group_ref_u4_1000.sh
bash scripts/run_cus15_frro_u4_dyn_mem_dde_1000.sh
bash scripts/run_cus15_dapg_u4_1000.sh
bash scripts/run_cus15_dapg_u4_dyn_1000.sh
bash scripts/watch_cus15_slppo_dapg_plot.sh
```

For the FRRO run, the launch script defaults to `FRRO_ALPHA=0.10` and `FRRO_EXPERT_WEIGHT=2.0`, producing run name `O2O_CUS15_FRRO_A010_LE2_DYN_MEM_DDE_KV_R40_U4_E1000`. Override these environment variables to run lambda-E sweeps, for example:

```bash
CUDA_DEVICE=1 FRRO_EXPERT_WEIGHT=4.0 FRRO_TAG=A010_LE4 bash scripts/run_cus15_frro_u4_dyn_mem_dde_1000.sh
```

The Cus15 plot helper writes:

```text
results/figures/cus15_offline_dynamic_update_ablation_r40_e1000.png
results/figures/cus15_offline_dynamic_update_ablation_r40_e1000.pdf
results/cus15_offline_dynamic_update_ablation_r40_e1000_summary.csv
```

If the validation Gurobi summary is not at `EVRPTW_DB_ROOT/results/offline_experts/dataset_v1/val/Cus15_Gurobi_2h/gurobi_summary.csv`, set:

```bash
export CUS15_GUROBI_VAL_SUMMARY=/path/to/Cus15/val/gurobi_summary.csv
```

## Cus50 Comparison Helpers

The current Cus50 comparison uses the same seed, rollout length, graph token, and attention bias for SL-PPO and DAPG. The checked-in helpers include u2, u4, and a DAPG u4 dynamic-embedding variant.

```bash
bash scripts/run_cus50_slppo_group_ref_1000.sh
bash scripts/run_cus50_slppo_group_ref_u4_1000.sh
bash scripts/run_cus50_frro_u4_dyn_mem_dde_1000.sh
bash scripts/run_cus50_dapg_1000.sh
bash scripts/run_cus50_dapg_u4_1000.sh
bash scripts/run_cus50_dapg_u4_dyn_1000.sh
bash scripts/watch_cus50_slppo_dapg_plot.sh
```

The current Cus50 plot helper overlays DAPG, FRRO lambda-E sweeps, and the validation-set Gurobi best average from `/data/Maojie/gurobi_mul/results/val/Cus50/gurobi_summary.csv`. Override that path with `CUS50_GUROBI_VAL_SUMMARY`.

## Outputs

Training writes logs and checkpoints to:

```text
results/logs/Cus_<N>_CS_<M>/<run_name>/seed_<seed>/
results/checkpoints/Cus_<N>_CS_<M>/<run_name>/seed_<seed>/
```

`checkpoint_best.pt` is selected by validation objective. `results/` is ignored by git.

For a fuller experiment protocol, method map, and smoke-test commands, see
[`docs/usage.md`](docs/usage.md).
