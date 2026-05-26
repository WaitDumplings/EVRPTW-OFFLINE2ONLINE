# EVRPTW-OFFLINE2ONLINE

Official offline-to-online EVRP-TW research code.

This first version isolates the model migration experiment:

- model: Ablation/TERRAN-style graph token + dynamic embedding backbone;
- training/eval/data: local path integration with `EVRPTW-DB`;
- objective: compare `O2O_TERRAN_FULL` against the current `EVRPTW-DB` TERRAN baseline on the same AC-v1 Cus50 eval set.

The initial implementation intentionally does not include group advantage,
reference advantage, SL-PPO, or offline ALNS demonstration buffers.

## Local Dependency

By default the code reads EVRPTW-DB from:

```bash
/data/Maojie/Github2/EVRPTW-DB
```

Override this location with:

```bash
export EVRPTW_DB_ROOT=/path/to/EVRPTW-DB
```

## Smoke Run

```bash
python -m offline2online.train \
  --config cus50_o2o_full.yaml \
  --seed 2005 \
  --epochs 2 \
  --num-envs-per-gpu 2 \
  --n-traj 2 \
  --rollout-steps 4 \
  --num-minibatches 1 \
  --eval-limit 2 \
  --eval-batch-size 2 \
  --eval-n-traj 2 \
  --device cpu
```

## Cus50 Comparison

Use `scripts/train_cus50_compare.sh` to launch `O2O_TERRAN_FULL` and the
`EVRPTW-DB` TERRAN baseline on the same AC-v1 Cus50 data with seeds `2005`
and `2006`.

