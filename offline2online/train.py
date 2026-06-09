from __future__ import annotations

import argparse
from typing import Any

from .trainer import load_config, train_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train migrated Ablation/TERRAN-full model on EVRPTW-D AC_v1.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-envs-per-gpu", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--ppo-step-chunk-size", type=int, default=None)
    parser.add_argument("--n-traj", type=int, default=None)
    parser.add_argument("--num-minibatches", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--service-territory-pool-size", "--mother-board-pool-size", dest="mother_board_pool_size", type=int, default=None)
    parser.add_argument("--territory-pool-path", "--region-pool-path", dest="territory_pool_path", type=str, default=None)
    parser.add_argument("--train-dataset-path", type=str, default=None)
    parser.add_argument("--train-sample-mode", type=str, default=None, choices=["shuffle_cycle", "cycle", "random"])
    parser.add_argument("--territory-pool-shuffle", "--region-pool-shuffle", dest="territory_pool_shuffle", action="store_true")
    parser.add_argument("--no-territory-pool-shuffle", "--no-region-pool-shuffle", dest="no_territory_pool_shuffle", action="store_true")
    parser.add_argument("--territory-pool-replacement-policy", "--region-pool-replacement-policy", dest="territory_pool_replacement_policy", type=str, default=None, choices=["cycle", "generate"])
    parser.add_argument("--async-instance-prefetch", action="store_true")
    parser.add_argument("--no-async-instance-prefetch", action="store_true")
    parser.add_argument("--async-instance-workers", type=int, default=None)
    parser.add_argument("--async-instance-queue-batches", type=int, default=None)
    parser.add_argument("--offline-method", type=str, default=None, choices=["ppo", "bc_ppo", "dapg", "route_bc_ppo", "sl_ppo", "frro"])
    parser.add_argument("--expert-solution-path", type=str, default=None)
    parser.add_argument("--expert-dataset-path", type=str, default=None)
    parser.add_argument("--expert-limit", type=int, default=None)
    parser.add_argument("--max-replay-records", type=int, default=None)
    parser.add_argument("--bc-warmup-epochs", type=int, default=None)
    parser.add_argument("--bc-coef", type=float, default=None)
    parser.add_argument("--bc-decay", type=float, default=None)
    parser.add_argument("--bc-batch-size", type=int, default=None)
    parser.add_argument("--bc-updates-per-epoch", type=int, default=None)
    parser.add_argument("--sl-coef", type=float, default=None)
    parser.add_argument("--route-loss-coef", type=float, default=None)
    parser.add_argument("--route-clip-eps", type=float, default=None)
    parser.add_argument("--route-batch-size", type=int, default=None)
    parser.add_argument("--route-updates-per-epoch", type=int, default=None)
    parser.add_argument("--use-group-advantage", action="store_true")
    parser.add_argument("--no-group-advantage", action="store_true")
    parser.add_argument("--use-reference-advantage", action="store_true")
    parser.add_argument("--no-reference-advantage", action="store_true")
    parser.add_argument("--group-adv-coef", type=float, default=None)
    parser.add_argument("--reference-adv-coef", type=float, default=None)
    parser.add_argument("--reference-adv-rho", type=float, default=None)
    parser.add_argument("--frro-coef", type=float, default=None)
    parser.add_argument("--frro-rho", type=float, default=None)
    parser.add_argument("--frro-clip", type=float, default=None)
    parser.add_argument("--frro-falsification-margin", type=float, default=None)
    parser.add_argument("--frro-falsification-eta", type=float, default=None)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-path", type=str, default=None)
    parser.add_argument("--eval-n-traj", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-num-batches", type=int, default=None)
    parser.add_argument("--eval-info-level", type=str, choices=["light", "full"], default=None)
    parser.add_argument("--eval-save-routes", action="store_true")
    parser.add_argument("--no-eval-save-routes", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-debug", action="store_true")
    parser.add_argument("--debug-log-every", type=int, default=None)
    parser.add_argument("--profile-timing", action="store_true")
    parser.add_argument("--no-profile-timing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    overrides: dict[str, Any] = {"data": {}, "training": {}, "evaluation": {}, "offline": {}, "advantage": {}}
    if args.mother_board_pool_size is not None:
        overrides["data"]["mother_board_pool_size"] = args.mother_board_pool_size
    if args.territory_pool_path is not None:
        overrides["data"]["territory_pool_path"] = args.territory_pool_path
    if args.train_dataset_path is not None:
        overrides["data"]["train_dataset_path"] = args.train_dataset_path
    if args.train_sample_mode is not None:
        overrides["data"]["train_sample_mode"] = args.train_sample_mode
    if args.territory_pool_shuffle:
        overrides["data"]["territory_pool_shuffle"] = True
    if args.no_territory_pool_shuffle:
        overrides["data"]["territory_pool_shuffle"] = False
    if args.territory_pool_replacement_policy is not None:
        overrides["data"]["region_pool_replacement_policy"] = args.territory_pool_replacement_policy
    if args.async_instance_prefetch:
        overrides["data"]["async_instance_prefetch"] = True
    if args.no_async_instance_prefetch:
        overrides["data"]["async_instance_prefetch"] = False
    if args.async_instance_workers is not None:
        overrides["data"]["async_instance_workers"] = args.async_instance_workers
    if args.async_instance_queue_batches is not None:
        overrides["data"]["async_instance_queue_batches"] = args.async_instance_queue_batches
    if args.epochs is not None:
        overrides["training"]["epochs"] = args.epochs
    if args.num_envs_per_gpu is not None:
        overrides["training"]["num_envs_per_gpu"] = args.num_envs_per_gpu
    if args.rollout_steps is not None:
        overrides["training"]["rollout_steps"] = args.rollout_steps
    if args.ppo_step_chunk_size is not None:
        overrides["training"]["ppo_step_chunk_size"] = args.ppo_step_chunk_size
    if args.n_traj is not None:
        overrides["training"]["n_traj"] = args.n_traj
    if args.num_minibatches is not None:
        overrides["training"]["num_minibatches"] = args.num_minibatches
    if args.gradient_accumulation_steps is not None:
        overrides["training"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.mixed_precision:
        overrides["training"]["mixed_precision"] = True
    if args.no_mixed_precision:
        overrides["training"]["mixed_precision"] = False
    if args.debug:
        overrides["training"]["debug"] = True
    if args.no_debug:
        overrides["training"]["debug"] = False
    if args.debug_log_every is not None:
        overrides["training"]["debug_log_every"] = args.debug_log_every
    if args.profile_timing:
        overrides["training"]["profile_timing"] = True
    if args.no_profile_timing:
        overrides["training"]["profile_timing"] = False
    if args.offline_method is not None:
        overrides["offline"]["method"] = args.offline_method
    if args.expert_solution_path is not None:
        overrides["offline"]["expert_solution_path"] = args.expert_solution_path
    if args.expert_dataset_path is not None:
        overrides["offline"]["expert_dataset_path"] = args.expert_dataset_path
    if args.expert_limit is not None:
        overrides["offline"]["expert_limit"] = args.expert_limit
    if args.max_replay_records is not None:
        overrides["offline"]["max_replay_records"] = args.max_replay_records
    if args.bc_warmup_epochs is not None:
        overrides["offline"]["bc_warmup_epochs"] = args.bc_warmup_epochs
    if args.bc_coef is not None:
        overrides["offline"]["bc_coef"] = args.bc_coef
    if args.bc_decay is not None:
        overrides["offline"]["bc_decay"] = args.bc_decay
    if args.bc_batch_size is not None:
        overrides["offline"]["bc_batch_size"] = args.bc_batch_size
    if args.bc_updates_per_epoch is not None:
        overrides["offline"]["bc_updates_per_epoch"] = args.bc_updates_per_epoch
    if args.sl_coef is not None:
        overrides["offline"]["sl_coef"] = args.sl_coef
    if args.route_loss_coef is not None:
        overrides["offline"]["route_loss_coef"] = args.route_loss_coef
    if args.route_clip_eps is not None:
        overrides["offline"]["route_clip_eps"] = args.route_clip_eps
    if args.route_batch_size is not None:
        overrides["offline"]["route_batch_size"] = args.route_batch_size
    if args.route_updates_per_epoch is not None:
        overrides["offline"]["route_updates_per_epoch"] = args.route_updates_per_epoch
    if args.use_group_advantage:
        overrides["advantage"]["use_group_advantage"] = True
    if args.no_group_advantage:
        overrides["advantage"]["use_group_advantage"] = False
    if args.use_reference_advantage:
        overrides["advantage"]["use_reference_advantage"] = True
    if args.no_reference_advantage:
        overrides["advantage"]["use_reference_advantage"] = False
    if args.group_adv_coef is not None:
        overrides["advantage"]["group_adv_coef"] = args.group_adv_coef
    if args.reference_adv_coef is not None:
        overrides["advantage"]["reference_adv_coef"] = args.reference_adv_coef
    if args.reference_adv_rho is not None:
        overrides["advantage"]["reference_adv_rho"] = args.reference_adv_rho
    if args.frro_coef is not None:
        overrides["advantage"]["frro_coef"] = args.frro_coef
    if args.frro_rho is not None:
        overrides["advantage"]["frro_rho"] = args.frro_rho
    if args.frro_clip is not None:
        overrides["advantage"]["frro_clip"] = args.frro_clip
    if args.frro_falsification_margin is not None:
        overrides["advantage"]["frro_falsification_margin"] = args.frro_falsification_margin
    if args.frro_falsification_eta is not None:
        overrides["advantage"]["frro_falsification_eta"] = args.frro_falsification_eta
    if args.eval_interval is not None:
        overrides["evaluation"]["eval_interval"] = args.eval_interval
    if args.eval_path is not None:
        overrides["evaluation"]["eval_path"] = args.eval_path
    if args.eval_n_traj is not None:
        overrides["evaluation"]["eval_n_traj"] = args.eval_n_traj
    if args.eval_limit is not None:
        overrides["evaluation"]["eval_limit"] = args.eval_limit
    if args.eval_batch_size is not None:
        overrides["evaluation"]["eval_batch_size"] = args.eval_batch_size
    if args.eval_num_batches is not None:
        overrides["evaluation"]["eval_num_batches"] = args.eval_num_batches
    if args.eval_info_level is not None:
        overrides["evaluation"]["eval_info_level"] = args.eval_info_level
    if args.eval_save_routes:
        overrides["evaluation"]["eval_save_routes"] = True
    if args.no_eval_save_routes:
        overrides["evaluation"]["eval_save_routes"] = False
    overrides = {key: value for key, value in overrides.items() if value}
    ckpt = train_from_config(cfg, seed=args.seed, device=args.device, overrides=overrides)
    print(f"Saved final checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
