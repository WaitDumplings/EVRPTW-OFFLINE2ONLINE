from __future__ import annotations

import csv
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import yaml

from .integrations.evrptw_db import configure_evrptw_db

EVRPTW_DB_ROOT = configure_evrptw_db()

from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.rollout import collect_rollout, compute_returns
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.trainer import (
    _debug_log,
    _format_float,
    evaluate_fixed_dataset,
    evaluate_policy_loss,
    make_envs,
    pbrs_scale_for_epoch,
    set_pbrs_reward_scale,
    summarize_train_infos,
)

from .models import Agent


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        local = REPO_ROOT / "configs" / cfg_path
        cfg_path = local if local.exists() else cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sync_cuda(device: str | torch.device) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()


def save_checkpoint(path: Path, agent: Agent, optimizer: torch.optim.Optimizer, cfg: dict[str, Any], epoch: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "seed": int(seed),
            "config": cfg,
            "model_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def train_from_config(
    cfg: dict[str, Any],
    seed: int,
    device: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> Path:
    cfg = deep_update(cfg, overrides or {})
    set_seed(seed)
    train_cfg = cfg["training"]
    eval_cfg = cfg.get("evaluation", {})
    model_cfg = cfg.get("model", {})
    run_name = str(cfg.get("run_name", "O2O_TERRAN_FULL"))
    num_customers = int(cfg["data"].get("num_customers", 50))
    num_cs = int(cfg["data"].get("num_charging_stations", 10))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = Agent(
        embedding_dim=int(model_cfg.get("embedding_dim", 256)),
        tanh_clipping=float(model_cfg.get("tanh_clipping", 15.0)),
        n_encode_layers=int(model_cfg.get("n_encode_layers", 2)),
        device=device,
        use_graph_token=bool(model_cfg.get("use_graph_token", True)),
        use_dynamic_embedding=bool(model_cfg.get("use_dynamic_embedding", True)),
        use_candidate_dynamic_embedding=model_cfg.get("use_candidate_dynamic_embedding"),
    ).to(device)
    optimizer = torch.optim.AdamW(
        agent.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        eps=1e-5,
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    initial_env_start = time.perf_counter()
    envs, pool = make_envs(cfg, seed)
    initial_env_pool_time_s = time.perf_counter() - initial_env_start

    gamma = float(train_cfg.get("gamma", 0.99))
    epochs = int(train_cfg.get("epochs", 500))
    rollout_steps = int(train_cfg.get("rollout_steps", 90))
    ppo_epochs = int(train_cfg.get("ppo_update_epochs", 3))
    num_minibatches = max(1, int(train_cfg.get("num_minibatches", 4)))
    gradient_accumulation_steps = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
    checkpoint_interval = int(train_cfg.get("checkpoint_interval", 50))
    eval_interval = int(eval_cfg.get("eval_interval", 0) or 0)
    debug_enabled = bool(train_cfg.get("debug", False))
    debug_log_every = max(1, int(train_cfg.get("debug_log_every", 1)))
    profile_timing = bool(train_cfg.get("profile_timing", False))
    ppo_step_chunk_size = int(train_cfg.get("ppo_step_chunk_size", 0) or 0)

    out_root = REPO_ROOT / "results"
    ckpt_dir = out_root / "checkpoints" / f"Cus_{num_customers}_CS_{num_cs}" / run_name / f"seed_{seed}"
    log_dir = out_root / "logs" / f"Cus_{num_customers}_CS_{num_cs}" / run_name / f"seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train_log.csv"
    eval_log_path = log_dir / "eval_log.csv"
    debug_log_path = log_dir / "debug_log.txt"

    train_fields = [
        "epoch",
        "reward_mean",
        "policy_loss",
        "value_loss",
        "entropy",
        "samples_seen",
        "num_envs",
        "n_traj",
        "rollout_steps",
        "num_minibatches",
        "gradient_accumulation_steps",
        "effective_instances_per_optimizer_step",
        "pbrs_scale",
        "initial_env_pool_time_s",
        "rollout_reset_time_s",
        "rollout_stack_obs_time_s",
        "rollout_model_action_time_s",
        "rollout_env_step_time_s",
        "rollout_interaction_time_s",
        "rollout_total_time_s",
        "ppo_update_time_s",
        "eval_wall_time_s",
        "epoch_wall_time_s",
        "train_feasible_rate",
        "train_avg_best_objective_distance_km",
        "train_avg_vehicle_count",
        "train_avg_served_customers",
        "eval_avg_objective_distance_km",
        "eval_avg_vehicle_count",
        "eval_feasible_rate",
        "eval_avg_runtime_s",
        "eval_num_instances",
        "eval_n_traj",
        "eval_batch_size",
        "eval_num_batches",
        "eval_decode_mode",
        "eval_info_level",
        "eval_save_routes",
        "eval_status",
    ]
    eval_fields = [
        "epoch",
        "eval_avg_objective_distance_km",
        "eval_avg_vehicle_count",
        "eval_feasible_rate",
        "eval_avg_runtime_s",
        "eval_num_instances",
        "eval_n_traj",
        "eval_batch_size",
        "eval_num_batches",
        "eval_decode_mode",
        "eval_info_level",
        "eval_save_routes",
        "eval_status",
    ]

    with (
        log_path.open("w", newline="", encoding="utf-8") as f,
        eval_log_path.open("w", newline="", encoding="utf-8") as ef,
        debug_log_path.open("w", encoding="utf-8") as df,
    ):
        writer = csv.DictWriter(f, fieldnames=train_fields)
        eval_writer = csv.DictWriter(ef, fieldnames=eval_fields)
        writer.writeheader()
        eval_writer.writeheader()
        _debug_log(
            debug_enabled,
            df,
            f"[Init] run={run_name} seed={seed} device={device} epochs={epochs} "
            f"n_traj={train_cfg.get('n_traj', 50)} rollout_steps={rollout_steps} "
            f"num_envs={train_cfg.get('num_envs_per_gpu', 128)} minibatches={num_minibatches} "
            f"accum_grad={gradient_accumulation_steps} "
            f"n_encode_layers={model_cfg.get('n_encode_layers', 2)} "
            f"use_graph_token={model_cfg.get('use_graph_token', True)} "
            f"use_dynamic_embedding={model_cfg.get('use_dynamic_embedding', True)} "
            f"initial_env_pool_time_s={initial_env_pool_time_s:.3f} "
            f"eval_interval={eval_interval} eval_n_traj={eval_cfg.get('eval_n_traj', 50)} "
            f"eval_batch_size={eval_cfg.get('eval_batch_size', 1000)}",
        )
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            pbrs_scale = pbrs_scale_for_epoch(cfg, epoch, epochs)
            set_pbrs_reward_scale(envs, pbrs_scale)
            agent.train()
            batch = collect_rollout(
                agent,
                envs,
                rollout_steps=rollout_steps,
                decode_mode="sample",
                device=device,
                seed=seed + epoch * 100_000,
                profile_timing=profile_timing,
            )
            returns = compute_returns(batch.rewards, batch.dones, gamma=gamma)
            advantages = returns - batch.values
            adv_vals = advantages[batch.valid]
            if adv_vals.numel() > 1:
                advantages = (advantages - adv_vals.mean()) / (adv_vals.std(unbiased=False) + 1e-8)

            losses = []
            num_envs = int(batch.actions.size(1))
            minibatches = min(num_minibatches, num_envs)
            env_order = np.arange(num_envs, dtype=np.int64)
            if profile_timing:
                _sync_cuda(device)
            ppo_start = time.perf_counter()
            total_steps = int(batch.actions.size(0))
            chunk_size = ppo_step_chunk_size if ppo_step_chunk_size > 0 else total_steps
            chunk_size = max(1, min(chunk_size, total_steps))
            for _ in range(ppo_epochs):
                np.random.shuffle(env_order)
                split_indices = [indices for indices in np.array_split(env_order, minibatches) if indices.size > 0]
                for group_start in range(0, len(split_indices), gradient_accumulation_steps):
                    accum_group = split_indices[group_start : group_start + gradient_accumulation_steps]
                    if not accum_group:
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    group_policy = 0.0
                    group_value = 0.0
                    group_entropy = 0.0
                    group_size = float(len(accum_group))
                    for env_indices in accum_group:
                        weighted_policy = 0.0
                        weighted_value = 0.0
                        weighted_entropy = 0.0
                        for step_start in range(0, total_steps, chunk_size):
                            step_end = min(step_start + chunk_size, total_steps)
                            chunk_weight = float(step_end - step_start) / max(float(total_steps), 1.0)
                            loss, policy_loss, value_loss, entropy = evaluate_policy_loss(
                                agent,
                                batch,
                                returns,
                                advantages.detach(),
                                cfg,
                                device,
                                env_indices=env_indices,
                                step_start=step_start,
                                step_end=step_end,
                            )
                            (loss * chunk_weight / group_size).backward()
                            weighted_policy += policy_loss.item() * chunk_weight
                            weighted_value += value_loss.item() * chunk_weight
                            weighted_entropy += entropy.item() * chunk_weight
                        group_policy += weighted_policy / group_size
                        group_value += weighted_value / group_size
                        group_entropy += weighted_entropy / group_size
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), float(train_cfg.get("max_grad_norm", 1.0)))
                    optimizer.step()
                    losses.append((group_policy, group_value, group_entropy))
            if profile_timing:
                _sync_cuda(device)
            ppo_update_time_s = time.perf_counter() - ppo_start

            reward_mean = float(batch.rewards[batch.valid].mean().detach().cpu().item()) if batch.valid.any() else 0.0
            loss_arr = np.asarray(losses, dtype=float)
            train_summary = summarize_train_infos(batch.final_infos)
            if epoch % debug_log_every == 0:
                _debug_log(
                    debug_enabled,
                    df,
                    "[Train] "
                    f"epoch={epoch}/{epochs} samples={pool.sample_count} "
                    f"reward={_format_float(reward_mean)} "
                    f"policy_loss={_format_float(loss_arr[:, 0].mean())} "
                    f"value_loss={_format_float(loss_arr[:, 1].mean())} "
                    f"entropy={_format_float(loss_arr[:, 2].mean())} "
                    f"train_fr={_format_float(train_summary['train_feasible_rate'])} "
                    f"train_obj={_format_float(train_summary['train_avg_best_objective_distance_km'])} "
                    f"timing_reset={batch.timings.get('rollout_reset_time_s', 0.0):.3f}s "
                    f"timing_model={batch.timings.get('rollout_model_action_time_s', 0.0):.3f}s "
                    f"timing_env={batch.timings.get('rollout_env_step_time_s', 0.0):.3f}s "
                    f"timing_ppo={ppo_update_time_s:.3f}s",
                )

            eval_row: dict[str, Any] = {}
            eval_wall_time_s = 0.0
            should_eval = eval_interval > 0 and (epoch % eval_interval == 0 or epoch == epochs)
            if should_eval:
                eval_start = time.perf_counter()
                eval_row = evaluate_fixed_dataset(agent, cfg, seed=seed, epoch=epoch, device=device)
                eval_wall_time_s = time.perf_counter() - eval_start
                eval_writer.writerow({"epoch": epoch, **eval_row})
                ef.flush()
                _debug_log(
                    debug_enabled,
                    df,
                    "[Eval] "
                    f"epoch={epoch}/{epochs} n={eval_row.get('eval_num_instances')} "
                    f"fr={_format_float(eval_row.get('eval_feasible_rate'))} "
                    f"obj={_format_float(eval_row.get('eval_avg_objective_distance_km'))} "
                    f"veh={_format_float(eval_row.get('eval_avg_vehicle_count'))} "
                    f"eval_wall={eval_wall_time_s:.3f}s status={eval_row.get('eval_status')}",
                )

            epoch_wall_time_s = time.perf_counter() - epoch_start
            writer.writerow(
                {
                    "epoch": epoch,
                    "reward_mean": reward_mean,
                    "policy_loss": float(loss_arr[:, 0].mean()),
                    "value_loss": float(loss_arr[:, 1].mean()),
                    "entropy": float(loss_arr[:, 2].mean()),
                    "samples_seen": pool.sample_count,
                    "num_envs": num_envs,
                    "n_traj": int(train_cfg.get("n_traj", 50)),
                    "rollout_steps": rollout_steps,
                    "num_minibatches": minibatches,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "effective_instances_per_optimizer_step": int(np.ceil(num_envs / max(minibatches, 1))) * gradient_accumulation_steps,
                    "pbrs_scale": pbrs_scale,
                    "initial_env_pool_time_s": initial_env_pool_time_s,
                    "rollout_reset_time_s": batch.timings.get("rollout_reset_time_s", ""),
                    "rollout_stack_obs_time_s": batch.timings.get("rollout_stack_obs_time_s", ""),
                    "rollout_model_action_time_s": batch.timings.get("rollout_model_action_time_s", ""),
                    "rollout_env_step_time_s": batch.timings.get("rollout_env_step_time_s", ""),
                    "rollout_interaction_time_s": batch.timings.get("rollout_interaction_time_s", ""),
                    "rollout_total_time_s": batch.timings.get("rollout_total_time_s", ""),
                    "ppo_update_time_s": ppo_update_time_s,
                    "eval_wall_time_s": eval_wall_time_s,
                    "epoch_wall_time_s": epoch_wall_time_s,
                    "train_feasible_rate": train_summary.get("train_feasible_rate", ""),
                    "train_avg_best_objective_distance_km": train_summary.get("train_avg_best_objective_distance_km", ""),
                    "train_avg_vehicle_count": train_summary.get("train_avg_vehicle_count", ""),
                    "train_avg_served_customers": train_summary.get("train_avg_served_customers", ""),
                    "eval_avg_objective_distance_km": eval_row.get("eval_avg_objective_distance_km", ""),
                    "eval_avg_vehicle_count": eval_row.get("eval_avg_vehicle_count", ""),
                    "eval_feasible_rate": eval_row.get("eval_feasible_rate", ""),
                    "eval_avg_runtime_s": eval_row.get("eval_avg_runtime_s", ""),
                    "eval_num_instances": eval_row.get("eval_num_instances", ""),
                    "eval_n_traj": eval_row.get("eval_n_traj", ""),
                    "eval_batch_size": eval_row.get("eval_batch_size", ""),
                    "eval_num_batches": eval_row.get("eval_num_batches", ""),
                    "eval_decode_mode": eval_row.get("eval_decode_mode", ""),
                    "eval_info_level": eval_row.get("eval_info_level", ""),
                    "eval_save_routes": eval_row.get("eval_save_routes", ""),
                    "eval_status": eval_row.get("eval_status", ""),
                }
            )
            f.flush()
            if epoch % checkpoint_interval == 0 or epoch == epochs:
                save_checkpoint(ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pt", agent, optimizer, cfg, epoch, seed)

    save_checkpoint(ckpt_dir / "checkpoint_final.pt", agent, optimizer, cfg, epochs, seed)
    close_pool = getattr(pool, "close", None)
    if callable(close_pool):
        close_pool(terminate=True)
    return ckpt_dir / "checkpoint_final.pt"

