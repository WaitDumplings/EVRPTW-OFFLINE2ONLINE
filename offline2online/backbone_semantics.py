from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .models import Agent
from .trainer import (
    _attach_decomposed_rewards,
    deep_update,
    load_config,
    make_envs,
    set_seed,
)
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.rollout import (
    collect_rollout,
    reset_envs,
    stack_observations,
    step_envs,
)


def _close_pool(pool) -> None:
    close_pool = getattr(pool, "close", None)
    if callable(close_pool):
        close_pool(terminate=True)


def _small_check_config(cfg: dict[str, Any], num_envs: int, n_traj: int, rollout_steps: int) -> dict[str, Any]:
    return deep_update(
        copy.deepcopy(cfg),
        {
            "training": {
                "num_envs_per_gpu": int(num_envs),
                "n_traj": int(n_traj),
                "rollout_steps": int(rollout_steps),
                "mixed_precision": False,
                "debug": False,
            },
            "data": {
                "async_instance_prefetch": False,
                "mother_board_pool_size": min(int(cfg.get("data", {}).get("mother_board_pool_size", 1) or 1), max(int(num_envs), 1)),
            },
            "env": {"info_level": "light"},
            "evaluation": {"eval_interval": 0},
            "offline": {"method": "ppo"},
        },
    )


def build_agent(cfg: dict[str, Any], device: str | torch.device) -> Agent:
    model_cfg = cfg.get("model", {}) or {}
    critic_cfg = cfg.get("critic", {}) or {}
    return Agent(
        embedding_dim=int(model_cfg.get("embedding_dim", 256)),
        tanh_clipping=float(model_cfg.get("tanh_clipping", 15.0)),
        n_encode_layers=int(model_cfg.get("n_encode_layers", 2)),
        device=device,
        use_graph_token=bool(model_cfg.get("use_graph_token", True)),
        use_dynamic_decision_encoder=bool(model_cfg.get("use_dynamic_decision_encoder", False)),
        dynamic_decision_heads=int(model_cfg.get("dynamic_decision_heads", 4)),
        use_decomposed_critic=bool(model_cfg.get("use_decomposed_critic", critic_cfg.get("use_decomposed_critic", True))),
    ).to(device)


def reward_decomposition_check(cfg: dict[str, Any], seed: int, device: str, num_envs: int, n_traj: int, rollout_steps: int) -> dict[str, Any]:
    cfg = _small_check_config(cfg, num_envs=num_envs, n_traj=n_traj, rollout_steps=rollout_steps)
    set_seed(seed)
    agent = build_agent(cfg, device)
    envs, pool = make_envs(cfg, seed)
    try:
        batch = collect_rollout(agent, envs, rollout_steps=rollout_steps, decode_mode="sample", device=device, seed=seed, profile_timing=False)
        stats = _attach_decomposed_rewards(batch)
        episode_total = batch.rewards_total.masked_fill(~batch.valid, 0.0).sum(dim=0)
        episode_decomp = (batch.rewards_boundary + batch.rewards_internal).masked_fill(~batch.valid, 0.0).sum(dim=0)
        stats.update(
            {
                "steps_collected": int(batch.rewards.size(0)),
                "num_envs": int(batch.rewards.size(1)),
                "n_traj": int(batch.rewards.size(2)),
                "episode_total_mean": float(episode_total.mean().detach().cpu().item()) if episode_total.numel() else 0.0,
                "episode_decomposed_mean": float(episode_decomp.mean().detach().cpu().item()) if episode_decomp.numel() else 0.0,
            }
        )
        return stats
    finally:
        _close_pool(pool)


def critic_shape_check(cfg: dict[str, Any], seed: int, device: str, num_envs: int, n_traj: int) -> dict[str, Any]:
    cfg = _small_check_config(cfg, num_envs=num_envs, n_traj=n_traj, rollout_steps=1)
    set_seed(seed)
    agent = build_agent(cfg, device)
    envs, pool = make_envs(cfg, seed)
    try:
        observations, _ = reset_envs(envs, seed=seed)
        obs_batch = stack_observations(observations)
        with torch.no_grad():
            values = agent.get_value(obs_batch)
        expected_heads = 3 if bool(cfg.get("critic", {}).get("use_decomposed_critic", True)) else 1
        return {
            "value_shape": list(values.shape),
            "expected_last_dim": expected_heads,
            "last_dim_ok": bool(values.dim() >= 1 and values.shape[-1] == expected_heads),
            "has_nan": bool(torch.isnan(values).any().detach().cpu().item()),
        }
    finally:
        _close_pool(pool)


def _traj_mask(mask: Any) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim == 2:
        return arr[0]
    if arr.ndim == 1:
        return arr
    raise ValueError(f"unsupported action_mask shape {arr.shape}")


def cs_once_per_route_check(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    cfg = _small_check_config(cfg, num_envs=1, n_traj=1, rollout_steps=4)
    envs, pool = make_envs(cfg, seed)
    try:
        observations, _ = reset_envs(envs, seed=seed)
        obs = observations[0]
        num_customers = int(cfg.get("data", {}).get("num_customers", 0))
        cs_start = 1 + num_customers
        mask0 = _traj_mask(obs["action_mask"])
        feasible_cs = [idx for idx in range(cs_start, mask0.size) if mask0[idx]]
        if not feasible_cs:
            return {"status": "skipped", "reason": "no feasible charging station from initial state"}
        cs_idx = int(feasible_cs[0])
        observations, _, _, _ = step_envs(envs, np.asarray([[cs_idx]], dtype=np.int64))
        mask_after_cs = _traj_mask(observations[0]["action_mask"])
        same_cs_blocked = not bool(mask_after_cs[cs_idx])
        depot_feasible = bool(mask_after_cs[0])
        reset_restores_cs = None
        if depot_feasible:
            observations, _, _, _ = step_envs(envs, np.asarray([[0]], dtype=np.int64))
            mask_after_depot = _traj_mask(observations[0]["action_mask"])
            reset_restores_cs = bool(mask_after_depot[cs_idx])
        return {
            "status": "ok",
            "cs_idx": cs_idx,
            "same_cs_blocked_before_depot": same_cs_blocked,
            "depot_feasible_after_cs": depot_feasible,
            "same_cs_feasible_after_depot_if_physically_feasible": reset_restores_cs,
        }
    finally:
        _close_pool(pool)


def fast_reference_consistency_check(cfg: dict[str, Any], seed: int, steps: int = 8) -> dict[str, Any]:
    base = _small_check_config(cfg, num_envs=1, n_traj=1, rollout_steps=steps)
    fast_cfg = deep_update(copy.deepcopy(base), {"env": {"use_fast_env": True, "use_jit_mask": True}})
    ref_cfg = deep_update(copy.deepcopy(base), {"env": {"use_fast_env": False, "use_jit_mask": False}})
    fast_envs, fast_pool = make_envs(fast_cfg, seed)
    ref_envs, ref_pool = make_envs(ref_cfg, seed)
    try:
        fast_obs, _ = reset_envs(fast_envs, seed=seed)
        ref_obs, _ = reset_envs(ref_envs, seed=seed)
        mask_mismatches = 0
        reward_mismatches = 0
        done_mismatches = 0
        executed = 0
        for _ in range(int(steps)):
            fast_mask = _traj_mask(fast_obs[0]["action_mask"])
            ref_mask = _traj_mask(ref_obs[0]["action_mask"])
            if fast_mask.shape != ref_mask.shape or not np.array_equal(fast_mask, ref_mask):
                mask_mismatches += 1
            feasible = np.flatnonzero(fast_mask & ref_mask)
            if feasible.size == 0:
                break
            action = int(feasible[0])
            action_arr = np.asarray([[action]], dtype=np.int64)
            fast_obs, fast_reward, fast_done, _ = step_envs(fast_envs, action_arr)
            ref_obs, ref_reward, ref_done, _ = step_envs(ref_envs, action_arr)
            if not np.allclose(fast_reward, ref_reward, atol=1e-6, rtol=1e-6):
                reward_mismatches += 1
            if not np.array_equal(fast_done, ref_done):
                done_mismatches += 1
            executed += 1
            if bool(fast_done.all()) or bool(ref_done.all()):
                break
        return {
            "status": "ok",
            "steps_executed": executed,
            "mask_mismatches": mask_mismatches,
            "reward_mismatches": reward_mismatches,
            "done_mismatches": done_mismatches,
        }
    finally:
        _close_pool(fast_pool)
        _close_pool(ref_pool)


def run_all_checks(cfg: dict[str, Any], seed: int, device: str, num_envs: int, n_traj: int, rollout_steps: int) -> dict[str, Any]:
    return {
        "reward_decomposition": reward_decomposition_check(cfg, seed, device, num_envs, n_traj, rollout_steps),
        "critic_shape": critic_shape_check(cfg, seed, device, num_envs, n_traj),
        "cs_once_per_route": cs_once_per_route_check(cfg, seed),
        "fast_reference_consistency": fast_reference_consistency_check(cfg, seed, steps=min(int(rollout_steps), 20)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backbone semantic checks for EVRPTW-OFFLINE2ONLINE.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=2005)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--n-traj", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    results = run_all_checks(cfg, args.seed, args.device, args.num_envs, args.n_traj, args.rollout_steps)
    text = json.dumps(results, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
