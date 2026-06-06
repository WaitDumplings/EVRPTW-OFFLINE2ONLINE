from __future__ import annotations

from contextlib import nullcontext
import csv
import json
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
    _slice_obs_by_env,
    evaluate_fixed_dataset,
    evaluate_policy_loss,
    make_envs,
    pbrs_scale_for_epoch,
    set_pbrs_reward_scale,
    summarize_train_infos,
)

from .models import Agent
from .offline_data import (
    ExpertReplayBuffer,
    build_expert_trajectories,
    compute_bc_loss,
    compute_route_supervised_loss,
    load_solver_expert_records,
)

from ablation.dapg import compute_dapg_demo_loss


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        local = REPO_ROOT / "configs" / cfg_path
        repo_local = REPO_ROOT / cfg_path
        cfg_path = local if local.exists() else repo_local if repo_local.exists() else cfg_path
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


def _amp_enabled(train_cfg: dict[str, Any], device: str | torch.device) -> bool:
    return bool(train_cfg.get("mixed_precision", train_cfg.get("amp", False))) and str(device).startswith("cuda") and torch.cuda.is_available()


def _autocast_context(device: str | torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    return torch.autocast(device_type=device_type, dtype=torch.float16, enabled=True)


def _new_grad_scaler(enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def _backward(loss: torch.Tensor, scaler, amp_enabled: bool) -> None:
    if amp_enabled and scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()


def _optimizer_step(
    optimizer: torch.optim.Optimizer,
    agent: Agent,
    max_grad_norm: float,
    scaler,
    amp_enabled: bool,
) -> None:
    if amp_enabled and scaler is not None:
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
    if amp_enabled and scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()


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


def _offline_method(cfg: dict[str, Any]) -> str:
    return str((cfg.get("offline", {}) or {}).get("method", "ppo")).strip().lower()


def _is_sl_ppo_method(method: str) -> bool:
    return method in {"sl_ppo", "sl-ppo", "slppo"}


def _is_route_bc_method(method: str) -> bool:
    return method in {"route_bc_ppo", "route-bc-ppo", "route_bc", "route-bc", "route_sl_ppo"}


def _requires_expert_routes(method: str) -> bool:
    return method in {"bc_ppo", "bc-ppo", "dapg", "bc_dapg", "bc+ppo"} or _is_route_bc_method(method)


def _offline_coef(offline_cfg: dict[str, Any], epoch: int) -> float:
    coef = float(offline_cfg.get("bc_coef", offline_cfg.get("dapg_bc_coef", 1.0)))
    decay = float(offline_cfg.get("bc_decay", offline_cfg.get("dapg_decay", 1.0)))
    min_coef = float(offline_cfg.get("min_bc_coef", 0.0))
    if decay != 1.0:
        coef *= decay ** max(int(epoch) - 1, 0)
    return max(coef, min_coef)


def _advantage_config(cfg: dict[str, Any]) -> dict[str, Any]:
    adv_cfg = dict(cfg.get("advantage", {}) or {})
    offline_cfg = cfg.get("offline", {}) or {}
    method = _offline_method(cfg)
    for key in (
        "use_group_advantage",
        "group_adv_coef",
        "group_adv_clip",
        "group_adv_std_floor",
        "group_infeasible_penalty",
        "use_reference_advantage",
        "reference_adv_coef",
        "reference_adv_rho",
        "reference_adv_clip",
        "reference_success_only",
        "use_reference_soft_gate",
        "reference_soft_gate_eta",
        "reference_policy_estimate",
        "renormalize_after_aux_advantage",
    ):
        if key in offline_cfg and key not in adv_cfg:
            adv_cfg[key] = offline_cfg[key]
    if _is_sl_ppo_method(method):
        adv_cfg.setdefault("use_reference_advantage", True)
        adv_cfg.setdefault("reference_adv_coef", 0.10)
        adv_cfg.setdefault("reference_adv_rho", 0.10)
        adv_cfg.setdefault("reference_adv_clip", 2.0)
        adv_cfg.setdefault("reference_success_only", True)
        adv_cfg.setdefault("use_reference_soft_gate", True)
        adv_cfg.setdefault("reference_soft_gate_eta", 0.05)
        adv_cfg.setdefault("reference_policy_estimate", "best")
    return adv_cfg


def _group_advantage_enabled(cfg: dict[str, Any]) -> bool:
    adv_cfg = _advantage_config(cfg)
    return bool(adv_cfg.get("use_group_advantage", False)) or float(adv_cfg.get("group_adv_coef", 0.0) or 0.0) != 0.0


def _reference_advantage_enabled(cfg: dict[str, Any]) -> bool:
    adv_cfg = _advantage_config(cfg)
    return bool(adv_cfg.get("use_reference_advantage", False)) or float(adv_cfg.get("reference_adv_coef", 0.0) or 0.0) != 0.0


def _env_instance_id(env) -> str | None:
    candidates = [env, getattr(env, "unwrapped", None), getattr(env, "env", None)]
    current = env
    for _ in range(8):
        current = getattr(current, "env", None)
        if current is None:
            break
        candidates.extend([current, getattr(current, "unwrapped", None)])
    for obj in candidates:
        instance = getattr(obj, "instance", None) if obj is not None else None
        instance_id = getattr(instance, "instance_id", None)
        if instance_id is not None:
            return str(instance_id)
    return None


def _final_info_arrays(final_infos: list[dict[str, Any]], num_envs: int, n_traj: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    objective = np.full((num_envs, n_traj), np.nan, dtype=np.float64)
    success = np.zeros((num_envs, n_traj), dtype=bool)
    served = np.zeros((num_envs, n_traj), dtype=np.float64)
    for env_idx, info in enumerate(final_infos[:num_envs]):
        obj_arr = np.asarray(info.get("objective_distance_km", []), dtype=np.float64).reshape(-1)
        suc_arr = np.asarray(info.get("success", []), dtype=bool).reshape(-1)
        srv_arr = np.asarray(info.get("served_customers", []), dtype=np.float64).reshape(-1)
        limit = min(n_traj, obj_arr.size)
        if limit > 0:
            objective[env_idx, :limit] = obj_arr[:limit]
        limit = min(n_traj, suc_arr.size)
        if limit > 0:
            success[env_idx, :limit] = suc_arr[:limit]
        limit = min(n_traj, srv_arr.size)
        if limit > 0:
            served[env_idx, :limit] = srv_arr[:limit]
    return objective, success, served


def _finite_mean_std(value: np.ndarray) -> tuple[float, float]:
    finite = value[np.isfinite(value)]
    if finite.size == 0:
        return 0.0, 0.0
    return float(finite.mean()), float(finite.std())


def _compute_gae_returns(batch, gamma: float, gae_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(batch.rewards)
    last_gae = torch.zeros_like(batch.rewards[0])
    for step in reversed(range(batch.rewards.size(0))):
        if step == batch.rewards.size(0) - 1:
            next_value = torch.zeros_like(batch.values[step])
        else:
            next_value = batch.values[step + 1]
        next_nonterminal = (~batch.dones[step]).float()
        delta = batch.rewards[step] + float(gamma) * next_value * next_nonterminal - batch.values[step]
        last_gae = delta + float(gamma) * float(gae_lambda) * next_nonterminal * last_gae
        advantages[step] = last_gae
    returns = advantages + batch.values
    return returns, advantages


def _apply_auxiliary_advantages(
    advantages: torch.Tensor,
    batch,
    cfg: dict[str, Any],
    envs,
    expert_buffer: ExpertReplayBuffer | None,
    device: str | torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    adv_cfg = _advantage_config(cfg)
    use_group = _group_advantage_enabled(cfg)
    use_ref = _reference_advantage_enabled(cfg)
    info = {
        "group_adv_mean": 0.0,
        "group_adv_std": 0.0,
        "ref_adv_mean": 0.0,
        "ref_adv_std": 0.0,
        "aux_adv_mean": 0.0,
        "aux_adv_std": 0.0,
    }
    if not use_group and not use_ref:
        return advantages, info

    num_envs = int(advantages.size(1))
    n_traj = int(advantages.size(2))
    objective, success, served = _final_info_arrays(batch.final_infos, num_envs, n_traj)
    traj_adv = np.zeros((num_envs, n_traj), dtype=np.float64)

    if use_group:
        num_customers = max(1, int(cfg.get("data", {}).get("num_customers", 1)))
        penalty = float(adv_cfg.get("group_infeasible_penalty", 10.0))
        score = -objective.copy()
        finite_score = np.isfinite(score)
        if np.any(finite_score):
            fallback = float(np.nanmin(score[finite_score]) - penalty * (num_customers + 1))
        else:
            fallback = -penalty * (num_customers + 1)
        score[~finite_score] = fallback
        missing_customers = np.maximum(float(num_customers) - served, 0.0)
        score[~success] -= penalty * (missing_customers[~success] + 1.0)
        mean = score.mean(axis=1, keepdims=True)
        std = np.maximum(score.std(axis=1, keepdims=True), float(adv_cfg.get("group_adv_std_floor", 1e-8)))
        group_adv = np.divide(score - mean, std + 1e-8, out=np.zeros_like(score), where=std > 1e-8)
        group_adv = np.clip(group_adv, -float(adv_cfg.get("group_adv_clip", 3.0)), float(adv_cfg.get("group_adv_clip", 3.0)))
        group_adv *= float(adv_cfg.get("group_adv_coef", 1.0))
        traj_adv += group_adv
        info["group_adv_mean"], info["group_adv_std"] = _finite_mean_std(group_adv)

    if use_ref and expert_buffer is not None:
        ref_clip = float(adv_cfg.get("reference_adv_clip", 1.0))
        ref_coef = float(adv_cfg.get("reference_adv_coef", 1.0))
        ref_rho = max(float(adv_cfg.get("reference_adv_rho", 1.0)), 1e-8)
        success_only = bool(adv_cfg.get("reference_success_only", True))
        ref_adv = np.zeros((num_envs, n_traj), dtype=np.float64)
        for env_idx, env in enumerate(envs[:num_envs]):
            ref_obj = expert_buffer.reference_objective(_env_instance_id(env))
            if ref_obj is None or not np.isfinite(ref_obj) or ref_obj <= 0.0:
                continue
            row = (float(ref_obj) - objective[env_idx]) / max(ref_rho * float(ref_obj), 1e-8)
            row[~np.isfinite(row)] = 0.0
            if success_only:
                row = np.where(success[env_idx], row, 0.0)
            ref_adv[env_idx] = np.clip(row, -ref_clip, ref_clip)
        ref_adv *= ref_coef
        traj_adv += ref_adv
        info["ref_adv_mean"], info["ref_adv_std"] = _finite_mean_std(ref_adv)

    info["aux_adv_mean"], info["aux_adv_std"] = _finite_mean_std(traj_adv)
    aux_tensor = torch.as_tensor(traj_adv, dtype=advantages.dtype, device=device).unsqueeze(0)
    advantages = advantages + aux_tensor * batch.valid.float()
    if bool(adv_cfg.get("renormalize_after_aux_advantage", True)):
        adv_vals = advantages[batch.valid]
        if adv_vals.numel() > 1:
            advantages = (advantages - adv_vals.mean()) / (adv_vals.std(unbiased=False) + 1e-8)
    return advantages, info


def _solution_level_advantage_tensors(
    batch,
    cfg: dict[str, Any],
    envs,
    expert_buffer: ExpertReplayBuffer | None,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    adv_cfg = _advantage_config(cfg)
    use_group = _group_advantage_enabled(cfg)
    use_ref = _reference_advantage_enabled(cfg)
    num_envs = int(batch.actions.size(1))
    n_traj = int(batch.actions.size(2))
    objective, success, served = _final_info_arrays(batch.final_infos, num_envs, n_traj)
    route_adv = np.zeros((num_envs, n_traj), dtype=np.float64)
    info = {
        "group_adv_mean": 0.0,
        "group_adv_std": 0.0,
        "ref_adv_mean": 0.0,
        "ref_adv_std": 0.0,
        "ref_gate_mean": 0.0,
        "ref_gate_std": 0.0,
        "route_adv_mean": 0.0,
        "route_adv_std": 0.0,
    }

    if use_group:
        num_customers = max(1, int(cfg.get("data", {}).get("num_customers", 1)))
        penalty = float(adv_cfg.get("group_infeasible_penalty", 10.0))
        score = -objective.copy()
        finite_score = np.isfinite(score)
        fallback = float(np.nanmin(score[finite_score]) - penalty * (num_customers + 1)) if np.any(finite_score) else -penalty * (num_customers + 1)
        score[~finite_score] = fallback
        missing_customers = np.maximum(float(num_customers) - served, 0.0)
        score[~success] -= penalty * (missing_customers[~success] + 1.0)
        mean = score.mean(axis=1, keepdims=True)
        std = np.maximum(score.std(axis=1, keepdims=True), float(adv_cfg.get("group_adv_std_floor", 5.0)))
        group_adv = (score - mean) / (std + 1e-8)
        group_adv = np.clip(group_adv, -float(adv_cfg.get("group_adv_clip", 3.0)), float(adv_cfg.get("group_adv_clip", 3.0)))
        group_adv *= float(adv_cfg.get("group_adv_coef", 0.30))
        route_adv += group_adv
        info["group_adv_mean"], info["group_adv_std"] = _finite_mean_std(group_adv)

    if use_ref and expert_buffer is not None:
        ref_clip = float(adv_cfg.get("reference_adv_clip", 2.0))
        ref_coef = float(adv_cfg.get("reference_adv_coef", 0.10))
        ref_rho = max(float(adv_cfg.get("reference_adv_rho", 0.10)), 1e-8)
        success_only = bool(adv_cfg.get("reference_success_only", True))
        use_gate = bool(adv_cfg.get("use_reference_soft_gate", True))
        gate_eta = max(float(adv_cfg.get("reference_soft_gate_eta", 0.05)), 1e-8)
        estimate_mode = str(adv_cfg.get("reference_policy_estimate", "best")).lower()
        ref_adv = np.zeros((num_envs, n_traj), dtype=np.float64)
        gate = np.ones((num_envs, 1), dtype=np.float64)
        for env_idx, env in enumerate(envs[:num_envs]):
            ref_obj = expert_buffer.reference_objective(_env_instance_id(env))
            if ref_obj is None or not np.isfinite(ref_obj) or ref_obj <= 0.0:
                ref_adv[env_idx] = 0.0
                gate[env_idx, 0] = 0.0
                continue
            row = (float(ref_obj) - objective[env_idx]) / max(ref_rho * float(ref_obj), 1e-8)
            row[~np.isfinite(row)] = 0.0
            if success_only:
                row = np.where(success[env_idx], row, 0.0)
            ref_adv[env_idx] = np.clip(row, -ref_clip, ref_clip)
            if use_gate:
                succ_obj = objective[env_idx][success[env_idx] & np.isfinite(objective[env_idx])]
                if succ_obj.size == 0:
                    estimate_obj = np.inf
                elif estimate_mode == "mean":
                    estimate_obj = float(np.mean(succ_obj))
                else:
                    estimate_obj = float(np.min(succ_obj))
                gap = (estimate_obj - float(ref_obj)) / max(float(ref_obj), 1e-8)
                gate[env_idx, 0] = float(np.clip(gap / gate_eta, 0.0, 1.0)) if np.isfinite(gap) else 1.0
        ref_used = ref_adv * gate
        ref_used *= ref_coef
        route_adv += ref_used
        info["ref_adv_mean"], info["ref_adv_std"] = _finite_mean_std(ref_used)
        info["ref_gate_mean"], info["ref_gate_std"] = _finite_mean_std(gate)

    info["route_adv_mean"], info["route_adv_std"] = _finite_mean_std(route_adv)
    return (
        torch.as_tensor(route_adv, dtype=batch.old_logprobs.dtype, device=device),
        torch.as_tensor(success, dtype=torch.bool, device=device),
        info,
    )


def _compute_solution_level_ppo_loss(
    agent: Agent,
    batch,
    route_adv: torch.Tensor,
    route_success: torch.Tensor,
    cfg: dict[str, Any],
    env_indices: np.ndarray,
    device: str | torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    del device
    offline_cfg = cfg.get("offline", {}) or {}
    route_clip_eps = float(offline_cfg.get("route_clip_eps", offline_cfg.get("sl_clip_coef", 0.20)))
    only_success = bool(offline_cfg.get("only_success_route_loss", True))
    total_steps = int(batch.actions.size(0))
    env_indices = np.asarray(env_indices, dtype=np.int64)
    cached_state = agent.backbone.encode(_slice_obs_by_env(batch.observations[0], env_indices))
    sum_delta = torch.zeros_like(batch.old_logprobs[0, env_indices])
    valid_counts = torch.zeros_like(sum_delta)
    for step in range(total_steps):
        obs_mb = _slice_obs_by_env(batch.observations[step], env_indices)
        actions = batch.actions[step, env_indices].long()
        _, new_logprob, _, _, _ = agent.get_action_and_value_cached(obs_mb, action=actions, state=cached_state)
        valid = batch.valid[step, env_indices].to(dtype=new_logprob.dtype)
        sum_delta = sum_delta + (new_logprob - batch.old_logprobs[step, env_indices]) * valid
        valid_counts = valid_counts + valid

    mean_delta = sum_delta / valid_counts.clamp_min(1.0)
    route_ratio = torch.exp(mean_delta)
    adv = route_adv[env_indices].detach()
    route_mask = valid_counts > 0
    if only_success:
        route_mask = route_mask & route_success[env_indices]
    route_mask = route_mask & torch.isfinite(adv) & (adv != 0)
    if not bool(route_mask.any()):
        zero = route_ratio.sum() * 0.0
        return zero, {
            "sl_route_loss": 0.0,
            "sl_route_ratio_mean": 1.0,
            "sl_route_ratio_std": 0.0,
            "sl_route_clip_frac": 0.0,
            "sl_route_adv_mean": 0.0,
            "sl_route_adv_std": 0.0,
            "sl_num_routes_used": 0.0,
        }

    r = route_ratio[route_mask]
    a = adv[route_mask]
    unclipped = r * a
    clipped = torch.clamp(r, 1.0 - route_clip_eps, 1.0 + route_clip_eps) * a
    route_loss = -torch.minimum(unclipped, clipped).mean()
    clip_frac = ((r > 1.0 + route_clip_eps) | (r < 1.0 - route_clip_eps)).float().mean()
    return route_loss, {
        "sl_route_loss": float(route_loss.detach().cpu().item()),
        "sl_route_ratio_mean": float(r.detach().mean().cpu().item()),
        "sl_route_ratio_std": float(r.detach().std(unbiased=False).cpu().item()) if r.numel() > 1 else 0.0,
        "sl_route_clip_frac": float(clip_frac.detach().cpu().item()),
        "sl_route_adv_mean": float(a.detach().mean().cpu().item()),
        "sl_route_adv_std": float(a.detach().std(unbiased=False).cpu().item()) if a.numel() > 1 else 0.0,
        "sl_num_routes_used": float(route_mask.sum().detach().cpu().item()),
    }


def _load_expert_buffer(cfg: dict[str, Any], seed: int, debug_enabled: bool, debug_file) -> ExpertReplayBuffer | None:
    offline_cfg = cfg.get("offline", {}) or {}
    method = _offline_method(cfg)
    need_archive = _requires_expert_routes(method) or _reference_advantage_enabled(cfg)
    if method in {"", "none", "ppo"} and not need_archive:
        return None
    if _is_sl_ppo_method(method):
        need_archive = _reference_advantage_enabled(cfg)
    if not need_archive:
        return None
    solution_path = offline_cfg.get("expert_solution_path") or offline_cfg.get("expert_csv_path")
    if not solution_path:
        raise ValueError(f"offline.method={method!r} or reference advantage requires offline.expert_solution_path")
    data_cfg = cfg.get("data", {}) or {}
    dataset_path = offline_cfg.get("expert_dataset_path") or data_cfg.get("train_dataset_path")
    if not dataset_path:
        raise ValueError("offline expert loading requires data.train_dataset_path or offline.expert_dataset_path")
    records = load_solver_expert_records(
        dataset_path=dataset_path,
        solution_csv_path=solution_path,
        num_customers=int(data_cfg.get("num_customers", 5)),
        num_charging_stations=int(data_cfg.get("num_charging_stations", 3)),
        limit=offline_cfg.get("expert_limit"),
    )
    trajectories, stats = build_expert_trajectories(
        records,
        cfg,
        max_records=offline_cfg.get("max_replay_records"),
        strict=bool(offline_cfg.get("strict_replay", True)),
    )
    _debug_log(
        debug_enabled,
        debug_file,
        "[OfflineArchive] "
        f"method={method} records={stats['records_seen']} trajectories={stats['trajectories']} "
        f"invalid={stats['invalid_records']} steps={stats['steps']} "
        f"avg_steps={stats['avg_steps_per_route']:.3f} solution_path={solution_path}",
    )
    return ExpertReplayBuffer(trajectories, seed=seed + int(offline_cfg.get("replay_seed_offset", 17_000)))


def _run_bc_updates(
    agent: Agent,
    optimizer: torch.optim.Optimizer,
    expert_buffer: ExpertReplayBuffer,
    cfg: dict[str, Any],
    device: str | torch.device,
    epoch: int,
    *,
    coef: float,
    updates: int,
    scaler=None,
    amp_enabled: bool = False,
) -> dict[str, Any]:
    offline_cfg = cfg.get("offline", {}) or {}
    batch_size = int(offline_cfg.get("bc_batch_size", 256))
    max_grad_norm = float(cfg.get("training", {}).get("max_grad_norm", 1.0))
    losses = []
    accs = []
    entropies = []
    steps = 0
    if coef <= 0.0 or updates <= 0:
        return {"bc_loss": 0.0, "bc_accuracy": 0.0, "bc_entropy": 0.0, "bc_steps": 0, "bc_coef": coef, "offline_updates": 0}
    agent.train()
    for _ in range(int(updates)):
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp_enabled):
            loss, info = compute_bc_loss(agent, expert_buffer, batch_size=batch_size, device=device)
        _backward(float(coef) * loss, scaler, amp_enabled)
        _optimizer_step(optimizer, agent, max_grad_norm, scaler, amp_enabled)
        losses.append(float(info["bc_loss"]))
        accs.append(float(info["bc_accuracy"]))
        entropies.append(float(info["bc_entropy"]))
        steps += int(info["bc_steps"])
    return {
        "bc_loss": float(np.mean(losses)) if losses else 0.0,
        "bc_accuracy": float(np.mean(accs)) if accs else 0.0,
        "bc_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "bc_steps": int(steps),
        "bc_coef": float(coef),
        "offline_updates": int(updates),
    }


def _run_route_bc_updates(
    agent: Agent,
    optimizer: torch.optim.Optimizer,
    expert_buffer: ExpertReplayBuffer,
    cfg: dict[str, Any],
    device: str | torch.device,
    epoch: int,
    *,
    coef: float,
    updates: int,
    scaler=None,
    amp_enabled: bool = False,
) -> dict[str, Any]:
    offline_cfg = cfg.get("offline", {}) or {}
    batch_size = int(offline_cfg.get("route_batch_size", offline_cfg.get("bc_batch_size", 256)))
    max_grad_norm = float(cfg.get("training", {}).get("max_grad_norm", 1.0))
    losses = []
    entropies = []
    route_counts = []
    route_lens = []
    steps = 0
    if coef <= 0.0 or updates <= 0:
        return {"route_bc_loss": 0.0, "route_bc_entropy": 0.0, "route_bc_count": 0, "route_bc_step_count": 0, "route_bc_avg_route_len": 0.0, "route_bc_coef": coef, "offline_updates": 0}
    agent.train()
    for _ in range(int(updates)):
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp_enabled):
            loss, info = compute_route_supervised_loss(agent, expert_buffer, batch_size=batch_size, device=device)
        _backward(float(coef) * loss, scaler, amp_enabled)
        _optimizer_step(optimizer, agent, max_grad_norm, scaler, amp_enabled)
        losses.append(float(info["sl_route_loss"]))
        entropies.append(float(info["sl_route_entropy"]))
        route_counts.append(float(info["sl_route_count"]))
        route_lens.append(float(info["sl_avg_route_len"]))
        steps += int(info["sl_step_count"])
    return {
        "route_bc_loss": float(np.mean(losses)) if losses else 0.0,
        "route_bc_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "route_bc_count": int(np.sum(route_counts)) if route_counts else 0,
        "route_bc_step_count": int(steps),
        "route_bc_avg_route_len": float(np.mean(route_lens)) if route_lens else 0.0,
        "route_bc_coef": float(coef),
        "offline_updates": int(updates),
    }


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
    offline_cfg = cfg.get("offline", {}) or {}
    offline_method = _offline_method(cfg)
    model_cfg = cfg.get("model", {})
    cfg.setdefault("env", {})["use_fast_env"] = True
    cfg["env"].setdefault("info_level", "light")
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
        use_dynamic_embedding=bool(model_cfg.get("use_dynamic_embedding", False)),
        use_candidate_dynamic_embedding=model_cfg.get("use_candidate_dynamic_embedding", False),
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
    use_gae = bool(train_cfg.get("use_gae", True))
    gae_lambda = float(train_cfg.get("gae_lambda", 0.95))
    amp_enabled = _amp_enabled(train_cfg, device)
    scaler = _new_grad_scaler(amp_enabled)
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))

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
        "train_mode",
        "mixed_precision",
        "use_gae",
        "gae_lambda",
        "bc_loss",
        "bc_accuracy",
        "bc_entropy",
        "bc_coef",
        "bc_steps",
        "route_bc_loss",
        "route_bc_entropy",
        "route_bc_coef",
        "route_bc_count",
        "route_bc_step_count",
        "route_bc_avg_route_len",
        "sl_route_loss",
        "sl_route_ratio_mean",
        "sl_route_ratio_std",
        "sl_route_clip_frac",
        "sl_route_adv_mean",
        "sl_route_adv_std",
        "sl_num_routes_used",
        "sl_coef",
        "sl_ref_gate_mean",
        "sl_ref_gate_std",
        "offline_updates",
        "group_adv_mean",
        "group_adv_std",
        "ref_adv_mean",
        "ref_adv_std",
        "aux_adv_mean",
        "aux_adv_std",
        "route_adv_mean",
        "route_adv_std",
        "best_eval_avg_objective_distance_km",
        "best_eval_epoch",
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
        expert_buffer = _load_expert_buffer(cfg, seed, debug_enabled, df)
        best_eval_objective = float("inf")
        best_eval_epoch = 0
        _debug_log(
            debug_enabled,
            df,
            f"[Init] run={run_name} seed={seed} device={device} epochs={epochs} "
            f"n_traj={train_cfg.get('n_traj', 50)} rollout_steps={rollout_steps} "
            f"num_envs={train_cfg.get('num_envs_per_gpu', 128)} minibatches={num_minibatches} "
            f"accum_grad={gradient_accumulation_steps} "
            f"n_encode_layers={model_cfg.get('n_encode_layers', 2)} "
            f"use_graph_token={model_cfg.get('use_graph_token', True)} "
            f"use_dynamic_embedding={model_cfg.get('use_dynamic_embedding', False)} "
            f"mixed_precision={amp_enabled} "
            f"offline_method={offline_method} use_gae={use_gae} gae_lambda={gae_lambda} "
            f"expert_steps={expert_buffer.num_steps if expert_buffer is not None else 0} "
            f"initial_env_pool_time_s={initial_env_pool_time_s:.3f} "
            f"eval_interval={eval_interval} eval_n_traj={eval_cfg.get('eval_n_traj', 50)} "
            f"eval_batch_size={eval_cfg.get('eval_batch_size', 1000)}",
        )
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            pbrs_scale = pbrs_scale_for_epoch(cfg, epoch, epochs)
            set_pbrs_reward_scale(envs, pbrs_scale)
            agent.train()
            offline_updates = 0
            bc_info: dict[str, Any] = {}
            sl_info: dict[str, Any] = {}
            adv_info: dict[str, Any] = {}
            bc_warmup_epochs = int(offline_cfg.get("bc_warmup_epochs", 0))
            bc_updates_per_epoch = int(offline_cfg.get("bc_updates_per_epoch", offline_cfg.get("offline_updates_per_epoch", 1)))
            route_updates_per_epoch = int(offline_cfg.get("route_updates_per_epoch", offline_cfg.get("offline_updates_per_epoch", 1)))
            offline_coef = _offline_coef(offline_cfg, epoch)
            do_bc_warmup = (
                offline_method in {"bc_ppo", "bc-ppo", "dapg", "bc_dapg", "bc+ppo"}
                and expert_buffer is not None
                and epoch <= bc_warmup_epochs
            )

            batch = None
            train_summary = {
                "train_feasible_rate": np.nan,
                "train_avg_best_objective_distance_km": np.nan,
                "train_avg_vehicle_count": np.nan,
                "train_avg_served_customers": np.nan,
            }
            reward_mean = float("nan")
            losses = [(0.0, 0.0, 0.0)]
            num_envs = int(train_cfg.get("num_envs_per_gpu", 0))
            minibatches = min(num_minibatches, max(num_envs, 1))
            ppo_update_time_s = 0.0
            rollout_timings: dict[str, Any] = {}

            if do_bc_warmup:
                ppo_start = time.perf_counter()
                bc_info = _run_bc_updates(
                    agent,
                    optimizer,
                    expert_buffer,
                    cfg,
                    device,
                    epoch,
                    coef=float(offline_cfg.get("bc_warmup_coef", 1.0)),
                    updates=max(1, bc_updates_per_epoch),
                    scaler=scaler,
                    amp_enabled=amp_enabled,
                )
                offline_updates += int(bc_info.get("offline_updates", 0))
                ppo_update_time_s = time.perf_counter() - ppo_start
            else:
                batch = collect_rollout(
                    agent,
                    envs,
                    rollout_steps=rollout_steps,
                    decode_mode="sample",
                    device=device,
                    seed=seed + epoch * 100_000,
                    profile_timing=profile_timing,
                )
                if use_gae:
                    returns, advantages = _compute_gae_returns(batch, gamma=gamma, gae_lambda=gae_lambda)
                else:
                    returns = compute_returns(batch.rewards, batch.dones, gamma=gamma)
                    advantages = returns - batch.values
                adv_vals = advantages[batch.valid]
                if adv_vals.numel() > 1:
                    advantages = (advantages - adv_vals.mean()) / (adv_vals.std(unbiased=False) + 1e-8)
                sl_enabled = _is_sl_ppo_method(offline_method)
                route_bc_enabled = _is_route_bc_method(offline_method)
                route_adv_tensor = None
                route_success_tensor = None
                if sl_enabled:
                    route_adv_tensor, route_success_tensor, adv_info = _solution_level_advantage_tensors(
                        batch,
                        cfg,
                        envs,
                        expert_buffer,
                        device,
                    )
                else:
                    advantages, adv_info = _apply_auxiliary_advantages(
                        advantages,
                        batch,
                        cfg,
                        envs,
                        expert_buffer,
                        device,
                    )

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
                dapg_enabled = expert_buffer is not None and offline_method in {"dapg", "bc_dapg", "bc+ppo"}
                dapg_adv_scale = 1.0
                dapg_demo_coef = float(offline_coef)
                if dapg_enabled:
                    dapg_iteration = max(int(epoch) - int(bc_warmup_epochs) - 1, 0)
                    dapg_base_coef = float(offline_cfg.get("dapg_lambda0", offline_cfg.get("bc_coef", 0.1)))
                    dapg_decay = float(offline_cfg.get("dapg_lambda1", offline_cfg.get("bc_decay", 0.95)))
                    dapg_min_coef = float(offline_cfg.get("min_bc_coef", 0.0))
                    valid_adv = advantages.detach()[batch.valid]
                    if valid_adv.numel() > 0:
                        dapg_adv_scale = max(float(valid_adv.max().detach().cpu().item()), 0.0)
                    dapg_demo_coef = max(dapg_base_coef * (dapg_decay ** dapg_iteration), dapg_min_coef)
                    dapg_demo_coef *= dapg_adv_scale
                dapg_bc_batch_size = int(offline_cfg.get("bc_batch_size", 256))
                dapg_bc_losses: list[float] = []
                dapg_bc_accs: list[float] = []
                dapg_bc_entropies: list[float] = []
                dapg_bc_steps = 0
                sl_losses: list[float] = []
                sl_ratio_means: list[float] = []
                sl_ratio_stds: list[float] = []
                sl_clip_fracs: list[float] = []
                sl_adv_means: list[float] = []
                sl_adv_stds: list[float] = []
                sl_route_counts: list[float] = []
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
                                with _autocast_context(device, amp_enabled):
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
                                _backward(loss * chunk_weight / group_size, scaler, amp_enabled)
                                weighted_policy += policy_loss.item() * chunk_weight
                                weighted_value += value_loss.item() * chunk_weight
                                weighted_entropy += entropy.item() * chunk_weight
                            if sl_enabled and route_adv_tensor is not None and route_success_tensor is not None:
                                with _autocast_context(device, amp_enabled):
                                    route_loss, route_info = _compute_solution_level_ppo_loss(
                                        agent,
                                        batch,
                                        route_adv_tensor,
                                        route_success_tensor,
                                        cfg,
                                        env_indices,
                                        device,
                                    )
                                sl_coef = float(offline_cfg.get("sl_coef", offline_cfg.get("route_loss_coef", 0.10)))
                                _backward(sl_coef * route_loss / group_size, scaler, amp_enabled)
                                sl_losses.append(float(route_info["sl_route_loss"]))
                                sl_ratio_means.append(float(route_info["sl_route_ratio_mean"]))
                                sl_ratio_stds.append(float(route_info["sl_route_ratio_std"]))
                                sl_clip_fracs.append(float(route_info["sl_route_clip_frac"]))
                                sl_adv_means.append(float(route_info["sl_route_adv_mean"]))
                                sl_adv_stds.append(float(route_info["sl_route_adv_std"]))
                                sl_route_counts.append(float(route_info["sl_num_routes_used"]))
                            group_policy += weighted_policy / group_size
                            group_value += weighted_value / group_size
                            group_entropy += weighted_entropy / group_size
                        if dapg_enabled and dapg_demo_coef > 0.0 and expert_buffer is not None:
                            with _autocast_context(device, amp_enabled):
                                demo_loss, demo_info = compute_dapg_demo_loss(
                                    agent,
                                    expert_buffer,
                                    batch_size=dapg_bc_batch_size,
                                    device=device,
                                )
                            _backward(float(dapg_demo_coef) * demo_loss, scaler, amp_enabled)
                            dapg_bc_losses.append(float(demo_info["bc_loss"]))
                            dapg_bc_accs.append(float(demo_info["bc_accuracy"]))
                            dapg_bc_entropies.append(float(demo_info["bc_entropy"]))
                            dapg_bc_steps += int(demo_info["bc_steps"])
                            offline_updates += 1
                        _optimizer_step(optimizer, agent, max_grad_norm, scaler, amp_enabled)
                        losses.append((group_policy, group_value, group_entropy))
                if dapg_enabled:
                    bc_info = {
                        "bc_loss": float(np.mean(dapg_bc_losses)) if dapg_bc_losses else 0.0,
                        "bc_accuracy": float(np.mean(dapg_bc_accs)) if dapg_bc_accs else 0.0,
                        "bc_entropy": float(np.mean(dapg_bc_entropies)) if dapg_bc_entropies else 0.0,
                        "bc_steps": int(dapg_bc_steps),
                        "bc_coef": float(dapg_demo_coef),
                        "offline_updates": int(offline_updates),
                    }
                if sl_enabled:
                    sl_info = {
                        "sl_route_loss": float(np.mean(sl_losses)) if sl_losses else 0.0,
                        "sl_route_ratio_mean": float(np.mean(sl_ratio_means)) if sl_ratio_means else 1.0,
                        "sl_route_ratio_std": float(np.mean(sl_ratio_stds)) if sl_ratio_stds else 0.0,
                        "sl_route_clip_frac": float(np.mean(sl_clip_fracs)) if sl_clip_fracs else 0.0,
                        "sl_route_adv_mean": float(np.mean(sl_adv_means)) if sl_adv_means else 0.0,
                        "sl_route_adv_std": float(np.mean(sl_adv_stds)) if sl_adv_stds else 0.0,
                        "sl_num_routes_used": int(np.sum(sl_route_counts)) if sl_route_counts else 0,
                        "sl_coef": float(offline_cfg.get("sl_coef", offline_cfg.get("route_loss_coef", 0.10))),
                        "sl_ref_gate_mean": adv_info.get("ref_gate_mean", 0.0),
                        "sl_ref_gate_std": adv_info.get("ref_gate_std", 0.0),
                    }
                if profile_timing:
                    _sync_cuda(device)
                ppo_update_time_s = time.perf_counter() - ppo_start

                if route_bc_enabled and expert_buffer is not None:
                    route_bc_info = _run_route_bc_updates(
                        agent,
                        optimizer,
                        expert_buffer,
                        cfg,
                        device,
                        epoch,
                        coef=float(offline_cfg.get("route_bc_coef", offline_cfg.get("bc_coef", 1.0))),
                        updates=max(1, route_updates_per_epoch),
                        scaler=scaler,
                        amp_enabled=amp_enabled,
                    )
                    bc_info.update(route_bc_info)
                    offline_updates += int(route_bc_info.get("offline_updates", 0))

                reward_mean = float(batch.rewards[batch.valid].mean().detach().cpu().item()) if batch.valid.any() else 0.0
                train_summary = summarize_train_infos(batch.final_infos)
                rollout_timings = batch.timings

            loss_arr = np.asarray(losses or [(0.0, 0.0, 0.0)], dtype=float)
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
                    f"timing_reset={rollout_timings.get('rollout_reset_time_s', 0.0):.3f}s "
                    f"timing_model={rollout_timings.get('rollout_model_action_time_s', 0.0):.3f}s "
                    f"timing_env={rollout_timings.get('rollout_env_step_time_s', 0.0):.3f}s "
                    f"timing_ppo={ppo_update_time_s:.3f}s "
                    f"group_adv={_format_float(adv_info.get('group_adv_mean', 0.0))}/"
                    f"{_format_float(adv_info.get('group_adv_std', 0.0))} "
                    f"ref_adv={_format_float(adv_info.get('ref_adv_mean', 0.0))}/"
                    f"{_format_float(adv_info.get('ref_adv_std', 0.0))} "
                    f"bc={_format_float(bc_info.get('bc_loss', 0.0))} "
                    f"bc_acc={_format_float(bc_info.get('bc_accuracy', 0.0))} "
                    f"route_bc={_format_float(bc_info.get('route_bc_loss', 0.0))} "
                    f"sl={_format_float(sl_info.get('sl_route_loss', 0.0))} "
                    f"sl_ratio={_format_float(sl_info.get('sl_route_ratio_mean', 1.0))} "
                    f"sl_clip={_format_float(sl_info.get('sl_route_clip_frac', 0.0))}",
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
                eval_obj = eval_row.get("eval_avg_objective_distance_km")
                eval_fr = eval_row.get("eval_feasible_rate", 0.0)
                try:
                    eval_obj_f = float(eval_obj)
                    eval_fr_f = float(eval_fr)
                except (TypeError, ValueError):
                    eval_obj_f = float("nan")
                    eval_fr_f = 0.0
                if eval_row.get("eval_status") == "ok" and np.isfinite(eval_obj_f) and eval_fr_f > 0.0 and eval_obj_f < best_eval_objective:
                    best_eval_objective = eval_obj_f
                    best_eval_epoch = int(epoch)
                    best_path = ckpt_dir / "checkpoint_best.pt"
                    save_checkpoint(best_path, agent, optimizer, cfg, epoch, seed)
                    (ckpt_dir / "best_checkpoint.json").write_text(
                        json.dumps({"epoch": best_eval_epoch, "eval_avg_objective_distance_km": best_eval_objective, "eval_feasible_rate": eval_fr_f}, indent=2),
                        encoding="utf-8",
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
                    "train_mode": offline_method,
                    "mixed_precision": amp_enabled,
                    "use_gae": use_gae,
                    "gae_lambda": gae_lambda,
                    "bc_loss": bc_info.get("bc_loss", ""),
                    "bc_accuracy": bc_info.get("bc_accuracy", ""),
                    "bc_entropy": bc_info.get("bc_entropy", ""),
                    "bc_coef": bc_info.get("bc_coef", ""),
                    "bc_steps": bc_info.get("bc_steps", ""),
                    "route_bc_loss": bc_info.get("route_bc_loss", ""),
                    "route_bc_entropy": bc_info.get("route_bc_entropy", ""),
                    "route_bc_coef": bc_info.get("route_bc_coef", ""),
                    "route_bc_count": bc_info.get("route_bc_count", ""),
                    "route_bc_step_count": bc_info.get("route_bc_step_count", ""),
                    "route_bc_avg_route_len": bc_info.get("route_bc_avg_route_len", ""),
                    "sl_route_loss": sl_info.get("sl_route_loss", ""),
                    "sl_route_ratio_mean": sl_info.get("sl_route_ratio_mean", ""),
                    "sl_route_ratio_std": sl_info.get("sl_route_ratio_std", ""),
                    "sl_route_clip_frac": sl_info.get("sl_route_clip_frac", ""),
                    "sl_route_adv_mean": sl_info.get("sl_route_adv_mean", ""),
                    "sl_route_adv_std": sl_info.get("sl_route_adv_std", ""),
                    "sl_num_routes_used": sl_info.get("sl_num_routes_used", ""),
                    "sl_coef": sl_info.get("sl_coef", ""),
                    "sl_ref_gate_mean": sl_info.get("sl_ref_gate_mean", ""),
                    "sl_ref_gate_std": sl_info.get("sl_ref_gate_std", ""),
                    "offline_updates": offline_updates,
                    "group_adv_mean": adv_info.get("group_adv_mean", ""),
                    "group_adv_std": adv_info.get("group_adv_std", ""),
                    "ref_adv_mean": adv_info.get("ref_adv_mean", ""),
                    "ref_adv_std": adv_info.get("ref_adv_std", ""),
                    "aux_adv_mean": adv_info.get("aux_adv_mean", ""),
                    "aux_adv_std": adv_info.get("aux_adv_std", ""),
                    "route_adv_mean": adv_info.get("route_adv_mean", ""),
                    "route_adv_std": adv_info.get("route_adv_std", ""),
                    "best_eval_avg_objective_distance_km": best_eval_objective if np.isfinite(best_eval_objective) else "",
                    "best_eval_epoch": best_eval_epoch,
                    "pbrs_scale": pbrs_scale,
                    "initial_env_pool_time_s": initial_env_pool_time_s,
                    "rollout_reset_time_s": rollout_timings.get("rollout_reset_time_s", ""),
                    "rollout_stack_obs_time_s": rollout_timings.get("rollout_stack_obs_time_s", ""),
                    "rollout_model_action_time_s": rollout_timings.get("rollout_model_action_time_s", ""),
                    "rollout_env_step_time_s": rollout_timings.get("rollout_env_step_time_s", ""),
                    "rollout_interaction_time_s": rollout_timings.get("rollout_interaction_time_s", ""),
                    "rollout_total_time_s": rollout_timings.get("rollout_total_time_s", ""),
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
