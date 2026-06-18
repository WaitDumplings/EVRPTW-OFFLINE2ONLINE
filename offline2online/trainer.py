from __future__ import annotations

from collections import deque
from contextlib import nullcontext
import csv
from dataclasses import dataclass
import itertools
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .integrations.evrptw_db import configure_evrptw_db

EVRPTW_DB_ROOT = configure_evrptw_db()

from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.env_factory import make_terran_env
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.rollout import (
    collect_rollout,
    compute_returns,
    reset_envs,
    sample_actions,
    stack_observations,
    step_envs,
)
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.trainer import (
    _debug_log,
    _format_float,
    _slice_obs_by_env,
    _configure_dataset_reward_scale,
    build_pbrs_config,
    FixedDatasetInstancePool,
    evaluate_fixed_dataset,
    evaluate_policy_loss,
    make_envs,
    pbrs_scale_for_epoch,
    set_pbrs_reward_scale,
    summarize_train_infos,
)
from evrptw_core.io import iter_instances

from .models import Agent
from .offline_data import (
    ExpertReplayBuffer,
    build_expert_trajectories,
    compute_awbc_loss,
    compute_bc_loss,
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
    return False


def _is_frro_method(method: str) -> bool:
    return False


def _is_dapg_method(method: str) -> bool:
    return method in {"dapg"}


def _is_awbc_method(method: str) -> bool:
    return method in {"awbc", "awbc_ppo", "awbc-ppo", "advantage_weighted_bc", "advantage-weighted-bc"}


def _is_hard_method(method: str) -> bool:
    return False


def _is_hard_full_method(method: str) -> bool:
    return False


def _is_bc_aux_method(method: str) -> bool:
    return False


def _is_bafipo_method(method: str) -> bool:
    return False


def _is_gcbpo_method(method: str) -> bool:
    return False


def _is_solution_level_method(method: str) -> bool:
    return _is_sl_ppo_method(method) or _is_frro_method(method)


def _is_route_bc_method(method: str) -> bool:
    return False


def _requires_expert_routes(method: str) -> bool:
    return method in {"bc_ppo", "bc-ppo"} or _is_dapg_method(method) or _is_awbc_method(method)


def _offline_coef(offline_cfg: dict[str, Any], epoch: int) -> float:
    coef = float(offline_cfg.get("bc_coef", offline_cfg.get("dapg_bc_coef", 1.0)))
    decay = float(offline_cfg.get("bc_decay", offline_cfg.get("dapg_decay", 1.0)))
    min_coef = float(offline_cfg.get("min_bc_coef", 0.0))
    if decay != 1.0:
        coef *= decay ** max(int(epoch) - 1, 0)
    return max(coef, min_coef)


def _advantage_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("advantage", {}) or {})


def _group_advantage_enabled(cfg: dict[str, Any]) -> bool:
    return False


def _reference_advantage_enabled(cfg: dict[str, Any]) -> bool:
    return False


def _frro_enabled(cfg: dict[str, Any]) -> bool:
    return False


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


def _final_vehicle_array(final_infos: list[dict[str, Any]], num_envs: int, n_traj: int) -> np.ndarray:
    vehicle = np.full((num_envs, n_traj), np.nan, dtype=np.float64)
    for env_idx, info in enumerate(final_infos[:num_envs]):
        vehicle_arr = np.asarray(info.get("vehicle_count", []), dtype=np.float64).reshape(-1)
        limit = min(n_traj, vehicle_arr.size)
        if limit > 0:
            vehicle[env_idx, :limit] = vehicle_arr[:limit]
    return vehicle


def _distance_matrix_from_env(env) -> np.ndarray:
    candidates = [env, getattr(env, "unwrapped", None), getattr(env, "env", None)]
    current = env
    for _ in range(8):
        current = getattr(current, "env", None)
        if current is None:
            break
        candidates.extend([current, getattr(current, "unwrapped", None)])
    for obj in candidates:
        instance = getattr(obj, "instance", None) if obj is not None else None
        if instance is None:
            continue
        matrix = getattr(instance, "distance_matrix_km", None)
        if matrix is None and hasattr(instance, "raw"):
            matrix = instance.raw.get("distance_matrix_km")
        if matrix is not None:
            return np.asarray(matrix, dtype=np.float64)
    return np.asarray([], dtype=np.float64)


def _reward_scale_from_env(env) -> float:
    candidates = [env, getattr(env, "unwrapped", None), getattr(env, "env", None)]
    current = env
    for _ in range(8):
        current = getattr(current, "env", None)
        if current is None:
            break
        candidates.extend([current, getattr(current, "unwrapped", None)])
    for obj in candidates:
        if obj is None:
            continue
        scale = getattr(obj, "reward_distance_scale_km", None)
        if scale is not None:
            try:
                return max(float(scale), 1e-9)
            except (TypeError, ValueError):
                pass
    return 1.0


def _route_order_distance(order: Sequence[int], dist: np.ndarray) -> float:
    clean = [int(x) for x in order]
    if not clean:
        return 0.0
    total = float(dist[0, clean[0]])
    for u, v in zip(clean[:-1], clean[1:]):
        total += float(dist[int(u), int(v)])
    total += float(dist[clean[-1], 0])
    return total


def _two_opt_route(order: Sequence[int], dist: np.ndarray, *, max_passes: int) -> list[int]:
    route = [int(x) for x in order]
    if len(route) < 4:
        return route
    n = len(route)
    for _ in range(max(0, int(max_passes))):
        best_delta = -1e-12
        best_pair: tuple[int, int] | None = None
        for i in range(n - 1):
            a = 0 if i == 0 else route[i - 1]
            b = route[i]
            for j in range(i + 1, n):
                c = route[j]
                d = 0 if j == n - 1 else route[j + 1]
                delta = float(dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d])
                if delta < best_delta:
                    best_delta = delta
                    best_pair = (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        route[i : j + 1] = reversed(route[i : j + 1])
    return route


def _nearest_neighbor_route(customers: Sequence[int], dist: np.ndarray, start: int) -> list[int]:
    unvisited = set(int(c) for c in customers)
    current = int(start)
    order = [current]
    unvisited.remove(current)
    while unvisited:
        nxt = min(unvisited, key=lambda c: (float(dist[current, c]), int(c)))
        order.append(int(nxt))
        unvisited.remove(int(nxt))
        current = int(nxt)
    return order


def _load_reference_metrics(path: str | Path | None) -> dict[str, dict[str, float]]:
    ref_path = _resolve_path(path)
    if ref_path is None or not ref_path.exists():
        return {}
    out: dict[str, dict[str, float]] = {}
    with ref_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            instance_id = str(row.get("instance_id", ""))
            if not instance_id:
                continue
            try:
                objective = float(row.get("objective_distance_km", "nan"))
            except (TypeError, ValueError):
                objective = float("nan")
            try:
                vehicle = float(row.get("vehicle_count", "nan"))
            except (TypeError, ValueError):
                vehicle = float("nan")
            feasible_raw = str(row.get("feasible", "")).strip().lower()
            feasible = feasible_raw in {"true", "1", "yes", "y"} or feasible_raw == ""
            if feasible and np.isfinite(objective):
                out[instance_id] = {
                    "objective_distance_km": objective,
                    "vehicle_count": vehicle,
                }
    return out


def _tail_gap_stats(rows: list[dict[str, Any]], references: dict[str, dict[str, float]]) -> dict[str, Any]:
    gaps: list[float] = []
    vehicle_gaps: list[float] = []
    expert_k_gaps: dict[int, list[float]] = {k: [] for k in range(1, 5)}
    expert_k_objs: dict[int, list[float]] = {k: [] for k in range(1, 5)}
    policy_k_gaps: dict[int, list[float]] = {k: [] for k in range(1, 5)}
    policy_k_objs: dict[int, list[float]] = {k: [] for k in range(1, 5)}
    for row in rows:
        ref = references.get(str(row.get("instance_id", "")))
        if ref is None:
            continue
        try:
            policy_obj = float(row.get("objective_distance_km", float("nan")))
            expert_obj = float(ref.get("objective_distance_km", float("nan")))
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(policy_obj) and np.isfinite(expert_obj)):
            continue
        gap = policy_obj - expert_obj
        gaps.append(float(gap))
        try:
            policy_vehicle = float(row.get("vehicle_count", float("nan")))
            expert_vehicle = float(ref.get("vehicle_count", float("nan")))
        except (TypeError, ValueError):
            policy_vehicle = float("nan")
            expert_vehicle = float("nan")
        if np.isfinite(policy_vehicle) and np.isfinite(expert_vehicle):
            vehicle_gaps.append(float(policy_vehicle - expert_vehicle))
            expert_bucket = min(max(int(round(expert_vehicle)), 1), 4)
            policy_bucket = min(max(int(round(policy_vehicle)), 1), 4)
            expert_k_gaps[expert_bucket].append(float(gap))
            expert_k_objs[expert_bucket].append(float(policy_obj))
            policy_k_gaps[policy_bucket].append(float(gap))
            policy_k_objs[policy_bucket].append(float(policy_obj))

    def _q(values: list[float], q: float) -> float:
        arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
        return float(np.quantile(arr, q)) if arr.size else float("nan")

    gap_arr = np.asarray([v for v in gaps if np.isfinite(v)], dtype=np.float64)
    vehicle_arr = np.asarray([v for v in vehicle_gaps if np.isfinite(v)], dtype=np.float64)
    top10 = np.sort(gap_arr)[-10:] if gap_arr.size else np.asarray([], dtype=np.float64)
    out = {
        "eval_gap_mean": float(np.mean(gap_arr)) if gap_arr.size else float("nan"),
        "eval_gap_median": _q(gaps, 0.50),
        "eval_gap_p75": _q(gaps, 0.75),
        "eval_gap_p90": _q(gaps, 0.90),
        "eval_gap_p95": _q(gaps, 0.95),
        "eval_gap_p99": _q(gaps, 0.99),
        "eval_gap_gt50_count": int(np.sum(gap_arr > 50.0)) if gap_arr.size else 0,
        "eval_gap_gt100_count": int(np.sum(gap_arr > 100.0)) if gap_arr.size else 0,
        "eval_top10_hard_gap_mean": float(np.mean(top10)) if top10.size else float("nan"),
        "eval_vehicle_gap_mean": float(np.mean(vehicle_arr)) if vehicle_arr.size else float("nan"),
        "eval_vehicle_gap_median": _q(vehicle_gaps, 0.50),
        "eval_vehicle_gap_p90": _q(vehicle_gaps, 0.90),
        "eval_vehicle_gap_p95": _q(vehicle_gaps, 0.95),
        "eval_vehicle_gap_gt0_count": int(np.sum(vehicle_arr > 0.0)) if vehicle_arr.size else 0,
    }
    for k in range(1, 5):
        expert_gap_arr = np.asarray([v for v in expert_k_gaps[k] if np.isfinite(v)], dtype=np.float64)
        expert_obj_arr = np.asarray([v for v in expert_k_objs[k] if np.isfinite(v)], dtype=np.float64)
        policy_gap_arr = np.asarray([v for v in policy_k_gaps[k] if np.isfinite(v)], dtype=np.float64)
        policy_obj_arr = np.asarray([v for v in policy_k_objs[k] if np.isfinite(v)], dtype=np.float64)
        out[f"eval_gap_expertK{k}_mean"] = float(np.mean(expert_gap_arr)) if expert_gap_arr.size else float("nan")
        out[f"eval_obj_expertK{k}_mean"] = float(np.mean(expert_obj_arr)) if expert_obj_arr.size else float("nan")
        out[f"eval_gap_expertK{k}_count"] = int(expert_gap_arr.size)
        out[f"eval_gap_policyK{k}_mean"] = float(np.mean(policy_gap_arr)) if policy_gap_arr.size else float("nan")
        out[f"eval_obj_policyK{k}_mean"] = float(np.mean(policy_obj_arr)) if policy_obj_arr.size else float("nan")
        out[f"eval_gap_policyK{k}_count"] = int(policy_gap_arr.size)
    return out


class HapsPrioritySampler:
    def __init__(
        self,
        base_pool: Any,
        references: dict[str, dict[str, float]],
        *,
        batch_size: int,
        seed: int,
        cfg: dict[str, Any],
    ) -> None:
        offline_cfg = cfg.get("offline", {}) or {}
        self.instances = list(getattr(base_pool, "instances"))
        self.references = references
        self.instance_to_idx = {
            str(getattr(instance, "instance_id")): idx
            for idx, instance in enumerate(self.instances)
            if getattr(instance, "instance_id", None) is not None
        }
        self.rng = np.random.default_rng(seed)
        self.batch_size = max(1, int(batch_size))
        self.mix_rho = float(offline_cfg.get("priority_mix_rho", 0.50))
        self.alpha = float(offline_cfg.get("priority_alpha", 0.70))
        self.priority_min = float(offline_cfg.get("priority_min", 0.05))
        self.beta = float(offline_cfg.get("priority_update_beta", 0.20))
        self.gap_scale = max(float(offline_cfg.get("gap_scale", 0.20)), 1e-8)
        self.vehicle_gap_scale = max(float(offline_cfg.get("vehicle_gap_scale", 1.0)), 1e-8)
        self.obj_weight = float(offline_cfg.get("priority_obj_weight", 0.75))
        self.vehicle_weight = float(offline_cfg.get("priority_vehicle_weight", 0.25))
        self.stale_window = max(float(offline_cfg.get("stale_window", 100)), 1e-8)
        self.stale_bonus_weight = float(offline_cfg.get("stale_bonus_weight", 0.05))
        init_priority = float(offline_cfg.get("priority_init", 1.0))
        n = len(self.instances)
        self.priority = np.full(n, init_priority, dtype=np.float64)
        self.gap_ema = np.full(n, init_priority, dtype=np.float64)
        self.vehicle_gap_ema = np.zeros(n, dtype=np.float64)
        self.last_update_epoch = np.zeros(n, dtype=np.int64)
        self.num_updates = np.zeros(n, dtype=np.int64)
        self.current_epoch = 0
        self.sample_count = 0
        self.region_pool_status = f"haps_priority_fixed_dataset:{getattr(base_pool, 'dataset_path', '')}"
        self._buffer: deque[int] = deque()
        self._last_sample_indices: np.ndarray = np.asarray([], dtype=np.int64)
        self._last_sample_stats: dict[str, Any] = {}

    def begin_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)
        self._buffer.clear()

    def _sample_priority(self) -> np.ndarray:
        stale = np.clip((float(self.current_epoch) - self.last_update_epoch.astype(np.float64)) / self.stale_window, 0.0, 1.0)
        return self.priority + self.stale_bonus_weight * stale

    def _refill(self) -> None:
        n = len(self.instances)
        if n <= 0:
            raise RuntimeError("HAPS priority sampler has no instances")
        batch_size = min(self.batch_size, n)
        rho = float(np.clip(self.mix_rho, 0.0, 1.0))
        uniform_count = int(batch_size * (1.0 - rho))
        priority_count = batch_size - uniform_count
        all_indices = np.arange(n, dtype=np.int64)
        uniform = (
            self.rng.choice(all_indices, size=min(uniform_count, n), replace=False)
            if uniform_count > 0
            else np.asarray([], dtype=np.int64)
        )
        available = np.setdiff1d(all_indices, uniform, assume_unique=False)
        priority_count = min(priority_count, available.size)
        if priority_count > 0:
            sample_priority = self._sample_priority()[available]
            weights = np.power(np.maximum(sample_priority, 0.0), max(self.alpha, 0.0))
            if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
                probs = None
            else:
                probs = weights / float(weights.sum())
            weighted = self.rng.choice(available, size=priority_count, replace=False, p=probs)
        else:
            weighted = np.asarray([], dtype=np.int64)
        selected = np.concatenate([uniform, weighted]).astype(np.int64)
        self.rng.shuffle(selected)
        stale_bonus = self._sample_priority() - self.priority
        self._last_sample_indices = selected
        self._last_sample_stats = {
            "sampled_priority_mean": float(np.mean(self.priority[selected])) if selected.size else 0.0,
            "priority_uniform_fraction": float(uniform.size / max(selected.size, 1)),
            "priority_weighted_fraction": float(weighted.size / max(selected.size, 1)),
            "unique_instance_ratio": float(np.unique(selected).size / max(selected.size, 1)),
            "stale_bonus_mean": float(np.mean(stale_bonus[selected])) if selected.size else 0.0,
        }
        self._buffer = deque(int(i) for i in selected)

    def sample(self):
        if not self._buffer:
            self._refill()
        idx = int(self._buffer.popleft())
        self.sample_count += 1
        return self.instances[idx]

    def update_from_rollout(self, batch, *, epoch: int) -> dict[str, Any]:
        if batch is None:
            return self.stats()
        num_envs = int(batch.actions.size(1))
        n_traj = int(batch.actions.size(2))
        objective, success, _ = _final_info_arrays(batch.final_infos, num_envs, n_traj)
        vehicle = _final_vehicle_array(batch.final_infos, num_envs, n_traj)
        instance_ids = list(getattr(batch, "instance_ids", []) or [])
        gap_scores: list[float] = []
        vehicle_scores: list[float] = []
        raw_gaps: list[float] = []
        raw_vehicle_gaps: list[float] = []
        for env_idx in range(num_envs):
            instance_id = str(instance_ids[env_idx]) if env_idx < len(instance_ids) else ""
            idx = self.instance_to_idx.get(instance_id)
            ref = self.references.get(instance_id)
            if idx is None or ref is None:
                continue
            obj_row = objective[env_idx]
            success_row = success[env_idx]
            valid_obj = success_row & np.isfinite(obj_row)
            if not np.any(valid_obj):
                continue
            policy_obj = float(np.mean(obj_row[valid_obj]))
            expert_obj = float(ref.get("objective_distance_km", float("nan")))
            if not np.isfinite(expert_obj) or expert_obj <= 0.0:
                continue
            gap = max((policy_obj - expert_obj) / (expert_obj + 1e-8), 0.0)
            obj_score = float(np.clip(gap / self.gap_scale, 0.0, 1.0))
            vehicle_row = vehicle[env_idx]
            valid_vehicle = valid_obj & np.isfinite(vehicle_row)
            if np.any(valid_vehicle) and np.isfinite(ref.get("vehicle_count", float("nan"))):
                policy_vehicle = float(np.mean(vehicle_row[valid_vehicle]))
                vehicle_gap = max(policy_vehicle - float(ref["vehicle_count"]), 0.0)
            else:
                vehicle_gap = 0.0
            vehicle_score = float(np.clip(vehicle_gap / self.vehicle_gap_scale, 0.0, 1.0))
            hard_score = self.obj_weight * obj_score + self.vehicle_weight * vehicle_score
            self.priority[idx] = max((1.0 - self.beta) * self.priority[idx] + self.beta * hard_score, self.priority_min)
            self.gap_ema[idx] = (1.0 - self.beta) * self.gap_ema[idx] + self.beta * gap
            self.vehicle_gap_ema[idx] = (1.0 - self.beta) * self.vehicle_gap_ema[idx] + self.beta * vehicle_gap
            self.last_update_epoch[idx] = int(epoch)
            self.num_updates[idx] += 1
            gap_scores.append(obj_score)
            vehicle_scores.append(vehicle_score)
            raw_gaps.append(gap)
            raw_vehicle_gaps.append(vehicle_gap)
        stats = self.stats()
        stats.update(
            {
                "gap_score_mean": float(np.mean(gap_scores)) if gap_scores else 0.0,
                "gap_score_std": float(np.std(gap_scores)) if gap_scores else 0.0,
                "vehicle_score_mean": float(np.mean(vehicle_scores)) if vehicle_scores else 0.0,
                "vehicle_score_std": float(np.std(vehicle_scores)) if vehicle_scores else 0.0,
                "sampled_gap_mean": float(np.mean(raw_gaps)) if raw_gaps else 0.0,
                "sampled_vehicle_gap_mean": float(np.mean(raw_vehicle_gaps)) if raw_vehicle_gaps else 0.0,
            }
        )
        return stats

    def stats(self) -> dict[str, Any]:
        base = {
            "priority_mean": float(np.mean(self.priority)) if self.priority.size else 0.0,
            "priority_std": float(np.std(self.priority)) if self.priority.size else 0.0,
            "priority_min": float(np.min(self.priority)) if self.priority.size else 0.0,
            "priority_max": float(np.max(self.priority)) if self.priority.size else 0.0,
            "num_priority_updates_mean": float(np.mean(self.num_updates)) if self.num_updates.size else 0.0,
            "gap_score_mean": 0.0,
            "gap_score_std": 0.0,
            "vehicle_score_mean": 0.0,
            "vehicle_score_std": 0.0,
            "sampled_gap_mean": 0.0,
            "sampled_vehicle_gap_mean": 0.0,
        }
        base.update(self._last_sample_stats)
        for key in (
            "sampled_priority_mean",
            "priority_uniform_fraction",
            "priority_weighted_fraction",
            "unique_instance_ratio",
            "stale_bonus_mean",
        ):
            base.setdefault(key, 0.0)
        return base


def _resolve_path(path: str | Path | None) -> Path | None:
    if path is None or str(path) == "":
        return None
    out = Path(path)
    return out if out.is_absolute() else REPO_ROOT / out


def _validate_dataset_metadata(
    path: str | Path | None,
    *,
    num_customers: int,
    num_charging_stations: int,
    label: str,
) -> None:
    dataset_path = _resolve_path(path)
    if dataset_path is None:
        return
    if not dataset_path.exists():
        raise FileNotFoundError(f"{label} dataset path does not exist: {dataset_path}")
    metadata_path = dataset_path / "metadata.json" if dataset_path.is_dir() else dataset_path.parent / "metadata.json"
    if not metadata_path.exists():
        return
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    meta_customers = metadata.get("num_customers")
    meta_cs = metadata.get("num_charging_stations")
    if meta_customers is not None and int(meta_customers) != int(num_customers):
        raise ValueError(
            f"{label} dataset metadata mismatch at {metadata_path}: "
            f"config num_customers={num_customers}, metadata num_customers={meta_customers}"
        )
    if meta_cs is not None and int(meta_cs) != int(num_charging_stations):
        raise ValueError(
            f"{label} dataset metadata mismatch at {metadata_path}: "
            f"config num_charging_stations={num_charging_stations}, "
            f"metadata num_charging_stations={meta_cs}"
        )


def _load_agent_checkpoint(
    agent: Agent,
    path: str | Path | None,
    device: str | torch.device,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    ckpt_path = _resolve_path(path)
    if ckpt_path is None:
        return {}
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("agent_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )
    else:
        state_dict = checkpoint
        checkpoint = {}
    result = agent.load_state_dict(state_dict, strict=strict)
    return {
        "checkpoint_path": str(ckpt_path),
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "seed": checkpoint.get("seed") if isinstance(checkpoint, dict) else None,
        "missing_keys": list(getattr(result, "missing_keys", [])),
        "unexpected_keys": list(getattr(result, "unexpected_keys", [])),
    }


def _eval_instance_batches(
    eval_path: Path,
    num_customers: int,
    num_charging_stations: int,
    batch_size: int,
    limit: int | None = None,
    num_batches_limit: int | None = None,
):
    max_count = None if limit is None else int(limit)
    if num_batches_limit is not None:
        by_batches = max(1, int(batch_size)) * int(num_batches_limit)
        max_count = by_batches if max_count is None else min(max_count, by_batches)
    batch = []
    seen = 0
    for instance in iter_instances(
        eval_path,
        num_customers=num_customers,
        num_charging_stations=num_charging_stations,
    ):
        if max_count is not None and seen >= max_count:
            break
        batch.append(instance)
        seen += 1
        if len(batch) >= max(1, int(batch_size)):
            yield batch
            batch = []
    if batch:
        yield batch


def _empty_eval_row(
    *,
    n_traj: int,
    batch_size: int,
    num_batches: int,
    decode_mode: str,
    eval_info_level: str,
    eval_save_routes: bool,
    status: str,
) -> dict[str, Any]:
    return {
        "eval_num_instances": 0,
        "eval_n_traj": n_traj,
        "eval_batch_size": batch_size,
        "eval_num_batches": num_batches,
        "eval_decode_mode": decode_mode,
        "eval_info_level": eval_info_level,
        "eval_save_routes": eval_save_routes,
        "eval_feasible_rate": np.nan,
        "eval_traj_feasible_rate": np.nan,
        "eval_avg_feasible_traj_count": np.nan,
        "eval_avg_objective_distance_km": np.nan,
        "eval_avg_min_objective_distance_km": np.nan,
        "eval_avg_median_objective_distance_km": np.nan,
        "eval_avg_vehicle_count": np.nan,
        "eval_avg_min_vehicle_count": np.nan,
        "eval_avg_median_vehicle_count": np.nan,
        "eval_avg_runtime_s": np.nan,
        "eval_status": status,
        "eval_gap_mean": np.nan,
        "eval_gap_median": np.nan,
        "eval_gap_p75": np.nan,
        "eval_gap_p90": np.nan,
        "eval_gap_p95": np.nan,
        "eval_gap_p99": np.nan,
        "eval_gap_gt50_count": 0,
        "eval_gap_gt100_count": 0,
        "eval_top10_hard_gap_mean": np.nan,
        "eval_vehicle_gap_mean": np.nan,
        "eval_vehicle_gap_median": np.nan,
        "eval_vehicle_gap_p90": np.nan,
        "eval_vehicle_gap_p95": np.nan,
        "eval_vehicle_gap_gt0_count": 0,
        **{
            key: value
            for k in range(1, 5)
            for key, value in {
                f"eval_gap_expertK{k}_mean": np.nan,
                f"eval_obj_expertK{k}_mean": np.nan,
                f"eval_gap_expertK{k}_count": 0,
                f"eval_gap_policyK{k}_mean": np.nan,
                f"eval_obj_policyK{k}_mean": np.nan,
                f"eval_gap_policyK{k}_count": 0,
            }.items()
        },
    }


def _select_min_median_trajectory_stats(info: dict[str, Any]) -> dict[str, Any]:
    objective = np.asarray(info.get("objective_distance_km", []), dtype=np.float64).reshape(-1)
    success = np.asarray(info.get("success", []), dtype=bool).reshape(-1)
    vehicle = np.asarray(info.get("vehicle_count", []), dtype=np.float64).reshape(-1)
    served = np.asarray(info.get("served_customers", []), dtype=np.float64).reshape(-1)

    finite_obj = np.isfinite(objective)
    success_mask = np.zeros(objective.shape, dtype=bool)
    success_mask[: min(success.size, objective.size)] = success[: min(success.size, objective.size)]
    candidate_mask = success_mask & finite_obj
    feasible = bool(np.any(candidate_mask))
    if not feasible:
        if served.size:
            served_pad = np.full(objective.shape, np.nan, dtype=np.float64)
            served_pad[: min(served.size, objective.size)] = served[: min(served.size, objective.size)]
            finite_served = np.isfinite(served_pad)
            if np.any(finite_served & finite_obj):
                max_served = float(np.nanmax(served_pad[finite_served & finite_obj]))
                candidate_mask = finite_obj & (served_pad == max_served)
        if not np.any(candidate_mask):
            candidate_mask = finite_obj

    candidate_idx = np.where(candidate_mask)[0]
    if candidate_idx.size == 0:
        return {
            "feasible": False,
            "traj_feasible_rate": float(np.mean(success)) if success.size else np.nan,
            "feasible_traj_count": 0.0,
            "objective_distance_km": np.nan,
            "min_objective_distance_km": np.nan,
            "median_objective_distance_km": np.nan,
            "vehicle_count": np.nan,
            "min_vehicle_count": np.nan,
            "median_vehicle_count": np.nan,
        }

    candidate_obj = objective[candidate_idx]
    order = np.argsort(candidate_obj)
    min_idx = int(candidate_idx[order[0]])
    median_idx = int(candidate_idx[order[len(order) // 2]])
    median_obj = float(np.median(candidate_obj))

    def _vehicle_at(idx: int) -> float:
        if 0 <= idx < vehicle.size and np.isfinite(vehicle[idx]):
            return float(vehicle[idx])
        return np.nan

    return {
        "feasible": feasible,
        "traj_feasible_rate": float(np.mean(success)) if success.size else np.nan,
        "feasible_traj_count": float(np.sum(success)) if success.size else np.nan,
        "objective_distance_km": float(objective[min_idx]),
        "min_objective_distance_km": float(objective[min_idx]),
        "median_objective_distance_km": median_obj,
        "vehicle_count": _vehicle_at(min_idx),
        "min_vehicle_count": _vehicle_at(min_idx),
        "median_vehicle_count": _vehicle_at(median_idx),
    }


def _rollout_eval_batch_min_median(
    agent: Agent,
    envs,
    decode_mode: str,
    max_steps: int,
    device: str | torch.device,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    if not envs:
        return []
    observations, infos = reset_envs(envs, seed=seed)
    n_traj = int(envs[0].unwrapped.n_traj)
    done = np.zeros((len(envs), n_traj), dtype=bool)
    start = time.perf_counter()
    for _ in range(int(max_steps)):
        obs_batch = stack_observations(observations)
        with torch.no_grad():
            actions, _, _, _, _ = sample_actions(agent, obs_batch, decode_mode=decode_mode, device=device)
        action_np = actions.detach().cpu().numpy().astype(np.int64)
        observations, _, step_done, infos = step_envs(envs, action_np)
        done = done | step_done
        if done.all():
            break
    elapsed = time.perf_counter() - start
    per_instance_runtime = float(elapsed) / max(len(envs), 1)
    rows = []
    for info in infos:
        row = _select_min_median_trajectory_stats(info)
        row["runtime_s"] = per_instance_runtime
        row["batch_runtime_s"] = float(elapsed)
        rows.append(row)
    return rows


def evaluate_fixed_dataset(agent: Agent, cfg: dict[str, Any], seed: int, epoch: int, device: str | torch.device) -> dict[str, Any]:
    eval_cfg = cfg.get("evaluation", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    num_customers = int(data_cfg.get("num_customers", 15))
    num_cs = int(data_cfg.get("num_charging_stations", 3))
    eval_path = _resolve_path(eval_cfg.get("eval_path"))
    n_traj = int(eval_cfg.get("eval_n_traj", 8))
    decode_mode = str(eval_cfg.get("eval_decode_mode", "sample"))
    max_steps = int(eval_cfg.get("eval_max_steps", 128))
    limit = eval_cfg.get("eval_limit", None)
    batch_size = max(1, int(eval_cfg.get("eval_batch_size", 1)))
    num_batches_limit = eval_cfg.get("eval_num_batches", None)
    eval_save_routes = bool(eval_cfg.get("eval_save_routes", False))
    eval_info_level = str(eval_cfg.get("eval_info_level", "light"))
    reference_path = (
        eval_cfg.get("gurobi_summary_path")
        or eval_cfg.get("eval_gurobi_summary")
        or eval_cfg.get("reference_summary_path")
        or eval_cfg.get("expert_solution_path")
    )
    reference_metrics = _load_reference_metrics(reference_path)
    if eval_path is None or not eval_path.exists():
        return _empty_eval_row(
            n_traj=n_traj,
            batch_size=batch_size,
            num_batches=0,
            decode_mode=decode_mode,
            eval_info_level=eval_info_level,
            eval_save_routes=eval_save_routes,
            status=f"missing_eval_path:{eval_path}",
        )

    was_training = agent.training
    agent.eval()
    rows: list[dict[str, Any]] = []
    num_batches = 0
    seen_before_batch = 0
    for instances in _eval_instance_batches(eval_path, num_customers, num_cs, batch_size, limit, num_batches_limit):
        eval_env_cfg = dict(cfg.get("env", {}) or {})
        if bool(eval_env_cfg.get("use_fast_env", True)):
            eval_env_cfg["info_level"] = "full" if eval_save_routes else eval_info_level
        envs = [make_terran_env(instance=instance, n_traj=n_traj, **eval_env_cfg) for instance in instances]
        batch_rows = _rollout_eval_batch_min_median(
            agent,
            envs,
            decode_mode=decode_mode,
            max_steps=max_steps,
            device=device,
            seed=seed + epoch * 1_000_000 + seen_before_batch,
        )
        for instance, row in zip(instances, batch_rows):
            row["instance_id"] = instance.instance_id
        rows.extend(batch_rows)
        num_batches += 1
        seen_before_batch += len(instances)

    if was_training:
        agent.train()
    if not rows:
        return _empty_eval_row(
            n_traj=n_traj,
            batch_size=batch_size,
            num_batches=0,
            decode_mode=decode_mode,
            eval_info_level=eval_info_level,
            eval_save_routes=eval_save_routes,
            status=f"no_instances:{eval_path}",
        )

    feasible_rows = [row for row in rows if row["feasible"]]
    out = {
        "eval_num_instances": len(rows),
        "eval_n_traj": n_traj,
        "eval_batch_size": batch_size,
        "eval_num_batches": num_batches,
        "eval_decode_mode": decode_mode,
        "eval_info_level": eval_info_level,
        "eval_save_routes": eval_save_routes,
        "eval_feasible_rate": float(np.mean([row["feasible"] for row in rows])),
        "eval_traj_feasible_rate": float(np.nanmean([row["traj_feasible_rate"] for row in rows])),
        "eval_avg_feasible_traj_count": float(np.nanmean([row["feasible_traj_count"] for row in rows])),
        "eval_avg_objective_distance_km": float(np.nanmean([row["objective_distance_km"] for row in feasible_rows])) if feasible_rows else np.nan,
        "eval_avg_min_objective_distance_km": float(np.nanmean([row["min_objective_distance_km"] for row in feasible_rows])) if feasible_rows else np.nan,
        "eval_avg_median_objective_distance_km": float(np.nanmean([row["median_objective_distance_km"] for row in feasible_rows])) if feasible_rows else np.nan,
        "eval_avg_vehicle_count": float(np.nanmean([row["vehicle_count"] for row in feasible_rows])) if feasible_rows else np.nan,
        "eval_avg_min_vehicle_count": float(np.nanmean([row["min_vehicle_count"] for row in feasible_rows])) if feasible_rows else np.nan,
        "eval_avg_median_vehicle_count": float(np.nanmean([row["median_vehicle_count"] for row in feasible_rows])) if feasible_rows else np.nan,
        "eval_avg_runtime_s": float(np.nanmean([row["runtime_s"] for row in rows])),
        "eval_status": "ok",
    }
    if reference_metrics:
        out.update(_tail_gap_stats(rows, reference_metrics))
    return out


def _policy_mean_successful_objective(batch) -> float | None:
    num_envs = int(batch.actions.size(1))
    n_traj = int(batch.actions.size(2))
    objective, success, _ = _final_info_arrays(batch.final_infos, num_envs, n_traj)
    successful = objective[success & np.isfinite(objective)]
    if successful.size == 0:
        return None
    return float(np.mean(successful))


def _update_policy_best_objectives(
    policy_best_objectives: dict[str, float],
    batch,
    envs,
) -> None:
    num_envs = int(batch.actions.size(1))
    n_traj = int(batch.actions.size(2))
    objective, success, _ = _final_info_arrays(batch.final_infos, num_envs, n_traj)
    for env_idx, env in enumerate(envs[:num_envs]):
        instance_id = _env_instance_id(env)
        if instance_id is None:
            continue
        successful = objective[env_idx][success[env_idx] & np.isfinite(objective[env_idx])]
        if successful.size == 0:
            continue
        current_best = float(np.min(successful))
        previous_best = policy_best_objectives.get(instance_id)
        if previous_best is None or current_best < previous_best:
            policy_best_objectives[instance_id] = current_best


def _finite_mean_std(value: np.ndarray) -> tuple[float, float]:
    finite = value[np.isfinite(value)]
    if finite.size == 0:
        return 0.0, 0.0
    return float(finite.mean()), float(finite.std())


def _objective_baseline(values: np.ndarray, mode: str) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    mode = str(mode).lower()
    if mode in {"best", "min", "minimum"}:
        return float(np.min(finite))
    if mode in {"median", "p50"}:
        return float(np.median(finite))
    return float(np.mean(finite))


@dataclass
class FrroExpertCandidate:
    env_idx: int
    observations: list[dict[str, np.ndarray]]
    actions: list[int]
    advantage: float
    gate: float
    old_mean_logprob: float = 0.0


@dataclass
class BafipoIncumbentCandidate:
    env_idx: int
    observations: list[dict[str, np.ndarray]]
    actions: list[int]
    objective: float
    old_mean_logprob: float = 0.0


@dataclass(frozen=True)
class BafipoPreferencePair:
    env_idx: int
    pos_kind: str
    pos_traj: int
    neg_kind: str
    neg_traj: int
    old_delta: float
    weight: float
    incumbent_pair: bool


@dataclass
class GcbpoBranchCandidate:
    env_idx: int
    observations: list[dict[str, np.ndarray]]
    actions: list[int]
    objective: float
    prefix_len: int
    prefix_weight: float
    old_mean_logprob: float = 0.0


@dataclass(frozen=True)
class GcbpoPreferencePair:
    env_idx: int
    branch_idx: int
    neg_traj: int
    old_delta: float
    weight: float
    strong: bool


def _frro_improvement_stats(
    objective_row: np.ndarray,
    success_row: np.ndarray,
    ref_obj: float,
    adv_cfg: dict[str, Any],
) -> tuple[float, float, float] | None:
    succ_obj = objective_row[success_row & np.isfinite(objective_row)]
    if succ_obj.size == 0 or not np.isfinite(ref_obj) or ref_obj <= 0.0:
        return None
    base_obj = _objective_baseline(succ_obj, str(adv_cfg.get("frro_gap_baseline", "mean")))
    if not np.isfinite(base_obj):
        return None
    remaining_gap = float(base_obj) - float(ref_obj)
    std_floor = max(float(adv_cfg.get("frro_std_floor", 5.0)), 0.0)
    sigma = max(float(np.std(succ_obj)), std_floor)
    gap_scale = max(float(adv_cfg.get("frro_gap_scale_coef", 1.0)), 0.0) * max(remaining_gap, 0.0)
    gap_floor = max(float(adv_cfg.get("frro_gap_floor_ratio", 0.01)), 0.0) * float(ref_obj)
    scale = max(sigma, gap_scale, gap_floor, 1e-8)
    return float(base_obj), remaining_gap, scale


def _frro_expert_gate(
    *,
    remaining_gap: float,
    ref_obj: float,
    current_success_objectives: np.ndarray,
    memory_obj: float | None,
    adv_cfg: dict[str, Any],
) -> tuple[float, bool, float | None]:
    quality_eta = max(float(adv_cfg.get("frro_quality_gate_eta", 0.05)), 1e-8)
    quality_gate = float(np.clip(remaining_gap / (quality_eta * float(ref_obj) + 1e-8), 0.0, 1.0))
    best_known = np.inf
    if bool(adv_cfg.get("frro_use_current_falsification", True)) and current_success_objectives.size > 0:
        best_known = min(best_known, float(np.min(current_success_objectives)))
    if (
        bool(adv_cfg.get("frro_use_memory_falsification", True))
        and memory_obj is not None
        and np.isfinite(memory_obj)
        and memory_obj > 0.0
    ):
        best_known = min(best_known, float(memory_obj))

    best_gap_ratio: float | None = None
    memory_gate = 1.0
    falsified = False
    if np.isfinite(best_known):
        margin = max(float(adv_cfg.get("frro_falsification_margin", 0.005)), 0.0)
        eta = max(float(adv_cfg.get("frro_falsification_eta", 0.05)), 1e-8)
        target_obj = float(ref_obj) * (1.0 - margin)
        gate_gap = (best_known - target_obj) / max(float(ref_obj), 1e-8)
        memory_gate = float(np.clip(gate_gap / eta, 0.0, 1.0))
        best_gap_ratio = (best_known - float(ref_obj)) / max(float(ref_obj), 1e-8)
        falsified = memory_gate <= 1e-6
    return quality_gate * memory_gate, falsified, best_gap_ratio


def _expert_route_mean_logprobs(
    agent: Agent,
    candidates: list[Any],
    device: str | torch.device,
    chunk_size: int,
) -> torch.Tensor:
    if not candidates:
        return torch.empty(0, dtype=torch.float32, device=device)
    route_sums: torch.Tensor | None = None
    route_lens: torch.Tensor | None = None
    observations: list[dict[str, np.ndarray]] = []
    actions: list[int] = []
    route_ids: list[int] = []
    for route_idx, candidate in enumerate(candidates):
        for obs, action in zip(candidate.observations, candidate.actions):
            observations.append(obs)
            actions.append(int(action))
            route_ids.append(route_idx)

    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(actions), chunk_size):
        end = min(start + chunk_size, len(actions))
        obs_batch = stack_observations(observations[start:end])
        action_tensor = torch.as_tensor(np.asarray(actions[start:end], dtype=np.int64)[:, None], dtype=torch.long, device=device)
        route_tensor = torch.as_tensor(np.asarray(route_ids[start:end], dtype=np.int64), dtype=torch.long, device=device)
        _, logprob, _, _ = agent.get_action_and_value(obs_batch, action=action_tensor)
        logprob_flat = logprob.reshape(-1)
        if route_sums is None:
            route_sums = torch.zeros(len(candidates), dtype=logprob_flat.dtype, device=logprob_flat.device)
            route_lens = torch.zeros_like(route_sums)
        route_sums.scatter_add_(0, route_tensor, logprob_flat)
        route_lens.scatter_add_(0, route_tensor, torch.ones_like(logprob_flat))
    assert route_sums is not None and route_lens is not None
    return route_sums / route_lens.clamp_min(1.0)


def _prepare_frro_expert_candidates(
    agent: Agent,
    batch,
    cfg: dict[str, Any],
    envs,
    expert_buffer: ExpertReplayBuffer | None,
    policy_best_objectives: dict[str, float] | None,
    device: str | torch.device,
) -> tuple[list[FrroExpertCandidate], dict[str, float]]:
    adv_cfg = _advantage_config(cfg)
    if not _frro_enabled(cfg) or expert_buffer is None or not bool(adv_cfg.get("frro_use_expert_candidate", True)):
        return [], {}
    num_envs = int(batch.actions.size(1))
    n_traj = int(batch.actions.size(2))
    objective, success, _ = _final_info_arrays(batch.final_infos, num_envs, n_traj)
    adv_clip = float(adv_cfg.get("frro_clip", 2.0))
    expert_weight = float(adv_cfg.get("frro_expert_candidate_weight", 2.0))
    candidates: list[FrroExpertCandidate] = []
    best_gap_ratios: list[float] = []
    for env_idx, env in enumerate(envs[:num_envs]):
        instance_id = _env_instance_id(env)
        traj = expert_buffer.trajectory_for_instance(instance_id)
        if traj is None or traj.length <= 0:
            continue
        ref_obj = float(traj.objective_distance_km)
        stats = _frro_improvement_stats(objective[env_idx], success[env_idx], ref_obj, adv_cfg)
        if stats is None:
            continue
        base_obj, remaining_gap, scale = stats
        del base_obj
        expert_adv = float(np.clip((remaining_gap / scale), -adv_clip, adv_clip))
        current_success = objective[env_idx][success[env_idx] & np.isfinite(objective[env_idx])]
        memory_obj = policy_best_objectives.get(instance_id) if policy_best_objectives is not None and instance_id is not None else None
        gate, _, best_gap = _frro_expert_gate(
            remaining_gap=remaining_gap,
            ref_obj=ref_obj,
            current_success_objectives=current_success,
            memory_obj=memory_obj,
            adv_cfg=adv_cfg,
        )
        if best_gap is not None:
            best_gap_ratios.append(best_gap)
        used_adv = expert_weight * gate * expert_adv
        if not np.isfinite(used_adv) or abs(used_adv) <= 1e-8:
            continue
        candidates.append(
            FrroExpertCandidate(
                env_idx=env_idx,
                observations=traj.observations,
                actions=traj.actions,
                advantage=used_adv,
                gate=gate,
            )
        )

    if candidates:
        chunk_size = int(adv_cfg.get("frro_expert_logprob_chunk_size", 4096))
        with torch.no_grad():
            old_mean = _expert_route_mean_logprobs(agent, candidates, device, chunk_size).detach().float().cpu().numpy()
        use_support_gate = bool(adv_cfg.get("frro_use_support_gate", False))
        support_min = float(adv_cfg.get("frro_support_logprob_min", -20.0))
        support_temp = max(float(adv_cfg.get("frro_support_gate_temperature", 1.0)), 1e-8)
        kept: list[FrroExpertCandidate] = []
        for candidate, old_logprob in zip(candidates, old_mean):
            candidate.old_mean_logprob = float(old_logprob)
            if use_support_gate:
                support_gate = float(1.0 / (1.0 + np.exp(-(candidate.old_mean_logprob - support_min) / support_temp)))
                candidate.advantage *= support_gate
                candidate.gate *= support_gate
            if abs(candidate.advantage) > 1e-8:
                kept.append(candidate)
        candidates = kept

    adv_values = np.asarray([candidate.advantage for candidate in candidates], dtype=np.float64)
    gates = np.asarray([candidate.gate for candidate in candidates], dtype=np.float64)
    info = {
        "frro_expert_adv_mean": float(adv_values.mean()) if adv_values.size else 0.0,
        "frro_expert_adv_std": float(adv_values.std()) if adv_values.size else 0.0,
        "frro_expert_gate_mean": float(gates.mean()) if gates.size else 0.0,
        "frro_expert_gate_std": float(gates.std()) if gates.size else 0.0,
        "frro_expert_num_routes": float(len(candidates)),
        "frro_expert_weight": expert_weight,
    }
    if best_gap_ratios:
        info["frro_best_gap_mean"] = float(np.mean(best_gap_ratios))
    return candidates, info


def _policy_route_old_mean_logprobs(batch) -> np.ndarray:
    old_logprobs = batch.old_logprobs.detach().float().cpu().numpy()
    valid = batch.valid.detach().float().cpu().numpy()
    sums = (old_logprobs * valid).sum(axis=0)
    counts = valid.sum(axis=0)
    return np.divide(sums, np.maximum(counts, 1.0), out=np.zeros_like(sums), where=counts > 0)


def _bafipo_config(cfg: dict[str, Any]) -> dict[str, Any]:
    adv_cfg = cfg.get("advantage", {}) or {}
    offline_cfg = {**adv_cfg, **(cfg.get("offline", {}) or {})}
    return {
        "pref_coef": float(offline_cfg.get("bafipo_pref_coef", 0.05)),
        "beta": float(offline_cfg.get("bafipo_beta", 1.0)),
        "policy_pairs_per_instance": int(offline_cfg.get("bafipo_policy_pairs_per_instance", 16)),
        "incumbent_pairs_per_instance": int(offline_cfg.get("bafipo_incumbent_pairs_per_instance", 8)),
        "top_quantile": float(offline_cfg.get("bafipo_top_quantile", 0.20)),
        "bottom_quantile": float(offline_cfg.get("bafipo_bottom_quantile", 0.20)),
        "gap_floor_ratio": float(offline_cfg.get("bafipo_gap_floor_ratio", 0.01)),
        "pair_weight_max": float(offline_cfg.get("bafipo_pair_weight_max", 2.0)),
        "quality_eta": max(float(offline_cfg.get("bafipo_quality_eta", 0.05)), 1e-8),
        "memory_margin": max(float(offline_cfg.get("bafipo_memory_margin", 0.005)), 0.0),
        "memory_eta": max(float(offline_cfg.get("bafipo_memory_eta", 0.05)), 1e-8),
        "spread_min": max(float(offline_cfg.get("bafipo_spread_min", 0.005)), 1e-8),
        "allow_incumbent_negative": bool(offline_cfg.get("bafipo_allow_incumbent_negative", False)),
        "expert_logprob_chunk_size": int(offline_cfg.get("bafipo_expert_logprob_chunk_size", offline_cfg.get("frro_expert_logprob_chunk_size", 4096))),
    }


def _prepare_bafipo_preference_pairs(
    agent: Agent,
    batch,
    cfg: dict[str, Any],
    envs,
    expert_buffer: ExpertReplayBuffer | None,
    policy_best_objectives: dict[str, float] | None,
    device: str | torch.device,
) -> tuple[list[BafipoPreferencePair], list[BafipoIncumbentCandidate], dict[str, float]]:
    if expert_buffer is None:
        return [], [], {}
    bafipo_cfg = _bafipo_config(cfg)
    num_envs = int(batch.actions.size(1))
    n_traj = int(batch.actions.size(2))
    objective, success, _ = _final_info_arrays(batch.final_infos, num_envs, n_traj)
    old_policy = _policy_route_old_mean_logprobs(batch)
    pairs: list[BafipoPreferencePair] = []
    incumbents: list[BafipoIncumbentCandidate] = []
    pending_inc_pairs: list[tuple[int, int, float, bool]] = []
    policy_pairs = 0
    incumbent_pair_count = 0
    weights: list[float] = []
    quality_gates: list[float] = []
    memory_gates: list[float] = []
    spread_gates: list[float] = []
    inc_beats_best = 0
    inc_beats_mean = 0
    inc_compared = 0
    top_quantile = max(min(float(bafipo_cfg["top_quantile"]), 1.0), 1e-6)
    bottom_quantile = max(min(float(bafipo_cfg["bottom_quantile"]), 1.0), 1e-6)
    for env_idx, env in enumerate(envs[:num_envs]):
        succ_idx = np.where(success[env_idx] & np.isfinite(objective[env_idx]))[0]
        if succ_idx.size < 2:
            continue
        succ_obj = objective[env_idx, succ_idx].astype(np.float64)
        mean_obj = float(np.mean(succ_obj))
        std_obj = float(np.std(succ_obj))
        scale = max(std_obj, float(bafipo_cfg["gap_floor_ratio"]) * max(mean_obj, 1e-8), 1e-8)
        spread = std_obj / max(mean_obj, 1e-8)
        spread_gate = float(np.clip(spread / float(bafipo_cfg["spread_min"]), 0.0, 1.0))
        spread_gates.append(spread_gate)
        if spread_gate <= 1e-8:
            continue
        order = succ_idx[np.argsort(objective[env_idx, succ_idx])]
        n_top = max(1, int(np.ceil(top_quantile * order.size)))
        n_bottom = max(1, int(np.ceil(bottom_quantile * order.size)))
        top = order[:n_top]
        bottom = order[-n_bottom:][::-1]
        max_policy_pairs = max(0, int(bafipo_cfg["policy_pairs_per_instance"]))
        for k in range(max_policy_pairs):
            pos = int(top[k % len(top)])
            neg = int(bottom[k % len(bottom)])
            gap = float(objective[env_idx, neg] - objective[env_idx, pos])
            if gap <= 1e-8:
                continue
            weight = spread_gate * float(np.clip(gap / scale, 0.0, float(bafipo_cfg["pair_weight_max"])))
            if weight <= 1e-8:
                continue
            pairs.append(
                BafipoPreferencePair(
                    env_idx=env_idx,
                    pos_kind="policy",
                    pos_traj=pos,
                    neg_kind="policy",
                    neg_traj=neg,
                    old_delta=float(old_policy[env_idx, pos] - old_policy[env_idx, neg]),
                    weight=weight,
                    incumbent_pair=False,
                )
            )
            policy_pairs += 1
            weights.append(weight)

        instance_id = _env_instance_id(env)
        traj = expert_buffer.trajectory_for_instance(instance_id)
        if traj is None or traj.length <= 0:
            continue
        ref_obj = float(traj.objective_distance_km)
        if not np.isfinite(ref_obj) or ref_obj <= 0.0:
            continue
        incumbent_idx = len(incumbents)
        incumbents.append(
            BafipoIncumbentCandidate(
                env_idx=env_idx,
                observations=traj.observations,
                actions=traj.actions,
                objective=ref_obj,
            )
        )
        quality_gap = (mean_obj - ref_obj) / max(float(bafipo_cfg["quality_eta"]) * ref_obj, 1e-8)
        quality_gate = float(np.clip(quality_gap, 0.0, 1.0))
        memory_gate = 1.0
        memory_obj = policy_best_objectives.get(instance_id) if policy_best_objectives is not None and instance_id is not None else None
        if memory_obj is not None and np.isfinite(memory_obj) and memory_obj > 0.0:
            target = ref_obj * (1.0 - float(bafipo_cfg["memory_margin"]))
            memory_gate = float(np.clip((memory_obj - target) / max(float(bafipo_cfg["memory_eta"]) * ref_obj, 1e-8), 0.0, 1.0))
        inc_gate = quality_gate * memory_gate
        quality_gates.append(quality_gate)
        memory_gates.append(memory_gate)
        best_policy = float(np.min(succ_obj))
        if ref_obj < best_policy:
            inc_beats_best += 1
        if ref_obj < mean_obj:
            inc_beats_mean += 1
        inc_compared += 1
        if inc_gate <= 1e-8:
            continue
        worse_policy = succ_idx[objective[env_idx, succ_idx] > ref_obj + 1e-8]
        worse_policy = worse_policy[np.argsort(objective[env_idx, worse_policy])[::-1]]
        max_inc_pairs = max(0, int(bafipo_cfg["incumbent_pairs_per_instance"]))
        for k in range(min(max_inc_pairs, len(worse_policy))):
            neg = int(worse_policy[k % len(worse_policy)])
            gap = float(objective[env_idx, neg] - ref_obj)
            weight = spread_gate * inc_gate * float(np.clip(gap / scale, 0.0, float(bafipo_cfg["pair_weight_max"])))
            if weight <= 1e-8:
                continue
            pending_inc_pairs.append((incumbent_idx, neg, weight, False))
            incumbent_pair_count += 1
            weights.append(weight)
        if bool(bafipo_cfg["allow_incumbent_negative"]):
            better_policy = succ_idx[objective[env_idx, succ_idx] < ref_obj - 1e-8]
            better_policy = better_policy[np.argsort(objective[env_idx, better_policy])]
            for k in range(min(max_inc_pairs, len(better_policy))):
                pos = int(better_policy[k % len(better_policy)])
                gap = float(ref_obj - objective[env_idx, pos])
                weight = spread_gate * inc_gate * float(np.clip(gap / scale, 0.0, float(bafipo_cfg["pair_weight_max"])))
                if weight <= 1e-8:
                    continue
                pending_inc_pairs.append((incumbent_idx, pos, weight, True))
                incumbent_pair_count += 1
                weights.append(weight)

    if pending_inc_pairs and incumbents:
        chunk_size = max(1, int(bafipo_cfg["expert_logprob_chunk_size"]))
        with torch.no_grad():
            old_inc = _expert_route_mean_logprobs(agent, incumbents, device, chunk_size).detach().float().cpu().numpy()
        for candidate, old_val in zip(incumbents, old_inc):
            candidate.old_mean_logprob = float(old_val)
        for incumbent_idx, policy_traj, weight, incumbent_is_negative in pending_inc_pairs:
            candidate = incumbents[incumbent_idx]
            env_idx = int(candidate.env_idx)
            if incumbent_is_negative:
                old_delta = float(old_policy[env_idx, policy_traj] - candidate.old_mean_logprob)
                pairs.append(
                    BafipoPreferencePair(
                        env_idx=env_idx,
                        pos_kind="policy",
                        pos_traj=int(policy_traj),
                        neg_kind="incumbent",
                        neg_traj=-1,
                        old_delta=old_delta,
                        weight=float(weight),
                        incumbent_pair=True,
                    )
                )
            else:
                old_delta = float(candidate.old_mean_logprob - old_policy[env_idx, policy_traj])
                pairs.append(
                    BafipoPreferencePair(
                        env_idx=env_idx,
                        pos_kind="incumbent",
                        pos_traj=-1,
                        neg_kind="policy",
                        neg_traj=int(policy_traj),
                        old_delta=old_delta,
                        weight=float(weight),
                        incumbent_pair=True,
                    )
                )

    info = {
        "bafipo_pref_pairs": float(len(pairs)),
        "bafipo_policy_pairs": float(policy_pairs),
        "bafipo_incumbent_pairs": float(incumbent_pair_count),
        "bafipo_quality_gate_mean": float(np.mean(quality_gates)) if quality_gates else 0.0,
        "bafipo_memory_gate_mean": float(np.mean(memory_gates)) if memory_gates else 0.0,
        "bafipo_spread_gate_mean": float(np.mean(spread_gates)) if spread_gates else 0.0,
        "bafipo_incumbent_beats_best_rate": float(inc_beats_best / max(inc_compared, 1)),
        "bafipo_incumbent_beats_mean_rate": float(inc_beats_mean / max(inc_compared, 1)),
        "bafipo_pair_weight_mean": float(np.mean(weights)) if weights else 0.0,
        "bafipo_pref_coef": float(bafipo_cfg["pref_coef"]),
    }
    return pairs, incumbents, info


def _dapg_demo_gate_from_rollout(
    batch,
    cfg: dict[str, Any],
    envs,
    expert_buffer: ExpertReplayBuffer | None,
    policy_best_objectives: dict[str, float] | None,
) -> tuple[float, dict[str, float]]:
    offline_cfg = cfg.get("offline", {}) or {}
    method = _offline_method(cfg)
    use_gate = bool(offline_cfg.get("use_dapg_demo_gate", method in {"gadapg", "ga_dapg", "ga-dapg", "group_dapg", "group-dapg"}))
    if not use_gate or expert_buffer is None:
        return 1.0, {
            "dapg_demo_gate_mean": 1.0,
            "dapg_demo_gate_std": 0.0,
            "dapg_demo_active_ratio": 1.0,
            "expert_better_ratio": 1.0,
            "dapg_memory_better_rate": 0.0,
            "dapg_memory_gap_mean": 0.0,
        }

    num_envs = int(batch.actions.size(1))
    n_traj = int(batch.actions.size(2))
    objective, success, _ = _final_info_arrays(batch.final_infos, num_envs, n_traj)
    eta = max(float(offline_cfg.get("dapg_demo_gate_eta", 0.05)), 1e-8)
    margin = max(float(offline_cfg.get("dapg_demo_gate_margin", 0.0)), 0.0)
    policy_base = str(offline_cfg.get("dapg_demo_gate_policy_base", "best_successful")).lower()
    use_memory = bool(offline_cfg.get("use_dapg_memory_gate", True))
    gates: list[float] = []
    memory_better = 0
    memory_gaps: list[float] = []
    for env_idx, env in enumerate(envs[:num_envs]):
        instance_id = _env_instance_id(env)
        ref_obj = expert_buffer.reference_objective(instance_id)
        if ref_obj is None or not np.isfinite(ref_obj) or ref_obj <= 0.0:
            continue
        current_success = objective[env_idx][success[env_idx] & np.isfinite(objective[env_idx])]
        if current_success.size > 0:
            if policy_base in {"mean_successful", "mean", "avg_successful", "average_successful"}:
                best_known = float(np.mean(current_success))
            else:
                best_known = float(np.min(current_success))
        else:
            best_known = np.inf
        if use_memory and policy_best_objectives is not None and instance_id is not None:
            memory_obj = policy_best_objectives.get(instance_id)
            if memory_obj is not None and np.isfinite(memory_obj) and memory_obj > 0.0:
                best_known = min(best_known, float(memory_obj))
        if not np.isfinite(best_known):
            gates.append(1.0)
            continue
        target_obj = float(ref_obj) * (1.0 - margin)
        gap_ratio = (best_known - target_obj) / max(float(ref_obj), 1e-8)
        gate = float(np.clip(gap_ratio / eta, 0.0, 1.0))
        gates.append(gate)
        memory_gaps.append(float(gap_ratio))
        if best_known <= target_obj:
            memory_better += 1

    if not gates:
        return 1.0, {
            "dapg_demo_gate_mean": 1.0,
            "dapg_demo_gate_std": 0.0,
            "dapg_demo_active_ratio": 1.0,
            "expert_better_ratio": 1.0,
            "dapg_memory_better_rate": 0.0,
            "dapg_memory_gap_mean": 0.0,
        }
    gate_arr = np.asarray(gates, dtype=np.float64)
    active_ratio = float(np.mean(gate_arr > 1e-8))
    return float(gate_arr.mean()), {
        "dapg_demo_gate_mean": float(gate_arr.mean()),
        "dapg_demo_gate_std": float(gate_arr.std()),
        "dapg_demo_active_ratio": active_ratio,
        "expert_better_ratio": active_ratio,
        "dapg_memory_better_rate": float(memory_better / max(len(gates), 1)),
        "dapg_memory_gap_mean": float(np.mean(memory_gaps)) if memory_gaps else 0.0,
    }


def _to_scalar_values(values: torch.Tensor) -> torch.Tensor:
    if values.dim() >= 1 and values.size(-1) == 1:
        return values.squeeze(-1)
    return values


def _value_head(values: torch.Tensor, head_idx: int) -> torch.Tensor:
    values = _to_scalar_values(values)
    if values.dim() >= 4 and values.size(-1) >= 3:
        return values[..., head_idx]
    return values


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=values.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def _evaluate_policy_loss_with_stats(
    agent,
    batch,
    returns,
    advantages,
    cfg,
    env_indices: Sequence[int] | np.ndarray | None = None,
    step_start: int = 0,
    step_end: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    train_cfg = cfg["training"]
    clip_coef = float(train_cfg.get("clip_coef", 0.2))
    vf_coef = float(train_cfg.get("vf_coef", 0.5))
    ent_coef = float(train_cfg.get("ent_coef", 0.01))
    if env_indices is None:
        env_indices = np.arange(batch.actions.size(1), dtype=np.int64)
    else:
        env_indices = np.asarray(env_indices, dtype=np.int64)

    if step_end is None:
        step_end = len(batch.observations)
    step_start = max(0, int(step_start))
    step_end = min(len(batch.observations), int(step_end))
    if step_start >= step_end:
        raise ValueError(f"empty PPO step range: [{step_start}, {step_end})")

    cached_state = agent.backbone.encode(_slice_obs_by_env(batch.observations[0], env_indices))
    policy_losses = []
    value_losses = []
    entropy_losses = []
    approx_kls = []
    clip_fracs = []
    for step in range(step_start, step_end):
        obs_mb = _slice_obs_by_env(batch.observations[step], env_indices)
        actions = batch.actions[step, env_indices].long()
        old_logprob = batch.old_logprobs[step, env_indices]
        _, new_logprob, entropy, value, _ = agent.get_action_and_value_cached(
            obs_mb,
            action=actions,
            state=cached_state,
        )
        value = value.squeeze(-1)
        logratio = new_logprob - old_logprob
        ratio = torch.exp(logratio)
        adv = advantages[step, env_indices]
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * adv
        valid = batch.valid[step, env_indices]
        policy_losses.append(-_masked_mean(torch.minimum(unclipped, clipped), valid))
        value_losses.append(_masked_mean(F.mse_loss(value, returns[step, env_indices], reduction="none"), valid))
        entropy_losses.append(_masked_mean(entropy, valid))
        with torch.no_grad():
            approx_kls.append(_masked_mean((ratio - 1.0) - logratio, valid))
            clip_fracs.append(_masked_mean((torch.abs(ratio - 1.0) > clip_coef).float(), valid))
    policy_loss = torch.stack(policy_losses).mean()
    value_loss = torch.stack(value_losses).mean()
    entropy_loss = torch.stack(entropy_losses).mean()
    total = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss
    stats = {
        "approx_kl": float(torch.stack(approx_kls).mean().detach().cpu().item()) if approx_kls else 0.0,
        "clip_fraction": float(torch.stack(clip_fracs).mean().detach().cpu().item()) if clip_fracs else 0.0,
    }
    return total, policy_loss.detach(), value_loss.detach(), entropy_loss.detach(), stats


def _evaluate_policy_loss_policy_only_with_stats(
    agent,
    batch,
    advantages,
    cfg,
    env_indices: Sequence[int] | np.ndarray | None = None,
    step_start: int = 0,
    step_end: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    train_cfg = cfg["training"]
    clip_coef = float(train_cfg.get("clip_coef", 0.2))
    ent_coef = float(train_cfg.get("ent_coef", 0.01))
    if env_indices is None:
        env_indices = np.arange(batch.actions.size(1), dtype=np.int64)
    else:
        env_indices = np.asarray(env_indices, dtype=np.int64)

    if step_end is None:
        step_end = len(batch.observations)
    step_start = max(0, int(step_start))
    step_end = min(len(batch.observations), int(step_end))
    if step_start >= step_end:
        raise ValueError(f"empty PPO step range: [{step_start}, {step_end})")

    cached_state = agent.backbone.encode(_slice_obs_by_env(batch.observations[0], env_indices))
    policy_losses = []
    entropy_losses = []
    approx_kls = []
    clip_fracs = []
    for step in range(step_start, step_end):
        obs_mb = _slice_obs_by_env(batch.observations[step], env_indices)
        actions = batch.actions[step, env_indices].long()
        old_logprob = batch.old_logprobs[step, env_indices]
        _, new_logprob, entropy, _value, _ = agent.get_action_and_value_cached(
            obs_mb,
            action=actions,
            state=cached_state,
        )
        logratio = new_logprob - old_logprob
        ratio = torch.exp(logratio)
        adv = advantages[step, env_indices]
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * adv
        valid = batch.valid[step, env_indices]
        policy_losses.append(-_masked_mean(torch.minimum(unclipped, clipped), valid))
        entropy_losses.append(_masked_mean(entropy, valid))
        with torch.no_grad():
            approx_kls.append(_masked_mean((ratio - 1.0) - logratio, valid))
            clip_fracs.append(_masked_mean((torch.abs(ratio - 1.0) > clip_coef).float(), valid))
    policy_loss = torch.stack(policy_losses).mean()
    value_loss = torch.zeros((), device=policy_loss.device, dtype=policy_loss.dtype)
    entropy_loss = torch.stack(entropy_losses).mean()
    total = policy_loss - ent_coef * entropy_loss
    stats = {
        "approx_kl": float(torch.stack(approx_kls).mean().detach().cpu().item()) if approx_kls else 0.0,
        "clip_fraction": float(torch.stack(clip_fracs).mean().detach().cpu().item()) if clip_fracs else 0.0,
    }
    return total, policy_loss.detach(), value_loss.detach(), entropy_loss.detach(), stats


def _compute_pomo_trajectory_advantages(
    batch,
    envs,
    cfg: dict[str, Any],
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    num_envs = int(batch.actions.size(1))
    n_traj = int(batch.actions.size(2))
    valid = batch.valid
    reward_sum = (batch.rewards * valid.float()).sum(dim=0)
    objective, success, _served = _final_info_arrays(batch.final_infos, num_envs, n_traj)

    rewards_np = reward_sum.detach().cpu().numpy().astype(np.float64, copy=True)
    used_objective = np.zeros((num_envs, n_traj), dtype=bool)
    success_mask = np.zeros((num_envs, n_traj), dtype=bool)
    for env_idx, env in enumerate(envs[:num_envs]):
        scale = _reward_scale_from_env(env)
        obj_row = objective[env_idx]
        suc_row = success[env_idx]
        finite_obj = np.isfinite(obj_row) & (obj_row > 0.0) & suc_row
        if finite_obj.any():
            rewards_np[env_idx, finite_obj] = -obj_row[finite_obj] / max(float(scale), 1e-9)
            used_objective[env_idx, finite_obj] = True
        success_mask[env_idx, : min(n_traj, suc_row.size)] = suc_row[:n_traj]

    traj_valid_np = valid.detach().cpu().numpy().any(axis=0)
    rewards_np = np.where(traj_valid_np & np.isfinite(rewards_np), rewards_np, 0.0)
    raw_adv = np.zeros_like(rewards_np, dtype=np.float64)
    baselines = np.zeros(num_envs, dtype=np.float64)
    within_stds: list[float] = []
    within_means: list[float] = []
    valid_counts: list[int] = []
    for env_idx in range(num_envs):
        mask = traj_valid_np[env_idx] & np.isfinite(rewards_np[env_idx])
        valid_counts.append(int(mask.sum()))
        if not mask.any():
            continue
        values = rewards_np[env_idx, mask]
        baseline = float(values.mean())
        baselines[env_idx] = baseline
        raw_adv[env_idx, mask] = values - baseline
        within_stds.append(float(values.std()))
        within_means.append(float(values.mean()))

    raw_valid = raw_adv[traj_valid_np]
    if raw_valid.size > 1:
        raw_mean = float(raw_valid.mean())
        raw_std = float(raw_valid.std())
        norm_adv = (raw_adv - raw_mean) / max(raw_std, 1e-8)
    else:
        raw_mean = float(raw_valid.mean()) if raw_valid.size else 0.0
        raw_std = 0.0
        norm_adv = raw_adv
    norm_adv = np.where(traj_valid_np, norm_adv, 0.0)
    adv_tensor = torch.as_tensor(norm_adv, device=device, dtype=batch.rewards.dtype).unsqueeze(0).expand_as(batch.rewards)
    adv_tensor = adv_tensor * valid.float()
    returns = torch.zeros_like(batch.rewards)

    norm_valid = norm_adv[traj_valid_np]
    reward_valid = rewards_np[traj_valid_np]
    info = {
        "advantage_mode": "pomo_trajectory",
        "pomo_reward_mean": float(reward_valid.mean()) if reward_valid.size else 0.0,
        "pomo_reward_std": float(reward_valid.std()) if reward_valid.size else 0.0,
        "pomo_baseline_mean": float(np.mean(baselines)) if baselines.size else 0.0,
        "pomo_within_reward_std_mean": float(np.mean(within_stds)) if within_stds else 0.0,
        "pomo_within_reward_std_p10": float(np.quantile(within_stds, 0.10)) if within_stds else 0.0,
        "pomo_within_reward_std_p90": float(np.quantile(within_stds, 0.90)) if within_stds else 0.0,
        "pomo_valid_traj_ratio": float(traj_valid_np.mean()) if traj_valid_np.size else 0.0,
        "pomo_success_traj_ratio": float(success_mask[traj_valid_np].mean()) if traj_valid_np.any() else 0.0,
        "pomo_objective_used_ratio": float(used_objective[traj_valid_np].mean()) if traj_valid_np.any() else 0.0,
        "pomo_adv_raw_mean": raw_mean,
        "pomo_adv_raw_std": raw_std,
        "pomo_adv_norm_mean": float(norm_valid.mean()) if norm_valid.size else 0.0,
        "pomo_adv_norm_std": float(norm_valid.std()) if norm_valid.size else 0.0,
        "pomo_valid_traj_count_mean": float(np.mean(valid_counts)) if valid_counts else 0.0,
    }
    return returns, adv_tensor, info


def _compute_gae_from_rewards(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    route_boundaries: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = _to_scalar_values(values)
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(rewards[0])
    for step in reversed(range(rewards.size(0))):
        if step == rewards.size(0) - 1:
            next_value = torch.zeros_like(values[step])
        else:
            next_value = values[step + 1]
        bootstrap_nonterminal = (~dones[step]).float()
        if route_boundaries is None:
            gae_nonterminal = bootstrap_nonterminal
        else:
            gae_nonterminal = (~(dones[step] | route_boundaries[step])).float()
        delta = rewards[step] + float(gamma) * next_value * bootstrap_nonterminal - values[step]
        last_gae = delta + float(gamma) * float(gae_lambda) * gae_nonterminal * last_gae
        advantages[step] = last_gae
    returns = advantages + values
    return returns, advantages


def _compute_discounted_returns(rewards: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    returns = torch.zeros_like(rewards)
    next_return = torch.zeros_like(rewards[0])
    for step in reversed(range(rewards.size(0))):
        next_nonterminal = (~dones[step]).float()
        next_return = rewards[step] + float(gamma) * next_return * next_nonterminal
        returns[step] = next_return
    return returns


def _compute_gae_returns(
    batch,
    gamma: float,
    gae_lambda: float,
    route_segmented: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    route_boundaries = getattr(batch, "route_boundaries", None) if route_segmented else None
    return _compute_gae_from_rewards(
        batch.rewards,
        _value_head(batch.values, 0),
        batch.dones,
        gamma,
        gae_lambda,
        route_boundaries=route_boundaries,
    )


def _normalize_valid(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    valid_values = values[valid]
    if valid_values.numel() <= 1:
        return values
    return (values - valid_values.mean()) / (valid_values.std(unbiased=False) + 1e-8)


def _decomposed_critic_config(cfg: dict[str, Any]) -> dict[str, Any]:
    train_cfg = cfg.get("training", {}) or {}
    critic_cfg = cfg.get("critic", {}) or {}
    out = dict(critic_cfg)
    for key in (
        "use_decomposed_critic",
        "advantage_mode",
        "value_coef_total",
        "value_coef_boundary",
        "value_coef_internal",
        "consistency_coef",
        "balanced_adv_weight_boundary",
        "balanced_adv_weight_internal",
    ):
        if key in train_cfg and key not in out:
            out[key] = train_cfg[key]
    out.setdefault("use_decomposed_critic", True)
    out.setdefault("advantage_mode", "total")
    out.setdefault("value_coef_total", 1.0)
    out.setdefault("value_coef_boundary", 0.5)
    out.setdefault("value_coef_internal", 0.5)
    out.setdefault("consistency_coef", 0.1)
    out.setdefault("balanced_adv_weight_boundary", 0.3)
    out.setdefault("balanced_adv_weight_internal", 0.7)
    return out


def _build_decomposed_rewards_from_actions(batch) -> dict[str, torch.Tensor]:
    rewards_total = batch.rewards
    rewards_boundary = torch.zeros_like(rewards_total)
    rewards_internal = torch.zeros_like(rewards_total)
    for step, obs in enumerate(batch.observations[: rewards_total.size(0)]):
        last_node = torch.as_tensor(obs["last_node_idx"], device=rewards_total.device, dtype=batch.actions.dtype)
        action = batch.actions[step].to(device=rewards_total.device)
        boundary_mask = (last_node == 0) | (action == 0)
        rewards_boundary[step] = torch.where(boundary_mask, rewards_total[step], torch.zeros_like(rewards_total[step]))
        rewards_internal[step] = torch.where(boundary_mask, torch.zeros_like(rewards_total[step]), rewards_total[step])
    return {"total": rewards_total, "boundary": rewards_boundary, "internal": rewards_internal}


def _attach_decomposed_rewards(batch) -> dict[str, float]:
    if not all(hasattr(batch, name) for name in ("rewards_total", "rewards_boundary", "rewards_internal")):
        rewards = _build_decomposed_rewards_from_actions(batch)
        batch.rewards_total = rewards["total"]
        batch.rewards_boundary = rewards["boundary"]
        batch.rewards_internal = rewards["internal"]
    residual = batch.rewards_total - (batch.rewards_boundary + batch.rewards_internal)
    step_abs = residual.abs()
    episode_residual = residual.masked_fill(~batch.valid, 0.0).sum(dim=0)
    return {
        "reward_decomposition_max_abs_error": float(step_abs.max().detach().cpu().item()) if step_abs.numel() else 0.0,
        "reward_decomposition_mean_abs_error": float(step_abs[batch.valid].mean().detach().cpu().item()) if batch.valid.any() else 0.0,
        "episode_decomposition_max_abs_error": float(episode_residual.abs().max().detach().cpu().item()) if episode_residual.numel() else 0.0,
        "episode_decomposition_mean_abs_error": float(episode_residual.abs().mean().detach().cpu().item()) if episode_residual.numel() else 0.0,
    }


def _decompose_distance_rewards(batch) -> dict[str, torch.Tensor]:
    _attach_decomposed_rewards(batch)
    return {
        "total": batch.rewards_total,
        "boundary": batch.rewards_boundary,
        "internal": batch.rewards_internal,
    }


def _explained_variance(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> float:
    pred_v = pred[valid].detach()
    target_v = target[valid].detach()
    if target_v.numel() <= 1:
        return float("nan")
    var_y = torch.var(target_v, unbiased=False)
    if float(var_y.detach().cpu().item()) <= 1e-12:
        return float("nan")
    ev = 1.0 - torch.var(target_v - pred_v, unbiased=False) / (var_y + 1e-8)
    return float(ev.detach().cpu().item())


def _stats(values: torch.Tensor, valid: torch.Tensor) -> tuple[float, float]:
    vals = values[valid].detach()
    if vals.numel() == 0:
        return 0.0, 0.0
    return float(vals.mean().cpu().item()), float(vals.std(unbiased=False).cpu().item())


def _compute_decomposed_returns_advantages(
    batch,
    gamma: float,
    gae_lambda: float,
    use_gae: bool,
    cfg: dict[str, Any],
    route_segmented: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, dict[str, float]]:
    decomposition_error_info = _attach_decomposed_rewards(batch)
    rewards = _decompose_distance_rewards(batch)
    values = {
        "total": _value_head(batch.values, 0),
        "boundary": _value_head(batch.values, 1),
        "internal": _value_head(batch.values, 2),
    }
    returns: dict[str, torch.Tensor] = {}
    advantages: dict[str, torch.Tensor] = {}
    route_boundaries = getattr(batch, "route_boundaries", None) if route_segmented else None
    for name in ("total", "boundary", "internal"):
        if use_gae:
            ret, adv = _compute_gae_from_rewards(
                rewards[name],
                values[name],
                batch.dones,
                gamma,
                gae_lambda,
                route_boundaries=route_boundaries,
            )
        else:
            ret = _compute_discounted_returns(rewards[name], batch.dones, gamma)
            adv = ret - values[name]
        returns[name] = ret
        advantages[name] = adv

    critic_cfg = _decomposed_critic_config(cfg)
    mode = str(critic_cfg.get("advantage_mode", "total")).strip().lower()
    if mode == "exact_decomp":
        actor_adv = advantages["boundary"] + advantages["internal"]
        actor_adv = _normalize_valid(actor_adv, batch.valid)
    elif mode == "balanced_decomp":
        boundary_adv = _normalize_valid(advantages["boundary"], batch.valid)
        internal_adv = _normalize_valid(advantages["internal"], batch.valid)
        actor_adv = (
            float(critic_cfg.get("balanced_adv_weight_boundary", 0.3)) * boundary_adv
            + float(critic_cfg.get("balanced_adv_weight_internal", 0.7)) * internal_adv
        )
        actor_adv = _normalize_valid(actor_adv, batch.valid)
    else:
        actor_adv = _normalize_valid(advantages["total"], batch.valid)

    info: dict[str, float] = dict(decomposition_error_info)
    for name in ("total", "boundary", "internal"):
        mean, std = _stats(advantages[name], batch.valid)
        info[f"adv_{name}_mean"] = mean
        info[f"adv_{name}_std"] = std
        ret_mean, _ = _stats(returns[name], batch.valid)
        info[f"return_{name}_mean"] = ret_mean
    actor_mean, actor_std = _stats(actor_adv, batch.valid)
    info["adv_actor_mean"] = actor_mean
    info["adv_actor_std"] = actor_std
    boundary_dist = -rewards["boundary"]
    internal_dist = -rewards["internal"]
    boundary_sum = float(boundary_dist[batch.valid].sum().detach().cpu().item()) if batch.valid.any() else 0.0
    internal_sum = float(internal_dist[batch.valid].sum().detach().cpu().item()) if batch.valid.any() else 0.0
    denom = max(boundary_sum + internal_sum, 1e-8)
    info["boundary_distance_mean"] = _stats(boundary_dist, batch.valid)[0]
    info["internal_distance_mean"] = _stats(internal_dist, batch.valid)[0]
    info["boundary_share"] = boundary_sum / denom
    info["internal_share"] = internal_sum / denom
    info["advantage_mode"] = mode
    return returns, advantages, actor_adv, info


def _evaluate_policy_loss_decomposed(
    agent,
    batch,
    returns: dict[str, torch.Tensor],
    advantages_actor: torch.Tensor,
    cfg: dict[str, Any],
    device,
    env_indices: np.ndarray | None = None,
    step_start: int = 0,
    step_end: int | None = None,
):
    del device
    train_cfg = cfg["training"]
    critic_cfg = _decomposed_critic_config(cfg)
    clip_coef = float(train_cfg.get("clip_coef", 0.2))
    vf_coef = float(train_cfg.get("vf_coef", 0.5))
    ent_coef = float(train_cfg.get("ent_coef", 0.01))
    value_coef_total = float(critic_cfg.get("value_coef_total", 1.0))
    value_coef_boundary = float(critic_cfg.get("value_coef_boundary", 0.5))
    value_coef_internal = float(critic_cfg.get("value_coef_internal", 0.5))
    consistency_coef = float(critic_cfg.get("consistency_coef", 0.1))
    if env_indices is None:
        env_indices = np.arange(batch.actions.size(1), dtype=np.int64)
    else:
        env_indices = np.asarray(env_indices, dtype=np.int64)
    if step_end is None:
        step_end = len(batch.observations)
    step_start = max(0, int(step_start))
    step_end = min(len(batch.observations), int(step_end))
    if step_start >= step_end:
        raise ValueError(f"empty PPO step range: [{step_start}, {step_end})")

    cached_state = agent.backbone.encode(_slice_obs_by_env(batch.observations[0], env_indices))
    policy_losses = []
    value_total_losses = []
    value_boundary_losses = []
    value_internal_losses = []
    consistency_losses = []
    entropy_losses = []
    for step in range(step_start, step_end):
        obs_mb = _slice_obs_by_env(batch.observations[step], env_indices)
        actions = batch.actions[step, env_indices].long()
        old_logprob = batch.old_logprobs[step, env_indices]
        _, new_logprob, entropy, value, _ = agent.get_action_and_value_cached(
            obs_mb,
            action=actions,
            state=cached_state,
        )
        value = _to_scalar_values(value)
        if value.dim() < 3 or value.size(-1) < 3:
            raise ValueError("decomposed critic requires value tensor with three heads")
        value_total = value[..., 0]
        value_boundary = value[..., 1]
        value_internal = value[..., 2]
        ratio = torch.exp(new_logprob - old_logprob)
        adv = advantages_actor[step, env_indices]
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * adv
        valid = batch.valid[step, env_indices]
        policy_losses.append(-_masked_mean(torch.minimum(unclipped, clipped), valid))
        value_total_losses.append(_masked_mean(F.mse_loss(value_total, returns["total"][step, env_indices], reduction="none"), valid))
        value_boundary_losses.append(_masked_mean(F.mse_loss(value_boundary, returns["boundary"][step, env_indices], reduction="none"), valid))
        value_internal_losses.append(_masked_mean(F.mse_loss(value_internal, returns["internal"][step, env_indices], reduction="none"), valid))
        consistency_losses.append(_masked_mean((value_total - value_boundary - value_internal).pow(2), valid))
        entropy_losses.append(_masked_mean(entropy, valid))
    policy_loss = torch.stack(policy_losses).mean()
    value_loss_total = torch.stack(value_total_losses).mean()
    value_loss_boundary = torch.stack(value_boundary_losses).mean()
    value_loss_internal = torch.stack(value_internal_losses).mean()
    value_consistency_loss = torch.stack(consistency_losses).mean()
    entropy_loss = torch.stack(entropy_losses).mean()
    value_loss = (
        value_coef_total * value_loss_total
        + value_coef_boundary * value_loss_boundary
        + value_coef_internal * value_loss_internal
        + consistency_coef * value_consistency_loss
    )
    total = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss
    loss_info = {
        "value_loss_total": float(value_loss_total.detach().cpu().item()),
        "value_loss_boundary": float(value_loss_boundary.detach().cpu().item()),
        "value_loss_internal": float(value_loss_internal.detach().cpu().item()),
        "value_consistency_loss": float(value_consistency_loss.detach().cpu().item()),
    }
    return total, policy_loss.detach(), value_loss.detach(), entropy_loss.detach(), loss_info


def _decomposed_value_diagnostics(batch, returns: dict[str, torch.Tensor]) -> dict[str, float]:
    values = {
        "total": _value_head(batch.values, 0),
        "boundary": _value_head(batch.values, 1),
        "internal": _value_head(batch.values, 2),
    }
    return {
        f"explained_variance_{name}": _explained_variance(values[name], returns[name], batch.valid)
        for name in ("total", "boundary", "internal")
    }


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
    policy_best_objectives: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    adv_cfg = _advantage_config(cfg)
    use_group = _group_advantage_enabled(cfg)
    use_ref = _reference_advantage_enabled(cfg)
    use_frro = _frro_enabled(cfg)
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
        "ref_memory_gate_mean": 0.0,
        "ref_memory_gate_std": 0.0,
        "ref_memory_better_rate": 0.0,
        "ref_memory_gap_mean": 0.0,
        "ref_base_gap_ratio_mean": 0.0,
        "frro_adv_mean": 0.0,
        "frro_adv_std": 0.0,
        "frro_positive_mean": 0.0,
        "frro_positive_std": 0.0,
        "frro_negative_mean": 0.0,
        "frro_negative_std": 0.0,
        "frro_gate_mean": 0.0,
        "frro_gate_std": 0.0,
        "frro_falsified_rate": 0.0,
        "frro_best_gap_mean": 0.0,
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

    if use_frro and expert_buffer is not None:
        frro_coef = float(adv_cfg.get("frro_coef", 0.10))
        frro_rho = max(float(adv_cfg.get("frro_rho", 0.10)), 1e-8)
        frro_clip = float(adv_cfg.get("frro_clip", 2.0))
        success_only = bool(adv_cfg.get("frro_success_only", True))
        positive_coef = float(adv_cfg.get("frro_positive_coef", 1.0))
        negative_coef = float(adv_cfg.get("frro_negative_coef", 1.0))
        frro_mode = str(adv_cfg.get("frro_advantage_mode", "remaining_gap")).lower()
        falsification_margin = max(float(adv_cfg.get("frro_falsification_margin", 0.0)), 0.0)
        falsification_eta = max(float(adv_cfg.get("frro_falsification_eta", 0.05)), 1e-8)
        use_memory_falsification = bool(adv_cfg.get("frro_use_memory_falsification", True))
        use_current_falsification = bool(adv_cfg.get("frro_use_current_falsification", True))
        frro_adv = np.zeros((num_envs, n_traj), dtype=np.float64)
        frro_positive = np.zeros_like(frro_adv)
        frro_negative = np.zeros_like(frro_adv)
        frro_gate = np.zeros((num_envs, 1), dtype=np.float64)
        falsified = np.zeros((num_envs, 1), dtype=np.float64)
        best_gap_ratios: list[float] = []
        for env_idx, env in enumerate(envs[:num_envs]):
            instance_id = _env_instance_id(env)
            ref_obj = expert_buffer.reference_objective(instance_id)
            if ref_obj is None or not np.isfinite(ref_obj) or ref_obj <= 0.0:
                continue
            if frro_mode in {"gap", "gap_reduction", "remaining_gap", "remaining-gap", "improvement"}:
                stats = _frro_improvement_stats(objective[env_idx], success[env_idx], float(ref_obj), adv_cfg)
                if stats is None:
                    continue
                base_obj, remaining_gap, scale = stats
                row = (base_obj - objective[env_idx]) / scale
                current_success = objective[env_idx][success[env_idx] & np.isfinite(objective[env_idx])]
                memory_obj = policy_best_objectives.get(instance_id) if policy_best_objectives is not None and instance_id is not None else None
                gate_value, is_falsified, best_gap = _frro_expert_gate(
                    remaining_gap=remaining_gap,
                    ref_obj=float(ref_obj),
                    current_success_objectives=current_success,
                    memory_obj=memory_obj,
                    adv_cfg=adv_cfg,
                )
                if best_gap is not None:
                    best_gap_ratios.append(best_gap)
                falsified[env_idx, 0] = 1.0 if is_falsified else 0.0
            else:
                row = (float(ref_obj) - objective[env_idx]) / max(frro_rho * float(ref_obj), 1e-8)
                best_known = np.inf
                if use_current_falsification:
                    succ_obj = objective[env_idx][success[env_idx] & np.isfinite(objective[env_idx])]
                    if succ_obj.size > 0:
                        best_known = min(best_known, float(np.min(succ_obj)))
                if (
                    use_memory_falsification
                    and policy_best_objectives is not None
                    and instance_id is not None
                ):
                    memory_obj = policy_best_objectives.get(instance_id)
                    if memory_obj is not None and np.isfinite(memory_obj) and memory_obj > 0.0:
                        best_known = min(best_known, float(memory_obj))

                if np.isfinite(best_known):
                    target_obj = float(ref_obj) * (1.0 - falsification_margin)
                    gate_gap = (best_known - target_obj) / max(float(ref_obj), 1e-8)
                    gate_value = float(np.clip(gate_gap / falsification_eta, 0.0, 1.0))
                    best_gap_ratios.append((best_known - float(ref_obj)) / max(float(ref_obj), 1e-8))
                else:
                    gate_value = 1.0
                if gate_value <= 1e-6:
                    falsified[env_idx, 0] = 1.0
            row[~np.isfinite(row)] = 0.0
            if success_only:
                row = np.where(success[env_idx], row, 0.0)

            pos = np.clip(np.maximum(row, 0.0), 0.0, frro_clip) * positive_coef
            neg = np.clip(np.minimum(row, 0.0), -frro_clip, 0.0) * negative_coef
            frro_gate[env_idx, 0] = gate_value

            frro_positive[env_idx] = pos
            if frro_mode in {"gap", "gap_reduction", "remaining_gap", "remaining-gap", "improvement"}:
                frro_negative[env_idx] = neg
                frro_adv[env_idx] = pos + neg
            else:
                frro_negative[env_idx] = neg * gate_value
                frro_adv[env_idx] = pos + neg * gate_value

        frro_used = frro_adv * frro_coef
        route_adv += frro_used
        info["frro_adv_mean"], info["frro_adv_std"] = _finite_mean_std(frro_used)
        info["frro_positive_mean"], info["frro_positive_std"] = _finite_mean_std(frro_positive * frro_coef)
        info["frro_negative_mean"], info["frro_negative_std"] = _finite_mean_std(frro_negative * frro_coef)
        info["frro_gate_mean"], info["frro_gate_std"] = _finite_mean_std(frro_gate)
        info["frro_falsified_rate"] = float(falsified.mean())
        if best_gap_ratios:
            info["frro_best_gap_mean"] = float(np.mean(best_gap_ratios))

    if use_ref and expert_buffer is not None:
        ref_clip = float(adv_cfg.get("reference_adv_clip", 2.0))
        ref_coef = float(adv_cfg.get("reference_adv_coef", 0.10))
        ref_rho = max(float(adv_cfg.get("reference_adv_rho", 0.10)), 1e-8)
        success_only = bool(adv_cfg.get("reference_success_only", True))
        ref_mode = str(adv_cfg.get("reference_advantage_mode", "absolute")).lower()
        gap_baseline_mode = str(adv_cfg.get("reference_gap_baseline", "mean")).lower()
        gap_floor_ratio = max(float(adv_cfg.get("reference_gap_floor_ratio", 0.01)), 0.0)
        use_gate = bool(adv_cfg.get("use_reference_soft_gate", True))
        gate_eta = max(float(adv_cfg.get("reference_soft_gate_eta", 0.05)), 1e-8)
        estimate_mode = str(adv_cfg.get("reference_policy_estimate", "best")).lower()
        use_memory_gate = bool(adv_cfg.get("use_reference_memory_gate", False))
        memory_gate_eta = max(float(adv_cfg.get("reference_memory_gate_eta", gate_eta)), 1e-8)
        memory_margin = max(float(adv_cfg.get("reference_memory_margin", 0.0)), 0.0)
        ref_adv = np.zeros((num_envs, n_traj), dtype=np.float64)
        gate = np.ones((num_envs, 1), dtype=np.float64)
        memory_gate = np.ones((num_envs, 1), dtype=np.float64)
        memory_better = np.zeros((num_envs, 1), dtype=np.float64)
        memory_gaps: list[float] = []
        base_gap_ratios: list[float] = []
        for env_idx, env in enumerate(envs[:num_envs]):
            instance_id = _env_instance_id(env)
            ref_obj = expert_buffer.reference_objective(instance_id)
            if ref_obj is None or not np.isfinite(ref_obj) or ref_obj <= 0.0:
                ref_adv[env_idx] = 0.0
                gate[env_idx, 0] = 0.0
                memory_gate[env_idx, 0] = 0.0
                continue
            succ_mask = success[env_idx] & np.isfinite(objective[env_idx])
            succ_obj = objective[env_idx][succ_mask]
            if ref_mode in {"gap", "gap_reduction", "remaining_gap", "remaining-gap"}:
                if succ_obj.size == 0:
                    ref_adv[env_idx] = 0.0
                    gate[env_idx, 0] = 0.0
                else:
                    base_obj = _objective_baseline(succ_obj, gap_baseline_mode)
                    remaining_gap = float(base_obj) - float(ref_obj)
                    base_gap_ratio = remaining_gap / max(float(ref_obj), 1e-8)
                    base_gap_ratios.append(float(base_gap_ratio))
                    gap_floor = gap_floor_ratio * float(ref_obj)
                    denom = max(remaining_gap, gap_floor, 1e-8)
                    row = (float(base_obj) - objective[env_idx]) / denom
                    row[~np.isfinite(row)] = 0.0
                    if success_only:
                        row = np.where(success[env_idx], row, 0.0)
                    ref_adv[env_idx] = np.clip(row, -ref_clip, ref_clip)
                    if use_gate:
                        gate[env_idx, 0] = float(np.clip(base_gap_ratio / gate_eta, 0.0, 1.0))
            else:
                row = (float(ref_obj) - objective[env_idx]) / max(ref_rho * float(ref_obj), 1e-8)
                row[~np.isfinite(row)] = 0.0
                if success_only:
                    row = np.where(success[env_idx], row, 0.0)
                ref_adv[env_idx] = np.clip(row, -ref_clip, ref_clip)
            if use_gate and ref_mode not in {"gap", "gap_reduction", "remaining_gap", "remaining-gap"}:
                succ_obj = objective[env_idx][success[env_idx] & np.isfinite(objective[env_idx])]
                if succ_obj.size == 0:
                    estimate_obj = np.inf
                elif estimate_mode == "mean":
                    estimate_obj = float(np.mean(succ_obj))
                else:
                    estimate_obj = float(np.min(succ_obj))
                gap = (estimate_obj - float(ref_obj)) / max(float(ref_obj), 1e-8)
                gate[env_idx, 0] = float(np.clip(gap / gate_eta, 0.0, 1.0)) if np.isfinite(gap) else 1.0
            if use_memory_gate and policy_best_objectives is not None and instance_id is not None:
                memory_obj = policy_best_objectives.get(instance_id)
                if memory_obj is not None and np.isfinite(memory_obj) and memory_obj > 0.0:
                    target_obj = float(ref_obj) * (1.0 - memory_margin)
                    memory_gap = (float(memory_obj) - target_obj) / max(float(ref_obj), 1e-8)
                    memory_gaps.append(float(memory_gap))
                    memory_gate[env_idx, 0] = float(np.clip(memory_gap / memory_gate_eta, 0.0, 1.0))
                    if float(memory_obj) <= target_obj:
                        memory_better[env_idx, 0] = 1.0
        combined_gate = gate * memory_gate if use_memory_gate else gate
        ref_used = ref_adv * combined_gate
        ref_used *= ref_coef
        route_adv += ref_used
        info["ref_adv_mean"], info["ref_adv_std"] = _finite_mean_std(ref_used)
        info["ref_gate_mean"], info["ref_gate_std"] = _finite_mean_std(combined_gate)
        if base_gap_ratios:
            info["ref_base_gap_ratio_mean"] = float(np.mean(base_gap_ratios))
        if use_memory_gate:
            info["ref_memory_gate_mean"], info["ref_memory_gate_std"] = _finite_mean_std(memory_gate)
            info["ref_memory_better_rate"] = float(memory_better.mean())
            if memory_gaps:
                info["ref_memory_gap_mean"] = float(np.mean(memory_gaps))

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


def _compute_frro_expert_candidate_loss(
    agent: Agent,
    candidates: list[FrroExpertCandidate],
    cfg: dict[str, Any],
    env_indices: np.ndarray,
    device: str | torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    adv_cfg = _advantage_config(cfg)
    selected_envs = {int(idx) for idx in np.asarray(env_indices, dtype=np.int64).reshape(-1)}
    selected = [candidate for candidate in candidates if candidate.env_idx in selected_envs and abs(candidate.advantage) > 1e-8]
    if not selected:
        zero = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
        return zero, {
            "frro_expert_loss": 0.0,
            "frro_expert_ratio_mean": 1.0,
            "frro_expert_ratio_std": 0.0,
            "frro_expert_clip_frac": 0.0,
            "frro_expert_adv_mean": 0.0,
            "frro_expert_adv_std": 0.0,
            "frro_expert_num_routes": 0.0,
        }

    chunk_size = int(adv_cfg.get("frro_expert_logprob_chunk_size", 4096))
    new_mean_logprob = _expert_route_mean_logprobs(agent, selected, device, chunk_size)
    old_mean_logprob = torch.as_tensor([candidate.old_mean_logprob for candidate in selected], dtype=new_mean_logprob.dtype, device=new_mean_logprob.device)
    adv = torch.as_tensor([candidate.advantage for candidate in selected], dtype=new_mean_logprob.dtype, device=new_mean_logprob.device)
    route_ratio = torch.exp(new_mean_logprob - old_mean_logprob.detach())
    route_clip_eps = float((cfg.get("offline", {}) or {}).get("route_clip_eps", (cfg.get("offline", {}) or {}).get("sl_clip_coef", 0.20)))
    unclipped = route_ratio * adv.detach()
    clipped = torch.clamp(route_ratio, 1.0 - route_clip_eps, 1.0 + route_clip_eps) * adv.detach()
    route_loss = -torch.minimum(unclipped, clipped).mean()
    clip_frac = ((route_ratio > 1.0 + route_clip_eps) | (route_ratio < 1.0 - route_clip_eps)).float().mean()
    return route_loss, {
        "frro_expert_loss": float(route_loss.detach().cpu().item()),
        "frro_expert_ratio_mean": float(route_ratio.detach().mean().cpu().item()),
        "frro_expert_ratio_std": float(route_ratio.detach().std(unbiased=False).cpu().item()) if route_ratio.numel() > 1 else 0.0,
        "frro_expert_clip_frac": float(clip_frac.detach().cpu().item()),
        "frro_expert_adv_mean": float(adv.detach().mean().cpu().item()),
        "frro_expert_adv_std": float(adv.detach().std(unbiased=False).cpu().item()) if adv.numel() > 1 else 0.0,
        "frro_expert_num_routes": float(len(selected)),
    }


def _compute_bafipo_preference_loss(
    agent: Agent,
    batch,
    pairs: list[BafipoPreferencePair],
    incumbents: list[BafipoIncumbentCandidate],
    cfg: dict[str, Any],
    env_indices: np.ndarray,
    device: str | torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    bafipo_cfg = _bafipo_config(cfg)
    selected_envs = {int(idx) for idx in np.asarray(env_indices, dtype=np.int64).reshape(-1)}
    selected_pairs = [pair for pair in pairs if pair.env_idx in selected_envs and pair.weight > 0.0]
    if not selected_pairs:
        zero = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
        return zero, {
            "bafipo_pref_loss": 0.0,
            "bafipo_pref_pair_count": 0.0,
            "bafipo_policy_pair_count": 0.0,
            "bafipo_incumbent_pair_count": 0.0,
            "bafipo_pref_weight_mean": 0.0,
            "bafipo_pref_logit_mean": 0.0,
        }

    env_indices = np.asarray(env_indices, dtype=np.int64)
    env_to_local = {int(env_idx): local_idx for local_idx, env_idx in enumerate(env_indices)}
    total_steps = int(batch.actions.size(0))
    n_traj = int(batch.actions.size(2))
    cached_state = agent.backbone.encode(_slice_obs_by_env(batch.observations[0], env_indices))
    route_sums = torch.zeros((len(env_indices), n_traj), dtype=batch.old_logprobs.dtype, device=device)
    route_counts = torch.zeros_like(route_sums)
    for step in range(total_steps):
        obs_mb = _slice_obs_by_env(batch.observations[step], env_indices)
        actions = batch.actions[step, env_indices].long()
        _, new_logprob, _, _, _ = agent.get_action_and_value_cached(obs_mb, action=actions, state=cached_state)
        valid = batch.valid[step, env_indices].to(dtype=new_logprob.dtype)
        route_sums = route_sums + new_logprob * valid
        route_counts = route_counts + valid
    policy_mean = route_sums / route_counts.clamp_min(1.0)

    needed_inc_envs = {
        pair.env_idx
        for pair in selected_pairs
        if pair.pos_kind == "incumbent" or pair.neg_kind == "incumbent"
    }
    selected_incumbents = [candidate for candidate in incumbents if candidate.env_idx in needed_inc_envs]
    incumbent_logprob: dict[int, torch.Tensor] = {}
    if selected_incumbents:
        chunk_size = max(1, int(bafipo_cfg["expert_logprob_chunk_size"]))
        inc_mean = _expert_route_mean_logprobs(agent, selected_incumbents, device, chunk_size)
        incumbent_logprob = {candidate.env_idx: value for candidate, value in zip(selected_incumbents, inc_mean)}

    pos_values: list[torch.Tensor] = []
    neg_values: list[torch.Tensor] = []
    old_deltas: list[float] = []
    weights: list[float] = []
    policy_pair_count = 0
    incumbent_pair_count = 0
    for pair in selected_pairs:
        local_idx = env_to_local.get(pair.env_idx)
        if local_idx is None:
            continue

        def route_value(kind: str, traj_idx: int) -> torch.Tensor | None:
            if kind == "policy":
                if traj_idx < 0 or traj_idx >= n_traj or route_counts[local_idx, traj_idx].item() <= 0:
                    return None
                return policy_mean[local_idx, traj_idx]
            if kind == "incumbent":
                return incumbent_logprob.get(pair.env_idx)
            return None

        pos = route_value(pair.pos_kind, pair.pos_traj)
        neg = route_value(pair.neg_kind, pair.neg_traj)
        if pos is None or neg is None:
            continue
        pos_values.append(pos)
        neg_values.append(neg)
        old_deltas.append(float(pair.old_delta))
        weights.append(float(pair.weight))
        if pair.incumbent_pair:
            incumbent_pair_count += 1
        else:
            policy_pair_count += 1

    if not pos_values:
        zero = route_sums.sum() * 0.0
        return zero, {
            "bafipo_pref_loss": 0.0,
            "bafipo_pref_pair_count": 0.0,
            "bafipo_policy_pair_count": 0.0,
            "bafipo_incumbent_pair_count": 0.0,
            "bafipo_pref_weight_mean": 0.0,
            "bafipo_pref_logit_mean": 0.0,
        }

    pos_t = torch.stack(pos_values)
    neg_t = torch.stack(neg_values)
    old_delta_t = torch.as_tensor(old_deltas, dtype=pos_t.dtype, device=pos_t.device)
    weight_t = torch.as_tensor(weights, dtype=pos_t.dtype, device=pos_t.device)
    beta = float(bafipo_cfg["beta"])
    logits = beta * ((pos_t - neg_t) - old_delta_t.detach())
    loss = -(weight_t.detach() * F.logsigmoid(logits)).mean()
    return loss, {
        "bafipo_pref_loss": float(loss.detach().cpu().item()),
        "bafipo_pref_pair_count": float(len(weights)),
        "bafipo_policy_pair_count": float(policy_pair_count),
        "bafipo_incumbent_pair_count": float(incumbent_pair_count),
        "bafipo_pref_weight_mean": float(weight_t.detach().mean().cpu().item()),
        "bafipo_pref_logit_mean": float(logits.detach().mean().cpu().item()),
    }



def _parse_int_list_config(value: Any, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        out = []
        for part in value.split(','):
            part = part.strip()
            if part:
                out.append(int(part))
        return out or list(default)
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return list(default)


def _gcbpo_config(cfg: dict[str, Any]) -> dict[str, Any]:
    adv_cfg = cfg.get("advantage", {}) or {}
    offline_cfg = {**adv_cfg, **(cfg.get("offline", {}) or {})}
    method = _offline_method(cfg)
    prefix_default = 0.02 if method in {"gcbpo_prefix", "gcbpo-prefix", "gcbpo_branch_prefix", "gcbpo-branch-prefix"} else 0.0
    return {
        "branch_coef": float(offline_cfg.get("gcbpo_branch_coef", offline_cfg.get("gcbpo_pref_coef", 0.10))),
        "prefix_coef": float(offline_cfg.get("gcbpo_prefix_coef", prefix_default)),
        "beta": float(offline_cfg.get("gcbpo_beta", 1.0)),
        "prefix_lengths": _parse_int_list_config(offline_cfg.get("gcbpo_prefix_lengths"), [1, 2, 3, 5, 8]),
        "branch_completions_per_prefix": int(offline_cfg.get("gcbpo_branch_completions_per_prefix", 1)),
        "branch_pairs_per_instance": int(offline_cfg.get("gcbpo_branch_pairs_per_instance", 8)),
        "top_quantile": float(offline_cfg.get("gcbpo_top_quantile", 0.20)),
        "bottom_quantile": float(offline_cfg.get("gcbpo_bottom_quantile", 0.20)),
        "gap_floor_ratio": float(offline_cfg.get("gcbpo_gap_floor_ratio", 0.01)),
        "pair_weight_max": float(offline_cfg.get("gcbpo_pair_weight_max", 2.0)),
        "soft_weight_coef": float(offline_cfg.get("gcbpo_soft_weight_coef", 0.50)),
        "margin_abs": float(offline_cfg.get("gcbpo_margin_abs", 0.0)),
        "max_instances_per_epoch": int(offline_cfg.get("gcbpo_max_instances_per_epoch", 64)),
        "expert_logprob_chunk_size": int(offline_cfg.get("gcbpo_expert_logprob_chunk_size", offline_cfg.get("frro_expert_logprob_chunk_size", 4096))),
    }


def _gcbpo_env_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    env_cfg = dict(cfg.get("env", {}) or {})
    scale_mode = str(env_cfg.get("reward_distance_scale_mode", ""))
    if scale_mode.startswith("dataset_"):
        env_cfg["reward_distance_scale_mode"] = scale_mode[len("dataset_"):]
    env_cfg["use_fast_env"] = True
    env_cfg["info_level"] = "light"
    return env_cfg


def _env_instance(env):
    candidates = [env, getattr(env, "unwrapped", None), getattr(env, "env", None)]
    current = env
    for _ in range(8):
        current = getattr(current, "env", None)
        if current is None:
            break
        candidates.extend([current, getattr(current, "unwrapped", None)])
    for obj in candidates:
        instance = getattr(obj, "instance", None) if obj is not None else None
        if instance is not None:
            return instance
    return None


def _copy_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value).copy() for key, value in obs.items()}


def _slice_obs_traj(obs: dict[str, np.ndarray], traj_idx: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    action_mask = np.asarray(obs.get("action_mask"))
    n_traj = int(action_mask.shape[0]) if action_mask.ndim > 0 else 1
    for key, value in obs.items():
        arr = np.asarray(value)
        if arr.ndim > 0 and arr.shape[0] == n_traj and int(traj_idx) < arr.shape[0]:
            out[key] = arr[int(traj_idx) : int(traj_idx) + 1].copy()
        else:
            out[key] = arr.copy()
    return out


def _successful_obj_indices(objective_row: np.ndarray, success_row: np.ndarray) -> np.ndarray:
    return np.where(success_row & np.isfinite(objective_row))[0]


def _top_mean_objective(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    n = max(1, int(np.ceil(max(min(float(q), 1.0), 1e-6) * finite.size)))
    return float(np.mean(np.sort(finite)[:n]))


def _complete_gcbpo_prefix_candidates(
    agent: Agent,
    instance: Any,
    prefix_actions: list[int],
    cfg: dict[str, Any],
    *,
    env_idx: int,
    prefix_len: int,
    prefix_weight: float,
    completions: int,
    max_steps: int,
    device: str | torch.device,
    seed: int,
) -> tuple[list[GcbpoBranchCandidate], dict[str, float]]:
    env = make_terran_env(
        instance=instance,
        n_traj=max(1, int(completions)),
        pbrs_config=None,
        **_gcbpo_env_cfg(cfg),
    )
    obs, info = env.reset(seed=int(seed))
    n_traj = int(env.unwrapped.n_traj)
    done = np.zeros(n_traj, dtype=bool)
    step_obs: list[dict[str, np.ndarray]] = []
    step_actions: list[np.ndarray] = []
    step_alive: list[np.ndarray] = []
    prefix_valid = True
    invalid_step = -1
    invalid_action = -1
    with torch.no_grad():
        for step_idx, action in enumerate(prefix_actions):
            alive = ~done
            if not bool(alive.any()):
                prefix_valid = False
                invalid_step = int(step_idx)
                invalid_action = int(action)
                break
            action_i = int(action)
            mask = np.asarray(obs["action_mask"], dtype=bool)
            if action_i < 0 or action_i >= mask.shape[1] or not bool(mask[alive, action_i].all()):
                prefix_valid = False
                invalid_step = int(step_idx)
                invalid_action = action_i
                break
            action_np = np.full(n_traj, action_i, dtype=np.int64)
            step_obs.append(_copy_obs(obs))
            step_actions.append(action_np.copy())
            step_alive.append(alive.copy())
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = done | np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
            if done.any() and step_idx + 1 < len(prefix_actions):
                prefix_valid = False
                invalid_step = int(step_idx)
                invalid_action = action_i
                break
        if prefix_valid and not done.all():
            for _ in range(max(0, int(max_steps) - len(prefix_actions))):
                alive = ~done
                if not bool(alive.any()):
                    break
                obs_batch = stack_observations([obs])
                actions, _, _, _, _ = sample_actions(agent, obs_batch, decode_mode="sample", device=device)
                action_np = actions.squeeze(0).detach().cpu().numpy().astype(np.int64)
                step_obs.append(_copy_obs(obs))
                step_actions.append(action_np.copy())
                step_alive.append(alive.copy())
                obs, reward, terminated, truncated, info = env.step(action_np)
                done = done | np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
                if done.all():
                    break
    if not prefix_valid:
        return [], {
            "prefix_valid": 0.0,
            "invalid_step": float(invalid_step),
            "invalid_action": float(invalid_action),
            "branch_success_count": 0.0,
        }
    success = np.asarray(info.get("success", []), dtype=bool).reshape(-1)
    objective = np.asarray(info.get("objective_distance_km", []), dtype=np.float64).reshape(-1)
    candidates: list[GcbpoBranchCandidate] = []
    for traj_idx in range(min(n_traj, success.size, objective.size)):
        if not bool(success[traj_idx]) or not np.isfinite(objective[traj_idx]):
            continue
        obs_list: list[dict[str, np.ndarray]] = []
        act_list: list[int] = []
        for obs_s, action_s, alive_s in zip(step_obs, step_actions, step_alive):
            if bool(alive_s[traj_idx]):
                obs_list.append(_slice_obs_traj(obs_s, traj_idx))
                act_list.append(int(action_s[traj_idx]))
        if act_list:
            candidates.append(
                GcbpoBranchCandidate(
                    env_idx=int(env_idx),
                    observations=obs_list,
                    actions=act_list,
                    objective=float(objective[traj_idx]),
                    prefix_len=min(int(prefix_len), len(act_list)),
                    prefix_weight=float(prefix_weight),
                )
            )
    return candidates, {
        "prefix_valid": 1.0,
        "invalid_step": -1.0,
        "invalid_action": -1.0,
        "branch_success_count": float(len(candidates)),
    }


def _prepare_gcbpo_preference_pairs(
    agent: Agent,
    batch,
    cfg: dict[str, Any],
    envs,
    expert_buffer: ExpertReplayBuffer | None,
    device: str | torch.device,
    epoch: int,
    seed: int,
) -> tuple[list[GcbpoPreferencePair], list[GcbpoBranchCandidate], dict[str, float]]:
    if expert_buffer is None:
        return [], [], {}
    gcbpo_cfg = _gcbpo_config(cfg)
    num_envs = int(batch.actions.size(1))
    n_traj = int(batch.actions.size(2))
    objective, success, _ = _final_info_arrays(batch.final_infos, num_envs, n_traj)
    old_policy = _policy_route_old_mean_logprobs(batch)
    max_instances = int(gcbpo_cfg["max_instances_per_epoch"])
    env_indices = np.arange(num_envs, dtype=np.int64)
    if max_instances > 0 and max_instances < num_envs:
        rng = np.random.default_rng(int(seed) * 1_000_003 + int(epoch))
        env_indices = np.sort(rng.choice(env_indices, size=max_instances, replace=False))
    top_q = max(min(float(gcbpo_cfg["top_quantile"]), 1.0), 1e-6)
    bottom_q = max(min(float(gcbpo_cfg["bottom_quantile"]), 1.0), 1e-6)
    gap_floor_ratio = max(float(gcbpo_cfg["gap_floor_ratio"]), 0.0)
    pair_weight_max = max(float(gcbpo_cfg["pair_weight_max"]), 0.0)
    soft_weight_coef = max(float(gcbpo_cfg["soft_weight_coef"]), 0.0)
    margin_abs = max(float(gcbpo_cfg["margin_abs"]), 0.0)
    prefix_lengths = [x for x in gcbpo_cfg["prefix_lengths"] if int(x) > 0]
    completions = max(1, int(gcbpo_cfg["branch_completions_per_prefix"]))
    max_steps = int(cfg.get("training", {}).get("rollout_steps", batch.actions.size(0)))
    max_pairs_per_instance = max(0, int(gcbpo_cfg["branch_pairs_per_instance"]))

    branch_candidates: list[GcbpoBranchCandidate] = []
    pending_pairs: list[tuple[int, int, float, bool]] = []
    prefix_valids: list[float] = []
    prefix_lens_used: list[int] = []
    branch_success_counts: list[float] = []
    branch_gap_closes: list[float] = []
    branch_beats_best = 0
    branch_beats_top = 0
    branch_compared = 0
    strong_pair_count = 0
    soft_pair_count = 0
    pair_weights: list[float] = []

    for env_idx in env_indices:
        env_idx_i = int(env_idx)
        succ_idx = _successful_obj_indices(objective[env_idx_i], success[env_idx_i])
        if succ_idx.size < 1:
            continue
        instance_id = _env_instance_id(envs[env_idx_i])
        traj = expert_buffer.trajectory_for_instance(instance_id)
        instance = _env_instance(envs[env_idx_i])
        if traj is None or traj.length <= 0 or instance is None:
            continue
        ref_obj = float(traj.objective_distance_km)
        if not np.isfinite(ref_obj) or ref_obj <= 0.0:
            continue
        succ_obj = objective[env_idx_i, succ_idx].astype(np.float64)
        policy_best = float(np.min(succ_obj))
        policy_top_mean = _top_mean_objective(succ_obj, top_q)
        order = succ_idx[np.argsort(objective[env_idx_i, succ_idx])]
        n_bottom = max(1, int(np.ceil(bottom_q * order.size)))
        bottom = order[-n_bottom:][::-1]
        gap_den_best = max(policy_best - ref_obj, gap_floor_ratio * max(ref_obj, 1e-8), 1e-8)
        soft_den = max(policy_top_mean - ref_obj, gap_floor_ratio * max(ref_obj, 1e-8), 1e-8)
        for prefix_len in prefix_lengths:
            prefix_len_i = int(prefix_len)
            if prefix_len_i > traj.length:
                continue
            candidates, replay_info = _complete_gcbpo_prefix_candidates(
                agent,
                instance,
                traj.actions[:prefix_len_i],
                cfg,
                env_idx=env_idx_i,
                prefix_len=prefix_len_i,
                prefix_weight=0.0,
                completions=completions,
                max_steps=max_steps,
                device=device,
                seed=int(seed) * 1_000_000 + int(epoch) * 10_000 + env_idx_i * 101 + prefix_len_i,
            )
            prefix_valids.append(float(replay_info.get("prefix_valid", 0.0)))
            branch_success_counts.append(float(replay_info.get("branch_success_count", 0.0)))
            if not candidates:
                continue
            best_branch = min(candidates, key=lambda x: x.objective)
            branch_obj = float(best_branch.objective)
            strong = bool(branch_obj < policy_best - margin_abs)
            soft = bool(branch_obj < policy_top_mean - margin_abs)
            if not strong and not soft:
                continue
            branch_compared += 1
            if strong:
                branch_beats_best += 1
            if soft:
                branch_beats_top += 1
            gap_close = 0.0
            if strong:
                gap_close = float(np.clip((policy_best - branch_obj) / gap_den_best, 0.0, pair_weight_max))
            else:
                gap_close = soft_weight_coef * float(np.clip((policy_top_mean - branch_obj) / soft_den, 0.0, pair_weight_max))
            if gap_close <= 1e-8:
                continue
            best_branch.prefix_weight = gap_close if strong else gap_close * 0.5
            branch_idx = len(branch_candidates)
            branch_candidates.append(best_branch)
            prefix_lens_used.append(prefix_len_i)
            branch_gap_closes.append(gap_close)
            worse = bottom[objective[env_idx_i, bottom] > branch_obj + margin_abs]
            if worse.size == 0:
                worse = succ_idx[objective[env_idx_i, succ_idx] > branch_obj + margin_abs]
                worse = worse[np.argsort(objective[env_idx_i, worse])[::-1]]
            for k, neg in enumerate(worse[:max_pairs_per_instance]):
                gap = float(objective[env_idx_i, int(neg)] - branch_obj)
                if gap <= 1e-8:
                    continue
                scale = gap_den_best if strong else soft_den
                weight = gap_close * float(np.clip(gap / scale, 0.0, pair_weight_max))
                if weight <= 1e-8:
                    continue
                pending_pairs.append((branch_idx, int(neg), float(weight), bool(strong)))
                pair_weights.append(float(weight))
                if strong:
                    strong_pair_count += 1
                else:
                    soft_pair_count += 1

    if branch_candidates:
        chunk_size = max(1, int(gcbpo_cfg["expert_logprob_chunk_size"]))
        with torch.no_grad():
            old_branch = _expert_route_mean_logprobs(agent, branch_candidates, device, chunk_size).detach().float().cpu().numpy()
        for candidate, old_val in zip(branch_candidates, old_branch):
            candidate.old_mean_logprob = float(old_val)

    pairs: list[GcbpoPreferencePair] = []
    for branch_idx, neg_traj, weight, strong in pending_pairs:
        candidate = branch_candidates[branch_idx]
        env_idx_i = int(candidate.env_idx)
        pairs.append(
            GcbpoPreferencePair(
                env_idx=env_idx_i,
                branch_idx=int(branch_idx),
                neg_traj=int(neg_traj),
                old_delta=float(candidate.old_mean_logprob - old_policy[env_idx_i, int(neg_traj)]),
                weight=float(weight),
                strong=bool(strong),
            )
        )

    info = {
        "gcbpo_branch_candidates": float(len(branch_candidates)),
        "gcbpo_pref_pairs": float(len(pairs)),
        "gcbpo_strong_pairs": float(strong_pair_count),
        "gcbpo_soft_pairs": float(soft_pair_count),
        "gcbpo_branch_beats_best_rate": float(branch_beats_best / max(branch_compared, 1)),
        "gcbpo_branch_beats_top_mean_rate": float(branch_beats_top / max(branch_compared, 1)),
        "gcbpo_branch_gap_close_mean": float(np.mean(branch_gap_closes)) if branch_gap_closes else 0.0,
        "gcbpo_prefix_valid_rate": float(np.mean(prefix_valids)) if prefix_valids else 0.0,
        "gcbpo_prefix_len_mean": float(np.mean(prefix_lens_used)) if prefix_lens_used else 0.0,
        "gcbpo_branch_success_count_mean": float(np.mean(branch_success_counts)) if branch_success_counts else 0.0,
        "gcbpo_pair_weight_mean": float(np.mean(pair_weights)) if pair_weights else 0.0,
        "gcbpo_branch_coef": float(gcbpo_cfg["branch_coef"]),
        "gcbpo_prefix_coef": float(gcbpo_cfg["prefix_coef"]),
    }
    return pairs, branch_candidates, info


def _compute_gcbpo_preference_loss(
    agent: Agent,
    batch,
    pairs: list[GcbpoPreferencePair],
    candidates: list[GcbpoBranchCandidate],
    cfg: dict[str, Any],
    env_indices: np.ndarray,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    gcbpo_cfg = _gcbpo_config(cfg)
    selected_envs = {int(idx) for idx in np.asarray(env_indices, dtype=np.int64).reshape(-1)}
    selected_pairs = [pair for pair in pairs if pair.env_idx in selected_envs and pair.weight > 0.0]
    selected_prefix = [idx for idx, candidate in enumerate(candidates) if candidate.env_idx in selected_envs and candidate.prefix_weight > 0.0]
    if not selected_pairs and not selected_prefix:
        zero = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
        return zero, zero, {
            "gcbpo_pref_loss": 0.0,
            "gcbpo_prefix_loss": 0.0,
            "gcbpo_pref_pair_count": 0.0,
            "gcbpo_strong_pair_count": 0.0,
            "gcbpo_soft_pair_count": 0.0,
            "gcbpo_pref_weight_mean": 0.0,
            "gcbpo_pref_logit_mean": 0.0,
            "gcbpo_prefix_route_count": 0.0,
        }

    env_indices = np.asarray(env_indices, dtype=np.int64)
    env_to_local = {int(env_idx): local_idx for local_idx, env_idx in enumerate(env_indices)}
    total_steps = int(batch.actions.size(0))
    n_traj = int(batch.actions.size(2))
    cached_state = agent.backbone.encode(_slice_obs_by_env(batch.observations[0], env_indices))
    route_sums = torch.zeros((len(env_indices), n_traj), dtype=batch.old_logprobs.dtype, device=device)
    route_counts = torch.zeros_like(route_sums)
    for step in range(total_steps):
        obs_mb = _slice_obs_by_env(batch.observations[step], env_indices)
        actions = batch.actions[step, env_indices].long()
        _, new_logprob, _, _, _ = agent.get_action_and_value_cached(obs_mb, action=actions, state=cached_state)
        valid = batch.valid[step, env_indices].to(dtype=new_logprob.dtype)
        route_sums = route_sums + new_logprob * valid
        route_counts = route_counts + valid
    policy_mean = route_sums / route_counts.clamp_min(1.0)

    needed_branch_indices = sorted({pair.branch_idx for pair in selected_pairs} | set(selected_prefix))
    branch_logprob: dict[int, torch.Tensor] = {}
    if needed_branch_indices:
        needed_candidates = [candidates[idx] for idx in needed_branch_indices]
        chunk_size = max(1, int(gcbpo_cfg["expert_logprob_chunk_size"]))
        branch_mean = _expert_route_mean_logprobs(agent, needed_candidates, device, chunk_size)
        branch_logprob = {idx: value for idx, value in zip(needed_branch_indices, branch_mean)}

    pos_values: list[torch.Tensor] = []
    neg_values: list[torch.Tensor] = []
    old_deltas: list[float] = []
    weights: list[float] = []
    strong_count = 0
    soft_count = 0
    for pair in selected_pairs:
        local_idx = env_to_local.get(pair.env_idx)
        if local_idx is None or pair.neg_traj < 0 or pair.neg_traj >= n_traj:
            continue
        if route_counts[local_idx, pair.neg_traj].item() <= 0:
            continue
        pos = branch_logprob.get(pair.branch_idx)
        if pos is None:
            continue
        pos_values.append(pos)
        neg_values.append(policy_mean[local_idx, pair.neg_traj])
        old_deltas.append(float(pair.old_delta))
        weights.append(float(pair.weight))
        if pair.strong:
            strong_count += 1
        else:
            soft_count += 1

    if pos_values:
        pos_t = torch.stack(pos_values)
        neg_t = torch.stack(neg_values)
        old_delta_t = torch.as_tensor(old_deltas, dtype=pos_t.dtype, device=pos_t.device)
        weight_t = torch.as_tensor(weights, dtype=pos_t.dtype, device=pos_t.device)
        logits = float(gcbpo_cfg["beta"]) * ((pos_t - neg_t) - old_delta_t.detach())
        pref_loss = -(weight_t.detach() * F.logsigmoid(logits)).mean()
        pref_weight_mean = float(weight_t.detach().mean().cpu().item())
        pref_logit_mean = float(logits.detach().mean().cpu().item())
    else:
        pref_loss = route_sums.sum() * 0.0
        pref_weight_mean = 0.0
        pref_logit_mean = 0.0

    prefix_values: list[torch.Tensor] = []
    prefix_weights: list[float] = []
    for branch_idx in selected_prefix:
        candidate = candidates[branch_idx]
        value = branch_logprob.get(branch_idx)
        if value is None:
            continue
        prefix_values.append(value)
        prefix_weights.append(float(candidate.prefix_weight))
    if prefix_values:
        prefix_t = torch.stack(prefix_values)
        prefix_weight_t = torch.as_tensor(prefix_weights, dtype=prefix_t.dtype, device=prefix_t.device)
        prefix_loss = -(prefix_weight_t.detach() * prefix_t).mean()
    else:
        prefix_loss = route_sums.sum() * 0.0

    return pref_loss, prefix_loss, {
        "gcbpo_pref_loss": float(pref_loss.detach().cpu().item()),
        "gcbpo_prefix_loss": float(prefix_loss.detach().cpu().item()),
        "gcbpo_pref_pair_count": float(len(weights)),
        "gcbpo_strong_pair_count": float(strong_count),
        "gcbpo_soft_pair_count": float(soft_count),
        "gcbpo_pref_weight_mean": pref_weight_mean,
        "gcbpo_pref_logit_mean": pref_logit_mean,
        "gcbpo_prefix_route_count": float(len(prefix_values)),
    }


def _load_expert_buffer(cfg: dict[str, Any], seed: int, debug_enabled: bool, debug_file) -> ExpertReplayBuffer | None:
    offline_cfg = cfg.get("offline", {}) or {}
    method = _offline_method(cfg)
    need_archive = _requires_expert_routes(method) or _reference_advantage_enabled(cfg) or _frro_enabled(cfg)
    if method in {"", "none", "ppo"} and not need_archive:
        return None
    if _is_sl_ppo_method(method):
        need_archive = _reference_advantage_enabled(cfg)
    if _is_frro_method(method):
        need_archive = True
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
        seed=seed + int(offline_cfg.get("replay_seed_offset", 17_000)),
    )
    _debug_log(
        debug_enabled,
        debug_file,
        "[OfflineArchive] "
        f"method={method} records={stats['records_seen']} trajectories={stats['trajectories']} "
        f"invalid={stats['invalid_records']} steps={stats['steps']} "
        f"avg_steps={stats['avg_steps_per_route']:.3f} "
        f"success_rate={stats.get('expert_replay_success_rate', 0.0):.6f} "
        f"valid_ratio={stats.get('expert_action_valid_ratio', 0.0):.6f} "
        f"obj_error_max={stats.get('expert_env_replay_obj_error_max', float('nan')):.6g} "
        f"route_count_mean={stats.get('expert_route_count_mean', 0.0):.3f} "
        f"solution_path={solution_path}",
    )
    buffer = ExpertReplayBuffer(
        trajectories,
        seed=seed + int(offline_cfg.get("replay_seed_offset", 17_000)),
        replay_stats=stats,
    )
    return buffer


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
    del epoch
    offline_cfg = cfg.get("offline", {}) or {}
    batch_size = int(offline_cfg.get("bc_batch_size", 256))
    max_grad_norm = float(cfg.get("training", {}).get("max_grad_norm", 1.0))
    losses: list[float] = []
    accs: list[float] = []
    action_accs: list[float] = []
    entropies: list[float] = []
    steps = 0
    agent.train()
    for _ in range(int(updates)):
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp_enabled):
            loss, info = compute_bc_loss(agent, expert_buffer, batch_size=batch_size, device=device)
        _backward(float(coef) * loss, scaler, amp_enabled)
        _optimizer_step(optimizer, agent, max_grad_norm, scaler, amp_enabled)
        losses.append(float(info["bc_loss"]))
        accs.append(float(info["bc_accuracy"]))
        action_accs.append(float(info.get("bc_action_accuracy", info["bc_accuracy"])))
        entropies.append(float(info["bc_entropy"]))
        steps += int(info["bc_steps"])
    return {
        "bc_loss": float(np.mean(losses)) if losses else 0.0,
        "bc_accuracy": float(np.mean(accs)) if accs else 0.0,
        "bc_action_accuracy": float(np.mean(action_accs)) if action_accs else 0.0,
        "bc_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "bc_steps": int(steps),
        "bc_coef": float(coef),
        "offline_updates": int(updates),
    }


def _run_route_bc_updates(*args, **kwargs) -> dict[str, Any]:
    raise RuntimeError("Route-local/RSEG updates were removed from this cleaned branch.")

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
    allowed_methods = {"ppo", "bc_ppo", "bc-ppo", "awbc", "awbc_ppo", "awbc-ppo", "dapg"}
    if offline_method not in allowed_methods:
        raise ValueError(
            f"Unsupported offline.method={offline_method!r}. "
            "This cleaned branch keeps only ppo, bc_ppo, awbc/awbc_ppo, and dapg."
        )
    model_cfg = cfg.get("model", {})
    critic_cfg = _decomposed_critic_config(cfg)
    advantage_mode_name = str(critic_cfg.get("advantage_mode", "total")).strip().lower()
    use_pomo_trajectory_advantage = advantage_mode_name in {"pomo_trajectory", "pomo", "trajectory_pomo"}
    use_decomposed_critic = bool(model_cfg.get("use_decomposed_critic", critic_cfg.get("use_decomposed_critic", True)))
    if use_pomo_trajectory_advantage:
        raise ValueError("POMO trajectory advantage was removed from this cleaned branch.")
    if use_pomo_trajectory_advantage and use_decomposed_critic:
        raise ValueError("advantage_mode=pomo_trajectory requires use_decomposed_critic=false")
    cfg.setdefault("env", {})["use_fast_env"] = True
    cfg["env"].setdefault("info_level", "light")
    run_name = str(cfg.get("run_name", "O2O_TERRAN_FULL"))
    data_cfg = cfg["data"]
    num_customers = int(data_cfg.get("num_customers", 50))
    num_cs = int(data_cfg.get("num_charging_stations", 10))
    train_dataset_path = data_cfg.get("train_dataset_path") or data_cfg.get("instance_dataset_path") or data_cfg.get("fixed_train_path")
    _validate_dataset_metadata(
        train_dataset_path,
        num_customers=num_customers,
        num_charging_stations=num_cs,
        label="train",
    )
    _validate_dataset_metadata(
        eval_cfg.get("eval_path"),
        num_customers=num_customers,
        num_charging_stations=num_cs,
        label="eval",
    )
    expert_dataset_path = offline_cfg.get("expert_dataset_path")
    if expert_dataset_path not in (None, "", train_dataset_path):
        _validate_dataset_metadata(
            expert_dataset_path,
            num_customers=num_customers,
            num_charging_stations=num_cs,
            label="expert",
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dynamic_decision_delta_k = bool(
        model_cfg.get(
            "dynamic_decision_delta_k",
            model_cfg.get("dynamic_delta_k", model_cfg.get("delta_k", True)),
        )
    )
    dynamic_decision_delta_v = bool(
        model_cfg.get(
            "dynamic_decision_delta_v",
            model_cfg.get("dynamic_delta_v", model_cfg.get("delta_v", True)),
        )
    )
    dynamic_decision_delta_action_key = bool(
        model_cfg.get(
            "dynamic_decision_delta_action_key",
            model_cfg.get("dynamic_delta_action_key", model_cfg.get("delta_action_key", True)),
        )
    )
    dynamic_decision_action_bias = bool(
        model_cfg.get(
            "dynamic_decision_action_bias",
            model_cfg.get("dynamic_action_bias", model_cfg.get("action_bias", True)),
        )
    )

    def _make_agent() -> Agent:
        return Agent(
            embedding_dim=int(model_cfg.get("embedding_dim", 256)),
            tanh_clipping=float(model_cfg.get("tanh_clipping", 15.0)),
            n_encode_layers=int(model_cfg.get("n_encode_layers", 2)),
            device=device,
            use_graph_token=bool(model_cfg.get("use_graph_token", True)),
            use_dynamic_decision_encoder=bool(
                model_cfg.get("use_dynamic_decision_encoder", False)
            ),
            dynamic_decision_heads=int(model_cfg.get("dynamic_decision_heads", 4)),
            dynamic_decision_delta_k=dynamic_decision_delta_k,
            dynamic_decision_delta_v=dynamic_decision_delta_v,
            dynamic_decision_delta_action_key=dynamic_decision_delta_action_key,
            dynamic_decision_action_bias=dynamic_decision_action_bias,
            use_decomposed_critic=use_decomposed_critic,
        ).to(device)

    agent = _make_agent()
    init_checkpoint_info: dict[str, Any] = {}
    init_checkpoint_path = (
        offline_cfg.get("init_checkpoint_path")
        or offline_cfg.get("initial_checkpoint_path")
        or train_cfg.get("init_checkpoint_path")
        or train_cfg.get("initial_checkpoint_path")
    )
    if init_checkpoint_path:
        init_checkpoint_info = _load_agent_checkpoint(
            agent,
            init_checkpoint_path,
            device,
            strict=bool(offline_cfg.get("init_checkpoint_strict", True)),
        )
    reference_agent: Agent | None = None
    reference_checkpoint_info: dict[str, Any] = {}
    hard_ref_kl_requested = _is_hard_method(offline_method) or float(
        offline_cfg.get("hard_ref_kl_coef", offline_cfg.get("lambda_ref_kl", 0.0)) or 0.0
    ) > 0.0
    if hard_ref_kl_requested:
        reference_agent = _make_agent()
        reference_checkpoint_path = offline_cfg.get("reference_checkpoint_path") or init_checkpoint_path
        if reference_checkpoint_path:
            reference_checkpoint_info = _load_agent_checkpoint(
                reference_agent,
                reference_checkpoint_path,
                device,
                strict=bool(offline_cfg.get("reference_checkpoint_strict", True)),
            )
        else:
            reference_agent.load_state_dict(agent.state_dict())
            reference_checkpoint_info = {"checkpoint_path": "current_initial_agent"}
        reference_agent.eval()
        for parameter in reference_agent.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        agent.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        eps=1e-5,
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    gamma = float(train_cfg.get("gamma", 0.99))
    epochs = int(train_cfg.get("epochs", 500))
    num_envs_cfg = int(train_cfg.get("num_envs_per_gpu", 128))
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
    use_route_segmented_gae = bool(train_cfg.get("use_route_segmented_gae", False))
    if use_route_segmented_gae:
        raise ValueError("Route-segmented GAE was removed from this cleaned branch.")
    use_oracle_ordering_hint = bool(
        train_cfg.get(
            "use_oracle_ordering_hint",
            offline_cfg.get("use_oracle_ordering_hint", False),
        )
    )
    if use_oracle_ordering_hint:
        raise ValueError("Oracle ordering hint was removed from this cleaned branch.")
    oracle_refresh_logprobs = bool(
        train_cfg.get(
            "oracle_ordering_refresh_logprobs",
            offline_cfg.get("oracle_ordering_refresh_logprobs", True),
        )
    )
    amp_enabled = _amp_enabled(train_cfg, device)
    scaler = _new_grad_scaler(amp_enabled)
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    record_eval_median = str(
        os.environ.get("O2O_RECORD_MEDIAN_EVAL", eval_cfg.get("record_median_eval", True))
    ).strip().lower() not in {"0", "false", "no", "off"}

    initial_env_start = time.perf_counter()
    use_priority_sampler = bool(offline_cfg.get("use_priority_sampler", False))
    if use_priority_sampler:
        train_dataset_path = data_cfg.get("train_dataset_path") or data_cfg.get("instance_dataset_path") or data_cfg.get("fixed_train_path")
        if train_dataset_path in (None, ""):
            raise ValueError("HAPS priority sampler requires a fixed data.train_dataset_path")
        base_pool = FixedDatasetInstancePool(
            dataset_path=train_dataset_path,
            num_customers=num_customers,
            num_charging_stations=num_cs,
            seed=seed,
            sample_mode=str(data_cfg.get("train_sample_mode", "shuffle_cycle")),
        )
        _configure_dataset_reward_scale(cfg, base_pool)
        references = _load_reference_metrics(offline_cfg.get("expert_solution_path") or offline_cfg.get("expert_csv_path"))
        if not references:
            raise ValueError("HAPS priority sampler requires offline.expert_solution_path with objective/vehicle references")
        pool = HapsPrioritySampler(
            base_pool,
            references,
            batch_size=num_envs_cfg,
            seed=seed + int(offline_cfg.get("priority_seed_offset", 31_000)),
            cfg=cfg,
        )
        pbrs_config = build_pbrs_config(cfg)
        env_cfg = dict(cfg.get("env", {}) or {})
        if bool(env_cfg.get("use_fast_env", True)):
            env_cfg.setdefault("info_level", "light")
        envs = [
            make_terran_env(
                instance_sampler=pool.sample,
                n_traj=int(train_cfg.get("n_traj", 50)),
                pbrs_config=pbrs_config,
                **env_cfg,
            )
            for _ in range(num_envs_cfg)
        ]
    else:
        envs, pool = make_envs(cfg, seed)
    initial_env_pool_time_s = time.perf_counter() - initial_env_start

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
        "approx_kl",
        "clip_fraction",
        "value_loss_total",
        "value_loss_boundary",
        "value_loss_internal",
        "value_consistency_loss",
        "explained_variance_total",
        "explained_variance_boundary",
        "explained_variance_internal",
        "return_total_mean",
        "return_boundary_mean",
        "return_internal_mean",
        "adv_total_mean",
        "adv_total_std",
        "adv_boundary_mean",
        "adv_boundary_std",
        "adv_internal_mean",
        "adv_internal_std",
        "adv_actor_mean",
        "adv_actor_std",
        "pomo_reward_mean",
        "pomo_reward_std",
        "pomo_baseline_mean",
        "pomo_within_reward_std_mean",
        "pomo_within_reward_std_p10",
        "pomo_within_reward_std_p90",
        "pomo_valid_traj_ratio",
        "pomo_success_traj_ratio",
        "pomo_objective_used_ratio",
        "pomo_adv_raw_mean",
        "pomo_adv_raw_std",
        "pomo_adv_norm_mean",
        "pomo_adv_norm_std",
        "pomo_valid_traj_count_mean",
        "boundary_distance_mean",
        "internal_distance_mean",
        "boundary_share",
        "internal_share",
        "reward_decomposition_max_abs_error",
        "reward_decomposition_mean_abs_error",
        "episode_decomposition_max_abs_error",
        "episode_decomposition_mean_abs_error",
        "advantage_mode",
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
        "use_route_segmented_gae",
        "route_boundary_count_mean",
        "route_boundary_step_ratio",
        "expert_replay_success_rate",
        "expert_action_valid_ratio",
        "expert_env_replay_obj_error_mean",
        "expert_env_replay_obj_error_max",
        "expert_route_count_mean",
        "route_set_num_permutations_sampled",
        "route_set_target_valid_ratio",
        "route_local_customer_count_mean",
        "route_local_cs_candidate_count_mean",
        "route_local_objective_mean",
        "same_route_examples",
        "same_route_positive_pairs_mean",
        "same_route_negative_pairs_mean",
        "rseg_structure_examples",
        "rseg_route_count_mean",
        "rseg_route_count_max",
        "rseg_route_start_count_mean",
        "rseg_progress_baselines",
        "priority_mean",
        "priority_std",
        "priority_min",
        "priority_max",
        "sampled_priority_mean",
        "gap_score_mean",
        "gap_score_std",
        "vehicle_score_mean",
        "vehicle_score_std",
        "sampled_gap_mean",
        "sampled_vehicle_gap_mean",
        "priority_uniform_fraction",
        "priority_weighted_fraction",
        "unique_instance_ratio",
        "stale_bonus_mean",
        "num_priority_updates_mean",
        "bc_loss",
        "bc_accuracy",
        "bc_action_accuracy",
        "bc_entropy",
        "bc_coef",
        "route_start_loss",
        "internal_successor_loss",
        "route_close_loss",
        "depot_multilabel_active_ratio",
        "start_target_set_size_mean",
        "start_target_set_size_max",
        "awbc_loss",
        "awbc_weight_mean",
        "awbc_weight_std",
        "awbc_active_ratio",
        "awbc_expert_better_ratio",
        "hard_ref_kl_loss",
        "hard_ref_match",
        "hard_ref_bc_action_accuracy",
        "hard_ref_entropy",
        "hard_ref_kl_coef",
        "hard_ref_temperature",
        "hard_demo_loss",
        "hard_demo_coef",
        "hard_eta",
        "hard_normalize",
        "hard_baseline",
        "dapg_demo_gate_mean",
        "dapg_demo_gate_std",
        "dapg_demo_active_ratio",
        "expert_better_ratio",
        "dapg_memory_better_rate",
        "dapg_memory_gap_mean",
        "bafipo_pref_loss",
        "bafipo_pref_pair_count",
        "bafipo_policy_pair_count",
        "bafipo_incumbent_pair_count",
        "bafipo_quality_gate_mean",
        "bafipo_memory_gate_mean",
        "bafipo_spread_gate_mean",
        "bafipo_incumbent_beats_best_rate",
        "bafipo_incumbent_beats_mean_rate",
        "bafipo_pref_weight_mean",
        "bafipo_pref_logit_mean",
        "bafipo_pref_coef",
        "bafipo_minibatches_per_ppo_epoch",
        "gcbpo_pref_loss",
        "gcbpo_prefix_loss",
        "gcbpo_pref_pair_count",
        "gcbpo_strong_pair_count",
        "gcbpo_soft_pair_count",
        "gcbpo_branch_candidates",
        "gcbpo_branch_beats_best_rate",
        "gcbpo_branch_beats_top_mean_rate",
        "gcbpo_branch_gap_close_mean",
        "gcbpo_prefix_valid_rate",
        "gcbpo_prefix_len_mean",
        "gcbpo_pref_weight_mean",
        "gcbpo_pref_logit_mean",
        "gcbpo_branch_coef",
        "gcbpo_prefix_coef",
        "bc_steps",
        "route_bc_loss",
        "route_bc_entropy",
        "route_bc_coef",
        "route_bc_count",
        "route_bc_step_count",
        "route_bc_avg_route_len",
        "same_route_loss",
        "same_route_accuracy",
        "same_route_pos_accuracy",
        "same_route_neg_accuracy",
        "same_route_pairs",
        "same_route_coef",
        "rseg_loss",
        "rseg_edge_loss",
        "rseg_edge_accuracy",
        "rseg_edge_steps",
        "rseg_start_loss",
        "rseg_start_recall",
        "rseg_start_precision",
        "rseg_start_jaccard",
        "rseg_close_loss",
        "rseg_close_accuracy",
        "rseg_close_positive_rate",
        "rseg_count_loss",
        "rseg_count_mae",
        "rseg_count_pred_mean",
        "rseg_progress_penalty_mean",
        "rseg_progress_active_ratio",
        "rseg_progress_gap_mean",
        "rseg_progress_points",
        "rseg_progress_coef",
        "route_quality_reward_mean",
        "route_quality_reward_sum",
        "route_quality_modified_steps",
        "route_quality_penalty_mean",
        "route_quality_gap_km_mean",
        "route_quality_gap_scaled_mean",
        "route_quality_active_ratio",
        "route_quality_routes",
        "route_quality_cache_hit_rate",
        "route_quality_cache_size",
        "route_quality_coef",
        "oracle_ordering_hint_enabled",
        "oracle_ordering_hint_routes",
        "oracle_ordering_hint_changed_routes",
        "oracle_ordering_hint_route_changed_ratio",
        "oracle_ordering_hint_customer_steps",
        "oracle_ordering_hint_flagged_steps",
        "oracle_ordering_hint_step_ratio",
        "oracle_ordering_hint_action_match_ratio",
        "notclose_loss",
        "notclose_points",
        "notclose_candidate_points",
        "notclose_active_ratio",
        "notclose_depot_action_ratio",
        "notclose_avg_route_size_mean",
        "notclose_coef",
        "member_loss",
        "member_loss_points",
        "member_candidate_points",
        "member_loss_active_ratio",
        "member_matched_route_count",
        "member_matched_traj_count",
        "member_k_equal_traj_count",
        "member_k2_matched_jaccard",
        "member_k2_match_gap",
        "member_coef",
        "anchor_loss",
        "anchor_loss_points",
        "anchor_candidate_points",
        "anchor_violation_ratio",
        "anchor_k_equal_traj_count",
        "anchor_k2_matched_jaccard",
        "anchor_k2_match_gap",
        "anchor_coef",
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
        "sl_ref_memory_gate_mean",
        "sl_ref_memory_gate_std",
        "sl_ref_memory_better_rate",
        "sl_ref_memory_gap_mean",
        "sl_ref_base_gap_ratio_mean",
        "frro_adv_mean",
        "frro_adv_std",
        "frro_positive_mean",
        "frro_positive_std",
        "frro_negative_mean",
        "frro_negative_std",
        "frro_gate_mean",
        "frro_gate_std",
        "frro_falsified_rate",
        "frro_best_gap_mean",
        "frro_expert_loss",
        "frro_expert_ratio_mean",
        "frro_expert_ratio_std",
        "frro_expert_clip_frac",
        "frro_expert_adv_mean",
        "frro_expert_adv_std",
        "frro_expert_gate_mean",
        "frro_expert_gate_std",
        "frro_expert_num_routes",
        "frro_expert_weight",
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
        "eval_avg_min_objective_distance_km",
        "eval_avg_median_objective_distance_km",
        "eval_avg_vehicle_count",
        "eval_avg_min_vehicle_count",
        "eval_avg_median_vehicle_count",
        "eval_feasible_rate",
        "eval_traj_feasible_rate",
        "eval_avg_feasible_traj_count",
        "eval_avg_runtime_s",
        "eval_num_instances",
        "eval_n_traj",
        "eval_batch_size",
        "eval_num_batches",
        "eval_decode_mode",
        "eval_info_level",
        "eval_save_routes",
        "eval_status",
        "eval_gap_mean",
        "eval_gap_median",
        "eval_gap_p75",
        "eval_gap_p90",
        "eval_gap_p95",
        "eval_gap_p99",
        "eval_gap_gt50_count",
        "eval_gap_gt100_count",
        "eval_top10_hard_gap_mean",
        "eval_vehicle_gap_mean",
        "eval_vehicle_gap_median",
        "eval_vehicle_gap_p90",
        "eval_vehicle_gap_p95",
        "eval_vehicle_gap_gt0_count",
        *[
            field
            for k in range(1, 5)
            for field in (
                f"eval_gap_expertK{k}_mean",
                f"eval_obj_expertK{k}_mean",
                f"eval_gap_expertK{k}_count",
                f"eval_gap_policyK{k}_mean",
                f"eval_obj_policyK{k}_mean",
                f"eval_gap_policyK{k}_count",
            )
        ],
    ]
    eval_fields = [
        "epoch",
        "eval_avg_objective_distance_km",
        "eval_avg_min_objective_distance_km",
        "eval_avg_median_objective_distance_km",
        "eval_avg_vehicle_count",
        "eval_avg_min_vehicle_count",
        "eval_avg_median_vehicle_count",
        "eval_feasible_rate",
        "eval_traj_feasible_rate",
        "eval_avg_feasible_traj_count",
        "eval_avg_runtime_s",
        "eval_num_instances",
        "eval_n_traj",
        "eval_batch_size",
        "eval_num_batches",
        "eval_decode_mode",
        "eval_info_level",
        "eval_save_routes",
        "eval_status",
        "eval_gap_mean",
        "eval_gap_median",
        "eval_gap_p75",
        "eval_gap_p90",
        "eval_gap_p95",
        "eval_gap_p99",
        "eval_gap_gt50_count",
        "eval_gap_gt100_count",
        "eval_top10_hard_gap_mean",
        "eval_vehicle_gap_mean",
        "eval_vehicle_gap_median",
        "eval_vehicle_gap_p90",
        "eval_vehicle_gap_p95",
        "eval_vehicle_gap_gt0_count",
        *[
            field
            for k in range(1, 5)
            for field in (
                f"eval_gap_expertK{k}_mean",
                f"eval_obj_expertK{k}_mean",
                f"eval_gap_expertK{k}_count",
                f"eval_gap_policyK{k}_mean",
                f"eval_obj_policyK{k}_mean",
                f"eval_gap_policyK{k}_count",
            )
        ],
    ]
    if not record_eval_median:
        median_fields = {
            "eval_avg_median_objective_distance_km",
            "eval_avg_median_vehicle_count",
        }
        train_fields = [field for field in train_fields if field not in median_fields]
        eval_fields = [field for field in eval_fields if field not in median_fields]

    with (
        log_path.open("w", newline="", encoding="utf-8") as f,
        eval_log_path.open("w", newline="", encoding="utf-8") as ef,
        debug_log_path.open("w", encoding="utf-8") as df,
    ):
        writer = csv.DictWriter(f, fieldnames=train_fields, extrasaction="ignore")
        eval_writer = csv.DictWriter(ef, fieldnames=eval_fields, extrasaction="ignore")
        writer.writeheader()
        eval_writer.writeheader()
        expert_buffer = _load_expert_buffer(cfg, seed, debug_enabled, df)
        best_eval_objective = float("inf")
        best_eval_epoch = 0
        policy_best_objectives: dict[str, float] = {}
        _debug_log(
            debug_enabled,
            df,
            f"[Init] run={run_name} seed={seed} device={device} epochs={epochs} "
            f"n_traj={train_cfg.get('n_traj', 50)} rollout_steps={rollout_steps} "
            f"num_envs={train_cfg.get('num_envs_per_gpu', 128)} minibatches={num_minibatches} "
            f"accum_grad={gradient_accumulation_steps} "
            f"n_encode_layers={model_cfg.get('n_encode_layers', 2)} "
            f"use_graph_token={model_cfg.get('use_graph_token', True)} "
            f"use_dynamic_decision_encoder={model_cfg.get('use_dynamic_decision_encoder', False)} "
            f"dde_flags=k{int(dynamic_decision_delta_k)}_v{int(dynamic_decision_delta_v)}_ak{int(dynamic_decision_delta_action_key)}_bias{int(dynamic_decision_action_bias)} "
            f"use_decomposed_critic={use_decomposed_critic} "
            f"advantage_mode={advantage_mode_name} "
            f"mixed_precision={amp_enabled} "
            f"offline_method={offline_method} use_gae={use_gae} gae_lambda={gae_lambda} "
            f"use_route_segmented_gae={use_route_segmented_gae} "
            f"use_oracle_ordering_hint={use_oracle_ordering_hint} "
            f"use_priority_sampler={use_priority_sampler} "
            f"expert_steps={expert_buffer.num_steps if expert_buffer is not None else 0} "
            f"initial_env_pool_time_s={initial_env_pool_time_s:.3f} "
            f"eval_interval={eval_interval} eval_n_traj={eval_cfg.get('eval_n_traj', 50)} "
            f"eval_batch_size={eval_cfg.get('eval_batch_size', 1000)}",
        )
        if init_checkpoint_info or reference_checkpoint_info:
            _debug_log(
                debug_enabled,
                df,
                "[CheckpointInit] "
                f"init={init_checkpoint_info.get('checkpoint_path', '')} "
                f"init_epoch={init_checkpoint_info.get('epoch', '')} "
                f"reference={reference_checkpoint_info.get('checkpoint_path', '')} "
                f"reference_epoch={reference_checkpoint_info.get('epoch', '')} "
                f"init_missing={len(init_checkpoint_info.get('missing_keys', [])) if init_checkpoint_info else 0} "
                f"init_unexpected={len(init_checkpoint_info.get('unexpected_keys', [])) if init_checkpoint_info else 0} "
                f"ref_missing={len(reference_checkpoint_info.get('missing_keys', [])) if reference_checkpoint_info else 0} "
                f"ref_unexpected={len(reference_checkpoint_info.get('unexpected_keys', [])) if reference_checkpoint_info else 0}",
            )
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            pbrs_scale = pbrs_scale_for_epoch(cfg, epoch, epochs)
            set_pbrs_reward_scale(envs, pbrs_scale)
            agent.train()
            offline_updates = 0
            bc_info: dict[str, Any] = {}
            haps_info: dict[str, Any] = {}
            sl_info: dict[str, Any] = {}
            bafipo_info: dict[str, Any] = {}
            gcbpo_info: dict[str, Any] = {}
            adv_info: dict[str, Any] = {}
            progress_info: dict[str, Any] = {}
            ppo_stats_info: dict[str, Any] = {"approx_kl": "", "clip_fraction": ""}
            decomposed_info: dict[str, Any] = {
                "value_loss_total": 0.0,
                "value_loss_boundary": 0.0,
                "value_loss_internal": 0.0,
                "value_consistency_loss": 0.0,
                "explained_variance_total": np.nan,
                "explained_variance_boundary": np.nan,
                "explained_variance_internal": np.nan,
                "return_total_mean": 0.0,
                "return_boundary_mean": 0.0,
                "return_internal_mean": 0.0,
                "adv_total_mean": 0.0,
                "adv_total_std": 0.0,
                "adv_boundary_mean": 0.0,
                "adv_boundary_std": 0.0,
                "adv_internal_mean": 0.0,
                "adv_internal_std": 0.0,
                "adv_actor_mean": 0.0,
                "adv_actor_std": 0.0,
                "pomo_reward_mean": "",
                "pomo_reward_std": "",
                "pomo_baseline_mean": "",
                "pomo_within_reward_std_mean": "",
                "pomo_within_reward_std_p10": "",
                "pomo_within_reward_std_p90": "",
                "pomo_valid_traj_ratio": "",
                "pomo_success_traj_ratio": "",
                "pomo_objective_used_ratio": "",
                "pomo_adv_raw_mean": "",
                "pomo_adv_raw_std": "",
                "pomo_adv_norm_mean": "",
                "pomo_adv_norm_std": "",
                "pomo_valid_traj_count_mean": "",
                "boundary_distance_mean": 0.0,
                "internal_distance_mean": 0.0,
                "boundary_share": 0.0,
                "internal_share": 0.0,
                "reward_decomposition_max_abs_error": 0.0,
                "reward_decomposition_mean_abs_error": 0.0,
                "episode_decomposition_max_abs_error": 0.0,
                "episode_decomposition_mean_abs_error": 0.0,
                "advantage_mode": critic_cfg.get("advantage_mode", "total"),
            }
            bc_warmup_epochs = int(offline_cfg.get("bc_warmup_epochs", 0))
            bc_updates_per_epoch = int(offline_cfg.get("bc_updates_per_epoch", offline_cfg.get("offline_updates_per_epoch", 1)))
            route_updates_per_epoch = int(offline_cfg.get("route_updates_per_epoch", offline_cfg.get("offline_updates_per_epoch", 1)))
            offline_coef = _offline_coef(offline_cfg, epoch)
            do_bc_warmup = (
                (offline_method in {"bc_ppo", "bc-ppo"} or _is_dapg_method(offline_method) or _is_frro_method(offline_method))
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
            route_boundary_info: dict[str, Any] = {
                "route_boundary_count_mean": "",
                "route_boundary_step_ratio": "",
            }
            if use_priority_sampler and hasattr(pool, "begin_epoch"):
                pool.begin_epoch(epoch)

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
                if use_priority_sampler and hasattr(pool, "update_from_rollout"):
                    haps_info = pool.update_from_rollout(batch, epoch=epoch)
                decomposed_returns: dict[str, torch.Tensor] | None = None
                if use_pomo_trajectory_advantage:
                    returns, advantages, pomo_info = _compute_pomo_trajectory_advantages(
                        batch,
                        envs,
                        cfg,
                        device,
                    )
                    decomposed_info.update(pomo_info)
                elif use_decomposed_critic:
                    decomposed_returns, decomposed_advantages, advantages, decomposed_info = _compute_decomposed_returns_advantages(
                        batch,
                        gamma=gamma,
                        gae_lambda=gae_lambda,
                        use_gae=use_gae,
                        cfg=cfg,
                        route_segmented=use_route_segmented_gae,
                    )
                    returns = decomposed_returns
                    decomposed_info.update(_decomposed_value_diagnostics(batch, decomposed_returns))
                elif use_gae:
                    returns, advantages = _compute_gae_returns(
                        batch,
                        gamma=gamma,
                        gae_lambda=gae_lambda,
                        route_segmented=use_route_segmented_gae,
                    )
                    advantages = _normalize_valid(advantages, batch.valid)
                else:
                    returns = compute_returns(batch.rewards, batch.dones, gamma=gamma)
                    values = _value_head(batch.values, 0)
                    advantages = returns - values
                    advantages = _normalize_valid(advantages, batch.valid)
                if getattr(batch, "route_boundaries", None) is not None:
                    valid_boundary = batch.route_boundaries & batch.valid
                    valid_count = int(batch.valid.sum().detach().cpu().item())
                    traj_count = max(int(batch.valid.size(1) * batch.valid.size(2)), 1)
                    boundary_count = int(valid_boundary.sum().detach().cpu().item())
                    route_boundary_info = {
                        "route_boundary_count_mean": boundary_count / traj_count,
                        "route_boundary_step_ratio": boundary_count / max(valid_count, 1),
                    }
                dapg_enabled = expert_buffer is not None and _is_dapg_method(offline_method)
                awbc_enabled = expert_buffer is not None and _is_awbc_method(offline_method)
                hard_enabled = expert_buffer is not None and _is_hard_method(offline_method)
                bc_aux_enabled = expert_buffer is not None and _is_bc_aux_method(offline_method)
                bafipo_enabled = expert_buffer is not None and _is_bafipo_method(offline_method)
                gcbpo_enabled = expert_buffer is not None and _is_gcbpo_method(offline_method)
                dapg_demo_gate = 1.0
                sl_enabled = _is_solution_level_method(offline_method)
                route_bc_enabled = _is_route_bc_method(offline_method)
                route_adv_tensor = None
                route_success_tensor = None
                frro_expert_candidates: list[FrroExpertCandidate] = []
                bafipo_pairs: list[BafipoPreferencePair] = []
                bafipo_incumbents: list[BafipoIncumbentCandidate] = []
                gcbpo_pairs: list[GcbpoPreferencePair] = []
                gcbpo_candidates: list[GcbpoBranchCandidate] = []
                if sl_enabled:
                    route_adv_tensor, route_success_tensor, adv_info = _solution_level_advantage_tensors(
                        batch,
                        cfg,
                        envs,
                        expert_buffer,
                        device,
                        policy_best_objectives=policy_best_objectives,
                    )
                    frro_expert_candidates, frro_expert_info = _prepare_frro_expert_candidates(
                        agent,
                        batch,
                        cfg,
                        envs,
                        expert_buffer,
                        policy_best_objectives,
                        device,
                    )
                    adv_info.update(frro_expert_info)
                    # Gate the current batch with historical policy memory only;
                    # the current rollout becomes memory for subsequent epochs.
                    _update_policy_best_objectives(policy_best_objectives, batch, envs)
                else:
                    advantages, adv_info = _apply_auxiliary_advantages(
                        advantages,
                        batch,
                        cfg,
                        envs,
                        expert_buffer,
                        device,
                    )
                    if use_decomposed_critic:
                        actor_mean, actor_std = _stats(advantages, batch.valid)
                        decomposed_info["adv_actor_mean"] = actor_mean
                        decomposed_info["adv_actor_std"] = actor_std
                    if bafipo_enabled:
                        bafipo_pairs, bafipo_incumbents, bafipo_prepare_info = _prepare_bafipo_preference_pairs(
                            agent,
                            batch,
                            cfg,
                            envs,
                            expert_buffer,
                            policy_best_objectives,
                            device,
                        )
                        adv_info.update(bafipo_prepare_info)
                        _update_policy_best_objectives(policy_best_objectives, batch, envs)
                    elif gcbpo_enabled:
                        gcbpo_pairs, gcbpo_candidates, gcbpo_prepare_info = _prepare_gcbpo_preference_pairs(
                            agent,
                            batch,
                            cfg,
                            envs,
                            expert_buffer,
                            device,
                            epoch,
                            seed,
                        )
                        adv_info.update(gcbpo_prepare_info)
                        _update_policy_best_objectives(policy_best_objectives, batch, envs)
                    elif dapg_enabled:
                        dapg_demo_gate, dapg_gate_info = _dapg_demo_gate_from_rollout(
                            batch,
                            cfg,
                            envs,
                            expert_buffer,
                            policy_best_objectives,
                        )
                        adv_info.update(dapg_gate_info)
                        _update_policy_best_objectives(policy_best_objectives, batch, envs)

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
                    dapg_demo_coef *= dapg_adv_scale * float(dapg_demo_gate)
                bc_aux_coef = float(offline_coef)
                dapg_bc_batch_size = int(offline_cfg.get("bc_batch_size", 256))
                dapg_bc_losses: list[float] = []
                dapg_bc_accs: list[float] = []
                dapg_bc_action_accs: list[float] = []
                dapg_bc_entropies: list[float] = []
                route_start_losses: list[float] = []
                internal_successor_losses: list[float] = []
                route_close_losses: list[float] = []
                depot_multilabel_active_ratios: list[float] = []
                start_target_size_means: list[float] = []
                start_target_size_maxes: list[float] = []
                dapg_bc_steps = 0
                awbc_weight_means: list[float] = []
                awbc_weight_stds: list[float] = []
                awbc_active_ratios: list[float] = []
                awbc_expert_better_ratios: list[float] = []
                awbc_coef = float(offline_cfg.get("awbc_coef", offline_cfg.get("bc_coef", 0.05)))
                awbc_eta = float(offline_cfg.get("awbc_eta", 0.05))
                awbc_normalize = str(offline_cfg.get("awbc_normalize", "p95"))
                awbc_baseline = str(offline_cfg.get("awbc_baseline", "batch_mean"))
                awbc_policy_baseline = (
                    _policy_mean_successful_objective(batch)
                    if awbc_baseline.lower() in {"mean_successful", "mean", "avg_successful", "average_successful"}
                    else None
                )
                hard_ref_kl_coef = float(offline_cfg.get("hard_ref_kl_coef", offline_cfg.get("lambda_ref_kl", 0.005 if hard_enabled else 0.0)))
                hard_ref_batch_size = int(offline_cfg.get("hard_ref_batch_size", offline_cfg.get("bc_batch_size", 512)))
                hard_ref_temperature = float(offline_cfg.get("hard_ref_temperature", 1.0))
                hard_ref_kl_enabled = hard_enabled and reference_agent is not None and hard_ref_kl_coef > 0.0
                hard_demo_coef = float(offline_cfg.get("hard_demo_coef", offline_cfg.get("lambda_demo", 0.10 if _is_hard_full_method(offline_method) else 0.0)))
                hard_demo_batch_size = int(offline_cfg.get("hard_demo_batch_size", offline_cfg.get("bc_batch_size", 512)))
                hard_eta = float(offline_cfg.get("hard_eta", offline_cfg.get("hard_eta_total", offline_cfg.get("awbc_eta", 0.10))))
                hard_normalize = str(offline_cfg.get("hard_normalize", offline_cfg.get("hard_component_weight_mode", offline_cfg.get("awbc_normalize", "p95"))))
                hard_baseline = str(offline_cfg.get("hard_baseline", offline_cfg.get("awbc_baseline", "mean_successful")))
                hard_policy_baseline = (
                    _policy_mean_successful_objective(batch)
                    if hard_baseline.lower() in {"mean_successful", "mean", "avg_successful", "average_successful"}
                    else None
                )
                hard_demo_enabled = hard_enabled and _is_hard_full_method(offline_method) and hard_demo_coef > 0.0
                hard_ref_kl_losses: list[float] = []
                hard_ref_matches: list[float] = []
                hard_ref_action_accs: list[float] = []
                hard_ref_entropies: list[float] = []
                hard_demo_losses: list[float] = []
                bafipo_pref_coef = float(_bafipo_config(cfg)["pref_coef"])
                bafipo_minibatches_per_ppo_epoch = int(offline_cfg.get("bafipo_minibatches_per_ppo_epoch", 0) or 0)
                bafipo_pref_losses: list[float] = []
                bafipo_pref_pair_counts: list[float] = []
                bafipo_policy_pair_counts: list[float] = []
                bafipo_incumbent_pair_counts: list[float] = []
                bafipo_pref_weight_means: list[float] = []
                bafipo_pref_logit_means: list[float] = []
                gcbpo_cfg = _gcbpo_config(cfg)
                gcbpo_branch_coef = float(gcbpo_cfg["branch_coef"])
                gcbpo_prefix_coef = float(gcbpo_cfg["prefix_coef"])
                gcbpo_pref_losses: list[float] = []
                gcbpo_prefix_losses: list[float] = []
                gcbpo_pref_pair_counts: list[float] = []
                gcbpo_strong_pair_counts: list[float] = []
                gcbpo_soft_pair_counts: list[float] = []
                gcbpo_pref_weight_means: list[float] = []
                gcbpo_pref_logit_means: list[float] = []
                gcbpo_prefix_route_counts: list[float] = []
                sl_losses: list[float] = []
                sl_ratio_means: list[float] = []
                sl_ratio_stds: list[float] = []
                sl_clip_fracs: list[float] = []
                sl_adv_means: list[float] = []
                sl_adv_stds: list[float] = []
                sl_route_counts: list[float] = []
                notclose_coef = float(offline_cfg.get("notclose_coef", offline_cfg.get("premature_close_coef", 0.0)) or 0.0)
                notclose_losses: list[float] = []
                notclose_points: list[float] = []
                notclose_candidate_points: list[float] = []
                notclose_active_ratios: list[float] = []
                notclose_depot_action_ratios: list[float] = []
                notclose_avg_route_sizes: list[float] = []
                member_coef = float(offline_cfg.get("member_coef", offline_cfg.get("onpolicy_member_coef", 0.0)) or 0.0)
                member_targets = (
                    _prepare_onpolicy_member_targets(batch, envs, expert_buffer, cfg)
                    if member_coef > 0.0 and expert_buffer is not None
                    else None
                )
                member_losses: list[float] = []
                member_loss_points: list[float] = []
                member_candidate_points: list[float] = []
                member_active_ratios: list[float] = []
                anchor_coef = float(offline_cfg.get("anchor_coef", offline_cfg.get("onpolicy_anchor_coef", 0.0)) or 0.0)
                anchor_targets = (
                    _prepare_onpolicy_anchor_targets(batch, envs, expert_buffer, cfg)
                    if anchor_coef > 0.0 and expert_buffer is not None
                    else None
                )
                anchor_losses: list[float] = []
                anchor_loss_points: list[float] = []
                anchor_candidate_points: list[float] = []
                anchor_violation_ratios: list[float] = []
                decomposed_loss_infos: list[dict[str, float]] = []
                approx_kls: list[float] = []
                clip_fracs: list[float] = []
                frro_expert_losses: list[float] = []
                frro_expert_ratio_means: list[float] = []
                frro_expert_ratio_stds: list[float] = []
                frro_expert_clip_fracs: list[float] = []
                frro_expert_adv_means: list[float] = []
                frro_expert_adv_stds: list[float] = []
                frro_expert_route_counts: list[float] = []
                for _ in range(ppo_epochs):
                    bafipo_minibatches_used_this_ppo_epoch = 0
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
                                    if use_decomposed_critic:
                                        if decomposed_returns is None:
                                            raise RuntimeError("decomposed critic enabled but decomposed returns are missing")
                                        loss, policy_loss, value_loss, entropy, value_info = _evaluate_policy_loss_decomposed(
                                            agent,
                                            batch,
                                            decomposed_returns,
                                            advantages.detach(),
                                            cfg,
                                            device,
                                            env_indices=env_indices,
                                            step_start=step_start,
                                            step_end=step_end,
                                        )
                                        decomposed_loss_infos.append(value_info)
                                    elif use_pomo_trajectory_advantage:
                                        loss, policy_loss, value_loss, entropy, ppo_stats = _evaluate_policy_loss_policy_only_with_stats(
                                            agent,
                                            batch,
                                            advantages.detach(),
                                            cfg,
                                            env_indices=env_indices,
                                            step_start=step_start,
                                            step_end=step_end,
                                        )
                                        approx_kls.append(float(ppo_stats["approx_kl"]))
                                        clip_fracs.append(float(ppo_stats["clip_fraction"]))
                                    else:
                                        loss, policy_loss, value_loss, entropy, ppo_stats = _evaluate_policy_loss_with_stats(
                                            agent,
                                            batch,
                                            returns,
                                            advantages.detach(),
                                            cfg,
                                            env_indices=env_indices,
                                            step_start=step_start,
                                            step_end=step_end,
                                        )
                                        approx_kls.append(float(ppo_stats["approx_kl"]))
                                        clip_fracs.append(float(ppo_stats["clip_fraction"]))
                                _backward(loss * chunk_weight / group_size, scaler, amp_enabled)
                                weighted_policy += policy_loss.item() * chunk_weight
                                weighted_value += value_loss.item() * chunk_weight
                                weighted_entropy += entropy.item() * chunk_weight
                                if notclose_coef > 0.0 and expert_buffer is not None:
                                    with _autocast_context(device, amp_enabled):
                                        notclose_loss, notclose_info = _compute_premature_close_loss(
                                            agent,
                                            batch,
                                            envs,
                                            expert_buffer,
                                            cfg,
                                            env_indices,
                                            device,
                                            step_start=step_start,
                                            step_end=step_end,
                                        )
                                    if int(notclose_info.get("notclose_points", 0)) > 0:
                                        _backward(notclose_coef * notclose_loss * chunk_weight / group_size, scaler, amp_enabled)
                                    notclose_losses.append(float(notclose_info.get("notclose_loss", 0.0)))
                                    notclose_points.append(float(notclose_info.get("notclose_points", 0.0)))
                                    notclose_candidate_points.append(float(notclose_info.get("notclose_candidate_points", 0.0)))
                                    notclose_active_ratios.append(float(notclose_info.get("notclose_active_ratio", 0.0)))
                                    notclose_depot_action_ratios.append(float(notclose_info.get("notclose_depot_action_ratio", 0.0)))
                                    notclose_avg_route_sizes.append(float(notclose_info.get("notclose_avg_route_size_mean", 0.0)))
                                if member_coef > 0.0 and member_targets is not None:
                                    with _autocast_context(device, amp_enabled):
                                        member_loss, member_info = _compute_onpolicy_member_loss(
                                            agent,
                                            batch,
                                            member_targets,
                                            cfg,
                                            env_indices,
                                            device,
                                            step_start=step_start,
                                            step_end=step_end,
                                        )
                                    if int(member_info.get("member_loss_points", 0)) > 0:
                                        _backward(member_coef * member_loss * chunk_weight / group_size, scaler, amp_enabled)
                                    member_losses.append(float(member_info.get("member_loss", 0.0)))
                                    member_loss_points.append(float(member_info.get("member_loss_points", 0.0)))
                                    member_candidate_points.append(float(member_info.get("member_candidate_points", 0.0)))
                                    member_active_ratios.append(float(member_info.get("member_loss_active_ratio", 0.0)))
                                if anchor_coef > 0.0 and anchor_targets is not None:
                                    with _autocast_context(device, amp_enabled):
                                        anchor_loss, anchor_info = _compute_onpolicy_anchor_loss(
                                            agent,
                                            batch,
                                            anchor_targets,
                                            cfg,
                                            env_indices,
                                            device,
                                            step_start=step_start,
                                            step_end=step_end,
                                        )
                                    if int(anchor_info.get("anchor_loss_points", 0)) > 0:
                                        _backward(anchor_coef * anchor_loss * chunk_weight / group_size, scaler, amp_enabled)
                                    anchor_losses.append(float(anchor_info.get("anchor_loss", 0.0)))
                                    anchor_loss_points.append(float(anchor_info.get("anchor_loss_points", 0.0)))
                                    anchor_candidate_points.append(float(anchor_info.get("anchor_candidate_points", 0.0)))
                                    anchor_violation_ratios.append(float(anchor_info.get("anchor_violation_ratio", 0.0)))
                            sl_coef = float(offline_cfg.get("sl_coef", offline_cfg.get("route_loss_coef", 0.10)))
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
                                _backward(sl_coef * route_loss / group_size, scaler, amp_enabled)
                                sl_losses.append(float(route_info["sl_route_loss"]))
                                sl_ratio_means.append(float(route_info["sl_route_ratio_mean"]))
                                sl_ratio_stds.append(float(route_info["sl_route_ratio_std"]))
                                sl_clip_fracs.append(float(route_info["sl_route_clip_frac"]))
                                sl_adv_means.append(float(route_info["sl_route_adv_mean"]))
                                sl_adv_stds.append(float(route_info["sl_route_adv_std"]))
                                sl_route_counts.append(float(route_info["sl_num_routes_used"]))
                            if frro_expert_candidates:
                                with _autocast_context(device, amp_enabled):
                                    expert_loss, expert_info = _compute_frro_expert_candidate_loss(
                                        agent,
                                        frro_expert_candidates,
                                        cfg,
                                        env_indices,
                                        device,
                                    )
                                _backward(sl_coef * expert_loss / group_size, scaler, amp_enabled)
                                frro_expert_losses.append(float(expert_info["frro_expert_loss"]))
                                frro_expert_ratio_means.append(float(expert_info["frro_expert_ratio_mean"]))
                                frro_expert_ratio_stds.append(float(expert_info["frro_expert_ratio_std"]))
                                frro_expert_clip_fracs.append(float(expert_info["frro_expert_clip_frac"]))
                                frro_expert_adv_means.append(float(expert_info["frro_expert_adv_mean"]))
                                frro_expert_adv_stds.append(float(expert_info["frro_expert_adv_std"]))
                                frro_expert_route_counts.append(float(expert_info["frro_expert_num_routes"]))
                            if (
                                bafipo_enabled
                                and bafipo_pairs
                                and bafipo_pref_coef > 0.0
                                and (
                                    bafipo_minibatches_per_ppo_epoch <= 0
                                    or bafipo_minibatches_used_this_ppo_epoch < bafipo_minibatches_per_ppo_epoch
                                )
                            ):
                                bafipo_minibatches_used_this_ppo_epoch += 1
                                with _autocast_context(device, amp_enabled):
                                    pref_loss, pref_info = _compute_bafipo_preference_loss(
                                        agent,
                                        batch,
                                        bafipo_pairs,
                                        bafipo_incumbents,
                                        cfg,
                                        env_indices,
                                        device,
                                    )
                                _backward(bafipo_pref_coef * pref_loss / group_size, scaler, amp_enabled)
                                if float(pref_info["bafipo_pref_pair_count"]) > 0.0:
                                    bafipo_pref_losses.append(float(pref_info["bafipo_pref_loss"]))
                                    bafipo_pref_pair_counts.append(float(pref_info["bafipo_pref_pair_count"]))
                                    bafipo_policy_pair_counts.append(float(pref_info["bafipo_policy_pair_count"]))
                                    bafipo_incumbent_pair_counts.append(float(pref_info["bafipo_incumbent_pair_count"]))
                                    bafipo_pref_weight_means.append(float(pref_info["bafipo_pref_weight_mean"]))
                                    bafipo_pref_logit_means.append(float(pref_info["bafipo_pref_logit_mean"]))
                            if (
                                gcbpo_enabled
                                and (gcbpo_pairs or gcbpo_candidates)
                                and (gcbpo_branch_coef > 0.0 or gcbpo_prefix_coef > 0.0)
                            ):
                                with _autocast_context(device, amp_enabled):
                                    gcbpo_pref_loss, gcbpo_prefix_loss, gcbpo_loss_info = _compute_gcbpo_preference_loss(
                                        agent,
                                        batch,
                                        gcbpo_pairs,
                                        gcbpo_candidates,
                                        cfg,
                                        env_indices,
                                        device,
                                    )
                                _backward(
                                    (gcbpo_branch_coef * gcbpo_pref_loss + gcbpo_prefix_coef * gcbpo_prefix_loss) / group_size,
                                    scaler,
                                    amp_enabled,
                                )
                                if float(gcbpo_loss_info["gcbpo_pref_pair_count"]) > 0.0:
                                    gcbpo_pref_losses.append(float(gcbpo_loss_info["gcbpo_pref_loss"]))
                                    gcbpo_prefix_losses.append(float(gcbpo_loss_info["gcbpo_prefix_loss"]))
                                    gcbpo_pref_pair_counts.append(float(gcbpo_loss_info["gcbpo_pref_pair_count"]))
                                    gcbpo_strong_pair_counts.append(float(gcbpo_loss_info["gcbpo_strong_pair_count"]))
                                    gcbpo_soft_pair_counts.append(float(gcbpo_loss_info["gcbpo_soft_pair_count"]))
                                    gcbpo_pref_weight_means.append(float(gcbpo_loss_info["gcbpo_pref_weight_mean"]))
                                    gcbpo_pref_logit_means.append(float(gcbpo_loss_info["gcbpo_pref_logit_mean"]))
                                    gcbpo_prefix_route_counts.append(float(gcbpo_loss_info["gcbpo_prefix_route_count"]))
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
                            dapg_bc_action_accs.append(float(demo_info.get("bc_action_accuracy", demo_info["bc_accuracy"])))
                            dapg_bc_entropies.append(float(demo_info["bc_entropy"]))
                            route_start_losses.append(float(demo_info.get("route_start_loss", 0.0)))
                            internal_successor_losses.append(float(demo_info.get("internal_successor_loss", 0.0)))
                            route_close_losses.append(float(demo_info.get("route_close_loss", 0.0)))
                            depot_multilabel_active_ratios.append(float(demo_info.get("depot_multilabel_active_ratio", 0.0)))
                            start_target_size_means.append(float(demo_info.get("start_target_set_size_mean", 0.0)))
                            start_target_size_maxes.append(float(demo_info.get("start_target_set_size_max", 0.0)))
                            dapg_bc_steps += int(demo_info["bc_steps"])
                            offline_updates += 1
                        if awbc_enabled and awbc_coef > 0.0 and expert_buffer is not None:
                            with _autocast_context(device, amp_enabled):
                                demo_loss, demo_info = compute_awbc_loss(
                                    agent,
                                    expert_buffer,
                                    batch_size=dapg_bc_batch_size,
                                    device=device,
                                    eta=awbc_eta,
                                    normalize=awbc_normalize,
                                    baseline=awbc_baseline,
                                    baseline_objective=awbc_policy_baseline,
                                )
                            _backward(float(awbc_coef) * demo_loss, scaler, amp_enabled)
                            dapg_bc_losses.append(float(demo_info["bc_loss"]))
                            dapg_bc_accs.append(float(demo_info["bc_accuracy"]))
                            dapg_bc_action_accs.append(float(demo_info.get("bc_action_accuracy", demo_info["bc_accuracy"])))
                            dapg_bc_entropies.append(float(demo_info["bc_entropy"]))
                            route_start_losses.append(float(demo_info.get("route_start_loss", 0.0)))
                            internal_successor_losses.append(float(demo_info.get("internal_successor_loss", 0.0)))
                            route_close_losses.append(float(demo_info.get("route_close_loss", 0.0)))
                            depot_multilabel_active_ratios.append(float(demo_info.get("depot_multilabel_active_ratio", 0.0)))
                            start_target_size_means.append(float(demo_info.get("start_target_set_size_mean", 0.0)))
                            start_target_size_maxes.append(float(demo_info.get("start_target_set_size_max", 0.0)))
                            dapg_bc_steps += int(demo_info["bc_steps"])
                            awbc_weight_means.append(float(demo_info["awbc_weight_mean"]))
                            awbc_weight_stds.append(float(demo_info["awbc_weight_std"]))
                            awbc_active_ratios.append(float(demo_info["awbc_active_ratio"]))
                            awbc_expert_better_ratios.append(float(demo_info["awbc_expert_better_ratio"]))
                            offline_updates += 1
                        _optimizer_step(optimizer, agent, max_grad_norm, scaler, amp_enabled)
                        losses.append((group_policy, group_value, group_entropy))
                if decomposed_loss_infos:
                    for key in (
                        "value_loss_total",
                        "value_loss_boundary",
                        "value_loss_internal",
                        "value_consistency_loss",
                    ):
                        decomposed_info[key] = float(np.mean([info[key] for info in decomposed_loss_infos]))
                ppo_stats_info = {
                    "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
                    "clip_fraction": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
                }
                if dapg_enabled or awbc_enabled:
                    bc_info = {
                        "bc_loss": float(np.mean(dapg_bc_losses)) if dapg_bc_losses else 0.0,
                        "bc_accuracy": float(np.mean(dapg_bc_accs)) if dapg_bc_accs else 0.0,
                        "bc_action_accuracy": float(np.mean(dapg_bc_action_accs)) if dapg_bc_action_accs else 0.0,
                        "bc_entropy": float(np.mean(dapg_bc_entropies)) if dapg_bc_entropies else 0.0,
                        "bc_steps": int(dapg_bc_steps),
                        "bc_coef": float(awbc_coef if awbc_enabled else dapg_demo_coef),
                        "route_start_loss": float(np.mean(route_start_losses)) if route_start_losses else 0.0,
                        "internal_successor_loss": float(np.mean(internal_successor_losses)) if internal_successor_losses else 0.0,
                        "route_close_loss": float(np.mean(route_close_losses)) if route_close_losses else 0.0,
                        "depot_multilabel_active_ratio": float(np.mean(depot_multilabel_active_ratios)) if depot_multilabel_active_ratios else 0.0,
                        "start_target_set_size_mean": float(np.mean(start_target_size_means)) if start_target_size_means else 0.0,
                        "start_target_set_size_max": float(np.max(start_target_size_maxes)) if start_target_size_maxes else 0.0,
                        "offline_updates": int(offline_updates),
                        "awbc_loss": float(np.mean(dapg_bc_losses)) if (awbc_enabled and dapg_bc_losses) else 0.0,
                        "awbc_weight_mean": float(np.mean(awbc_weight_means)) if awbc_weight_means else 0.0,
                        "awbc_weight_std": float(np.mean(awbc_weight_stds)) if awbc_weight_stds else 0.0,
                        "awbc_active_ratio": float(np.mean(awbc_active_ratios)) if awbc_active_ratios else 0.0,
                        "awbc_expert_better_ratio": float(np.mean(awbc_expert_better_ratios)) if awbc_expert_better_ratios else 0.0,
                        "hard_ref_kl_loss": float(np.mean(hard_ref_kl_losses)) if hard_ref_kl_losses else 0.0,
                        "hard_ref_match": float(np.mean(hard_ref_matches)) if hard_ref_matches else 0.0,
                        "hard_ref_bc_action_accuracy": float(np.mean(hard_ref_action_accs)) if hard_ref_action_accs else 0.0,
                        "hard_ref_entropy": float(np.mean(hard_ref_entropies)) if hard_ref_entropies else 0.0,
                        "hard_ref_kl_coef": float(hard_ref_kl_coef),
                        "hard_ref_temperature": float(hard_ref_temperature),
                        "hard_demo_loss": float(np.mean(hard_demo_losses)) if hard_demo_losses else 0.0,
                        "hard_demo_coef": float(hard_demo_coef if hard_demo_enabled else 0.0),
                        "hard_eta": float(hard_eta),
                        "hard_normalize": hard_normalize,
                        "hard_baseline": hard_baseline,
                    }
                if notclose_coef > 0.0:
                    bc_info.update(
                        {
                            "notclose_loss": float(np.mean(notclose_losses)) if notclose_losses else 0.0,
                            "notclose_points": int(np.sum(notclose_points)) if notclose_points else 0,
                            "notclose_candidate_points": int(np.sum(notclose_candidate_points)) if notclose_candidate_points else 0,
                            "notclose_active_ratio": float(np.mean(notclose_active_ratios)) if notclose_active_ratios else 0.0,
                            "notclose_depot_action_ratio": float(np.mean(notclose_depot_action_ratios)) if notclose_depot_action_ratios else 0.0,
                            "notclose_avg_route_size_mean": float(np.mean(notclose_avg_route_sizes)) if notclose_avg_route_sizes else 0.0,
                            "notclose_coef": float(notclose_coef),
                        }
                    )
                if member_coef > 0.0:
                    member_info_for_epoch = _zero_member_info(member_coef, member_targets)
                    member_points_sum = int(np.sum(member_loss_points)) if member_loss_points else 0
                    member_candidates_sum = int(np.sum(member_candidate_points)) if member_candidate_points else 0
                    member_info_for_epoch.update(
                        {
                            "member_loss": float(np.mean(member_losses)) if member_losses else 0.0,
                            "member_loss_points": member_points_sum,
                            "member_candidate_points": member_candidates_sum,
                            "member_loss_active_ratio": float(member_points_sum / max(member_candidates_sum, 1)),
                        }
                    )
                    bc_info.update(member_info_for_epoch)
                if anchor_coef > 0.0:
                    anchor_info_for_epoch = _zero_anchor_info(anchor_coef, anchor_targets)
                    anchor_points_sum = int(np.sum(anchor_loss_points)) if anchor_loss_points else 0
                    anchor_candidates_sum = int(np.sum(anchor_candidate_points)) if anchor_candidate_points else 0
                    anchor_info_for_epoch.update(
                        {
                            "anchor_loss": float(np.mean(anchor_losses)) if anchor_losses else 0.0,
                            "anchor_loss_points": anchor_points_sum,
                            "anchor_candidate_points": anchor_candidates_sum,
                            "anchor_violation_ratio": float(anchor_points_sum / max(anchor_candidates_sum, 1)),
                        }
                    )
                    bc_info.update(anchor_info_for_epoch)
                if bafipo_enabled:
                    bafipo_info = {
                        "bafipo_pref_loss": float(np.mean(bafipo_pref_losses)) if bafipo_pref_losses else 0.0,
                        "bafipo_pref_pair_count": int(np.sum(bafipo_pref_pair_counts)) if bafipo_pref_pair_counts else 0,
                        "bafipo_policy_pair_count": int(np.sum(bafipo_policy_pair_counts)) if bafipo_policy_pair_counts else 0,
                        "bafipo_incumbent_pair_count": int(np.sum(bafipo_incumbent_pair_counts)) if bafipo_incumbent_pair_counts else 0,
                        "bafipo_quality_gate_mean": adv_info.get("bafipo_quality_gate_mean", 0.0),
                        "bafipo_memory_gate_mean": adv_info.get("bafipo_memory_gate_mean", 0.0),
                        "bafipo_spread_gate_mean": adv_info.get("bafipo_spread_gate_mean", 0.0),
                        "bafipo_incumbent_beats_best_rate": adv_info.get("bafipo_incumbent_beats_best_rate", 0.0),
                        "bafipo_incumbent_beats_mean_rate": adv_info.get("bafipo_incumbent_beats_mean_rate", 0.0),
                        "bafipo_pref_weight_mean": float(np.mean(bafipo_pref_weight_means)) if bafipo_pref_weight_means else adv_info.get("bafipo_pair_weight_mean", 0.0),
                        "bafipo_pref_logit_mean": float(np.mean(bafipo_pref_logit_means)) if bafipo_pref_logit_means else 0.0,
                        "bafipo_pref_coef": float(bafipo_pref_coef),
                        "bafipo_minibatches_per_ppo_epoch": int(bafipo_minibatches_per_ppo_epoch),
                    }
                if gcbpo_enabled:
                    gcbpo_info = {
                        "gcbpo_pref_loss": float(np.mean(gcbpo_pref_losses)) if gcbpo_pref_losses else 0.0,
                        "gcbpo_prefix_loss": float(np.mean(gcbpo_prefix_losses)) if gcbpo_prefix_losses else 0.0,
                        "gcbpo_pref_pair_count": int(np.sum(gcbpo_pref_pair_counts)) if gcbpo_pref_pair_counts else 0,
                        "gcbpo_strong_pair_count": int(np.sum(gcbpo_strong_pair_counts)) if gcbpo_strong_pair_counts else 0,
                        "gcbpo_soft_pair_count": int(np.sum(gcbpo_soft_pair_counts)) if gcbpo_soft_pair_counts else 0,
                        "gcbpo_branch_candidates": int(adv_info.get("gcbpo_branch_candidates", 0)),
                        "gcbpo_branch_beats_best_rate": adv_info.get("gcbpo_branch_beats_best_rate", 0.0),
                        "gcbpo_branch_beats_top_mean_rate": adv_info.get("gcbpo_branch_beats_top_mean_rate", 0.0),
                        "gcbpo_branch_gap_close_mean": adv_info.get("gcbpo_branch_gap_close_mean", 0.0),
                        "gcbpo_prefix_valid_rate": adv_info.get("gcbpo_prefix_valid_rate", 0.0),
                        "gcbpo_prefix_len_mean": adv_info.get("gcbpo_prefix_len_mean", 0.0),
                        "gcbpo_pref_weight_mean": float(np.mean(gcbpo_pref_weight_means)) if gcbpo_pref_weight_means else adv_info.get("gcbpo_pair_weight_mean", 0.0),
                        "gcbpo_pref_logit_mean": float(np.mean(gcbpo_pref_logit_means)) if gcbpo_pref_logit_means else 0.0,
                        "gcbpo_branch_coef": float(gcbpo_branch_coef),
                        "gcbpo_prefix_coef": float(gcbpo_prefix_coef),
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
                        "sl_ref_memory_gate_mean": adv_info.get("ref_memory_gate_mean", 0.0),
                        "sl_ref_memory_gate_std": adv_info.get("ref_memory_gate_std", 0.0),
                        "sl_ref_memory_better_rate": adv_info.get("ref_memory_better_rate", 0.0),
                        "sl_ref_memory_gap_mean": adv_info.get("ref_memory_gap_mean", 0.0),
                        "sl_ref_base_gap_ratio_mean": adv_info.get("ref_base_gap_ratio_mean", 0.0),
                        "frro_adv_mean": adv_info.get("frro_adv_mean", 0.0),
                        "frro_adv_std": adv_info.get("frro_adv_std", 0.0),
                        "frro_positive_mean": adv_info.get("frro_positive_mean", 0.0),
                        "frro_positive_std": adv_info.get("frro_positive_std", 0.0),
                        "frro_negative_mean": adv_info.get("frro_negative_mean", 0.0),
                        "frro_negative_std": adv_info.get("frro_negative_std", 0.0),
                        "frro_gate_mean": adv_info.get("frro_gate_mean", 0.0),
                        "frro_gate_std": adv_info.get("frro_gate_std", 0.0),
                        "frro_falsified_rate": adv_info.get("frro_falsified_rate", 0.0),
                        "frro_best_gap_mean": adv_info.get("frro_best_gap_mean", 0.0),
                        "frro_expert_loss": float(np.mean(frro_expert_losses)) if frro_expert_losses else 0.0,
                        "frro_expert_ratio_mean": float(np.mean(frro_expert_ratio_means)) if frro_expert_ratio_means else 1.0,
                        "frro_expert_ratio_std": float(np.mean(frro_expert_ratio_stds)) if frro_expert_ratio_stds else 0.0,
                        "frro_expert_clip_frac": float(np.mean(frro_expert_clip_fracs)) if frro_expert_clip_fracs else 0.0,
                        "frro_expert_adv_mean": float(np.mean(frro_expert_adv_means)) if frro_expert_adv_means else adv_info.get("frro_expert_adv_mean", 0.0),
                        "frro_expert_adv_std": float(np.mean(frro_expert_adv_stds)) if frro_expert_adv_stds else adv_info.get("frro_expert_adv_std", 0.0),
                        "frro_expert_gate_mean": adv_info.get("frro_expert_gate_mean", 0.0),
                        "frro_expert_gate_std": adv_info.get("frro_expert_gate_std", 0.0),
                        "frro_expert_num_routes": int(np.sum(frro_expert_route_counts)) if frro_expert_route_counts else int(adv_info.get("frro_expert_num_routes", 0)),
                        "frro_expert_weight": adv_info.get("frro_expert_weight", 0.0),
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
                    f"value_heads={_format_float(decomposed_info.get('value_loss_total', 0.0))}/"
                    f"{_format_float(decomposed_info.get('value_loss_boundary', 0.0))}/"
                    f"{_format_float(decomposed_info.get('value_loss_internal', 0.0))} "
                    f"entropy={_format_float(loss_arr[:, 2].mean())} "
                    f"train_fr={_format_float(train_summary['train_feasible_rate'])} "
                    f"train_obj={_format_float(train_summary['train_avg_best_objective_distance_km'])} "
                    f"timing_reset={rollout_timings.get('rollout_reset_time_s', 0.0):.3f}s "
                    f"timing_model={rollout_timings.get('rollout_model_action_time_s', 0.0):.3f}s "
                    f"timing_env={rollout_timings.get('rollout_env_step_time_s', 0.0):.3f}s "
                    f"timing_ppo={ppo_update_time_s:.3f}s "
                    f"dapg_gate={_format_float(adv_info.get('dapg_demo_gate_mean', 1.0))} "
                    f"dapg_mem_better={_format_float(adv_info.get('dapg_memory_better_rate', 0.0))} "
                    f"bc={_format_float(bc_info.get('bc_loss', 0.0))} "
                    f"bc_acc={_format_float(bc_info.get('bc_accuracy', 0.0))} "
                    f"awbc={_format_float(bc_info.get('awbc_loss', 0.0))} "
                    f"awbc_w={_format_float(bc_info.get('awbc_weight_mean', 0.0))} "
                    f"bc_coef={_format_float(bc_info.get('bc_coef', 0.0))}",
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
                eval_message = (
                    "[Eval] "
                    f"epoch={epoch}/{epochs} n={eval_row.get('eval_num_instances')} "
                    f"fr={_format_float(eval_row.get('eval_feasible_rate'))} "
                    f"min_obj={_format_float(eval_row.get('eval_avg_min_objective_distance_km', eval_row.get('eval_avg_objective_distance_km')))} "
                    f"min_veh={_format_float(eval_row.get('eval_avg_min_vehicle_count', eval_row.get('eval_avg_vehicle_count')))} "
                )
                if record_eval_median:
                    eval_message += (
                        f"med_obj={_format_float(eval_row.get('eval_avg_median_objective_distance_km'))} "
                        f"med_veh={_format_float(eval_row.get('eval_avg_median_vehicle_count'))} "
                    )
                eval_message += f"eval_wall={eval_wall_time_s:.3f}s status={eval_row.get('eval_status')}"
                _debug_log(debug_enabled, df, eval_message)
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
            expert_stats = getattr(expert_buffer, "replay_stats", {}) if expert_buffer is not None else {}
            writer.writerow(
                {
                    "epoch": epoch,
                    "samples_seen": pool.sample_count,
                    "train_mode": offline_method,
                    "reward_mean": reward_mean,
                    "policy_loss": float(loss_arr[:, 0].mean()),
                    "value_loss": float(loss_arr[:, 1].mean()),
                    "entropy": float(loss_arr[:, 2].mean()),
                    "approx_kl": ppo_stats_info.get("approx_kl", ""),
                    "clip_fraction": ppo_stats_info.get("clip_fraction", ""),
                    "train_feasible_rate": train_summary.get("train_feasible_rate", ""),
                    "train_avg_best_objective_distance_km": train_summary.get("train_avg_best_objective_distance_km", ""),
                    "train_avg_vehicle_count": train_summary.get("train_avg_vehicle_count", ""),
                    "train_avg_served_customers": train_summary.get("train_avg_served_customers", ""),
                    "bc_loss": bc_info.get("bc_loss", ""),
                    "bc_accuracy": bc_info.get("bc_accuracy", ""),
                    "bc_action_accuracy": bc_info.get("bc_action_accuracy", ""),
                    "bc_entropy": bc_info.get("bc_entropy", ""),
                    "bc_coef": bc_info.get("bc_coef", ""),
                    "awbc_loss": bc_info.get("awbc_loss", ""),
                    "awbc_weight_mean": bc_info.get("awbc_weight_mean", ""),
                    "awbc_weight_std": bc_info.get("awbc_weight_std", ""),
                    "awbc_active_ratio": bc_info.get("awbc_active_ratio", ""),
                    "awbc_expert_better_ratio": bc_info.get("awbc_expert_better_ratio", ""),
                    "dapg_demo_gate_mean": adv_info.get("dapg_demo_gate_mean", ""),
                    "dapg_demo_gate_std": adv_info.get("dapg_demo_gate_std", ""),
                    "dapg_demo_active_ratio": adv_info.get("dapg_demo_active_ratio", ""),
                    "dapg_memory_better_rate": adv_info.get("dapg_memory_better_rate", ""),
                    "dapg_memory_gap_mean": adv_info.get("dapg_memory_gap_mean", ""),
                    "offline_updates": offline_updates,
                    "num_envs": num_envs,
                    "n_traj": int(train_cfg.get("n_traj", 50)),
                    "rollout_steps": rollout_steps,
                    "num_minibatches": minibatches,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "mixed_precision": amp_enabled,
                    "use_gae": use_gae,
                    "gae_lambda": gae_lambda,
                    "best_eval_avg_objective_distance_km": best_eval_objective if np.isfinite(best_eval_objective) else "",
                    "best_eval_epoch": best_eval_epoch,
                    "rollout_model_action_time_s": rollout_timings.get("rollout_model_action_time_s", ""),
                    "rollout_env_step_time_s": rollout_timings.get("rollout_env_step_time_s", ""),
                    "rollout_total_time_s": rollout_timings.get("rollout_total_time_s", ""),
                    "ppo_update_time_s": ppo_update_time_s,
                    "eval_wall_time_s": eval_wall_time_s,
                    "epoch_wall_time_s": epoch_wall_time_s,
                    "eval_avg_objective_distance_km": eval_row.get("eval_avg_objective_distance_km", ""),
                    "eval_avg_min_objective_distance_km": eval_row.get("eval_avg_min_objective_distance_km", ""),
                    "eval_avg_vehicle_count": eval_row.get("eval_avg_vehicle_count", ""),
                    "eval_avg_min_vehicle_count": eval_row.get("eval_avg_min_vehicle_count", ""),
                    "eval_feasible_rate": eval_row.get("eval_feasible_rate", ""),
                    "eval_gap_mean": eval_row.get("eval_gap_mean", ""),
                    "eval_gap_median": eval_row.get("eval_gap_median", ""),
                    "eval_gap_p90": eval_row.get("eval_gap_p90", ""),
                    "eval_vehicle_gap_mean": eval_row.get("eval_vehicle_gap_mean", ""),
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
