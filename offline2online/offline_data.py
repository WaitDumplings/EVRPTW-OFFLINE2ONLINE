from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .integrations.evrptw_db import configure_evrptw_db

EVRPTW_DB_ROOT = configure_evrptw_db()

from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.env_factory import make_terran_env
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.rollout import stack_observations
from evrptw_core.io import iter_instances


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ExpertRecord:
    instance_id: str
    instance: Any
    routes: list[list[int]]
    objective_distance_km: float
    vehicle_count: int


@dataclass
class ExpertTrajectory:
    instance_id: str
    observations: list[dict[str, np.ndarray]]
    actions: list[int]
    objective_distance_km: float
    vehicle_count: int

    @property
    def length(self) -> int:
        return len(self.actions)


def resolve_repo_path(path: str | Path) -> Path:
    out = Path(path)
    if out.is_absolute():
        return out
    if str(path).startswith("EVRPTW_DB_ROOT/"):
        return EVRPTW_DB_ROOT / str(path).split("/", 1)[1]
    return REPO_ROOT / out


def _route_payload(row: dict[str, str]) -> str:
    for key in ("routes_json", "routes", "route_json", "solution_routes"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _parse_routes(payload: str) -> list[list[int]]:
    data = json.loads(payload)
    if isinstance(data, dict):
        for key in ("routes", "solution", "vehicles"):
            if key in data:
                data = data[key]
                break
    routes: list[list[int]] = []
    for route in data:
        if isinstance(route, dict):
            route = route.get("route") or route.get("nodes") or route.get("path") or []
        nodes = [int(node) for node in route]
        if nodes:
            routes.append(nodes)
    return routes


def _clean_routes(routes: Sequence[Sequence[int]]) -> list[list[int]]:
    clean: list[list[int]] = []
    for route in routes:
        nodes = [int(node) for node in route]
        if not nodes or not any(node != 0 for node in nodes):
            continue
        if nodes[0] != 0:
            nodes.insert(0, 0)
        if nodes[-1] != 0:
            nodes.append(0)
        clean.append(nodes)
    return clean


def route_actions(routes: Sequence[Sequence[int]]) -> list[int]:
    actions: list[int] = []
    for route in _clean_routes(routes):
        actions.extend(int(node) for node in route[1:])
    return actions


def _is_usable_solution(row: dict[str, str]) -> bool:
    status = str(row.get("status", "")).strip().upper()
    status_name = str(row.get("status_name", "")).strip().upper()
    allowed = {"2", "9", "OPTIMAL", "TIME_LIMIT", "RUNNING", "SUBOPTIMAL", "FEASIBLE", "SUCCESS", "OK"}
    status_ok = not status or status in allowed or status_name in allowed
    if not status_ok:
        return False
    obj = row.get("objective_distance_km", "")
    routes = _route_payload(row)
    return obj not in {"", "nan", "NaN", "None"} and routes not in {"", "nan", "NaN", "None"}


def load_solver_expert_records(
    dataset_path: str | Path,
    solution_csv_path: str | Path,
    num_customers: int,
    num_charging_stations: int,
    limit: int | None = None,
) -> list[ExpertRecord]:
    dataset_root = resolve_repo_path(dataset_path)
    solution_path = Path(solution_csv_path)
    if not solution_path.is_absolute():
        solution_path = REPO_ROOT / solution_path
    instances = {
        instance.instance_id: instance
        for instance in iter_instances(
            dataset_root,
            num_customers=int(num_customers),
            num_charging_stations=int(num_charging_stations),
        )
    }
    best_by_instance: dict[str, ExpertRecord] = {}
    with solution_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not _is_usable_solution(row):
                continue
            instance_id = row.get("instance_id", "")
            instance = instances.get(instance_id)
            if instance is None:
                continue
            try:
                objective = float(row["objective_distance_km"])
                routes = _parse_routes(_route_payload(row))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not np.isfinite(objective) or not routes:
                continue
            previous = best_by_instance.get(instance_id)
            if previous is not None and previous.objective_distance_km <= objective:
                continue
            best_by_instance[instance_id] = ExpertRecord(
                instance_id=instance_id,
                instance=instance,
                routes=routes,
                objective_distance_km=objective,
                vehicle_count=int(float(row.get("vehicle_count", 0) or 0)),
            )
    records = list(best_by_instance.values())
    if limit is not None:
        records = records[: int(limit)]
    if not records:
        raise ValueError(f"no usable expert records loaded from {solution_path}")
    return records


load_gurobi_expert_records = load_solver_expert_records


def build_expert_trajectories(
    records: Sequence[ExpertRecord],
    cfg: dict[str, Any],
    *,
    max_records: int | None = None,
    strict: bool = True,
    seed: int = 0,
) -> tuple[list[ExpertTrajectory], dict[str, Any]]:
    del seed
    env_cfg = dict(cfg.get("env", {}) or {})
    env_cfg["use_fast_env"] = True
    env_cfg.setdefault("info_level", "light")
    scale_mode = str(env_cfg.get("reward_distance_scale_mode", ""))
    if scale_mode.startswith("dataset_"):
        env_cfg["reward_distance_scale_mode"] = scale_mode[len("dataset_") :]

    selected = list(records[: max_records or len(records)])
    trajectories: list[ExpertTrajectory] = []
    invalid_records: list[dict[str, Any]] = []
    objective_errors: list[float] = []
    route_counts: list[int] = []

    for record in selected:
        clean_routes = _clean_routes(record.routes)
        route_counts.append(len(clean_routes))
        env = make_terran_env(instance=record.instance, n_traj=1, pbrs_config=None, **env_cfg)
        obs, info = env.reset()
        observations: list[dict[str, np.ndarray]] = []
        actions: list[int] = []
        invalid_step: dict[str, Any] | None = None
        for step_idx, action in enumerate(route_actions(clean_routes)):
            action_i = int(action)
            mask = np.asarray(obs["action_mask"], dtype=bool)
            if mask.shape[0] != 1 or action_i < 0 or action_i >= mask.shape[1] or not bool(mask[0, action_i]):
                invalid_step = {
                    "instance_id": record.instance_id,
                    "step": step_idx,
                    "action": action_i,
                    "mask_shape": tuple(mask.shape),
                }
                break
            observations.append({key: np.asarray(value).copy() for key, value in obs.items()})
            actions.append(action_i)
            obs, reward, terminated, truncated, info = env.step(np.asarray([action_i], dtype=np.int64))
            if bool(np.asarray(truncated, dtype=bool)[0]):
                invalid_step = {
                    "instance_id": record.instance_id,
                    "step": step_idx,
                    "action": action_i,
                    "reason": "truncated_after_expert_action",
                }
                break
        success = bool(np.asarray(info.get("success", [False]), dtype=bool)[0])
        if invalid_step is not None or not success:
            invalid_records.append(
                invalid_step
                or {
                    "instance_id": record.instance_id,
                    "reason": "expert_route_replay_not_successful",
                    "served": int(np.asarray(info.get("served_customers", [0]))[0]),
                }
            )
            continue
        if actions:
            objective = np.asarray(info.get("objective_distance_km", []), dtype=np.float64).reshape(-1)
            if objective.size > 0 and np.isfinite(objective[0]):
                objective_errors.append(abs(float(objective[0]) - float(record.objective_distance_km)))
            trajectories.append(
                ExpertTrajectory(
                    instance_id=record.instance_id,
                    observations=observations,
                    actions=actions,
                    objective_distance_km=record.objective_distance_km,
                    vehicle_count=record.vehicle_count,
                )
            )
    if strict and invalid_records:
        raise ValueError(f"{len(invalid_records)} expert routes failed replay; first={invalid_records[0]}")
    stats = {
        "records_seen": len(selected),
        "trajectories": len(trajectories),
        "invalid_records": len(invalid_records),
        "steps": int(sum(traj.length for traj in trajectories)),
        "avg_steps_per_route": float(np.mean([traj.length for traj in trajectories])) if trajectories else 0.0,
        "expert_replay_success_rate": float(len(trajectories) / max(len(selected), 1)),
        "expert_action_valid_ratio": float(1.0 - len(invalid_records) / max(len(selected), 1)),
        "expert_env_replay_obj_error_mean": float(np.mean(objective_errors)) if objective_errors else float("nan"),
        "expert_env_replay_obj_error_max": float(np.max(objective_errors)) if objective_errors else float("nan"),
        "expert_route_count_mean": float(np.mean(route_counts)) if route_counts else 0.0,
    }
    return trajectories, stats


class ExpertReplayBuffer:
    def __init__(
        self,
        trajectories: Sequence[ExpertTrajectory],
        seed: int = 0,
        replay_stats: dict[str, Any] | None = None,
    ) -> None:
        self.trajectories = list(trajectories)
        if not self.trajectories:
            raise ValueError("ExpertReplayBuffer requires at least one trajectory")
        self.rng = np.random.default_rng(int(seed))
        self.replay_stats = dict(replay_stats or {})
        self.objective_by_instance_id = {
            str(traj.instance_id): float(traj.objective_distance_km)
            for traj in self.trajectories
        }
        self.trajectory_by_instance_id = {
            str(traj.instance_id): traj
            for traj in self.trajectories
        }
        self._step_index: list[tuple[int, int]] = [
            (traj_idx, step_idx)
            for traj_idx, traj in enumerate(self.trajectories)
            for step_idx in range(traj.length)
        ]
        if not self._step_index:
            raise ValueError("ExpertReplayBuffer has no expert steps")

    @property
    def num_trajectories(self) -> int:
        return len(self.trajectories)

    @property
    def num_steps(self) -> int:
        return len(self._step_index)

    def reference_objective(self, instance_id: str | None) -> float | None:
        if instance_id is None:
            return None
        value = self.objective_by_instance_id.get(str(instance_id))
        return None if value is None else float(value)

    def trajectory_for_instance(self, instance_id: str | None) -> ExpertTrajectory | None:
        if instance_id is None:
            return None
        return self.trajectory_by_instance_id.get(str(instance_id))

    def sample_step_batch(self, batch_size: int) -> tuple[dict[str, np.ndarray], torch.Tensor]:
        obs_batch, action_tensor, _ = self.sample_step_batch_with_objectives(batch_size)
        return obs_batch, action_tensor

    def sample_step_batch_with_objectives(
        self,
        batch_size: int,
    ) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor]:
        indices = self.rng.integers(0, len(self._step_index), size=max(1, int(batch_size)))
        observations = []
        actions = []
        objectives = []
        for index in indices:
            traj_idx, step_idx = self._step_index[int(index)]
            traj = self.trajectories[traj_idx]
            observations.append(traj.observations[step_idx])
            actions.append(traj.actions[step_idx])
            objectives.append(float(traj.objective_distance_km))
        obs_batch = stack_observations(observations)
        action_tensor = torch.as_tensor(np.asarray(actions, dtype=np.int64)[:, None], dtype=torch.long)
        objective_tensor = torch.as_tensor(np.asarray(objectives, dtype=np.float32), dtype=torch.float32)
        return obs_batch, action_tensor, objective_tensor


def compute_bc_loss(agent, buffer: ExpertReplayBuffer, batch_size: int, device: str | torch.device):
    obs_batch, action = buffer.sample_step_batch(batch_size)
    action = action.to(device)
    _, logprob, entropy, _ = agent.get_action_and_value(obs_batch, action=action)
    logprob_flat = logprob.reshape(-1)
    entropy_flat = entropy.reshape(-1)
    loss = -logprob_flat.mean()
    with torch.no_grad():
        _, logits = agent(obs_batch)
        pred_action = logits.reshape(-1, logits.size(-1)).argmax(dim=-1)
        acc = (pred_action == action.reshape(-1)).float().mean()
    return loss, {
        "bc_loss": float(loss.detach().cpu().item()),
        "bc_accuracy": float(acc.detach().cpu().item()),
        "bc_action_accuracy": float(acc.detach().cpu().item()),
        "bc_entropy": float(entropy_flat.mean().detach().cpu().item()),
        "bc_steps": int(action.numel()),
    }


def compute_awbc_loss(
    agent,
    buffer: ExpertReplayBuffer,
    batch_size: int,
    device: str | torch.device,
    *,
    eta: float = 0.05,
    normalize: str = "p95",
    baseline: str = "batch_mean",
    baseline_objective: float | None = None,
):
    obs_batch, action, objective = buffer.sample_step_batch_with_objectives(batch_size)
    action = action.to(device)
    objective = objective.to(device)
    _, logprob, entropy, _ = agent.get_action_and_value(obs_batch, action=action)
    logprob_flat = logprob.reshape(-1)
    entropy_flat = entropy.reshape(-1)
    obj_flat = objective.reshape(-1).to(logprob_flat.dtype)

    with torch.no_grad():
        if baseline_objective is not None and np.isfinite(float(baseline_objective)):
            base = torch.as_tensor(float(baseline_objective), dtype=obj_flat.dtype, device=obj_flat.device)
        elif baseline == "batch_median":
            base = torch.quantile(obj_flat, 0.5)
        elif baseline == "batch_p75":
            base = torch.quantile(obj_flat, 0.75)
        elif baseline == "batch_p90":
            base = torch.quantile(obj_flat, 0.90)
        else:
            base = obj_flat.mean()
        adv_pos = torch.clamp(base - obj_flat, min=0.0)
        positive = adv_pos[adv_pos > 0]
        if normalize == "eta_objective":
            denom = torch.clamp(float(eta) * obj_flat.abs(), min=1e-8)
            weights = torch.clamp(adv_pos / denom, 0.0, 1.0)
        elif positive.numel() > 0:
            q = torch.quantile(positive, 0.95)
            weights = torch.clamp(adv_pos / torch.clamp(q, min=1e-8), 0.0, 1.0)
        else:
            weights = torch.zeros_like(adv_pos)
    denom = weights.sum().clamp_min(1.0)
    loss = -(weights * logprob_flat).sum() / denom
    with torch.no_grad():
        _, logits = agent(obs_batch)
        pred_action = logits.reshape(-1, logits.size(-1)).argmax(dim=-1)
        acc = (pred_action == action.reshape(-1)).float().mean()
        active = weights > 0
    return loss, {
        "bc_loss": float(loss.detach().cpu().item()),
        "bc_accuracy": float(acc.detach().cpu().item()),
        "bc_action_accuracy": float(acc.detach().cpu().item()),
        "bc_entropy": float(entropy_flat.mean().detach().cpu().item()),
        "bc_steps": int(action.numel()),
        "awbc_loss": float(loss.detach().cpu().item()),
        "awbc_weight_mean": float(weights.mean().detach().cpu().item()),
        "awbc_weight_std": float(weights.std(unbiased=False).detach().cpu().item()),
        "awbc_active_ratio": float(active.float().mean().detach().cpu().item()),
        "awbc_expert_better_ratio": float(active.float().mean().detach().cpu().item()),
    }
