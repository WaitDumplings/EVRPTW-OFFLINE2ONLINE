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

from evrptw_core.io import iter_instances
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.env_factory import make_terran_env
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.rollout import stack_observations


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
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
    return out if out.is_absolute() else EVRPTW_DB_ROOT / out


def route_actions(routes: Sequence[Sequence[int]]) -> list[int]:
    actions: list[int] = []
    for route in routes:
        route_list = [int(x) for x in route]
        if len(route_list) < 2:
            continue
        if route_list[0] != 0:
            raise ValueError(f"expert route must start at depot 0, got {route_list}")
        actions.extend(route_list[1:])
    return actions


def _parse_routes(raw: str) -> list[list[int]]:
    routes = json.loads(raw)
    if not routes:
        return []
    if isinstance(routes[0], (int, float, str)):
        return [[int(node) for node in routes]]
    return [[int(node) for node in route] for route in routes]


def _route_payload(row: dict[str, str]) -> str:
    return str(row.get("routes_json") or row.get("route_sequence_json") or "")


def _is_usable_solution(row: dict[str, str]) -> bool:
    status = str(row.get("status", "")).strip().upper()
    status_name = str(row.get("status_name", "")).strip().upper()
    # Empty status is allowed for generic solver/heuristic archives. Gurobi may
    # use numeric status 2, while checkpoint traces often store RUNNING with an
    # incumbent.
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


# Backward-compatible alias for older configs and scripts.
load_gurobi_expert_records = load_solver_expert_records


def build_expert_trajectories(
    records: Sequence[ExpertRecord],
    cfg: dict[str, Any],
    *,
    max_records: int | None = None,
    strict: bool = True,
) -> tuple[list[ExpertTrajectory], dict[str, Any]]:
    env_cfg = dict(cfg.get("env", {}) or {})
    env_cfg["use_fast_env"] = True
    env_cfg.setdefault("info_level", "light")
    trajectories: list[ExpertTrajectory] = []
    invalid_records: list[dict[str, Any]] = []
    selected_records = list(records[: max_records or len(records)])
    for record in selected_records:
        env = make_terran_env(
            instance=record.instance,
            n_traj=1,
            pbrs_config=None,
            **env_cfg,
        )
        obs, info = env.reset()
        observations: list[dict[str, np.ndarray]] = []
        actions: list[int] = []
        invalid_step: dict[str, Any] | None = None
        for step_idx, action in enumerate(route_actions(record.routes)):
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
        first = invalid_records[0]
        raise ValueError(f"{len(invalid_records)} expert routes failed replay; first={first}")
    stats = {
        "records_seen": len(selected_records),
        "trajectories": len(trajectories),
        "invalid_records": len(invalid_records),
        "steps": int(sum(traj.length for traj in trajectories)),
        "avg_steps_per_route": float(np.mean([traj.length for traj in trajectories])) if trajectories else 0.0,
    }
    return trajectories, stats


class ExpertReplayBuffer:
    def __init__(self, trajectories: Sequence[ExpertTrajectory], seed: int = 0) -> None:
        self.trajectories = list(trajectories)
        if not self.trajectories:
            raise ValueError("ExpertReplayBuffer requires at least one trajectory")
        self.objective_by_instance_id = {
            traj.instance_id: float(traj.objective_distance_km)
            for traj in self.trajectories
        }
        self.rng = np.random.default_rng(int(seed))
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

    def sample_step_batch(self, batch_size: int) -> tuple[dict[str, np.ndarray], torch.Tensor]:
        indices = self.rng.integers(0, len(self._step_index), size=max(1, int(batch_size)))
        observations = []
        actions = []
        for index in indices:
            traj_idx, step_idx = self._step_index[int(index)]
            traj = self.trajectories[traj_idx]
            observations.append(traj.observations[step_idx])
            actions.append(traj.actions[step_idx])
        obs_batch = stack_observations(observations)
        action_tensor = torch.as_tensor(np.asarray(actions, dtype=np.int64)[:, None], dtype=torch.long)
        return obs_batch, action_tensor

    def sample_route_batch(self, batch_size: int) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor]:
        traj_indices = self.rng.integers(0, len(self.trajectories), size=max(1, int(batch_size)))
        observations = []
        actions = []
        route_ids = []
        for local_route_idx, traj_index in enumerate(traj_indices):
            traj = self.trajectories[int(traj_index)]
            for obs, action in zip(traj.observations, traj.actions):
                observations.append(obs)
                actions.append(action)
                route_ids.append(local_route_idx)
        obs_batch = stack_observations(observations)
        action_tensor = torch.as_tensor(np.asarray(actions, dtype=np.int64)[:, None], dtype=torch.long)
        route_tensor = torch.as_tensor(np.asarray(route_ids, dtype=np.int64), dtype=torch.long)
        return obs_batch, action_tensor, route_tensor


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
        "bc_entropy": float(entropy_flat.mean().detach().cpu().item()),
        "bc_steps": int(action.numel()),
    }


def compute_route_supervised_loss(agent, buffer: ExpertReplayBuffer, batch_size: int, device: str | torch.device):
    obs_batch, action, route_id = buffer.sample_route_batch(batch_size)
    action = action.to(device)
    route_id = route_id.to(device)
    _, logprob, entropy, _ = agent.get_action_and_value(obs_batch, action=action)
    logprob_flat = logprob.reshape(-1)
    entropy_flat = entropy.reshape(-1)
    num_routes = int(route_id.max().detach().cpu().item()) + 1 if route_id.numel() else 0
    route_logprob_sum = torch.zeros(num_routes, dtype=logprob_flat.dtype, device=logprob_flat.device)
    route_len = torch.zeros(num_routes, dtype=logprob_flat.dtype, device=logprob_flat.device)
    route_logprob_sum.scatter_add_(0, route_id, logprob_flat)
    route_len.scatter_add_(0, route_id, torch.ones_like(logprob_flat))
    route_mean_logprob = route_logprob_sum / route_len.clamp_min(1.0)
    loss = -route_mean_logprob.mean()
    return loss, {
        "sl_route_loss": float(loss.detach().cpu().item()),
        "sl_route_entropy": float(entropy_flat.mean().detach().cpu().item()),
        "sl_route_count": int(num_routes),
        "sl_step_count": int(action.numel()),
        "sl_avg_route_len": float(route_len.mean().detach().cpu().item()) if num_routes else 0.0,
    }
