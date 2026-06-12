from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .models import Agent
from .offline_data import load_solver_expert_records, resolve_repo_path, route_actions
from .trainer import load_config, set_seed
from .integrations.evrptw_db import configure_evrptw_db

EVRPTW_DB_ROOT = configure_evrptw_db()

from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.env_factory import make_terran_env
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.rollout import sample_actions, stack_observations

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_int_list(raw: str) -> list[int]:
    values = []
    for part in str(raw).split(','):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def build_agent(cfg: dict[str, Any], device: str | torch.device) -> Agent:
    model_cfg = cfg.get('model', {}) or {}
    agent = Agent(
        embedding_dim=int(model_cfg.get('embedding_dim', 256)),
        tanh_clipping=float(model_cfg.get('tanh_clipping', 15.0)),
        n_encode_layers=int(model_cfg.get('n_encode_layers', 2)),
        device=device,
        use_graph_token=bool(model_cfg.get('use_graph_token', True)),
        use_dynamic_decision_encoder=bool(model_cfg.get('use_dynamic_decision_encoder', False)),
        dynamic_decision_heads=int(model_cfg.get('dynamic_decision_heads', 4)),
    ).to(device)
    return agent


def load_checkpoint(agent: Agent, checkpoint_path: str | Path | None, device: str | torch.device) -> dict[str, Any]:
    if checkpoint_path is None or str(checkpoint_path) == '':
        return {'loaded': False, 'missing_keys': [], 'unexpected_keys': []}
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_absolute():
        ckpt_path = REPO_ROOT / ckpt_path
    payload = torch.load(ckpt_path, map_location=device)
    state = payload.get('model_state_dict', payload)
    result = agent.load_state_dict(state, strict=False)
    return {
        'loaded': True,
        'checkpoint_path': str(ckpt_path),
        'epoch': payload.get('epoch', ''),
        'seed': payload.get('seed', ''),
        'missing_keys': list(result.missing_keys),
        'unexpected_keys': list(result.unexpected_keys),
    }


def make_env_cfg(cfg: dict[str, Any], info_level: str = 'light') -> dict[str, Any]:
    env_cfg = dict(cfg.get('env', {}) or {})
    # The trainer resolves dataset-level reward-scale modes through the train
    # dataset pool before constructing envs. Diagnostics only need objective
    # distances, so map dataset-prefixed modes back to the env-supported local
    # mode to keep fixed-instance probing self-contained.
    scale_mode = str(env_cfg.get('reward_distance_scale_mode', ''))
    if scale_mode.startswith('dataset_'):
        env_cfg['reward_distance_scale_mode'] = scale_mode[len('dataset_'):]
    env_cfg['use_fast_env'] = True
    env_cfg['info_level'] = info_level
    return env_cfg


def _info_arrays(info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    success = np.asarray(info.get('success', []), dtype=bool)
    objective = np.asarray(info.get('objective_distance_km', []), dtype=np.float64)
    served = np.asarray(info.get('served_customers', []), dtype=np.int64)
    if objective.size == 0:
        objective = np.full(success.shape, np.nan, dtype=np.float64)
    if served.size == 0:
        served = np.zeros(success.shape, dtype=np.int64)
    return success, objective, served


def _successful_objectives(info: dict[str, Any]) -> np.ndarray:
    success, objective, _ = _info_arrays(info)
    if success.size == 0:
        return np.asarray([], dtype=np.float64)
    mask = success & np.isfinite(objective)
    return objective[mask].astype(np.float64)


def _top_mean(values: np.ndarray, q: float = 0.20) -> float:
    if values.size == 0:
        return float('nan')
    n = max(1, int(math.ceil(float(q) * values.size)))
    return float(np.mean(np.sort(values)[:n]))


def rollout_policy(agent: Agent, env, *, max_steps: int, decode_mode: str, device: str | torch.device, seed: int | None) -> dict[str, Any]:
    obs, info = env.reset(seed=seed) if seed is not None else env.reset()
    n_traj = int(env.unwrapped.n_traj)
    done = np.zeros(n_traj, dtype=bool)
    for _ in range(int(max_steps)):
        obs_batch = stack_observations([obs])
        with torch.no_grad():
            actions, _, _, _, _ = sample_actions(agent, obs_batch, decode_mode=decode_mode, device=device)
        action_np = actions.squeeze(0).detach().cpu().numpy().astype(np.int64)
        obs, reward, terminated, truncated, info = env.step(action_np)
        done = done | np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
        if done.all():
            break
    return info


def complete_from_incumbent_prefix(
    agent: Agent,
    instance: Any,
    prefix_actions: list[int],
    cfg: dict[str, Any],
    *,
    completions: int,
    max_steps: int,
    decode_mode: str,
    device: str | torch.device,
    seed: int | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    env = make_terran_env(
        instance=instance,
        n_traj=max(1, int(completions)),
        pbrs_config=None,
        **make_env_cfg(cfg, info_level='light'),
    )
    obs, info = env.reset(seed=seed) if seed is not None else env.reset()
    n_traj = int(env.unwrapped.n_traj)
    done = np.zeros(n_traj, dtype=bool)
    replay_info = {
        'prefix_valid': True,
        'invalid_step': '',
        'invalid_action': '',
        'terminated_during_prefix': False,
    }
    for step_idx, action in enumerate(prefix_actions):
        mask = np.asarray(obs['action_mask'], dtype=bool)
        action_i = int(action)
        if action_i < 0 or action_i >= mask.shape[1] or not bool(mask[:, action_i].all()):
            replay_info.update(
                {
                    'prefix_valid': False,
                    'invalid_step': int(step_idx),
                    'invalid_action': int(action_i),
                }
            )
            return None, replay_info
        action_np = np.full(n_traj, action_i, dtype=np.int64)
        obs, reward, terminated, truncated, info = env.step(action_np)
        step_done = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
        done = done | step_done
        if done.any() and step_idx + 1 < len(prefix_actions):
            replay_info['terminated_during_prefix'] = True
            replay_info.update(
                {
                    'prefix_valid': False,
                    'invalid_step': int(step_idx),
                    'invalid_action': int(action_i),
                }
            )
            return None, replay_info
    if not done.all():
        for _ in range(max(0, int(max_steps) - len(prefix_actions))):
            obs_batch = stack_observations([obs])
            with torch.no_grad():
                actions, _, _, _, _ = sample_actions(agent, obs_batch, decode_mode=decode_mode, device=device)
            action_np = actions.squeeze(0).detach().cpu().numpy().astype(np.int64)
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = done | np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
            if done.all():
                break
    return info, replay_info


def summarize_policy(info: dict[str, Any], incumbent_obj: float, gap_floor_ratio: float) -> dict[str, float]:
    success_objs = _successful_objectives(info)
    if success_objs.size == 0:
        return {
            'policy_success_count': 0,
            'policy_best': float('nan'),
            'policy_mean': float('nan'),
            'policy_std': float('nan'),
            'policy_top_mean': float('nan'),
            'gap_best': float('nan'),
            'gap_mean': float('nan'),
            'gap_den': float('nan'),
        }
    best = float(np.min(success_objs))
    mean = float(np.mean(success_objs))
    std = float(np.std(success_objs))
    top_mean = _top_mean(success_objs, 0.20)
    gap_best = best - float(incumbent_obj)
    gap_mean = mean - float(incumbent_obj)
    gap_den = max(gap_best, float(gap_floor_ratio) * max(float(incumbent_obj), 1e-8), 1e-8)
    return {
        'policy_success_count': int(success_objs.size),
        'policy_best': best,
        'policy_mean': mean,
        'policy_std': std,
        'policy_top_mean': top_mean,
        'gap_best': gap_best,
        'gap_mean': gap_mean,
        'gap_den': gap_den,
    }


def run_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    set_seed(int(args.seed))
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    agent = build_agent(cfg, device=device)
    ckpt_info = load_checkpoint(agent, args.checkpoint, device=device)
    agent.eval()

    data_cfg = cfg.get('data', {}) or {}
    offline_cfg = cfg.get('offline', {}) or {}
    dataset_path = args.dataset_path or offline_cfg.get('expert_dataset_path') or data_cfg.get('train_dataset_path')
    solution_path = args.expert_solution_path or offline_cfg.get('expert_solution_path')
    if dataset_path is None or solution_path is None:
        raise ValueError('Provide --dataset-path and --expert-solution-path or set them in config.offline/data.')
    records = load_solver_expert_records(
        dataset_path=dataset_path,
        solution_csv_path=solution_path,
        num_customers=int(data_cfg.get('num_customers', args.num_customers or 15)),
        num_charging_stations=int(data_cfg.get('num_charging_stations', args.num_charging_stations or 3)),
        limit=args.limit,
    )
    prefix_lengths = parse_int_list(args.prefix_lengths)
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / 'results' / 'diagnostics' / 'gcbpo_branch'
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = args.tag or f"cus{data_cfg.get('num_customers', 'x')}_seed{args.seed}_n{len(records)}_k{args.n_traj}_m{args.branch_completions}"
    per_prefix_path = output_dir / f'{stamp}_per_prefix.csv'
    summary_path = output_dir / f'{stamp}_summary.json'

    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for rec_idx, record in enumerate(records):
        env = make_terran_env(
            instance=record.instance,
            n_traj=int(args.n_traj),
            pbrs_config=None,
            **make_env_cfg(cfg, info_level='light'),
        )
        policy_info = rollout_policy(
            agent,
            env,
            max_steps=int(args.max_steps),
            decode_mode=str(args.decode_mode),
            device=device,
            seed=int(args.seed) * 1_000_000 + rec_idx,
        )
        policy_stats = summarize_policy(policy_info, record.objective_distance_km, float(args.gap_floor_ratio))
        expert_actions = route_actions(record.routes)
        for prefix_len in prefix_lengths:
            prefix_len_i = int(prefix_len)
            if prefix_len_i <= 0 or prefix_len_i > len(expert_actions):
                rows.append(
                    {
                        'instance_id': record.instance_id,
                        'prefix_len': prefix_len_i,
                        'prefix_valid': False,
                        'invalid_reason': 'prefix_too_long',
                        'incumbent_obj': record.objective_distance_km,
                        **policy_stats,
                    }
                )
                continue
            branch_info, replay = complete_from_incumbent_prefix(
                agent,
                record.instance,
                expert_actions[:prefix_len_i],
                cfg,
                completions=int(args.branch_completions),
                max_steps=int(args.max_steps),
                decode_mode=str(args.decode_mode),
                device=device,
                seed=int(args.seed) * 1_000_000 + 100_000 + rec_idx * 100 + prefix_len_i,
            )
            branch_objs = _successful_objectives(branch_info or {})
            branch_success_count = int(branch_objs.size)
            branch_best = float(np.min(branch_objs)) if branch_objs.size else float('nan')
            branch_mean = float(np.mean(branch_objs)) if branch_objs.size else float('nan')
            policy_best = float(policy_stats['policy_best'])
            policy_top_mean = float(policy_stats['policy_top_mean'])
            gap_den = float(policy_stats['gap_den'])
            incumbent_obj = float(record.objective_distance_km)
            strong = bool(np.isfinite(branch_best) and np.isfinite(policy_best) and branch_best < policy_best - float(args.margin_abs))
            soft = bool(np.isfinite(branch_best) and np.isfinite(policy_top_mean) and branch_best < policy_top_mean - float(args.margin_abs))
            gap_close = 0.0
            soft_weight = 0.0
            if np.isfinite(branch_best) and np.isfinite(policy_best) and np.isfinite(gap_den):
                gap_close = float(np.clip((policy_best - branch_best) / max(gap_den, 1e-8), 0.0, 1.0))
            if np.isfinite(branch_best) and np.isfinite(policy_top_mean):
                soft_den = max(policy_top_mean - incumbent_obj, float(args.gap_floor_ratio) * max(incumbent_obj, 1e-8), 1e-8)
                soft_weight = float(np.clip((policy_top_mean - branch_best) / soft_den, 0.0, 1.0))
            rows.append(
                {
                    'instance_id': record.instance_id,
                    'record_index': rec_idx,
                    'prefix_len': prefix_len_i,
                    'prefix_valid': bool(replay.get('prefix_valid', False)),
                    'invalid_step': replay.get('invalid_step', ''),
                    'invalid_action': replay.get('invalid_action', ''),
                    'terminated_during_prefix': bool(replay.get('terminated_during_prefix', False)),
                    'incumbent_obj': incumbent_obj,
                    **policy_stats,
                    'branch_success_count': branch_success_count,
                    'branch_best': branch_best,
                    'branch_mean': branch_mean,
                    'branch_strong': strong,
                    'branch_soft': soft,
                    'branch_gap_close': gap_close,
                    'branch_soft_weight': soft_weight,
                    'branch_improvement_vs_policy_best': policy_best - branch_best if np.isfinite(policy_best) and np.isfinite(branch_best) else float('nan'),
                    'branch_improvement_vs_policy_top_mean': policy_top_mean - branch_best if np.isfinite(policy_top_mean) and np.isfinite(branch_best) else float('nan'),
                }
            )

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with per_prefix_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [row for row in rows if bool(row.get('prefix_valid'))]
    strong_rows = [row for row in valid_rows if bool(row.get('branch_strong'))]
    soft_rows = [row for row in valid_rows if bool(row.get('branch_soft'))]
    instance_ids = sorted({str(row['instance_id']) for row in rows})
    strong_instances = sorted({str(row['instance_id']) for row in strong_rows})
    soft_instances = sorted({str(row['instance_id']) for row in soft_rows})
    prefix_counter_strong = Counter(int(row['prefix_len']) for row in strong_rows)
    prefix_counter_soft = Counter(int(row['prefix_len']) for row in soft_rows)
    gap_closes = np.asarray([float(row.get('branch_gap_close', 0.0)) for row in valid_rows], dtype=np.float64)
    positive_gap_closes = gap_closes[gap_closes > 0]
    policy_best_gaps_by_instance = {}
    for row in rows:
        iid = str(row['instance_id'])
        if iid not in policy_best_gaps_by_instance:
            policy_best_gaps_by_instance[iid] = row.get('gap_best', float('nan'))
    gap_values = np.asarray([float(v) for v in policy_best_gaps_by_instance.values() if np.isfinite(float(v))], dtype=np.float64)
    summary = {
        'method': 'GCBPO_v2.0_branch_diagnostics',
        'config': str(args.config),
        'checkpoint': ckpt_info,
        'dataset_path': str(dataset_path),
        'expert_solution_path': str(solution_path),
        'num_instances': len(instance_ids),
        'num_prefix_rows': len(rows),
        'valid_prefix_rows': len(valid_rows),
        'invalid_prefix_rows': len(rows) - len(valid_rows),
        'branch_beats_pomo_best_rows': len(strong_rows),
        'branch_beats_pomo_top_mean_rows': len(soft_rows),
        'branch_beats_pomo_best_ratio_per_prefix': len(strong_rows) / max(len(valid_rows), 1),
        'branch_beats_pomo_top_mean_ratio_per_prefix': len(soft_rows) / max(len(valid_rows), 1),
        'branch_beats_pomo_best_instances': len(strong_instances),
        'branch_beats_pomo_top_mean_instances': len(soft_instances),
        'branch_beats_pomo_best_ratio_per_instance': len(strong_instances) / max(len(instance_ids), 1),
        'branch_beats_pomo_top_mean_ratio_per_instance': len(soft_instances) / max(len(instance_ids), 1),
        'mean_branch_gap_close_all_valid': float(np.mean(gap_closes)) if gap_closes.size else 0.0,
        'mean_branch_gap_close_positive': float(np.mean(positive_gap_closes)) if positive_gap_closes.size else 0.0,
        'policy_gap_best_mean': float(np.mean(gap_values)) if gap_values.size else float('nan'),
        'policy_gap_best_median': float(np.median(gap_values)) if gap_values.size else float('nan'),
        'strong_prefix_histogram': dict(sorted(prefix_counter_strong.items())),
        'soft_prefix_histogram': dict(sorted(prefix_counter_soft.items())),
        'n_traj': int(args.n_traj),
        'branch_completions_per_prefix': int(args.branch_completions),
        'prefix_lengths': prefix_lengths,
        'elapsed_s': float(time.perf_counter() - start),
        'per_prefix_csv': str(per_prefix_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='GCBPO v2.0 incumbent-prefix branch diagnostics.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--seed', type=int, default=2005)
    parser.add_argument('--device', default=None)
    parser.add_argument('--dataset-path', default=None)
    parser.add_argument('--expert-solution-path', default=None)
    parser.add_argument('--num-customers', type=int, default=None)
    parser.add_argument('--num-charging-stations', type=int, default=None)
    parser.add_argument('--limit', type=int, default=32)
    parser.add_argument('--n-traj', type=int, default=50)
    parser.add_argument('--branch-completions', type=int, default=2)
    parser.add_argument('--prefix-lengths', default='1,2,3,5,8')
    parser.add_argument('--max-steps', type=int, default=40)
    parser.add_argument('--decode-mode', choices=['sample', 'greedy'], default='sample')
    parser.add_argument('--gap-floor-ratio', type=float, default=0.01)
    parser.add_argument('--margin-abs', type=float, default=0.0)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--tag', default=None)
    return parser.parse_args()


def main() -> None:
    run_diagnostics(parse_args())


if __name__ == '__main__':
    main()
