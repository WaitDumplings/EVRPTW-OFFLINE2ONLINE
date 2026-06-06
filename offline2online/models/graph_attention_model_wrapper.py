from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .nets.graph_model.decoder import Decoder
from .nets.graph_model.embedding import AutoEmbedding
from .nets.graph_model.encoder import GraphAttentionEncoder


class Problem:
    def __init__(self, name: str):
        self.NAME = name


def _to_tensor(value: Any, device: str | torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.as_tensor(value, device=device)


def prepare_observation_batch(obs: dict[str, Any]) -> dict[str, Any]:
    """Add the outer env-batch dimension when a single EVRPTW env obs is passed."""
    out: dict[str, Any] = {}
    for key, value in obs.items():
        arr = np.asarray(value)
        if key in {"cus_loc", "rs_loc", "time_window", "action_mask"}:
            out[key] = arr[None, ...] if arr.ndim == 2 else value
        elif key == "depot_loc":
            out[key] = arr[None, ...] if arr.ndim == 2 else value
        elif key in {
            "demand",
            "service_time",
            "last_node_idx",
            "prev_node_idx",
            "current_load",
            "current_battery",
            "remaining_battery",
            "current_time",
        }:
            out[key] = arr[None, ...] if arr.ndim == 1 else value
        elif key in {
            "visited_customers_ratio",
            "visited_customers_raio",
            "remain_feasible_customers_ratio",
            "remain_feasible_customers_raio",
            "route_served_customers_ratio",
            "rs_streak_ratio",
        }:
            out[key] = arr[None, ...] if arr.ndim == 2 else value
        elif key in {"battery_capacity", "loading_capacity"}:
            out[key] = arr[None, ...] if arr.ndim == 1 else value
        else:
            out[key] = value
    if "visited_customers_ratio" not in out and "visited_customers_raio" in out:
        out["visited_customers_ratio"] = out["visited_customers_raio"]
    if "visited_customers_raio" not in out and "visited_customers_ratio" in out:
        out["visited_customers_raio"] = out["visited_customers_ratio"]
    if "remain_feasible_customers_ratio" not in out and "remain_feasible_customers_raio" in out:
        out["remain_feasible_customers_ratio"] = out["remain_feasible_customers_raio"]
    if "remain_feasible_customers_raio" not in out and "remain_feasible_customers_ratio" in out:
        out["remain_feasible_customers_raio"] = out["remain_feasible_customers_ratio"]
    return out


def orthogonal_init(layer: nn.Module, gain: float = 1.0) -> None:
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class StateWrapper:
    """Adapt EVRPTW-DB Gymnasium observations to the Ablation graph model."""

    def __init__(self, states: dict[str, Any], device: str | torch.device, problem: str = "evrptw"):
        self.device = device
        self.problem = problem
        states = prepare_observation_batch(states)
        self.states = {key: _to_tensor(value, device=device) for key, value in states.items()}
        if problem == "evrptw":
            self._build_evrptw_state()

    @staticmethod
    def _as_node_scalar(value: torch.Tensor) -> torch.Tensor:
        value = value.float()
        if value.dim() == 3 and value.size(-1) == 1:
            return value
        if value.dim() == 2:
            return value.unsqueeze(-1)
        raise ValueError(f"Expected node scalar with shape [B,N] or [B,N,1], got {tuple(value.shape)}")

    def _build_evrptw_state(self) -> None:
        demand = self._as_node_scalar(self.states["demand"])
        service_time = self._as_node_scalar(self.states["service_time"])
        time_window = self.states["time_window"].float()

        observations = {
            "depot_loc": self.states["depot_loc"].float(),
            "cus_loc": self.states["cus_loc"].float(),
            "rs_loc": self.states["rs_loc"].float(),
            "time_window": time_window,
            "demand": demand,
            "service_time": service_time,
        }
        self.states["observations"] = observations
        self.VEHICLE_CAPACITY = self.states["loading_capacity"]
        self.VEHICLE_BATTERY = self.states["battery_capacity"]
        self.used_capacity = self.states["current_load"].float()
        self.used_battery = self.states["current_battery"].float()
        self.current_time = self.states["current_time"].float()
        self.visited_customers_ratio = self.states["visited_customers_ratio"].float()
        self.remain_feasible_customers_ratio = self.states["remain_feasible_customers_ratio"].float()

    @property
    def observations(self) -> dict[str, torch.Tensor]:
        return self.states["observations"]

    def get_current_node(self) -> torch.Tensor:
        return self.states["last_node_idx"].long()

    def get_mask(self) -> torch.Tensor:
        return ~self.states["action_mask"].bool()


class Backbone(nn.Module):
    """Ablation TERRAN-style backbone with graph token and dynamic embeddings."""

    def __init__(
        self,
        embedding_dim: int = 256,
        problem_name: str = "evrptw",
        n_encode_layers: int = 2,
        tanh_clipping: float = 15.0,
        n_heads: int = 16,
        device: str | torch.device = "cpu",
        use_graph_token: bool = True,
        use_dynamic_embedding: bool = False,
        use_candidate_dynamic_embedding: bool | None = None,
    ):
        super().__init__()
        del use_graph_token  # graph token is intrinsic to the migrated graph encoder.
        self.device = device
        self.problem = Problem(problem_name)
        if use_candidate_dynamic_embedding is None:
            use_candidate_dynamic_embedding = use_dynamic_embedding
        self.use_candidate_dynamic_embedding = bool(use_candidate_dynamic_embedding)

        self.embedding = AutoEmbedding(self.problem.NAME, {"embedding_dim": embedding_dim})
        self.encoder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=n_encode_layers,
        )
        self.decoder = Decoder(
            embedding_dim=embedding_dim,
            step_context_dim=embedding_dim + 5,
            n_heads=n_heads,
            problem=self.problem,
            tanh_clipping=tanh_clipping,
            use_candidate_dynamic_embedding=self.use_candidate_dynamic_embedding,
        )

        self.dist_bias_scale = nn.Parameter(torch.tensor(1.0))
        self.type_pair_bias = nn.Embedding(3 * 3, 1)
        nn.init.zeros_(self.type_pair_bias.weight)

    def _build_node_type(self, node_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        depot_loc = node_inputs["depot_loc"]
        batch_size = depot_loc.size(0)
        n_cus = node_inputs["cus_loc"].size(1)
        n_rs = node_inputs["rs_loc"].size(1)
        device = node_inputs["cus_loc"].device
        depot_type = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        customer_type = torch.full((batch_size, n_cus), 2, dtype=torch.long, device=device)
        station_type = torch.ones(batch_size, n_rs, dtype=torch.long, device=device)
        return torch.cat([depot_type, customer_type, station_type], dim=1)

    def _build_distance_matrix(self, node_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        depot_loc = node_inputs["depot_loc"]
        if depot_loc.dim() == 2:
            depot_loc = depot_loc.unsqueeze(1)
        nodes = torch.cat([depot_loc, node_inputs["cus_loc"], node_inputs["rs_loc"]], dim=1)
        return torch.cdist(nodes, nodes, p=2)

    def _build_attn_bias(self, state: StateWrapper) -> torch.Tensor:
        node_inputs = state.observations
        dist_mat = self._build_distance_matrix(node_inputs)
        node_type = self._build_node_type(node_inputs)
        dist_bias = -self.dist_bias_scale * dist_mat
        pair_id = node_type.unsqueeze(2) * 3 + node_type.unsqueeze(1)
        type_bias = self.type_pair_bias(pair_id).squeeze(-1)
        attn_bias = dist_bias + type_bias

        edge_energy = state.states.get("edge_energy")
        battery_capacity = state.states.get("battery_capacity")
        if edge_energy is None or battery_capacity is None:
            return attn_bias
        edge_energy = edge_energy.to(device=attn_bias.device, dtype=attn_bias.dtype)
        if edge_energy.dim() == 2:
            edge_energy = edge_energy.unsqueeze(0)
        if edge_energy.size(0) == 1 and attn_bias.size(0) != 1:
            edge_energy = edge_energy.expand(attn_bias.size(0), -1, -1)
        battery_capacity = battery_capacity.to(device=attn_bias.device, dtype=attn_bias.dtype)
        if battery_capacity.dim() == 0:
            battery_capacity = battery_capacity.view(1, 1, 1)
        elif battery_capacity.dim() == 1:
            battery_capacity = battery_capacity.view(-1, 1, 1)
        else:
            battery_capacity = battery_capacity.reshape(battery_capacity.size(0), -1)[:, :1].view(-1, 1, 1)
        if battery_capacity.size(0) == 1 and attn_bias.size(0) != 1:
            battery_capacity = battery_capacity.expand(attn_bias.size(0), -1, -1)
        unreachable = edge_energy > (battery_capacity + 1e-6)
        eye = torch.eye(attn_bias.size(-1), dtype=torch.bool, device=attn_bias.device).unsqueeze(0)
        return attn_bias.masked_fill(unreachable & ~eye, -1e9)

    def _build_state(self, obs: dict[str, Any]) -> StateWrapper:
        return StateWrapper(obs, device=self.device, problem=self.problem.NAME)

    def _encode_from_state(self, state: StateWrapper, use_mask: bool = False):
        node_mask = state.states.get("instance_mask") if use_mask else None
        if node_mask is not None:
            node_mask = node_mask.bool()
        node_embeddings = self.embedding(state.observations)
        encoded_nodes = self.encoder(
            node_embeddings,
            mask=None,
            attn_bias=self._build_attn_bias(state),
        )
        return self.decoder._precompute(encoded_nodes, mask=node_mask), node_mask

    def forward(self, obs: dict[str, Any], use_mask: bool = False):
        state = self._build_state(obs)
        cached_embeddings, node_mask = self._encode_from_state(state, use_mask=use_mask)
        logits, glimpse = self.decoder.advance(cached_embeddings, state, node_mask=node_mask)
        return logits, glimpse

    def encode(self, obs: dict[str, Any], use_mask: bool = False):
        state = self._build_state(obs)
        cached_embeddings, _ = self._encode_from_state(state, use_mask=use_mask)
        return cached_embeddings

    def decode(self, obs: dict[str, Any], cached_embeddings, use_mask: bool = False):
        state = self._build_state(obs)
        node_mask = state.states.get("instance_mask") if use_mask else None
        if node_mask is not None:
            node_mask = node_mask.bool()
        return self.decoder.advance(cached_embeddings, state, node_mask=node_mask)


class Actor(nn.Module):
    def forward(self, backbone_output):
        return backbone_output[0]


class Critic(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        for layer in self.mlp:
            orthogonal_init(layer, gain=0.01)

    def forward(self, backbone_output):
        return self.mlp(backbone_output[1])


class Agent(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        tanh_clipping: float = 15.0,
        n_encode_layers: int = 2,
        device: str | torch.device = "cpu",
        name: str = "evrptw",
        use_graph_token: bool = True,
        use_dynamic_embedding: bool = False,
        use_candidate_dynamic_embedding: bool | None = None,
    ):
        super().__init__()
        self.backbone = Backbone(
            embedding_dim=embedding_dim,
            device=device,
            tanh_clipping=tanh_clipping,
            n_encode_layers=n_encode_layers,
            problem_name=name,
            use_graph_token=use_graph_token,
            use_dynamic_embedding=use_dynamic_embedding,
            use_candidate_dynamic_embedding=use_candidate_dynamic_embedding,
        )
        self.actor = Actor()
        self.critic = Critic(hidden_size=embedding_dim)

    def forward(self, x, use_mask: bool = False, return_logits: bool = True):
        backbone_output = self.backbone(x, use_mask=use_mask)
        logits = self.actor(backbone_output)
        action = logits.max(2)[1]
        if not return_logits:
            return action
        return action, logits

    def get_value(self, x, use_mask: bool = False):
        return self.critic(self.backbone(x, use_mask=use_mask))

    def get_action_and_value(self, x, action=None, use_mask: bool = False):
        backbone_output = self.backbone(x, use_mask=use_mask)
        logits = self.actor(backbone_output)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(backbone_output)

    def get_acction_and_value(self, x, action=None, use_mask: bool = False):
        return self.get_action_and_value(x, action=action, use_mask=use_mask)

    def get_value_cached(self, x, state):
        return self.critic(self.backbone.decode(x, state))

    def get_action_and_value_cached(self, x, action=None, state=None, cached_embeddings=None, print_probs: bool = False):
        del print_probs
        if cached_embeddings is None:
            cached_embeddings = state
        if cached_embeddings is None:
            cached_embeddings = self.backbone.encode(x)
        backbone_output = self.backbone.decode(x, cached_embeddings)
        logits = self.actor(backbone_output)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(backbone_output), cached_embeddings

