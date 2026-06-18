import torch
from torch import nn

from ...nets.graph_model.multi_head_attention import (
    AttentionScore,
    MultiHeadAttention,
)



class DriverQueryEncoder(nn.Module):
    """Encode the driver/vehicle/route state into the decoder query.

    The query is deliberately route-side only. It contains the static graph token,
    the current node embedding, and scalar vehicle/route state. It does not pool
    remaining graph nodes or candidate features; those belong to
    DynamicGraphKVEncoder on the key/value/action side.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.feature_dim = 12
        self.state_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.query_proj = nn.Sequential(
            nn.LayerNorm(3 * embedding_dim),
            nn.Linear(3 * embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        nn.init.xavier_uniform_(self.state_proj[1].weight, gain=0.5)
        nn.init.zeros_(self.state_proj[1].bias)
        nn.init.xavier_uniform_(self.state_proj[3].weight, gain=0.5)
        nn.init.zeros_(self.state_proj[3].bias)
        nn.init.xavier_uniform_(self.query_proj[1].weight, gain=0.5)
        nn.init.zeros_(self.query_proj[1].bias)
        nn.init.xavier_uniform_(self.query_proj[3].weight, gain=0.5)
        nn.init.zeros_(self.query_proj[3].bias)

    @staticmethod
    def _step_count(state, fallback=1):
        action_mask = state.states.get("action_mask", None)
        if torch.is_tensor(action_mask) and action_mask.dim() >= 3:
            return int(action_mask.size(1))
        current = state.get_current_node()
        if current.dim() >= 2:
            return int(current.size(1))
        return int(fallback)

    @staticmethod
    def _expand_step(x, T):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.size(1) == 1 and T != 1:
            x = x.expand(-1, T, -1)
        return x

    @staticmethod
    def _as_step_scalar(x, T, like):
        if x is None:
            return like.new_zeros(like.size(0), T, 1)
        x = x.to(device=like.device, dtype=like.dtype)
        if x.dim() == 1:
            x = x[:, None, None]
        elif x.dim() == 2:
            x = x[:, :, None]
        if x.size(1) == 1 and T != 1:
            x = x.expand(-1, T, -1)
        return x

    @staticmethod
    def _as_step_index(node_idx, T, node_embeddings):
        if node_idx.dim() == 1:
            node_idx = node_idx.unsqueeze(1)
        node_idx = node_idx.to(device=node_embeddings.device, dtype=torch.long)
        if node_idx.size(1) == 1 and T != 1:
            node_idx = node_idx.expand(-1, T)
        return node_idx.clamp(min=0, max=node_embeddings.size(1) - 1)

    @staticmethod
    def _gather_node(node_embeddings, node_idx):
        if node_idx.dim() == 1:
            node_idx = node_idx.unsqueeze(1)
        node_idx = node_idx.to(device=node_embeddings.device, dtype=torch.long)
        node_idx = node_idx.clamp(min=0, max=node_embeddings.size(1) - 1)
        gather_idx = node_idx.unsqueeze(-1).expand(-1, -1, node_embeddings.size(-1))
        return torch.gather(node_embeddings, dim=1, index=gather_idx)

    def forward(self, node_embeddings, graph_context, state):
        T = self._step_count(state, fallback=1)
        graph_context = self._expand_step(graph_context, T)
        current_node_idx = self._as_step_index(state.get_current_node(), T, node_embeddings)
        current_node = self._gather_node(node_embeddings, current_node_idx)

        B, _, _ = node_embeddings.shape
        dtype = node_embeddings.dtype
        device = node_embeddings.device
        n_cus = int(state.states["cus_loc"].size(1))
        n_rs = int(state.states["rs_loc"].size(1))
        depot_mask = current_node_idx == 0
        customer_mask = (current_node_idx >= 1) & (current_node_idx < 1 + n_cus)
        rs_mask = current_node_idx >= 1 + n_cus

        current_load = self._as_step_scalar(state.states.get("current_load"), T, node_embeddings)
        current_battery = self._as_step_scalar(state.states.get("current_battery"), T, node_embeddings)
        remaining_battery = self._as_step_scalar(state.states.get("remaining_battery"), T, node_embeddings)
        current_time = self._as_step_scalar(state.states.get("current_time"), T, node_embeddings)
        visited_ratio = self._as_step_scalar(state.states.get("visited_customers_ratio"), T, node_embeddings)
        remain_feasible = self._as_step_scalar(state.states.get("remain_feasible_customers_ratio"), T, node_embeddings)
        route_served = self._as_step_scalar(state.states.get("route_served_customers_ratio"), T, node_embeddings)
        route_steps = self._as_step_scalar(state.states.get("current_route_step_count"), T, node_embeddings)
        route_customers = self._as_step_scalar(state.states.get("current_route_customer_count"), T, node_embeddings)

        current_type = torch.cat(
            [
                depot_mask.to(dtype).unsqueeze(-1),
                customer_mask.to(dtype).unsqueeze(-1),
                rs_mask.to(dtype).unsqueeze(-1),
            ],
            dim=-1,
        ).to(device=device)
        features = torch.cat(
            [
                current_load,
                current_battery,
                remaining_battery,
                current_time,
                visited_ratio,
                remain_feasible,
                route_served,
                route_steps,
                route_customers,
                current_type,
            ],
            dim=-1,
        )
        state_context = self.state_proj(features)
        return self.query_proj(torch.cat([graph_context, current_node, state_context], dim=-1))


class DynamicGraphKVEncoder(nn.Module):
    """
    Candidate-side dynamic graph encoder for the decoder.

    Static node embeddings provide the base graph memory. This module is the only
    dynamic graph path: it reads current remaining-graph/candidate state and
    produces dynamic corrections for attention keys, values, action keys, and a
    scalar action bias.
    """

    def __init__(
        self,
        embedding_dim,
        n_heads=4,
        enabled=False,
        enable_delta_k=True,
        enable_delta_v=True,
        enable_delta_action_key=True,
        enable_action_bias=True,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.enabled = bool(enabled)
        self.enable_delta_k = bool(enable_delta_k)
        self.enable_delta_v = bool(enable_delta_v)
        self.enable_delta_action_key = bool(enable_delta_action_key)
        self.enable_action_bias = bool(enable_action_bias)
        self.routing_system_feature_dim = 10
        self.problem_system_feature_dim = 5
        self.system_feature_dim = (
            self.routing_system_feature_dim + self.problem_system_feature_dim
        )
        self.routing_candidate_feature_dim = 16
        self.problem_candidate_feature_dim = 14
        self.candidate_feature_dim = (
            self.routing_candidate_feature_dim + self.problem_candidate_feature_dim
        )
        self.num_tokens = 9

        n_heads = max(1, int(n_heads))
        if self.embedding_dim % n_heads != 0:
            n_heads = 1

        self.state_proj = nn.Sequential(
            nn.LayerNorm(self.system_feature_dim),
            nn.Linear(self.system_feature_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.token_type = nn.Parameter(torch.zeros(1, self.num_tokens, embedding_dim))
        self.token_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        self.token_norm = nn.LayerNorm(embedding_dim)
        self.token_ff = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.SiLU(),
            nn.Linear(2 * embedding_dim, embedding_dim),
        )
        self.token_ff_norm = nn.LayerNorm(embedding_dim)
        self.route_pos_proj = nn.Sequential(
            nn.LayerNorm(4 * embedding_dim),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.node_state_proj = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.decision_state_proj = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.step_state_proj = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.candidate_feature_proj = nn.Sequential(
            nn.LayerNorm(self.candidate_feature_dim),
            nn.Linear(self.candidate_feature_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, 3 * embedding_dim),
        )
        self.candidate_delta_base = nn.Sequential(
            nn.LayerNorm(self.candidate_feature_dim),
            nn.Linear(self.candidate_feature_dim, embedding_dim),
            nn.SiLU(),
        )
        self.candidate_key_delta_proj = nn.Linear(embedding_dim, embedding_dim)
        self.candidate_value_delta_proj = nn.Linear(embedding_dim, embedding_dim)
        self.candidate_action_key_delta_proj = nn.Linear(embedding_dim, embedding_dim)
        self.action_bias_proj = nn.Sequential(
            nn.LayerNorm(self.candidate_feature_dim),
            nn.Linear(self.candidate_feature_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, 1),
        )
        self.key_scale = nn.Parameter(torch.tensor(0.1))
        self.value_scale = nn.Parameter(torch.tensor(0.1))
        self.action_key_scale = nn.Parameter(torch.tensor(0.1))
        self.action_bias_scale = nn.Parameter(torch.tensor(0.1))

        nn.init.normal_(self.token_type, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.state_proj[1].weight, gain=0.5)
        nn.init.zeros_(self.state_proj[1].bias)
        nn.init.xavier_uniform_(self.state_proj[3].weight, gain=0.5)
        nn.init.zeros_(self.state_proj[3].bias)
        nn.init.xavier_uniform_(self.route_pos_proj[1].weight, gain=0.5)
        nn.init.zeros_(self.route_pos_proj[1].bias)
        nn.init.zeros_(self.route_pos_proj[3].weight)
        nn.init.zeros_(self.route_pos_proj[3].bias)
        nn.init.zeros_(self.node_state_proj.weight)
        nn.init.zeros_(self.decision_state_proj.weight)
        nn.init.zeros_(self.step_state_proj.weight)
        nn.init.xavier_uniform_(self.candidate_feature_proj[1].weight, gain=0.5)
        nn.init.zeros_(self.candidate_feature_proj[1].bias)
        nn.init.zeros_(self.candidate_feature_proj[3].weight)
        nn.init.zeros_(self.candidate_feature_proj[3].bias)
        nn.init.xavier_uniform_(self.candidate_delta_base[1].weight, gain=0.5)
        nn.init.zeros_(self.candidate_delta_base[1].bias)
        nn.init.zeros_(self.candidate_key_delta_proj.weight)
        nn.init.zeros_(self.candidate_key_delta_proj.bias)
        nn.init.zeros_(self.candidate_value_delta_proj.weight)
        nn.init.zeros_(self.candidate_value_delta_proj.bias)
        nn.init.zeros_(self.candidate_action_key_delta_proj.weight)
        nn.init.zeros_(self.candidate_action_key_delta_proj.bias)
        nn.init.xavier_uniform_(self.action_bias_proj[1].weight, gain=0.5)
        nn.init.zeros_(self.action_bias_proj[1].bias)
        nn.init.zeros_(self.action_bias_proj[3].weight)
        nn.init.zeros_(self.action_bias_proj[3].bias)

    @staticmethod
    def _step_count(state, fallback=1):
        action_mask = state.states.get("action_mask", None)
        if torch.is_tensor(action_mask) and action_mask.dim() >= 3:
            return int(action_mask.size(1))
        current = state.get_current_node()
        if current.dim() >= 2:
            return int(current.size(1))
        return int(fallback)

    @staticmethod
    def _expand_step(x, T):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.size(1) == 1 and T != 1:
            x = x.expand(-1, T, -1)
        return x

    @staticmethod
    def _expand_mask(mask, T):
        if mask.dim() == 2:
            mask = mask.unsqueeze(1)
        if mask.size(1) == 1 and T != 1:
            mask = mask.expand(-1, T, -1)
        return mask

    @staticmethod
    def _as_step_scalar(x, T, like):
        B = like.size(0)
        if x is None:
            return like.new_zeros(B, T, 1)
        x = x.to(device=like.device, dtype=like.dtype)
        if x.dim() == 0:
            x = x.view(1, 1, 1)
        elif x.dim() == 1:
            x = x[:, None, None]
        elif x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() == 3 and x.size(-1) != 1:
            x = x[..., :1]
        if x.size(0) == 1 and B != 1:
            x = x.expand(B, -1, -1)
        if x.size(1) == 1 and T != 1:
            x = x.expand(-1, T, -1)
        return x

    @staticmethod
    def _as_step_index(node_idx, T, node_embeddings):
        if node_idx.dim() == 1:
            node_idx = node_idx.unsqueeze(1)
        node_idx = node_idx.to(device=node_embeddings.device, dtype=torch.long)
        if node_idx.size(1) == 1 and T != 1:
            node_idx = node_idx.expand(-1, T)
        return node_idx.clamp(min=0, max=node_embeddings.size(1) - 1)

    @staticmethod
    def _masked_mean(node_embeddings, mask, weights=None):
        mask = mask.to(device=node_embeddings.device, dtype=node_embeddings.dtype)
        if weights is not None:
            weights = weights.to(device=node_embeddings.device, dtype=node_embeddings.dtype)
            mask = mask * weights.clamp_min(0.0)
        denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.matmul(mask, node_embeddings) / denom

    @staticmethod
    def _signed_weighted_mean(node_embeddings, mask, weights):
        mask = mask.to(device=node_embeddings.device, dtype=node_embeddings.dtype)
        weights = weights.to(device=node_embeddings.device, dtype=node_embeddings.dtype)
        denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.matmul(mask * weights, node_embeddings) / denom

    def _route_position_summary(self, node_embeddings, route_mask, route_order):
        order = route_order.to(device=node_embeddings.device, dtype=node_embeddings.dtype).clamp(0.0, 1.0)
        angle = order * (2.0 * torch.pi)
        features = torch.cat(
            [
                self._signed_weighted_mean(node_embeddings, route_mask, torch.sin(angle)),
                self._signed_weighted_mean(node_embeddings, route_mask, torch.cos(angle)),
                self._signed_weighted_mean(node_embeddings, route_mask, torch.sin(2.0 * angle)),
                self._signed_weighted_mean(node_embeddings, route_mask, torch.cos(2.0 * angle)),
            ],
            dim=-1,
        )
        return self.route_pos_proj(features)

    @staticmethod
    def _gather_node(node_embeddings, node_idx):
        if node_idx.dim() == 1:
            node_idx = node_idx.unsqueeze(1)
        node_idx = node_idx.to(device=node_embeddings.device, dtype=torch.long)
        node_idx = node_idx.clamp(min=0, max=node_embeddings.size(1) - 1)
        gather_idx = node_idx.unsqueeze(-1).expand(-1, -1, node_embeddings.size(-1))
        return torch.gather(node_embeddings, dim=1, index=gather_idx)

    @staticmethod
    def _edge_matrix(state, name: str, *, B: int, device, dtype):
        edge = state.states.get(name, None)
        if edge is None:
            return None
        edge = edge.to(device=device, dtype=dtype)
        if edge.dim() == 2:
            edge = edge.unsqueeze(0)
        if edge.size(0) == 1 and B != 1:
            edge = edge.expand(B, -1, -1)
        return edge

    @staticmethod
    def _gather_edge_from_current(edge_matrix, current_node_idx):
        B, T = current_node_idx.shape
        batch_idx = torch.arange(B, device=edge_matrix.device).view(B, 1).expand(B, T)
        return edge_matrix[batch_idx, current_node_idx, :]

    @staticmethod
    def _gather_edge_to_depot(edge_matrix, current_node_idx):
        B, T = current_node_idx.shape
        batch_idx = torch.arange(B, device=edge_matrix.device).view(B, 1).expand(B, T)
        return edge_matrix[batch_idx, current_node_idx, 0].unsqueeze(-1)

    def _node_type_masks(self, state, node_embeddings, T):
        B, N, _ = node_embeddings.shape
        n_cus = int(state.states["cus_loc"].size(1))
        n_rs = int(state.states["rs_loc"].size(1))
        customer = torch.zeros(B, 1, N, dtype=torch.bool, device=node_embeddings.device)
        rs = torch.zeros_like(customer)
        customer[:, :, 1 : 1 + n_cus] = True
        if n_rs > 0:
            rs[:, :, 1 + n_cus : 1 + n_cus + n_rs] = True
        depot = torch.zeros_like(customer)
        depot[:, :, 0] = True
        if T != 1:
            customer = customer.expand(-1, T, -1)
            rs = rs.expand(-1, T, -1)
            depot = depot.expand(-1, T, -1)
        return depot, customer, rs

    def _system_features(
        self,
        state,
        node_embeddings,
        T,
        action_mask,
        depot_mask,
        customer_mask,
        rs_mask,
        route_mask,
        current_node_idx,
    ):
        dtype = node_embeddings.dtype
        feasible_customer_ratio = (
            (action_mask & customer_mask).to(dtype).sum(-1, keepdim=True)
            / customer_mask.to(dtype).sum(-1, keepdim=True).clamp_min(1.0)
        )
        feasible_rs_ratio = (
            (action_mask & rs_mask).to(dtype).sum(-1, keepdim=True)
            / rs_mask.to(dtype).sum(-1, keepdim=True).clamp_min(1.0)
        )
        depot_feasible = (action_mask & depot_mask).to(dtype).sum(-1, keepdim=True).clamp(max=1.0)
        current_is_depot = torch.gather(
            depot_mask.to(dtype),
            dim=2,
            index=current_node_idx.unsqueeze(-1),
        )
        current_is_customer = torch.gather(
            customer_mask.to(dtype),
            dim=2,
            index=current_node_idx.unsqueeze(-1),
        )
        route_len_ratio = (
            route_mask.to(dtype).sum(dim=-1, keepdim=True)
            / float(max(node_embeddings.size(1), 1))
        )
        route_is_empty = (route_len_ratio <= 1e-6).to(dtype)

        routing_core = torch.cat(
            [
                self._as_step_scalar(
                    state.states.get("visited_customers_ratio"), T, node_embeddings
                ),
                self._as_step_scalar(
                    state.states.get("remain_feasible_customers_ratio"),
                    T,
                    node_embeddings,
                ),
                self._as_step_scalar(
                    state.states.get("route_served_customers_ratio"), T, node_embeddings
                ),
                depot_feasible,
                feasible_customer_ratio,
                feasible_rs_ratio,
                current_is_depot,
                current_is_customer,
                route_len_ratio,
                route_is_empty,
            ],
            dim=-1,
        )
        current_is_rs = torch.gather(
            rs_mask.to(dtype),
            dim=2,
            index=current_node_idx.unsqueeze(-1),
        )
        constraint_context = torch.cat(
            [
                self._as_step_scalar(state.states.get("current_load"), T, node_embeddings),
                self._as_step_scalar(state.states.get("current_battery"), T, node_embeddings),
                self._as_step_scalar(state.states.get("current_time"), T, node_embeddings),
                current_is_rs,
                self._as_step_scalar(state.states.get("rs_streak_ratio"), T, node_embeddings),
            ],
            dim=-1,
        )
        return torch.cat([routing_core, constraint_context], dim=-1)

    def _candidate_features(
        self,
        state,
        node_embeddings,
        T,
        action_mask,
        depot_mask,
        customer_mask,
        rs_mask,
        route_mask,
        visit_count,
        route_order,
        current_node_idx,
        prev_node_idx,
    ):
        depot_loc = state.states["depot_loc"]
        if depot_loc.dim() == 2:
            depot_loc = depot_loc.unsqueeze(1)

        cus_loc = state.states["cus_loc"]
        rs_loc = state.states["rs_loc"]
        node_loc = torch.cat([depot_loc, cus_loc, rs_loc], dim=1)
        B, N, _ = node_loc.shape
        device = node_embeddings.device
        dtype = node_embeddings.dtype

        current_loc = torch.gather(
            node_loc.to(device=device, dtype=dtype),
            dim=1,
            index=current_node_idx.unsqueeze(-1).expand(-1, -1, node_loc.size(-1)),
        )
        rel = node_loc.to(device=device, dtype=dtype).unsqueeze(1) - current_loc.unsqueeze(2)
        coord_travel_proxy = torch.linalg.norm(rel, dim=-1).clamp(min=0.0) / (2.0 ** 0.5)
        depot_step_loc = node_loc[:, :1, :].to(device=device, dtype=dtype)
        coord_return_to_depot = torch.linalg.norm(
            node_loc.to(device=device, dtype=dtype).unsqueeze(1)
            - depot_step_loc.unsqueeze(2),
            dim=-1,
        ).clamp(min=0.0) / (2.0 ** 0.5)
        if coord_return_to_depot.size(1) == 1 and T != 1:
            coord_return_to_depot = coord_return_to_depot.expand(-1, T, -1)
        coord_current_to_depot = torch.linalg.norm(
            current_loc - depot_step_loc,
            dim=-1,
            keepdim=True,
        ).clamp(min=0.0) / (2.0 ** 0.5)
        travel_proxy = coord_travel_proxy
        return_to_depot = coord_return_to_depot
        current_to_depot = coord_current_to_depot

        edge_distance = self._edge_matrix(state, "edge_distance", B=B, device=device, dtype=dtype)
        if edge_distance is not None:
            travel_proxy = self._gather_edge_from_current(edge_distance, current_node_idx)
            return_to_depot = edge_distance[:, None, :, 0].expand(-1, T, -1)
            current_to_depot = self._gather_edge_to_depot(edge_distance, current_node_idx)
        depot_detour = travel_proxy + return_to_depot - current_to_depot

        edge_energy = self._edge_matrix(state, "edge_energy", B=B, device=device, dtype=dtype)
        if edge_energy is None:
            energy_cost = travel_proxy
        else:
            energy_cost = self._gather_edge_from_current(edge_energy, current_node_idx)
        edge_time = self._edge_matrix(state, "edge_time", B=B, device=device, dtype=dtype)
        if edge_time is None:
            travel_time = coord_travel_proxy
        else:
            travel_time = self._gather_edge_from_current(edge_time, current_node_idx)

        time_window = state.states["time_window"].to(device=device, dtype=dtype)
        service_time = state.states["service_time"].to(device=device, dtype=dtype)
        if service_time.dim() == 3:
            service_time = service_time.squeeze(-1)
        demand = state.states["demand"].to(device=device, dtype=dtype)
        if demand.dim() == 3:
            demand = demand.squeeze(-1)

        tw_open = time_window[..., 0].unsqueeze(1)
        tw_close = time_window[..., 1].unsqueeze(1)
        service = service_time.unsqueeze(1).expand(B, T, N)
        demand_step = demand.unsqueeze(1).expand(B, T, N)

        current_time = self._as_step_scalar(state.current_time.float(), T, node_embeddings)
        current_load = self._as_step_scalar(state.used_capacity.float(), T, node_embeddings)
        current_battery = self._as_step_scalar(state.used_battery.float(), T, node_embeddings)

        arrival = current_time + travel_time
        wait = torch.relu(tw_open - arrival)
        service_start = torch.maximum(arrival, tw_open)
        finish = service_start + service
        arrival_slack = tw_close - arrival
        finish_slack = tw_close - finish
        load_after = current_load + demand_step
        battery_after = current_battery + energy_cost
        capacity_margin = (1.0 - load_after).clamp(-1.0, 1.0)

        battery_capacity = state.states.get("battery_capacity", None)
        if battery_capacity is None:
            battery_capacity = torch.ones(B, 1, 1, device=device, dtype=dtype)
        else:
            battery_capacity = battery_capacity.to(device=device, dtype=dtype)
            if battery_capacity.dim() == 0:
                battery_capacity = battery_capacity.view(1, 1, 1)
            elif battery_capacity.dim() == 1:
                battery_capacity = battery_capacity.view(-1, 1, 1)
            else:
                battery_capacity = battery_capacity.reshape(battery_capacity.size(0), -1)
                battery_capacity = battery_capacity[:, :1].view(-1, 1, 1)
            if battery_capacity.size(0) == 1 and B != 1:
                battery_capacity = battery_capacity.expand(B, -1, -1)
        battery_capacity = battery_capacity.clamp_min(1e-6)
        energy_ratio = (energy_cost / battery_capacity).clamp(0.0, 2.0)
        current_battery_feasible = (battery_after <= battery_capacity).to(dtype)
        battery_margin = (1.0 - battery_after).clamp(-1.0, 1.0)

        node_ids = torch.arange(N, device=device).view(1, 1, N)
        current_candidate = node_ids == current_node_idx.unsqueeze(-1)
        prev_candidate = node_ids == prev_node_idx.unsqueeze(-1)
        visit_norm = visit_count.clamp(0.0, 5.0) / 5.0
        route_order_clamped = route_order.clamp(0.0, 1.0)
        unvisited_customer = customer_mask & (visit_count <= 0)
        route_served_ratio = self._as_step_scalar(
            state.states.get("route_served_customers_ratio"),
            T,
            node_embeddings,
        ).clamp(0.0, 1.0)
        no_customer_route = (route_served_ratio <= 1e-6).to(dtype)
        rs_streak_ratio = self._as_step_scalar(
            state.states.get("rs_streak_ratio"),
            T,
            node_embeddings,
        ).clamp(0.0, 1.0)
        repeated_rs_candidate = (visit_norm > 0).to(dtype) * rs_mask.to(dtype)

        routing_core = torch.stack(
            [
                action_mask.to(dtype),
                (~action_mask).to(dtype),
                depot_mask.to(dtype),
                customer_mask.to(dtype),
                rs_mask.to(dtype),
                unvisited_customer.to(dtype),
                route_mask.to(dtype),
                visit_norm,
                route_order_clamped,
                current_candidate.to(dtype),
                prev_candidate.to(dtype),
                (prev_candidate & rs_mask).to(dtype),
                travel_proxy.clamp(0.0, 2.0),
                return_to_depot.clamp(0.0, 2.0),
                depot_detour.clamp(-1.0, 2.0),
                (action_mask & unvisited_customer).to(dtype),
            ],
            dim=-1,
        )
        constraint_supplement = torch.stack(
            [
                demand_step,
                load_after.clamp(0.0, 2.0),
                capacity_margin,
                energy_ratio,
                arrival.clamp(0.0, 2.0),
                wait.clamp(0.0, 1.0),
                arrival_slack.clamp(-1.0, 1.0),
                finish_slack.clamp(-1.0, 1.0),
                battery_after.clamp(0.0, 2.0),
                battery_margin,
                current_battery_feasible,
                route_served_ratio.expand(B, T, N),
                no_customer_route.expand(B, T, N),
                rs_streak_ratio.expand(B, T, N) * repeated_rs_candidate,
            ],
            dim=-1,
        )
        return torch.cat([routing_core, constraint_supplement], dim=-1)

    def forward(
        self,
        node_embeddings,
        graph_context,
        driver_query,
        state,
    ):
        if not self.enabled:
            return 0, 0, 0, 0

        T = self._step_count(state, fallback=driver_query.size(1))
        graph_context = self._expand_step(graph_context, T)
        driver_query = self._expand_step(driver_query, T)

        depot_mask, customer_mask, rs_mask = self._node_type_masks(
            state, node_embeddings, T
        )
        action_mask = self._expand_mask(state.states["action_mask"].to(torch.bool), T)

        node_visit_count = state.states.get("node_visit_count", None)
        if node_visit_count is None:
            visit_count = action_mask.new_zeros(action_mask.shape, dtype=node_embeddings.dtype)
        else:
            visit_count = node_visit_count.to(
                device=node_embeddings.device,
                dtype=node_embeddings.dtype,
            )
            visit_count = self._expand_mask(visit_count, T)

        route_order = state.states.get("route_order_rank", None)
        if route_order is None:
            route_order = (visit_count > 0).to(dtype=node_embeddings.dtype)
        else:
            route_order = route_order.to(
                device=node_embeddings.device,
                dtype=node_embeddings.dtype,
            )
            route_order = self._expand_mask(route_order, T)
        route_mask = route_order > 0

        current_node_idx = self._as_step_index(
            state.get_current_node(), T, node_embeddings
        )
        prev_node_idx = self._as_step_index(
            state.states.get("prev_node_idx", state.get_current_node()),
            T,
            node_embeddings,
        )
        current_node = self._gather_node(node_embeddings, current_node_idx)

        route_summary = self._masked_mean(
            node_embeddings,
            route_mask,
            weights=1.0 + route_order,
        )
        route_summary = route_summary + self._route_position_summary(
            node_embeddings,
            route_mask,
            route_order,
        )
        feasible_customer = action_mask & customer_mask
        feasible_rs = action_mask & rs_mask
        unvisited_customer = customer_mask & (visit_count <= 0)
        feasible_customer_summary = self._masked_mean(node_embeddings, feasible_customer)
        feasible_rs_summary = self._masked_mean(node_embeddings, feasible_rs)
        unvisited_customer_summary = self._masked_mean(node_embeddings, unvisited_customer)
        depot_summary = self._masked_mean(node_embeddings, depot_mask)

        system_features = self._system_features(
            state=state,
            node_embeddings=node_embeddings,
            T=T,
            action_mask=action_mask,
            depot_mask=depot_mask,
            customer_mask=customer_mask,
            rs_mask=rs_mask,
            route_mask=route_mask,
            current_node_idx=current_node_idx,
        )
        state_token = self.state_proj(system_features)

        tokens = torch.stack(
            [
                driver_query,
                state_token,
                current_node,
                route_summary,
                feasible_customer_summary,
                feasible_rs_summary,
                unvisited_customer_summary,
                depot_summary,
                graph_context,
            ],
            dim=2,
        )
        B, _, S, D = tokens.shape
        flat_tokens = tokens.reshape(B * T, S, D)
        flat_tokens = flat_tokens + self.token_type[:, :S, :].to(
            device=flat_tokens.device,
            dtype=flat_tokens.dtype,
        )
        attended_tokens, _ = self.token_attn(
            flat_tokens,
            flat_tokens,
            flat_tokens,
            need_weights=False,
        )
        flat_tokens = self.token_norm(flat_tokens + attended_tokens)
        flat_tokens = self.token_ff_norm(flat_tokens + self.token_ff(flat_tokens))
        decision_token = flat_tokens[:, 0, :].reshape(B, T, D)

        candidate_features = self._candidate_features(
            state=state,
            node_embeddings=node_embeddings,
            T=T,
            action_mask=action_mask,
            depot_mask=depot_mask,
            customer_mask=customer_mask,
            rs_mask=rs_mask,
            route_mask=route_mask,
            visit_count=visit_count,
            route_order=route_order,
            current_node_idx=current_node_idx,
            prev_node_idx=prev_node_idx,
        )
        key_delta = 0
        value_delta = 0
        action_key_delta = 0
        if self.enable_delta_k or self.enable_delta_v or self.enable_delta_action_key:
            candidate_base = self.candidate_delta_base(candidate_features)
            node_key, node_value, node_action_key = self.node_state_proj(node_embeddings).chunk(3, dim=-1)
            decision_key, decision_value, decision_action_key = self.decision_state_proj(decision_token).chunk(3, dim=-1)
            step_key, step_value, step_action_key = self.step_state_proj(state_token).chunk(3, dim=-1)
            if self.enable_delta_k:
                key_delta = self.candidate_key_delta_proj(candidate_base)
                key_delta = key_delta + node_key.unsqueeze(1)
                key_delta = key_delta + decision_key.unsqueeze(2)
                key_delta = key_delta + step_key.unsqueeze(2)
                key_delta = torch.tanh(self.key_scale) * key_delta
            if self.enable_delta_v:
                value_delta = self.candidate_value_delta_proj(candidate_base)
                value_delta = value_delta + node_value.unsqueeze(1)
                value_delta = value_delta + decision_value.unsqueeze(2)
                value_delta = value_delta + step_value.unsqueeze(2)
                value_delta = torch.tanh(self.value_scale) * value_delta
            if self.enable_delta_action_key:
                action_key_delta = self.candidate_action_key_delta_proj(candidate_base)
                action_key_delta = action_key_delta + node_action_key.unsqueeze(1)
                action_key_delta = action_key_delta + decision_action_key.unsqueeze(2)
                action_key_delta = action_key_delta + step_action_key.unsqueeze(2)
                action_key_delta = torch.tanh(self.action_key_scale) * action_key_delta
        if self.enable_action_bias:
            action_bias = self.action_bias_proj(candidate_features).squeeze(-1)
            action_bias = torch.tanh(self.action_bias_scale) * action_bias
        else:
            action_bias = 0
        return key_delta, value_delta, action_key_delta, action_bias


class Decoder(nn.Module):
    """
    Pointer-style decoder.

    Assumption:
        Encoder output MUST include graph token at index 0.

    Encoder output:
        embeddings: [B, N+1, D]
            embeddings[:, 0, :]   -> graph token
            embeddings[:, 1:, :]  -> real nodes

    Real node order:
        [depot, customers, RS]
    """

    def __init__(
        self,
        embedding_dim,
        step_context_dim,
        n_heads,
        problem,
        tanh_clipping,
        use_dynamic_decision_encoder=False,
        dynamic_decision_heads=4,
        dynamic_decision_delta_k=True,
        dynamic_decision_delta_v=True,
        dynamic_decision_delta_action_key=True,
        dynamic_decision_action_bias=True,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.problem = problem

        # project node embeddings -> (attention K, attention V, action key)
        self.project_node_embeddings = nn.Linear(
            embedding_dim, 3 * embedding_dim, bias=False
        )

        # graph token -> fixed graph context
        self.project_fixed_context = nn.Linear(
            embedding_dim, embedding_dim, bias=False
        )

        del step_context_dim

        # driver-side query and candidate-side dynamic graph encoder
        self.driver_query_encoder = DriverQueryEncoder(embedding_dim)
        self.dynamic_graph_kv_encoder = DynamicGraphKVEncoder(
            embedding_dim=embedding_dim,
            n_heads=dynamic_decision_heads,
            enabled=use_dynamic_decision_encoder,
            enable_delta_k=dynamic_decision_delta_k,
            enable_delta_v=dynamic_decision_delta_v,
            enable_delta_action_key=dynamic_decision_delta_action_key,
            enable_action_bias=dynamic_decision_action_bias,
        )

        # glimpse + pointer
        self.glimpse = MultiHeadAttention(
            embedding_dim=embedding_dim,
            n_heads=n_heads,
        )
        self.pointer = AttentionScore(
            use_tanh=True,
            C=tanh_clipping,
            learn_scale=True,
            learn_C=False,
        )
        self.action_query_proj = nn.Sequential(
            nn.LayerNorm(2 * embedding_dim),
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.decode_type = None

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def set_decode_type(self, decode_type):
        assert decode_type in ["greedy", "sampling"]
        self.decode_type = decode_type

    # ------------------------------------------------------------------
    # Precompute
    # ------------------------------------------------------------------

    def _precompute(self, embeddings, mask=None):
        """
        embeddings: [B, N+1, D], MUST include graph token at index 0
        mask: [B, N] or None, mask over REAL nodes only
        """
        if embeddings.dim() != 3:
            raise ValueError(f"Expected embeddings to be [B, N+1, D], got {embeddings.shape}")

        if embeddings.size(1) < 2:
            raise ValueError(
                f"Embeddings must include graph token + at least one real node, got {embeddings.shape}"
            )

        graph_embed = embeddings[:, 0, :]     # [B,D]
        node_embed = embeddings[:, 1:, :]     # [B,N,D]

        if mask is not None and mask.size(1) != node_embed.size(1):
            raise ValueError(
                f"Mask shape {mask.shape} incompatible with real node embeddings {node_embed.shape}"
            )

        graph_context = self.project_fixed_context(graph_embed).unsqueeze(1)  # [B,1,D]

        glimpse_key, glimpse_val, action_key = self.project_node_embeddings(node_embed).chunk(
            3, dim=-1
        )

        cache = (node_embed, graph_context, glimpse_key, glimpse_val, action_key)
        return cache

    # ------------------------------------------------------------------
    # One-step decoding
    # ------------------------------------------------------------------

    def advance(self, cached_embeddings, state, node_mask=None):
        """
        cached_embeddings: output of _precompute()
        state: StateWrapper
        node_mask: [B,N] optional extra mask over real nodes
        """
        node_embeddings, graph_context, glimpse_K, glimpse_V, action_key = cached_embeddings

        query = self.driver_query_encoder(node_embeddings, graph_context, state)
        key_delta, val_delta, action_key_delta, action_bias = self.dynamic_graph_kv_encoder(
            node_embeddings=node_embeddings,
            graph_context=graph_context,
            driver_query=query,
            state=state,
        )

        tensor_deltas = [
            delta
            for delta in (key_delta, val_delta, action_key_delta)
            if torch.is_tensor(delta)
        ]
        stepwise_t = 1
        for delta in tensor_deltas:
            if delta.dim() == 4:
                stepwise_t = max(stepwise_t, int(delta.size(1)))
        if stepwise_t > 1:
            glimpse_K = glimpse_K.unsqueeze(1).expand(-1, stepwise_t, -1, -1)
            glimpse_V = glimpse_V.unsqueeze(1).expand(-1, stepwise_t, -1, -1)
            action_key = action_key.unsqueeze(1).expand(-1, stepwise_t, -1, -1)

            def _align_stepwise(delta):
                if not torch.is_tensor(delta):
                    return None
                if delta.dim() == 3:
                    return delta.unsqueeze(1).expand(-1, stepwise_t, -1, -1)
                return delta

            aligned_key_delta = _align_stepwise(key_delta)
            aligned_val_delta = _align_stepwise(val_delta)
            aligned_action_key_delta = _align_stepwise(action_key_delta)
            if aligned_key_delta is not None:
                glimpse_K = glimpse_K + aligned_key_delta
            if aligned_val_delta is not None:
                glimpse_V = glimpse_V + aligned_val_delta
            if aligned_action_key_delta is not None:
                action_key = action_key + aligned_action_key_delta
        elif tensor_deltas:
            if torch.is_tensor(key_delta):
                glimpse_K = glimpse_K + key_delta.squeeze(1) if key_delta.dim() == 4 else glimpse_K + key_delta
            if torch.is_tensor(val_delta):
                glimpse_V = glimpse_V + val_delta.squeeze(1) if val_delta.dim() == 4 else glimpse_V + val_delta
            if torch.is_tensor(action_key_delta):
                action_key = action_key + action_key_delta.squeeze(1) if action_key_delta.dim() == 4 else action_key + action_key_delta

        # base feasibility mask from env, over real nodes only
        mask = state.get_mask()   # [B,N]

        if node_mask is not None:
            # optional extra mask from outside (e.g., padded nodes)
            node_mask = node_mask.to(mask.device)
            mask = mask | node_mask

        logits, glimpse = self.calc_logits(
            query=query,
            glimpse_K=glimpse_K,
            glimpse_V=glimpse_V,
            action_key=action_key,
            mask=mask,
            action_bias=action_bias,
        )
        return logits, glimpse

    def calc_logits(self, query, glimpse_K, glimpse_V, action_key, mask, action_bias=0):
        """
        query: [B,T,D] or [B,1,D]
        glimpse_K/V/action_key: [B,N,D] or [B,T,N,D]
        mask: [B,N] or [B,T,N]
        action_bias: scalar or [B,T,N]/[B,N]
        """
        if glimpse_K.dim() == 4:
            B, T, N, D = glimpse_K.shape
            query_flat = query.reshape(B * T, 1, D)
            glimpse_K_flat = glimpse_K.reshape(B * T, N, D)
            glimpse_V_flat = glimpse_V.reshape(B * T, N, D)
            action_key_flat = action_key.reshape(B * T, N, D)
            mask_flat = mask.reshape(B * T, N)

            glimpse_flat = self.glimpse(
                query_flat,
                glimpse_K_flat,
                glimpse_V_flat,
                mask_flat,
            )
            action_query_flat = self.action_query_proj(torch.cat([query_flat, glimpse_flat], dim=-1))
            logits_flat = self.pointer(action_query_flat, action_key_flat, mask_flat)
            if torch.is_tensor(action_bias):
                bias_flat = action_bias.reshape(B * T, N).unsqueeze(1).to(logits_flat.dtype)
                logits_flat = logits_flat + bias_flat
            return logits_flat.reshape(B, T, N), glimpse_flat.reshape(B, T, D)

        glimpse = self.glimpse(query, glimpse_K, glimpse_V, mask)  # [B,1,D]
        action_query = self.action_query_proj(torch.cat([query, glimpse], dim=-1))
        logits = self.pointer(action_query, action_key, mask)      # [B,1,N]
        if torch.is_tensor(action_bias):
            if action_bias.dim() == 2:
                action_bias = action_bias.unsqueeze(1)
            logits = logits + action_bias.to(device=logits.device, dtype=logits.dtype)
        return logits, glimpse

    # ------------------------------------------------------------------
    # Decode strategy
    # ------------------------------------------------------------------

    def decode(self, probs, mask):
        """
        probs: [B,N]
        mask: [B,N], True = infeasible
        """
        assert (probs == probs).all(), "Probs should not contain NaNs"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(-1)).data.any(), \
                "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print("Sampled bad values, resampling!")
                selected = probs.multinomial(1).squeeze(1)
        else:
            raise ValueError(f"Unknown decode type: {self.decode_type}")

        return selected
