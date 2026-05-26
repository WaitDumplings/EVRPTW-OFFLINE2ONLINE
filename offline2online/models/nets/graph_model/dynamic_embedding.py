"""
Problem specific node embedding for dynamic feature.
"""

import torch
import torch.nn as nn

def AutoDynamicContextEmbedding(problem_name, config):
    """
    Automatically select the corresponding module according to ``problem_name``
    """
    mapping = {
        "evrptw": EVRPTWDynamicContextEmbedding,
    }
    embeddingClass = mapping[problem_name]
    embedding = embeddingClass(**config)
    return embedding

def AutoDynamicEmbedding(problem_name, config):
    """
    Automatically select the corresponding module according to ``problem_name``
    """
    if (
        problem_name == "evrptw"
        and bool(config.get("use_candidate_dynamic_embedding", True))
    ):
        return EVRPTWDynamicEmbedding(**config)

    mapping = {
        "tsp": NonDyanmicEmbedding,
        "evrptw": NonDyanmicEmbedding,
        "cvrp": NonDyanmicEmbedding,
        "sdvrp": SDVRPDynamicEmbedding,
        "pctsp": NonDyanmicEmbedding,
        "op": NonDyanmicEmbedding,
    }
    embeddingClass = mapping[problem_name]
    embedding = embeddingClass(**config)
    return embedding


class SDVRPDynamicEmbedding(nn.Module):
    """
    Embedding for dynamic node feature for the split delivery vehicle routing problem.

    It is implemented as a linear projection of the demands left in each node.

    Args:
        embedding_dim: dimension of output
    Inputs: state
        * **state** : a class that provide ``state.demands_with_depot`` tensor
    Outputs: glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic
        * **glimpse_key_dynamic** : [batch, graph_size, embedding_dim]
        * **glimpse_val_dynamic** : [batch, graph_size, embedding_dim]
        * **logit_key_dynamic** : [batch, graph_size, embedding_dim]

    """

    def __init__(self, embedding_dim):
        super(SDVRPDynamicEmbedding, self).__init__()
        self.projection = nn.Linear(1, 3 * embedding_dim, bias=False)

    def forward(self, state):
        glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic = self.projection(
            state.demands_with_depot[:, 0, :, None].clone()
        ).chunk(3, dim=-1)
        return glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic


class EVRPTWDynamicEmbedding(nn.Module):
    """
    Candidate-level dynamic embedding for EVRPTW.

    Static node embeddings already contain location, demand, time window and
    service time. This module injects state-conditioned candidate features into
    the decoder keys/logits, so each action can see its current arrival time,
    waiting time and time-window slack.
    """

    def __init__(self, embedding_dim, use_candidate_dynamic_embedding=True):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.enabled = bool(use_candidate_dynamic_embedding)
        self.feature_dim = 22

        self.projection = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, 3 * embedding_dim),
        )

        # Start as an exact no-op when continuing old experiments.
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    @staticmethod
    def _ensure_step_dim(x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        return x

    def forward(self, state):
        if not self.enabled:
            return 0, 0, 0

        depot_loc = state.states["depot_loc"]
        if depot_loc.dim() == 2:
            depot_loc = depot_loc.unsqueeze(1)

        cus_loc = state.states["cus_loc"]
        rs_loc = state.states["rs_loc"]
        node_loc = torch.cat([depot_loc, cus_loc, rs_loc], dim=1)
        B, N, _ = node_loc.shape
        device = node_loc.device

        action_mask = state.states["action_mask"].float()
        if action_mask.dim() == 2:
            action_mask = action_mask.unsqueeze(1)
        T = action_mask.size(1)

        current_node = state.get_current_node().long()
        if current_node.dim() == 1:
            current_node = current_node.unsqueeze(1)
        if current_node.size(1) != T:
            current_node = current_node.expand(-1, T)

        prev_node = state.states.get("prev_node_idx", current_node)
        prev_node = prev_node.long()
        if prev_node.dim() == 1:
            prev_node = prev_node.unsqueeze(1)
        if prev_node.size(1) != T:
            prev_node = prev_node.expand(-1, T)

        current_loc = torch.gather(
            node_loc,
            dim=1,
            index=current_node.unsqueeze(-1).expand(-1, -1, node_loc.size(-1)),
        )

        rel = node_loc.unsqueeze(1) - current_loc.unsqueeze(2)
        travel_proxy = torch.linalg.norm(rel, dim=-1).clamp(min=0.0)
        travel_proxy = travel_proxy / (2.0 ** 0.5)

        edge_energy = state.states.get("edge_energy", None)
        if edge_energy is None:
            energy_cost = travel_proxy
        else:
            edge_energy = edge_energy.to(device=device, dtype=travel_proxy.dtype)
            if edge_energy.dim() == 2:
                edge_energy = edge_energy.unsqueeze(0)
            if edge_energy.size(0) == 1 and B != 1:
                edge_energy = edge_energy.expand(B, -1, -1)
            edge_energy = edge_energy.unsqueeze(1).expand(-1, T, -1, -1)
            gather_idx = current_node.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, N)
            energy_cost = torch.gather(edge_energy, dim=2, index=gather_idx).squeeze(2)

        time_window = state.states["time_window"].float()
        service_time = state.states["service_time"].float()
        if service_time.dim() == 3:
            service_time = service_time.squeeze(-1)
        demand = state.states["demand"].float()
        if demand.dim() == 3:
            demand = demand.squeeze(-1)

        tw_open = time_window[..., 0].unsqueeze(1)
        tw_close = time_window[..., 1].unsqueeze(1)
        service = service_time.unsqueeze(1)
        demand = demand.unsqueeze(1)

        current_time = self._ensure_step_dim(state.current_time.float()).unsqueeze(-1)
        current_load = self._ensure_step_dim(state.used_capacity.float()).unsqueeze(-1)
        current_battery = self._ensure_step_dim(state.used_battery.float()).unsqueeze(-1)

        arrival = current_time + travel_proxy
        wait = torch.relu(tw_open - arrival)
        service_start = torch.maximum(arrival, tw_open)
        finish = service_start + service
        arrival_slack = tw_close - arrival
        finish_slack = tw_close - finish

        load_after = current_load + demand
        battery_after = current_battery + energy_cost

        battery_capacity = state.states.get("battery_capacity", None)
        if battery_capacity is None:
            battery_capacity = torch.ones(B, 1, 1, device=device, dtype=travel_proxy.dtype)
        else:
            battery_capacity = battery_capacity.to(device=device, dtype=travel_proxy.dtype)
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
        full_battery_feasible = (energy_cost <= battery_capacity).float()
        current_battery_feasible = (battery_after <= battery_capacity).float()
        energy_ratio = (energy_cost / battery_capacity).clamp(0.0, 2.0)

        n_cus = cus_loc.size(1)
        n_rs = rs_loc.size(1)
        node_ids = torch.arange(N, device=device)
        is_depot = (node_ids == 0).float().view(1, 1, N)
        is_customer = ((node_ids >= 1) & (node_ids < 1 + n_cus)).float().view(1, 1, N)
        is_rs = (
            (node_ids >= 1 + n_cus)
            & (node_ids < 1 + n_cus + n_rs)
        ).float().view(1, 1, N)

        node_visit_count = state.states.get("node_visit_count", None)
        if node_visit_count is None:
            candidate_visit = torch.zeros(B, T, N, device=device, dtype=travel_proxy.dtype)
        else:
            candidate_visit = node_visit_count.float()
            if candidate_visit.dim() == 2:
                candidate_visit = candidate_visit.unsqueeze(1)
            if candidate_visit.size(1) != T:
                candidate_visit = candidate_visit.expand(-1, T, -1)
            candidate_visit = (candidate_visit.clamp(0.0, 5.0) / 5.0)

        candidate_is_prev = (
            node_ids.view(1, 1, N) == prev_node.unsqueeze(-1)
        ).float()
        current_is_rs = (current_node >= 1 + n_cus).float().unsqueeze(-1)
        immediate_backtrack = candidate_is_prev * current_is_rs

        route_served_ratio = state.states.get("route_served_customers_ratio", None)
        if route_served_ratio is None:
            route_served_ratio = torch.zeros(B, T, 1, device=device, dtype=travel_proxy.dtype)
        else:
            route_served_ratio = route_served_ratio.float()
            if route_served_ratio.dim() == 2:
                route_served_ratio = route_served_ratio.unsqueeze(-1)
            if route_served_ratio.size(1) != T:
                route_served_ratio = route_served_ratio.expand(-1, T, -1)
            route_served_ratio = route_served_ratio.clamp(0.0, 1.0)

        no_customer_route = (route_served_ratio <= 1e-6).float()

        rs_streak_ratio = state.states.get("rs_streak_ratio", None)
        if rs_streak_ratio is None:
            rs_streak_ratio = torch.zeros(B, T, 1, device=device, dtype=travel_proxy.dtype)
        else:
            rs_streak_ratio = rs_streak_ratio.float()
            if rs_streak_ratio.dim() == 2:
                rs_streak_ratio = rs_streak_ratio.unsqueeze(-1)
            if rs_streak_ratio.size(1) != T:
                rs_streak_ratio = rs_streak_ratio.expand(-1, T, -1)
            rs_streak_ratio = rs_streak_ratio.clamp(0.0, 1.0)

        repeated_rs_candidate = (candidate_visit > 0).float() * is_rs.expand(B, T, N)

        features = torch.stack(
            [
                travel_proxy,
                energy_cost.clamp(0.0, 2.0),
                energy_ratio,
                arrival.clamp(0.0, 2.0),
                wait.clamp(0.0, 1.0),
                arrival_slack.clamp(-1.0, 1.0),
                finish_slack.clamp(-1.0, 1.0),
                service.expand(B, T, N),
                demand.expand(B, T, N),
                load_after.clamp(0.0, 2.0),
                battery_after.clamp(0.0, 2.0),
                full_battery_feasible.expand(B, T, N),
                current_battery_feasible.expand(B, T, N),
                action_mask,
                is_customer.expand(B, T, N),
                (is_depot + is_rs).expand(B, T, N),
                candidate_visit,
                candidate_is_prev,
                immediate_backtrack,
                route_served_ratio.expand(B, T, N),
                no_customer_route.expand(B, T, N),
                rs_streak_ratio.expand(B, T, N) * repeated_rs_candidate,
            ],
            dim=-1,
        )

        out = self.projection(features)
        return out.chunk(3, dim=-1)


class EVRPTWDynamicContextEmbedding(nn.Module):
    """
    Context embedding for dynamic feature for the electric vehicle routing problem with time windows.

    It is implemented as a concatenation of normalized used capacity, used battery and current time.

    Args:
        context_dim: dimension of output
    Inputs: embeddings, state
    """
    def __init__(self, embedding_dim):
        super(EVRPTWDynamicContextEmbedding, self).__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(9),
            nn.Linear(9, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Keep the new path initially neutral for stable continuation runs.
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, state):
        action_mask = state.states["action_mask"].float()
        if action_mask.dim() == 2:
            action_mask = action_mask.unsqueeze(1)

        n_cus = state.states["cus_loc"].size(1)
        n_rs = state.states["rs_loc"].size(1)

        current_load = state.used_capacity.float().unsqueeze(-1)
        current_battery = state.used_battery.float().unsqueeze(-1)
        current_time = state.current_time.float().unsqueeze(-1)
        visited_ratio = state.visited_customers_ratio.float()
        remain_feasible_ratio = state.remain_feasible_customers_ratio.float()

        depot_feasible = action_mask[..., 0:1]
        customer_feasible_ratio = action_mask[..., 1:1 + n_cus].mean(
            dim=-1,
            keepdim=True,
        )

        if n_rs > 0:
            rs_feasible_ratio = action_mask[..., 1 + n_cus:1 + n_cus + n_rs].mean(
                dim=-1,
                keepdim=True,
            )
        else:
            rs_feasible_ratio = torch.zeros_like(depot_feasible)

        current_is_rs = (state.get_current_node() > n_cus).float().unsqueeze(-1)

        features = torch.cat(
            [
                current_load,
                current_battery,
                current_time,
                visited_ratio,
                remain_feasible_ratio,
                depot_feasible,
                customer_feasible_ratio,
                rs_feasible_ratio,
                current_is_rs,
            ],
            dim=-1,
        )

        return self.proj(features)



class NonDyanmicEmbedding(nn.Module):
    """
    Embedding for problems that do not have any dynamic node feature.

    It is implemented as simply returning zeros.

    Args:
        embedding_dim: dimension of output
    Inputs: state
        * **state** : not used, just for consistency
    Outputs: glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic
        * **glimpse_key_dynamic** : [batch, graph_size, embedding_dim]
        * **glimpse_val_dynamic** : [batch, graph_size, embedding_dim]
        * **logit_key_dynamic** : [batch, graph_size, embedding_dim]

    """

    def __init__(self, embedding_dim):
        super(NonDyanmicEmbedding, self).__init__()

    def forward(self, state):
        return 0, 0, 0
