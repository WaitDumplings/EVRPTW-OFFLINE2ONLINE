"""
Problem specific decoder context for EVRPTW.

Expected shapes:
    embeddings: [B, N, D]
        B = batch size
        N = number of real nodes = 1 + n_cus + n_rs
        D = embedding dim

    state.get_current_node(): [B, T] or [B]
        T = number of parallel trajectories per instance

    state.get_mask(): [B, T, N]
        True = infeasible
"""

import torch
from torch import nn


def AutoContext(problem_name, config):
    """
    Automatically select the corresponding context module according to problem_name.
    """
    mapping = {
        "evrptw": VRPContext,
    }
    context_class = mapping[problem_name]
    context = context_class(**config)
    return context


def _gather_by_index(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Gather node embeddings by index.

    Args:
        source: [B, N, D]
        index:  [B] or [B, T]

    Returns:
        gathered: [B, T, D]
            If input index is [B], it is treated as [B, 1].
    """
    if index.dim() == 1:
        index = index.unsqueeze(1)  # [B,1]

    B, N, D = source.shape
    T = index.size(1)

    # Expand source to [B, T, N, D]
    source_expand = source.unsqueeze(1).expand(-1, T, -1, -1)

    # Expand index to [B, T, 1, D]
    index_expand = index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, D)

    gathered = torch.gather(source_expand, dim=2, index=index_expand).squeeze(2)  # [B,T,D]
    return gathered


class PrevNodeContext(nn.Module):
    """
    Base class for decoder context.

    By default the output is:
        [prev_node_embedding, state_embedding]

    Shapes:
        prev_node_embedding: [B, T, D]
        state_embedding:     [B, T, S]
        output:              [B, T, D+S]
    """

    def __init__(self, context_dim: int):
        super().__init__()
        self.context_dim = context_dim

    def _prev_node_embedding(self, embeddings: torch.Tensor, state) -> torch.Tensor:
        """
        embeddings: [B, N, D]
        current_node: [B, T] or [B]
        return: [B, T, D]
        """
        current_node = state.get_current_node()
        prev_node_embedding = _gather_by_index(embeddings, current_node)
        return prev_node_embedding

    def _state_embedding(self, embeddings: torch.Tensor, state) -> torch.Tensor:
        raise NotImplementedError("Please implement _state_embedding for your problem.")

    def forward(self, embeddings: torch.Tensor, state) -> torch.Tensor:
        """
        embeddings: [B, N, D]
        return: [B, T, context_dim]
        """
        prev_node_embedding = self._prev_node_embedding(embeddings, state)  # [B,T,D]
        state_embedding = self._state_embedding(embeddings, state)          # [B,T,S]
        context_embedding = torch.cat((prev_node_embedding, state_embedding), dim=-1)
        return context_embedding


class VRPContext(PrevNodeContext):
    """
    Context for EVRPTW.

    Context structure:
        [ prev_node_embedding ,
          feasible_customer_summary ,
          scalar_state ]

    Where:
        prev_node_embedding:      [B, T, D]
        feasible_customer_summary:[B, T, D]
        scalar_state:             [B, T, 5]

    So total output dim is:
        2 * D + 5
    """

    def __init__(self, context_dim: int):
        super().__init__(context_dim)

    @staticmethod
    def _to_step_scalar(x: torch.Tensor) -> torch.Tensor:
        """
        Convert scalar-like state tensor to [B, T, 1].

        Accepts:
            [B]
            [B, T]
            [B, T, 1]

        Returns:
            [B, T, 1]
        """
        if x.dim() == 1:
            x = x.unsqueeze(1)      # [B,1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)     # [B,T,1]
        return x

    @staticmethod
    def _masked_mean_bt(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Masked mean pooling over customer dimension.

        Args:
            x: [B, C, D]
                Customer embeddings.
            valid_mask: [B, T, C]
                True = valid / included in pooling.

        Returns:
            pooled: [B, T, D]
        """
        # [B,1,C,D]
        x_expand = x.unsqueeze(1)

        # [B,T,C,1]
        valid_mask_f = valid_mask.unsqueeze(-1).type_as(x)

        # [B,T,D]
        x_sum = (x_expand * valid_mask_f).sum(dim=2)

        # [B,T,1]
        denom = valid_mask_f.sum(dim=2).clamp(min=1e-6)

        pooled = x_sum / denom
        return pooled

    def _feasible_customer_summary(self, embeddings: torch.Tensor, state) -> torch.Tensor:
        """
        Build current feasible-customer summary.

        Args:
            embeddings: [B, N, D], node order = [depot, customers, RS]
            state.get_mask(): [B, T, N], True = infeasible

        Returns:
            feasible_summary: [B, T, D]
        """
        n_nodes = embeddings.size(1)

        infeasible_mask = state.get_mask()  # [B, T, N]
        if infeasible_mask.size(-1) != n_nodes:
            raise ValueError(
                f"State mask shape {infeasible_mask.shape} incompatible with embeddings {embeddings.shape}"
            )

        # Number of customers from observation
        n_cus = state.states["cus_loc"].size(1)

        # Customer embeddings only: [B, C, D]
        cus_embeddings = embeddings[:, 1:1 + n_cus, :]

        # Customer infeasible mask: [B, T, C]
        cus_infeasible = infeasible_mask[:, :, 1:1 + n_cus]

        # Feasible customer mask: True = valid
        cus_feasible = ~cus_infeasible

        # [B, T, D]
        feasible_summary = self._masked_mean_bt(cus_embeddings, cus_feasible)
        return feasible_summary

    def _state_embedding(self, embeddings: torch.Tensor, state) -> torch.Tensor:
        """
        Build dynamic state embedding.

        Returns:
            [B, T, D + 5]
            = feasible_customer_summary [B,T,D]
            + scalar_state              [B,T,5]
        """
        feasible_customer_summary = self._feasible_customer_summary(embeddings, state)  # [B,T,D]

        state_embedding_capacity = self._to_step_scalar(state.used_capacity)                         # [B,T,1]
        state_embedding_battery = self._to_step_scalar(state.used_battery)                           # [B,T,1]
        state_embedding_time = self._to_step_scalar(state.current_time)                              # [B,T,1]
        state_visited_customers_ratio = self._to_step_scalar(state.visited_customers_ratio)         # [B,T,1]
        state_remain_feasible_customers_ratio = self._to_step_scalar(
            state.remain_feasible_customers_ratio
        )                                                                                            # [B,T,1]

        scalar_state = torch.cat(
            (
                state_embedding_capacity,
                state_embedding_battery,
                state_embedding_time,
                state_visited_customers_ratio,
                state_remain_feasible_customers_ratio,
            ),
            dim=-1,
        )  # [B,T,5]

        state_embedding = torch.cat(
            (
                feasible_customer_summary,   # [B,T,D]
                scalar_state,                # [B,T,5]
            ),
            dim=-1,
        )  # [B,T,D+5]

        return state_embedding