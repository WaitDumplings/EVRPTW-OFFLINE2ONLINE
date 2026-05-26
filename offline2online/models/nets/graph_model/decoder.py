import torch
from torch import nn

from ...nets.graph_model.context import AutoContext
from ...nets.graph_model.dynamic_embedding import (
    AutoDynamicEmbedding,
    AutoDynamicContextEmbedding,
)
from ...nets.graph_model.multi_head_attention import (
    AttentionScore,
    MultiHeadAttention,
)


class StructuredGraphStateFusion(nn.Module):
    """
    Fuse the static graph embedding with current remaining/route graph summaries
    and scalar vehicle/system state before pointer attention.

    This is intentionally local to the decoder: the encoder still builds the
    static graph representation once, while the decoder receives lightweight
    state-conditioned summaries for the remaining graph and current route.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.system_feature_dim = 12
        self.candidate_feature_dim = 14

        self.system_proj = nn.Sequential(
            nn.LayerNorm(self.system_feature_dim),
            nn.Linear(self.system_feature_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.route_order_embed = nn.Embedding(33, embedding_dim)
        self.route_visit_embed = nn.Embedding(8, embedding_dim)
        self.route_event_norm = nn.LayerNorm(embedding_dim)
        self.route_event_gru = nn.GRU(
            embedding_dim,
            embedding_dim,
            num_layers=1,
            batch_first=True,
        )
        self.route_event_seq_norm = nn.LayerNorm(embedding_dim)
        self.query_fuse = nn.Sequential(
            nn.LayerNorm(8 * embedding_dim),
            nn.Linear(8 * embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.candidate_state_proj = nn.Sequential(
            nn.LayerNorm(self.candidate_feature_dim),
            nn.Linear(self.candidate_feature_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, 3 * embedding_dim),
        )
        self.fusion_scale = nn.Parameter(torch.tensor(0.1))
        self.candidate_scale = nn.Parameter(torch.tensor(0.1))

        nn.init.xavier_uniform_(self.system_proj[1].weight, gain=0.5)
        nn.init.zeros_(self.system_proj[1].bias)
        nn.init.xavier_uniform_(self.system_proj[3].weight, gain=0.5)
        nn.init.zeros_(self.system_proj[3].bias)
        nn.init.normal_(self.route_order_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.route_visit_embed.weight, mean=0.0, std=0.02)
        for name, param in self.route_event_gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param, gain=0.5)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.query_fuse[1].weight, gain=0.5)
        nn.init.zeros_(self.query_fuse[1].bias)
        nn.init.xavier_uniform_(self.query_fuse[3].weight, gain=0.05)
        nn.init.zeros_(self.query_fuse[3].bias)
        nn.init.xavier_uniform_(self.candidate_state_proj[1].weight, gain=0.5)
        nn.init.zeros_(self.candidate_state_proj[1].bias)
        nn.init.xavier_uniform_(self.candidate_state_proj[3].weight, gain=0.05)
        nn.init.zeros_(self.candidate_state_proj[3].bias)

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
    def _expand_mask(mask, T):
        if mask.dim() == 2:
            mask = mask.unsqueeze(1)
        if mask.size(1) == 1 and T != 1:
            mask = mask.expand(-1, T, -1)
        return mask

    @staticmethod
    def _masked_mean(node_embeddings, mask, weights=None):
        mask = mask.to(device=node_embeddings.device, dtype=node_embeddings.dtype)
        if weights is not None:
            weights = weights.to(device=node_embeddings.device, dtype=node_embeddings.dtype)
            mask = mask * weights.clamp_min(0.0)
        denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.matmul(mask, node_embeddings) / denom

    @staticmethod
    def _gather_node(node_embeddings, node_idx):
        if node_idx.dim() == 1:
            node_idx = node_idx.unsqueeze(1)
        node_idx = node_idx.to(device=node_embeddings.device, dtype=torch.long)
        node_idx = node_idx.clamp(min=0, max=node_embeddings.size(1) - 1)
        gather_idx = node_idx.unsqueeze(-1).expand(-1, -1, node_embeddings.size(-1))
        return torch.gather(node_embeddings, dim=1, index=gather_idx)

    def _route_event_summary(self, node_embeddings, query, state, T):
        route_nodes = state.states.get("route_event_nodes", None)
        if route_nodes is None:
            return node_embeddings.new_zeros(node_embeddings.size(0), T, node_embeddings.size(-1))

        route_nodes = route_nodes.to(device=node_embeddings.device, dtype=torch.long)
        route_nodes = self._expand_mask(route_nodes, T)
        route_nodes = route_nodes.clamp(min=0, max=node_embeddings.size(1) - 1)

        route_mask = state.states.get("route_event_mask", None)
        if route_mask is None:
            route_mask = route_nodes.new_ones(route_nodes.shape, dtype=torch.bool)
        else:
            route_mask = route_mask.to(device=node_embeddings.device, dtype=torch.bool)
            route_mask = self._expand_mask(route_mask, T)

        order_rank = state.states.get("route_event_order_rank", None)
        if order_rank is None:
            order_rank = route_nodes.new_zeros(route_nodes.shape, dtype=node_embeddings.dtype)
        else:
            order_rank = order_rank.to(device=node_embeddings.device, dtype=node_embeddings.dtype)
            order_rank = self._expand_mask(order_rank, T).clamp(0.0, 1.0)

        visit_count = state.states.get("route_event_visit_count", None)
        if visit_count is None:
            visit_count = route_nodes.new_zeros(route_nodes.shape)
        else:
            visit_count = visit_count.to(device=node_embeddings.device, dtype=torch.long)
            visit_count = self._expand_mask(visit_count, T)

        B, _, D = node_embeddings.shape
        L = route_nodes.size(-1)
        expanded_nodes = node_embeddings.unsqueeze(1).expand(-1, T, -1, -1)
        gather_idx = route_nodes.unsqueeze(-1).expand(B, T, L, D)
        event_node_emb = torch.gather(expanded_nodes, dim=2, index=gather_idx)

        order_bucket = torch.round(order_rank * 32.0).to(torch.long).clamp(0, 32)
        visit_bucket = visit_count.clamp(0, 7)
        event_repr = self.route_event_norm(
            event_node_emb
            + self.route_order_embed(order_bucket)
            + self.route_visit_embed(visit_bucket)
        )

        flat_event = event_repr.reshape(B * T, L, D)
        flat_mask = route_mask.reshape(B * T, L, 1).to(dtype=flat_event.dtype)
        flat_event = flat_event * flat_mask
        flat_encoded, _ = self.route_event_gru(flat_event)
        event_repr = self.route_event_seq_norm(flat_encoded + flat_event).reshape(B, T, L, D)

        scores = (event_repr * query.unsqueeze(2)).sum(dim=-1) / (float(D) ** 0.5)
        scores = scores.masked_fill(~route_mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * route_mask.to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.matmul(weights.unsqueeze(2), event_repr).squeeze(2)

    @staticmethod
    def _as_step_index(node_idx, T, node_embeddings):
        if node_idx.dim() == 1:
            node_idx = node_idx.unsqueeze(1)
        node_idx = node_idx.to(device=node_embeddings.device, dtype=torch.long)
        if node_idx.size(1) == 1 and T != 1:
            node_idx = node_idx.expand(-1, T)
        return node_idx

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

    def forward(
        self,
        node_embeddings,
        graph_context,
        step_context,
        dynamic_context,
        state,
    ):
        T = self._step_count(state, fallback=step_context.size(1))
        graph_context = self._expand_step(graph_context, T)
        step_context = self._expand_step(step_context, T)
        dynamic_context = self._expand_step(dynamic_context, T)

        depot_mask, customer_mask, rs_mask = self._node_type_masks(
            state, node_embeddings, T
        )

        action_mask = state.states["action_mask"].to(torch.bool)
        action_mask = self._expand_mask(action_mask, T)
        node_visit_count = state.states.get("node_visit_count", None)
        if node_visit_count is None:
            visit_count = action_mask.new_zeros(action_mask.shape, dtype=torch.float32)
        else:
            visit_count = node_visit_count.to(
                device=node_embeddings.device, dtype=node_embeddings.dtype
            )
            visit_count = self._expand_mask(visit_count, T)

        route_order = state.states.get("route_order_rank", None)
        if route_order is None:
            route_order = (visit_count > 0).to(dtype=node_embeddings.dtype)
        else:
            route_order = route_order.to(
                device=node_embeddings.device, dtype=node_embeddings.dtype
            )
            route_order = self._expand_mask(route_order, T)
        route_mask = route_order > 0
        route_weights = 1.0 + route_order
        route_node_summary = self._masked_mean(
            node_embeddings,
            route_mask,
            weights=route_weights,
        )

        unvisited_customer = customer_mask & (visit_count <= 0)
        available_rs = action_mask & rs_mask
        current_node_idx = self._as_step_index(
            state.get_current_node(), T, node_embeddings
        )
        current_node = self._gather_node(node_embeddings, current_node_idx)
        prev_node_idx = state.states.get("prev_node_idx", state.get_current_node())
        prev_node_idx = self._as_step_index(prev_node_idx, T, node_embeddings)
        prev_node = self._gather_node(node_embeddings, prev_node_idx)

        feasible_customer_ratio = (
            (action_mask & customer_mask).to(node_embeddings.dtype).sum(-1, keepdim=True)
            / customer_mask.to(node_embeddings.dtype).sum(-1, keepdim=True).clamp_min(1.0)
        )
        feasible_rs_ratio = (
            (action_mask & rs_mask).to(node_embeddings.dtype).sum(-1, keepdim=True)
            / rs_mask.to(node_embeddings.dtype).sum(-1, keepdim=True).clamp_min(1.0)
        )
        depot_feasible = (action_mask & depot_mask).to(node_embeddings.dtype).sum(
            -1, keepdim=True
        ).clamp(max=1.0)
        current_is_rs = torch.gather(
            rs_mask.to(node_embeddings.dtype),
            dim=2,
            index=current_node_idx.unsqueeze(-1).clamp(
                min=0, max=node_embeddings.size(1) - 1
            ),
        )
        route_len_ratio = (
            route_mask.to(node_embeddings.dtype).sum(dim=-1, keepdim=True)
            / float(max(node_embeddings.size(1), 1))
        )

        like = node_embeddings
        system_features = torch.cat(
            [
                self._as_step_scalar(state.states.get("current_load"), T, like),
                self._as_step_scalar(state.states.get("current_battery"), T, like),
                self._as_step_scalar(state.states.get("current_time"), T, like),
                self._as_step_scalar(state.states.get("visited_customers_ratio"), T, like),
                self._as_step_scalar(
                    state.states.get("remain_feasible_customers_ratio"), T, like
                ),
                self._as_step_scalar(
                    state.states.get("route_served_customers_ratio"), T, like
                ),
                self._as_step_scalar(state.states.get("rs_streak_ratio"), T, like),
                depot_feasible,
                feasible_customer_ratio,
                feasible_rs_ratio,
                current_is_rs,
                route_len_ratio,
            ],
            dim=-1,
        )
        system_context = self.system_proj(system_features)
        base_query = graph_context + step_context + dynamic_context
        route_event_summary = self._route_event_summary(
            node_embeddings,
            base_query,
            state,
            T,
        )

        query_input = torch.cat(
            [
                graph_context,
                step_context,
                dynamic_context,
                route_node_summary,
                route_event_summary,
                current_node,
                prev_node,
                system_context,
            ],
            dim=-1,
        )
        query_delta = torch.tanh(self.fusion_scale) * self.query_fuse(query_input)

        B, N, _ = node_embeddings.shape
        node_ids = torch.arange(N, device=node_embeddings.device).view(1, 1, N)
        current_candidate = node_ids == current_node_idx.unsqueeze(-1)
        prev_candidate = node_ids == prev_node_idx.unsqueeze(-1)
        visit_norm = visit_count.clamp(0.0, 5.0) / 5.0
        route_order_clamped = route_order.clamp(0.0, 1.0)
        action_float = action_mask.to(node_embeddings.dtype)
        candidate_features = torch.stack(
            [
                depot_mask.to(node_embeddings.dtype),
                customer_mask.to(node_embeddings.dtype),
                rs_mask.to(node_embeddings.dtype),
                action_float,
                (~action_mask).to(node_embeddings.dtype),
                unvisited_customer.to(node_embeddings.dtype),
                available_rs.to(node_embeddings.dtype),
                visit_norm,
                route_order_clamped,
                route_mask.to(node_embeddings.dtype),
                current_candidate.to(node_embeddings.dtype),
                prev_candidate.to(node_embeddings.dtype),
                (action_mask & depot_mask).to(node_embeddings.dtype),
                ((visit_count > 0) & rs_mask).to(node_embeddings.dtype),
            ],
            dim=-1,
        )
        candidate_delta = torch.tanh(self.candidate_scale) * self.candidate_state_proj(
            candidate_features
        )
        key_delta, value_delta, logit_delta = candidate_delta.chunk(3, dim=-1)
        return query_delta, key_delta, value_delta, logit_delta


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
        use_candidate_dynamic_embedding=True,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.problem = problem

        # project node embeddings -> (glimpse_K, glimpse_V, logit_K)
        self.project_node_embeddings = nn.Linear(
            embedding_dim, 3 * embedding_dim, bias=False
        )

        # graph token -> fixed graph context
        self.project_fixed_context = nn.Linear(
            embedding_dim, embedding_dim, bias=False
        )

        # dynamic step context
        self.project_step_context = nn.Sequential(
            nn.LayerNorm(step_context_dim),
            nn.Linear(step_context_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # state-dependent context builders
        self.context = AutoContext(
            problem.NAME,
            {"context_dim": step_context_dim},
        )
        self.dynamic_embedding = AutoDynamicEmbedding(
            problem.NAME,
            {
                "embedding_dim": embedding_dim,
                "use_candidate_dynamic_embedding": use_candidate_dynamic_embedding,
            },
        )
        self.dynamic_context_embedding = AutoDynamicContextEmbedding(
            problem.NAME,
            {"embedding_dim": embedding_dim},
        )
        self.structured_state_fusion = StructuredGraphStateFusion(embedding_dim)

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

        glimpse_key, glimpse_val, logit_key = self.project_node_embeddings(node_embed).chunk(
            3, dim=-1
        )

        cache = (node_embed, graph_context, glimpse_key, glimpse_val, logit_key)
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
        node_embeddings, graph_context, glimpse_K, glimpse_V, logit_K = cached_embeddings

        # current step context
        context = self.context(node_embeddings, state)       # [B,1,step_context_dim]
        step_context = self.project_step_context(context)    # [B,1,D]

        dynamic_context = self.dynamic_context_embedding(state)
        (
            structured_query_delta,
            structured_key_delta,
            structured_val_delta,
            structured_logit_delta,
        ) = self.structured_state_fusion(
            node_embeddings=node_embeddings,
            graph_context=graph_context,
            step_context=step_context,
            dynamic_context=dynamic_context,
            state=state,
        )
        query = graph_context + step_context + dynamic_context + structured_query_delta

        # dynamic node-wise modifiers
        glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic = self.dynamic_embedding(state)
        if torch.is_tensor(glimpse_key_dynamic) and glimpse_key_dynamic.dim() == 4:
            glimpse_K = (
                glimpse_K.unsqueeze(1)
                + glimpse_key_dynamic
                + structured_key_delta
            )
            glimpse_V = (
                glimpse_V.unsqueeze(1)
                + glimpse_val_dynamic
                + structured_val_delta
            )
            logit_K = (
                logit_K.unsqueeze(1)
                + logit_key_dynamic
                + structured_logit_delta
            )
        else:
            structured_key_delta = structured_key_delta.squeeze(1)
            structured_val_delta = structured_val_delta.squeeze(1)
            structured_logit_delta = structured_logit_delta.squeeze(1)
            glimpse_K = glimpse_K + glimpse_key_dynamic + structured_key_delta
            glimpse_V = glimpse_V + glimpse_val_dynamic + structured_val_delta
            logit_K = logit_K + logit_key_dynamic + structured_logit_delta

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
            logit_K=logit_K,
            mask=mask,
        )
        return logits, glimpse

    def calc_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        """
        query: [B,1,D]
        glimpse_K/V/logit_K: [B,N,D]
        mask: [B,N]
        """
        if glimpse_K.dim() == 4:
            B, T, N, D = glimpse_K.shape
            query_flat = query.reshape(B * T, 1, D)
            glimpse_K_flat = glimpse_K.reshape(B * T, N, D)
            glimpse_V_flat = glimpse_V.reshape(B * T, N, D)
            logit_K_flat = logit_K.reshape(B * T, N, D)
            mask_flat = mask.reshape(B * T, N)

            glimpse_flat = self.glimpse(
                query_flat,
                glimpse_K_flat,
                glimpse_V_flat,
                mask_flat,
            )
            logits_flat = self.pointer(glimpse_flat, logit_K_flat, mask_flat)
            return logits_flat.reshape(B, T, N), glimpse_flat.reshape(B, T, D)

        glimpse = self.glimpse(query, glimpse_K, glimpse_V, mask)  # [B,1,D]
        logits = self.pointer(glimpse, logit_K, mask)              # [B,1,N]
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
