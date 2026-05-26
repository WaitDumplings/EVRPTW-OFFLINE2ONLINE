import torch
import torch.nn as nn

def AutoEmbedding(problem_name, config):
    """
    Automatically select the corresponding module according to ``problem_name``
    """
    mapping = {
        "evrptw": EVRPTWEmbedding,
    }
    embeddingClass = mapping[problem_name]
    embedding = embeddingClass(**config)
    return embedding

# Embedding Layer for EVRPTW
class EVRPTWEmbedding(nn.Module):
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = None):
        super().__init__()
        self.embed_dim = embedding_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else embedding_dim

        # raw feature dim = 2(x,y) + 1(demand) + 2(tw) + 1(service_time) = 6
        in_dim = 6

        # Type-specific projections
        self.depot_proj = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, embedding_dim),
        )

        self.customer_proj = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, embedding_dim),
        )

        self.rs_proj = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, embedding_dim),
        )

        # 0: depot, 1: RS, 2: customer
        self.type_embed = nn.Embedding(3, embedding_dim)
        self.post_fusion = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.norm = nn.LayerNorm(embedding_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for module in [self.depot_proj, self.customer_proj, self.rs_proj]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        nn.init.zeros_(self.type_embed.weight)

    def _ensure_depot_shape(self, depot_loc: torch.Tensor) -> torch.Tensor:
        # Accept either [B,2] or [B,1,2]
        if depot_loc.dim() == 2:
            depot_loc = depot_loc.unsqueeze(1)
        return depot_loc

    def forward(self, x):
        """
        Build node embeddings in fixed order: [depot, customers, RS].
        """
        depot_loc = self._ensure_depot_shape(x["depot_loc"])   # [B,1,2]
        cus_loc = x["cus_loc"]                                 # [B,n_cus,2]
        rs_loc = x["rs_loc"]                                   # [B,n_rs,2]

        B = depot_loc.size(0)
        device = depot_loc.device
        n_cus = cus_loc.size(1)
        n_rs = rs_loc.size(1)

        demand = x["demand"]                                   # [B,N,1]
        time_window = x["time_window"]                         # [B,N,2]
        service_time = x["service_time"]                       # [B,N,1]

        # ----- depot -----
        depot_demand = demand[:, :1, :]
        depot_tw = time_window[:, :1, :]
        depot_service = service_time[:, :1, :]
        depot_feat = torch.cat(
            [depot_loc, depot_demand, depot_tw, depot_service], dim=-1
        )  # [B,1,6]

        # ----- customers -----
        cus_demand = demand[:, 1:1 + n_cus, :]
        cus_tw = time_window[:, 1:1 + n_cus, :]
        cus_service = service_time[:, 1:1 + n_cus, :]
        cus_feat = torch.cat(
            [cus_loc, cus_demand, cus_tw, cus_service], dim=-1
        )  # [B,n_cus,6]

        # ----- RS -----
        rs_demand = demand[:, 1 + n_cus:, :]
        rs_tw = time_window[:, 1 + n_cus:, :]
        rs_service = service_time[:, 1 + n_cus:, :]
        rs_feat = torch.cat(
            [rs_loc, rs_demand, rs_tw, rs_service], dim=-1
        )  # [B,n_rs,6]

        # Type-specific projections
        depot_emb = self.depot_proj(depot_feat)       # [B,1,D]
        cus_emb = self.customer_proj(cus_feat)        # [B,n_cus,D]
        rs_emb = self.rs_proj(rs_feat)                # [B,n_rs,D]

        node_emb = torch.cat([depot_emb, cus_emb, rs_emb], dim=1)  # [B,N,D]

        # type embedding
        depot_type = torch.zeros(B, 1, dtype=torch.long, device=device)          # 0
        cus_type = torch.full((B, n_cus), 2, dtype=torch.long, device=device)    # 2
        rs_type = torch.ones(B, n_rs, dtype=torch.long, device=device)            # 1
        node_type = torch.cat([depot_type, cus_type, rs_type], dim=1)            # [B,N]

        type_emb = self.type_embed(node_type)                                     # [B,N,D]

        node_emb = node_emb + type_emb
        node_emb = node_emb + self.post_fusion(node_emb)
        node_emb = self.norm(node_emb)

        return node_emb