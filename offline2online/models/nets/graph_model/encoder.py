import torch
from torch import nn
from ...nets.graph_model.multi_head_attention import MultiHeadAttentionProj

class GraphBiasBuilder(nn.Module):
    def __init__(self, num_node_types: int = 3):
        super().__init__()
        # 0: depot, 1: RS, 2: customer
        self.type_pair_bias = nn.Embedding(num_node_types * num_node_types, 1)
        nn.init.zeros_(self.type_pair_bias.weight)

        self.dist_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, dist_mat, node_type):
        """
        dist_mat:  [B, N, N]
        node_type: [B, N]   (0 depot, 1 RS, 2 customer)

        return:
            attn_bias: [B, N, N]
        """
        # type-pair ids
        type_i = node_type.unsqueeze(2)   # [B,N,1]
        type_j = node_type.unsqueeze(1)   # [B,1,N]
        pair_id = type_i * 3 + type_j     # [B,N,N]

        type_bias = self.type_pair_bias(pair_id).squeeze(-1)  # [B,N,N]

        # simple distance bias: closer = larger
        dist_bias = -self.dist_scale * dist_mat

        attn_bias = type_bias + dist_bias
        return attn_bias

class SwiGLUFFN(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.value_proj = nn.Linear(embed_dim, hidden_dim)
        self.gate_proj = nn.Linear(embed_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        value = self.value_proj(x)
        gate = torch.nn.functional.silu(self.gate_proj(x))
        return self.out_proj(value * gate)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(
        self,
        n_heads: int,
        embedding_dim: int,
        feed_forward_hidden: int = 512,
    ):
        super().__init__()

        self.attn = MultiHeadAttentionProj(
            embedding_dim=embedding_dim,
            n_heads=n_heads,
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.ff = SwiGLUFFN(
            embed_dim=embedding_dim,
            hidden_dim=feed_forward_hidden,
        )

    def forward(self, x, attn_bias=None):
        # Attention block (Pre-LN)
        h = self.norm1(x)
        h = self.attn(h, mask=None, attn_bias=attn_bias)
        x = x + h

        # FFN block (Pre-LN)
        h = self.norm2(x)
        h = self.ff(h)
        x = x + h

        return x


class GraphAttentionEncoder(nn.Module):
    """
    v1 graph encoder:
    - graph token is always used
    - wrapper passes raw node embeddings [B, N, D]
    - output becomes [B, N+1, D], with graph token at index 0
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        n_layers: int,
        feed_forward_hidden: int = 512,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.graph_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.graph_token)

        self.layers = nn.ModuleList(
            [
                MultiHeadAttentionLayer(
                    n_heads=n_heads,
                    embedding_dim=embed_dim,
                    feed_forward_hidden=feed_forward_hidden,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(embed_dim)

    def _prepend_graph_token(self, x, attn_bias=None):
        """
        x: [B, N, D]
        attn_bias: [B, N, N] or [B, 1, N, N] or None
        """
        B, _, D = x.shape
        graph_token = self.graph_token.expand(B, 1, D)   # [B,1,D]
        x = torch.cat([graph_token, x], dim=1)           # [B,N+1,D]

        if attn_bias is not None:
            if attn_bias.dim() == 3:
                # [B, N, N] -> [B, N+1, N+1]
                B2, N, _ = attn_bias.shape
                new_bias = torch.zeros(
                    B2, N + 1, N + 1,
                    dtype=attn_bias.dtype,
                    device=attn_bias.device
                )
                new_bias[:, 1:, 1:] = attn_bias
                attn_bias = new_bias

            elif attn_bias.dim() == 4:
                # [B, H, N, N] or [B, 1, N, N] -> [B, H, N+1, N+1]
                B2, H, N, _ = attn_bias.shape
                new_bias = torch.zeros(
                    B2, H, N + 1, N + 1,
                    dtype=attn_bias.dtype,
                    device=attn_bias.device
                )
                new_bias[:, :, 1:, 1:] = attn_bias
                attn_bias = new_bias

            else:
                raise ValueError(f"Unsupported attn_bias shape: {attn_bias.shape}")

        return x, attn_bias

    def forward(self, x, mask=None, attn_bias=None):
        """
        x: [B, N, D]
        mask: ignored by design in current graph-token-only setup
        attn_bias: [B, N, N] or [B, H, N, N]
        """
        x, attn_bias = self._prepend_graph_token(x, attn_bias=attn_bias)
        for layer in self.layers:
            x = layer(x, attn_bias=attn_bias)

        x = self.final_norm(x)   # [B, N+1, D]
        return x
        
    @staticmethod
    def _mean_without_graph_token(x):
        """
        x: [B, N+1, D], graph token at index 0
        """
        return x[:, 1:, :].mean(dim=1)
