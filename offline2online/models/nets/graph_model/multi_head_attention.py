import math

import torch
from torch import nn

################################ Decoder Attention ################################
class AttentionScore(nn.Module):
    def __init__(self, use_tanh=True, C=10.0, learn_scale=True, learn_C=False):
        super().__init__()
        self.use_tanh = use_tanh

        if learn_scale:
            self.scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("scale", torch.tensor(1.0))

        if learn_C:
            self._C = nn.Parameter(torch.tensor(math.log(C)))
        else:
            self.register_buffer("_C", torch.tensor(math.log(C)))

    @property
    def C(self):
        return torch.exp(self._C)

    def forward(self, query, key, mask=None):
        d_k = query.size(-1)
        u_score = torch.matmul(query, key.transpose(-2, -1)) * self.scale / math.sqrt(d_k)

        with torch.no_grad():
            self.last_sat = (torch.abs(torch.tanh(u_score)) > 0.98).float().mean()

        if self.use_tanh:
            logits = torch.tanh(u_score) * self.C
        else:
            logits = u_score

        if mask is not None:
            # mask: [B, K] or [B,1,K]
            if mask.dim() == 2:      # [B,K]
                mask_exp = mask.unsqueeze(1)   # [B,1,K]
            else:
                mask_exp = mask               # [B,1,K]
            mask_exp = mask_exp.to(logits.device)  # [B,1,K]
            logits = logits.masked_fill(mask_exp, float("-inf"))

        return logits

class MultiHeadAttention(nn.Module):
    r"""
    Compute the multi-head attention.

    .. math::
        q^\prime = \mathrm{MultiHeadAttention}(q,\pmb{k},\pmb{v},\mathrm{mask})

    The following is computed:

    .. math::
        \begin{aligned}
        \pmb{a}^{(j)} &= \mathrm{Softmax}(\mathrm{AttentionScore}(q^{(j)},\pmb{k}^{(j)}, \mathrm{mask}))\\
        h^{(j)} &= \sum\nolimits_i \pmb{a}^{(j)}_i\pmb{v}_i \\
        q^\prime &= W^O \left[h^{(1)},...,h^{(J)}\right]
        \end{aligned}

    Args:
        embedding_dim: dimension of the query, keys, values
        n_head: number of heads
    Inputs: query, keys, value, mask
        * **query** : [batch, n_querys, embedding_dim]
        * **keys**: [batch, n_keys, embedding_dim]
        * **value**: [batch, n_keys, embedding_dim]
        * **mask**: [batch, 1, n_keys] ``logits[batch,j]==-inf`` if ``mask[batch, 0, j]==True``
    Outputs: logits, out
        * **out**: [batch, 1, embedding_dim] The output of the multi-head attention
    """

    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.attentionScore = AttentionScore()
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, query, key, value, mask):
        query_heads = self._make_heads(query)
        key_heads = self._make_heads(key)
        value_heads = self._make_heads(value)

        # [n_heads, batch, 1, nkeys]
        compatibility = self.attentionScore(query_heads, key_heads, mask)

        # [n_heads, batch, 1, head_dim]
        # torch.Size([16, 1, 49, 16])
        out_heads = torch.matmul(torch.softmax(compatibility, dim=-1), value_heads)

        # from multihead [nhead, batch, 1, head_dim] -> [batch, 1, nhead* head_dim]
        out = self.project_out(self._unmake_heads(out_heads))
        return out

    def _make_heads(self, v):
        batch_size, nkeys, h_dim = v.shape
        #  [batch_size, ..., n_heads* head_dim] --> [n_heads, batch_size, ..., head_dim]
        out = v.reshape(batch_size, nkeys, self.n_heads, h_dim // self.n_heads).movedim(-2, 0)
        return out

    def _unmake_heads(self, v):
        #  [n_heads, batch_size, ..., head_dim] --> [batch_size, ..., n_heads* head_dim]
        out = v.movedim(0, -2).flatten(-2)
        return out

################################ Encoder Attention ################################

class Vanilla_AttentionScore(nn.Module):
    """
    Computes scaled dot-product attention scores.

    Args:
        query: [H, B, Q, D]
        key:   [H, B, K, D]
        mask:  [B, K] (bool, True = masked)
        attn_bias: Optional attention bias of shape:
            - [B, Q, K] (recommended)
            - [B, K, K] (for self-attention)
            - Will be broadcast to [H, B, Q, K]
    """
    def __init__(self):
        super().__init__()

    def forward(self, query, key, mask=None, attn_bias=None):
        d_k = query.size(-1)

        # Standard scaled dot-product attention score
        u_score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)   # [H,B,Q,K]

        # Optional external attention bias
        if attn_bias is not None:
            # attn_bias: [B,Q,K] → [1,B,Q,K] → broadcast to [H,B,Q,K]
            u_score = u_score + attn_bias.unsqueeze(0).to(u_score.device)

        # Apply mask (masked positions receive -inf)
        if mask is not None:
            # mask: [B,K] → [1,B,1,K]
            mask = mask.to(u_score.device).unsqueeze(0).unsqueeze(2)
            u_score = u_score.masked_fill(mask, float("-inf"))

        return u_score



class MultiHeadAttentionEncoder(nn.Module):
    """
    Multi-head attention layer operating on pre-projected queries, keys, and values.

    Args:
        query: [B, Nq, D]
        key:   [B, Nk, D]
        value: [B, Nk, D]
        mask:  [B, Nk] (bool)
        attn_bias: optional [B, Nq, Nk] (broadcast-safe)
    """
    def __init__(self, embedding_dim, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.attentionScore = Vanilla_AttentionScore()
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, query, key, value, mask=None, attn_bias=None):
        # Split into multiple heads
        query_heads = self._make_heads(query)   # [H,B,Nq,D_head]
        key_heads   = self._make_heads(key)     # [H,B,Nk,D_head]
        value_heads = self._make_heads(value)   # [H,B,Nk,D_head]

        # Compute attention scores
        compatibility = self.attentionScore(
            query_heads, key_heads,
            mask=mask,
            attn_bias=attn_bias
        )  # [H,B,Nq,Nk]

        attn = torch.softmax(compatibility, dim=-1)

        # Weighted sum of value vectors
        out_heads = torch.matmul(attn, value_heads)  # [H,B,Nq,D_head]

        # Merge heads and project back to embedding_dim
        out = self.project_out(self._unmake_heads(out_heads))  # [B,Nq,D]
        return out

    def _make_heads(self, v):
        B, N, D = v.shape
        head_dim = D // self.n_heads
        # [B,N,D] → [H,B,N,head_dim]
        return v.reshape(B, N, self.n_heads, head_dim).movedim(-2, 0)

    def _unmake_heads(self, v):
        # [H,B,N,head_dim] → [B,N,H*head_dim]
        return v.movedim(0, -2).flatten(-2)



class MultiHeadAttentionProj(nn.Module):
    """
    Multi-head attention with linear projections of q/k/v.

    Args:
        q: [B, Nq, D]
        h: [B, Nk, D] or None (defaults to q for self-attention)
        mask: [B, Nk] (bool)
        attn_bias: Optional attention bias, e.g. [B, Nq, Nk]
    """
    def __init__(self, embedding_dim, n_heads=8):
        super().__init__()
        self.queryEncoder = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.keyEncoder   = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.valueEncoder = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.MHA = MultiHeadAttentionEncoder(embedding_dim, n_heads)

    def forward(self, q, h=None, mask=None, attn_bias=None):
        if h is None:
            h = q  # self-attention case

        # Linear projections
        query = self.queryEncoder(q)
        key   = self.keyEncoder(h)
        value = self.valueEncoder(h)

        # Multi-head attention with optional bias
        out = self.MHA(query, key, value, mask=mask, attn_bias=attn_bias)
        return out