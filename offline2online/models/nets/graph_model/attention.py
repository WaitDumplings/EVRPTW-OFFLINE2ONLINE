
import torch
import torch.nn as nn
import math

# Skip Connection Module
class SkipConnection(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)

# Normalization Module
class Normalization(nn.Module):
    """
    Normalization module to apply LayerNorm, BatchNorm1d, or InstanceNorm1d.

    Args:
        embed_dim (int): Embedding dimension for normalization.
        normalization (str): Type of normalization ('layer', 'batch', 'instance').

    Returns:
        Normalized tensor of the same shape as input.
    """

    def __init__(self, embed_dim, normalization='layer'):
        super().__init__()
        if normalization == 'layer':
            self.normalizer = nn.LayerNorm(embed_dim)
        elif normalization == 'batch':
            self.normalizer = nn.BatchNorm1d(embed_dim)
        elif normalization == "instance":
            self.normalizer = nn.InstanceNorm1d(embed_dim)
        else:
            raise ValueError(f"Unsupported normalization type: {normalization}")

    def forward(self, x):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
        return self.normalizer(x)

# Feed-Forward Network with Residuals
class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation=nn.ReLU):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.residual = SkipConnection(self.ffn)

    def forward(self, x):
        return self.residual(x)

# Multi-Head Attention Module
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, input_dim, embed_dim):
        super().__init__()
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.proj_qkv = nn.Linear(input_dim, embed_dim * 3)
        self.proj_out = nn.Linear(embed_dim, input_dim)
        self.norm_factor = 1 / math.sqrt(embed_dim // n_heads)

    def forward(self, x, mask=None):
        assert len(x.size()) == 3, "The input shape should follow the rule: Batch Size, Seq_len, Hidden dim"
        bs, seq_len, _ = x.size()
        qkv = self.proj_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(bs, seq_len, self.n_heads, -1).transpose(1, 2) for t in qkv]

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.norm_factor
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)

        # Weighted sum of values
        out = torch.matmul(attn, v).transpose(1, 2).reshape(bs, seq_len, -1)
        return self.proj_out(out)

# Multi-Head Attention Block
class MHA_Block(nn.Module):
    def __init__(self, num_head, input_dim, attention_dim, ffn_dim, normalization='batch'):
        super().__init__()
        self.mha = MultiHeadAttention(num_head, input_dim, attention_dim)
        self.norm1 = Normalization(input_dim, normalization)
        self.ffn = FFN(input_dim, ffn_dim, input_dim)
        self.norm2 = Normalization(input_dim, normalization)

    def forward(self, x, mask=None):
        # Multi-Head Attention with Residual Connection and Normalization
        x = self.norm1(x + self.mha(x, mask=mask))
        # Feed-Forward Network with Residual Connection and Normalization
        x = self.norm2(x + self.ffn(x))
        return x