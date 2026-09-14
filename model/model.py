import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from patterns.dense_attention import dense_attention
from patterns.sparse_patterns import sliding_window_attention, block_sparse_bigbird_attention
from patterns.utils import causal_mask

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attention_type="dense", attn_kwargs=None):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attention_type = attention_type
        self.attn_kwargs = attn_kwargs or {}
        if attention_type == "block_sparse":
            self._pattern_generator = torch.Generator().manual_seed(1234)
        else:
            self._pattern_generator = None

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, N, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        k = k.view(B, N, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.n_head, self.head_dim).transpose(1, 2)

        if self.attention_type == "dense":
            mask = causal_mask(N, device=x.device)
            out, _ = dense_attention(q, k, v, mask=mask)
        elif self.attention_type == "sliding_window":
            out = sliding_window_attention(q, k, v, causal=True, **self.attn_kwargs)
        elif self.attention_type == "block_sparse":
            out, _ = block_sparse_bigbird_attention(
                q, k, v, causal=True, generator=self._pattern_generator, **self.attn_kwargs
            )
        else:
            raise ValueError(f"unknown attention_type {self.attention_type}")

        out = out.transpose(1, 2).contiguous().view(B, N, C)
        return self.proj(out)

class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, n_embd, n_head, attention_type, attn_kwargs):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attention_type, attn_kwargs)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

