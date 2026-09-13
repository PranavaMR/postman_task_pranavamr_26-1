
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.dense_attention import dense_attention
from src.sparse_patterns import (
    sliding_window_attention,
    block_sparse_bigbird_attention,
    pattern_to_full_mask,
)
from src.utils import sliding_window_full_mask, safe_softmax

torch.manual_seed(0)

ATOL = 1e-5
RTOL = 1e-4


def make_qkv(B, H, N, D, device="cpu"):
    Q = torch.randn(B, H, N, D, device=device)
    K = torch.randn(B, H, N, D, device=device)
    V = torch.randn(B, H, N, D, device=device)
    return Q, K, V


def report(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name} {detail}")
    assert ok, f"{name} failed: {detail}"


# ---------------------------------------------------------------------
# 1.3 — sliding window vs dense-with-equivalent-mask
# ---------------------------------------------------------------------

def test_sliding_window_matches_dense_causal():
    B, H, N, D = 2, 3, 97, 16  # N not a multiple of block_size on purpose
    window = 12
    Q, K, V = make_qkv(B, H, N, D)

    out_sparse = sliding_window_attention(Q, K, V, window=window, block_size=16, causal=True)

    ref_mask = sliding_window_full_mask(N, window, causal=True)
    out_dense, _ = dense_attention(Q, K, V, mask=ref_mask)

    diff = (out_sparse - out_dense).abs().max().item()
    ok = torch.allclose(out_sparse, out_dense, atol=ATOL, rtol=RTOL)
    report("sliding_window (causal) matches dense", ok, f"max abs diff={diff:.2e}")


def test_sliding_window_matches_dense_noncausal():
    B, H, N, D = 2, 2, 80, 8
    window = 5
    Q, K, V = make_qkv(B, H, N, D)

    out_sparse = sliding_window_attention(Q, K, V, window=window, block_size=8, causal=False)
    ref_mask = sliding_window_full_mask(N, window, causal=False)
    out_dense, _ = dense_attention(Q, K, V, mask=ref_mask)

    ok = torch.allclose(out_sparse, out_dense, atol=ATOL, rtol=RTOL)
    diff = (out_sparse - out_dense).abs().max().item()
    report("sliding_window (non-causal) matches dense", ok, f"max abs diff={diff:.2e}")


