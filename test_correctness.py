
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


#1.3 sliding window vs dense attention

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


#block sparse vs dense attention with mask

def test_block_sparse_matches_dense_causal():
    B, H, N, D = 2, 3, 100, 16
    block_size = 10
    Q, K, V = make_qkv(B, H, N, D)

    gen = torch.Generator().manual_seed(42)
    out_sparse, pattern = block_sparse_bigbird_attention(
        Q, K, V, block_size=block_size, num_global_blocks=1, num_random_blocks=2,
        local_window_blocks=1, causal=True, generator=gen,
    )

    ref_mask = pattern_to_full_mask(N, block_size, pattern, causal=True)
    out_dense, _ = dense_attention(Q, K, V, mask=ref_mask)

    ok = torch.allclose(out_sparse, out_dense, atol=ATOL, rtol=RTOL)
    diff = (out_sparse - out_dense).abs().max().item()
    report("block_sparse (BigBird, causal) matches dense", ok, f"max abs diff={diff:.2e}")


def test_block_sparse_matches_dense_no_random():
    B, H, N, D = 1, 2, 64, 8
    block_size = 8
    Q, K, V = make_qkv(B, H, N, D)

    out_sparse, pattern = block_sparse_bigbird_attention(
        Q, K, V, block_size=block_size, num_global_blocks=1, num_random_blocks=0,
        local_window_blocks=1, causal=True,
    )
    ref_mask = pattern_to_full_mask(N, block_size, pattern, causal=True)
    out_dense, _ = dense_attention(Q, K, V, mask=ref_mask)

    ok = torch.allclose(out_sparse, out_dense, atol=ATOL, rtol=RTOL)
    diff = (out_sparse - out_dense).abs().max().item()
    report("block_sparse (local+global only) matches dense", ok, f"max abs diff={diff:.2e}")

#NaN handling
def test_dense_attention_no_nan_on_fully_masked_row():
    B, H, N, D = 1, 1, 5, 4
    Q, K, V = make_qkv(B, H, N, D)
    mask = torch.ones(N, N, dtype=torch.bool)
    mask[2, :] = False  # query 2 has zero valid keys

    out, attn = dense_attention(Q, K, V, mask=mask)
    ok_no_nan = not torch.isnan(out).any() and not torch.isnan(attn).any()
    ok_zero_row = torch.allclose(out[:, :, 2, :], torch.zeros_like(out[:, :, 2, :]))
    ok_zero_attn = torch.allclose(attn[:, :, 2, :], torch.zeros_like(attn[:, :, 2, :]))
    report("dense_attention: fully-masked row -> no NaN", ok_no_nan and ok_zero_row and ok_zero_attn)


def test_sliding_window_no_nan_at_block_boundary():

    B, H, N, D = 1, 2, 37, 8  # 37 is not a multiple of block_size below
    Q, K, V = make_qkv(B, H, N, D)
    out = sliding_window_attention(Q, K, V, window=1, block_size=16, causal=True)
    ok = not torch.isnan(out).any()
    report("sliding_window: no NaN at block boundary (N=37, block_size=16)", ok)


def test_block_sparse_no_nan_first_query_block():

    B, H, N, D = 1, 1, 40, 8
    Q, K, V = make_qkv(B, H, N, D)
    out, pattern = block_sparse_bigbird_attention(
        Q, K, V, block_size=8, num_global_blocks=0, num_random_blocks=3,
        local_window_blocks=0, causal=True,  # window=0: block only ever sees itself
    )
    ok = not torch.isnan(out).any()
    report("block_sparse: no NaN in first query block (no global, window=0)", ok, f"pattern[0]={pattern[0]}")


def test_empty_pattern_block_no_nan():
    from src.sparse_patterns import block_sparse_attention
    B, H, N, D = 1, 1, 16, 4
    Q, K, V = make_qkv(B, H, N, D)
    pattern = [[0], []]  # second block has NO allowed key blocks at all
    out = block_sparse_attention(Q, K, V, block_size=8, pattern=pattern, causal=True)
    ok = not torch.isnan(out).any()
    ok_zero = torch.allclose(out[:, :, 8:, :], torch.zeros_like(out[:, :, 8:, :]))
    report("block_sparse_attention: empty pattern block -> zero output, no NaN", ok and ok_zero)


ALL_TESTS = [
    test_sliding_window_matches_dense_causal,
    test_sliding_window_matches_dense_noncausal,
    test_block_sparse_matches_dense_causal,
    test_block_sparse_matches_dense_no_random,
    test_dense_attention_no_nan_on_fully_masked_row,
    test_sliding_window_no_nan_at_block_boundary,
    test_block_sparse_no_nan_first_query_block,
    test_empty_pattern_block_no_nan,
]


if __name__ == "__main__":
    failures = 0
    for t in ALL_TESTS:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"  -> {e}")
    print()
    if failures:
        print(f"{failures}/{len(ALL_TESTS)} tests FAILED")
        sys.exit(1)
    else:
        print(f"All {len(ALL_TESTS)} tests passed.")
