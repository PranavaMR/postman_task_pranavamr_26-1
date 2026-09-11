import torch
import math
from typing import List, Optional, Callable

from .utils import safe_softmax, causal_mask

def _pad_to_blocks(x: torch.Tensor, block_size: int):
    B, H, N, D = x.shape
    pad = (block_size - N % block_size) % block_size
    if pad:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad))
    return x, pad


def build_block_pattern(
    num_blocks: int,
    kind: str,
    local_window_blocks: int = 1,
    num_global_blocks: int = 0,
    num_random_blocks: int = 0,
    causal: bool = True,
    generator: Optional[torch.Generator] = None,
) -> List[List[int]]:

    assert kind in ("sliding_window", "block_sparse")
    pattern: List[List[int]] = []

    for qb in range(num_blocks):
        if kind == "sliding_window":
            lo = max(0, qb - local_window_blocks)
            hi = qb if causal else min(num_blocks - 1, qb + local_window_blocks)
            pattern.append(list(range(lo, hi + 1)))
            continue

        if qb < num_global_blocks:
            hi = qb if causal else num_blocks - 1
            pattern.append(list(range(0, hi + 1)))
            continue

        allowed = set()
        lo = max(0, qb - local_window_blocks)
        hi = qb if causal else min(num_blocks - 1, qb + local_window_blocks)
        allowed.update(range(lo, hi + 1))

        for gb in range(min(num_global_blocks, num_blocks)):
            if not causal or gb <= qb:
                allowed.add(gb)

        if num_random_blocks > 0:
            candidates = [
                b for b in range(num_blocks)
                if b not in allowed and (not causal or b <= qb)
            ]
            if candidates:
                k = min(num_random_blocks, len(candidates))
                if generator is not None:
                    perm = torch.randperm(len(candidates), generator=generator)[:k]
                else:
                    perm = torch.randperm(len(candidates))[:k]
                for p in perm.tolist():
                    allowed.add(candidates[p])

        pattern.append(sorted(allowed))

    return pattern



def pattern_to_full_mask(N: int, block_size: int, pattern: List[List[int]], causal: bool = True, device=None) -> torch.Tensor:
    mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    for qb, key_blocks in enumerate(pattern):
        q_lo, q_hi = qb * block_size, min((qb + 1) * block_size, N)
        if q_lo >= N:
            break
        for kb in key_blocks:
            k_lo, k_hi = kb * block_size, min((kb + 1) * block_size, N)
            if k_lo >= N:
                continue
            mask[q_lo:q_hi, k_lo:k_hi] = True
    if causal:
        mask = mask & causal_mask(N, device=device)
    return mask

def block_sparse_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_size: int,
    pattern: List[List[int]],
    causal: bool = True,
    extra_valid_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
):
    B, H, N, D = Q.shape
    Qp, _ = _pad_to_blocks(Q, block_size)
    Kp, _ = _pad_to_blocks(K, block_size)
    Vp, _ = _pad_to_blocks(V, block_size)
    Np = Qp.shape[2]
    num_blocks = Np // block_size
    assert len(pattern) == num_blocks, f"pattern has {len(pattern)} blocks, expected {num_blocks}"

    out = torch.zeros_like(Qp)

    for qb in range(num_blocks):
        key_blocks = pattern[qb]
        if not key_blocks:
            continue  

        q_lo, q_hi = qb * block_size, (qb + 1) * block_size
        q_slice = Qp[:, :, q_lo:q_hi, :]  # (B, H, bs, D) ( for my reference )

        k_idx = torch.cat([
            torch.arange(kb * block_size, (kb + 1) * block_size, device=Qp.device)
            for kb in key_blocks
        ])
        k_gather = Kp[:, :, k_idx, :]  # (B, H, gathered_len, D) ( for my own referencee)
        v_gather = Vp[:, :, k_idx, :]

        scores = torch.matmul(q_slice, k_gather.transpose(-2, -1)) / math.sqrt(D)  # (B, H, bs, gathered_len)

        q_pos = torch.arange(q_lo, q_hi, device=Qp.device).unsqueeze(1)  # (bs, 1)
        k_pos = k_idx.unsqueeze(0)  # (1, gathered_len)

        valid = (k_pos <= q_pos) if causal else torch.ones(q_pos.shape[0], k_pos.shape[1], dtype=torch.bool, device=Qp.device)
        valid = valid & (k_pos < N)  
        if extra_valid_fn is not None:
            valid = valid & extra_valid_fn(q_pos, k_pos)

        attn = safe_softmax(scores, valid.unsqueeze(0).unsqueeze(0))  # broadcast over (B, H)
        out[:, :, q_lo:q_hi, :] = torch.matmul(attn, v_gather)

    return out[:, :, :N, :]
