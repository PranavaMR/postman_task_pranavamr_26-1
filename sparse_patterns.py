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
    """
    Returns pattern[qb] = sorted list of key-block indices query block qb
    may attend to (before fine-grained, token-level trimming).

    kind='sliding_window': local blocks only (a safe superset later
        trimmed to the exact token window by the engine's extra_valid_fn).
    kind='block_sparse': BigBird-style local + global + random, matching
        the classic construction:
          - local: |query_block - key_block| <= local_window_blocks
          - global: the first `num_global_blocks` blocks are attended to
            by every query (global "columns"), AND themselves attend to
            everything causally available (global "rows" — a global
            query gets full causal attention, not just local/random).
          - random: each non-global query block additionally attends to
            `num_random_blocks` randomly chosen blocks.
    """
    assert kind in ("sliding_window", "block_sparse")
    pattern: List[List[int]] = []

    for qb in range(num_blocks):
        if kind == "sliding_window":
            lo = max(0, qb - local_window_blocks)
            hi = qb if causal else min(num_blocks - 1, qb + local_window_blocks)
            pattern.append(list(range(lo, hi + 1)))
            continue

        # kind == "block_sparse"
        if qb < num_global_blocks:
            # Global query block: full causal attention (matches the
            # "global rows attend to everything" BigBird property).
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
