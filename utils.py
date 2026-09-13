import torch
import torch.nn.functional as F


"""
Returns attention weights of the same shape, softmaxed over the last
    dim within valid_mask, with fully-invalid rows set to all-zero
    (never NaN).
"""
def safe_softmax(scores: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:

    valid_mask = valid_mask.expand_as(scores)

    masked_scores = scores.masked_fill(~valid_mask, float('-inf'))

    row_has_valid = valid_mask.any(dim=-1, keepdim=True)

    safe_scores = masked_scores.masked_fill(~row_has_valid, 0.0)

    attn = F.softmax(safe_scores, dim=-1)

    attn = attn.masked_fill(~valid_mask, 0.0)

    return attn

def causal_mask(N: int, device=None) -> torch.Tensor:
    idx = torch.arange(N, device=device)
    return idx.unsqueeze(0) <= idx.unsqueeze(1)

def sliding_window_full_mask(N: int, window: int, causal: bool = True, device=None) -> torch.Tensor:
    i = torch.arange(N, device=device).unsqueeze(1)
    j = torch.arange(N, device=device).unsqueeze(0)
    if causal:
        mask = (j <= i) & (j >= i - window)
    else:
        mask = (j >= i - window) & (j <= i + window)
    return mask


def block_sparse_full_mask(
    N: int,
    block_size: int,
    num_global_blocks: int,
    num_random_blocks: int,
    causal: bool = True,
    device=None,
    generator: torch.Generator = None,
) -> torch.Tensor:
    
    import math
    num_blocks = math.ceil(N / block_size)
    block_id = torch.arange(N, device=device) // block_size

    bi = block_id.unsqueeze(1)
    bj = block_id.unsqueeze(0)
    local = (bi - bj).abs() <= 1

    is_global = block_id < num_global_blocks
    global_rows = is_global.unsqueeze(1).expand(N, N)
    global_cols = is_global.unsqueeze(0).expand(N, N)

    random_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    if num_random_blocks > 0:
        for qb in range(num_blocks):
            candidates = [b for b in range(num_blocks) if b != qb]
            if not candidates:
                continue
            k = min(num_random_blocks, len(candidates))
            if generator is not None:
                perm = torch.randperm(len(candidates), generator=generator)[:k]
            else:
                perm = torch.randperm(len(candidates))[:k]
            for p in perm.tolist():
                kb = candidates[p]
                q_lo, q_hi = qb * block_size, min((qb + 1) * block_size, N)
                k_lo, k_hi = kb * block_size, min((kb + 1) * block_size, N)
                random_mask[q_lo:q_hi, k_lo:k_hi] = True

    mask = local | global_rows | global_cols | random_mask
    if causal:
        mask = mask & causal_mask(N, device=device)
    return mask





