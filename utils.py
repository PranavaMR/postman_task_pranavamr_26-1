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





