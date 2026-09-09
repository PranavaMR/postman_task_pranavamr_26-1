#Task 1.1, implenenting dense attention ( matmul + mask + attention )
# note: ALL THE COMMENTS ARE MADE FOR ME TO UNDERSTAND THE CODE.
import math
import torch
import torch.nn.functional as F

# Never NaN - since it makes every row with no valid/inf entries to be uniform
def safe_softmax(scores: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:

    valid_mask = valid_mask.expand_as(scores)

    masked_scores = scores.masked_fill(~valid_mask, float('-inf'))

    row_has_valid = valid_mask.any(dim=-1, keepdim=True)

    safe_scores = masked_scores.masked_fill(~row_has_valid, 0.0)

    attn = F.softmax(safe_scores, dim=-1)

    attn = attn.masked_fill(~valid_mask, 0.0)

    return attn

def dense_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # shape of Q, K, V: (B, H, N, D) is the same,
    # B is batch size
    # H is number of attention heads
    # N is sequence length
    # D is embedding dimension/head
    B, H, N, D = Q.shape
    # i do transpose (-2,-1) to get the correct shape for matmul, 
    # since i want to multiply Q (B,H,N,D) with K (B,H,D,N) to get scores (B,H,N,N)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)  # (B, H, N, N)
    # 1. apply mask of ones if not there
    # 2. apply safe softmax to get attention weights then multiply with V
    if mask is None:
        mask = torch.ones(N, N, dtype=torch.bool, device=Q.device)

    attn = safe_softmax(scores, mask)
    out = torch.matmul(attn, V)
    return out, attn




