#Task 1.1, implenenting dense attention ( matmul + mask + attention )
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




