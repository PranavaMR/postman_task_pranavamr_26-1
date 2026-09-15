#training chargpt on tiny shakespeare dataset with different attention patterns

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from models.model import CharGPT


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tinyshakespeare.txt")
BLOCK_SIZE = 128
BATCH_SIZE = 32
N_EMBD = 64
N_HEAD = 4
N_LAYER = 2
LR = 3e-3
EVAL_ITERS = 20

ATTN_KWARGS = {
    "dense": {},
    "sliding_window": {"window": 32, "block_size": 16},
    "block_sparse": {"block_size": 16, "num_global_blocks": 1, "num_random_blocks": 2},
}


def load_data():
    with open(DATA_PATH) as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], len(chars)


def get_batch(data):
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    model.eval()
    out = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(data)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def train(attention_type, steps, log_every, seed=0):
    torch.manual_seed(seed)
    train_data, val_data, vocab_size = load_data()
    model = CharGPT(
        vocab_size, BLOCK_SIZE, n_embd=N_EMBD, n_head=N_HEAD, n_layer=N_LAYER,
        attention_type=attention_type, attn_kwargs=ATTN_KWARGS[attention_type],
    )
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    history = []
    t0 = time.time()
    for step in range(steps + 1):
        if step % log_every == 0 or step == steps:
            losses = estimate_loss(model, train_data, val_data)
            elapsed = time.time() - t0
            history.append({"step": step, "train_loss": losses["train"],
                             "val_loss": losses["val"], "elapsed_s": elapsed})
            print(f"[{attention_type}] step {step:4d}  train {losses['train']:.4f}  "
                  f"val {losses['val']:.4f}  ({elapsed:.1f}s elapsed)")
        x, y = get_batch(train_data)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return {"attention_type": attention_type, "n_params": n_params,
            "total_time_s": time.time() - t0, "history": history}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention", required=True, choices=["dense", "sliding_window", "block_sparse"])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = train(args.attention, args.steps, args.log_every, args.seed)
    out_path = os.path.join(os.path.dirname(__file__), f"loss_history_{args.attention}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {out_path}")
