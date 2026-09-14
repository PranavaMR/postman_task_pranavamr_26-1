import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from patterns.dense_attention import dense_attention
from patterns.sparse_patterns import sliding_window_attention, block_sparse_bigbird_attention
from patterns.utils import causal_mask

assert torch.cuda.is_available(), (
    "No CUDA GPU visible. On Colab: Runtime -> Change runtime type -> "
    "Hardware accelerator -> T4 GPU, then re-run this cell. Running this "
    "benchmark on CPU by accident gives numbers you'll mistake for GPU numbers."
)
DEVICE = "cuda"
print(f"Running on: {torch.cuda.get_device_name(0)}", file=sys.stderr)

#batch size, number of heads, embedding dimension per head
#number of heads chosen as 4 for colab T4 GPU
#for nvidia rtx 3060 16gb you can use H=8
#16384 will be OOM for dense attention on T4.
B, H, D = 1, 4, 64
WINDOW = 128
BLOCK_SIZE = 64
NUM_GLOBAL = 1
NUM_RANDOM = 3
N_VALUES = [512, 1024, 2048, 4096, 8192, 16384]
WARMUP_RUNS = 3
TIMED_RUNS = 5


def run_single(variant: str, N: int):
    torch.manual_seed(0)
    Q = torch.randn(B, H, N, D, device=DEVICE)
    K = torch.randn(B, H, N, D, device=DEVICE)
    V = torch.randn(B, H, N, D, device=DEVICE)

    if variant == "dense":
        mask = causal_mask(N, device=DEVICE)
        fn = lambda: dense_attention(Q, K, V, mask=mask)[0]
    elif variant == "sliding_window":
        fn = lambda: sliding_window_attention(Q, K, V, window=WINDOW, block_size=BLOCK_SIZE, causal=True)
    elif variant == "block_sparse":
        fn = lambda: block_sparse_bigbird_attention(
            Q, K, V, block_size=BLOCK_SIZE, num_global_blocks=NUM_GLOBAL,
            num_random_blocks=NUM_RANDOM, causal=True,
        )[0]
    else:
        raise ValueError(variant)

    for _ in range(WARMUP_RUNS):
        fn()
    torch.cuda.synchronize()

    # Reset after warmup so warmup's allocations don't inflate the reading,
    # and before the timed loop so this config's peak isn't contaminated by
    # whatever the previous config (in the same process) used.
    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(TIMED_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()  # wait for the GPU to actually finish before stopping the clock
        times.append(time.perf_counter() - t0)

    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
    return {"variant": variant, "N": N, "time_s": min(times), "peak_mem_mb": peak_mem_mb}


def make_plots(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = ["dense", "sliding_window", "block_sparse"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for variant in variants:
        pts = [(r["N"], r["time_s"]) for r in results if r["variant"] == variant and r.get("time_s") is not None]
        if not pts:
            continue
        xs, ys = zip(*sorted(pts))
        axes[0].plot(xs, ys, marker="o", label=variant)
    axes[0].set_xlabel("Sequence length N")
    axes[0].set_ylabel("Wall-clock time (s, forward pass, best of 5)")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_title("Time vs N (T4 GPU)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for variant in variants:
        pts = [(r["N"], r["peak_mem_mb"]) for r in results if r["variant"] == variant and r.get("peak_mem_mb") is not None]
        if not pts:
            continue
        xs, ys = zip(*sorted(pts))
        axes[1].plot(xs, ys, marker="o", label=variant)
    axes[1].set_xlabel("Sequence length N")
    axes[1].set_ylabel("Peak GPU memory (MB), reset per config")
    axes[1].set_xscale("log", base=2)
    axes[1].set_title("Peak memory vs N (T4 GPU)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "benchmark_plots.png")
    plt.savefig(path, dpi=150)
    print(f"Saved plot to {path}", file=sys.stderr)


def main():
    results = []
    for N in N_VALUES:
        for variant in ["dense", "sliding_window", "block_sparse"]:
            print(f"Running {variant} @ N={N} ...", file=sys.stderr)
            try:
                results.append(run_single(variant, N))
            except torch.cuda.OutOfMemoryError as e:
                print(f"  -> OOM at N={N}: {e}", file=sys.stderr)
                results.append({"variant": variant, "N": N, "time_s": None, "peak_mem_mb": None, "failed": True})
                torch.cuda.empty_cache()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    payload = {
        "hardware": torch.cuda.get_device_name(0),
        "shape": {
            "B": B, "H": H, "D": D, "window": WINDOW, "block_size": BLOCK_SIZE,
            "num_global_blocks": NUM_GLOBAL, "num_random_blocks": NUM_RANDOM,
        },
        "results": results,
    }
    with open(os.path.join(out_dir, "benchmark_results.json"), "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    make_plots(results, out_dir)


if __name__ == "__main__":
    main()