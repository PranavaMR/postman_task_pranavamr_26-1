# Sparse Attention from Scratch

This repository implements and studies two sparse attention patterns from scratch:

- **Sliding-window attention**
- **BigBird-style block-sparse attention** with local, global and random connections

It also contains a dense attention reference implementation, a correctness harness, a Tiny Shakespeare character-level GPT experiment, and GPU benchmarks.

The point of the project is not just to reproduce a known sparse-attention pattern. I wanted to understand what changes when we stop computing the full $N \times N$ attention matrix, how to verify that the sparse implementation is correct, and what the trade-offs actually look like on a real GPU.

## What is being compared?

Dense causal attention computes

$$
\mathrm{softmax}\left(\frac{QK^T}{\sqrt{D}}\right)V
$$

over every query-key pair, subject to the causal mask.

The sparse implementations use the same basic attention calculation, but they first decide which key/value blocks a query block is allowed to access.

### Sliding window

A query at position $i$ can attend only to nearby positions.

For causal attention:

$$
i - \text{window} \le j \le i
$$

The implementation uses block-level routing as a safe superset and then applies a token-level mask to get the exact window.

### BigBird-style block sparse attention

Each query block can use:

- nearby local blocks
- a small number of global blocks
- a small number of random blocks

Only those K/V blocks are gathered for the attention calculation.

The important distinction is that the implementation does **not** build the full $N \times N$ score matrix and then throw most of it away. The unnecessary blocks are never used in the sparse computation.

## Project structure

The main pieces are:

```text
patterns/
    dense_attention.py
    sparse_patterns.py
    utils.py

model/
    model.py

tests/
    test_correctness.py

benchmarks/
    benchmarks.py

train.py
data/
    tinyshakespeare.txt
```

The exact surrounding filenames can be changed without affecting the core idea; the important modules are the dense attention reference, sparse attention implementation, model, training script, correctness tests, and benchmark script.

## Data ( important )

As the instructions had said, I haven't added the data/ directory in the repository.

To run it in an alternative efficient way, you must download the entire repo, into a folder named Sparse_Attention, and add a directory named 'data' with tinyshakespeare.txt in it.

Once that's done, convert the Sparse_Attention folder into a ".zip" file.
Make sure the directory structure is : 
Sparse_Attention.zip/Sparse_Attention/...

Then using this colab link : 
https://colab.research.google.com/drive/1U71dVKyKjdcfkSZjfEt_LErZLaicyIEp?usp=sharing

You may upload Sparse_Attention.zip and run benchmarks and the training loop to evaluate the model.

## Correctness

The correctness tests compare the sparse implementations against dense attention using the same effective attention pattern.

For sliding-window attention, the test constructs the full token-level sliding-window mask and compares the sparse result with dense attention using that mask.

For BigBird attention, the test takes the actual block routing pattern and expands it into a full $N \times N$ mask before running dense attention.

There are also tests for:

- causal and non-causal sliding windows
- BigBird with and without random connections
- fully masked rows
- non-multiple-of-block-size sequence lengths
- empty pattern blocks
- NaN safety at block boundaries

Run the correctness test from the repository root with the path where the test file lives, for example:

```bash
python tests/test_correctness.py
```

A successful run ends with:

```text
All 8 tests passed.
```

## Tiny Shakespeare experiment

The training code builds a small character-level GPT and swaps only the attention mechanism.

The recorded configuration uses:

```text
context length: 128
batch size:     32
embedding size: 64
heads:          4
layers:         2
learning rate:  3e-3
```

The training script supports:

```text
dense
sliding_window
block_sparse
```

and writes a JSON loss history for each run.

Example:

```bash
python train.py --attention dense --steps 800 --log_every 100
python train.py --attention sliding_window --steps 800 --log_every 100
python train.py --attention block_sparse --steps 800 --log_every 100
```

The training data is expected at:

```text
data/tinyshakespeare.txt
```

The recorded experiments use a character-level vocabulary built directly from the dataset.

## Benchmark

The benchmark measures forward-pass wall-clock time and peak GPU memory for:

```text
N = 512, 1024, 2048, 4096, 8192, 16384
```

The recorded benchmark was run on a **Tesla T4** with:

```text
B = 1
H = 4
D = 64
sliding window = 128
block size = 64
global blocks = 1
random blocks = 3
warmup runs = 3
timed runs = 5
```

The benchmark synchronizes the CUDA device before and after each timed run and reports the best of five measured forward passes.

Run it on a CUDA GPU:

```bash
python benchmarks/benchmarks.py
```

The script produces:

```text
benchmark_results.json
benchmark_plots.png
```

Dense attention eventually runs out of memory at the largest tested sequence length on the T4, while the sparse variants continue to run.

## What to expect

This implementation is intentionally simple and written for understanding rather than maximum GPU performance.

Because the sparse engine loops over query blocks in Python and launches smaller GPU operations, sparse attention is not guaranteed to be faster at short sequence lengths. In the recorded T4 benchmark, dense attention was faster through $N=4096$, while the sparse variants became faster at $N=8192$. Dense attention then failed with OOM at $N=16384$, while both sparse implementations completed.

So the main result is not:

> sparse attention is always faster.

It is:

> sparse attention can trade some small/medium-sequence overhead for much lower memory growth and the ability to handle longer sequences.

That trade-off is the main thing this project is intended to make visible.
