# Task 1 — Sparse Attention

## What I was trying to do

The goal of this task was to understand what is actually gained by making self-attention sparse, rather than just using a different masking pattern.

I implemented dense causal attention from the basic equation

```
Attention(Q, K, V) = softmax(Q · Kᵀ / √D) · V
```

and then built two sparse variants on top of a shared block-based execution engine:

- sliding-window attention
- BigBird-style block-sparse attention with local, global and random connections

The important implementation choice was not to compute the full N×N attention matrix and then mask most of it. Instead, the sparse implementation selects the key/value blocks a query block is allowed to see, gathers only those blocks, and computes attention on that smaller set. The mask is then used for token-level restrictions such as causality, padding, and the exact sliding-window boundary.

I also wrote a correctness harness that compares the sparse implementations against dense attention using the same effective attention pattern.

---

## 1. Implementation

The code is split into a few simple pieces.

`dense_attention()` is the reference implementation. It computes the complete Q·Kᵀ matrix, applies a boolean mask, performs a safe softmax, and multiplies the attention weights by V.

The sparse implementation is built around two ideas. First, `build_block_pattern()` decides which key blocks each query block is allowed to access. Second, `block_sparse_attention()` actually gathers those K/V blocks and runs attention on them.

This separation turned out to be important. A sparse pattern is not useful for performance if the program still computes the full dense N×N matrix first. The block-level routing is what lets the implementation avoid those unnecessary scores.

For sliding-window attention, I use a deliberately safe superset of nearby blocks and then apply a finer token-level validity function to trim it to the exact window. For BigBird-style attention, the pattern combines local blocks, global blocks, and randomly selected blocks.

I also had to handle sequence lengths that are not multiples of the block size. Internally, the tensors are padded to whole blocks, but padded key positions are explicitly excluded. Fully masked rows are handled with a `safe_softmax` so they produce zero attention/output instead of NaNs.

---

## 2. Correctness

Before benchmarking speed, I wanted to make sure the sparse implementations were actually doing the same computation they were supposed to do.

The correctness tests compare:

1. sparse sliding-window attention vs dense attention with the exact sliding-window mask
2. sparse BigBird attention vs dense attention with the corresponding full block-sparse mask
3. causal and non-causal sliding-window cases
4. BigBird with and without random connections

The tests also deliberately include awkward cases: a sequence length that is not divisible by the block size, fully masked rows, an empty pattern for a query block, and the first query block in a causal BigBird setup.

The reason for having a dense reference at all is simple: for the sparse implementation, many entries of the conceptual N×N attention matrix are never computed. The reference mask reconstructs those same allowed connections at full resolution, which gives a straightforward way to check the sparse result.

---

## 3. Character-level GPT experiment

For the quality experiment, I trained the same small character-level GPT with each attention variant on Tiny Shakespeare. The model has 2 transformer layers, 4 attention heads and 64-dimensional embeddings. The training script uses a context length of 128 and compares the three attention mechanisms while keeping the model architecture and optimizer setup the same.

All three models learned the task normally. The final validation losses were in a very similar range rather than showing a dramatic collapse for either sparse method.

At step 3000, the validation losses were:

| Attention | Validation loss |
|---|---:|
| Dense | 1.6632 |
| Sliding window | 1.6527 |
| Block sparse | 1.6657 |

At step 3100, they were:

| Attention | Validation loss |
|---|---:|
| Dense | 1.6915 |
| Sliding window | 1.6880 |
| Block sparse | 1.6999 |

The block-sparse run continued further than the other two in the recorded run; at step 3300 its validation loss was 1.6853.

I would not interpret the small differences here as proof that one pattern is universally better. This is a small character-level experiment, and the losses fluctuate during training. The useful result is that both sparse patterns remained capable of learning the task rather than becoming unusable because of the information they removed.

### What information does each pattern lose?

**Sliding-window attention loses long-range direct interactions.**

A token can only directly attend to nearby tokens. Information from far away in the sequence can still influence a representation indirectly through intermediate tokens and later layers, but there is no single attention operation that jumps from one distant token to another. The pattern therefore preserves local structure well but deliberately throws away direct long-range connections.

For text, that can matter when the useful dependency is not local: a character or word may depend on something much earlier in the sequence.

**BigBird-style attention loses fewer long-range connections, but it does not preserve all of them.**

Local attention handles nearby structure. Global blocks provide a small number of broadly connected positions. Random connections add a few long-range routes. The result is a sparse graph rather than a fully connected attention matrix.

This is a better approximation to dense attention when long-range information matters because the model still has explicit paths to information outside its local neighborhood. But it is still sparse: most possible query-key pairs are missing.

---

## 4. Why global tokens can matter disproportionately

Global tokens are interesting because they are not just one more category of connection.

A normal local token participates in attention mostly with its neighbors. A global token can become a shared communication point: many query blocks can read from it, and a global query can aggregate information from a much larger part of the sequence.

That means a small number of global blocks can influence many parts of the sequence. Removing one ordinary local connection affects a relatively small region; removing a well-placed global connection can remove a route that many query positions were relying on.

I did not run a separate ablation that varies the number of global blocks, so I am treating this as an explanation of the pattern rather than claiming a measured "global-token effect" from this experiment.

---

## 5. Benchmark: time and memory

I benchmarked the three variants on a Tesla T4 using:

- batch size: 1
- heads: 4
- head dimension: 64
- sliding window: 128
- block size: 64
- 1 global block
- 3 random blocks for BigBird
- 3 warmup runs
- 5 timed runs

I tested sequence lengths from 512 to 16384.

The results were more interesting than I initially expected.

### Memory

Memory is where sparse attention showed a clear advantage at every tested sequence length.

| N | Dense time (s) | Sliding time (s) | Block-sparse time (s) | Dense peak MB | Sliding peak MB | Block-sparse peak MB |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.00059 | 0.00439 | 0.00426 | 30.9 | 12.2 | 13.2 |
| 1024 | 0.00209 | 0.00867 | 0.00865 | 96.1 | 14.2 | 15.2 |
| 2048 | 0.00743 | 0.01685 | 0.01700 | 354.1 | 18.2 | 19.2 |
| 4096 | 0.02617 | 0.03418 | 0.03442 | 1380.1 | 26.2 | 27.2 |
| 8192 | 0.10450 | 0.08871 | 0.09585 | 5472.2 | 42.2 | 43.2 |
| 16384 | OOM | 0.13498 | 0.13552 | OOM | 74.2 | 75.2 |

Dense attention's memory grows very quickly with sequence length because the score matrix is quadratic in N. The sparse implementations stay much flatter because they never materialize that full score matrix.

At N = 8192, dense attention used about 5.47 GB of peak GPU memory in this benchmark, compared with about 42 MB for sliding-window attention and 43 MB for block-sparse attention.
Hence dense attention uses almost 130x the amount of compute than sliding window and block sparse attention.

At N = 16384, dense attention ran out of GPU memory, while both sparse variants still completed the forward pass.

This is the clearest practical result from the benchmark: sparse attention is not just an optimization of the same workload. It makes sequence lengths feasible that the dense implementation cannot run on the same GPU.

### Time

The timing result was less straightforward.

Dense attention was faster for the smaller sequence lengths:

- 512: dense 0.00059 s vs ~0.0043 s sparse
- 1024: dense 0.00209 s vs ~0.0087 s sparse
- 2048: dense 0.00743 s vs ~0.0169 s sparse
- 4096: dense 0.02617 s vs ~0.0342 s sparse

At N = 8192, the ordering changed:

- dense: 0.10450 s
- sliding window: 0.08871 s
- block sparse: 0.09585 s

At N = 16384, dense could not run, while sliding window and block sparse both completed in about 0.135 s.

I did not expect the sparse implementation to be slower at small N, but the result makes sense for this particular implementation. The sparse engine processes query blocks in a Python loop and launches many smaller GPU operations. Dense attention can hand a single large matrix multiplication to the GPU, and for moderate sequence lengths the GPU is very good at doing that efficiently.

So this implementation does **not** demonstrate "sparse attention is always faster." It demonstrates something more useful: there is a crossover. At small enough sequence lengths, the overhead of my simple block-gather implementation dominates. As the sequence gets longer, the quadratic cost of dense attention starts to matter enough that sparse attention catches up and then wins.

This is also a good reminder not to judge an algorithm only by its asymptotic complexity. The way it is implemented matters. A more optimized sparse kernel could have a very different wall-clock profile.

---

## 6. What I learned from the results

The most useful lesson from this experiment was that the benefit of sparsity is not a single number called "speedup."

There are really two separate questions:

**How much computation did I avoid?**

and

**How much overhead did my implementation introduce?**

The first question favors sparse attention as sequences become long. The second question hurt this implementation at shorter lengths because of Python-level block iteration and many small GPU launches.

The memory result was much cleaner than the timing result. Sparse attention reduced memory usage across the entire benchmark and allowed the experiment to continue to N = 16384, where dense attention hit OOM.

The quality experiment also made the trade-off more concrete. Sliding-window attention removes direct long-range connections, while BigBird-style attention keeps a small number of routes to distant information through global and random connections. In this small Tiny Shakespeare experiment, neither sparse variant showed a dramatic quality collapse, but the losses were not exactly identical either. That is the cost of throwing away attention edges: some information is genuinely no longer available through a direct connection.

The main takeaway for me is therefore:

> Sparse attention is not "the same attention, but faster." It is a different connectivity pattern with different computational and information trade-offs.

For short sequences, dense attention can still be the better engineering choice. For long sequences, reducing the amount of attention that has to be computed can become the difference between "runs" and "does not fit in memory."
