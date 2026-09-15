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

A note on how I worked: I used an AI assistant heavily while writing the code, which the task brief explicitly allows. `models/model.py` is the part I wrote from my own understanding, based on what I had learned about attention and transformer blocks. For the rest, I spent a lot of time reading through it, running pieces of it by hand on small examples, and figuring out why it was written the way it was. The analysis in this writeup — including the problems I found in my own benchmark and training setup, described below — is mine.

---

## 1. Implementation

The code is split into a few simple pieces.

`dense_attention()` is the reference implementation. It computes the complete Q·Kᵀ matrix, applies a boolean mask, performs a safe softmax, and multiplies the attention weights by V.

The sparse implementation is built around two ideas. First, `build_block_pattern()` decides which key blocks each query block is allowed to access. Second, `block_sparse_attention()` actually gathers those K/V blocks and runs attention on them.

This separation turned out to be important. A sparse pattern is not useful for performance if the program still computes the full dense N×N matrix first. The block-level routing is what lets the implementation avoid those unnecessary scores.

For sliding-window attention, I use a deliberately safe superset of nearby blocks and then apply a finer token-level validity function to trim it to the exact window. For BigBird-style attention, the pattern combines local blocks, global blocks, and randomly selected blocks.

I also had to handle sequence lengths that are not multiples of the block size. Internally, the tensors are padded to whole blocks, but padded key positions are explicitly excluded. Fully masked rows are handled with a `safe_softmax` so they produce zero attention/output instead of NaNs.

### Why the NaN fix is done before the softmax, not after

The obvious way to deal with the NaN is to run the softmax and then replace any NaN in the output with zero. I did not do that, and the reason is worth stating.

If a whole row is `-inf`, softmax produces NaN. Cleaning it up afterwards fixes the numbers you can see in the forward pass, but the NaN has already been produced inside the graph, and during training the gradient flowing back through that row is NaN too. That poisons the parameters even though the forward output looked fine.

So instead, `safe_softmax` detects rows with no valid keys *before* the softmax and fills those rows with `0.0` instead of `-inf`. Softmax of an all-zero row is a uniform distribution, which is meaningless but finite. Then the second `masked_fill` zeroes those entries out, so the row contributes nothing to the output and nothing NaN ever exists in the graph.

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

One honest limitation of this harness. For the sliding-window tests, the reference mask is built by `sliding_window_full_mask()`, which is written independently of the pattern builder — so that test really does check both halves of the sparse path. For the BigBird tests, the reference mask is built from the pattern that `build_block_pattern()` returned. That means those tests check the gather-and-compute engine properly, but they cannot catch a bug in the pattern builder itself, because a wrong pattern would produce an equally wrong reference. If I had more time I would build the BigBird reference mask independently too.

---

## 3. Character-level GPT experiment

For the quality experiment, I trained the same small character-level GPT with each attention variant on Tiny Shakespeare. The model has 2 transformer layers, 4 attention heads and 64-dimensional embeddings. The training script uses a context length of 128 and compares the three attention mechanisms while keeping the model architecture and optimizer setup the same.

All three models learned the task normally. Rather than quote a single step, which bounces around a lot, here is the mean validation loss over the last five evaluations that all three runs share (steps 2700 to 3100):

| Attention | Mean val loss, steps 2700–3100 | Best single eval |
|---|---:|---:|
| Dense | 1.6911 | 1.6632 |
| Sliding window | 1.6838 | 1.6527 |
| Block sparse | 1.6995 | 1.6657 |

The whole spread is about 0.016, and the ordering flips depending on which step you look at. I ran one seed each, so I do not think any of these differences mean anything. The block-sparse run also continued to step 3300 while the other two stopped at 3100, which is another reason not to read the final numbers directly against each other.

### Why this experiment could not have found a degradation

When I set this up, I expected the sparse variants to be at least a little worse and was mildly surprised when they weren't. Looking at the hyperparameters again, I don't think that result means what I wanted it to mean.

The training context length is 128 tokens. The sliding window is 32 tokens, and the block-sparse config uses a block size of 16 with 1 global block and 2 random blocks, on a sequence that is only 8 blocks long. So each query can already see a large fraction of the context, and the part it cannot see is 100 characters back in a character-level task where almost all of the useful signal is a few characters away anyway.

In other words, I did not fail to find a quality cost. I built an experiment that could not have found one. The patterns barely removed anything that mattered at this scale.

If I ran this again, I would use a much longer context — 1024 or more — with a window that is a genuinely small fraction of it, and I would use a task where the dependency is definitely long-range rather than character-level text. The needle-in-a-haystack style setup used in the KV-cache literature would be a better fit, because it tests whether one specific distant fact survived, rather than averaging over a loss where local prediction dominates.

### A bug I found in the block-sparse training run

While going back over `model.py` I found something wrong with how the random pattern is generated.

The generator is created once in `__init__`:

```python
self._pattern_generator = torch.Generator().manual_seed(1234)
```

and then passed into the pattern builder on every forward pass. But a generator advances its internal state every time you draw from it. So step 1 draws one set of random blocks, step 2 draws a different set, step 3 a different set again, and eval batches do the same.

The run is still reproducible — rerun it and you get exactly the same sequence of patterns — but the pattern is not *fixed*, which is what BigBird actually specifies. What I trained is closer to a model whose random connections are resampled every step, a bit like a routing-level dropout.

The fix is to store the seed instead of the generator and build a freshly seeded one inside `forward`:

```python
gen = torch.Generator().manual_seed(self._pattern_seed)
```

I found this too late to retrain, so the block-sparse quality numbers above should be read with this in mind. Given that all patterns have the same density, I would expect the effect on loss to be small, but I have not verified that. The benchmark has a related issue: it calls the BigBird wrapper with no generator at all, so it falls back to the global RNG and uses a different pattern on each call. That does not affect timing, since the amount of work is the same either way.

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

Another way to see it is in terms of how many attention layers it takes for information to travel. With a pure sliding window of size w, information moves at most w positions per layer, so two tokens N apart need roughly N/w layers before they can influence each other at all. A global token collapses that to two hops: everything writes into the global position, and everything reads back out of it. With only 2 layers in my model, that difference is large.

I did not run a separate ablation that varies the number of global blocks, so I am treating this as an explanation of the pattern rather than a measured effect from my experiment. It is the ablation I would run first if I continued.

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

At N = 8192, dense attention used about 5.47 GB of peak GPU memory, compared with about 42 MB for sliding-window and 43 MB for block-sparse. That is roughly **130× the peak memory**, not 130× the compute — the timing column right next to it shows the three variants within about 15% of each other at that length, so the two numbers are measuring completely different things.

At N = 16384, dense attention ran out of GPU memory, while both sparse variants still completed the forward pass.

### How much of that 130× is real?

I wanted to check whether the 130× was actually the quadratic scaling or partly my own implementation, so I worked out what dense *should* use.

At N = 8192 with B=1, H=4, in fp32, one N×N score tensor is:

```
4 heads × 8192 × 8192 × 4 bytes = 1024 MB
```

So the score matrix itself is 1 GB. Measured peak was 5472 MB. The gap is `safe_softmax`, which does everything out of place:

```python
masked_scores = scores.masked_fill(...)          # copy 2
safe_scores   = masked_scores.masked_fill(...)   # copy 3
attn          = F.softmax(safe_scores, dim=-1)   # copy 4
attn          = attn.masked_fill(...)            # copy 5
```

That is five live 1024 MB tensors = 5120 MB, plus a couple of hundred MB for the boolean masks being materialized at full shape by `~valid_mask`. Which lands almost exactly on the 5472 MB I measured.

So the honest breakdown of the 130× is:

- about **24×** is inherent — 1024 MB of score matrix against 43 MB of gathered blocks
- the rest is my dense baseline making five copies of a tensor it only needed one or two of

An in-place version would look like this:

```python
def safe_softmax(scores, valid_mask, inplace=False):
    valid_mask = valid_mask.expand_as(scores)
    invalid = ~valid_mask
    row_has_valid = valid_mask.any(dim=-1, keepdim=True)

    s = scores if inplace else scores.clone()
    s.masked_fill_(invalid, float('-inf'))
    s.masked_fill_(~row_has_valid, 0.0)

    attn = torch.softmax(s, dim=-1)
    attn.masked_fill_(invalid, 0.0)
    return attn
```

`inplace=True` is safe inside `dense_attention` because `scores` comes straight out of the matmul and nothing else holds a reference to it. That brings peak down to roughly scores + attn + one boolean mask, about 2.3 GB instead of 5.5 GB.

This matters for the headline result. At N = 16384 the score matrix is 4 GB, so a memory-tight dense implementation would need something like 9 GB, which probably **would** fit on a 15 GB T4. In other words the OOM I reported is partly a property of my baseline rather than a hard wall of the algorithm. The sparse advantage is real, but at these lengths it is closer to 25× than 130×, and dense dies later than my table suggests.

I did not rerun the benchmark with the fixed version before the deadline, so the numbers in the table are the ones I actually measured with the code as submitted.

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

### Something wrong with how I timed it

Going back over `benchmarks.py`, the timing comparison is not as fair as I thought, and it is unfair in the direction of making sparse look worse.

For dense, the causal mask is built once, *outside* the timed function:

```python
mask = causal_mask(N, device=DEVICE)
fn = lambda: dense_attention(Q, K, V, mask=mask)[0]
```

For the sparse variants, `fn` calls the wrapper functions, and those call `build_block_pattern()` internally — so the block pattern is rebuilt on **every timed iteration**. At N = 16384 with block size 64 that is 256 query blocks, each running a Python list comprehension over 256 candidates, plus a `randperm` for the BigBird case. All of that is CPU work with no GPU involvement, and I timed it as if it were part of attention.

There is a smaller version of the same problem inside `block_sparse_attention`, where the gather indices `k_idx` are rebuilt with `torch.cat([torch.arange(...)])` for every query block on every call. A real implementation would compute those once and reuse them.

The fix is to build the pattern once and pass it straight to `block_sparse_attention` inside the timed lambda. I did not rerun the benchmark with this change before the deadline, so the numbers above stand as measured, but the correct reading is:

- the **memory** results are unaffected, since none of this allocates GPU memory
- the **timing** results overstate the sparse cost at small N, and the real crossover point is earlier than 8192

I would rather report this than quietly present a comparison I now know is tilted.

---

## 6. What I learned from the results

The most useful lesson from this experiment was that the benefit of sparsity is not a single number called "speedup."

There are really two separate questions:

**How much computation did I avoid?**

and

**How much overhead did my implementation introduce?**

The first question favors sparse attention as sequences become long. The second question hurt this implementation at shorter lengths because of Python-level block iteration and many small GPU launches — and, as it turns out, because of measurement overhead I built into the benchmark itself.

The memory result was much cleaner than the timing result. But even there, working out the arithmetic afterwards showed that a good chunk of my dense baseline's memory was avoidable copies rather than the algorithm. The score matrix is what makes dense attention expensive; my implementation made it about five times more expensive than it needed to be.

The quality experiment taught me something different and slightly uncomfortable: it is easy to run an experiment, get a clean-looking result, and not notice that the setup could never have produced the result you were looking for. A 32-token window on a 128-token context does not test long-range information loss.

The main takeaway for me is therefore:

> Sparse attention is not "the same attention, but faster." It is a different connectivity pattern with different computational and information trade-offs — and measuring those trade-offs correctly is a separate skill from implementing them.

For short sequences, dense attention can still be the better engineering choice. For long sequences, reducing the amount of attention that has to be computed can become the difference between "runs" and "does not fit in memory."

---

## 7. What I would do next

In rough priority order:

1. Rerun the benchmark with the pattern built outside the timed region, and with the in-place `safe_softmax`, to get a fair time curve and an honest OOM point for dense.
2. Fix the resampling generator and retrain block-sparse.
3. Rerun the quality experiment at context length 1024+ with a window that is a genuinely small fraction of it, so the comparison can actually fail.
4. Ablate the number of global blocks, which is the one claim in this writeup I explained but did not measure.
5. Batch the query-block loop instead of iterating in Python, which is the main reason sparse loses at small N.
