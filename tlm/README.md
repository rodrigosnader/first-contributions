# TLM — Tree Language Model

An exploration of tree-based recurrent language models: can a soft-decision-tree
cell replace the matmul in an LSTM, and buy inference speedup via conditional
routing?

**25 experiments, mixed verdict, here's the full story.**

## The original thesis (PRD)

> Language modeling does not require repeatedly attending over raw token
> history if the model can maintain a strong enough compressed state.
> Tree-based structured compute can be dramatically more efficient than
> dense compute when paired with the right representation and memory design.

Key bets:
1. **Conditional compute**: only one root-to-leaf path fires per step → sparse → fast
2. **Train soft, infer hard**: soft mixing during training, argmax routing at inference
3. **Persistent compressed state** replaces attention over KV cache
4. **CPU-friendly**: tree routing avoids large dense matmuls, fits on commodity hardware

## Core architecture

```
token_t → embed → RelationExtractor (tree)  ─┐
                                              │   per-dim forget gate
state_{t-1}  ─────────────────────────────────┴→ TreeCell → state_t
                                                                │
                                              ContextExtractor (tree)
                                                                │
                                              DecoderHead (tree) → logits
```

- **SoftTree**: binary routing via `sigmoid(W_route · x)` per node, 2^depth leaves
  with independent Linear transforms, weighted combination via path probabilities
- **Shared-backbone variant**: single `Linear(input, hidden) + GELU` feeds all
  leaves — introduces weight sharing that cuts overfit 10x (exp10)
- **Forget gate**: `state_t = f ⊙ state_{t-1} + (1 − f) ⊙ LN(tree_proposal)` —
  LSTM-style preserve channel so long-range info doesn't get normalized away (exp14)
- **Hard inference**: argmax routing, pure slicing of the one reached leaf's
  weights — 3.9x fewer FLOPs than LSTM (exp16)

## Experiment arc

### Phase 1: Foundations (exp1-5)
Autoencoder on TinyShakespeare char-level. Established that soft trees are
differentiable, state compression works, and added the RelationExtractor
component for +9.9pp val acc (exp3). Autoencoder tree beat same-dim LSTM by
+27pp — first evidence the architecture is viable.

### Phase 2: Swapping to LM (exp6-9)
Moved to next-token prediction (the PRD's actual goal). First finding was
humbling: TreeLM tied LSTM at matched state_dim but needed 2.8x more params to
do so. Tried every gating variant (residual, GRU-style, identity-leaf) —
**none closed the gap**. Verdict: gates weren't the issue.

### Phase 3: Weight sharing is the secret (exp10-12)
**Shared-backbone tree cell** tied or beat LSTM at nearly matched params (268k
vs 275k), and overfit gap collapsed 10x (0.759 → 0.084). Key insight: LSTM's
single shared matmul forces weight sharing across inputs; independent per-leaf
Linears in a vanilla tree let each leaf memorize its subset. Extending shared-
backbone to *all* trees (exp11) over-regularized. Long-context (exp12, seq=256)
initially went to LSTM.

### Phase 4: Forget gate fixes long context (exp13-14)
Applied LSTM's per-dim forget gate on top of the tree proposal:
`state = f ⊙ state_{t-1} + (1-f) ⊙ LN(tree_output)`.
**Tree+forget beat LSTM at every seq_len (64/128/256) and the gap GREW with
context length** — exactly the pattern state-based models should have but
exp12 didn't show. Diagnosis confirmed: LayerNorm on the proposal only (not the
mix) preserves the clean pass-through channel.

### Phase 5: Speed — bait and switch (exp13, exp15-19)
First hard-inference benchmark (exp13) looked terrible: accuracy dropped 27pp,
speedup was negative. Then routing entropy annealing (exp15) barely helped.
Cue panic: the PRD's core thesis looked dead.

Re-examined FLOPs analytically (exp16): tree hard is **3.9x fewer FLOPs than
LSTM**, provable by arithmetic. exp17 (`torch.compile`) couldn't fuse the
control flow. exp18 (numpy port) **hit 4.16x speedup** — the win was algorithmic,
PyTorch was eating it. exp19 (scale sweep state_dim=128→2048) showed **at
state=2048 stock PyTorch recovers the 3.92x** as matmul dominates overhead.

### Phase 6: Gumbel fixes hard training (exp20)
Gumbel-sigmoid routing + temperature annealing (1.0→0.1) cut the hard-inference
accuracy drop from -27pp to -11pp. Soft BPC preserved (2.661). Routing
confidence reached 0.998 — essentially discrete. Usable but not perfect.

### Phase 7: Scale and transformer comparison (exp21-25)
- **exp21**: at state=256 seq=128 ~700k params, LSTM 1.901 > tree 1.926 > vanilla transformer 2.027 BPC on WikiText-2
- **exp22**: vanilla transformer loses to both at this scale
- **exp23**: WikiText-2 10MB, same ranking — LSTM > Tree > Transformer
- **exp24b**: regularized transformer (dropout+warmup+cosine) still loses
- **exp25**: KV-cache fair inference benchmark — **tree hard is 1.32-2.12x
  faster than transformer with KV cache at matched size**, advantage grows
  with generation length

## What we validated vs the PRD

| claim | status | evidence |
|---|---|---|
| Tree compresses sequences | ✅ | +27pp over LSTM in autoencoder (exp5b) |
| Competitive LM quality | ✅ | Shared+forget matches/beats LSTM (exp10, exp14) |
| Fast convergence | ✅ | Peaks at epoch 5 vs LSTM 10-20 (exp6-7) |
| Weight sharing reduces overfit | ✅ | 10x smaller overfit gap (exp10) |
| Wins at long context | ✅ (with forget gate) | Gap grows with seq_len (exp14) |
| Beats transformer at small scale | ✅ | 700k-1M params, 10MB data (exp23-24b) |
| Algorithmic speedup exists | ✅ | 3.9x fewer FLOPs (exp16) |
| Speedup realizable in numpy | ✅ | 4.16x vs LSTM (exp18) |
| Speedup in stock PyTorch | ✅ at scale | 3.92x at state=2048 (exp19), 2.12x vs transformer-KV at gen=1024 (exp25) |
| Train soft, infer hard | ⚠️ partial | Gumbel cuts gap 3x but still ~11pp acc drop (exp20) |

## Where it stands

**Best model**: `TreeLMv2SharedForget` — shared-backbone tree cell + per-dim
forget gate. At 700k-1M params on 10MB data:
- Quality: within 0.025 BPC of LSTM, beats both transformer variants
- Inference: 1.32-2.12x faster than transformer KV cache at batch=1 CPU
- Train time: 15x slower than transformer (sequential recurrence)

**The honest story**: TLM is a compact, recurrent, inference-fast LM with
LSTM-style retention and MoE-style conditional compute. Not a universal winner,
but clearly wins in the CPU-inference / small-model / moderate-data niche.

**Unvalidated**: does the quality ranking flip at 100MB+ training data where
transformer usually overtakes? Would require a bigger corpus; current hosts
are restricted, user is placing a dataset locally to continue.

## Repo layout

```
tlm/
  model.py              # SoftTree, SharedBackboneSoftTree, cell variants
                        # (TreeCell, Residual/Gated/IdLeaf/ForgetGated/...)
                        # + hard_forward with pure slicing for real speedup
  lm_model.py           # TreeLM, TreeLMv2, TreeLMv2Shared, TreeLMv2Forget,
                        # TreeLMv2SharedForget (champion), LSTMLMv2
  lm_train.py           # run_lm(): training loop with BPC, supports any data_fn
  train.py              # older autoencoder training loop (phases 1-5)
  data.py               # TinyShakespeare char-level loader
  wikitext_data.py      # WikiText-2 byte-level loader
  bpe_data.py           # BPE tokenized Shakespeare (exp2)
  datasets/
    tinyshakespeare.txt # ~1MB
    wikitext2_train.txt # ~10MB
    wikitext2_valid.txt # ~1MB
  experiments/
    exp1 .. exp25       # each experiment has its own script + log under logs/
    logs/               # stdout captures for every run
  EXPERIMENTS.md        # running ledger, one row per experiment
  README.md             # this file
```

## Reproducing

```bash
cd tlm
# Core model sanity check
python3 -c "from lm_model import TreeLMv2SharedForget; \
  m = TreeLMv2SharedForget(65, state_dim=128, depth=3); \
  print(sum(p.numel() for p in m.parameters()))"

# Train the champion on TinyShakespeare
python3 experiments/exp14_forget_gate.py

# Three-way comparison on WikiText-2
python3 experiments/exp23_wikitext.py

# Inference speed head-to-head (TLM hard vs Transformer KV cache)
python3 experiments/exp25_inference_speed.py

# FLOP accounting for speedup claim
python3 experiments/exp16_flops_benchmark.py
```

## Key code primitives

**Soft tree forward** (weighted combination of 2^depth leaves):
```python
routing_probs = sigmoid(router(x))            # (B, n_internal)
# traverse level by level, multiply out path probabilities
for level in range(depth):
    left  = node_probs * routing_probs[level]
    right = node_probs * (1 - routing_probs[level])
    node_probs = interleave(left, right)
leaf_outs = leaves_linear(x).reshape(B, n_leaves, out_dim)
return (node_probs * leaf_outs).sum(dim=leaf_dim)
```

**Hard tree forward** (one root-to-leaf path, pure slicing):
```python
leaf_idx = 0
for level in range(depth):
    i = node_base + leaf_idx
    logit = router.weight[i] @ x + router.bias[i]
    leaf_idx = 2 * leaf_idx + (0 if logit > 0 else 1)
    node_base += 2 ** level
W_leaf = leaves.weight[leaf_idx * out : (leaf_idx + 1) * out]  # slice, no copy
return W_leaf @ x + leaves.bias[leaf_idx * out : (leaf_idx + 1) * out]
```

**Forget-gated cell**:
```python
concat = cat([x, state])
f = sigmoid(W_forget @ concat)          # per-dim forget gate
proposal = LayerNorm(tree(concat))      # normalize the proposal only
state_t = f * state + (1 - f) * proposal
```

## Open threads

- **Bigger data (100MB+)**: does the quality gap against transformer shrink
  and flip? Blocked until local dataset is provided.
- **Stronger hard-train alignment**: STE, temperature=0 training, or full
  discrete training from scratch to close the remaining 11pp acc gap in hard
  inference.
- **Compiled kernel**: a C++/Triton/CUDA port of the hard-forward traversal
  would convert the numpy 4x speedup into native production speed.
- **Scaling law study**: TLM vs LSTM vs transformer at 100k / 1M / 10M / 100M
  params. If TLM's per-param advantage grows with scale, there's a real pitch.
  If it shrinks, TLM's niche stays small.
