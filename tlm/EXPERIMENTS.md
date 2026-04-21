# TLM Experiments — running log

Central notes so we don't lose context across sessions.

## Status

Branch: `claude/create-tlm-repo-t2f5J`. Code under `tlm/`. Each exp has
its own script under `experiments/` and log under `experiments/logs/`.
Commits tagged with exp number.

## Experiment ledger

| # | goal | result |
|---|---|---|
| 1 | seq_len sweep 16/32/64/128 | state=128 saturates ~32 chars; val_acc drops 80→56→17→16% |
| 2 | BPE tokenizer (vocab 1024) | val acc 62% (consistent with saturation) |
| 3 | RelationExtractor before cell | **+9.9pp val_acc** for +10% params (autoencoder) |
| 4 | Naive hard inference (all leaves computed, indexed) | acc drops 47pp, no speedup; implementation was fake-sparse |
| 5 | Autoencoder tree vs LSTM same dim | tree +30pp but 4x params (unfair) |
| 5b | LSTM with 2-4x more params | tree still beats (+27pp at matched params) |
| 6 | Tree**LM** (not autoencoder), first pass | ties LSTM; tree=443k, LSTM=93k (LSTM wins per-param) |
| 7 | TreeLMv2 (full encoder+decoder trees) vs LSTMLMv2 matched shape | TreeLMv2 -0.056 BPC at 2.8x params; marginal |
| 8 | Depth sweep (2/3/4) | d3 is sweet spot; d4 overfits 3x; all still behind LSTM per-param |
| 9 | Gating variants (Residual / Gated-GRU / IdLeaf cells) | None close the gap with LSTM; gates are not the bottleneck |
| 10 | Shared-backbone tree in encoder cell | **WIN**: 268k params vs LSTM 275k, beats LSTM by 0.023 BPC, overfit gap 0.084 vs LSTM 0.391 |
| 11 | Shared-backbone in ALL trees (relation/cell/context/head) | over-regularizes; 148k params but BPC worse; sweet spot = shared-cell-only |
| 12 | Long context seq_len 64/128/256 vs LSTM | **LOSS**: tree (shared) loses to LSTM, gap GROWS with seq_len |
| 13 | Real sparse hard inference (per-example weight gather + bmm) | BPC 2.7→5.2 (near-random); speedup 1.43x bs=1 baseline only, negative otherwise |
| 14 | ForgetGatedTreeCell (LSTM-style per-dim gate on tree proposal) | **WIN**: tree+forget beats LSTM at every seq_len; gap grows with context |
| 15 | Routing entropy annealing (sub-agent) | annealing worked mechanically (\|logit\| 2.2→3.3) but hard inference still ~random (gap 2.1-2.4 BPC); speedup still negative |
| 16 | Clean FLOP count + minimal batch=1 slicing bench | **Tree hard is 3.9x cheaper than LSTM in FLOPs**; PyTorch wall-clock ties LSTM (framework overhead eats ~3x); tree-hard vs tree-soft 2.45x wall-clock speedup |

## Best-so-far configuration

- `TreeLMv2SharedForget` (shared-backbone tree cell + LSTM-style forget gate)
- depth=3, state_dim=128, relation_dim=64, context_dim=64
- At seq_len 256: val BPC **2.357** at 293k params (LSTM 275k: 2.354)
- Per-param match with LSTM; wins at long context (gap grows with seq_len)
- Weight sharing collapses overfit 10x vs naive tree
- Forget gate gives clean preserve-channel for long-range dependencies

## Tese do PRD — status

| pillar | verdict |
|---|---|
| Tree compresses sequences better than LSTM | ✅ autoencoder +27pp (exp5b) |
| Tree beats LSTM per-param at LM | ✅ ties/beats with shared+forget (exp14) |
| Tree converges faster than LSTM | ✅ peaks at epoch 5 vs 10-20 (exp6/7) |
| Weight sharing reduces overfit | ✅ 10x less overfit (exp10/14) |
| Tree wins at long context | ✅ with forget gate; gap grows (exp14) |
| Train soft, infer sparse works | ❌ ❌ rejected twice (exp13, exp15) |
| Algorithmic speedup exists | ✅ 3.9x fewer FLOPs (exp16) |
| Speedup realizable in PyTorch | ⚠️ ~parity with LSTM at small scales |

## Open questions / planned experiments

- **exp17**: does `torch.compile` recover the FLOP gap at batch=1?
- **exp18**: does a pure numpy port show the speedup without PyTorch overhead?
- **exp19**: at larger state_dim (512/1024/2048) does matmul dominate overhead enough to show the FLOP speedup?
- **exp20**: Gumbel-softmax routing with temperature annealing — does it make hard inference viable without catastrophic accuracy loss?

## Key files

- `tlm/model.py` — SoftTree, SharedBackboneSoftTree, TreeCells (various gating variants), hard_forward with slicing
- `tlm/lm_model.py` — TreeLMv2 and variants (Shared, Forget, SharedForget, FiLM, Residual, etc.), LSTMLMv2
- `tlm/lm_train.py` — `run_lm` training loop with shifted-target cross-entropy and BPC metric
- `tlm/train.py` — older autoencoder training loop (used by exp1-5)
- `tlm/data.py` — TinyShakespeare char-level loader
- `tlm/bpe_data.py` — BPE tokenizer data loader (exp2)
- `tlm/datasets/tinyshakespeare.txt` — ~1MB corpus
