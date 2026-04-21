"""Experiment 23: scale up to WikiText-2 (10x more data than Shakespeare).

WikiText-2 is ~10MB (vs Shakespeare 1MB), byte-level vocab ~178. Tests
whether the exp22 surprise (transformer loses to LSTM/TLM) was a
small-data artifact or holds at more realistic data scale.

Config matched to exp21/22: ~700k params, seq_len=128, 20 epochs,
n_train=6000 (more unique windows available).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lm_train import LMConfig, run_lm
from lm_model import TreeLMv2SharedForget, LSTMLMv2
from wikitext_data import make_wikitext_splits
from experiments.exp22_transformer import TransformerLM


def make_cfg(name):
    return LMConfig(seq_len=128, state_dim=256, depth=3, relation_dim=64,
                    context_dim=64, n_train=6000, n_val=800, batch_size=32,
                    epochs=20, lr=3e-3, grad_clip=5.0, log_every=5, name=name)


import os
# allow running a subset via env var (e.g. EXP23_ONLY=lstm,transformer)
only = os.environ.get("EXP23_ONLY", "").lower()
all_variants = [
    ("TreeLMv2SharedForget", TreeLMv2SharedForget),
    ("LSTMLMv2",             LSTMLMv2),
    ("TransformerLM",        TransformerLM),
]
if only:
    keep = [v for v in only.split(",") if v]
    variants = [(n, c) for n, c in all_variants
                if any(k in n.lower() for k in keep)]
else:
    variants = all_variants

results = []
for label, ctor in variants:
    print(f"\n=== {label} on WikiText-2 ===")
    r = run_lm(make_cfg(label), model_ctor=ctor, data_fn=make_wikitext_splits)
    results.append((label, r))

print("\n=== SUMMARY (WikiText-2, byte-level, ~700k params) ===")
print(f"{'model':<26} {'params':>10} {'best_vbpc':>11} {'val_acc':>10} {'time':>8}")
for label, r in results:
    print(f"{label:<26} {r['n_params']:>10,} {r['best_val_bpc']:>11.3f} "
          f"{r['best_val_acc']:>10.3f} {r['time_s']:>7.0f}s")
