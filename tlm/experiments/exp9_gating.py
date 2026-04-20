"""Experiment 9: test three keep/forget mechanisms in the encoder cell:
  - baseline (TreeLMv2): full rewrite each step via tree output
  - residual (TreeLMv2Residual): state + tree_delta, preserve by default
  - gated (TreeLMv2Gated): GRU-style g*state + (1-g)*tree_out
  - identity-leaf (TreeLMv2IdLeaf): one leaf is identity on state

Depth 3 from exp8 is the sweet spot; use it for all variants. LSTMLMv2
for reference."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lm_train import LMConfig, run_lm
from lm_model import (
    TreeLMv2, TreeLMv2Residual, TreeLMv2Gated, TreeLMv2IdLeaf, LSTMLMv2,
)


def make_cfg(name):
    return LMConfig(seq_len=64, state_dim=128, depth=3, relation_dim=64,
                    context_dim=64, epochs=30, n_train=4000, name=name)


variants = [
    ("baseline (full rewrite)", TreeLMv2, "tree_baseline"),
    ("residual (state + delta)", TreeLMv2Residual, "tree_residual"),
    ("gated (GRU-style)",        TreeLMv2Gated, "tree_gated"),
    ("identity-leaf",            TreeLMv2IdLeaf, "tree_idleaf"),
    ("LSTMLMv2 reference",       LSTMLMv2, "lstm_v2"),
]

results = []
for desc, ctor, name in variants:
    print(f"\n=== {desc} ===")
    r = run_lm(make_cfg(name), model_ctor=ctor)
    results.append((desc, r))

print("\n=== SUMMARY ===")
print(f"{'variant':<30} {'params':>10} {'val_bpc':>10} {'val_acc':>10} {'final_vbpc':>11}")
for desc, r in results:
    # include final-epoch val_bpc to reveal overfitting degree
    print(f"{desc:<30} {r['n_params']:>10,} {r['best_val_bpc']:>10.3f} "
          f"{r['best_val_acc']:>10.3f} {'-':>11}")
