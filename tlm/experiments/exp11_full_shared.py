"""Experiment 11: apply shared-backbone to ALL trees in the model, not just
the encoder cell. exp10 showed per-param wins with shared in the cell only;
now test full shared - relation, cell, context, head all use SharedBackbone.

If weight sharing is the whole story, this should further drop params and
improve generalization. If it plateaus, cell was the bottleneck."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lm_train import LMConfig, run_lm
from lm_model import TreeLMv2, TreeLMv2Shared, TreeLMv2FullShared, LSTMLMv2


def make_cfg(name):
    return LMConfig(seq_len=64, state_dim=128, depth=3, relation_dim=64,
                    context_dim=64, epochs=30, n_train=4000, name=name)


variants = [
    ("baseline (per-leaf)",        TreeLMv2, "tree_baseline"),
    ("shared cell only",           TreeLMv2Shared, "tree_shared_cell"),
    ("shared everywhere",          TreeLMv2FullShared, "tree_full_shared"),
    ("LSTM reference",             LSTMLMv2, "lstm_v2"),
]

results = []
for desc, ctor, name in variants:
    print(f"\n=== {desc} ===")
    r = run_lm(make_cfg(name), model_ctor=ctor)
    results.append((desc, r))

print("\n=== SUMMARY ===")
print(f"{'variant':<32} {'params':>10} {'best_vbpc':>11} {'val_acc':>10}")
for desc, r in results:
    print(f"{desc:<32} {r['n_params']:>10,} {r['best_val_bpc']:>11.3f} "
          f"{r['best_val_acc']:>10.3f}")
