"""Experiment 10: weight sharing in the encoder cell tree.

The hypothesis from exp9: LSTM wins because its single shared matmul forces
weight sharing across inputs, while our tree has independent per-leaf Linears.
Test two weight-sharing schemes:

  - TreeLMv2Shared: shared backbone (one Linear + GELU) + per-leaf output map.
    Analog of 2-layer MLP with routing between the layers.
  - TreeLMv2FiLM: single shared Linear + per-leaf (gamma, beta) modulation.
    Strongest sharing: leaves can only scale/shift the same base output.

Depth 3 (sweet spot from exp8). Compare vs baseline TreeLMv2 and LSTMLMv2."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lm_train import LMConfig, run_lm
from lm_model import TreeLMv2, TreeLMv2Shared, TreeLMv2FiLM, LSTMLMv2


def make_cfg(name):
    return LMConfig(seq_len=64, state_dim=128, depth=3, relation_dim=64,
                    context_dim=64, epochs=30, n_train=4000, name=name)


variants = [
    ("baseline (per-leaf Linear)", TreeLMv2, "tree_baseline"),
    ("shared backbone + low-rank", TreeLMv2Shared, "tree_shared"),
    ("shared + FiLM modulation",   TreeLMv2FiLM, "tree_film"),
    ("LSTMLMv2 reference",         LSTMLMv2, "lstm_v2"),
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
