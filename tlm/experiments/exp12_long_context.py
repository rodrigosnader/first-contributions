"""Experiment 12: long context. State-based models should pay off at
longer sequences (where attention gets expensive and LSTM's limited
hidden state might saturate). Use TreeLMv2Shared (exp10 winner) vs
LSTMLMv2 at seq_len 64, 128, 256.

At seq_len=256, LSTM has to compress 256 chars of context into 128
dims of hidden state; same constraint on TLM. If TLM's tree routing
helps organize longer contexts, gap should grow with seq_len."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lm_train import LMConfig, run_lm
from lm_model import TreeLMv2Shared, LSTMLMv2


# Reduce epochs at long seq_len since each step costs more CPU
def make_cfg(seq_len, ctor, name):
    epochs = {64: 20, 128: 15, 256: 10}[seq_len]
    return LMConfig(seq_len=seq_len, state_dim=128, depth=3,
                    relation_dim=64, context_dim=64,
                    epochs=epochs, n_train=3000, name=name)


results = []
for seq_len in [64, 128, 256]:
    for label, ctor in [("tree_shared", TreeLMv2Shared), ("lstm", LSTMLMv2)]:
        name = f"{label}_seq{seq_len}"
        print(f"\n=== {name} ===")
        r = run_lm(make_cfg(seq_len, ctor, name), model_ctor=ctor)
        results.append((seq_len, label, r))

print("\n=== SUMMARY ===")
print(f"{'seq_len':>8} {'model':<14} {'params':>10} {'best_vbpc':>11} {'val_acc':>10}")
for seq_len, label, r in results:
    print(f"{seq_len:>8} {label:<14} {r['n_params']:>10,} {r['best_val_bpc']:>11.3f} "
          f"{r['best_val_acc']:>10.3f}")

print("\n=== TREE vs LSTM delta per seq_len ===")
by_seq = {}
for seq_len, label, r in results:
    by_seq.setdefault(seq_len, {})[label] = r
for seq_len, d in sorted(by_seq.items()):
    delta = d["tree_shared"]["best_val_bpc"] - d["lstm"]["best_val_bpc"]
    print(f"  seq_len={seq_len}: tree {d['tree_shared']['best_val_bpc']:.3f} "
          f"vs lstm {d['lstm']['best_val_bpc']:.3f} -> delta {delta:+.3f}")
