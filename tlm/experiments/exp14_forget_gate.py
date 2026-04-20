"""Experiment 14: LSTM-style per-dim forget gate on top of tree proposal.

exp12 showed the tree loses to LSTM at long context and the gap grows
with seq_len. Hypothesis: LSTM has a clean 'preserve' channel via
c_t = f * c_{t-1} + ... with per-dim sigmoid forget. Our tree has no
such channel - LayerNorm erases slow drift, tree output fully rewrites
state.

Test: ForgetGatedTreeCell = f * state_{t-1} + (1-f) * norm(tree_out).
f is per-dim. Normalize ONLY the tree proposal, not the final mix
(that was the bug in exp9 GatedTreeCell). Forget bias init to 2.0
so gates start ~0.88 (preserve by default)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lm_train import LMConfig, run_lm
from lm_model import (
    TreeLMv2Shared,               # exp12 contender (no forget gate)
    TreeLMv2Forget,               # baseline tree + forget gate
    TreeLMv2SharedForget,         # shared backbone + forget gate
    LSTMLMv2,                     # reference
)


def make_cfg(seq_len, name):
    epochs = {64: 20, 128: 15, 256: 10}[seq_len]
    return LMConfig(seq_len=seq_len, state_dim=128, depth=3,
                    relation_dim=64, context_dim=64,
                    epochs=epochs, n_train=3000, name=name)


variants = [
    ("shared (no forget)",      TreeLMv2Shared),
    ("baseline + forget gate",  TreeLMv2Forget),
    ("shared + forget gate",    TreeLMv2SharedForget),
    ("LSTM reference",          LSTMLMv2),
]

results = {}
for seq_len in [64, 128, 256]:
    print(f"\n{'=' * 40}\nseq_len={seq_len}\n{'=' * 40}")
    for label, ctor in variants:
        name = f"{label.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '')}_seq{seq_len}"
        print(f"\n--- {label} @ seq_len={seq_len} ---")
        r = run_lm(make_cfg(seq_len, name), model_ctor=ctor)
        results.setdefault(seq_len, {})[label] = r

print("\n=== SUMMARY ===")
print(f"{'seq_len':>8} {'model':<28} {'params':>10} {'best_vbpc':>11} {'val_acc':>10}")
for seq_len in sorted(results):
    for label, _ in variants:
        r = results[seq_len][label]
        print(f"{seq_len:>8} {label:<28} {r['n_params']:>10,} "
              f"{r['best_val_bpc']:>11.3f} {r['best_val_acc']:>10.3f}")

print("\n=== TREE vs LSTM delta per seq_len (best_val_bpc - lstm_best_val_bpc) ===")
print("(negative = tree wins)")
for seq_len in sorted(results):
    lstm_bpc = results[seq_len]["LSTM reference"]["best_val_bpc"]
    for label, _ in variants:
        if label == "LSTM reference": continue
        delta = results[seq_len][label]["best_val_bpc"] - lstm_bpc
        print(f"  seq_len={seq_len:3d}  {label:<28}  delta={delta:+.3f}")
