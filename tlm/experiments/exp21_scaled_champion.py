"""Experiment 21: scale up the TLM champion.

exp14's TreeLMv2SharedForget tied LSTM at matched-params per-seq_len=256.
Now scale to ~700k params and longer training, with LSTM matched, to
see if the architecture advantage grows or shrinks with scale.

Config: state_dim=256, seq_len=128, 40 epochs, n_train=5000. Larger than
anything we've trained so far.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lm_train import LMConfig, run_lm
from lm_model import TreeLMv2SharedForget, LSTMLMv2


def make_cfg(name):
    return LMConfig(seq_len=128, state_dim=256, depth=3, relation_dim=64,
                    context_dim=64, n_train=4000, n_val=500, batch_size=32,
                    epochs=25, lr=3e-3, grad_clip=5.0, log_every=5, name=name)


print("=== TLM champion scaled (TreeLMv2SharedForget) ===")
cfg_tree = make_cfg("tree_shared_forget_256")
r_tree = run_lm(cfg_tree, model_ctor=TreeLMv2SharedForget)

print("\n=== LSTM matched-param ===")
cfg_lstm = make_cfg("lstm_v2_256")
r_lstm = run_lm(cfg_lstm, model_ctor=LSTMLMv2)

print("\n=== SUMMARY ===")
print(f"{'model':<30} {'params':>10} {'best_vbpc':>11} {'val_acc':>10} {'time':>8}")
for r in (r_tree, r_lstm):
    print(f"{r['name']:<30} {r['n_params']:>10,} {r['best_val_bpc']:>11.3f} "
          f"{r['best_val_acc']:>10.3f} {r['time_s']:>7.0f}s")
print(f"\ndelta tree-lstm: {r_tree['best_val_bpc'] - r_lstm['best_val_bpc']:+.3f} BPC")
