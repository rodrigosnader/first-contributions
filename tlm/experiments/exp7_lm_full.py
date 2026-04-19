"""Experiment 7: TreeLMv2 (trees on both sides, mirroring autoencoder
structure) vs LSTMLMv2 (2-layer MLPs in the same positions). Fair
architectural comparison - only the inner operator changes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lm_train import LMConfig, run_lm
from lm_model import TreeLMv2, LSTMLMv2


print("=== TreeLMv2 (trees both sides) ===")
cfg_tree = LMConfig(seq_len=64, state_dim=128, depth=4, relation_dim=64,
                    epochs=30, n_train=4000, name="tree_lm_v2")
r_tree = run_lm(cfg_tree, model_ctor=TreeLMv2)

print("\n=== LSTMLMv2 (MLPs both sides, same shape) ===")
cfg_lstm = LMConfig(seq_len=64, state_dim=128, relation_dim=64,
                    epochs=30, n_train=4000, name="lstm_lm_v2")
r_lstm = run_lm(cfg_lstm, model_ctor=LSTMLMv2)

print("\n=== SUMMARY ===")
print(f"{'model':<20} {'params':>10} {'val_bpc':>10} {'val_acc':>10} {'time':>8}")
for r in (r_tree, r_lstm):
    print(f"{r['name']:<20} {r['n_params']:>10,} {r['best_val_bpc']:>10.3f} "
          f"{r['best_val_acc']:>10.3f} {r['time_s']:>7.0f}s")
print(f"\ndelta tree - lstm: {r_tree['best_val_bpc'] - r_lstm['best_val_bpc']:+.3f} BPC")
