"""Experiment 8: depth sweep for TreeLMv2. Before reaching for regularization,
test if the overfit in exp7 is just overparameterization. Smaller trees =
fewer leaves = less capacity = maybe less overfit."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lm_train import LMConfig, run_lm
from lm_model import TreeLMv2


results = []
for depth in [2, 3, 4]:
    cfg = LMConfig(seq_len=64, state_dim=128, depth=depth, relation_dim=64,
                   context_dim=64, epochs=30, n_train=4000,
                   name=f"tree_lm_v2_d{depth}")
    r = run_lm(cfg, model_ctor=TreeLMv2)
    results.append(r)
    print()

print("\n=== SUMMARY ===")
print(f"{'depth':<8} {'params':>10} {'val_bpc':>10} {'val_acc':>10} {'time':>8}")
for r in results:
    depth_str = r["name"].split("_d")[-1]
    print(f"d{depth_str:<7} {r['n_params']:>10,} {r['best_val_bpc']:>10.3f} "
          f"{r['best_val_acc']:>10.3f} {r['time_s']:>7.0f}s")
