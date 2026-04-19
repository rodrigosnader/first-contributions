"""Experiment 1: vary seq_len, keep everything else fixed. Measure how
compression quality degrades as we ask the state to hold more characters."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train import Config, run

results = []
for seq_len in [16, 32, 64, 128]:
    cfg = Config(
        seq_len=seq_len,
        epochs=30,
        n_train=3000,
        n_val=500,
        name=f"seq{seq_len}",
    )
    r = run(cfg)
    results.append((seq_len, r["n_params"], r["best_val_acc"], r["final_train_acc"], r["time_s"]))
    print()

print("\n=== SUMMARY ===")
print(f"{'seq_len':>8} {'params':>10} {'train_acc':>10} {'val_acc':>10} {'time_s':>8}")
for sl, np, va, ta, t in results:
    print(f"{sl:>8d} {np:>10,} {ta:>10.3f} {va:>10.3f} {t:>8.0f}")
