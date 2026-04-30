"""Experiment 32: Hierarchical vs Transformer at seq=1024.

exp30 at seq=256 was a tie - too short to expose attention's quadratic cost.
This runs the same comparison at seq=1024 where transformer's O(T^2) attention
is expensive (4x longer seq = 16x more attention compute per layer).
Hierarchical's O(T*W + T*Tc) cost grows linearly with T.

Hypothesis: at seq=1024, hierarchical wins quality OR matches with much
less compute. If transformer still wins, the architectural pitch fails.

Configs (~700k params, seq=1024):
- Transformer: full causal self-attention, 4 layers
- Hierarchical: chunk_size=32 (32 chunks of 32 tokens),
                local_window=64, 3 layers
"""
import sys
import math
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from data import make_shakespeare_splits
from lm_train import LMConfig, shifted_loss

import importlib.util, os
spec = importlib.util.spec_from_file_location(
    "exp30_mod",
    os.path.join(os.path.dirname(__file__), "exp30_hierarchical_ssm_attention.py"),
)
exp30 = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp30)
HierarchicalLM = exp30.HierarchicalLM
TransformerLM = exp30.TransformerLM
train_lm = exp30.train_lm


def main():
    torch.manual_seed(0)
    SEQ_LEN = 1024
    # Reduced n_train and epochs because seq=1024 is expensive on CPU
    train_data, val_data, stoi, _ = make_shakespeare_splits(2500, 300, SEQ_LEN)
    vocab = len(stoi)

    print("\n=== Transformer (full causal attn, seq=1024) ===")
    txf = TransformerLM(vocab, max_len=SEQ_LEN, d_model=128,
                        n_heads=4, n_layers=4, ff_mult=3)
    txf_params = sum(p.numel() for p in txf.parameters())
    print(f"params: {txf_params:,}")
    txf_bpc, txf_acc, txf_time = train_lm(
        txf, train_data, val_data, n_epochs=12, batch_size=8, label="transformer")

    print("\n=== Hierarchical (chunk=32, local=64, seq=1024) ===")
    hier = HierarchicalLM(vocab, max_len=SEQ_LEN, d_model=128, n_heads=4,
                          n_layers=3, ff_mult=3,
                          chunk_size=32, local_window=64)
    hier_params = sum(p.numel() for p in hier.parameters())
    print(f"params: {hier_params:,}")
    hier_bpc, hier_acc, hier_time = train_lm(
        hier, train_data, val_data, n_epochs=12, batch_size=8, label="hierarchical")

    print("\n=== SUMMARY (Shakespeare seq=1024, 12 epochs) ===")
    print(f"{'model':<22} {'params':>10} {'val_bpc':>9} {'val_acc':>9} {'time':>8}")
    print(f"{'transformer':<22} {txf_params:>10,} {txf_bpc:>9.3f} {txf_acc:>9.3f} {txf_time:>7.0f}s")
    print(f"{'hierarchical':<22} {hier_params:>10,} {hier_bpc:>9.3f} {hier_acc:>9.3f} {hier_time:>7.0f}s")

    delta = hier_bpc - txf_bpc
    speed_ratio = txf_time / hier_time
    print(f"\nval_bpc delta (hier - txf): {delta:+.3f}")
    print(f"train time ratio (txf / hier): {speed_ratio:.2f}x  "
          f"({'hier faster' if speed_ratio > 1 else 'txf faster'})")
    print(f"\nVerdict:")
    if delta < -0.02:
        print("  hierarchical WINS quality at long seq (architecture justified)")
    elif delta > 0.02:
        print("  transformer still wins - hierarchical loss did not invert at long seq")
    else:
        print("  effective tie - hierarchical's value at long seq must come from speed/compute")


if __name__ == "__main__":
    main()
