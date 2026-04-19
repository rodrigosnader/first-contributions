"""Experiment 5b: fair LSTM baseline. Match param count (state_dim ~2x bigger)
and give LSTM enough epochs to converge. Previous run used matched state_dim
but that gave LSTM only ~25% of the tree's parameters."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from train import Config, run
from experiments.exp5_lstm_baseline import LSTMAutoencoder


print("=== LSTM state_dim=256 (param-matched-ish), 60 epochs ===")
cfg = Config(seq_len=16, epochs=60, n_train=3000, n_val=500,
             state_dim=256, name="lstm256_seq16")
r = run(cfg, model_ctor=LSTMAutoencoder)

print("\n=== LSTM state_dim=384, 60 epochs ===")
cfg2 = Config(seq_len=16, epochs=60, n_train=3000, n_val=500,
              state_dim=384, name="lstm384_seq16")
r2 = run(cfg2, model_ctor=LSTMAutoencoder)

print("\n=== SUMMARY ===")
print(f"lstm256: params={r['n_params']:,}  val_acc={r['best_val_acc']:.3f}  time={r['time_s']:.0f}s")
print(f"lstm384: params={r2['n_params']:,}  val_acc={r2['best_val_acc']:.3f}  time={r2['time_s']:.0f}s")
print(f"(compare to tree state_dim=128 at 940k params, val_acc=0.805)")
