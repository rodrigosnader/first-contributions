"""Experiment 19: speedup at scale. In exp16/18, the torch-vs-numpy gap
showed that framework overhead dominates at state_dim=128. At larger
state_dims matmul cost should dominate Python/tensor overhead, so the
theoretical 3.9x should appear in PyTorch too.

Sweep state_dim in {128, 512, 1024, 2048} for both numpy and torch,
all at batch=1. FLOPs scale quadratically, overhead scales linearly,
so the crossover point should be visible.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib.util, os
spec = importlib.util.spec_from_file_location(
    "exp16_mod",
    os.path.join(os.path.dirname(__file__), "exp16_flops_benchmark.py"),
)
exp16 = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp16)
MinimalSoftTree = exp16.MinimalSoftTree
MinimalHardTree = exp16.MinimalHardTree

spec18 = importlib.util.spec_from_file_location(
    "exp18_mod",
    os.path.join(os.path.dirname(__file__), "exp18_numpy_port.py"),
)
exp18 = importlib.util.module_from_spec(spec18); spec18.loader.exec_module(exp18)
NumpyLSTM = exp18.NumpyLSTM
NumpySoftTree = exp18.NumpySoftTree
NumpyHardTree = exp18.NumpyHardTree

import numpy as np
import torch
import torch.nn as nn


def bench(fn, n_runs, n_warm):
    for _ in range(n_warm):
        fn()
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(n_runs):
            fn()
        best = min(best, time.perf_counter() - t0)
    return best / n_runs


def measure(state_dim, input_dim=64, depth=3):
    full_in = input_dim + state_dim
    # scale run counts down for very large state to keep total time reasonable
    n_runs = max(200, min(3000, 2_000_000 // (state_dim * state_dim)))
    n_warm = max(50, n_runs // 10)

    # numpy
    x_np = np.random.randn(full_in).astype(np.float32)
    x_lstm_np = np.random.randn(input_dim).astype(np.float32)
    h_np = np.zeros(state_dim, dtype=np.float32)
    c_np = np.zeros(state_dim, dtype=np.float32)
    lstm_np = NumpyLSTM(input_dim, state_dim)
    soft_np = NumpySoftTree(full_in, state_dim, depth)
    hard_np = NumpyHardTree(full_in, state_dim, depth,
                            soft_np.W_r, soft_np.b_r, soft_np.W_l, soft_np.b_l)
    t_lstm_np = bench(lambda: lstm_np.step(x_lstm_np, h_np, c_np), n_runs, n_warm)
    t_hard_np = bench(lambda: hard_np.forward(x_np), n_runs, n_warm)

    # torch
    torch.set_num_threads(1)
    x_t = torch.from_numpy(x_np)
    x_lstm_t = torch.from_numpy(x_lstm_np).unsqueeze(0)
    h_t = torch.zeros(1, state_dim)
    c_t = torch.zeros(1, state_dim)
    lstm_t = nn.LSTMCell(input_dim, state_dim).eval()
    hard_t = MinimalHardTree(full_in, state_dim, depth).eval()

    def run_lstm_t():
        with torch.no_grad():
            lstm_t(x_lstm_t, (h_t, c_t))
    def run_hard_t():
        with torch.no_grad():
            hard_t(x_t)
    t_lstm_t = bench(run_lstm_t, n_runs, n_warm)
    t_hard_t = bench(run_hard_t, n_runs, n_warm)

    return {
        "state_dim": state_dim, "n_runs": n_runs,
        "lstm_np_us": t_lstm_np * 1e6, "hard_np_us": t_hard_np * 1e6,
        "lstm_t_us": t_lstm_t * 1e6, "hard_t_us": t_hard_t * 1e6,
        "flops_lstm": exp16.flop_lstm(state_dim, input_dim),
        "flops_hard": exp16.flop_tree_hard(full_in, state_dim, depth),
    }


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    print(f"{'state':>6} {'nruns':>6} "
          f"{'lstm_np':>8} {'hard_np':>8} {'np_spd':>7} "
          f"{'lstm_t':>8} {'hard_t':>8} {'torch_spd':>10} "
          f"{'FLOP_ratio':>11}")
    print(f"{'':>6} {'':>6} "
          f"{'us':>8} {'us':>8} {'':>7} "
          f"{'us':>8} {'us':>8} {'':>10} "
          f"{'LSTM/hard':>11}")
    for sd in [128, 512, 1024, 2048]:
        r = measure(sd)
        flop_ratio = r["flops_lstm"] / r["flops_hard"]
        print(f"{r['state_dim']:>6d} {r['n_runs']:>6d} "
              f"{r['lstm_np_us']:>8.2f} {r['hard_np_us']:>8.2f} "
              f"{r['lstm_np_us']/r['hard_np_us']:>6.2f}x "
              f"{r['lstm_t_us']:>8.2f} {r['hard_t_us']:>8.2f} "
              f"{r['lstm_t_us']/r['hard_t_us']:>9.2f}x "
              f"{flop_ratio:>10.2f}x")


if __name__ == "__main__":
    main()
