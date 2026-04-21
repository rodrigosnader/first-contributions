"""Experiment 18: numpy port of hard tree - can we actually hit the FLOP speedup?

PyTorch's per-op Python overhead swamps small matrices. numpy with a
straight Python/BLAS call and no autograd should be the lower bound
for achievable wall-clock at this scale.

Bench batch=1 single-step for:
- LSTMCell equivalent in numpy (4 gates, fused matmul)
- Minimal soft tree in numpy
- Minimal hard tree in numpy (pure slicing, visited-only routing)
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn


def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


class NumpyLSTM:
    def __init__(self, input_dim, state_dim, seed=0):
        rng = np.random.default_rng(seed)
        bound = 1.0 / np.sqrt(state_dim)
        self.W_ih = rng.uniform(-bound, bound, (4 * state_dim, input_dim)).astype(np.float32)
        self.W_hh = rng.uniform(-bound, bound, (4 * state_dim, state_dim)).astype(np.float32)
        self.b_ih = rng.uniform(-bound, bound, (4 * state_dim,)).astype(np.float32)
        self.b_hh = rng.uniform(-bound, bound, (4 * state_dim,)).astype(np.float32)
        self.state_dim = state_dim

    def step(self, x, h, c):
        gates = self.W_ih @ x + self.b_ih + self.W_hh @ h + self.b_hh
        i, f, g, o = np.split(gates, 4)
        i = sigmoid(i); f = sigmoid(f); g = np.tanh(g); o = sigmoid(o)
        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)
        return h_new, c_new


class NumpySoftTree:
    def __init__(self, input_dim, output_dim, depth, seed=0):
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.n_internal = 2 ** depth - 1
        self.n_leaves = 2 ** depth
        bound = 1.0 / np.sqrt(input_dim)
        self.W_r = rng.uniform(-bound, bound, (self.n_internal, input_dim)).astype(np.float32)
        self.b_r = rng.uniform(-bound, bound, (self.n_internal,)).astype(np.float32)
        self.W_l = rng.uniform(-bound, bound, (self.n_leaves * output_dim, input_dim)).astype(np.float32)
        self.b_l = rng.uniform(-bound, bound, (self.n_leaves * output_dim,)).astype(np.float32)

    def forward(self, x):
        logits = self.W_r @ x + self.b_r
        probs = sigmoid(logits)
        node_probs = np.ones(1, dtype=np.float32)
        idx = 0
        for level in range(self.depth):
            n = 2 ** level
            lp = probs[idx:idx + n]
            left = node_probs * lp
            right = node_probs * (1 - lp)
            node_probs = np.stack([left, right], axis=1).reshape(-1)
            idx += n
        leaf_outs = (self.W_l @ x + self.b_l).reshape(self.n_leaves, self.output_dim)
        return (node_probs[:, None] * leaf_outs).sum(axis=0)


class NumpyHardTree:
    def __init__(self, input_dim, output_dim, depth, W_r, b_r, W_l, b_l):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.n_leaves = 2 ** depth
        self.W_r = W_r
        self.b_r = b_r
        self.W_l = W_l
        self.b_l = b_l

    def forward(self, x):
        leaf_idx = 0
        node_idx = 0
        for level in range(self.depth):
            i = node_idx + leaf_idx
            logit = self.W_r[i] @ x + self.b_r[i]
            if logit > 0:
                leaf_idx = leaf_idx * 2
            else:
                leaf_idx = leaf_idx * 2 + 1
            node_idx += 2 ** level
        start = leaf_idx * self.output_dim
        end = start + self.output_dim
        return self.W_l[start:end] @ x + self.b_l[start:end]


def bench(fn, n_runs=5000, n_warm=500):
    for _ in range(n_warm):
        fn()
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(n_runs):
            fn()
        best = min(best, time.perf_counter() - t0)
    return best / n_runs


def main():
    np.random.seed(0)
    STATE_DIM = 128
    INPUT_DIM = 64
    DEPTH = 3
    FULL_IN = INPUT_DIM + STATE_DIM

    # numpy
    x_np = np.random.randn(FULL_IN).astype(np.float32)
    x_lstm = np.random.randn(INPUT_DIM).astype(np.float32)
    h_np = np.zeros(STATE_DIM, dtype=np.float32)
    c_np = np.zeros(STATE_DIM, dtype=np.float32)

    lstm_np = NumpyLSTM(INPUT_DIM, STATE_DIM)
    soft_np = NumpySoftTree(FULL_IN, STATE_DIM, DEPTH)
    hard_np = NumpyHardTree(FULL_IN, STATE_DIM, DEPTH,
                            soft_np.W_r, soft_np.b_r, soft_np.W_l, soft_np.b_l)

    t_lstm_np = bench(lambda: lstm_np.step(x_lstm, h_np, c_np))
    t_soft_np = bench(lambda: soft_np.forward(x_np))
    t_hard_np = bench(lambda: hard_np.forward(x_np))

    # torch (eager) for reference
    torch.manual_seed(0)
    torch.set_num_threads(1)
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "exp16_mod",
        os.path.join(os.path.dirname(__file__), "exp16_flops_benchmark.py"),
    )
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    MinimalSoftTree = mod.MinimalSoftTree
    MinimalHardTree = mod.MinimalHardTree

    x_t = torch.from_numpy(x_np)
    x_lstm_t = torch.from_numpy(x_lstm).unsqueeze(0)
    h_t = torch.zeros(1, STATE_DIM)
    c_t = torch.zeros(1, STATE_DIM)
    lstm_t = nn.LSTMCell(INPUT_DIM, STATE_DIM).eval()
    soft_t = MinimalSoftTree(FULL_IN, STATE_DIM, DEPTH).eval()
    hard_t = MinimalHardTree(FULL_IN, STATE_DIM, DEPTH).eval()

    def run_lstm_t():
        with torch.no_grad():
            lstm_t(x_lstm_t, (h_t, c_t))
    def run_soft_t():
        with torch.no_grad():
            soft_t(x_t)
    def run_hard_t():
        with torch.no_grad():
            hard_t(x_t)

    t_lstm_t = bench(run_lstm_t)
    t_soft_t = bench(run_soft_t)
    t_hard_t = bench(run_hard_t)

    print(f"Per-step wall-clock, batch=1, single thread:")
    print(f"{'op':<14} {'torch us':>10} {'numpy us':>10} {'numpy speedup':>15}")
    print(f"{'LSTM':<14} {t_lstm_t*1e6:>10.2f} {t_lstm_np*1e6:>10.2f} {t_lstm_t/t_lstm_np:>14.2f}x")
    print(f"{'Tree soft':<14} {t_soft_t*1e6:>10.2f} {t_soft_np*1e6:>10.2f} {t_soft_t/t_soft_np:>14.2f}x")
    print(f"{'Tree hard':<14} {t_hard_t*1e6:>10.2f} {t_hard_np*1e6:>10.2f} {t_hard_t/t_hard_np:>14.2f}x")

    print(f"\n=== numpy head-to-head ===")
    print(f"  LSTM numpy:     {t_lstm_np*1e6:>6.2f} us")
    print(f"  Tree soft numpy:{t_soft_np*1e6:>6.2f} us  ({t_lstm_np/t_soft_np:.2f}x vs LSTM)")
    print(f"  Tree hard numpy:{t_hard_np*1e6:>6.2f} us  ({t_lstm_np/t_hard_np:.2f}x vs LSTM)")
    print(f"  tree hard vs tree soft: {t_soft_np/t_hard_np:.2f}x (theoretical: 7.87x)")
    print(f"  tree hard vs LSTM:      {t_lstm_np/t_hard_np:.2f}x (theoretical: 3.91x)")


if __name__ == "__main__":
    main()
