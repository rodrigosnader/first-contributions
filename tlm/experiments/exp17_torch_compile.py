"""Experiment 17: does torch.compile recover the FLOP speedup?

exp16 showed tree hard = 3.9x fewer FLOPs than LSTM but PyTorch wall-clock
just ties LSTM (34 vs 35 us). The ~3x gap is Python/PyTorch overhead.
torch.compile should JIT the forward into a fused graph and remove
most of that overhead.

Bench: same batch=1, single-thread setup as exp16, with and without
torch.compile on each variant. Compare speedups.
"""
import time
import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.exp16_flops_benchmark import MinimalSoftTree, MinimalHardTree


def bench(fn, n_runs=3000, n_warm=300):
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
    torch.manual_seed(0)
    torch.set_num_threads(1)

    STATE_DIM = 128
    INPUT_DIM = 64
    DEPTH = 3
    FULL_IN = INPUT_DIM + STATE_DIM

    x_vec = torch.randn(FULL_IN)
    lstm_cell = nn.LSTMCell(INPUT_DIM, STATE_DIM).eval()
    h = torch.zeros(1, STATE_DIM)
    c = torch.zeros(1, STATE_DIM)
    x_lstm = torch.randn(1, INPUT_DIM)

    soft_tree = MinimalSoftTree(FULL_IN, STATE_DIM, DEPTH).eval()
    hard_tree = MinimalHardTree(FULL_IN, STATE_DIM, DEPTH).eval()
    hard_tree.router.load_state_dict(soft_tree.router.state_dict())
    hard_tree.leaves.load_state_dict(soft_tree.leaves.state_dict())

    # Uncompiled baseline runs
    def run_lstm():
        with torch.no_grad():
            lstm_cell(x_lstm, (h, c))
    def run_soft():
        with torch.no_grad():
            soft_tree(x_vec)
    def run_hard():
        with torch.no_grad():
            hard_tree(x_vec)

    t_lstm = bench(run_lstm, n_runs=3000)
    t_soft = bench(run_soft, n_runs=3000)
    t_hard = bench(run_hard, n_runs=3000)

    # Compiled versions - try to JIT away Python overhead
    try:
        lstm_c = torch.compile(lstm_cell, mode="reduce-overhead")
        soft_c = torch.compile(soft_tree, mode="reduce-overhead")
        hard_c = torch.compile(hard_tree, mode="reduce-overhead")

        def run_lstm_c():
            with torch.no_grad():
                lstm_c(x_lstm, (h, c))
        def run_soft_c():
            with torch.no_grad():
                soft_c(x_vec)
        def run_hard_c():
            with torch.no_grad():
                hard_c(x_vec)

        t_lstm_c = bench(run_lstm_c, n_runs=3000)
        t_soft_c = bench(run_soft_c, n_runs=3000)
        t_hard_c = bench(run_hard_c, n_runs=3000)
        compiled_ok = True
    except Exception as e:
        print(f"torch.compile failed: {e}")
        compiled_ok = False

    print(f"\nWall-clock per step, batch=1, single thread:")
    print(f"{'op':<14} {'eager us':>10} {'compiled us':>13} {'compile speedup':>17}")
    print(f"{'LSTM':<14} {t_lstm*1e6:>10.2f} "
          f"{'-' if not compiled_ok else f'{t_lstm_c*1e6:.2f}':>13} "
          f"{'-' if not compiled_ok else f'{t_lstm/t_lstm_c:.2f}x':>17}")
    print(f"{'Tree soft':<14} {t_soft*1e6:>10.2f} "
          f"{'-' if not compiled_ok else f'{t_soft_c*1e6:.2f}':>13} "
          f"{'-' if not compiled_ok else f'{t_soft/t_soft_c:.2f}x':>17}")
    print(f"{'Tree hard':<14} {t_hard*1e6:>10.2f} "
          f"{'-' if not compiled_ok else f'{t_hard_c*1e6:.2f}':>13} "
          f"{'-' if not compiled_ok else f'{t_hard/t_hard_c:.2f}x':>17}")

    if compiled_ok:
        print(f"\n=== COMPILED: tree-hard vs LSTM ===")
        print(f"  LSTM compiled:     {t_lstm_c*1e6:>6.2f} us")
        print(f"  Tree hard compiled:{t_hard_c*1e6:>6.2f} us  ({t_lstm_c/t_hard_c:.2f}x vs LSTM)")
        print(f"  theoretical speedup from FLOPs: 3.91x")


if __name__ == "__main__":
    main()
