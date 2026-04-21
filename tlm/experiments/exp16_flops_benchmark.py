"""Experiment 16: clean speedup measurement.

exp13/15 measured PyTorch wall-clock which was dominated by implementation
overhead (bmm, gather per-example weights, Python per-op). Here:

1. Count FLOPs analytically for each op - architectural truth, independent
   of framework.
2. Implement a minimal batch=1 hard-inference forward with pure weight
   slicing, no bmm/gather, no Python loop overhead in the hot path.
3. Bench the minimal hard-tree vs nn.LSTMCell, one step at a time.
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------- FLOP counts (analytical) -------------

def flop_lstm(state_dim: int, input_dim: int) -> int:
    # LSTMCell: [4*state, input] + [4*state, state] matmuls per step
    return 4 * (input_dim + state_dim) * state_dim


def flop_tree_soft(inp: int, out: int, depth: int) -> int:
    n_leaves = 2 ** depth
    n_internal = 2 ** depth - 1
    return inp * n_internal + n_leaves * inp * out


def flop_tree_hard(inp: int, out: int, depth: int) -> int:
    # Router: only `depth` inner products needed (visit one per level)
    # Leaf: 1 matmul (out x inp)
    return depth * inp + inp * out


def flop_shared_soft(inp: int, out: int, depth: int, hidden: int) -> int:
    n_leaves = 2 ** depth
    n_internal = 2 ** depth - 1
    return inp * n_internal + inp * hidden + n_leaves * hidden * out


def flop_shared_hard(inp: int, out: int, depth: int, hidden: int) -> int:
    return depth * inp + inp * hidden + hidden * out


# ------------- Minimal hard-tree forward (batch=1, pure slicing) -------------

class MinimalHardTree(nn.Module):
    """Batch=1 hard-routing tree. Only computes the router logits we visit and
    only the one leaf we reach - no gather, no bmm, no per-example weight
    expansion."""

    def __init__(self, input_dim: int, output_dim: int, depth: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.n_internal = 2 ** depth - 1
        self.n_leaves = 2 ** depth
        self.router = nn.Linear(input_dim, self.n_internal)
        self.leaves = nn.Linear(input_dim, self.n_leaves * output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (input_dim,)
        W_r, b_r = self.router.weight, self.router.bias
        leaf_idx = 0
        node_idx = 0
        for level in range(self.depth):
            i = node_idx + leaf_idx
            logit = W_r[i].dot(x) + b_r[i]
            if logit.item() > 0:
                leaf_idx = leaf_idx * 2
            else:
                leaf_idx = leaf_idx * 2 + 1
            node_idx += 2 ** level
        start = leaf_idx * self.output_dim
        end = start + self.output_dim
        W_l = self.leaves.weight[start:end]
        b_l = self.leaves.bias[start:end]
        return W_l @ x + b_l


class MinimalSoftTree(nn.Module):
    """Batch=1 soft-routing tree. All leaves computed, path-weighted sum."""
    def __init__(self, input_dim: int, output_dim: int, depth: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.n_internal = 2 ** depth - 1
        self.n_leaves = 2 ** depth
        self.router = nn.Linear(input_dim, self.n_internal)
        self.leaves = nn.Linear(input_dim, self.n_leaves * output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routing = torch.sigmoid(self.router(x))
        node_probs = torch.ones(1, device=x.device, dtype=x.dtype)
        idx = 0
        for level in range(self.depth):
            n_at_level = 2 ** level
            level_probs = routing[idx:idx + n_at_level]
            left = node_probs * level_probs
            right = node_probs * (1 - level_probs)
            node_probs = torch.stack([left, right], dim=1).reshape(-1)
            idx += n_at_level
        leaf_outs = self.leaves(x).view(self.n_leaves, self.output_dim)
        return (node_probs.unsqueeze(-1) * leaf_outs).sum(dim=0)


# ------------- Bench -------------

def bench(fn, n_runs=2000, n_warm=200):
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
    torch.set_num_threads(1)  # level the playing field

    STATE_DIM = 128
    INPUT_DIM = 64
    DEPTH = 3
    HIDDEN = 64
    FULL_IN = INPUT_DIM + STATE_DIM  # tree gets concat(x, state)

    # --- FLOP report ---
    lstm_f = flop_lstm(STATE_DIM, INPUT_DIM)
    tree_soft_f = flop_tree_soft(FULL_IN, STATE_DIM, DEPTH)
    tree_hard_f = flop_tree_hard(FULL_IN, STATE_DIM, DEPTH)
    sh_soft_f = flop_shared_soft(FULL_IN, STATE_DIM, DEPTH, HIDDEN)
    sh_hard_f = flop_shared_hard(FULL_IN, STATE_DIM, DEPTH, HIDDEN)

    print(f"Per-step FLOPs at state={STATE_DIM}, input={INPUT_DIM}, depth={DEPTH}:")
    print(f"  LSTM:              {lstm_f:>10,}")
    print(f"  Tree soft:         {tree_soft_f:>10,}   (vs LSTM: {tree_soft_f/lstm_f:.2f}x)")
    print(f"  Tree hard:         {tree_hard_f:>10,}   (vs LSTM: {lstm_f/tree_hard_f:.2f}x faster)")
    print(f"  Shared-tree soft:  {sh_soft_f:>10,}   (vs LSTM: {sh_soft_f/lstm_f:.2f}x)")
    print(f"  Shared-tree hard:  {sh_hard_f:>10,}   (vs LSTM: {lstm_f/sh_hard_f:.2f}x faster)")

    # --- Wall-clock bench (batch=1, single step) ---
    x_vec = torch.randn(FULL_IN)
    x_row = x_vec.unsqueeze(0)  # (1, input) for LSTMCell

    # LSTMCell regime: input=64, state=128 (its standard shape)
    lstm_cell = nn.LSTMCell(INPUT_DIM, STATE_DIM)
    lstm_cell.eval()
    h = torch.zeros(1, STATE_DIM)
    c = torch.zeros(1, STATE_DIM)
    x_lstm = torch.randn(1, INPUT_DIM)

    def run_lstm():
        with torch.no_grad():
            lstm_cell(x_lstm, (h, c))

    soft_tree = MinimalSoftTree(FULL_IN, STATE_DIM, DEPTH)
    soft_tree.eval()
    hard_tree = MinimalHardTree(FULL_IN, STATE_DIM, DEPTH)
    hard_tree.eval()
    # copy weights so comparison is apples-to-apples routing
    hard_tree.router.load_state_dict(soft_tree.router.state_dict())
    hard_tree.leaves.load_state_dict(soft_tree.leaves.state_dict())

    def run_soft():
        with torch.no_grad():
            soft_tree(x_vec)

    def run_hard():
        with torch.no_grad():
            hard_tree(x_vec)

    # warm up torch
    for _ in range(50):
        run_lstm(); run_soft(); run_hard()

    t_lstm = bench(run_lstm, n_runs=3000)
    t_soft = bench(run_soft, n_runs=3000)
    t_hard = bench(run_hard, n_runs=3000)

    print(f"\nWall-clock per step, batch=1, torch.set_num_threads(1):")
    print(f"  LSTM:     {t_lstm*1e6:>8.2f} us")
    print(f"  Soft:     {t_soft*1e6:>8.2f} us  ({t_lstm/t_soft:.2f}x vs LSTM)")
    print(f"  Hard:     {t_hard*1e6:>8.2f} us  ({t_lstm/t_hard:.2f}x vs LSTM)")
    print(f"  hard / soft speedup: {t_soft/t_hard:.2f}x")
    print(f"  theoretical hard / soft speedup: {tree_soft_f/tree_hard_f:.2f}x")
    print(f"  theoretical LSTM / hard speedup: {lstm_f/tree_hard_f:.2f}x")


if __name__ == "__main__":
    main()
