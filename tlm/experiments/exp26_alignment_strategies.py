"""Experiment 26: align train and inference - four ways to fix soft/hard gap.

Comparison of training strategies to make hard inference work after training:

1. Baseline: vanilla soft training, hard at test (exp13 reference, gap -27pp)
2. Gumbel + tau anneal: stochastic Gumbel-sigmoid, tau 1.0->0.1 (exp20, gap -11pp)
3. Straight-through estimator: forward uses hard routing, backward uses soft gradient
4. Full hard from start: argmax routing at all times - no soft warmup
5. Curriculum soft->hard: first half epochs soft, second half hard

Goal: find the routing-alignment strategy that minimizes the soft-to-hard
accuracy drop while keeping training stable.
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
from model import (
    SoftTree, SharedBackboneSoftTree, ForgetGatedSharedTreeCell,
)
from lm_model import (
    TreeLMv2SharedForget,
    RelationExtractor, ContextExtractor, TreeDecoderHead,
)


# -------- Routing strategies --------
# All implemented by overriding the forward of SharedBackboneSoftTree.
# Strategy is selected via a per-tree attribute `_routing_mode`.

_ORIG_FORWARD = SharedBackboneSoftTree.forward


def soft_forward(self, x):
    """Original soft mixing forward."""
    return _ORIG_FORWARD(self, x)


def gumbel_forward(self, x):
    """Gumbel-sigmoid routing with self._tau."""
    if self._hard:
        return self.hard_forward(x)
    batch = x.shape[0]
    logits = self.router(x)
    if self.training:
        u1 = torch.rand_like(logits).clamp(1e-20, 1 - 1e-20)
        u2 = torch.rand_like(logits).clamp(1e-20, 1 - 1e-20)
        g = -torch.log(-torch.log(u1)) - (-torch.log(-torch.log(u2)))
        routing = torch.sigmoid((logits + g) / self._tau)
    else:
        routing = torch.sigmoid(logits / max(self._tau, 0.1))
    self._last_routing_probs = routing
    return _propagate(self, x, routing, batch)


def ste_forward(self, x):
    """Straight-through: forward uses hard {0,1} routing, backward uses soft gradient."""
    if self._hard:
        return self.hard_forward(x)
    batch = x.shape[0]
    logits = self.router(x)
    soft = torch.sigmoid(logits)
    hard = (logits > 0).float()
    # Trick: hard in forward, gradient flows from soft (since hard - soft.detach() + soft = hard, but grad is soft)
    routing = hard + soft - soft.detach()
    self._last_routing_probs = routing
    return _propagate(self, x, routing, batch)


def full_hard_forward(self, x):
    """Pure argmax routing during both training and inference. No soft path."""
    if self.training:
        # During training we need a gradient-friendly approximation of hard:
        # use STE (so gradient still flows). Otherwise model can't learn.
        return ste_forward(self, x)
    return self.hard_forward(x)


def _propagate(self, x, routing, batch):
    """Shared tail: tree path probability propagation + leaf computation."""
    node_probs = torch.ones(batch, 1, device=x.device, dtype=x.dtype)
    idx = 0
    for level in range(self.depth):
        n_at_level = 2 ** level
        level_probs = routing[:, idx:idx + n_at_level]
        left = node_probs * level_probs
        right = node_probs * (1.0 - level_probs)
        node_probs = torch.stack([left, right], dim=2).reshape(batch, 2 * n_at_level)
        idx += n_at_level
    h = torch.nn.functional.gelu(self.shared(x))
    leaf_outs = self.leaves(h).reshape(batch, self.n_leaves, self.output_dim)
    return (node_probs.unsqueeze(-1) * leaf_outs).sum(dim=1)


def set_routing_mode(model, mode, tau=1.0):
    """Patch every SharedBackboneSoftTree to use the given routing strategy."""
    for m in model.modules():
        if isinstance(m, SharedBackboneSoftTree):
            m._tau = tau
            if mode == "soft":
                m.forward = soft_forward.__get__(m, type(m))
            elif mode == "gumbel":
                m.forward = gumbel_forward.__get__(m, type(m))
            elif mode == "ste":
                m.forward = ste_forward.__get__(m, type(m))
            elif mode == "full_hard":
                m.forward = full_hard_forward.__get__(m, type(m))
            else:
                raise ValueError(mode)


def set_hard_mode(model, hard):
    for m in model.modules():
        if isinstance(m, (SoftTree, SharedBackboneSoftTree)):
            m._hard = hard


# -------- Train + eval --------

def train(strategy, n_epochs=20, n_train=3000, seq_len=64, seed=0):
    torch.manual_seed(seed)
    train_data, val_data, stoi, itos = make_shakespeare_splits(n_train, 500, seq_len)
    vocab = len(stoi)

    model = TreeLMv2SharedForget(vocab, embed_dim=32, state_dim=128, depth=3,
                                 relation_dim=64, context_dim=64, max_len=seq_len)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    t0 = time.time()

    def tau_at(epoch):
        if strategy != "gumbel":
            return 1.0
        if epoch <= 3:
            return 1.0
        progress = min(1.0, (epoch - 3) / max(1, n_epochs - 3))
        return 1.0 * (0.1 / 1.0) ** progress

    for epoch in range(1, n_epochs + 1):
        if strategy == "curriculum":
            mode = "soft" if epoch <= n_epochs // 2 else "ste"
        else:
            mode = strategy if strategy != "baseline" else "soft"
        set_routing_mode(model, mode, tau=tau_at(epoch))
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0; total_acc = 0.0; n_batches = 0
        for i in range(0, n_train, 64):
            batch = train_data[perm[i:i + 64]]
            logits = model(batch)
            loss, acc = shifted_loss(logits, batch, loss_fn)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item(); total_acc += acc; n_batches += 1
        if epoch % 5 == 0 or epoch == n_epochs:
            print(f"  epoch {epoch:2d}  mode={mode}  tau={tau_at(epoch):.2f}  "
                  f"train_bpc {total_loss/n_batches/math.log(2):.3f}  "
                  f"acc {total_acc/n_batches:.3f}")

    # Eval soft and hard
    model.eval()
    set_routing_mode(model, "soft", tau=0.1)
    set_hard_mode(model, False)
    with torch.no_grad():
        sl, sa = shifted_loss(model(val_data), val_data, loss_fn)
    soft_bpc = sl.item() / math.log(2)
    soft_acc = sa

    set_hard_mode(model, True)
    with torch.no_grad():
        hl, ha = shifted_loss(model(val_data), val_data, loss_fn)
    hard_bpc = hl.item() / math.log(2)
    hard_acc = ha
    set_hard_mode(model, False)

    return {
        "strategy": strategy, "n_params": n_params, "time_s": time.time() - t0,
        "soft_bpc": soft_bpc, "soft_acc": soft_acc,
        "hard_bpc": hard_bpc, "hard_acc": hard_acc,
        "gap_bpc": hard_bpc - soft_bpc, "gap_acc": soft_acc - hard_acc,
    }


def main():
    strategies = ["baseline", "gumbel", "ste", "full_hard", "curriculum"]
    results = []
    for s in strategies:
        print(f"\n=== strategy: {s} ===")
        r = train(s, n_epochs=20)
        results.append(r)
        print(f"  -> soft BPC {r['soft_bpc']:.3f} acc {r['soft_acc']:.3f}  |  "
              f"hard BPC {r['hard_bpc']:.3f} acc {r['hard_acc']:.3f}  |  "
              f"gap BPC {r['gap_bpc']:+.3f} acc {r['gap_acc']:+.3f}")

    print("\n=== SUMMARY ===")
    print(f"{'strategy':<12} {'soft_bpc':>9} {'soft_acc':>9} {'hard_bpc':>9} "
          f"{'hard_acc':>9} {'gap_acc':>9} {'time_s':>7}")
    for r in results:
        print(f"{r['strategy']:<12} {r['soft_bpc']:>9.3f} {r['soft_acc']:>9.3f} "
              f"{r['hard_bpc']:>9.3f} {r['hard_acc']:>9.3f} "
              f"{r['gap_acc']:>+9.3f} {r['time_s']:>6.0f}s")


if __name__ == "__main__":
    main()
