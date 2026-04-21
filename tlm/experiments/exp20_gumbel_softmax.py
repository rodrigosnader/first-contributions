"""Experiment 20: Gumbel-sigmoid routing with temperature annealing.

exp15 tried simple entropy annealing - routing got more committed but
97/3 splits still cascade to random when hard-ified (exp13 result).
Gumbel-softmax is stronger: injects Gumbel noise during training at
each routing decision, then anneals the temperature. Training distribution
converges to pure discrete sampling; hard inference then matches training.

At inference we can either:
- Run soft with low tau (~0.01): essentially discrete, but still
  differentiable-style forward
- Run hard with _hard=True: full sparse (for real speedup from exp18/19)

Test: does Gumbel+annealing close the soft-vs-hard accuracy gap that
entropy annealing couldn't?
"""
import sys
import math
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from model import SharedBackboneSoftTree, SoftTree
from lm_model import (
    TreeLMv2, RelationExtractor, ContextExtractor, TreeDecoderHead,
)
from lm_train import LMConfig, shifted_loss
from data import make_shakespeare_splits


# -------- Gumbel-sigmoid wrapper --------

class GumbelSharedTree(SharedBackboneSoftTree):
    """SharedBackboneSoftTree with Gumbel-sigmoid routing during training."""
    def __init__(self, input_dim, output_dim, depth, hidden_dim=None):
        super().__init__(input_dim, output_dim, depth, hidden_dim)
        self._tau = 1.0  # annealed externally

    def forward(self, x):
        if self._hard:
            return self.hard_forward(x)
        batch = x.shape[0]
        logits = self.router(x)
        if self.training:
            u1 = torch.rand_like(logits).clamp(1e-20, 1 - 1e-20)
            u2 = torch.rand_like(logits).clamp(1e-20, 1 - 1e-20)
            g1 = -torch.log(-torch.log(u1))
            g2 = -torch.log(-torch.log(u2))
            routing_probs = torch.sigmoid((logits + g1 - g2) / self._tau)
        else:
            # Deterministic at eval: low-tau sigmoid = sharpened soft
            routing_probs = torch.sigmoid(logits / max(self._tau, 0.1))
        self._last_routing_probs = routing_probs

        node_probs = torch.ones(batch, 1, device=x.device, dtype=x.dtype)
        idx = 0
        for level in range(self.depth):
            n_at_level = 2 ** level
            level_probs = routing_probs[:, idx:idx + n_at_level]
            left = node_probs * level_probs
            right = node_probs * (1.0 - level_probs)
            node_probs = torch.stack([left, right], dim=2).reshape(batch, 2 * n_at_level)
            idx += n_at_level

        h = torch.nn.functional.gelu(self.shared(x))
        leaf_outs = self.leaves(h).reshape(batch, self.n_leaves, self.output_dim)
        return (node_probs.unsqueeze(-1) * leaf_outs).sum(dim=1)


class GumbelSharedCell(nn.Module):
    def __init__(self, input_dim, state_dim, depth):
        super().__init__()
        self.state_dim = state_dim
        self.tree = GumbelSharedTree(input_dim + state_dim, state_dim, depth)
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, x, state):
        return self.norm(self.tree(torch.cat([x, state], dim=-1)))


class TreeLMv2GumbelShared(TreeLMv2):
    def __init__(self, vocab_size, embed_dim=32, state_dim=128, depth=4,
                 relation_dim=64, context_dim=64, max_len=128):
        super().__init__(vocab_size, embed_dim, state_dim, depth,
                         relation_dim, context_dim, max_len)
        self.cell = GumbelSharedCell(relation_dim, state_dim, depth)


def set_tau(model, tau):
    for m in model.modules():
        if isinstance(m, GumbelSharedTree):
            m._tau = tau


def set_hard_mode(model, hard):
    for m in model.modules():
        if isinstance(m, (SoftTree, SharedBackboneSoftTree)):
            m._hard = hard


def train_and_eval():
    torch.manual_seed(0)
    cfg = LMConfig(seq_len=64, state_dim=128, depth=3, relation_dim=64,
                   context_dim=64, epochs=25, n_train=3000, name="gumbel")

    train_data, val_data, stoi, itos = make_shakespeare_splits(
        cfg.n_train, cfg.n_val, cfg.seq_len
    )
    vocab_size = len(stoi)

    model = TreeLMv2GumbelShared(vocab_size, embed_dim=cfg.embed_dim,
                                 state_dim=cfg.state_dim, depth=cfg.depth,
                                 relation_dim=cfg.relation_dim,
                                 context_dim=cfg.context_dim, max_len=cfg.seq_len)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TreeLMv2GumbelShared, params: {n_params:,}")
    print(f"vocab={vocab_size}, seq_len={cfg.seq_len}, state={cfg.state_dim}, depth={cfg.depth}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    # tau schedule: start high (noisy soft), anneal to low (near-discrete)
    def tau_at(epoch):
        tau_start, tau_end = 1.0, 0.1
        if epoch <= 3:
            return tau_start
        progress = min(1.0, (epoch - 3) / (cfg.epochs - 3))
        # exponential anneal
        return tau_start * (tau_end / tau_start) ** progress

    t0 = time.time()
    best_val_bpc = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        tau = tau_at(epoch)
        set_tau(model, tau)
        model.train()
        perm = torch.randperm(cfg.n_train)
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0
        for i in range(0, cfg.n_train, cfg.batch_size):
            batch = train_data[perm[i:i + cfg.batch_size]]
            logits = model(batch)
            loss, acc = shifted_loss(logits, batch, loss_fn)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total_loss += loss.item()
            total_acc += acc
            n_batches += 1

        if epoch % 5 == 0 or epoch == 1 or epoch == cfg.epochs:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_data)
                val_loss, val_acc = shifted_loss(val_logits, val_data, loss_fn)
            val_bpc = val_loss.item() / math.log(2)
            best_val_bpc = min(best_val_bpc, val_bpc)
            train_bpc = total_loss / n_batches / math.log(2)
            print(f"epoch {epoch:2d}  tau={tau:.3f}  train_bpc {train_bpc:.3f}  "
                  f"val_bpc {val_bpc:.3f}  val_acc {val_acc:.3f}  "
                  f"{time.time()-t0:.0f}s")

    # --- final eval: soft (tau=0.1) vs hard (_hard=True) ---
    model.eval()
    set_tau(model, 0.1)
    with torch.no_grad():
        soft_logits = model(val_data)
        soft_loss, soft_acc = shifted_loss(soft_logits, val_data, loss_fn)
    soft_bpc = soft_loss.item() / math.log(2)

    set_hard_mode(model, True)
    with torch.no_grad():
        hard_logits = model(val_data)
        hard_loss, hard_acc = shifted_loss(hard_logits, val_data, loss_fn)
    hard_bpc = hard_loss.item() / math.log(2)
    set_hard_mode(model, False)

    # routing confidence
    conf_vals = []
    with torch.no_grad():
        _ = model(val_data[:32])
        for m in model.modules():
            if isinstance(m, GumbelSharedTree) and m._last_routing_probs is not None:
                p = m._last_routing_probs
                conf_vals.append(((p - 0.5).abs() * 2).mean().item())  # in [0,1], 1=committed
    confidence = sum(conf_vals) / len(conf_vals) if conf_vals else 0.0

    print("\n=== Gumbel + annealing final eval ===")
    print(f"routing confidence (|p-0.5|*2, 1=committed): {confidence:.3f}")
    print(f"soft (tau=0.1): BPC {soft_bpc:.3f}  acc {soft_acc:.3f}")
    print(f"hard (_hard):   BPC {hard_bpc:.3f}  acc {hard_acc:.3f}")
    print(f"acc drop:  {soft_acc - hard_acc:+.3f}")
    print(f"BPC delta: {hard_bpc - soft_bpc:+.3f}")

    # reminder of prior results
    print("\n=== REFERENCE (exp15, entropy annealing, shared_forget) ===")
    print(f"  soft BPC 2.933, hard BPC 5.041, gap +2.11")
    print(f"=== REFERENCE (exp13, no annealing, shared) ===")
    print(f"  soft BPC 2.662, hard BPC 5.410, gap +2.75")


if __name__ == "__main__":
    train_and_eval()
