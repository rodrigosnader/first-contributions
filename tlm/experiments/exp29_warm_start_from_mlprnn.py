"""Experiment 29: warm-start a tree from a trained 1-leaf model, fine-tune,
see if the tree improves the seed.

User's thesis: a tree is at LEAST as expressive as an MLP-RNN. With routing
collapsed to one leaf, the tree IS the MLP-RNN. So warm-starting from a
trained MLP-RNN should at minimum match it; if routing learns to specialize,
it should improve.

Setup:
1. Train an MLP-RNN (single-leaf cell, identical structure to TreeCell with
   depth=0). Other trees (relation/context/head) at depth=3 as usual.
2. Build a TreeLMv2SharedForget at cell depth=3, warm-start from the
   MLP-RNN with router pinned to leaf 0 so it computes the same function
   at init.
3. Verify the initial val_bpc matches the seed.
4. Fine-tune (free all weights including router).
5. Compare seed, warm-started fine-tuned, from-scratch tree.

Hypotheses:
- H1: warm-started tree at init = seed (sanity check)
- H2: fine-tuned warm-started tree IMPROVES on seed (routing helps)
- H3: fine-tuned warm-started tree DEGRADES (tree optim landscape is harder
   even with a good init - tells us tree's problem is fundamentally
   optimization, not capacity)
"""
import sys
import math
import time
import copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from data import make_shakespeare_splits
from lm_train import LMConfig, shifted_loss
from lm_model import (
    TreeLMv2, TreeLMv2SharedForget, RelationExtractor,
    ContextExtractor, TreeDecoderHead,
)
from model import (
    SoftTree, SharedBackboneSoftTree, ForgetGatedSharedTreeCell,
)


# 1-leaf cell with same shape as ForgetGatedSharedTreeCell at depth=0.
# Computation:  state_t = f * state_{t-1} + (1 - f) * LayerNorm(W_leaf * GELU(W_shared * [x, state]))
class MLPRNNCell(nn.Module):
    def __init__(self, input_dim, state_dim, hidden_dim=None):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim or max(state_dim // 2, 32)
        self.shared = nn.Linear(input_dim + state_dim, self.hidden_dim)
        self.leaf = nn.Linear(self.hidden_dim, state_dim)
        self.forget = nn.Linear(input_dim + state_dim, state_dim)
        self.norm = nn.LayerNorm(state_dim)
        nn.init.constant_(self.forget.bias, 2.0)

    def forward(self, x, state):
        concat = torch.cat([x, state], dim=-1)
        f = torch.sigmoid(self.forget(concat))
        h = torch.nn.functional.gelu(self.shared(concat))
        proposal = self.norm(self.leaf(h))
        return f * state + (1 - f) * proposal


class TreeLMv2MLPRNN(TreeLMv2):
    """TreeLMv2 with single-leaf MLPRNN cell. Relation/context/head still
    use tree at depth=3 - those components are isolated to keep this
    experiment about the cell only."""
    def __init__(self, vocab_size, embed_dim=32, state_dim=128, depth=3,
                 relation_dim=64, context_dim=64, max_len=128):
        super().__init__(vocab_size, embed_dim, state_dim, depth,
                         relation_dim, context_dim, max_len)
        self.cell = MLPRNNCell(relation_dim, state_dim)


def warm_start_tree_from_mlprnn(mlp_lm, depth=3):
    """Build a TreeLMv2SharedForget that initially computes the SAME function
    as the trained MLPRNN-based LM. Router pinned to leaf 0, leaf 0 = MLPRNN
    leaf weights, other leaves = small zero-ish, all other components copied."""
    sd = mlp_lm.cell.state_dim
    rel_in = mlp_lm.relation.tree.input_dim if hasattr(mlp_lm.relation.tree, "input_dim") else None
    rel_out = mlp_lm.relation.tree.output_dim
    ctx_out = mlp_lm.context.tree.output_dim
    embed_dim = mlp_lm.embed.embedding_dim
    vocab = mlp_lm.embed.num_embeddings
    max_len = mlp_lm.decoder.pos_embed.num_embeddings if hasattr(mlp_lm, "decoder") else 128

    new_lm = TreeLMv2SharedForget(vocab, embed_dim=embed_dim, state_dim=sd, depth=depth,
                                  relation_dim=rel_out, context_dim=ctx_out,
                                  max_len=max_len)
    # Copy non-cell components verbatim
    new_lm.embed.load_state_dict(mlp_lm.embed.state_dict())
    new_lm.relation.load_state_dict(mlp_lm.relation.state_dict())
    new_lm.context.load_state_dict(mlp_lm.context.state_dict())
    new_lm.head.load_state_dict(mlp_lm.head.state_dict())

    # Cell warm start: ForgetGatedSharedTreeCell has self.tree (SharedBackboneSoftTree),
    # self.forget, self.norm. Inside self.tree: router, shared, leaves.
    new_cell = new_lm.cell
    src_cell = mlp_lm.cell
    new_cell.forget.load_state_dict(src_cell.forget.state_dict())
    new_cell.norm.load_state_dict(src_cell.norm.state_dict())
    new_cell.tree.shared.load_state_dict(src_cell.shared.state_dict())

    # Router: pin to leaf 0. Routing logic: at each level, go LEFT if logit > 0.
    # leaf_idx = 0 = always-left path. Set router weights to 0 and bias to large positive.
    with torch.no_grad():
        new_cell.tree.router.weight.zero_()
        new_cell.tree.router.bias.fill_(5.0)  # sigmoid(5) ~ 0.993, soft routing strongly to left

    # Leaves: leaves linear is (n_leaves * state_dim, hidden_dim). Want chunk 0 (rows 0:state_dim)
    # to equal src_cell.leaf.weight. Other chunks to zero.
    with torch.no_grad():
        new_cell.tree.leaves.weight.zero_()
        new_cell.tree.leaves.bias.zero_()
        n_leaves = new_cell.tree.n_leaves
        sd_out = new_cell.tree.output_dim
        new_cell.tree.leaves.weight[:sd_out] = src_cell.leaf.weight
        new_cell.tree.leaves.bias[:sd_out] = src_cell.leaf.bias

    return new_lm


def train_lm(model, train_data, val_data, n_epochs=20, batch_size=64, lr=3e-3, name=""):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n = train_data.shape[0]
    t0 = time.time()
    best_bpc = float("inf"); best_acc = 0.0
    for ep in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0; nb = 0
        for i in range(0, n, batch_size):
            batch = train_data[perm[i:i + batch_size]]
            logits = model(batch)
            loss, _ = shifted_loss(logits, batch, loss_fn)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item(); nb += 1
        if ep % 5 == 0 or ep == 1 or ep == n_epochs:
            model.eval()
            with torch.no_grad():
                vl, va = shifted_loss(model(val_data), val_data, loss_fn)
            vbpc = vl.item() / math.log(2)
            best_bpc = min(best_bpc, vbpc); best_acc = max(best_acc, va)
            print(f"  [{name}] ep {ep:3d}  train_bpc {total_loss/nb/math.log(2):.3f}  "
                  f"val_bpc {vbpc:.3f}  val_acc {va:.3f}  ({time.time()-t0:.0f}s)")
    return best_bpc, best_acc


def eval_lm(model, val_data):
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        l, acc = shifted_loss(model(val_data), val_data, loss_fn)
    return l.item() / math.log(2), acc


def main():
    torch.manual_seed(0)
    train_data, val_data, stoi, _ = make_shakespeare_splits(3000, 500, 64)
    vocab = len(stoi)

    print("\n=== Phase 1: train MLP-RNN baseline (single-leaf cell) ===")
    mlp_lm = TreeLMv2MLPRNN(vocab, embed_dim=32, state_dim=128, depth=3,
                            relation_dim=64, context_dim=64, max_len=64)
    mlp_params = sum(p.numel() for p in mlp_lm.parameters())
    print(f"MLPRNN-LM params: {mlp_params:,}")
    mlp_bpc, mlp_acc = train_lm(mlp_lm, train_data, val_data, n_epochs=20, name="mlprnn")

    print("\n=== Phase 2: warm-start TLM from MLP-RNN, verify init parity ===")
    tlm_warm = warm_start_tree_from_mlprnn(mlp_lm, depth=3)
    warm_params = sum(p.numel() for p in tlm_warm.parameters())
    print(f"warmed TLM params: {warm_params:,}")
    init_bpc, init_acc = eval_lm(tlm_warm, val_data)
    print(f"warm-start init val_bpc={init_bpc:.3f} (seed final {mlp_bpc:.3f}) "
          f"-- diff {init_bpc - mlp_bpc:+.3f}")

    print("\n=== Phase 3: fine-tune the warm-started TLM ===")
    warm_bpc, warm_acc = train_lm(tlm_warm, train_data, val_data, n_epochs=15, lr=1e-3, name="warm")

    print("\n=== Phase 4: from-scratch TLM control ===")
    torch.manual_seed(0)
    scratch_lm = TreeLMv2SharedForget(vocab, embed_dim=32, state_dim=128, depth=3,
                                      relation_dim=64, context_dim=64, max_len=64)
    scratch_params = sum(p.numel() for p in scratch_lm.parameters())
    print(f"scratch TLM params: {scratch_params:,}")
    scratch_bpc, scratch_acc = train_lm(scratch_lm, train_data, val_data, n_epochs=20, name="scratch")

    print("\n=== SUMMARY ===")
    print(f"{'model':<28} {'params':>10} {'val_bpc':>9} {'val_acc':>9}")
    print(f"{'MLPRNN baseline (seed)':<28} {mlp_params:>10,} {mlp_bpc:>9.3f} {mlp_acc:>9.3f}")
    print(f"{'warm-started TLM at init':<28} {warm_params:>10,} {init_bpc:>9.3f} {init_acc:>9.3f}")
    print(f"{'warm-started TLM fine-tuned':<28} {warm_params:>10,} {warm_bpc:>9.3f} {warm_acc:>9.3f}")
    print(f"{'from-scratch TLM control':<28} {scratch_params:>10,} {scratch_bpc:>9.3f} {scratch_acc:>9.3f}")

    print("\n=== INTERPRETATION ===")
    print(f"  H1 (warm init = seed):   diff {init_bpc - mlp_bpc:+.3f}  "
          f"({'OK' if abs(init_bpc - mlp_bpc) < 0.1 else 'FAILED'})")
    print(f"  H2 (warm-tuned > seed):  delta {warm_bpc - mlp_bpc:+.3f}  "
          f"({'YES tree improves over seed' if warm_bpc < mlp_bpc else 'NO tree fails to improve'})")
    print(f"  warm-tuned vs from-scratch tree:  delta {warm_bpc - scratch_bpc:+.3f}  "
          f"({'warm beats scratch' if warm_bpc < scratch_bpc else 'scratch wins'})")


if __name__ == "__main__":
    main()
