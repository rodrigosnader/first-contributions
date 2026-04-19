"""Experiment 4: hard inference vs soft inference. Train soft, then at
inference time pick only the argmax branch at each routing node. The big
TLM hypothesis: hard routing preserves accuracy while cutting compute
dramatically (only one leaf evaluated per tree instead of all 2^depth)."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from train import Config, run
from model import SoftTree, TreeCell, TreeEncoder, TreeDecoder, TreeAutoencoder


def hard_forward_soft_tree(tree: SoftTree, x: torch.Tensor) -> torch.Tensor:
    """Hard-argmax routing: traverse one path per example, evaluate only the
    reached leaf. Loses gradient but is what production inference would use."""
    batch = x.shape[0]
    logits = tree.router(x)
    leaf_idx = torch.zeros(batch, dtype=torch.long, device=x.device)
    node_idx = 0
    for level in range(tree.depth):
        n_at_level = 2 ** level
        node_logits = logits[:, node_idx:node_idx + n_at_level]
        cur = torch.gather(node_logits, 1, leaf_idx.unsqueeze(1)).squeeze(1)
        go_left = (cur > 0).long()
        leaf_idx = leaf_idx * 2 + (1 - go_left)
        node_idx += n_at_level
    leaf_outs = tree.leaves(x).reshape(batch, tree.n_leaves, tree.output_dim)
    return leaf_outs[torch.arange(batch), leaf_idx]


def patch_hard(model: nn.Module):
    """Replace forward() on every SoftTree with hard routing."""
    for m in model.modules():
        if isinstance(m, SoftTree):
            m._soft_forward = m.forward
            m.forward = lambda x, _m=m: hard_forward_soft_tree(_m, x)


def unpatch(model: nn.Module):
    for m in model.modules():
        if isinstance(m, SoftTree) and hasattr(m, "_soft_forward"):
            m.forward = m._soft_forward
            del m._soft_forward


def accuracy(logits, targets):
    return (logits.argmax(-1) == targets).float().mean().item()


def bench(model, data, n_runs=3):
    model.eval()
    with torch.no_grad():
        for _ in range(2):
            _ = model(data[:32])
        times = []
        for _ in range(n_runs):
            t0 = time.time()
            for i in range(0, len(data), 32):
                _ = model(data[i:i+32])
            times.append(time.time() - t0)
    return min(times)


cfg = Config(seq_len=16, epochs=30, n_train=3000, n_val=500, name="train_soft")
r = run(cfg)
model = r["model"]
val = r["val_data"]

with torch.no_grad():
    soft_logits = model(val)
soft_acc = accuracy(soft_logits, val)
soft_time = bench(model, val)

patch_hard(model)
with torch.no_grad():
    hard_logits = model(val)
hard_acc = accuracy(hard_logits, val)
hard_time = bench(model, val)
unpatch(model)

print(f"\n=== SUMMARY ===")
print(f"soft inference: val_acc={soft_acc:.3f}  batch_time={soft_time:.3f}s")
print(f"hard inference: val_acc={hard_acc:.3f}  batch_time={hard_time:.3f}s")
print(f"accuracy drop: {soft_acc - hard_acc:+.3f}")
print(f"speedup: {soft_time / hard_time:.2f}x")
