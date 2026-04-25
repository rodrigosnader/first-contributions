"""Experiment 28: speed-vs-quality Pareto frontier - does TLM-hard dominate LSTM?

Train TLM with curriculum (soft warmup -> STE) at several state_dim sizes,
LSTM at matched and larger sizes. For each model evaluate:
- val_bpc on Shakespeare
- inference tokens/sec at batch=1, single thread

Plot/list (speed, bpc) pairs. The selling story is:
  for any speed budget, TLM-hard achieves equal or better bpc than LSTM.

If TLM-hard dominates the LSTM curve, the project's value prop is
solid even with the -9.5pp soft-vs-hard tax (exp26).
"""
import sys
import time
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from data import make_shakespeare_splits
from lm_train import LMConfig, shifted_loss
from lm_model import TreeLMv2SharedForget, LSTMLMv2
from model import SoftTree, SharedBackboneSoftTree

# Reuse curriculum trainer from exp26
import importlib.util, os
spec = importlib.util.spec_from_file_location(
    "exp26_mod",
    os.path.join(os.path.dirname(__file__), "exp26_alignment_strategies.py"),
)
exp26 = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp26)
set_routing_mode = exp26.set_routing_mode
set_hard_mode = exp26.set_hard_mode


def train_tlm_curriculum(state_dim, n_epochs=20, n_train=3000, seq_len=64):
    torch.manual_seed(0)
    train_data, val_data, stoi, _ = make_shakespeare_splits(n_train, 500, seq_len)
    vocab = len(stoi)
    model = TreeLMv2SharedForget(vocab, embed_dim=32, state_dim=state_dim, depth=3,
                                 relation_dim=64, context_dim=64, max_len=seq_len)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    half = n_epochs // 2
    for epoch in range(1, n_epochs + 1):
        mode = "soft" if epoch <= half else "ste"
        set_routing_mode(model, mode)
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, 64):
            batch = train_data[perm[i:i + 64]]
            logits = model(batch)
            loss, _ = shifted_loss(logits, batch, loss_fn)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    # eval
    model.eval()
    set_routing_mode(model, "soft", tau=0.1)
    set_hard_mode(model, False)
    with torch.no_grad():
        sl, sa = shifted_loss(model(val_data), val_data, loss_fn)
    soft_bpc = sl.item() / math.log(2); soft_acc = sa
    set_hard_mode(model, True)
    with torch.no_grad():
        hl, ha = shifted_loss(model(val_data), val_data, loss_fn)
    hard_bpc = hl.item() / math.log(2); hard_acc = ha
    set_hard_mode(model, False)
    return model, n_params, vocab, val_data, soft_bpc, soft_acc, hard_bpc, hard_acc


def train_lstm(state_dim, n_epochs=20, n_train=3000, seq_len=64):
    torch.manual_seed(0)
    train_data, val_data, stoi, _ = make_shakespeare_splits(n_train, 500, seq_len)
    vocab = len(stoi)
    model = LSTMLMv2(vocab, embed_dim=32, state_dim=state_dim,
                     relation_dim=64, context_dim=64, max_len=seq_len)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, 64):
            batch = train_data[perm[i:i + 64]]
            logits = model(batch)
            loss, _ = shifted_loss(logits, batch, loss_fn)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        l, acc = shifted_loss(model(val_data), val_data, loss_fn)
    bpc = l.item() / math.log(2)
    return model, n_params, vocab, val_data, bpc, acc


def time_generation(generate_fn, prompt, n_tokens, n_runs=2, n_warm=1):
    for _ in range(n_warm):
        generate_fn(prompt, n_tokens)
    best = float("inf")
    for _ in range(n_runs):
        t0 = time.perf_counter()
        generate_fn(prompt, n_tokens)
        best = min(best, time.perf_counter() - t0)
    return best


def bench_tlm(model, vocab, hard, n_tokens=256):
    set_hard_mode(model, hard)
    model.eval()
    prompt = torch.randint(0, vocab, (1, 8))
    fn = lambda p, n: model.generate(p, n, temperature=1.0)
    with torch.no_grad():
        dt = time_generation(fn, prompt, n_tokens)
    set_hard_mode(model, False)
    return n_tokens / dt


def bench_lstm(model, vocab, n_tokens=256):
    model.eval()
    prompt = torch.randint(0, vocab, (1, 8))
    fn = lambda p, n: model.generate(p, n, temperature=1.0)
    with torch.no_grad():
        dt = time_generation(fn, prompt, n_tokens)
    return n_tokens / dt


def main():
    torch.set_num_threads(1)
    points = []  # (label, params, val_bpc, val_acc, tok_per_s)

    print("=== training TLMs (curriculum) ===")
    for sd in [128, 256]:
        print(f"  TLM curriculum state={sd} ...")
        t0 = time.time()
        model, params, vocab, val_data, sb, sa, hb, ha = train_tlm_curriculum(sd)
        print(f"    soft: BPC {sb:.3f} acc {sa:.3f}  |  hard: BPC {hb:.3f} acc {ha:.3f}  "
              f"({time.time()-t0:.0f}s)")
        # bench both modes
        soft_rate = bench_tlm(model, vocab, hard=False)
        hard_rate = bench_tlm(model, vocab, hard=True)
        points.append((f"tlm{sd}_soft", params, sb, sa, soft_rate))
        points.append((f"tlm{sd}_hard", params, hb, ha, hard_rate))
        print(f"    soft rate {soft_rate:.0f} tok/s  |  hard rate {hard_rate:.0f} tok/s")

    print("\n=== training LSTMs ===")
    for sd in [128, 192, 256, 384]:
        print(f"  LSTM state={sd} ...")
        t0 = time.time()
        model, params, vocab, val_data, bpc, acc = train_lstm(sd)
        rate = bench_lstm(model, vocab)
        print(f"    BPC {bpc:.3f}  acc {acc:.3f}  rate {rate:.0f} tok/s  ({time.time()-t0:.0f}s)")
        points.append((f"lstm{sd}", params, bpc, acc, rate))

    print("\n=== PARETO TABLE (sorted by tok/s descending) ===")
    points.sort(key=lambda p: -p[4])
    print(f"{'model':<14} {'params':>10} {'val_bpc':>9} {'val_acc':>9} {'tok/s':>8}")
    for label, params, bpc, acc, rate in points:
        print(f"{label:<14} {params:>10,} {bpc:>9.3f} {acc:>9.3f} {rate:>8.1f}")

    print("\n=== PARETO FRONTIER (sorted by tok/s, best bpc per row keeps the model alive) ===")
    points.sort(key=lambda p: -p[4])
    best_bpc = float("inf")
    frontier = []
    for label, params, bpc, acc, rate in points:
        # any model with strictly worse bpc AND lower speed is dominated
        # walk from fastest to slowest, keep models that improve bpc
        if bpc < best_bpc:
            frontier.append((label, params, bpc, acc, rate))
            best_bpc = bpc
    print(f"{'model':<14} {'params':>10} {'val_bpc':>9} {'tok/s':>8}")
    for label, params, bpc, acc, rate in frontier:
        print(f"{label:<14} {params:>10,} {bpc:>9.3f} {rate:>8.1f}")


if __name__ == "__main__":
    main()
