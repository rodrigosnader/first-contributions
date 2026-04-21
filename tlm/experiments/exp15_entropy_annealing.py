"""Experiment 15: routing entropy annealing to make soft-trained models
tolerate hard inference.

Diagnosis from exp13: routing sigmoids train to |logit| avg ~2.0 (sigmoid
~0.88/0.12, semi-committed). Flipping the uncertain ~12% to hard-argmax
cascades through recurrent steps; BPC collapses from 2.7 soft to 5.2 hard.

Fix attempted here: add a per-routing-decision binary entropy penalty
  H(p) = -[ p log p + (1 - p) log(1 - p) ]
averaged over all routing decisions from every SoftTree /
SharedBackboneSoftTree in the model. Total loss:
  loss = cross_entropy + entropy_weight * entropy_mean
Anneal entropy_weight linearly: 0 for epochs 1-5 (let the routing find
useful splits under soft mixing), ramp to target over epochs 6-20, full
target for epochs 21-25. Target tried: 0, 0.05, 0.1.

Hypothesis: at target=0.1 the entropy penalty drives sigmoids to 0/1,
hard == soft, and the exp13 failure disappears.

Tested on the two architectures exp13 / exp14 highlighted:
  TreeLMv2Shared        (exp10 short-context winner)
  TreeLMv2SharedForget  (exp14 long-context equivalent of LSTM)
"""
import sys
import time
import math
import inspect
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from data import make_shakespeare_splits
from lm_train import LMConfig, shifted_loss
from lm_model import TreeLMv2Shared, TreeLMv2SharedForget
from model import SoftTree, SharedBackboneSoftTree


# ---------- entropy annealing schedule ----------

def entropy_weight_schedule(epoch: int, total_epochs: int, target: float) -> float:
    """0 for epochs 1..5, linear ramp 6..20, full target after.
    Returns the weight to use at the START of `epoch` (1-indexed)."""
    warmup_end = 5
    ramp_end = 20
    if epoch <= warmup_end:
        return 0.0
    if epoch >= ramp_end:
        return target
    # linear ramp over (warmup_end, ramp_end]
    frac = (epoch - warmup_end) / (ramp_end - warmup_end)
    return target * frac


# ---------- collect routing probs across the whole model ----------

def collect_routing_probs(model):
    probs = []
    for m in model.modules():
        if isinstance(m, (SoftTree, SharedBackboneSoftTree)):
            if m._last_routing_probs is not None:
                probs.append(m._last_routing_probs.reshape(-1))
    if not probs:
        return None
    return torch.cat(probs)


def binary_entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-7, 1 - 1e-7)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def set_hard_mode(model, hard: bool):
    for m in model.modules():
        if isinstance(m, (SoftTree, SharedBackboneSoftTree)):
            m._hard = hard


# ---------- training loop with entropy annealing ----------

def run_lm_with_entropy(cfg: LMConfig, model_ctor, target_entropy_weight: float):
    """Same as lm_train.run_lm but adds a binary-entropy penalty on every
    SoftTree's routing probabilities, with a linear anneal schedule."""
    torch.manual_seed(cfg.seed)

    train_data, val_data, stoi, itos = make_shakespeare_splits(
        cfg.n_train, cfg.n_val, cfg.seq_len
    )
    vocab_size = len(stoi)

    sig = inspect.signature(model_ctor)
    kwargs = dict(embed_dim=cfg.embed_dim, state_dim=cfg.state_dim,
                  depth=cfg.depth, relation_dim=cfg.relation_dim, max_len=cfg.seq_len)
    if "context_dim" in sig.parameters:
        kwargs["context_dim"] = cfg.context_dim
    model = model_ctor(vocab_size, **kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"=== {cfg.name} ===")
    print(f"model: {model.__class__.__name__}, params: {n_params:,}")
    print(f"vocab={vocab_size}, seq_len={cfg.seq_len}, state={cfg.state_dim}")
    print(f"entropy target weight: {target_entropy_weight}")
    print(f"random baseline BPC: {math.log2(vocab_size):.3f}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    t0 = time.time()
    best_val_bpc = float("inf")
    best_val_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        w = entropy_weight_schedule(epoch, cfg.epochs, target_entropy_weight)
        model.train()
        perm = torch.randperm(cfg.n_train)
        total_loss = 0.0
        total_acc = 0.0
        total_ent = 0.0
        n_batches = 0
        for i in range(0, cfg.n_train, cfg.batch_size):
            batch = train_data[perm[i:i + cfg.batch_size]]
            logits = model(batch)
            ce_loss, acc = shifted_loss(logits, batch, loss_fn)
            probs = collect_routing_probs(model)
            if probs is not None:
                ent_mean = binary_entropy(probs).mean()
            else:
                ent_mean = torch.tensor(0.0)
            loss = ce_loss + w * ent_mean
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total_loss += ce_loss.item()
            total_acc += acc
            total_ent += ent_mean.item()
            n_batches += 1

        if epoch % cfg.log_every == 0 or epoch == 1 or epoch == cfg.epochs:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_data)
                val_loss, val_acc = shifted_loss(val_logits, val_data, loss_fn)
            train_bpc = total_loss / n_batches / math.log(2)
            val_bpc = val_loss.item() / math.log(2)
            best_val_bpc = min(best_val_bpc, val_bpc)
            best_val_acc = max(best_val_acc, val_acc)
            avg_ent = total_ent / n_batches
            print(f"epoch {epoch:3d} | w={w:.3f} | train BPC {train_bpc:.3f} "
                  f"acc {total_acc/n_batches:.3f} ent {avg_ent:.3f} "
                  f"| val BPC {val_bpc:.3f} acc {val_acc:.3f} | {time.time()-t0:.0f}s")

    return {
        "name": cfg.name, "n_params": n_params,
        "best_val_bpc": best_val_bpc, "best_val_acc": best_val_acc,
        "time_s": time.time() - t0, "model": model,
        "val_data": val_data, "train_data": train_data,
        "stoi": stoi, "itos": itos, "vocab_size": vocab_size,
        "target_entropy_weight": target_entropy_weight,
    }


# ---------- eval: soft vs hard BPC + routing |logit| + timing ----------

def routing_abs_logit(model, data):
    total_abs = 0.0
    total_n = 0
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (SoftTree, SharedBackboneSoftTree)):
                hook_out = []
                def hook(module, inp, out, store=hook_out):
                    store.append(out.abs().mean().item())
                    store.append(out.numel())
                h = m.router.register_forward_hook(hook)
                _ = model(data)
                h.remove()
                if hook_out:
                    abs_means = hook_out[::2]
                    counts = hook_out[1::2]
                    total_abs += sum(a * c for a, c in zip(abs_means, counts))
                    total_n += sum(counts)
    return total_abs / total_n if total_n else 0.0


def bench_and_eval(model, val_data, batch_sizes=(1, 32), hard=False, n_runs=2):
    set_hard_mode(model, hard)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        logits = model(val_data)
        loss, acc = shifted_loss(logits, val_data, loss_fn)
    bpc = loss.item() / math.log(2)

    times = {}
    for bs in batch_sizes:
        with torch.no_grad():
            subset = val_data[:max(bs * 4, bs)]
            for _ in range(2):  # warm
                for i in range(0, len(subset), bs):
                    _ = model(subset[i:i + bs])
            best = float("inf")
            for _ in range(n_runs):
                t0 = time.time()
                for i in range(0, len(subset), bs):
                    _ = model(subset[i:i + bs])
                best = min(best, time.time() - t0)
            times[bs] = best / (len(subset) / bs)
    set_hard_mode(model, False)
    return bpc, acc, times


# ---------- experiment runner ----------

def measure(model_cls, model_label, target_entropy_weight):
    name = f"{model_label}_ent{target_entropy_weight}"
    cfg = LMConfig(seq_len=64, state_dim=128, depth=3, relation_dim=64,
                   context_dim=64, epochs=25, n_train=3000, name=name)
    r = run_lm_with_entropy(cfg, model_cls, target_entropy_weight)
    model = r["model"]
    val = r["val_data"]

    conf = routing_abs_logit(model, val[:32])
    soft_bpc, soft_acc, soft_times = bench_and_eval(model, val, hard=False)
    hard_bpc, hard_acc, hard_times = bench_and_eval(model, val, hard=True)

    print(f"\n--- {name} hard-inference report ---")
    print(f"routing |logit| avg: {conf:.3f}  (higher = more committed)")
    print(f"soft: BPC {soft_bpc:.3f}  acc {soft_acc:.3f}")
    print(f"hard: BPC {hard_bpc:.3f}  acc {hard_acc:.3f}")
    print(f"acc drop: {soft_acc - hard_acc:+.3f}  bpc delta: {hard_bpc - soft_bpc:+.3f}")
    for bs in (1, 32):
        s = soft_times[bs]
        h = hard_times[bs]
        print(f"bs={bs:2d}: soft {s*1000:.2f}ms/batch, hard {h*1000:.2f}ms/batch, "
              f"speedup {s/h:.2f}x")
    return {
        "name": name, "model_label": model_label,
        "target_entropy_weight": target_entropy_weight,
        "conf": conf,
        "soft_bpc": soft_bpc, "soft_acc": soft_acc,
        "hard_bpc": hard_bpc, "hard_acc": hard_acc,
        "soft_times": soft_times, "hard_times": hard_times,
        "n_params": r["n_params"],
    }


if __name__ == "__main__":
    variants = [
        ("shared", TreeLMv2Shared),
        ("sharedforget", TreeLMv2SharedForget),
    ]
    targets = [0.0, 0.05, 0.1]

    all_results = []
    for model_label, model_cls in variants:
        print("\n" + "=" * 60)
        print(f"{model_label}  ({model_cls.__name__})")
        print("=" * 60)
        for target in targets:
            print(f"\n--- target entropy weight = {target} ---")
            all_results.append(measure(model_cls, model_label, target))

    print("\n=== SUMMARY: entropy annealing vs hard-inference gap ===")
    print(f"{'model':<14} {'ent_w':>6} {'conf':>6} "
          f"{'soft_bpc':>9} {'hard_bpc':>9} {'gap':>7} "
          f"{'soft_acc':>9} {'hard_acc':>9} "
          f"{'b1_soft':>9} {'b1_hard':>9} {'spd':>6}")
    for r in all_results:
        gap = r["hard_bpc"] - r["soft_bpc"]
        print(f"{r['model_label']:<14} {r['target_entropy_weight']:>6.2f} "
              f"{r['conf']:>6.2f} "
              f"{r['soft_bpc']:>9.3f} {r['hard_bpc']:>9.3f} {gap:>+7.3f} "
              f"{r['soft_acc']:>9.3f} {r['hard_acc']:>9.3f} "
              f"{r['soft_times'][1]*1000:>9.2f} {r['hard_times'][1]*1000:>9.2f} "
              f"{r['soft_times'][1]/r['hard_times'][1]:>5.2f}x")

    print("\n=== hypothesis check ===")
    print("(1) routing |logit| should grow MUCH with entropy_weight target")
    print("(2) hard_bpc - soft_bpc gap should shrink toward 0 with target=0.1")
    for model_label, _ in variants:
        rows = [r for r in all_results if r["model_label"] == model_label]
        print(f"\n  {model_label}:")
        for r in rows:
            gap = r["hard_bpc"] - r["soft_bpc"]
            print(f"    target={r['target_entropy_weight']:.2f}  "
                  f"|logit|={r['conf']:.2f}  gap={gap:+.3f}")
