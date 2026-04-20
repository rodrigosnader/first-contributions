"""Experiment 13: real sparse hard inference with FLOP accounting.

Unlike exp4 (which computed all leaves then indexed - no savings), this
implements genuinely sparse hard forward: gather only the reached leaf's
weight row per example, compute one matmul per example instead of 2^depth.

Measures on both TreeLMv2 (baseline per-leaf) and TreeLMv2Shared
(weight-shared backbone) at batch=1 (real single-example inference, the
regime that matters for CPU deployment) and batch=32.

Metric: val BPC (soft vs hard), wall-clock time, and effective speedup.
Also logs routing confidence (avg |logit|) - predicts whether hard will
match soft."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import torch
import torch.nn as nn

from lm_train import LMConfig, run_lm, shifted_loss
from lm_model import TreeLMv2, TreeLMv2Shared
from model import SoftTree, SharedBackboneSoftTree


def set_hard_mode(model, hard: bool):
    for m in model.modules():
        if isinstance(m, (SoftTree, SharedBackboneSoftTree)):
            m._hard = hard


def routing_confidence(model, data):
    """Average |logit| across all routing decisions on the val data.
    Larger = more confident routing = hard inference more likely to match soft."""
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
                # each forward stores many entries (one per call / per step); average them
                if hook_out:
                    abs_means = hook_out[::2]
                    counts = hook_out[1::2]
                    total_abs += sum(a * c for a, c in zip(abs_means, counts))
                    total_n += sum(counts)
    return total_abs / total_n if total_n else 0.0


def bench_and_eval(model, val_data, batch_sizes=(1, 32), hard=False, n_runs=2):
    set_hard_mode(model, hard)
    model.eval()
    vocab_size = model.head.tree.output_dim if hasattr(model.head, "tree") else model.head.out_features
    loss_fn = nn.CrossEntropyLoss()

    # accuracy / BPC on full val
    with torch.no_grad():
        logits = model(val_data)
        loss, acc = shifted_loss(logits, val_data, loss_fn)
    bpc = loss.item() / math.log(2)

    # timing at each batch size
    times = {}
    for bs in batch_sizes:
        with torch.no_grad():
            subset = val_data[:max(bs * 4, bs)]  # warm + measure
            for _ in range(2):  # warm
                for i in range(0, len(subset), bs):
                    _ = model(subset[i:i + bs])
            best = float("inf")
            for _ in range(n_runs):
                t0 = time.time()
                for i in range(0, len(subset), bs):
                    _ = model(subset[i:i + bs])
            best = min(best, time.time() - t0)
            times[bs] = best / (len(subset) / bs)  # time per batch
    set_hard_mode(model, False)
    return bpc, acc, times


def measure(model_cls, name, depth=3):
    cfg = LMConfig(seq_len=64, state_dim=128, depth=depth, relation_dim=64,
                   context_dim=64, epochs=20, n_train=3000, name=name)
    r = run_lm(cfg, model_ctor=model_cls)
    model = r["model"]
    val = r["val_data"]

    conf = routing_confidence(model, val[:32])
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
        "name": name, "conf": conf,
        "soft_bpc": soft_bpc, "soft_acc": soft_acc,
        "hard_bpc": hard_bpc, "hard_acc": hard_acc,
        "soft_times": soft_times, "hard_times": hard_times,
    }


if __name__ == "__main__":
    results = []
    print("=" * 40)
    print("baseline tree (per-leaf Linears)")
    print("=" * 40)
    results.append(measure(TreeLMv2, "tree_baseline"))

    print("\n" + "=" * 40)
    print("shared-backbone tree (exp10 winner)")
    print("=" * 40)
    results.append(measure(TreeLMv2Shared, "tree_shared"))

    print("\n=== SUMMARY ===")
    print(f"{'variant':<16} {'conf':>6} {'soft_bpc':>9} {'hard_bpc':>9} "
          f"{'soft_b1_ms':>11} {'hard_b1_ms':>11} {'speedup':>8}")
    for r in results:
        print(f"{r['name']:<16} {r['conf']:>6.2f} {r['soft_bpc']:>9.3f} "
              f"{r['hard_bpc']:>9.3f} {r['soft_times'][1]*1000:>11.2f} "
              f"{r['hard_times'][1]*1000:>11.2f} {r['soft_times'][1]/r['hard_times'][1]:>7.2f}x")
