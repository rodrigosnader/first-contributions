"""Experiment 25: TLM hard vs Transformer KV-cache inference speed.

At generation time:
  - TLM hard per step: O(state^2) for cell matmul. Constant per step.
  - Transformer with KV cache per step: O(layers * (d^2 + L * d)) where L
    is the growing context length. Cost grows with generated length.

Predict crossover: for short generations transformer wins, for long ones
TLM wins because per-step cost is constant while transformer grows linearly.

Bench: generate N=64, 256, 1024 new tokens starting from a small prompt,
batch=1, single thread, multiple model sizes. Pure inference speed (no
training involved). Use freshly initialized models - weights don't
affect timing.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import importlib.util, os

from lm_model import TreeLMv2SharedForget
from model import SoftTree, SharedBackboneSoftTree

spec22 = importlib.util.spec_from_file_location(
    "exp22_mod",
    os.path.join(os.path.dirname(__file__), "exp22_transformer.py"),
)
exp22 = importlib.util.module_from_spec(spec22); spec22.loader.exec_module(exp22)
TransformerLM = exp22.TransformerLM


def set_hard_mode(model, hard):
    for m in model.modules():
        if isinstance(m, (SoftTree, SharedBackboneSoftTree)):
            m._hard = hard


def bench(fn, n_runs=3, n_warm=1):
    for _ in range(n_warm):
        fn()
    best = float("inf")
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def time_tree_generation(model, prompt, n_new_tokens):
    model.eval()
    set_hard_mode(model, True)
    with torch.no_grad():
        t0 = time.perf_counter()
        out = model.generate(prompt, n_new_tokens, temperature=1.0)
        dt = time.perf_counter() - t0
    set_hard_mode(model, False)
    return dt


def time_transformer_generation(model, prompt, n_new_tokens):
    """Slow path: re-encodes full context each step (no KV cache)."""
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        out = model.generate(prompt, n_new_tokens, temperature=1.0)
        dt = time.perf_counter() - t0
    return dt


def time_transformer_kv_generation(model, prompt, n_new_tokens):
    """KV-cache path: prompt prefilled once, 1-token forward per step."""
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        out = model.generate_kv(prompt, n_new_tokens, temperature=1.0)
        dt = time.perf_counter() - t0
    return dt


def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)

    VOCAB = 256
    configs = [
        ("small",  {"state_dim": 128, "d_model": 128}),
        ("medium", {"state_dim": 256, "d_model": 192}),
        ("large",  {"state_dim": 512, "d_model": 256}),
    ]
    gen_lens = [64, 256, 1024]

    print(f"{'config':<8} {'gen':>5} {'tree_s':>9} {'txf_kv_s':>9} "
          f"{'tree_tok/s':>11} {'txf_kv_tok/s':>13} {'ratio':>8}")
    for cfg_name, cfg in configs:
        state = cfg["state_dim"]; dmodel = cfg["d_model"]
        tree = TreeLMv2SharedForget(VOCAB, embed_dim=32, state_dim=state,
                                    depth=3, relation_dim=64, context_dim=64,
                                    max_len=2048)
        txf = TransformerLM(VOCAB, max_len=2048, d_model=dmodel,
                            n_heads=4, n_layers=4, ff_mult=3)
        tree_params = sum(p.numel() for p in tree.parameters())
        txf_params = sum(p.numel() for p in txf.parameters())
        prompt = torch.randint(0, VOCAB, (1, 8))

        for n in gen_lens:
            t_tree = bench(lambda: time_tree_generation(tree, prompt, n))
            t_kv = bench(lambda: time_transformer_kv_generation(txf, prompt, n))
            tree_rate = n / t_tree
            kv_rate = n / t_kv
            print(f"{cfg_name:<8} {n:>5} {t_tree:>9.3f} {t_kv:>9.3f} "
                  f"{tree_rate:>11.1f} {kv_rate:>13.1f} {tree_rate/kv_rate:>7.2f}x")

        print(f"          tree params: {tree_params:,}  txf params: {txf_params:,}\n")


if __name__ == "__main__":
    main()
