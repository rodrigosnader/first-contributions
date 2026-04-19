"""Next-token language model training. Targets are inputs shifted by 1.
Reports bits-per-character (BPC) - the standard char-level LM metric.
Random baseline for vocab=65 is log2(65) ~= 6.02 BPC."""
import math
import time
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn

from data import make_shakespeare_splits, decode
from lm_model import TreeLM


@dataclass
class LMConfig:
    seq_len: int = 64
    embed_dim: int = 32
    state_dim: int = 128
    depth: int = 4
    relation_dim: int = 64
    context_dim: int = 64
    n_train: int = 4000
    n_val: int = 500
    batch_size: int = 64
    epochs: int = 30
    lr: float = 3e-3
    grad_clip: float = 5.0
    seed: int = 0
    log_every: int = 5
    name: str = "default"


def shifted_loss(logits: torch.Tensor, tokens: torch.Tensor, loss_fn) -> tuple[torch.Tensor, float]:
    """logits[:, t] predicts tokens[:, t+1]. Drop the last logit and the
    first token to align."""
    pred = logits[:, :-1].reshape(-1, logits.shape[-1])
    target = tokens[:, 1:].reshape(-1)
    loss = loss_fn(pred, target)
    acc = (pred.argmax(-1) == target).float().mean().item()
    return loss, acc


def run_lm(cfg: LMConfig, model_ctor=TreeLM):
    torch.manual_seed(cfg.seed)

    train_data, val_data, stoi, itos = make_shakespeare_splits(
        cfg.n_train, cfg.n_val, cfg.seq_len
    )
    vocab_size = len(stoi)

    import inspect
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
    print(f"random baseline BPC: {math.log2(vocab_size):.3f}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    t0 = time.time()
    best_val_bpc = float("inf")
    best_val_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
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

        if epoch % cfg.log_every == 0 or epoch == 1 or epoch == cfg.epochs:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_data)
                val_loss, val_acc = shifted_loss(val_logits, val_data, loss_fn)
            train_bpc = total_loss / n_batches / math.log(2)
            val_bpc = val_loss.item() / math.log(2)
            best_val_bpc = min(best_val_bpc, val_bpc)
            best_val_acc = max(best_val_acc, val_acc)
            print(f"epoch {epoch:3d} | train BPC {train_bpc:.3f} acc {total_acc/n_batches:.3f} "
                  f"| val BPC {val_bpc:.3f} acc {val_acc:.3f} | {time.time()-t0:.0f}s")

    # generation samples
    model.eval()
    seed_text = "ROMEO:\n"
    seed_ids = torch.tensor([[stoi[c] for c in seed_text]], dtype=torch.long)
    gen_ids = model.generate(seed_ids, n_new_tokens=200, temperature=0.8)
    gen_text = "".join(itos[int(t)] for t in gen_ids[0])
    print("\n--- generated continuation (200 chars from seed) ---")
    print(gen_text)
    print("--- end ---")

    return {
        "name": cfg.name, "n_params": n_params,
        "best_val_bpc": best_val_bpc, "best_val_acc": best_val_acc,
        "time_s": time.time() - t0, "model": model,
    }


if __name__ == "__main__":
    run_lm(LMConfig(name="default_lm"))
