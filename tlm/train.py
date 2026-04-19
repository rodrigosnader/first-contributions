import time
from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn

from data import make_shakespeare_splits, decode
from model import TreeAutoencoder


@dataclass
class Config:
    seq_len: int = 16
    embed_dim: int = 32
    state_dim: int = 128
    depth: int = 4
    n_train: int = 4000
    n_val: int = 500
    batch_size: int = 64
    epochs: int = 40
    lr: float = 3e-3
    grad_clip: float = 5.0
    seed: int = 0
    log_every: int = 5
    name: str = "default"


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def run(cfg: Config, model_ctor=TreeAutoencoder, extra_model_kwargs: Optional[dict] = None,
        data_fn=None, decode_fn=None):
    """data_fn(n_train, n_val, seq_len) -> (train, val, vocab_size, decoder_ctx)
    decode_fn(tokens, ctx) -> list[str] for printing reconstructions."""
    torch.manual_seed(cfg.seed)
    extra_model_kwargs = extra_model_kwargs or {}

    if data_fn is None:
        train_data, val_data, stoi, itos = make_shakespeare_splits(
            cfg.n_train, cfg.n_val, cfg.seq_len
        )
        vocab_size = len(stoi)
        ctx = itos
        _decode = lambda t, c: decode(t, c)
    else:
        train_data, val_data, vocab_size, ctx = data_fn(cfg.n_train, cfg.n_val, cfg.seq_len)
        _decode = decode_fn

    majority = torch.bincount(train_data.reshape(-1), minlength=vocab_size).argmax()
    baseline_acc = (val_data == majority).float().mean().item()

    model = model_ctor(vocab_size, cfg.embed_dim, cfg.state_dim, cfg.depth,
                       max_len=cfg.seq_len, **extra_model_kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    info_bits = cfg.seq_len * torch.log2(torch.tensor(float(vocab_size))).item()

    print(f"=== {cfg.name} ===")
    print(f"model params: {n_params:,}")
    print(f"vocab={vocab_size}, seq_len={cfg.seq_len}, embed={cfg.embed_dim}, "
          f"state={cfg.state_dim}, depth={cfg.depth}")
    print(f"bottleneck: {cfg.state_dim} floats vs {info_bits:.1f} bits per window")
    print(f"majority baseline val acc: {baseline_acc:.3f}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    t0 = time.time()
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
            loss = loss_fn(logits.reshape(-1, vocab_size), batch.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total_loss += loss.item()
            total_acc += accuracy(logits, batch)
            n_batches += 1

        if epoch % cfg.log_every == 0 or epoch == 1 or epoch == cfg.epochs:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_data)
                val_loss = loss_fn(val_logits.reshape(-1, vocab_size), val_data.reshape(-1)).item()
                val_acc = accuracy(val_logits, val_data)
            best_val_acc = max(best_val_acc, val_acc)
            print(f"epoch {epoch:3d} | train_loss {total_loss/n_batches:.4f} acc {total_acc/n_batches:.3f} "
                  f"| val_loss {val_loss:.4f} acc {val_acc:.3f} | {time.time()-t0:.0f}s")

    model.eval()
    with torch.no_grad():
        sample = val_data[:4]
        recon = model(sample).argmax(dim=-1)
    print("\nreconstruction samples (val, unseen):")
    for src, dst in zip(_decode(sample, ctx), _decode(recon, ctx)):
        cm = sum(a == b for a, b in zip(src, dst))
        print(f"  [{cm}/{len(src)}]  {src!r}  ->  {dst!r}")

    return {
        "name": cfg.name,
        "n_params": n_params,
        "best_val_acc": best_val_acc,
        "final_val_acc": val_acc,
        "final_train_acc": total_acc / n_batches,
        "time_s": time.time() - t0,
        "model": model,
        "vocab_size": vocab_size,
        "train_data": train_data,
        "val_data": val_data,
        "ctx": ctx,
    }


if __name__ == "__main__":
    run(Config(name="default"))
