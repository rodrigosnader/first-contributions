"""Experiment 24: transformer with proper regularization vs vanilla.

exp22 vanilla transformer lost to LSTM. Hypothesis: it was undertrained
/ poorly regularized. Add dropout, LR warmup + cosine decay, and label
smoothing. See if a properly-tuned transformer wins at the same
~700k params / seq=128 / n_train=4000 regime.
"""
import sys
import math
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from lm_train import LMConfig, shifted_loss
from data import make_shakespeare_splits


class CausalSelfAttentionDropout(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)
        out = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj_drop(self.proj(out))


class TransformerBlockReg(nn.Module):
    def __init__(self, d_model, n_heads, ff_mult=3, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttentionDropout(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class TransformerLMReg(nn.Module):
    def __init__(self, vocab_size, embed_dim=None, state_dim=None, depth=None,
                 relation_dim=None, context_dim=None, max_len=128,
                 d_model=128, n_heads=4, n_layers=4, ff_mult=3, dropout=0.1):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.embed_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlockReg(d_model, n_heads, ff_mult, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, tokens):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        x = self.embed_drop(self.tok_embed(tokens) + self.pos_embed(pos))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def lr_at(step, warmup, total, base_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    # cosine decay
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def train_regularized(cfg: LMConfig, model_ctor, label_smoothing=0.1, dropout=0.1,
                      data_fn=None):
    torch.manual_seed(cfg.seed)
    if data_fn is None:
        data_fn = make_shakespeare_splits
    train_data, val_data, stoi, itos = data_fn(cfg.n_train, cfg.n_val, cfg.seq_len)
    vocab_size = len(stoi)

    model = model_ctor(vocab_size, max_len=cfg.seq_len, dropout=dropout)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"=== {cfg.name} ===")
    print(f"params: {n_params:,}  dropout={dropout}  label_smoothing={label_smoothing}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    steps_per_epoch = (cfg.n_train + cfg.batch_size - 1) // cfg.batch_size
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = min(500, total_steps // 10)
    step = 0
    t0 = time.time()
    best_val_bpc = float("inf")
    best_val_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(cfg.n_train)
        total_loss = 0.0; total_acc = 0.0; n_batches = 0
        for i in range(0, cfg.n_train, cfg.batch_size):
            cur_lr = lr_at(step, warmup_steps, total_steps, cfg.lr)
            for g in opt.param_groups:
                g["lr"] = cur_lr
            batch = train_data[perm[i:i + cfg.batch_size]]
            logits = model(batch)
            loss, acc = shifted_loss(logits, batch, loss_fn)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total_loss += loss.item(); total_acc += acc; n_batches += 1
            step += 1

        if epoch % cfg.log_every == 0 or epoch == 1 or epoch == cfg.epochs:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_data)
                val_loss, val_acc = shifted_loss(val_logits, val_data, loss_fn)
            val_bpc = val_loss.item() / math.log(2)
            train_bpc = total_loss / n_batches / math.log(2)
            best_val_bpc = min(best_val_bpc, val_bpc)
            best_val_acc = max(best_val_acc, val_acc)
            print(f"epoch {epoch:3d}  lr {cur_lr:.4f}  train_bpc {train_bpc:.3f}  "
                  f"val_bpc {val_bpc:.3f}  val_acc {val_acc:.3f}  {time.time()-t0:.0f}s")

    return {"name": cfg.name, "n_params": n_params,
            "best_val_bpc": best_val_bpc, "best_val_acc": best_val_acc,
            "time_s": time.time() - t0}


if __name__ == "__main__":
    cfg = LMConfig(seq_len=128, n_train=4000, n_val=500, batch_size=32,
                   epochs=25, lr=3e-3, grad_clip=1.0, log_every=5,
                   name="transformer_reg")
    r = train_regularized(cfg, TransformerLMReg, label_smoothing=0.1, dropout=0.1)
    print(f"\n=== SUMMARY ===")
    print(f"{r['name']:<25} params={r['n_params']:,}  "
          f"best_vbpc={r['best_val_bpc']:.3f}  best_val_acc={r['best_val_acc']:.3f}  "
          f"time={r['time_s']:.0f}s")
    print(f"\n--- compare to exp22 vanilla transformer: vbpc=2.583 acc=0.483 ---")
    print(f"--- compare to exp21 LSTM ~700k:            vbpc=2.383 acc=0.522 ---")
    print(f"--- compare to exp21 TLM ~700k:             vbpc=2.426 acc=0.513 ---")
