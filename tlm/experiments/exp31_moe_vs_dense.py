"""Experiment 31: MoE Transformer vs Dense Transformer at matched params.

Validates the thesis 'sparse compute has SOTA potential'. The canonical
MoE result: at matched total parameters, MoE FFN with top-K routing beats
dense FFN because:
  - More total capacity (K_experts * smaller_FFN_size > one FFN of same size)
  - But only top-K experts active per token = lower active compute
  - Routing learns to specialize experts to input regions

If this replicates on Shakespeare at small scale, the path 'sparse compute
-> SOTA' is empirically valid.

Configs (~700k params each):
- Dense: 4 layers, d_model=128, FFN mult=3
- MoE:   4 layers, d_model=128, 4 experts per layer, top-2, FFN mult=0.75
        (each expert is 1/4 size of dense FFN; total FFN params = same)
"""
import sys
import math
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import make_shakespeare_splits
from lm_train import LMConfig, shifted_loss


class MoEFFN(nn.Module):
    """Top-K routing over n_experts, each a 2-layer MLP. Per token, only
    top_k experts run; their outputs are weighted-summed by softmax(top_k logits)."""

    def __init__(self, d_model: int, n_experts: int = 4, top_k: int = 2,
                 expert_mult: float = 0.75):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        hidden = max(4, int(d_model * expert_mult))
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, d_model),
            )
            for _ in range(n_experts)
        ])

    def forward(self, x):
        B, T, D = x.shape
        flat = x.reshape(-1, D)  # (BT, D)
        logits = self.router(flat)
        top_vals, top_idx = logits.topk(self.top_k, dim=-1)  # (BT, k)
        weights = torch.softmax(top_vals, dim=-1)

        out = torch.zeros_like(flat)
        for e in range(self.n_experts):
            mask = (top_idx == e)  # (BT, k)
            if not mask.any():
                continue
            wsum = (mask.float() * weights).sum(dim=-1)  # (BT,)
            sel = wsum > 0
            if not sel.any():
                continue
            expert_out = self.experts[e](flat[sel])
            out[sel] += wsum[sel].unsqueeze(-1) * expert_out
        return out.reshape(B, T, D)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = ffn

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, max_len=128, d_model=128, n_heads=4,
                 n_layers=4, use_moe=False, ff_mult=3,
                 n_experts=4, top_k=2, expert_mult=0.75):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        blocks = []
        for _ in range(n_layers):
            if use_moe:
                ffn = MoEFFN(d_model, n_experts=n_experts, top_k=top_k,
                             expert_mult=expert_mult)
            else:
                ffn = nn.Sequential(
                    nn.Linear(d_model, d_model * ff_mult),
                    nn.GELU(),
                    nn.Linear(d_model * ff_mult, d_model),
                )
            blocks.append(TransformerBlock(d_model, n_heads, ffn))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, tokens):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        x = self.tok_embed(tokens) + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def train_lm(model, train_data, val_data, n_epochs=20, batch_size=32, lr=3e-3,
             grad_clip=1.0, label="model"):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total_loss += loss.item(); nb += 1
        if ep % 5 == 0 or ep == 1 or ep == n_epochs:
            model.eval()
            with torch.no_grad():
                vl, va = shifted_loss(model(val_data), val_data, loss_fn)
            vbpc = vl.item() / math.log(2)
            best_bpc = min(best_bpc, vbpc); best_acc = max(best_acc, va)
            print(f"  [{label}] ep {ep:3d}  train_bpc {total_loss/nb/math.log(2):.3f}  "
                  f"val_bpc {vbpc:.3f}  val_acc {va:.3f}  ({time.time()-t0:.0f}s)")
    return best_bpc, best_acc, time.time() - t0


def main():
    torch.manual_seed(0)
    SEQ_LEN = 128
    train_data, val_data, stoi, _ = make_shakespeare_splits(4000, 500, SEQ_LEN)
    vocab = len(stoi)

    print("\n=== Dense Transformer ===")
    dense = TransformerLM(vocab, max_len=SEQ_LEN, d_model=128, n_heads=4,
                          n_layers=4, use_moe=False, ff_mult=3)
    dense_params = sum(p.numel() for p in dense.parameters())
    print(f"params: {dense_params:,}")
    dense_bpc, dense_acc, dense_time = train_lm(dense, train_data, val_data,
                                                n_epochs=20, label="dense")

    print("\n=== MoE Transformer (matched params, top-2 of 4 experts) ===")
    moe = TransformerLM(vocab, max_len=SEQ_LEN, d_model=128, n_heads=4,
                        n_layers=4, use_moe=True,
                        n_experts=4, top_k=2, expert_mult=0.75)
    moe_params = sum(p.numel() for p in moe.parameters())
    print(f"params: {moe_params:,} (~50% FFN compute per token vs dense)")
    moe_bpc, moe_acc, moe_time = train_lm(moe, train_data, val_data,
                                          n_epochs=20, label="moe")

    print("\n=== MoE Transformer (more experts, top-2 of 8) ===")
    moe2 = TransformerLM(vocab, max_len=SEQ_LEN, d_model=128, n_heads=4,
                         n_layers=4, use_moe=True,
                         n_experts=8, top_k=2, expert_mult=0.375)
    moe2_params = sum(p.numel() for p in moe2.parameters())
    print(f"params: {moe2_params:,} (top-2 of 8 = 25% FFN compute per token)")
    moe2_bpc, moe2_acc, moe2_time = train_lm(moe2, train_data, val_data,
                                             n_epochs=20, label="moe8")

    print("\n=== SUMMARY (Shakespeare seq=128, 20 epochs) ===")
    print(f"{'model':<20} {'params':>10} {'val_bpc':>9} {'val_acc':>9} {'time':>7}")
    print(f"{'dense':<20} {dense_params:>10,} {dense_bpc:>9.3f} {dense_acc:>9.3f} {dense_time:>6.0f}s")
    print(f"{'MoE 4exp top-2':<20} {moe_params:>10,} {moe_bpc:>9.3f} {moe_acc:>9.3f} {moe_time:>6.0f}s")
    print(f"{'MoE 8exp top-2':<20} {moe2_params:>10,} {moe2_bpc:>9.3f} {moe2_acc:>9.3f} {moe2_time:>6.0f}s")

    print("\n=== INTERPRETATION ===")
    d_moe = moe_bpc - dense_bpc
    d_moe2 = moe2_bpc - dense_bpc
    print(f"  MoE 4exp vs dense: {d_moe:+.3f} BPC  ({'sparse wins' if d_moe < 0 else 'dense wins'})")
    print(f"  MoE 8exp vs dense: {d_moe2:+.3f} BPC  ({'sparse wins' if d_moe2 < 0 else 'dense wins'})")
    print("  (matched params; MoE has lower active compute per token)")


if __name__ == "__main__":
    main()
