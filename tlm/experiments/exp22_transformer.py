"""Experiment 22: basic transformer baseline, matched params to exp21 TLM.

Compare TLM champion (from exp21) against a nanoGPT-style transformer
at ~700k params, same Shakespeare regime (seq_len=128, n_train=4000,
25 epochs). If transformer crushes TLM even at matched params, we know
the Shakespeare task genuinely needs attention-style content routing.
If TLM ties or beats, we have a real competitive story.
"""
import sys
import math
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from lm_train import LMConfig, run_lm


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, T, Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        out = att @ v  # (B, H, T, Dh)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    """Standard decoder-only transformer LM. d_model chosen so total params
    are ~matched to exp21 TLM (~700k). Ignores the LM-config 'state_dim',
    'depth', 'relation_dim', 'context_dim' - they don't apply here."""

    def __init__(self, vocab_size, embed_dim=None, state_dim=None, depth=None,
                 relation_dim=None, context_dim=None, max_len=128,
                 d_model=128, n_heads=4, n_layers=4, ff_mult=3):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_mult) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, tokens):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        x = self.tok_embed(tokens) + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, prompt, n_new_tokens, temperature=1.0):
        self.eval()
        out = prompt.clone()
        for _ in range(n_new_tokens):
            ctx = out[:, -self.max_len:]
            logits = self(ctx)[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            out = torch.cat([out, nxt], dim=1)
        return out


def make_cfg(name):
    return LMConfig(seq_len=128, state_dim=256, depth=3, relation_dim=64,
                    context_dim=64, n_train=4000, n_val=500, batch_size=32,
                    epochs=25, lr=3e-3, grad_clip=5.0, log_every=5, name=name)


if __name__ == "__main__":
    print("=== Transformer baseline matched-params ===")
    r = run_lm(make_cfg("transformer_matched"), model_ctor=TransformerLM)
    print(f"\n{r['name']:<30} params={r['n_params']:,}  best_vbpc={r['best_val_bpc']:.3f}  "
          f"best_val_acc={r['best_val_acc']:.3f}  time={r['time_s']:.0f}s")
