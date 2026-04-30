"""Experiment 30: Hierarchical SSM + Attention vs nano Transformer.

User's architecture: avoid quadratic attention on long sequences by:
  1. Local sliding-window self-attention over recent W tokens (cheap, O(T*W))
  2. SSM (here: LSTM) compresses chunks of N tokens into summary vectors
  3. Cross-attention from each token to all PAST chunk summaries (so the
     model can see compressed long-range context without O(T^2) cost)

Compare against a vanilla nano transformer with full causal self-attention,
matched param count, same data regime.

Hypotheses:
- H1: at seq_len long enough for full attention to matter, hierarchical
  matches or beats vanilla on val_bpc per-param
- H2: hierarchical is faster in inference (lower attn cost per step)
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


class CompressiveAttention(nn.Module):
    """Each token attends over (sliding causal local window of W tokens) +
    (all past chunk summaries). Chunks are causal: chunk j is visible to
    token t iff (j+1)*chunk_size <= t (entire chunk is in token t's past)."""

    def __init__(self, d_model: int, n_heads: int, local_window: int, chunk_size: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.local_window = local_window
        self.chunk_size = chunk_size
        self.q = nn.Linear(d_model, d_model)
        self.k_local = nn.Linear(d_model, d_model)
        self.v_local = nn.Linear(d_model, d_model)
        self.k_chunk = nn.Linear(d_model, d_model)
        self.v_chunk = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, chunk_summaries):
        B, T, D = x.shape
        Tc = chunk_summaries.shape[1]
        H, Dh = self.n_heads, self.d_head

        def split(t, n):
            return t.reshape(B, n, H, Dh).transpose(1, 2)  # (B, H, n, Dh)

        q = split(self.q(x), T)
        kl = split(self.k_local(x), T)
        vl = split(self.v_local(x), T)
        kc = split(self.k_chunk(chunk_summaries), Tc)
        vc = split(self.v_chunk(chunk_summaries), Tc)

        k = torch.cat([kc, kl], dim=2)  # (B, H, Tc+T, Dh)
        v = torch.cat([vc, vl], dim=2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)  # (B, H, T, Tc+T)

        token_pos = torch.arange(T, device=x.device)
        chunk_pos = torch.arange(Tc, device=x.device)
        # chunk j visible to token t iff (j+1)*chunk_size <= t
        chunk_mask = (chunk_pos[None, :] + 1) * self.chunk_size <= token_pos[:, None]
        # local i visible to token t iff i <= t and t - i < window
        local_mask = (token_pos[None, :] <= token_pos[:, None]) & \
                     ((token_pos[:, None] - token_pos[None, :]) < self.local_window)
        full_mask = torch.cat([chunk_mask, local_mask], dim=1)  # (T, Tc+T)
        att = att.masked_fill(~full_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        att = torch.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class HierarchicalBlock(nn.Module):
    def __init__(self, d_model, n_heads, local_window, chunk_size, ff_mult=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CompressiveAttention(d_model, n_heads, local_window, chunk_size)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )

    def forward(self, x, chunk_summaries):
        x = x + self.attn(self.norm1(x), chunk_summaries)
        x = x + self.ff(self.norm2(x))
        return x


class HierarchicalLM(nn.Module):
    """Token embed -> LSTM compressor -> sample chunk summaries every N tokens
    -> several HierarchicalBlocks with compressive attention over (local window,
    past chunks) -> output head."""

    def __init__(self, vocab_size, embed_dim=None, state_dim=None, depth=None,
                 relation_dim=None, context_dim=None, max_len=256,
                 d_model=128, n_heads=4, n_layers=3, ff_mult=3,
                 chunk_size=16, local_window=32):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.compressor = nn.LSTM(d_model, d_model, batch_first=True)
        self.blocks = nn.ModuleList([
            HierarchicalBlock(d_model, n_heads, local_window, chunk_size, ff_mult)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.chunk_size = chunk_size
        self.local_window = local_window
        self.max_len = max_len

    def forward(self, tokens):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        x = self.tok_embed(tokens) + self.pos_embed(pos)

        # Compress: run LSTM over the sequence, then take state every chunk_size positions
        lstm_out, _ = self.compressor(x)
        # Sample every chunk_size positions (last token of each chunk)
        # Indices: chunk_size-1, 2*chunk_size-1, ...
        idx = torch.arange(self.chunk_size - 1, T, self.chunk_size, device=tokens.device)
        if len(idx) == 0:
            chunk_summaries = lstm_out[:, :0]
        else:
            chunk_summaries = lstm_out[:, idx]  # (B, Tc, D)

        for block in self.blocks:
            x = block(x, chunk_summaries)
        return self.head(self.norm(x))

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


# Reuse the vanilla TransformerLM from exp22 for the baseline. exp22 was
# later modified to return (logits, kv_caches) for KV-cache experiments;
# wrap it so it returns plain logits like our run_lm expects.
import importlib.util, os
spec = importlib.util.spec_from_file_location(
    "exp22_mod",
    os.path.join(os.path.dirname(__file__), "exp22_transformer.py"),
)
exp22 = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp22)
_BaseTransformerLM = exp22.TransformerLM


class TransformerLM(_BaseTransformerLM):
    def forward(self, tokens, **kw):
        out = super().forward(tokens, **kw)
        return out[0] if isinstance(out, tuple) else out


def train_lm(model, train_data, val_data, n_epochs=25, batch_size=32, lr=3e-3,
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
    SEQ_LEN = 256
    train_data, val_data, stoi, _ = make_shakespeare_splits(4000, 500, SEQ_LEN)
    vocab = len(stoi)

    print("\n=== nano Transformer (vanilla causal self-attention) ===")
    txf = TransformerLM(vocab, max_len=SEQ_LEN, d_model=128,
                        n_heads=4, n_layers=4, ff_mult=3)
    txf_params = sum(p.numel() for p in txf.parameters())
    print(f"params: {txf_params:,}")
    txf_bpc, txf_acc, txf_time = train_lm(txf, train_data, val_data,
                                          n_epochs=25, label="transformer")

    print("\n=== Hierarchical LM (LSTM compress + local attn + chunk attn) ===")
    hier = HierarchicalLM(vocab, max_len=SEQ_LEN, d_model=128, n_heads=4,
                          n_layers=3, ff_mult=3,
                          chunk_size=16, local_window=32)
    hier_params = sum(p.numel() for p in hier.parameters())
    print(f"params: {hier_params:,}")
    hier_bpc, hier_acc, hier_time = train_lm(hier, train_data, val_data,
                                             n_epochs=25, label="hierarchical")

    print("\n=== SUMMARY (Shakespeare seq=256, ~700k params, 25 epochs) ===")
    print(f"{'model':<22} {'params':>10} {'val_bpc':>9} {'val_acc':>9} {'time':>7}")
    print(f"{'nano transformer':<22} {txf_params:>10,} {txf_bpc:>9.3f} {txf_acc:>9.3f} {txf_time:>6.0f}s")
    print(f"{'hierarchical':<22} {hier_params:>10,} {hier_bpc:>9.3f} {hier_acc:>9.3f} {hier_time:>6.0f}s")

    delta = hier_bpc - txf_bpc
    print(f"\nhierarchical - transformer val_bpc delta: {delta:+.3f}")
    print(f"  ({'hierarchical wins' if delta < 0 else 'transformer wins'})")

    # Inference timing: tokens/sec at batch=1, generating 256 new from 8-token prompt
    torch.set_num_threads(1)
    prompt = torch.randint(0, vocab, (1, 8))
    n_gen = 256
    print(f"\n=== inference speed (batch=1, single thread, gen {n_gen} tokens) ===")
    for label, m in [("nano transformer", txf), ("hierarchical", hier)]:
        m.eval()
        with torch.no_grad():
            for _ in range(2):
                m.generate(prompt, n_gen, temperature=1.0)
            best = float("inf")
            for _ in range(2):
                t0 = time.perf_counter()
                m.generate(prompt, n_gen, temperature=1.0)
                best = min(best, time.perf_counter() - t0)
        print(f"  {label}: {best:.2f}s for {n_gen} tokens = {n_gen/best:.1f} tok/s")


if __name__ == "__main__":
    main()
