"""Experiment 27: TLM autoencoder as text embedding model.

The exp5b autoencoder result was the strongest TLM win (+27pp over LSTM).
Embedding models compress text into a fixed vector for downstream tasks
(retrieval, classification, similarity). Test if TLM's compression
advantage carries over to a real embedding benchmark.

Setup:
- Train TLM and LSTM autoencoders on text (Shakespeare or NLTK movies)
- Use bottleneck state as the embedding
- Downstream task: NLTK movie_reviews sentiment classification
- Compare to bag-of-bytes mean embedding (no learning) and random

Pipeline:
  text -> tokenize -> encoder -> bottleneck state (embedding)
  embedding -> linear probe -> binary class (positive/negative)

Metric: probe accuracy on held-out reviews.
"""
import sys
import time
import io
import zipfile
import math
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F


NLTK_BASE = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora"
DATASETS_DIR = Path(__file__).parent.parent / "datasets"


def download_movie_reviews():
    target = DATASETS_DIR / "movie_reviews.zip"
    if target.exists():
        return target
    url = f"{NLTK_BASE}/movie_reviews.zip"
    print(f"downloading {url} ...")
    data = urllib.request.urlopen(url).read()
    target.write_bytes(data)
    return target


def load_movie_reviews():
    """Returns list of (text, label) tuples. Label 0=neg, 1=pos."""
    zpath = download_movie_reviews()
    examples = []
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            if info.filename.endswith(".txt"):
                if "/neg/" in info.filename:
                    label = 0
                elif "/pos/" in info.filename:
                    label = 1
                else:
                    continue
                with z.open(info) as f:
                    text = f.read().decode("utf-8", errors="replace")
                examples.append((text, label))
    return examples


def encode_bytes(text: str, max_len: int = 256) -> torch.Tensor:
    b = text.encode("utf-8", errors="replace")[:max_len]
    if len(b) < max_len:
        b = b + b" " * (max_len - len(b))
    return torch.tensor(list(b), dtype=torch.long)


# ------- Encoders -------

class LSTMEncoderForEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, state_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.cell = nn.LSTMCell(embed_dim, state_dim)
        self.state_dim = state_dim

    def forward(self, tokens):
        B, T = tokens.shape
        h = torch.zeros(B, self.state_dim, device=tokens.device)
        c = torch.zeros(B, self.state_dim, device=tokens.device)
        embeds = self.embed(tokens)
        for t in range(T):
            h, c = self.cell(embeds[:, t], (h, c))
        return h


class TreeEncoderForEmbedding(nn.Module):
    """Reuses RelationExtractor + TreeCell from lm_model. Recurrent encoder
    that compresses sequence into a fixed-size state."""
    def __init__(self, vocab_size, embed_dim, state_dim, depth, relation_dim):
        super().__init__()
        from lm_model import RelationExtractor
        from model import ForgetGatedSharedTreeCell
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.relation = RelationExtractor(embed_dim, relation_dim, depth)
        self.cell = ForgetGatedSharedTreeCell(relation_dim, state_dim, depth)
        self.state_dim = state_dim

    def forward(self, tokens):
        B, T = tokens.shape
        state = torch.zeros(B, self.state_dim, device=tokens.device)
        embeds = self.embed(tokens)
        for t in range(T):
            rel = self.relation(embeds[:, t])
            state = self.cell(rel, state)
        return state


class BagOfBytesEmbedding(nn.Module):
    """Trivial baseline: mean of byte embeddings."""
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.state_dim = embed_dim

    def forward(self, tokens):
        return self.embed(tokens).mean(dim=1)


# ------- Decoder for autoencoder pretraining -------

class AutoencoderDecoder(nn.Module):
    """Reads the encoder's final state and reconstructs the input."""
    def __init__(self, vocab_size, state_dim, max_len, embed_dim):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.cell = nn.LSTMCell(embed_dim + state_dim, state_dim)
        self.head = nn.Linear(state_dim, vocab_size)
        self.state_dim = state_dim

    def forward(self, compressed_state, seq_len):
        B = compressed_state.shape[0]
        h = torch.zeros(B, self.state_dim, device=compressed_state.device)
        c = torch.zeros(B, self.state_dim, device=compressed_state.device)
        logits = []
        for t in range(seq_len):
            pos = torch.full((B,), t, device=compressed_state.device, dtype=torch.long)
            step_in = torch.cat([self.pos_embed(pos), compressed_state], dim=-1)
            h, c = self.cell(step_in, (h, c))
            logits.append(self.head(h))
        return torch.stack(logits, dim=1)


class Autoencoder(nn.Module):
    def __init__(self, encoder, vocab_size, state_dim, max_len, embed_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = AutoencoderDecoder(vocab_size, state_dim, max_len, embed_dim)

    def forward(self, tokens):
        state = self.encoder(tokens)
        return self.decoder(state, tokens.shape[1])


# ------- Training routines -------

def train_autoencoder(model, train_tokens, n_epochs=15, batch_size=32, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n = train_tokens.shape[0]
    for ep in range(1, n_epochs + 1):
        perm = torch.randperm(n)
        total_loss = 0.0; total_acc = 0.0; nb = 0
        for i in range(0, n, batch_size):
            batch = train_tokens[perm[i:i + batch_size]]
            logits = model(batch)
            V = logits.shape[-1]
            loss = loss_fn(logits.reshape(-1, V), batch.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item()
            total_acc += (logits.argmax(-1) == batch).float().mean().item()
            nb += 1
        if ep % 3 == 0 or ep == 1:
            print(f"    ae epoch {ep:2d}  loss {total_loss/nb:.3f}  acc {total_acc/nb:.3f}")
    return model


def linear_probe(emb_train, y_train, emb_val, y_val, n_epochs=300, lr=0.1):
    D = emb_train.shape[1]
    probe = nn.Linear(D, 2)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_val = 0.0
    for ep in range(n_epochs):
        probe.train()
        opt.zero_grad()
        logits = probe(emb_train)
        loss = loss_fn(logits, y_train)
        loss.backward(); opt.step()
        if ep % 20 == 0:
            probe.eval()
            with torch.no_grad():
                val_acc = (probe(emb_val).argmax(-1) == y_val).float().mean().item()
                best_val = max(best_val, val_acc)
    probe.eval()
    with torch.no_grad():
        train_acc = (probe(emb_train).argmax(-1) == y_train).float().mean().item()
        val_acc = (probe(emb_val).argmax(-1) == y_val).float().mean().item()
    return train_acc, max(best_val, val_acc)


def main():
    torch.manual_seed(0)
    print("loading movie_reviews...")
    examples = load_movie_reviews()
    print(f"loaded {len(examples)} reviews")

    # Shuffle then split
    rng = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(examples), generator=rng).tolist()
    examples = [examples[i] for i in perm]
    n_train = int(0.8 * len(examples))
    train_pairs = examples[:n_train]
    val_pairs = examples[n_train:]
    print(f"train: {len(train_pairs)}  val: {len(val_pairs)}")

    MAX_LEN = 256
    VOCAB = 256
    train_tokens = torch.stack([encode_bytes(t, MAX_LEN) for t, _ in train_pairs])
    val_tokens = torch.stack([encode_bytes(t, MAX_LEN) for t, _ in val_pairs])
    y_train = torch.tensor([y for _, y in train_pairs])
    y_val = torch.tensor([y for _, y in val_pairs])

    EMBED_DIM = 32
    STATE_DIM = 128
    DEPTH = 3
    RELATION_DIM = 64

    encoders = {
        "tree": TreeEncoderForEmbedding(VOCAB, EMBED_DIM, STATE_DIM, DEPTH, RELATION_DIM),
        "lstm": LSTMEncoderForEmbedding(VOCAB, EMBED_DIM, STATE_DIM),
        "bag":  BagOfBytesEmbedding(VOCAB, STATE_DIM),
    }

    results = []

    # 1. Random embeddings (no training): just compute embedding from untrained model
    print("\n=== untrained embeddings (random) ===")
    for name, enc in encoders.items():
        enc.eval()
        with torch.no_grad():
            tr = enc(train_tokens)
            va = enc(val_tokens)
        train_acc, val_acc = linear_probe(tr, y_train, va, y_val)
        params = sum(p.numel() for p in enc.parameters())
        print(f"  {name:<6} (untrained, params={params:,}): probe train={train_acc:.3f} val={val_acc:.3f}")
        results.append((f"{name}_untrained", params, train_acc, val_acc))

    # 2. Autoencoder-pretrained encoders
    print("\n=== autoencoder-pretrained encoders ===")
    for name in ("tree", "lstm"):
        print(f"  pretraining {name} autoencoder...")
        # rebuild encoder fresh for ae training
        if name == "tree":
            enc = TreeEncoderForEmbedding(VOCAB, EMBED_DIM, STATE_DIM, DEPTH, RELATION_DIM)
        else:
            enc = LSTMEncoderForEmbedding(VOCAB, EMBED_DIM, STATE_DIM)
        ae = Autoencoder(enc, VOCAB, STATE_DIM, MAX_LEN, EMBED_DIM)
        train_autoencoder(ae, train_tokens, n_epochs=10, batch_size=16)
        ae.eval()
        with torch.no_grad():
            tr = ae.encoder(train_tokens)
            va = ae.encoder(val_tokens)
        train_acc, val_acc = linear_probe(tr, y_train, va, y_val)
        params = sum(p.numel() for p in ae.encoder.parameters())
        print(f"  {name:<6} (AE-pretrained): probe train={train_acc:.3f} val={val_acc:.3f}")
        results.append((f"{name}_ae", params, train_acc, val_acc))

    print("\n=== SUMMARY (NLTK movie_reviews sentiment, MAX_LEN=256, state=128) ===")
    print(f"{'embedding':<18} {'params':>10} {'probe_train':>12} {'probe_val':>11}")
    for name, params, tr_acc, va_acc in results:
        print(f"{name:<18} {params:>10,} {tr_acc:>12.3f} {va_acc:>11.3f}")

    print(f"\n--- baseline: random class is 0.5 (binary balanced) ---")


if __name__ == "__main__":
    main()
