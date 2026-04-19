"""Experiment 2: replace char-level tokenizer with a small BPE. Tests whether
the tree autoencoder handles subword units (larger effective vocabulary,
semantically richer tokens, shorter sequences for same text coverage)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train import Config, run
from bpe_data import make_bpe_splits, decode_bpe


BPE_VOCAB = 1024


def data_fn(n_train, n_val, seq_len):
    train, val, tok = make_bpe_splits(n_train, n_val, seq_len, vocab_size=BPE_VOCAB)
    return train, val, tok.get_vocab_size(), tok


cfg = Config(
    seq_len=16,
    embed_dim=32,
    state_dim=128,
    depth=4,
    n_train=4000,
    n_val=500,
    epochs=30,
    name=f"bpe{BPE_VOCAB}_seq16",
)
r = run(cfg, data_fn=data_fn, decode_fn=decode_bpe)
print(f"\n=== SUMMARY ===")
print(f"BPE vocab: {BPE_VOCAB}, best_val_acc: {r['best_val_acc']:.3f}, time: {r['time_s']:.0f}s")
