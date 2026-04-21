"""WikiText-2 byte-level loader. Treat UTF-8 bytes as tokens - vocab ~178,
10x bigger corpus than TinyShakespeare (10MB vs 1MB)."""
from pathlib import Path
import torch

WIKI_TRAIN = Path(__file__).parent / "datasets" / "wikitext2_train.txt"
WIKI_VALID = Path(__file__).parent / "datasets" / "wikitext2_valid.txt"


def load_wikitext_bytes():
    train_bytes = WIKI_TRAIN.read_bytes()
    valid_bytes = WIKI_VALID.read_bytes()
    seen = set(train_bytes) | set(valid_bytes)
    sorted_ids = sorted(seen)
    stoi = {b: i for i, b in enumerate(sorted_ids)}
    itos = {i: b for i, b in enumerate(sorted_ids)}
    train_ids = torch.tensor([stoi[b] for b in train_bytes], dtype=torch.long)
    valid_ids = torch.tensor([stoi[b] for b in valid_bytes], dtype=torch.long)
    return train_ids, valid_ids, stoi, itos


def sample_windows(source: torch.Tensor, n_samples: int, seq_len: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    max_start = len(source) - seq_len
    starts = torch.randint(0, max_start + 1, (n_samples,), generator=g)
    return torch.stack([source[s:s + seq_len] for s in starts])


def make_wikitext_splits(n_train: int, n_val: int, seq_len: int, seed: int = 42):
    train_src, val_src, stoi, itos = load_wikitext_bytes()
    train = sample_windows(train_src, n_train, seq_len, seed=seed)
    val = sample_windows(val_src, n_val, seq_len, seed=seed + 1)
    return train, val, stoi, itos
