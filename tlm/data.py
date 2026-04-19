from pathlib import Path
import torch

SHAKESPEARE_PATH = Path(__file__).parent / "datasets" / "tinyshakespeare.txt"


def load_shakespeare() -> tuple[torch.Tensor, dict, dict]:
    """Returns (ids, stoi, itos). Char-level tokenization."""
    text = SHAKESPEARE_PATH.read_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return ids, stoi, itos


def sample_windows(source: torch.Tensor, n_samples: int, seq_len: int,
                   seed: int) -> torch.Tensor:
    """Sample n_samples random contiguous windows of length seq_len from source."""
    g = torch.Generator().manual_seed(seed)
    max_start = len(source) - seq_len
    starts = torch.randint(0, max_start + 1, (n_samples,), generator=g)
    return torch.stack([source[s:s + seq_len] for s in starts])


def make_shakespeare_splits(n_train: int, n_val: int, seq_len: int,
                            seed: int = 42) -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
    """Train windows from first 90% of text, val windows from last 10%.
    Disjoint char ranges -> val windows cannot overlap train windows."""
    ids, stoi, itos = load_shakespeare()
    split = int(len(ids) * 0.9)
    train_src = ids[:split]
    val_src = ids[split:]
    train = sample_windows(train_src, n_train, seq_len, seed=seed)
    val = sample_windows(val_src, n_val, seq_len, seed=seed + 1)
    return train, val, stoi, itos


def decode(tokens: torch.Tensor, itos: dict) -> list[str]:
    return ["".join(itos[int(t)] for t in row) for row in tokens]
