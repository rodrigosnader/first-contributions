"""BPE-tokenized Shakespeare. Trains a small BPE once and caches it."""
from pathlib import Path
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from data import SHAKESPEARE_PATH

CACHE_DIR = Path(__file__).parent / "datasets"
BPE_PATH = CACHE_DIR / "shakespeare_bpe.json"


def get_or_train_bpe(vocab_size: int = 1024) -> Tokenizer:
    cache = CACHE_DIR / f"shakespeare_bpe_{vocab_size}.json"
    if cache.exists():
        return Tokenizer.from_file(str(cache))
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>"])
    tok.train([str(SHAKESPEARE_PATH)], trainer)
    tok.save(str(cache))
    return tok


def load_shakespeare_bpe(vocab_size: int = 1024) -> tuple[torch.Tensor, Tokenizer]:
    tok = get_or_train_bpe(vocab_size)
    text = SHAKESPEARE_PATH.read_text()
    ids = torch.tensor(tok.encode(text).ids, dtype=torch.long)
    return ids, tok


def sample_windows(source: torch.Tensor, n_samples: int, seq_len: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    max_start = len(source) - seq_len
    starts = torch.randint(0, max_start + 1, (n_samples,), generator=g)
    return torch.stack([source[s:s + seq_len] for s in starts])


def make_bpe_splits(n_train: int, n_val: int, seq_len: int, vocab_size: int = 1024,
                    seed: int = 42) -> tuple[torch.Tensor, torch.Tensor, Tokenizer]:
    ids, tok = load_shakespeare_bpe(vocab_size)
    split = int(len(ids) * 0.9)
    train_src = ids[:split]
    val_src = ids[split:]
    train = sample_windows(train_src, n_train, seq_len, seed=seed)
    val = sample_windows(val_src, n_val, seq_len, seed=seed + 1)
    return train, val, tok


def decode_bpe(tokens: torch.Tensor, tok: Tokenizer) -> list[str]:
    return [tok.decode(row.tolist()) for row in tokens]
