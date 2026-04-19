import torch


def make_dataset(n_samples: int, seq_len: int, vocab_size: int, seed: int = 42) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size, (n_samples, seq_len), generator=g)


def decode(tokens: torch.Tensor) -> list[str]:
    return [" ".join(f"{int(t):>5d}" for t in row) for row in tokens]
