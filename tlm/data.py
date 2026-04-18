import torch

VOCAB = [
    "the", "a", "cat", "dog", "bird", "fish",
    "runs", "jumps", "sleeps", "eats", "sees", "likes",
    "fast", "slow", "big", "small", "red", "blue",
    "here", "there",
]
VOCAB_SIZE = len(VOCAB)
TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}
ID_TO_TOKEN = {i: tok for i, tok in enumerate(VOCAB)}


def make_dataset(n_samples: int, seq_len: int, seed: int = 42) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB_SIZE, (n_samples, seq_len), generator=g)


def decode(tokens: torch.Tensor) -> list[str]:
    return [" ".join(ID_TO_TOKEN[int(t)] for t in row) for row in tokens]
