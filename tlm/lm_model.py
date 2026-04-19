"""TLM as a language model: emit next-token logits at every step. Same
TreeCell + RelationExtractor as the autoencoder, but no decoder - the
state itself is the prediction substrate."""
import torch
import torch.nn as nn

from model import SoftTree, TreeCell


class RelationExtractor(nn.Module):
    def __init__(self, embed_dim: int, relation_dim: int, depth: int):
        super().__init__()
        self.tree = SoftTree(embed_dim, relation_dim, depth)
        self.norm = nn.LayerNorm(relation_dim)

    def forward(self, x):
        return self.norm(self.tree(x))


class TreeLM(nn.Module):
    """Causal next-token LM. At each step t the state has integrated tokens
    0..t and emits logits for token t+1."""

    def __init__(self, vocab_size: int, embed_dim: int = 32, state_dim: int = 128,
                 depth: int = 4, relation_dim: int = 64, max_len: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.relation = RelationExtractor(embed_dim, relation_dim, depth)
        self.cell = TreeCell(relation_dim, state_dim, depth)
        self.output_head = nn.Linear(state_dim, vocab_size)
        self.state_dim = state_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T). returns logits (B, T, V) where logits[:, t] predicts
        the token AFTER tokens[:, t] (so target is tokens[:, t+1])."""
        batch, seq_len = tokens.shape
        state = torch.zeros(batch, self.state_dim, device=tokens.device)
        embeds = self.embed(tokens)
        all_logits = []
        for t in range(seq_len):
            rel = self.relation(embeds[:, t])
            state = self.cell(rel, state)
            all_logits.append(self.output_head(state))
        return torch.stack(all_logits, dim=1)

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, n_new_tokens: int,
                 temperature: float = 1.0) -> torch.Tensor:
        """Continue prompt with n_new_tokens sampled from the model."""
        self.eval()
        batch = prompt.shape[0]
        state = torch.zeros(batch, self.state_dim, device=prompt.device)
        # consume prompt
        for t in range(prompt.shape[1]):
            rel = self.relation(self.embed(prompt[:, t]))
            state = self.cell(rel, state)
        out = [prompt]
        cur = prompt[:, -1:]
        for _ in range(n_new_tokens):
            rel = self.relation(self.embed(cur[:, 0]))
            state = self.cell(rel, state)
            logits = self.output_head(state) / temperature
            probs = torch.softmax(logits, dim=-1)
            cur = torch.multinomial(probs, 1)
            out.append(cur)
        return torch.cat(out, dim=1)


class LSTMLM(nn.Module):
    """Standard LSTM language model baseline."""

    def __init__(self, vocab_size: int, embed_dim: int = 32, state_dim: int = 128,
                 depth: int = 0, relation_dim: int = 0, max_len: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.cell = nn.LSTMCell(embed_dim, state_dim)
        self.output_head = nn.Linear(state_dim, vocab_size)
        self.state_dim = state_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq_len = tokens.shape
        h = torch.zeros(batch, self.state_dim, device=tokens.device)
        c = torch.zeros(batch, self.state_dim, device=tokens.device)
        embeds = self.embed(tokens)
        all_logits = []
        for t in range(seq_len):
            h, c = self.cell(embeds[:, t], (h, c))
            all_logits.append(self.output_head(h))
        return torch.stack(all_logits, dim=1)

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, n_new_tokens: int,
                 temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        batch = prompt.shape[0]
        h = torch.zeros(batch, self.state_dim, device=prompt.device)
        c = torch.zeros(batch, self.state_dim, device=prompt.device)
        for t in range(prompt.shape[1]):
            h, c = self.cell(self.embed(prompt[:, t]), (h, c))
        out = [prompt]
        cur = prompt[:, -1:]
        for _ in range(n_new_tokens):
            h, c = self.cell(self.embed(cur[:, 0]), (h, c))
            logits = self.output_head(h) / temperature
            probs = torch.softmax(logits, dim=-1)
            cur = torch.multinomial(probs, 1)
            out.append(cur)
        return torch.cat(out, dim=1)
