import torch
import torch.nn as nn


class SoftTree(nn.Module):
    """Soft binary decision tree. Differentiable: mixes all leaf outputs weighted
    by the probability of each root-to-leaf path under the routing decisions."""

    def __init__(self, input_dim: int, output_dim: int, depth: int):
        super().__init__()
        self.depth = depth
        self.output_dim = output_dim
        self.n_internal = 2 ** depth - 1
        self.n_leaves = 2 ** depth
        self.router = nn.Linear(input_dim, self.n_internal)
        self.leaves = nn.Linear(input_dim, self.n_leaves * output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        routing_probs = torch.sigmoid(self.router(x))  # (B, n_internal), prob of going LEFT

        node_probs = torch.ones(batch, 1, device=x.device, dtype=x.dtype)
        idx = 0
        for level in range(self.depth):
            n_at_level = 2 ** level
            level_probs = routing_probs[:, idx:idx + n_at_level]
            left = node_probs * level_probs
            right = node_probs * (1.0 - level_probs)
            node_probs = torch.stack([left, right], dim=2).reshape(batch, 2 * n_at_level)
            idx += n_at_level

        leaf_outs = self.leaves(x).reshape(batch, self.n_leaves, self.output_dim)
        return (node_probs.unsqueeze(-1) * leaf_outs).sum(dim=1)

    def routing_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Average binary entropy across internal nodes. ~0 = collapsed, ~ln(2) = uncommitted."""
        with torch.no_grad():
            p = torch.sigmoid(self.router(x)).clamp(1e-7, 1 - 1e-7)
            ent = -(p * p.log() + (1 - p) * (1 - p).log())
            return ent.mean()


class TreeCell(nn.Module):
    """Recurrent cell: state_t = LayerNorm(Tree(concat(input_t, state_{t-1}))).
    LayerNorm keeps state magnitude stable across long recurrent unrolls."""

    def __init__(self, input_dim: int, state_dim: int, depth: int):
        super().__init__()
        self.state_dim = state_dim
        self.tree = SoftTree(input_dim + state_dim, state_dim, depth)
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.norm(self.tree(torch.cat([x, state], dim=-1)))


class ResidualTreeCell(nn.Module):
    """state_t = LayerNorm(state_{t-1} + Tree(x, state)). Tree proposes a
    delta; state is preserved by default. Closest analog to a residual
    block in transformers - lets the cell learn 'do nothing' by outputting
    a small delta."""

    def __init__(self, input_dim: int, state_dim: int, depth: int):
        super().__init__()
        self.state_dim = state_dim
        self.tree = SoftTree(input_dim + state_dim, state_dim, depth)
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        delta = self.tree(torch.cat([x, state], dim=-1))
        return self.norm(state + delta)


class GatedTreeCell(nn.Module):
    """GRU-style update: state_t = g * state_{t-1} + (1-g) * tree_output.
    A learned sigmoid gate decides, per dimension, how much of the old
    state to keep vs replace with the tree's proposal."""

    def __init__(self, input_dim: int, state_dim: int, depth: int):
        super().__init__()
        self.state_dim = state_dim
        self.tree = SoftTree(input_dim + state_dim, state_dim, depth)
        self.gate = nn.Linear(input_dim + state_dim, state_dim)
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([x, state], dim=-1)
        g = torch.sigmoid(self.gate(concat))
        proposed = self.tree(concat)
        return self.norm(g * state + (1 - g) * proposed)


class IdentityLeafTreeCell(nn.Module):
    """Soft tree where leaf 0 is hardcoded to return state_{t-1} verbatim.
    Other leaves are learned transforms. Routing can literally select
    'preserve state' as one of the outcomes."""

    def __init__(self, input_dim: int, state_dim: int, depth: int):
        super().__init__()
        self.state_dim = state_dim
        self.depth = depth
        self.n_internal = 2 ** depth - 1
        self.n_leaves = 2 ** depth
        full_in = input_dim + state_dim
        self.router = nn.Linear(full_in, self.n_internal)
        # learned leaves: one fewer than total; leaf 0 is identity-on-state
        self.leaves = nn.Linear(full_in, (self.n_leaves - 1) * state_dim)
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        concat = torch.cat([x, state], dim=-1)

        routing_probs = torch.sigmoid(self.router(concat))
        node_probs = torch.ones(batch, 1, device=concat.device, dtype=concat.dtype)
        idx = 0
        for level in range(self.depth):
            n_at_level = 2 ** level
            level_probs = routing_probs[:, idx:idx + n_at_level]
            left = node_probs * level_probs
            right = node_probs * (1.0 - level_probs)
            node_probs = torch.stack([left, right], dim=2).reshape(batch, 2 * n_at_level)
            idx += n_at_level

        learned = self.leaves(concat).reshape(batch, self.n_leaves - 1, self.state_dim)
        identity = state.unsqueeze(1)
        leaves = torch.cat([identity, learned], dim=1)  # leaf 0 = state
        return self.norm((node_probs.unsqueeze(-1) * leaves).sum(dim=1))


class TreeEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, state_dim: int, depth: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.cell = TreeCell(embed_dim, state_dim, depth)
        self.state_dim = state_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq_len = tokens.shape
        state = torch.zeros(batch, self.state_dim, device=tokens.device)
        embeds = self.embed(tokens)
        for t in range(seq_len):
            state = self.cell(embeds[:, t], state)
        return state


class TreeDecoder(nn.Module):
    """Reads from compressed memory at each step. Input per step =
    concat(position_embed, compressed_state). Decoder's own recurrent state is
    kept separately so the memory is re-injected every step instead of
    decaying through LayerNorm recurrence."""

    def __init__(self, vocab_size: int, embed_dim: int, state_dim: int, depth: int, max_len: int):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.cell = TreeCell(embed_dim + state_dim, state_dim, depth)
        self.output_head = nn.Linear(state_dim, vocab_size)
        self.state_dim = state_dim

    def forward(self, compressed_state: torch.Tensor, seq_len: int) -> torch.Tensor:
        batch = compressed_state.shape[0]
        state = torch.zeros(batch, self.state_dim, device=compressed_state.device)
        logits = []
        for t in range(seq_len):
            pos = torch.full((batch,), t, device=state.device, dtype=torch.long)
            step_input = torch.cat([self.pos_embed(pos), compressed_state], dim=-1)
            state = self.cell(step_input, state)
            logits.append(self.output_head(state))
        return torch.stack(logits, dim=1)


class TreeAutoencoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 16, state_dim: int = 32,
                 depth: int = 3, max_len: int = 5):
        super().__init__()
        self.encoder = TreeEncoder(vocab_size, embed_dim, state_dim, depth)
        self.decoder = TreeDecoder(vocab_size, embed_dim, state_dim, depth, max_len)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        state = self.encoder(tokens)
        return self.decoder(state, tokens.shape[1])
