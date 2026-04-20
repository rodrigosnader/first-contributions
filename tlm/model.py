import torch
import torch.nn as nn


class SoftTree(nn.Module):
    """Soft binary decision tree. Differentiable: mixes all leaf outputs weighted
    by the probability of each root-to-leaf path under the routing decisions."""

    def __init__(self, input_dim: int, output_dim: int, depth: int):
        super().__init__()
        self.depth = depth
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_internal = 2 ** depth - 1
        self.n_leaves = 2 ** depth
        self.router = nn.Linear(input_dim, self.n_internal)
        self.leaves = nn.Linear(input_dim, self.n_leaves * output_dim)
        self._hard = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._hard:
            return self.hard_forward(x)
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

    def hard_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Genuinely sparse: traverse per-example, compute only reached leaf."""
        batch = x.shape[0]
        logits = self.router(x)
        leaf_idx = torch.zeros(batch, dtype=torch.long, device=x.device)
        node_idx = 0
        for level in range(self.depth):
            n_at_level = 2 ** level
            node_logits = logits[:, node_idx:node_idx + n_at_level]
            cur = torch.gather(node_logits, 1, leaf_idx.unsqueeze(1)).squeeze(1)
            go_left = (cur > 0).long()
            leaf_idx = leaf_idx * 2 + (1 - go_left)
            node_idx += n_at_level

        W = self.leaves.weight.view(self.n_leaves, self.output_dim, self.input_dim)
        b = self.leaves.bias.view(self.n_leaves, self.output_dim)
        W_sel = W[leaf_idx]
        b_sel = b[leaf_idx]
        return torch.bmm(W_sel, x.unsqueeze(-1)).squeeze(-1) + b_sel

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


class SharedBackboneSoftTree(nn.Module):
    """Soft tree with weight-shared backbone. All leaves read from the same
    shared projection of the input; they differ only in how they map the
    shared hidden representation to the output. Forces the tree to agree
    on 'what features of the input matter', differ only in 'how to use them'."""

    def __init__(self, input_dim: int, output_dim: int, depth: int,
                 hidden_dim: int = None):
        super().__init__()
        self.depth = depth
        self.output_dim = output_dim
        self.n_internal = 2 ** depth - 1
        self.n_leaves = 2 ** depth
        self.hidden_dim = hidden_dim or max(output_dim // 2, 32)
        self.router = nn.Linear(input_dim, self.n_internal)
        self.shared = nn.Linear(input_dim, self.hidden_dim)
        self.leaves = nn.Linear(self.hidden_dim, self.n_leaves * output_dim)
        self._hard = False  # when True, hard_forward is used (no soft mixing)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._hard:
            return self.hard_forward(x)
        batch = x.shape[0]
        routing_probs = torch.sigmoid(self.router(x))

        node_probs = torch.ones(batch, 1, device=x.device, dtype=x.dtype)
        idx = 0
        for level in range(self.depth):
            n_at_level = 2 ** level
            level_probs = routing_probs[:, idx:idx + n_at_level]
            left = node_probs * level_probs
            right = node_probs * (1.0 - level_probs)
            node_probs = torch.stack([left, right], dim=2).reshape(batch, 2 * n_at_level)
            idx += n_at_level

        h = torch.nn.functional.gelu(self.shared(x))
        leaf_outs = self.leaves(h).reshape(batch, self.n_leaves, self.output_dim)
        return (node_probs.unsqueeze(-1) * leaf_outs).sum(dim=1)

    def hard_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Hard-argmax routing: traverse tree per-example, evaluate ONLY the
        reached leaf's Linear. Genuinely sparse - computes 1 leaf instead of
        n_leaves per example. Gathers weight rows per example via advanced
        indexing, then a batched matmul."""
        batch = x.shape[0]
        logits = self.router(x)  # always compute routing logits
        leaf_idx = torch.zeros(batch, dtype=torch.long, device=x.device)
        node_idx = 0
        for level in range(self.depth):
            n_at_level = 2 ** level
            node_logits = logits[:, node_idx:node_idx + n_at_level]
            cur = torch.gather(node_logits, 1, leaf_idx.unsqueeze(1)).squeeze(1)
            go_left = (cur > 0).long()
            leaf_idx = leaf_idx * 2 + (1 - go_left)
            node_idx += n_at_level

        h = torch.nn.functional.gelu(self.shared(x))  # (B, hidden)
        # leaves.weight: (n_leaves * output_dim, hidden_dim)
        W = self.leaves.weight.view(self.n_leaves, self.output_dim, self.hidden_dim)
        b = self.leaves.bias.view(self.n_leaves, self.output_dim)
        W_sel = W[leaf_idx]  # (B, output_dim, hidden_dim)
        b_sel = b[leaf_idx]  # (B, output_dim)
        return torch.bmm(W_sel, h.unsqueeze(-1)).squeeze(-1) + b_sel


class FiLMSoftTree(nn.Module):
    """Soft tree where leaves are FiLM modulations of a single shared
    transform. All leaves see the same base = Linear(x); they only
    multiply/add per-leaf gamma/beta. Strongest weight sharing: leaves
    are variations of the same theme, cannot learn independent matrices."""

    def __init__(self, input_dim: int, output_dim: int, depth: int):
        super().__init__()
        self.depth = depth
        self.output_dim = output_dim
        self.n_internal = 2 ** depth - 1
        self.n_leaves = 2 ** depth
        self.router = nn.Linear(input_dim, self.n_internal)
        self.shared = nn.Linear(input_dim, output_dim)
        self.gammas = nn.Parameter(torch.ones(self.n_leaves, output_dim))
        self.betas = nn.Parameter(torch.zeros(self.n_leaves, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        routing_probs = torch.sigmoid(self.router(x))

        node_probs = torch.ones(batch, 1, device=x.device, dtype=x.dtype)
        idx = 0
        for level in range(self.depth):
            n_at_level = 2 ** level
            level_probs = routing_probs[:, idx:idx + n_at_level]
            left = node_probs * level_probs
            right = node_probs * (1.0 - level_probs)
            node_probs = torch.stack([left, right], dim=2).reshape(batch, 2 * n_at_level)
            idx += n_at_level

        base = self.shared(x).unsqueeze(1)  # (batch, 1, output_dim)
        leaf_outs = self.gammas.unsqueeze(0) * base + self.betas.unsqueeze(0)
        return (node_probs.unsqueeze(-1) * leaf_outs).sum(dim=1)


class SharedBackboneTreeCell(nn.Module):
    """Recurrent cell using SharedBackboneSoftTree."""

    def __init__(self, input_dim: int, state_dim: int, depth: int):
        super().__init__()
        self.state_dim = state_dim
        self.tree = SharedBackboneSoftTree(input_dim + state_dim, state_dim, depth)
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.norm(self.tree(torch.cat([x, state], dim=-1)))


class FiLMTreeCell(nn.Module):
    """Recurrent cell using FiLMSoftTree."""

    def __init__(self, input_dim: int, state_dim: int, depth: int):
        super().__init__()
        self.state_dim = state_dim
        self.tree = FiLMSoftTree(input_dim + state_dim, state_dim, depth)
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.norm(self.tree(torch.cat([x, state], dim=-1)))


class ForgetGatedTreeCell(nn.Module):
    """LSTM-style per-dim forget gate on top of a tree proposal.
    state_t = f * state_{t-1} + (1 - f) * LayerNorm(tree(x, state_{t-1}))

    Unlike GatedTreeCell (exp9) the LayerNorm is applied ONLY to the
    proposal, not to the final gated mix - that preserves the clean
    state_{t-1} pass-through channel, which is the whole point of the
    LSTM-style gate and what enables long-context retention.

    Forget bias initialized high so f starts ~0.88 (preserve by default)."""

    def __init__(self, input_dim: int, state_dim: int, depth: int):
        super().__init__()
        self.state_dim = state_dim
        self.tree = SoftTree(input_dim + state_dim, state_dim, depth)
        self.forget = nn.Linear(input_dim + state_dim, state_dim)
        self.norm = nn.LayerNorm(state_dim)
        nn.init.constant_(self.forget.bias, 2.0)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([x, state], dim=-1)
        f = torch.sigmoid(self.forget(concat))
        proposed = self.norm(self.tree(concat))
        return f * state + (1 - f) * proposed


class ForgetGatedSharedTreeCell(nn.Module):
    """Same as ForgetGatedTreeCell but uses SharedBackboneSoftTree for the
    proposal. Combines the exp10 winner (shared backbone) with the exp14
    hypothesis (LSTM-style forget gate is needed for long context)."""

    def __init__(self, input_dim: int, state_dim: int, depth: int):
        super().__init__()
        self.state_dim = state_dim
        self.tree = SharedBackboneSoftTree(input_dim + state_dim, state_dim, depth)
        self.forget = nn.Linear(input_dim + state_dim, state_dim)
        self.norm = nn.LayerNorm(state_dim)
        nn.init.constant_(self.forget.bias, 2.0)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([x, state], dim=-1)
        f = torch.sigmoid(self.forget(concat))
        proposed = self.norm(self.tree(concat))
        return f * state + (1 - f) * proposed


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
