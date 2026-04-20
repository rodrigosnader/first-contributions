"""TLM as a language model: emit next-token logits at every step. Same
TreeCell + RelationExtractor as the autoencoder, but no decoder - the
state itself is the prediction substrate."""
import torch
import torch.nn as nn

from model import (
    SoftTree, TreeCell,
    ResidualTreeCell, GatedTreeCell, IdentityLeafTreeCell,
    SharedBackboneTreeCell, FiLMTreeCell,
    SharedBackboneSoftTree,
    ForgetGatedTreeCell, ForgetGatedSharedTreeCell,
)


class RelationExtractor(nn.Module):
    def __init__(self, embed_dim: int, relation_dim: int, depth: int):
        super().__init__()
        self.tree = SoftTree(embed_dim, relation_dim, depth)
        self.norm = nn.LayerNorm(relation_dim)

    def forward(self, x):
        return self.norm(self.tree(x))


class ContextExtractor(nn.Module):
    """Mirror of RelationExtractor on the decoder side: extracts a context
    vector from the current state before producing next-token logits."""

    def __init__(self, state_dim: int, context_dim: int, depth: int):
        super().__init__()
        self.tree = SoftTree(state_dim, context_dim, depth)
        self.norm = nn.LayerNorm(context_dim)

    def forward(self, state):
        return self.norm(self.tree(state))


class TreeDecoderHead(nn.Module):
    """Final tree: takes (state, context) and emits vocab logits. No output
    LayerNorm - raw logits feed into cross-entropy."""

    def __init__(self, state_dim: int, context_dim: int, vocab_size: int, depth: int):
        super().__init__()
        self.tree = SoftTree(state_dim + context_dim, vocab_size, depth)

    def forward(self, state, context):
        return self.tree(torch.cat([state, context], dim=-1))


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


class TreeLMv2(nn.Module):
    """Full TLM: two trees on the encoder side (relation + cell), two on the
    decoder side (context + head). Mirrors the autoencoder's symmetric
    tree-encoder / tree-decoder structure but applied per-step for LM."""

    def __init__(self, vocab_size: int, embed_dim: int = 32, state_dim: int = 128,
                 depth: int = 4, relation_dim: int = 64, context_dim: int = 64,
                 max_len: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.relation = RelationExtractor(embed_dim, relation_dim, depth)
        self.cell = TreeCell(relation_dim, state_dim, depth)
        self.context = ContextExtractor(state_dim, context_dim, depth)
        self.head = TreeDecoderHead(state_dim, context_dim, vocab_size, depth)
        self.state_dim = state_dim

    def forward(self, tokens):
        batch, seq_len = tokens.shape
        state = torch.zeros(batch, self.state_dim, device=tokens.device)
        embeds = self.embed(tokens)
        all_logits = []
        for t in range(seq_len):
            rel = self.relation(embeds[:, t])
            state = self.cell(rel, state)
            ctx = self.context(state)
            all_logits.append(self.head(state, ctx))
        return torch.stack(all_logits, dim=1)

    @torch.no_grad()
    def generate(self, prompt, n_new_tokens, temperature=1.0):
        self.eval()
        batch = prompt.shape[0]
        state = torch.zeros(batch, self.state_dim, device=prompt.device)
        for t in range(prompt.shape[1]):
            rel = self.relation(self.embed(prompt[:, t]))
            state = self.cell(rel, state)
        out = [prompt]
        cur = prompt[:, -1:]
        for _ in range(n_new_tokens):
            rel = self.relation(self.embed(cur[:, 0]))
            state = self.cell(rel, state)
            ctx = self.context(state)
            logits = self.head(state, ctx) / temperature
            probs = torch.softmax(logits, dim=-1)
            cur = torch.multinomial(probs, 1)
            out.append(cur)
        return torch.cat(out, dim=1)


class TreeLMv2Residual(TreeLMv2):
    """TreeLMv2 but the encoder cell is ResidualTreeCell (preserve state by default)."""
    def __init__(self, vocab_size, embed_dim=32, state_dim=128, depth=4,
                 relation_dim=64, context_dim=64, max_len=128):
        super().__init__(vocab_size, embed_dim, state_dim, depth,
                         relation_dim, context_dim, max_len)
        self.cell = ResidualTreeCell(relation_dim, state_dim, depth)


class TreeLMv2Gated(TreeLMv2):
    """TreeLMv2 but the encoder cell is GatedTreeCell (GRU-style forget/keep gate)."""
    def __init__(self, vocab_size, embed_dim=32, state_dim=128, depth=4,
                 relation_dim=64, context_dim=64, max_len=128):
        super().__init__(vocab_size, embed_dim, state_dim, depth,
                         relation_dim, context_dim, max_len)
        self.cell = GatedTreeCell(relation_dim, state_dim, depth)


class TreeLMv2Shared(TreeLMv2):
    """TreeLMv2 with SharedBackboneTreeCell in the encoder: all leaves read
    from a shared projection of (input, state), differ only in output mapping."""
    def __init__(self, vocab_size, embed_dim=32, state_dim=128, depth=4,
                 relation_dim=64, context_dim=64, max_len=128):
        super().__init__(vocab_size, embed_dim, state_dim, depth,
                         relation_dim, context_dim, max_len)
        self.cell = SharedBackboneTreeCell(relation_dim, state_dim, depth)


class SharedRelationExtractor(nn.Module):
    def __init__(self, embed_dim, relation_dim, depth):
        super().__init__()
        self.tree = SharedBackboneSoftTree(embed_dim, relation_dim, depth)
        self.norm = nn.LayerNorm(relation_dim)
    def forward(self, x):
        return self.norm(self.tree(x))


class SharedContextExtractor(nn.Module):
    def __init__(self, state_dim, context_dim, depth):
        super().__init__()
        self.tree = SharedBackboneSoftTree(state_dim, context_dim, depth)
        self.norm = nn.LayerNorm(context_dim)
    def forward(self, state):
        return self.norm(self.tree(state))


class SharedTreeDecoderHead(nn.Module):
    def __init__(self, state_dim, context_dim, vocab_size, depth):
        super().__init__()
        self.tree = SharedBackboneSoftTree(state_dim + context_dim, vocab_size, depth)
    def forward(self, state, context):
        return self.tree(torch.cat([state, context], dim=-1))


class TreeLMv2FullShared(TreeLMv2):
    """Every SoftTree in the model is replaced with a SharedBackboneSoftTree:
    relation extractor, encoder cell, context extractor, decoder head. The
    most aggressive weight-sharing variant; expected to further reduce
    parameters and overfit beyond TreeLMv2Shared (which only shares in cell)."""
    def __init__(self, vocab_size, embed_dim=32, state_dim=128, depth=4,
                 relation_dim=64, context_dim=64, max_len=128):
        super().__init__(vocab_size, embed_dim, state_dim, depth,
                         relation_dim, context_dim, max_len)
        self.relation = SharedRelationExtractor(embed_dim, relation_dim, depth)
        self.cell = SharedBackboneTreeCell(relation_dim, state_dim, depth)
        self.context = SharedContextExtractor(state_dim, context_dim, depth)
        self.head = SharedTreeDecoderHead(state_dim, context_dim, vocab_size, depth)


class TreeLMv2FiLM(TreeLMv2):
    """TreeLMv2 with FiLMTreeCell in the encoder: all leaves are FiLM
    modulations (gamma, beta) of a single shared Linear transform."""
    def __init__(self, vocab_size, embed_dim=32, state_dim=128, depth=4,
                 relation_dim=64, context_dim=64, max_len=128):
        super().__init__(vocab_size, embed_dim, state_dim, depth,
                         relation_dim, context_dim, max_len)
        self.cell = FiLMTreeCell(relation_dim, state_dim, depth)


class TreeLMv2Forget(TreeLMv2):
    """TreeLMv2 with ForgetGatedTreeCell: baseline tree + per-dim forget gate.
    Tests whether an LSTM-style preservation channel closes the exp12 gap."""
    def __init__(self, vocab_size, embed_dim=32, state_dim=128, depth=4,
                 relation_dim=64, context_dim=64, max_len=128):
        super().__init__(vocab_size, embed_dim, state_dim, depth,
                         relation_dim, context_dim, max_len)
        self.cell = ForgetGatedTreeCell(relation_dim, state_dim, depth)


class TreeLMv2SharedForget(TreeLMv2):
    """TreeLMv2 with ForgetGatedSharedTreeCell: shared-backbone tree proposal
    + per-dim forget gate. Combines exp10 winner with exp14 hypothesis."""
    def __init__(self, vocab_size, embed_dim=32, state_dim=128, depth=4,
                 relation_dim=64, context_dim=64, max_len=128):
        super().__init__(vocab_size, embed_dim, state_dim, depth,
                         relation_dim, context_dim, max_len)
        self.cell = ForgetGatedSharedTreeCell(relation_dim, state_dim, depth)


class TreeLMv2IdLeaf(TreeLMv2):
    """TreeLMv2 but the encoder cell has an identity leaf (one routing option
    preserves state verbatim)."""
    def __init__(self, vocab_size, embed_dim=32, state_dim=128, depth=4,
                 relation_dim=64, context_dim=64, max_len=128):
        super().__init__(vocab_size, embed_dim, state_dim, depth,
                         relation_dim, context_dim, max_len)
        self.cell = IdentityLeafTreeCell(relation_dim, state_dim, depth)


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


class LSTMLMv2(nn.Module):
    """LSTM baseline with the same shape as TreeLMv2: 2-layer MLP before the
    cell (mirrors RelationExtractor), LSTM cell, 2-layer MLP after (mirrors
    ContextExtractor), MLP head (mirrors TreeDecoderHead). Matmul-based
    analog of the tree components for a fair architectural comparison."""

    def __init__(self, vocab_size: int, embed_dim: int = 32, state_dim: int = 128,
                 depth: int = 0, relation_dim: int = 64, context_dim: int = 64,
                 max_len: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pre = nn.Sequential(
            nn.Linear(embed_dim, relation_dim * 4),
            nn.GELU(),
            nn.Linear(relation_dim * 4, relation_dim),
            nn.LayerNorm(relation_dim),
        )
        self.cell = nn.LSTMCell(relation_dim, state_dim)
        self.post = nn.Sequential(
            nn.Linear(state_dim, context_dim * 4),
            nn.GELU(),
            nn.Linear(context_dim * 4, context_dim),
            nn.LayerNorm(context_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(state_dim + context_dim, (state_dim + context_dim) * 2),
            nn.GELU(),
            nn.Linear((state_dim + context_dim) * 2, vocab_size),
        )
        self.state_dim = state_dim

    def forward(self, tokens):
        batch, seq_len = tokens.shape
        h = torch.zeros(batch, self.state_dim, device=tokens.device)
        c = torch.zeros(batch, self.state_dim, device=tokens.device)
        embeds = self.embed(tokens)
        all_logits = []
        for t in range(seq_len):
            rel = self.pre(embeds[:, t])
            h, c = self.cell(rel, (h, c))
            ctx = self.post(h)
            all_logits.append(self.head(torch.cat([h, ctx], dim=-1)))
        return torch.stack(all_logits, dim=1)

    @torch.no_grad()
    def generate(self, prompt, n_new_tokens, temperature=1.0):
        self.eval()
        batch = prompt.shape[0]
        h = torch.zeros(batch, self.state_dim, device=prompt.device)
        c = torch.zeros(batch, self.state_dim, device=prompt.device)
        for t in range(prompt.shape[1]):
            rel = self.pre(self.embed(prompt[:, t]))
            h, c = self.cell(rel, (h, c))
        out = [prompt]
        cur = prompt[:, -1:]
        for _ in range(n_new_tokens):
            rel = self.pre(self.embed(cur[:, 0]))
            h, c = self.cell(rel, (h, c))
            ctx = self.post(h)
            logits = self.head(torch.cat([h, ctx], dim=-1)) / temperature
            probs = torch.softmax(logits, dim=-1)
            cur = torch.multinomial(probs, 1)
            out.append(cur)
        return torch.cat(out, dim=1)
