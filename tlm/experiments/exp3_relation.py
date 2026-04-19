"""Experiment 3: add a tree-based RelationExtractor between the token embedding
and the encoder cell (PRD step 6). Hypothesis: transforming raw embeddings
into relational features before memory update improves reconstruction without
needing bigger state."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from train import Config, run
from model import SoftTree, TreeCell, TreeDecoder, TreeAutoencoder


class RelationExtractor(nn.Module):
    """One tree that maps embedding -> relation features."""
    def __init__(self, embed_dim: int, relation_dim: int, depth: int):
        super().__init__()
        self.tree = SoftTree(embed_dim, relation_dim, depth)
        self.norm = nn.LayerNorm(relation_dim)

    def forward(self, x):
        return self.norm(self.tree(x))


class RelationEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, state_dim, depth, relation_dim=None):
        super().__init__()
        relation_dim = relation_dim or embed_dim * 2
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.relation = RelationExtractor(embed_dim, relation_dim, depth)
        self.cell = TreeCell(relation_dim, state_dim, depth)
        self.state_dim = state_dim

    def forward(self, tokens):
        batch, seq_len = tokens.shape
        state = torch.zeros(batch, self.state_dim, device=tokens.device)
        embeds = self.embed(tokens)
        for t in range(seq_len):
            rel = self.relation(embeds[:, t])
            state = self.cell(rel, state)
        return state


class RelationAutoencoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, state_dim=32, depth=3, max_len=5,
                 relation_dim=None):
        super().__init__()
        self.encoder = RelationEncoder(vocab_size, embed_dim, state_dim, depth, relation_dim)
        self.decoder = TreeDecoder(vocab_size, embed_dim, state_dim, depth, max_len)

    def forward(self, tokens):
        state = self.encoder(tokens)
        return self.decoder(state, tokens.shape[1])


print("=== baseline (no relation extractor) ===")
cfg_base = Config(seq_len=16, epochs=30, n_train=3000, n_val=500, name="base_seq16")
r_base = run(cfg_base)

print("\n=== with relation extractor ===")
cfg_rel = Config(seq_len=16, epochs=30, n_train=3000, n_val=500, name="rel_seq16")
r_rel = run(cfg_rel, model_ctor=RelationAutoencoder, extra_model_kwargs={"relation_dim": 64})

print("\n=== SUMMARY ===")
print(f"baseline:  params={r_base['n_params']:,}  val_acc={r_base['best_val_acc']:.3f}")
print(f"relation:  params={r_rel['n_params']:,}  val_acc={r_rel['best_val_acc']:.3f}")
print(f"delta: {r_rel['best_val_acc'] - r_base['best_val_acc']:+.3f}")
