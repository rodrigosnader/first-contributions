"""Experiment 5: LSTM baseline with the same encoder-decoder-bottleneck
structure. Fair comparison for whether the tree recurrent cell matches a
standard dense recurrent cell at similar parameter count."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from train import Config, run


class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, state_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.cell = nn.LSTMCell(embed_dim, state_dim)
        self.state_dim = state_dim

    def forward(self, tokens):
        batch, seq_len = tokens.shape
        h = torch.zeros(batch, self.state_dim, device=tokens.device)
        c = torch.zeros(batch, self.state_dim, device=tokens.device)
        embeds = self.embed(tokens)
        for t in range(seq_len):
            h, c = self.cell(embeds[:, t], (h, c))
        return h


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, state_dim, max_len):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.cell = nn.LSTMCell(embed_dim + state_dim, state_dim)
        self.output_head = nn.Linear(state_dim, vocab_size)
        self.state_dim = state_dim

    def forward(self, compressed, seq_len):
        batch = compressed.shape[0]
        h = torch.zeros(batch, self.state_dim, device=compressed.device)
        c = torch.zeros(batch, self.state_dim, device=compressed.device)
        logits = []
        for t in range(seq_len):
            pos = torch.full((batch,), t, device=compressed.device, dtype=torch.long)
            step_in = torch.cat([self.pos_embed(pos), compressed], dim=-1)
            h, c = self.cell(step_in, (h, c))
            logits.append(self.output_head(h))
        return torch.stack(logits, dim=1)


class LSTMAutoencoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, state_dim=128, depth=4, max_len=16):
        super().__init__()
        self.encoder = LSTMEncoder(vocab_size, embed_dim, state_dim)
        self.decoder = LSTMDecoder(vocab_size, embed_dim, state_dim, max_len)

    def forward(self, tokens):
        state = self.encoder(tokens)
        return self.decoder(state, tokens.shape[1])


print("=== Tree baseline ===")
cfg_tree = Config(seq_len=16, epochs=30, n_train=3000, n_val=500, name="tree_seq16")
r_tree = run(cfg_tree)

print("\n=== LSTM baseline (same dims) ===")
cfg_lstm = Config(seq_len=16, epochs=30, n_train=3000, n_val=500, name="lstm_seq16")
r_lstm = run(cfg_lstm, model_ctor=LSTMAutoencoder)

print("\n=== SUMMARY ===")
print(f"tree: params={r_tree['n_params']:,}  val_acc={r_tree['best_val_acc']:.3f}  time={r_tree['time_s']:.0f}s")
print(f"lstm: params={r_lstm['n_params']:,}  val_acc={r_lstm['best_val_acc']:.3f}  time={r_lstm['time_s']:.0f}s")
print(f"tree vs lstm val_acc delta: {r_tree['best_val_acc'] - r_lstm['best_val_acc']:+.3f}")
