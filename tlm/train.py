import time
import torch
import torch.nn as nn

from data import make_dataset, decode
from model import TreeAutoencoder


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def train():
    torch.manual_seed(0)

    vocab_size = 10_000
    seq_len = 5
    embed_dim = 32
    state_dim = 128
    depth = 4
    n_train = 2000
    n_val = 400
    batch_size = 64
    epochs = 200
    lr = 3e-3

    train_data = make_dataset(n_train, seq_len, vocab_size, seed=1)
    val_data = make_dataset(n_val, seq_len, vocab_size, seed=2)

    model = TreeAutoencoder(vocab_size, embed_dim, state_dim, depth, max_len=seq_len)
    n_params = sum(p.numel() for p in model.parameters())
    info_bits = seq_len * torch.log2(torch.tensor(float(vocab_size))).item()
    print(f"model params: {n_params:,}")
    print(f"vocab={vocab_size}, seq_len={seq_len}, embed={embed_dim}, state={state_dim}, depth={depth}")
    print(f"bottleneck: {state_dim} floats (~{state_dim*32} bits) vs {info_bits:.1f} bits of info per seq\n")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0
        for i in range(0, n_train, batch_size):
            batch = train_data[perm[i:i + batch_size]]
            logits = model(batch)
            loss = loss_fn(logits.reshape(-1, vocab_size), batch.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_acc += accuracy(logits, batch)
            n_batches += 1

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_data)
                val_loss = loss_fn(val_logits.reshape(-1, vocab_size), val_data.reshape(-1)).item()
                val_acc = accuracy(val_logits, val_data)
                ent_enc = model.encoder.cell.tree.routing_entropy(
                    torch.cat([model.encoder.embed(val_data[:, 0]),
                               torch.zeros(val_data.shape[0], state_dim)], dim=-1)
                ).item()
            elapsed = time.time() - t0
            print(f"epoch {epoch:3d} | train_loss {total_loss/n_batches:.4f} acc {total_acc/n_batches:.3f} "
                  f"| val_loss {val_loss:.4f} acc {val_acc:.3f} | enc_ent {ent_enc:.3f} | {elapsed:.0f}s")

    model.eval()
    with torch.no_grad():
        sample = val_data[:5]
        recon = model(sample).argmax(dim=-1)
    print("\nreconstruction samples (val, unseen during training):")
    for src, dst in zip(decode(sample), decode(recon)):
        match = "OK" if src == dst else "!="
        print(f"  {match}  in:  {src}\n        out: {dst}")


if __name__ == "__main__":
    train()
