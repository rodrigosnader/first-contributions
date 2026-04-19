import time
import torch
import torch.nn as nn

from data import make_shakespeare_splits, decode
from model import TreeAutoencoder


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def train():
    torch.manual_seed(0)

    seq_len = 16
    embed_dim = 32
    state_dim = 128
    depth = 4
    n_train = 4000
    n_val = 500
    batch_size = 64
    epochs = 40
    lr = 3e-3
    grad_clip = 5.0

    train_data, val_data, stoi, itos = make_shakespeare_splits(n_train, n_val, seq_len)
    vocab_size = len(stoi)

    # Majority-char baseline: train val acc if we always predicted the most common char
    most_common_id = torch.bincount(train_data.reshape(-1), minlength=vocab_size).argmax()
    baseline_acc = (val_data == most_common_id).float().mean().item()

    model = TreeAutoencoder(vocab_size, embed_dim, state_dim, depth, max_len=seq_len)
    n_params = sum(p.numel() for p in model.parameters())
    info_bits = seq_len * torch.log2(torch.tensor(float(vocab_size))).item()
    print(f"model params: {n_params:,}")
    print(f"vocab={vocab_size}, seq_len={seq_len}, embed={embed_dim}, state={state_dim}, depth={depth}")
    print(f"bottleneck: {state_dim} floats vs {info_bits:.1f} bits of info per window")
    print(f"majority-char baseline val acc: {baseline_acc:.3f}\n")

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total_loss += loss.item()
            total_acc += accuracy(logits, batch)
            n_batches += 1

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_data)
                val_loss = loss_fn(val_logits.reshape(-1, vocab_size), val_data.reshape(-1)).item()
                val_acc = accuracy(val_logits, val_data)
                ent_enc = model.encoder.cell.tree.routing_entropy(
                    torch.cat([model.encoder.embed(val_data[:, 0]),
                               torch.zeros(val_data.shape[0], state_dim)], dim=-1)
                ).item()
            print(f"epoch {epoch:3d} | train_loss {total_loss/n_batches:.4f} acc {total_acc/n_batches:.3f} "
                  f"| val_loss {val_loss:.4f} acc {val_acc:.3f} | enc_ent {ent_enc:.3f} | {time.time()-t0:.0f}s")

    model.eval()
    with torch.no_grad():
        sample = val_data[:4]
        recon = model(sample).argmax(dim=-1)
    print("\nreconstruction samples (val, from unseen portion of Shakespeare):")
    for src, dst in zip(decode(sample, itos), decode(recon, itos)):
        char_match = sum(a == b for a, b in zip(src, dst))
        print(f"  [{char_match}/{len(src)} chars]")
        print(f"    in:  {src!r}")
        print(f"    out: {dst!r}")


if __name__ == "__main__":
    train()
