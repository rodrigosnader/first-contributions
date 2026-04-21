"""Experiment 24b: re-run regularized transformer WITHOUT label smoothing.

exp24 with label_smoothing=0.1 inflated BPC artificially (floor ~1.21).
Here keep dropout + warmup + cosine + weight_decay but drop label_smoothing
so BPC is directly comparable to exp23."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lm_train import LMConfig
from wikitext_data import make_wikitext_splits
from experiments.exp24_transformer_regularized import TransformerLMReg, train_regularized


if __name__ == "__main__":
    cfg = LMConfig(seq_len=128, n_train=6000, n_val=800, batch_size=32,
                   epochs=20, lr=3e-3, grad_clip=1.0, log_every=5,
                   name="transformer_reg_nosmooth")
    r = train_regularized(cfg, TransformerLMReg, label_smoothing=0.0, dropout=0.1,
                          data_fn=make_wikitext_splits)
    print(f"\n=== SUMMARY ===")
    print(f"{r['name']:<30} params={r['n_params']:,}  "
          f"best_vbpc={r['best_val_bpc']:.3f}  best_val_acc={r['best_val_acc']:.3f}  "
          f"time={r['time_s']:.0f}s")
    print(f"\n--- exp23 vanilla transformer:  vbpc=2.027 acc=0.593 ---")
    print(f"--- exp23 LSTMLMv2:              vbpc=1.901 acc=0.619 ---")
    print(f"--- exp23 TreeLMv2SharedForget:  vbpc=1.926 acc=0.613 ---")
