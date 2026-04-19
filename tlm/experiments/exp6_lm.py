"""Experiment 6: TLM as next-token language model. Compare BPC against
a parameter-matched LSTM. Tests the central PRD hypothesis: persistent
state can replace attention over history for generation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lm_train import LMConfig, run_lm
from lm_model import TreeLM, LSTMLM


print("=== TreeLM ===")
cfg_tree = LMConfig(seq_len=64, state_dim=128, depth=4, relation_dim=64,
                    epochs=30, n_train=4000, name="tree_lm")
r_tree = run_lm(cfg_tree, model_ctor=TreeLM)

# match LSTM params: TreeLM ~600k, LSTM with state=384 -> ~280k. Need state ~512.
print("\n=== LSTM LM (state=128, smaller) ===")
cfg_lstm_small = LMConfig(seq_len=64, state_dim=128, epochs=30, n_train=4000, name="lstm_lm_128")
r_lstm_small = run_lm(cfg_lstm_small, model_ctor=LSTMLM)

print("\n=== LSTM LM (state=512, param-matched) ===")
cfg_lstm = LMConfig(seq_len=64, state_dim=512, epochs=30, n_train=4000, name="lstm_lm_512")
r_lstm = run_lm(cfg_lstm, model_ctor=LSTMLM)

print("\n=== SUMMARY ===")
print(f"{'model':<20} {'params':>10} {'val_bpc':>10} {'val_acc':>10} {'time':>8}")
for r in (r_tree, r_lstm_small, r_lstm):
    print(f"{r['name']:<20} {r['n_params']:>10,} {r['best_val_bpc']:>10.3f} "
          f"{r['best_val_acc']:>10.3f} {r['time_s']:>7.0f}s")
print(f"\nrandom baseline BPC: 6.022 (vocab=65)")
