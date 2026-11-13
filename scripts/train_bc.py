"""Command-line version of notebook 04's training (carsim/bc_train.py), e.g. to train another random seed.

usage: python scripts/train_bc.py --head car [--seed 1] [--steps 5000] [--prep runs/datasets/scripted_close_prep.json]
                                  [--out runs/checkpoints/bc_car_head_s1.pt]
Writes the checkpoint (+ _meta.json, _quiz_head.pt, _train_log.json next to it) and prints a log line every 500 steps.
The defaults are the notebook's recipe (BCConfig); --seed changes only the random seed."""
import argparse, json, os, sys
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from carsim.bc_train import BCConfig, BCTrainer
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="smolvla", choices=["smolvla", "car"], help="action head: smolVLA's pretrained one or a fresh car one")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--prep", type=Path, default=ROOT / "runs/datasets/scripted_close_prep.json", help="training file (notebook 03)")
    ap.add_argument("--out", type=Path, default=None, help="checkpoint path (default runs/checkpoints/bc_<head>_head_s<seed>.pt)")
    a = ap.parse_args()
    out = a.out or ROOT / "runs/checkpoints" / f"bc_{a.head}_head_s{a.seed}.pt"
    tr = BCTrainer(BCConfig(head=a.head, seed=a.seed, steps=a.steps), a.prep)
    print(out.stem, "|", tr.data_note, "|", {k: f"{v / 1e6:.1f}M" for k, v in tr.n_trainable.items()}, flush=True)
    for row in tr.train():
        if row["step"] % 500 == 0 or "quiz_val_acc" in row:
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
    print("saved", tr.save(out), flush=True)
