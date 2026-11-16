"""Notebook 06 car training with overhead guidance (carsim/guided_train.py), from the command line.

usage: python scripts/train_guided.py --model Cstar|C|Cn|Rstar|R|A|I|F [--seed 0] [--steps 5000] [--init CKPT] [--out PATH]
  F      front camera only (new route-following demos)        C*  true facts (upper bound)
  C      the server's predicted facts                          A   the server's 8 x 128 message (main design)
  I      the overhead picture's whole view (64 tokens)
  Rstar  perfect ROUTE help: next points on the shortest obstacle-free path (carsim/route.py), help dropout 20%
  R      REAL server route: car pose from marker tracking, obstacles from the learned map, goal from the server
  Cn     true facts + the server's real errors resampled (carsim/guidance.py ServerLikeNoise), help dropout 20%"""
import argparse, json, os, sys
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS = {"F": dict(guidance="none"), "Cstar": dict(guidance="facts", source="true"), "C": dict(guidance="facts", source="pred"),
          "Rstar": dict(guidance="facts", source="route", drop_p=0.2),
          "R": dict(guidance="facts", source="route_real", drop_p=0.2),
          "Cn": dict(guidance="facts", source="noisy", val_source="pred", drop_p=0.2),
          "A": dict(guidance="message"), "I": dict(guidance="image")}

if __name__ == "__main__":
    import torch
    from carsim.guided_train import GuidedBCTrainer, GuidedConfig
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--init", default=None)
    ap.add_argument("--prep", type=Path, default=ROOT / "runs/datasets/route_follow_prep.json")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--route_file", default="route_tracked.npy", help="model R: which real-route file to train on")
    a = ap.parse_args()
    out = a.out or ROOT / "runs/checkpoints" / f"car06_{a.model}_s{a.seed}.pt"
    cfg = GuidedConfig(seed=a.seed, steps=a.steps, init=a.init, route_file=a.route_file, **MODELS[a.model])
    tr = GuidedBCTrainer(cfg, a.prep)
    print(out.stem, "|", tr.data_note, "|", {k: f"{v / 1e6:.2f}M" for k, v in tr.n_trainable.items()}, "| init", a.init, flush=True)
    for row in tr.train():
        if row["step"] % 500 == 0 or "val_loss_with_help" in row or row["step"] == 20:
            row["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
    print("saved", tr.save(out), flush=True)
