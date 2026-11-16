"""Pre-train the NATIVE SmolVLM2 server (carsim/server_native.py) on scripts/overhead_data_native.py's practice pictures.

usage: python scripts/train_server_native.py [--unlock 4] [--steps 6000] [--no_steer] [--identity] [--out ...]
  --identity   diagnosis: the reader gets no words; the compressor is handed the asked object's identity directly"""
import argparse, json, os, sys
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    import torch
    from carsim.server_native import NativeServerConfig, NativeServerTrainer
    ap = argparse.ArgumentParser()
    ap.add_argument("--unlock", type=int, default=4)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_steer", action="store_true")
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--heading", action="store_true", help="heading retraining: + arrow practice split, heading/direction weighted 3x")
    ap.add_argument("--init", type=str, default=None, help="warm start from this server checkpoint")
    ap.add_argument("--data", type=Path, default=ROOT / "runs/overhead_native")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    tag = f"native_u{a.unlock}{'_nosteer' if a.no_steer else ''}{'_identity' if a.identity else ''}{'_heading' if a.heading else ''}_s{a.seed}"
    out = a.out or ROOT / "runs/checkpoints" / f"server_{tag}.pt"
    cfg = NativeServerConfig(unlock_layers=a.unlock, steps=a.steps, batch=a.batch, seed=a.seed, steer=not a.no_steer, identity=a.identity,
                             init=a.init)
    if a.heading:
        cfg.train_splits = ("train", "train_heading")
        cfg.w_heading, cfg.w_rel_dir, cfg.w_dir4 = 3.0, 3.0, 1.5
    tr = NativeServerTrainer(cfg, a.data)
    print(tag, {k: f"{v / 1e6:.1f}M" for k, v in tr.n_trainable.items()}, flush=True)
    for row in tr.train():
        if "val_rooms/dir4_acc" in row or row["step"] == 100:
            row["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
            print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
    print("saved", tr.save(out), flush=True)
