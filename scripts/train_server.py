"""Pre-train the server's overhead reader + compressor (carsim/server.py) on scripts/overhead_data.py's practice pictures.

usage: python scripts/train_server.py [--unlock 4] [--steps 6000] [--seed 0] [--data runs/overhead_pretrain]
                                      [--out runs/checkpoints/server_u4_s0.pt]
Prints a log line every 500 steps (with validation scores) and saves the checkpoint (its log included)."""
import argparse, json, os, sys
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from carsim.server import ServerConfig, ServerTrainer
    ap = argparse.ArgumentParser()
    ap.add_argument("--unlock", type=int, default=4, help="last N language layers of the reader that may change (0 = frozen)")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", type=Path, default=ROOT / "runs/overhead_pretrain")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or ROOT / "runs/checkpoints" / f"server_u{a.unlock}_s{a.seed}.pt"
    tr = ServerTrainer(ServerConfig(unlock_layers=a.unlock, steps=a.steps, seed=a.seed), a.data)
    print(out.stem, {k: f"{v / 1e6:.1f}M" for k, v in tr.n_trainable.items()}, flush=True)
    for row in tr.train():
        if row["step"] % 500 == 0 or row["step"] == 100:
            print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
    print("saved", tr.save(out), flush=True)
