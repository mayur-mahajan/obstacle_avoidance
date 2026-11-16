"""Perfect-route help (carsim/route.py) for every frame of the driving data, in the same row order as
scripts/overhead_frames.py (so carsim/guidance.py GuidanceStore can use it as source "route").

usage: python scripts/route_frames.py [--prep runs/datasets/route_follow_prep.json] [--out runs/overhead_frames/route_follow]"""
import argparse, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

if __name__ == "__main__":
    from carsim.data import load_training_data
    from carsim.route import route_facts
    from carsim.world import EnvConfig, Grid, sample_episode
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", type=Path, default=ROOT / "runs/datasets/route_follow_prep.json")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/overhead_frames/route_follow")
    a = ap.parse_args()
    _p, table, _f, _e = load_training_data(a.prep)
    meta = np.load(a.out / "meta.npz", allow_pickle=True)
    assert (meta["row"] == table.index.to_numpy()).all(), "row order differs from overhead_frames"
    grid, cur, spec, t0 = Grid(EnvConfig()), None, None, time.time()
    out = np.zeros((len(table), 9), np.float32)
    for n, (_, r) in enumerate(table.iterrows()):
        seed = int(r["sim.episode_seed"])
        if seed != cur:
            spec, cur = sample_episode(seed), seed
        out[n] = route_facts(grid, spec.dist_field, (r["sim.pose.x"], r["sim.pose.y"], r["sim.pose.yaw"]), spec.goal[2:4])
    np.save(a.out / "route_true.npy", out)
    print(f"done {len(out)} frames {time.time() - t0:.0f}s", flush=True)
