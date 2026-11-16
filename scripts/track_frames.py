"""Classic marker tracking (carsim/tracker.py) for every frame of the driving data: re-render the overhead picture
(stored car pose + camera jitter, as scripts/overhead_frames.py) and track the car (calibrated camera).
Saved: OUT/tracker_pose.npy (N, 4): x, y, yaw, found (1/0); OUT/goal_refined.npy (N, 3): the goal's x, y = the server's
rough goal position refined by colour + geometry (carsim/tracker.py locate_goal), found (1/0).
usage: python scripts/track_frames.py [--out runs/overhead_frames/route_follow]"""
import argparse, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

if __name__ == "__main__":
    from carsim import overhead
    from carsim.env import NavEnv
    from carsim.data import load_training_data
    from carsim.tracker import locate_goal, track
    from carsim.world import sample_episode
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "runs/overhead_frames/route_follow")
    a = ap.parse_args()
    meta = np.load(a.out / "meta.npz", allow_pickle=True)
    env = NavEnv(car_cfg=overhead.car_config(two_colour=True), overview_camera_res=(1024, 1024), backend="cpu")
    N, cur, t0 = len(meta["seed"]), None, time.time()
    out, goal = np.zeros((N, 4), np.float32), np.zeros((N, 3), np.float32)
    fp = np.load(a.out / "facts_pred.npy")
    _p, table, _f, _e = load_training_data(ROOT / "runs/datasets/route_follow_prep.json")
    for k in range(N):
        seed = int(meta["seed"][k])
        if seed != cur:
            env.reset(spec=sample_episode(seed)); cur = seed
        env.car.reset(*map(float, meta["pose"][k]))
        pic = overhead.render(env, meta["jitter"][k])
        t = track(pic, meta["jitter"][k])
        out[k] = [t["x"], t["y"], t["yaw"], float(t["ok"])]
        (gx, gy), ok = locate_goal(pic, meta["jitter"][k], table.iloc[k]["task"], fp[k, 4:6] * 4, (t["x"], t["y"]) if t["ok"] else None)
        goal[k] = [gx, gy, float(ok)]
        if k % 2000 == 0:
            print(f"{k}/{N} {time.time() - t0:.0f}s", flush=True)
    np.save(a.out / "tracker_pose.npy", out); np.save(a.out / "goal_refined.npy", goal)
    print(f"done, found {100 * out[:, 3].mean():.2f}%", flush=True)
