"""REAL server routes for every frame of the driving data (the car's training input for the "real route" car), plus
where the route errors come from. All server parts are out-of-sample here: the obstacle map was trained on other rooms,
and the facts head never saw these rooms either.

For each frame: the server's obstacle map (carsim/server_map.py, from the cached picture features) + the server's goal
position + the server's car position/heading (facts head) -> planner -> the 9 route numbers (carsim/route.py).
Saved: route_pred.npy (all from the server) and, for the error breakdown, route_map_only.npy (server map, but TRUE goal
and car pose). --tracker: the car's pose comes from classic marker tracking (scripts/track_frames.py) instead of the
learned facts head -> route_tracked.npy.
usage: python scripts/server_route_frames.py --map runs/checkpoints/server_map_s0.pt [--out runs/overhead_frames/route_follow]"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch


def angle_err(a, b):
    """degrees between the directions to the 0.6 m route point of two route arrays (n, 9)"""
    d = np.arctan2(a[:, 3], a[:, 2]) - np.arctan2(b[:, 3], b[:, 2])
    return np.abs(np.degrees((d + np.pi) % (2 * np.pi) - np.pi))


if __name__ == "__main__":
    from carsim.data import load_training_data
    from carsim.route import server_route_facts
    from carsim.server_map import MapHead, map_scores
    from carsim.world import EnvConfig, Grid, sample_episode
    from carsim.server_map import label_map
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", type=Path, default=ROOT / "runs/checkpoints/server_map_s0.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/overhead_frames/route_follow")
    ap.add_argument("--prep", type=Path, default=ROOT / "runs/datasets/route_follow_prep.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="", help="suffix for the output files (e.g. _long for another map)")
    ap.add_argument("--tracker", action="store_true", help="car pose from tracker_pose.npy (fallback: server facts)")
    a = ap.parse_args()
    _p, table, _f, _e = load_training_data(a.prep)
    N = len(table) if a.limit is None else a.limit
    feats = np.load(a.out / "feats.npy", mmap_mode="r")
    fp = np.load(a.out / "facts_pred.npy")[:, :9]
    true_route = np.load(a.out / "route_true.npy")
    tp = np.load(a.out / "tracker_pose.npy") if a.tracker else None
    gr = np.load(a.out / "goal_refined.npy") if a.tracker else None      # goal: server's pick, colour-refined position
    head = MapHead().cuda(); head.load_state_dict(torch.load(a.map, map_location="cpu")["state"]); head.eval()
    grid, t0 = Grid(EnvConfig()), time.time()
    pred, map_only = np.zeros((N, 9), np.float32), np.zeros((N, 9), np.float32)
    scores, cur, spec = [], None, None
    for s in range(0, N, 64):
        idx = np.arange(s, min(N, s + 64))
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            lg = head(torch.from_numpy(np.array(feats[idx])).cuda()).float()
        maps = (lg.sigmoid() > 0.5).cpu().numpy()
        for k, n in enumerate(idx):
            r = table.iloc[n]
            seed = int(r["sim.episode_seed"])
            if seed != cur:
                spec, cur = sample_episode(seed), seed
                truth = torch.from_numpy(label_map(spec, grid))[None]
            if n == idx[0] or seed != int(table.iloc[n - 1]["sim.episode_seed"]):
                scores.append(map_scores(lg[k:k + 1].cpu(), truth))
            f = fp[n]
            pose_srv = (f[0] * 4, f[1] * 4, float(np.arctan2(f[3], f[2])))
            if tp is not None and tp[n, 3] > 0:
                pose_srv = tuple(float(v) for v in tp[n, :3])
            pose_true = (r["sim.pose.x"], r["sim.pose.y"], r["sim.pose.yaw"])
            goal_srv = (f[4] * 4, f[5] * 4) if gr is None else (float(gr[n, 0]), float(gr[n, 1]))
            pred[n] = server_route_facts(grid, maps[k], r["task"], goal_srv, pose_srv)
            map_only[n] = server_route_facts(grid, maps[k], r["task"], spec.goal[2:4], pose_true)
        if s % 2048 == 0:
            print(f"{s}/{N} {time.time() - t0:.0f}s", flush=True)
    np.save(a.out / (("route_tracked" if a.tracker else "route_pred") + a.tag + ".npy"), pred)
    np.save(a.out / f"route_map_only{a.tag}.npy", map_only)
    t = true_route[:N]
    # how often a route point lies where the car (radius ~0.28 m) would TOUCH a real obstacle: the failure that matters
    from scipy import ndimage
    from carsim.world import occupancy
    touch, cur = np.zeros((N, 2), bool), None
    for n in range(N):
        r = table.iloc[n]
        seed = int(r["sim.episode_seed"])
        if seed != cur:
            sp, cur = sample_episode(seed), seed
            hit = ndimage.distance_transform_edt(~occupancy(grid, sp.objects, sp.obstacles)) * grid.res < grid.cfg.car_radius
            gshape = sp.goal
        x, y, yaw = r["sim.pose.x"], r["sim.pose.y"], r["sim.pose.yaw"]
        c, s_ = np.cos(yaw), np.sin(yaw)
        for col, route in enumerate((pred, t)):
            pts = route[n, :6].reshape(3, 2)
            wx, wy = x + c * pts[:, 0] - s_ * pts[:, 1], y + s_ * pts[:, 0] + c * pts[:, 1]
            near_goal = np.hypot(wx - gshape[2], wy - gshape[3]) < 0.45            # touching the GOAL is the point
            touch[n, col] = any(hit[grid.cell(px, py)] and not ng for px, py, ng in zip(wx, wy, near_goal))
    rep = {"route point touches a real obstacle (this route)": float(100 * touch[:, 0].mean()),
           "route point touches a real obstacle (perfect route)": float(100 * touch[:, 1].mean()),"map_iou": float(np.mean([x["iou"] for x in scores])), "map_recall": float(np.mean([x["recall"] for x in scores])),
           "map_precision": float(np.mean([x["precision"] for x in scores]))}
    for name, r in [("server map only (true goal + pose)", map_only), ("tracker pose + server map + refined goal" if a.tracker else "full server (map + goal + pose)", pred)]:
        e = angle_err(r, t)
        rep[name] = {"median_deg": float(np.median(e)), "p90_deg": float(np.percentile(e, 90)),
                     "over45_pct": float(100 * (e > 45).mean()), "over90_pct": float(100 * (e > 90).mean())}
    print(json.dumps(rep, indent=1), flush=True)
