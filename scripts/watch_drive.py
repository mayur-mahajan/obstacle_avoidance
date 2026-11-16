"""Watch a trained car drive one room in Genesis: saves a video, and can also open the live Genesis 3-D window.

The video shows two views side by side:
  left   the overhead camera (for notebooks 04-05 the car never sees it; for 06-07 it is the server's view).
         Blue = where the car has driven. For route cars: orange = the server's planned route, ring = its goal estimate.
  right  the car's front camera: the only picture the car's own model sees

The drive works as in the tests: one decision every 0.5 s (5 actions of 0.1 s), no random jitter, 60 s limit.
Route cars (notebooks 06-07) get their help from the real server at every decision, as in scripts/car06_tests.py:
SmolVLM2 server + learned obstacle map + colour goal refinement, plus the roof-marker tracker, then the planner.

usage:
  python scripts/watch_drive.py --car runs/checkpoints/bc_car_head.pt --seed 623600484 --out drive.mp4
  python scripts/watch_drive.py --car runs/checkpoints/bc_car_head.pt --ppo runs/rl/ppo_20261113_1842/best.pt ...
  python scripts/watch_drive.py --car runs/checkpoints/car06_R_long_s0.pt [--ppo runs/rl/car07_route_ppo_s0/best.pt] ...
  add --live to also open the Genesis window (needs a display; the drive then runs at real-time speed)
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

PANEL = 576                     # pixels per view in the video
BANNER = 48                     # text strip on top (576 + 48 = 624, a multiple of 16 for the video encoder)


def put_text(img, text, xy, size=0.6, colour=(255, 255, 255), thick=1):
    import cv2
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, thick)
    x, y = xy
    box = img[max(0, y - h - 4): y + base + 2, max(0, x - 4): x + w + 4]
    box[:] = (box * 0.35).astype(img.dtype)                       # darkened strip behind the text
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, size, colour, thick, cv2.LINE_AA)


def to_px(xy, z=0.0):
    """Room points (n, 2) -> pixels in the PANEL-sized overhead view (nominal camera, no jitter)."""
    from carsim.tracker import floor_to_pixel
    xy = np.atleast_2d(np.asarray(xy, float))
    p = floor_to_pixel(np.c_[xy, np.full(len(xy), z)], None, 1024) * PANEL / 1024
    return np.round(p).astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--car", required=True, help="imitation-trained car (notebook 04 or 06)")
    ap.add_argument("--ppo", default=None, help="PPO checkpoint (best.pt) loaded on top of --car (notebooks 05, 07)")
    ap.add_argument("--server", default=str(ROOT / "runs/checkpoints/server_native2_u4_heading_s0.pt"))
    ap.add_argument("--map", default=str(ROOT / "runs/checkpoints/server_map_long_s0.pt"))
    ap.add_argument("--seed", type=int, default=623600484, help="room (layout seed); the default is a standard-test room (blue cylinder, hidden at the start)")
    ap.add_argument("--prompt", default=None, help="instruction to use instead of the room's own")
    ap.add_argument("--out", required=True, help="video file (.mp4)")
    ap.add_argument("--speed", type=float, default=2.0, help="video playback speed (1 = real time)")
    ap.add_argument("--title", default="")
    ap.add_argument("--live", action="store_true", help="also show the drive in the Genesis 3-D window")
    a = ap.parse_args()

    import cv2
    import imageio.v2 as imageio
    import torch
    from carsim import overhead
    from carsim.config import CarConfig
    from carsim.data import ActionScaler
    from carsim.device import pick_device
    from carsim.env import NavEnv
    from carsim.guidance import GuidanceAdapter, install
    from carsim.rl_policy import FlowRLPolicy
    from carsim.vla import load_bc_policy
    from carsim.world import sample_episode

    dev = pick_device()
    rl = FlowRLPolicy(load_bc_policy(a.car, dev), ActionScaler.from_car(CarConfig()), 4, 5, device=dev, full_precision=True)
    if a.ppo:                                   # PPO-trained weights (action expert) on top of the car
        ck = torch.load(a.ppo, map_location="cpu", weights_only=False)
        named = dict(rl.m.named_parameters())
        for n, p in ck["policy"].items():
            named[n].data.copy_(p.to(dev))
        print("loaded", Path(a.ppo).name, "round", ck.get("round"), flush=True)

    # ---- route help (notebooks 06-07) --------------------------------------------------------------------
    gpath = Path(a.car).with_name(Path(a.car).stem + "_guidance.pt")
    route = gpath.exists()
    if route:
        from carsim.route import goal_shape_from_prompt, planned_field, server_route_facts
        from carsim.server_map import MapHead
        from carsim.server_native import ServerRuntime
        from carsim.tracker import locate_goal, track
        from carsim.world import EnvConfig, Grid, shortest_path
        g = torch.load(gpath, map_location="cpu", weights_only=False)
        if g["source"] != "route_real":
            sys.exit(f"{gpath.name}: this viewer handles the real-route cars (notebooks 06-07), not '{g['source']}' cars")
        adapter = GuidanceAdapter(g["kind"]).to(dev); adapter.load_state_dict(g["state"]); adapter.eval()
        install(rl.policy, adapter)
        server = ServerRuntime(a.server, dev)
        map_head = MapHead().to(dev); map_head.load_state_dict(torch.load(a.map, map_location="cpu")["state"]); map_head.eval()
        grid = Grid(EnvConfig())

    @torch.no_grad()
    def route_help(pic, jitter, prompt):
        """One server reading -> (guidance tokens, planned path (n, 2), goal estimate, tracked pose)."""
        s = server(pic[None], [prompt])
        with torch.autocast("cuda", torch.bfloat16, enabled=str(dev).startswith("cuda")):
            blocked = (map_head(s["feats"]).float().sigmoid() > 0.5).cpu().numpy()[0]
        fp = s["facts_pred"][0, :9].cpu().numpy()
        t = track(pic, jitter)
        pose = (t["x"], t["y"], t["yaw"]) if t["ok"] else (fp[0] * 4, fp[1] * 4, float(np.arctan2(fp[3], fp[2])))
        goal, _ok = locate_goal(pic, jitter, prompt, fp[4:6] * 4, (t["x"], t["y"]) if t["ok"] else None)
        x = server_route_facts(grid, blocked, prompt, goal, pose)
        field = planned_field(grid, blocked, goal_shape_from_prompt(prompt), goal[0], goal[1])   # for drawing only
        path = shortest_path(grid, field, pose[0], pose[1]) if np.isfinite(field).any() else np.array([pose[:2], goal])
        tokens = adapter(torch.from_numpy(x)[None].to(dev), torch.ones(1, dtype=torch.bool, device=dev))
        return tokens, path, goal, pose

    # ---- the room ------------------------------------------------------------------------------------------
    car_cfg = overhead.car_config(two_colour=True) if route else CarConfig()
    env = NavEnv(backend="cpu", car_cfg=car_cfg, overview_camera_res=(1024, 1024), show_viewer=a.live)
    spec = sample_episode(a.seed)
    prompt = a.prompt or spec.prompt
    obs, info = env.reset(spec=spec)
    jrng = np.random.default_rng(a.seed + 777)
    gen = torch.Generator(device=dev).manual_seed(0)
    print(f'room {a.seed}: "{prompt}" (goal {"visible" if spec.start_visible else "hidden"} at the start)', flush=True)

    def chase(pose, dist=1.6, height=0.9):
        import genesis.utils.geom as gu
        x, y, yaw = pose
        T = gu.pos_lookat_up_to_T(np.array([x - dist * math.cos(yaw), y - dist * math.sin(yaw), height]),
                                  np.array([x + 0.6 * math.cos(yaw), y + 0.6 * math.sin(yaw), 0.1]), np.array([0.0, 0.0, 1.0]))
        env.scene.viewer.set_camera_pose(pose=T)

    trail, frames, plan, goal_est = [info["pose"][:2]], [], None, None
    status = "driving"

    def frame(info):
        top = cv2.resize(overhead.render(env), (PANEL, PANEL), interpolation=cv2.INTER_AREA)
        if plan is not None and len(plan) > 1:
            cv2.polylines(top, [to_px(plan)], False, (255, 150, 0), 2, cv2.LINE_AA)
        if goal_est is not None:
            cv2.circle(top, tuple(int(v) for v in to_px(goal_est, 0.15)[0]), 14, (255, 150, 0), 2, cv2.LINE_AA)
        cv2.polylines(top, [to_px(np.array(trail))], False, (40, 110, 255), 2, cv2.LINE_AA)
        front = cv2.resize(np.ascontiguousarray(obs["image"]), (PANEL, PANEL), interpolation=cv2.INTER_LINEAR)
        put_text(top, "overhead camera" + (" (the server's view)" if route else " (the car does NOT see this)"), (10, 24), 0.55)
        if route:
            put_text(top, "orange = server's route and goal estimate", (10, PANEL - 34), 0.5, (255, 190, 90))
        put_text(top, "blue = path driven", (10, PANEL - 12), 0.5, (120, 170, 255))
        put_text(front, "front camera (what the car sees)", (10, 24), 0.55)
        img = np.zeros((BANNER + PANEL, 2 * PANEL, 3), np.uint8)
        img[BANNER:, :PANEL], img[BANNER:, PANEL:] = top, front
        col = {"driving": (255, 255, 255), "touching": (255, 120, 120)}.get(status, (120, 255, 120) if "REACHED" in status else (255, 200, 80))
        put_text(img, f'{a.title}  "{prompt}"', (10, 20), 0.55)
        put_text(img, f'{info["t"] / 10:4.1f} s   {status}', (10, 42), 0.55, col)
        return img

    t0 = time.time()
    done = False
    while not done:
        if route:                                         # a fresh overhead picture for the server (camera jitter as in the tests)
            j = overhead.sample_jitter(jrng)
            tokens, plan, goal_est, _pose = route_help(overhead.render(env, j), j, prompt)
            rl.m._guidance = tokens
        act = rl.act(np.ascontiguousarray(obs["image"])[None], [prompt], obs["state"][None], stochastic=False, generator=gen)
        for u in act["actions"][0]:
            obs, term, trunc, info = env.step(u)
            trail.append(info["pose"][:2])
            status = "touching an obstacle" if info["collision"] else "driving"
            if term:
                status = f'REACHED the goal in {info["t"] / 10:.1f} s'
            elif trunc:
                status = "time is up (60 s): failed"
            frames.append(frame(info))
            if a.live:
                chase(info["pose"])
                time.sleep(max(0.0, t0 + info["t"] / 10 - time.time()))     # real-time pace in the window
            if term or trunc:
                done = True
                break
    frames += [frames[-1]] * int(10 * a.speed * 1.5)     # hold the last picture for 1.5 s
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(a.out, frames, fps=10 * a.speed, codec="libx264", quality=6, macro_block_size=16)
    res = {"room": a.seed, "prompt": prompt, "success": bool(info["success"]), "seconds": info["t"] / 10,
           "seconds_touching": info["n_collision_steps"] / 10, "video": str(a.out)}
    print("RESULT", json.dumps(res), f"({time.time() - t0:.0f} s to run)", flush=True)


if __name__ == "__main__":
    main()
