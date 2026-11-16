"""Notebook 06 driving tests: the standard 128 rooms (same rooms as notebooks 04/05: scripts/rl_tests.py standard), with
the overhead server running LIVE at every decision.

Modes (for cars that take guidance):
  help        the normal case: guidance from this room, now
  no_help     the learned "no help" tokens (what the car does on its own)
  other_room  guidance taken from ANOTHER simulator's room (wrong help): if the car really uses the help, this hurts
Cars without guidance (F, notebook 04's car head) are tested once.

usage: python scripts/car06_tests.py --car runs/checkpoints/car06_A_s0.pt [--modes help no_help other_room] --out results.json"""
import argparse, json, os, sys, time
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np


def main():
    import torch
    from carsim import overhead
    from carsim.config import CarConfig
    from carsim.data import ActionScaler
    from carsim.guidance import GuidanceAdapter, install
    from carsim.rl_env import ParallelNav
    from carsim.rl_policy import FlowRLPolicy
    from carsim.server import XY_SCALE
    from carsim.vla import load_bc_policy

    ap = argparse.ArgumentParser()
    ap.add_argument("--car", required=True)
    ap.add_argument("--server", default=str(ROOT / "runs/checkpoints/server_native2_u4_heading_s0.pt"))
    ap.add_argument("--modes", nargs="+", default=["help", "no_help", "other_room"])
    ap.add_argument("--per_env", type=int, default=16)
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--map", default=str(ROOT / "runs/checkpoints/server_map_s0.pt"), help="server obstacle map (real-route cars)")
    ap.add_argument("--ppo", default=None, help="a PPO checkpoint (RUN/best.pt) to load on top of --car")
    ap.add_argument("--facts", choices=["auto", "true", "server"], default="auto",
                    help="facts cars: where the help comes from (auto: true facts for C*, the live server otherwise)")
    a = ap.parse_args()

    dev = "cuda"
    rl = FlowRLPolicy(load_bc_policy(a.car, dev), ActionScaler.from_car(CarConfig()), 4, 5, device=dev, full_precision=True)
    if a.ppo:                                            # PPO-trained weights (action expert) on top of the car
        ck = torch.load(a.ppo, map_location="cpu", weights_only=False)
        named = dict(rl.m.named_parameters())
        for n, p in ck["policy"].items():
            named[n].data.copy_(p.to(dev))
        print("loaded", a.ppo, "round", ck.get("round"), flush=True)
    gpath = Path(a.car).with_name(Path(a.car).stem + "_guidance.pt")
    adapter, kind, source = None, "none", None
    if gpath.exists():
        g = torch.load(gpath, map_location="cpu", weights_only=False)
        kind, source = g["kind"], g["source"]
        adapter = GuidanceAdapter(kind).to(dev); adapter.load_state_dict(g["state"]); adapter.eval()
        install(rl.policy, adapter)
    route = source == "route"                        # perfect route help: computed by each simulator from its map
    real_route = source == "route_real"              # real route: tracked car + server's map + server's goal -> planner
    if source == "noisy":                            # trained on server-like errors: tested on the real server
        source = "pred"
    if kind == "facts" and a.facts != "auto":
        source = "true" if a.facts == "true" else "pred"
    modes = a.modes if adapter is not None else ["help"]
    server = None
    if real_route:
        from carsim.route import server_route_facts
        from carsim.server_map import MapHead
        from carsim.tracker import locate_goal, track
        from carsim.world import EnvConfig, Grid
        map_head = MapHead().to(dev); map_head.load_state_dict(torch.load(a.map, map_location="cpu")["state"]); map_head.eval()
        grid = Grid(EnvConfig())
    if kind in ("message", "image") or (kind == "facts" and source in ("pred", "route_real")):
        from carsim.server_native import ServerRuntime
        server = ServerRuntime(a.server, dev)
    print(f"car {Path(a.car).name}: guidance {kind} ({source}), modes {modes}", flush=True)

    def guidance(views, prompts, mode):
        """-> guidance tokens for this decision (or None for cars without guidance)."""
        if adapter is None:
            return None
        B = len(prompts)
        if route:
            x = torch.from_numpy(views["route"].astype(np.float32)).to(dev)
        elif real_route:
            s = server(views["overhead"], prompts)
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                maps = (map_head(s["feats"]).float().sigmoid() > 0.5).cpu().numpy()
            fp = s["facts_pred"][:, :9].cpu().numpy()
            rows = []
            for i in range(B):
                t = track(views["overhead"][i], views["jitter"][i])
                pose = (t["x"], t["y"], t["yaw"]) if t["ok"] else (fp[i, 0] * 4, fp[i, 1] * 4, float(np.arctan2(fp[i, 3], fp[i, 2])))
                goal, _ok = locate_goal(views["overhead"][i], views["jitter"][i], prompts[i], fp[i, 4:6] * 4,
                                        (t["x"], t["y"]) if t["ok"] else None)
                rows.append(server_route_facts(grid, maps[i], prompts[i], goal, pose))
            x = torch.from_numpy(np.stack(rows)).to(dev)
        elif kind == "facts" and source == "true":
            f = overhead.facts(views["pose"].astype(np.float64), views["goal_xy"].astype(np.float64))
            x = torch.from_numpy(np.concatenate([f["car_xy"] / XY_SCALE, f["heading"], f["obj_xy"] / XY_SCALE, f["rel_dir"],
                                                 f["rel_dist"] / XY_SCALE], 1).astype(np.float32)).to(dev)
        else:
            s = server(views["overhead"], prompts)
            x = {"facts": s["facts_pred"][:, :9], "message": s["message"], "image": s["global_tokens"]}[kind]
        if mode == "other_room":
            x = torch.roll(x, 1, 0)                          # each car gets its neighbour simulator's guidance
        present = torch.full((B,), mode != "no_help", device=dev)
        with torch.no_grad():
            return adapter(x, present)

    results = {}
    for mode in modes:
        t0 = time.time()
        venv = ParallelNav(a.n_envs, seed=0, overhead_view=not route, poses=route)
        venv.reseed(2026)
        imgs, states, prompts = venv.reset()
        per = [[] for _ in range(a.n_envs)]
        gen = torch.Generator(device=dev).manual_seed(0)
        while min(len(p) for p in per) < a.per_env:
            rl.m._guidance = guidance(venv.last_views, prompts, mode)
            act = rl.act(imgs, prompts, states, stochastic=False, generator=gen)
            out = venv.step(act["actions"])
            for i, e in enumerate(out["episodes"]):
                if e is not None and len(per[i]) < a.per_env:
                    e["env"], e["k"] = i, len(per[i]); per[i].append(e)
            imgs, states, prompts = out["images"], out["states"], out["prompts"]
        venv.close(); rl.m._guidance = None
        eps = [e for p in per for e in p]
        m = lambda xs: float(np.mean(xs)) if len(xs) else float("nan")
        summ = {"n": len(eps), "success": m([e["success"] for e in eps]),
                "success_hidden_start": m([e["success"] for e in eps if not e["start_visible"]]),
                "success_visible_start": m([e["success"] for e in eps if e["start_visible"]]),
                "touched": m([e["collision_steps"] > 0 for e in eps]),
                "failed_after_seeing_goal": m([(not e["success"]) and e["goal_seen"] for e in eps]),
                "seconds_success": m([e["seconds"] for e in eps if e["success"]]), "minutes": (time.time() - t0) / 60}
        results[mode] = {"summary": summ, "episodes": eps}
        print(mode, json.dumps({k: round(v, 3) for k, v in summ.items()}), flush=True)
    json.dump({"car": a.car, "guidance": kind, "source": source, "server": a.server, **results}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
