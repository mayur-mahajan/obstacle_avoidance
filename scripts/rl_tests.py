"""Scorecard tests for an RL-trained (or the starting) policy. No jitter; fixed seeds, so results are repeatable
and different checkpoints face exactly the same drives.

  standard   128 standard layouts (8 simulators x 16, seed 2026) -> success, goal hidden at start, touching
  which      controlled "which object?" tests on an empty floor (30 layouts each):
               alone        goal alone in view                      -> can it drive to an object?
               pair         goal + decoy in view (same colour / same shape / unrelated); every layout is
                            driven twice, asking for each object -> does it follow the WORDS?
               behind       goal behind, nothing in front           -> does it search?
               behind+decoy goal behind, decoy in front             -> does it settle for the decoy?

usage: python scripts/rl_tests.py standard|which --bc BC.pt [--ckpt RUN/best.pt] --out results.json [--detail drives.json]
       --bc = an imitation checkpoint (notebook 04); --ckpt = an RL checkpoint trained from it (notebook 05)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

DETAIL = None  # set to a list to log every drive in detail (--detail)
REACH = 0.35   # m from the front bumper to an object's surface = "went to it"


def load_policy(ckpt, bc):
    import torch
    from carsim.config import CarConfig
    from carsim.data import ActionScaler
    from carsim.device import pick_device
    from carsim.rl_policy import FlowRLPolicy
    from carsim.vla import load_bc_policy

    dev = pick_device()
    rl = FlowRLPolicy(load_bc_policy(bc, dev), ActionScaler.from_car(CarConfig()), 4, 5, device=dev, full_precision=True)
    if ckpt:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        named = dict(rl.m.named_parameters())
        for n, p in ck["policy"].items():
            named[n].data.copy_(p.to(dev))
        print("loaded", ckpt, "round", ck.get("round"), flush=True)
    return rl


def standard_test(rl, per_env=16, n_envs=8, seed=2026):
    import torch
    from carsim.rl_env import ParallelNav

    venv = ParallelNav(n_envs, seed=0)
    venv.reseed(seed)
    imgs, states, prompts = venv.reset()
    per = [[] for _ in range(n_envs)]
    gen = torch.Generator(device=rl.device).manual_seed(0)
    while min(len(p) for p in per) < per_env:
        a = rl.act(imgs, prompts, states, stochastic=False, generator=gen)
        out = venv.step(a["actions"])
        for i, e in enumerate(out["episodes"]):
            if e is not None and len(per[i]) < per_env:
                e["env"], e["k"] = i, len(per[i])
                per[i].append(e)
        imgs, states, prompts = out["images"], out["states"], out["prompts"]
    venv.close()
    eps = [e for p in per for e in p]
    m = lambda xs: float(np.mean(xs)) if xs else float("nan")
    summary = {"n": len(eps), "success": m([e["success"] for e in eps]),
               "success_hidden_start": m([e["success"] for e in eps if not e["start_visible"]]),
               "success_visible_start": m([e["success"] for e in eps if e["start_visible"]]),
               "touched": m([e["collision_steps"] > 0 for e in eps])}
    return {"summary": summary, "episodes": eps}


def which_test(rl, n=30, seed=7):
    import torch
    from carsim.data import Paraphraser
    from carsim.env import NavEnv
    from carsim.curriculum import _decoy, _pick_goal, _looks
    from carsim.world import EpisodeSpec, Grid, geodesic_field, goal_in_view, occupancy, surface_distance

    env = NavEnv(backend="cpu")
    grid, cfg = Grid(env.cfg), env.cfg
    rng = np.random.default_rng(seed)
    para = Paraphraser("train", seed=seed)
    gen = torch.Generator(device=rl.device).manual_seed(0)

    def build(rel, goal_index):
        floor, wall, light = _looks(rng)
        x0, y0, yaw = rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), rng.uniform(-math.pi, math.pi)
        objects = [(o[0], o[1], float(x0 + d * math.cos(yaw + math.radians(b))), float(y0 + d * math.sin(yaw + math.radians(b))))
                   for o, b, d in rel]
        g = objects[goal_index]
        dist, _ = geodesic_field(grid, occupancy(grid, objects, []), g[0], g[2], g[3])
        vis = goal_in_view(cfg, grid, occupancy(grid, objects, [], skip_object=goal_index), (x0, y0, yaw), (g[2], g[3]))
        return EpisodeSpec(seed=-1, goal_index=goal_index, objects=objects, obstacles=[], start=(float(x0), float(y0), float(yaw)),
                           prompt="", floor_index=floor, wall_index=wall, light=light, hard_distractors=False,
                           start_visible=bool(vis), geodesic_start=float(grid.lookup(dist, x0, y0)), dist_field=dist)

    def drive(spec, seconds):
        obs, info = env.reset(spec=spec)
        prompt = para(spec.goal[1], spec.goal[0])
        reached = None
        trace = {"path": [], "min_dist": [9.9] * len(spec.objects), "actions": []}   # detail log (does not change scoring)
        for _ in range(int(seconds * 2)):
            a = rl.act(np.ascontiguousarray(obs["image"])[None], [prompt], obs["state"][None], stochastic=False, generator=gen)["actions"][0]
            for act in a:
                obs, term, trunc, info = env.step(act)
                x, y, th = info["pose"]
                bx, by = x + cfg.car_half_length * math.cos(th), y + cfg.car_half_length * math.sin(th)
                ds = [surface_distance(s, ox, oy, bx, by) for (s, c, ox, oy) in spec.objects]
                trace["path"].append([round(float(x), 3), round(float(y), 3), round(float(th), 3)])
                trace["actions"].append([round(float(v), 3) for v in act])
                trace["min_dist"] = [min(m, float(d)) for m, d in zip(trace["min_dist"], ds)]
                trace["final_dist"] = [float(d) for d in ds]
                for k, (s, c, ox, oy) in enumerate(spec.objects):
                    if surface_distance(s, ox, oy, bx, by) <= REACH:
                        reached = k
                        break
                if reached is not None or term:
                    break
            if reached is not None or term:
                break
        if DETAIL is not None:
            DETAIL.append({"prompt": prompt, "objects": spec.objects, "start": spec.start, "goal_index": spec.goal_index,
                           "reached": reached, "steps": len(trace["path"]), **trace})
        return prompt, reached

    res = []
    for _ in range(n):
        g = _pick_goal(rng)
        p, k = drive(build([(g, rng.uniform(-15, 15), rng.uniform(2.0, 3.0))], 0), 20)
        res.append({"test": "alone", "goal": f"{g[1]} {g[0]}", "went": "goal" if k == 0 else "none"})
    for kind in ("same_colour", "same_shape", "unrelated"):
        for i in range(n):
            g = _pick_goal(rng); dcy = _decoy(rng, g, kind)
            side = rng.choice([-1, 1]); b = rng.uniform(12, 18); d = rng.uniform(2.0, 3.0)
            rel = [(g, side * b, d), (dcy, -side * b, d)]
            st = rng.bit_generator.state
            for ask in (0, 1):  # same layout, ask for each object in turn
                rng.bit_generator.state = st
                p, k = drive(build(rel, ask), 20)
                goal_left = (side if ask == 0 else -side) > 0
                res.append({"test": "pair", "kind": kind, "layout": i, "ask": ask, "went_object": k,
                            "went": "none" if k is None else ("goal" if k == ask else "decoy"),
                            "went_side": None if k is None else ("left" if (k == ask) == goal_left else "right")})
    for with_decoy in (False, True):
        for i in range(n):
            g = _pick_goal(rng)
            rel = [(g, 180 + rng.uniform(-30, 30), rng.uniform(2.0, 3.0))]
            if with_decoy:
                rel.append((_decoy(rng, g, ["same_colour", "same_shape", "unrelated"][i % 3]), rng.uniform(-10, 10), rng.uniform(2.0, 3.0)))
            p, k = drive(build(rel, 0), 40)
            res.append({"test": "behind+decoy" if with_decoy else "behind", "went": "none" if k is None else ("goal" if k == 0 else "decoy")})

    def rate(rs, what="goal"):
        return float(np.mean([r["went"] == what for r in rs])) if rs else float("nan")
    pairs = [r for r in res if r["test"] == "pair"]
    lay = {}
    for r in pairs:
        lay.setdefault((r["kind"], r["layout"]), []).append(r)
    same_obj = np.mean([len(v) == 2 and v[0]["went_object"] is not None and v[0]["went_object"] == v[1]["went_object"] for v in lay.values()])
    summary = {"alone": rate([r for r in res if r["test"] == "alone"]),
               "pair_right": rate(pairs),
               **{f"pair_right_{k}": rate([r for r in pairs if r["kind"] == k]) for k in ("same_colour", "same_shape", "unrelated")},
               "pair_same_object_whatever_asked": float(same_obj),
               "pair_went_left": float(np.mean([r["went_side"] == "left" for r in pairs if r["went_side"]])),
               "behind": rate([r for r in res if r["test"] == "behind"]),
               "behind_decoy_goal": rate([r for r in res if r["test"] == "behind+decoy"]),
               "behind_decoy_decoy": rate([r for r in res if r["test"] == "behind+decoy"], "decoy")}
    return {"summary": summary, "drives": res}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["standard", "which"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--bc", required=True, help="imitation checkpoint (notebook 04)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--detail", default=None, help="also write every drive's path/distances to this json")
    a = ap.parse_args()
    if a.detail:
        DETAIL = []
    t = time.time()
    rl = load_policy(a.ckpt, a.bc)
    out = standard_test(rl) if a.test == "standard" else which_test(rl)
    out["ckpt"], out["bc"] = a.ckpt, a.bc
    json.dump(out, open(a.out, "w"), indent=1)
    if a.detail:
        json.dump(DETAIL, open(a.detail, "w"))
    print("SUMMARY", json.dumps({k: round(v, 3) for k, v in out["summary"].items()}), f"({time.time() - t:.0f}s)", flush=True)
