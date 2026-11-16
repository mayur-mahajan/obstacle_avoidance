"""Notebook 07: PPO practice (notebook 05's recipe and reward) for the REAL-ROUTE car of notebook 06, with the real
server giving it route help during practice, exactly as in the tests.

Where the help comes from during practice (the same pipeline as notebook 06's live tests):
    at the start of every drive   the simulator sends its overhead picture; the server reads it once:
                                  SmolVLM2 -> which object is the goal (rough position) and the picture tokens,
                                  the learned obstacle map, colour -> the goal's exact position, then the planner's
                                  distance field (shortest path from every spot to the goal)
    every half second             the simulator tracks the car's roof marker in a fresh overhead picture (cheap, on
                                  the CPU); the route points are read off the drive's distance field from that pose
Why once per drive: obstacles and objects don't move, so their map and the plan don't change during a drive. (The
notebook 06 tests rebuilt them every half second; the result is the same up to small map flicker.)

What learns: as in notebook 05, the action expert, the critic and the jitter size. The reader and the help layer stay
frozen. The "practise the demos" lesson uses the real-route demos with their real routes (20% without help, as in the
car's imitation training).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .guidance import GuidanceAdapter, GuidanceStore, install
from .ppo import PPOConfig, PPOTrainer

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RoutePPOConfig(PPOConfig):
    server: str = str(ROOT / "runs/checkpoints/server_native2_u4_heading_s0.pt")
    map: str = str(ROOT / "runs/checkpoints/server_map_long_s0.pt")
    route_file: str = "route_tracked_long.npy"      # the demos' real routes (for the "practise the demos" lesson)
    bc_drop_p: float = 0.2
    store: str = str(ROOT / "runs/overhead_frames/route_follow")


class RoutePPOTrainer(PPOTrainer):
    venv_extra = {"route_view": True}

    def __init__(self, cfg: RoutePPOConfig, init_checkpoint, run_dir, prep_path, device=None):
        super().__init__(cfg, init_checkpoint, run_dir, prep_path, device)
        from .server_map import MapHead
        from .server_native import ServerRuntime
        from .world import EnvConfig, Grid
        g = torch.load(Path(init_checkpoint).with_name(Path(init_checkpoint).stem + "_guidance.pt"), map_location="cpu",
                       weights_only=False)
        self.adapter = GuidanceAdapter("facts").to(self.device)
        self.adapter.load_state_dict(g["state"]); self.adapter.eval()
        for p in self.adapter.parameters():
            p.requires_grad = False
        install(self.rl.policy, self.adapter)
        self.server = ServerRuntime(cfg.server, self.device)
        self.map_head = MapHead().to(self.device)
        self.map_head.load_state_dict(torch.load(cfg.map, map_location="cpu")["state"]); self.map_head.eval()
        self.grid = Grid(EnvConfig())
        self.demo_store = GuidanceStore(cfg.store, "facts", "route_real", route_file=cfg.route_file)
        self.hrng = np.random.default_rng(cfg.seed + 11)
        N = cfg.n_envs
        self.field = [None] * N                  # each simulator's current plan (distance field) ...
        self.goal = [(0.0, 0.0)] * N             # ... and goal position
        self.pose = [(0.0, 0.0, 0.0)] * N        # last tracked pose (if the marker is hidden for a moment)
        self.server_s = 0.0                      # time spent in the server (for the progress print)

    # ---- the server, once per drive -----------------------------------------------------------------
    @torch.no_grad()
    def _plan(self, idx, pics, jitters, prompts, tracked):
        from .route import goal_shape_from_prompt, planned_field
        from .tracker import locate_goal
        import time
        t0 = time.time()
        s = self.server(pics, prompts)
        with torch.autocast("cuda", torch.bfloat16):
            maps = (self.map_head(s["feats"]).float().sigmoid() > 0.5).cpu().numpy()
        fp = s["facts_pred"][:, :9].cpu().numpy()
        for k, i in enumerate(idx):
            car = tuple(tracked[k, :2]) if tracked[k, 3] > 0 else None
            goal, _ok = locate_goal(pics[k], jitters[k], prompts[k], fp[k, 4:6] * 4, car)
            self.goal[i] = goal
            self.field[i] = planned_field(self.grid, maps[k], goal_shape_from_prompt(prompts[k]), goal[0], goal[1])
        self.server_s += time.time() - t0

    def help_step(self, peek=False, eval_mode=False):
        from .route import route_facts, server_route_facts
        v, prompts = self.venv.last_views, self.venv.last_prompts
        fresh = [i for i, f in enumerate(v["fresh"]) if f is not None]
        if fresh:
            self._plan(fresh, [v["fresh"][i][0] for i in fresh], [v["fresh"][i][1] for i in fresh],
                       [prompts[i] for i in fresh], v["tracked"][fresh])
            for i in fresh:
                v["fresh"][i] = None                # planned: don't redo it on the next call for the same view
        rows = []
        for i, t in enumerate(v["tracked"]):
            if t[3] > 0:
                self.pose[i] = (float(t[0]), float(t[1]), float(t[2]))
            f = self.field[i]
            if f is not None and np.isfinite(f).any():
                rows.append(route_facts(self.grid, f, self.pose[i], self.goal[i]))
            else:                                    # no path on the map: straight at the goal (as in the tests)
                rows.append(server_route_facts(self.grid, np.zeros((80, 80), bool), prompts[i], self.goal[i], self.pose[i]))
        x = torch.from_numpy(np.stack(rows))
        return {"x": x, "present": torch.ones(len(rows), dtype=torch.bool)}

    # ---- feeding the help to the car (as carsim/guided_ppo.py) --------------------------------------
    @torch.no_grad()
    def use_help(self, h, idx=None):
        if h is None:
            self.rl.m._guidance = None
            return
        x, pr = h["x"], h["present"]
        if idx is not None:
            idx = torch.as_tensor(idx)
            x, pr = x[idx], pr[idx]
        self.rl.m._guidance = self.adapter(x.to(self.device), pr.to(self.device))

    def stack_help(self, hs):
        return {"x": torch.cat([h["x"] for h in hs]), "present": torch.cat([h["present"] for h in hs])}

    def use_bc_help(self, b, val: bool = False):
        x = self.demo_store.get(b["frame"])
        present = torch.ones(len(x), dtype=torch.bool) if val else torch.from_numpy(self.hrng.random(len(x)) >= self.cfg.bc_drop_p)
        self.use_help({"x": x, "present": present})
