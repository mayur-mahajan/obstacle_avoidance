"""Notebook 06 step 2a: PPO (notebook 05's recipe and reward) for a car that gets the overhead server's help in BURSTS.

Why bursts: later (step 2b) the car will ASK for help, and one ask gives a fresh reading at each of the next
``burst`` decisions (2 s). So during this practice, help already arrives the way asks will deliver it:

    per drive   25%  no help at all          (the car must still drive alone: "A Good Time to Ask", 2206.10606,
                25%  help all the time        found help-always training fails 95% of drives without help)
                50%  random bursts: at each decision without help, a burst starts with chance p (p drawn per drive,
                     0.05-0.3), lasting ``burst`` decisions

The help is SIMULATED: the true facts (from the simulator) with the real server's errors resampled
(carsim/guidance.py ServerLikeNoise; per drive 60% normal / 30% x1.5 / 10% useless). Running the real server on every
practice decision would multiply the training time; the tests use the real server.

The guidance adapter (numbers -> tokens) stays frozen; PPO trains what notebook 05 trains (action expert, critic,
jitter size). The critic sees the guidance tokens too, so it can learn how much help is worth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .guidance import GuidanceAdapter, GuidanceStore, ServerLikeNoise, install
from .ppo import PPOConfig, PPOTrainer


@dataclass
class GuidedPPOConfig(PPOConfig):
    regime_p: tuple = (0.25, 0.25, 0.5)        # per drive: never / always / bursts
    burst: int = 4                             # decisions per burst (4 x 0.5 s = 2 s)
    burst_p: tuple = (0.05, 0.3)               # per-decision chance a burst starts (drawn per drive)
    bc_drop_p: float = 0.2                     # demo samples: help removed (as in the step-1 imitation training)
    store: str = "runs/overhead_frames/route_follow"


def true_facts(pose, goal_xy):
    """The 9 facts from the simulator (same layout as scripts/overhead_frames.py)."""
    from . import overhead
    from .server import XY_SCALE
    f = overhead.facts(np.asarray(pose, np.float64), np.asarray(goal_xy, np.float64))
    return np.concatenate([f["car_xy"] / XY_SCALE, f["heading"], f["obj_xy"] / XY_SCALE, f["rel_dir"],
                           f["rel_dist"] / XY_SCALE], 1).astype(np.float32)


class GuidedPPOTrainer(PPOTrainer):
    venv_extra = {"poses": True}

    def __init__(self, cfg: GuidedPPOConfig, init_checkpoint, run_dir, prep_path, device=None):
        super().__init__(cfg, init_checkpoint, run_dir, prep_path, device)
        g = torch.load(Path(init_checkpoint).with_name(Path(init_checkpoint).stem + "_guidance.pt"), map_location="cpu",
                       weights_only=False)
        assert g["kind"] == "facts", "step 2a is for facts cars"
        self.adapter = GuidanceAdapter("facts").to(self.device)
        self.adapter.load_state_dict(g["state"]); self.adapter.eval()
        for p in self.adapter.parameters():
            p.requires_grad = False
        install(self.rl.policy, self.adapter)
        self.store = GuidanceStore(cfg.store, "facts", "noisy", seed=cfg.seed)
        self.noise = self.store.noise
        self.hrng = np.random.default_rng(cfg.seed + 11)
        N = cfg.n_envs
        self.regime, self.level, self.p = np.zeros(N, int), np.zeros(N, int), np.zeros(N)
        self.left = np.zeros(N, int)                         # decisions left in the current burst
        self.eval_present = None                             # tests: True = help always, False = never
        for i in range(N):
            self._new_drive(i)

    # ---- the help schedule ------------------------------------------------------------------------
    def _new_drive(self, i):
        c = self.cfg
        self.regime[i] = self.hrng.choice(3, p=c.regime_p)
        self.level[i] = self.hrng.choice(3, p=ServerLikeNoise.LEVELS)
        self.p[i] = self.hrng.uniform(*c.burst_p)
        self.left[i] = 0

    def help_step(self, peek=False, eval_mode=False):
        v = self.venv.last_views
        N = len(v["pose"])
        if eval_mode:
            present = np.full(N, bool(self.eval_present))
            level = np.zeros(N, int)
        else:
            present = np.zeros(N, bool)
            for i in range(N):
                if self.regime[i] == 1:
                    present[i] = True
                elif self.regime[i] == 2:
                    if self.left[i] == 0 and not peek and self.hrng.random() < self.p[i]:
                        self.left[i] = self.cfg.burst
                    present[i] = self.left[i] > 0
                    if not peek and self.left[i] > 0:
                        self.left[i] -= 1
            level = self.level
        x = self.noise.perturb(true_facts(v["pose"], v["goal_xy"]), level)
        return {"x": torch.from_numpy(x), "present": torch.from_numpy(present)}

    def help_after_step(self, out):
        for i, d in enumerate(out["done"]):
            if d:
                self._new_drive(i)

    def help_eval_start(self):
        self.noise.rng = np.random.default_rng(12345)      # the same simulated readings in every test

    # ---- feeding the help to the car --------------------------------------------------------------
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
        """Demo samples get noisy help (or none, 20%), as in step 1's imitation training."""
        x = self.store.get(b["frame"])
        present = torch.ones(len(x), dtype=torch.bool) if val else torch.from_numpy(self.hrng.random(len(x)) >= self.cfg.bc_drop_p)
        self.use_help({"x": x, "present": present})

    # ---- tests: with help (always, normal noise) and without --------------------------------------
    def evaluate(self, episodes_per_env=None, seed=None, layout=None, tag="standard"):
        self.eval_present = tag != "no_help"
        try:
            return super().evaluate(episodes_per_env, seed, layout, tag)
        finally:
            for i in range(self.cfg.n_envs):
                self._new_drive(i)

    def extra_evals(self):
        self.evaluate(tag="no_help")
