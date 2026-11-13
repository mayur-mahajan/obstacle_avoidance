"""Layout builders for "which object?" scenes, by difficulty level.

Used by the words layouts of the training data (carsim/words.py), by the words test (scripts/rl_tests.py) and,
optionally, by the RL arena (carsim/rl_env.py, a "levels" practice-layout setting). Practising easy levels first and
harder ones later is called curriculum learning (Narvekar et al. 2020; Portelas et al. 2020).

Levels (empty floor except L4):
  L1 pick1of2   goal + 1 decoy, both in view; the request names either one          20 s
  L2 pick1of4   goal + 2-3 decoys in view, at least one look-alike                   20 s
  L3 search     goal out of view, 1-2 decoys in view: "not that one, keep looking"   40 s
  L4 full       the standard random layouts with obstacles                           60 s
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .world import (COLORS, FLOOR_COLORS, SHAPES, WALL_COLORS, Grid, goal_in_view, geodesic_field, occupancy,
                    EpisodeSpec)

LEVELS = ["pick1of2", "pick1of4", "search", "full"]
LEVEL_SECONDS = {"pick1of2": 20.0, "pick1of4": 20.0, "search": 40.0, "full": 60.0}
HELD_OUT_GOAL = ("sphere", "orange")


def _pick_goal(rng):
    colours = list(COLORS)
    while True:
        g = (SHAPES[rng.integers(3)], colours[rng.integers(len(colours))])
        if g != HELD_OUT_GOAL:
            return g


def _decoy(rng, goal, kind, exclude=()):
    colours = list(COLORS)
    if kind == "same_colour":
        opts = [(s, goal[1]) for s in SHAPES if s != goal[0]]
    elif kind == "same_shape":
        opts = [(goal[0], c) for c in colours if c != goal[1]]
    else:
        opts = [(s, c) for s in SHAPES for c in colours if s != goal[0] and c != goal[1]]
    opts = [o for o in opts if o not in exclude]
    if not opts:  # every option of this kind is already used: any other object
        opts = [(s, c) for s in SHAPES for c in colours if (s, c) != goal and (s, c) not in exclude]
    return opts[rng.integers(len(opts))]


def _looks(rng):
    light = {"azimuth": float(rng.uniform(0, 2 * math.pi)), "elevation": float(rng.uniform(math.radians(35), math.radians(70))),
             "intensity": float(rng.uniform(3.5, 6.5))}
    return int(rng.integers(len(FLOOR_COLORS))), int(rng.integers(len(WALL_COLORS))), light


def make_level_spec(rng: np.random.Generator, level: str, cfg) -> EpisodeSpec:
    """One random practice/test layout of a curriculum level (L4 'full' = the standard sampler)."""
    if level == "full":
        from .rl_env import random_spec
        return random_spec(rng, cfg)
    grid = Grid(cfg)
    lim = cfg.arena_size / 2 - cfg.wall_thickness / 2 - 0.35
    for _ in range(200):
        g = _pick_goal(rng)
        kinds = ["same_colour", "same_shape", "unrelated"]
        if level == "pick1of2":
            d = _decoy(rng, g, kinds[rng.integers(3)])
            b = rng.uniform(8, 20)
            side = rng.choice([-1, 1])
            rel = [(g, side * b, rng.uniform(1.8, 3.0)), (d, -side * rng.uniform(8, 20), rng.uniform(1.8, 3.0))]
            # the request names either object (that is what makes the WORDS matter)
            if rng.random() < 0.5:
                rel = [rel[1], rel[0]]
        elif level == "pick1of4":
            n_dec = int(rng.integers(2, 4))
            decs = [_decoy(rng, g, rng.choice(["same_colour", "same_shape"]))]
            while len(decs) < n_dec:
                decs.append(_decoy(rng, g, kinds[rng.integers(3)], exclude=decs))
            objs = [g] + decs
            bearings = np.sort(rng.uniform(-24, 24, size=len(objs)))
            rel = [(o, float(bb), rng.uniform(1.8, 3.2)) for o, bb in zip([objs[i] for i in rng.permutation(len(objs))], bearings)]
            rel.sort(key=lambda r: r[0] != g)  # goal first (index 0)
        elif level == "search":
            n_dec = int(rng.integers(1, 3))
            decs = []
            while len(decs) < n_dec:
                decs.append(_decoy(rng, g, kinds[rng.integers(3)], exclude=decs))
            rel = [(g, 180 + rng.uniform(-110, 110), rng.uniform(1.8, 3.0))]
            rel += [(dd, rng.uniform(-20, 20), rng.uniform(1.8, 3.0)) for dd in decs]
        else:
            raise ValueError(level)
        goal_index = [r[0] for r in rel].index(g)
        x0, y0, yaw = rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-math.pi, math.pi)
        objects = [(o[0], o[1], float(x0 + dd * math.cos(yaw + math.radians(bb))), float(y0 + dd * math.sin(yaw + math.radians(bb))))
                   for o, bb, dd in rel]
        # inside the arena, not overlapping each other or the car
        if any(abs(x) > lim or abs(y) > lim for _s, _c, x, y in objects):
            continue
        if any(math.hypot(a[2] - b[2], a[3] - b[3]) < 0.55 for i, a in enumerate(objects) for b in objects[i + 1:]):
            continue
        occ = occupancy(grid, objects, [])
        gs = objects[goal_index]
        dist, _blocked = geodesic_field(grid, occ, gs[0], gs[2], gs[3])
        if not np.isfinite(grid.lookup(dist, x0, y0)):
            continue
        occ_ng = occupancy(grid, objects, [], skip_object=goal_index)
        visible = goal_in_view(cfg, grid, occ_ng, (x0, y0, yaw), (gs[2], gs[3]))
        if level in ("pick1of2", "pick1of4") and not visible:
            continue
        if level == "search":
            if visible:
                continue
            # at least one decoy must be in view (the temptation)
            if not any(goal_in_view(cfg, grid, occupancy(grid, objects, [], skip_object=k), (x0, y0, yaw), (o[2], o[3]))
                       for k, o in enumerate(objects) if k != goal_index):
                continue
        floor, wall, light = _looks(rng)
        return EpisodeSpec(seed=-1, goal_index=goal_index, objects=objects, obstacles=[], start=(float(x0), float(y0), float(yaw)),
                           prompt=f"go to the {gs[1]} {gs[0]}", floor_index=floor, wall_index=wall, light=light,
                           hard_distractors=level != "pick1of2", start_visible=bool(visible),
                           geodesic_start=float(grid.lookup(dist, x0, y0)), dist_field=dist)
    raise RuntimeError(f"could not build a {level} layout")


# -------------------------------------------------------------------------------------------------
@dataclass
class CurriculumConfig:
    levels: list = field(default_factory=lambda: list(LEVELS))
    promote_at: float = 0.70        # practice success on the current level (last `window` drives there) to move up
    window: int = 100
    current_share: float = 0.70     # share of practice drives from the current level; the rest = earlier levels
    eval_levels: list = field(default_factory=lambda: ["pick1of2", "pick1of4", "search"])
    eval_per_env: int = 4           # 8 simulators x 4 = 32 test drives per level


def level_mix(cc: CurriculumConfig, idx: int) -> dict:
    """Practice layout dict for the workers: {"levels": {name: probability}}."""
    if idx == 0:
        return {"levels": {cc.levels[0]: 1.0}}
    mix = {cc.levels[idx]: cc.current_share}
    for lv in cc.levels[:idx]:
        mix[lv] = (1 - cc.current_share) / idx
    return {"levels": mix}


def make_trainer_class():
    """Defined lazily so importing this module (in simulator workers) does not import torch/lerobot."""
    from .ppo import PPOTrainer

    class CurriculumTrainer(PPOTrainer):
        def __init__(self, cfg, cc: CurriculumConfig, *args, **kw):
            self.cc, self.level_idx = cc, 0
            cfg.practice_layout = level_mix(cc, 0)
            super().__init__(cfg, *args, **kw)

        @property
        def level(self):
            return self.cc.levels[self.level_idx]

        def level_success(self, level):
            eps = [e for e in self.episodes if e.get("level") == level][-self.cc.window:]
            return (float(np.mean([e["success"] for e in eps])) if eps else None), len(eps)

        def _set_level(self, idx):
            self.level_idx = idx
            self.cfg.practice_layout = level_mix(self.cc, idx)
            self.venv.set_layout(self.cfg.practice_layout)

        def after_round(self, entry):
            succ, n = self.level_success(self.level)
            entry["level"], entry["level_index"] = self.level, self.level_idx
            entry["level_success"], entry["level_drives"] = succ, n
            entry["success_by_level"] = {lv: self.level_success(lv)[0] for lv in self.cc.levels}
            entry["promoted"] = False
            if (succ is not None and n >= self.cc.window and succ >= self.cc.promote_at
                    and self.level_idx + 1 < len(self.cc.levels)):
                self._set_level(self.level_idx + 1)
                entry["promoted"] = True

        def extra_evals(self):
            for lv in self.cc.eval_levels:
                self.evaluate(episodes_per_env=self.cc.eval_per_env, layout={"levels": {lv: 1.0}}, tag=lv)

        def load(self, path):
            super().load(path)
            if self.log and "level_index" in self.log[-1]:
                idx = self.log[-1]["level_index"] + int(self.log[-1].get("promoted", False))
                self._set_level(idx)

    return CurriculumTrainer


def plot_curriculum(trainer, fig=None):
    """Live graphs: success per level (practice and test), the current level, and training health."""
    import matplotlib.pyplot as plt

    log, cc = trainer.log, trainer.cc
    fig = fig or plt.figure(figsize=(15, 8))
    fig.clf()
    ax = fig.subplots(2, 3).ravel()
    r = [e["round"] for e in log]
    colours = {"pick1of2": "C0", "pick1of4": "C1", "search": "C2", "full": "C3", "standard": "C3"}
    for lv in cc.levels:
        ys = [(e.get("success_by_level") or {}).get(lv) for e in log]
        ax[0].plot(r, [np.nan if y is None else y for y in ys], color=colours[lv], label=lv)
    ax[0].axhline(cc.promote_at, ls=":", color="k", lw=1)
    ax[0].set_title("Practice success per level (last 100 drives, with jitter)"); ax[0].set_ylim(0, 1); ax[0].legend(fontsize=8)
    for tag in cc.eval_levels + ["standard"]:
        ev = [e for e in trainer.evals if e.get("tag", "standard") == tag]
        if ev:
            ax[1].plot([e["round"] for e in ev], [e["success"] for e in ev], "o-", color=colours[tag], label=f"test: {tag}")
    ax[1].set_title("Test success (fixed drives, no jitter)"); ax[1].set_ylim(0, 1); ax[1].legend(fontsize=8)
    ax[2].step(r, [e.get("level_index", 0) + 1 for e in log], where="post")
    ax[2].set_yticks(range(1, len(cc.levels) + 1)); ax[2].set_yticklabels([f"L{i + 1} {lv}" for i, lv in enumerate(cc.levels)])
    ax[2].set_title("Current level")

    def line(a, key, label, **kw):
        a.plot(r, [e[key] if e.get(key) is not None else np.nan for e in log], label=label, **kw)

    line(ax[3], "explained_variance", "critic")
    ax[3].set_title("Critic accuracy (explained variance, 1 = perfect)"); ax[3].set_ylim(-1, 1)
    line(ax[4], "kl", "KL (how far the policy moved)"); line(ax[4], "clip_frac", "fraction clipped")
    line(ax[4], "study_done", "share of study done before KL stop", ls=":")
    ax[4].set_title("Size of each policy update"); ax[4].legend(fontsize=8)
    line(ax[5], "return_last100", "practice score (last 100, all levels)")
    ax[5].set_title("Score per drive"); ax[5].legend(fontsize=8)
    for a in ax:
        a.set_xlabel("round"); a.grid(alpha=0.3)
    fig.tight_layout()
    return fig
