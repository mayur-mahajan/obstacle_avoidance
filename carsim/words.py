"""Layouts where the WORDS decide which object to drive to (half of the scripted training drives, notebook 02).

Why: in ordinary random rooms the goal is the only, or the most obvious, object in view in ~95% of frames, so
imitation learns "drive to the obvious object" and ignores the prompt. These layouts put 2-4 objects in view
(or a decoy in front of a hidden goal), so only reading the prompt gives the right drive.

Scene types (chosen per layout; shares of the "standard" mix, see MIXES):
  pair       2 objects side by side; recorded TWICE - once asking for each (same picture, different words)  40%
  pick1of4   3-4 objects in view, decoys share a colour or shape with the goal                               20%
  search     goal behind the car, 1-2 decoys in front                                                        20%
  room       an ordinary room with obstacles, goal in view AND at least one other goal-type object in view  20%
The first three reuse the layout builder in carsim.curriculum (make_level_spec).

Seeds: every layout must be rebuildable from the number stored in the dataset (sim.episode_seed), like any layout.
Words seeds are WORDS_SEED_BASE + 2 * layout_id + ask (ask = 0 or 1 = which object of a pair is the goal);
carsim.world.sample_episode recognises them and calls words_spec, so all existing tools keep working.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from .world import (PROMPT_TEMPLATES, SHAPE_WORDS, EnvConfig, Grid, geodesic_field, goal_in_view, occupancy,
                    sample_episode)

WORDS_SEED_BASE = 10**12
# Scene mixes. The mix is part of the seed (see words_seed), so a recorded seed always rebuilds the same scene even
# after new mixes are added. NEVER change an existing entry - only append.
MIXES = {
    "standard": {"pair": 0.4, "pick1of4": 0.2, "search": 0.2, "room": 0.2},
    "search":   {"pair": 0.2, "pick1of4": 0.15, "search": 0.5, "room": 0.15},  # favours goal-behind-a-decoy scenes (the training data)
}
WORDS_MIX = MIXES["standard"]
_MIX_BITS = 41   # layout_id = mix number * 2**41 + random part
_LEVEL = {"pair": "pick1of2", "pick1of4": "pick1of4", "search": "search"}


def is_words_seed(seed: int) -> bool:
    return int(seed) >= WORDS_SEED_BASE


def words_seed(layout_id: int, ask: int = 0, mix: str = "standard") -> int:
    """layout_id: a random number below 2**40."""
    return WORDS_SEED_BASE + 2 * (list(MIXES).index(mix) * 2**_MIX_BITS + int(layout_id)) + int(ask)


def _split(seed: int):
    layout_id, ask = divmod(int(seed) - WORDS_SEED_BASE, 2)
    return layout_id, ask


def words_mix(seed: int) -> str:
    return list(MIXES)[_split(seed)[0] >> _MIX_BITS]


def words_kind(seed: int) -> str:
    layout_id, _ = _split(seed)
    mix = MIXES[words_mix(seed)]
    rng = np.random.default_rng([layout_id, 31])
    kinds = list(mix)
    return kinds[rng.choice(len(kinds), p=list(mix.values()))]


def _visible(cfg, grid, spec, k, pose):
    o = spec.objects[k]
    return bool(goal_in_view(cfg, grid, occupancy(grid, spec.objects, spec.obstacles, skip_object=k), pose, (o[2], o[3])))


def _with_goal(spec, k, cfg, grid):
    """The same layout with object k as the goal (its own shortest-path map, visibility)."""
    o = spec.objects[k]
    dist, _ = geodesic_field(grid, occupancy(grid, spec.objects, spec.obstacles), o[0], o[2], o[3])
    return replace(spec, goal_index=k, dist_field=dist, start_visible=_visible(cfg, grid, spec, k, spec.start),
                   geodesic_start=float(grid.lookup(dist, spec.start[0], spec.start[1])))


def words_spec(seed: int, cfg: EnvConfig | None = None):
    """The layout for a words seed (deterministic)."""
    from .curriculum import make_level_spec

    cfg = cfg or EnvConfig()
    grid = Grid(cfg)
    layout_id, ask = _split(seed)
    kind = words_kind(seed)
    rng = np.random.default_rng([layout_id, 97])
    if kind == "room":
        room_cfg = replace(cfg, p_goal_visible=1.0, p_hard_distractor=1.0)
        for _ in range(100):
            spec = sample_episode(int(rng.integers(2**31 - 1)), room_cfg)
            if spec.start_visible and any(_visible(cfg, grid, spec, k, spec.start)
                                          for k in range(len(spec.objects)) if k != spec.goal_index):
                break
        else:
            raise RuntimeError("could not build a 'room' layout")
    else:
        from .curriculum import HELD_OUT_GOAL
        for _ in range(100):
            spec = make_level_spec(rng, _LEVEL[kind], cfg)
            # a pair is asked both ways, so neither object may be the held-out test goal ("orange sphere")
            if kind != "pair" or all(o[:2] != HELD_OUT_GOAL for o in spec.objects):
                break
    if kind == "pair" and ask == 1:
        spec = _with_goal(spec, 1 - spec.goal_index, cfg, grid)
    shape, colour = spec.goal[:2]
    words = SHAPE_WORDS[shape]
    prompt = PROMPT_TEMPLATES[rng.integers(len(PROMPT_TEMPLATES))].format(c=colour, s=words[rng.integers(len(words))])
    return replace(spec, seed=int(seed), prompt=prompt)
