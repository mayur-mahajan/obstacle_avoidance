"""The RL practice arena: the reward, and many simulators running in parallel CPU processes.

Reward, given after every 0.1 s control step:

    progress   +1 per metre the shortest path to the goal got shorter (-1 per metre longer)
    success    +2.5 when the car arrives and stops at the RIGHT object (the episode ends)
    time       -0.01 per step
    crash      -0.1 per step spent touching anything except the goal
    sight      +0.2 when the goal comes into view, -0.2 when it leaves

"progress" and "sight" are both differences of a quantity (distance, visible-or-not), so over an
episode they add up to (start - end): driving back and forth or looking away and back cannot farm
points ("potential-based shaping", Ng et al. 1999).
The shortest-path distance comes from the simulator's map. Only the scorer uses it, during training;
the policy never sees it.

Episodes end only on success or at the 60 s limit (a "stuck -> end early" rule would let the robot
end a bad episode on purpose to stop the time penalty).

Why processes and the CPU: Genesis allows one physics backend per process, and for this small scene
the CPU backend is ~2x faster than the GPU and scales across cores (measured on a 20-core CPU: ~730
steps/s with 6 processes vs ~67 for any number of GPU copies). Each worker process owns one NavEnv.
"""

from __future__ import annotations

import math
import multiprocessing as mp
from dataclasses import dataclass, asdict, replace

import numpy as np
from scipy import ndimage

HELD_OUT_GOAL = ("sphere", "orange")  # never a goal in training: compositional generalisation test


@dataclass
class RewardConfig:
    progress: float = 1.0
    success: float = 2.5
    time: float = -0.01
    crash: float = -0.1
    sight: float = 0.2


class RewardTracker:
    """Scores one episode, step by step, from the env's ``info`` dicts."""

    def __init__(self, env, info0: dict, cfg: RewardConfig | None = None):
        self.cfg = cfg or RewardConfig()
        # The map's distance field is +inf where the car's CENTRE is closer to an obstacle than the planner's
        # safety margin (the car can still physically be there, e.g. scraping a wall). Fill those cells with
        # "distance of the nearest reachable cell + the way there" so progress is defined everywhere.
        f = env.spec.dist_field
        finite = np.isfinite(f)
        if finite.all() or not finite.any():
            self.field = f
        else:
            gap, (ii, jj) = ndimage.distance_transform_edt(~finite, return_indices=True)
            self.field = np.where(finite, f, f[ii, jj] + gap * env.grid.res)
        self.grid = env.grid
        self.d = self._dist(info0["pose"])
        self.vis = float(info0["goal_visible"])
        self.totals = {k: 0.0 for k in asdict(self.cfg)}

    def _dist(self, pose) -> float:
        return float(self.grid.lookup(self.field, pose[0], pose[1]))

    def __call__(self, info: dict) -> float:
        c = self.cfg
        d, vis = self._dist(info["pose"]), float(info["goal_visible"])
        if not math.isfinite(d):  # goal unreachable on the map (should not happen): no progress signal
            d = self.d
        parts = {"progress": c.progress * (self.d - d),
                 "success": c.success * float(info["success"]),
                 "time": c.time,
                 "crash": c.crash * float(info["collision"]),
                 "sight": c.sight * (vis - self.vis)}
        self.d, self.vis = d, vis
        for k, v in parts.items():
            self.totals[k] += v
        return sum(parts.values())


def random_spec(rng: np.random.Generator, env_cfg=None):
    """A random training layout whose goal is not the held-out orange sphere."""
    from .world import sample_episode

    while True:
        spec = sample_episode(int(rng.integers(2**31 - 1)), env_cfg)
        if tuple(spec.goal[:2]) != HELD_OUT_GOAL:
            return spec


# -------------------------------------------------------------------------------------------------
# One simulator per process
# -------------------------------------------------------------------------------------------------
def _worker(conn, seed: int, reward_cfg: dict, gamma: float, paraphrase: bool, layout: dict):
    from .data import Paraphraser
    from .env import NavEnv

    from .curriculum import LEVEL_SECONDS, make_level_spec

    env = NavEnv(backend="cpu")
    base_cfg = env.cfg
    # layout: changes to the layout dials for these drives, e.g. {"p_goal_visible": 0.25}; {} = standard layouts.
    # A "levels" entry ({level name: probability}) switches to curriculum layouts (carsim/curriculum.py).
    def dials(lay):
        return replace(base_cfg, **{k: v for k, v in lay.items() if k != "levels"})
    layout_cfg = dials(layout)
    rng = np.random.default_rng(seed)
    para = Paraphraser("train", seed=seed)
    rcfg = RewardConfig(**reward_cfg)
    state = {}

    def new_episode():
        levels = layout.get("levels")
        if levels:
            names = list(levels)
            p = np.array([levels[k] for k in names], float)
            level = names[int(rng.choice(len(names), p=p / p.sum()))]
            spec = make_level_spec(rng, level, layout_cfg)
            env.cfg = replace(base_cfg, max_episode_s=LEVEL_SECONDS[level])  # shorter drives on easy levels
        else:
            level = "standard"
            spec = random_spec(rng, layout_cfg)
            env.cfg = base_cfg
        state["level"] = level
        obs, info = env.reset(spec=spec)
        shape, colour = spec.goal[:2]
        state.update(tracker=RewardTracker(env, info, rcfg), ret=0.0,
                     prompt=para(colour, shape) if paraphrase else spec.prompt)
        return obs

    obs = None
    while True:
        cmd, arg = conn.recv()
        if cmd == "reset":
            obs = new_episode()
            conn.send((obs, state["prompt"]))
        elif cmd == "step":  # arg: (k, 3) raw actions; executes them unless the episode ends first
            disc_reward, n, done, truncated, ep = 0.0, 0, False, False, None
            for a in arg:
                obs, term, trunc, info = env.step(a)
                r = state["tracker"](info)
                disc_reward += (gamma ** n) * r
                state["ret"] += r
                n += 1
                if term or trunc:
                    done, truncated = True, bool(trunc and not term)
                    ep = {"success": bool(info["success"]), "return": state["ret"], "seconds": info["t"] / 10,
                          "collision_steps": info["n_collision_steps"], "goal": env.spec.goal_name,
                          "start_visible": env.spec.start_visible, "lookalike": env.spec.hard_distractors,
                          "level": state["level"], "parts": dict(state["tracker"].totals)}
                    break
            final_obs = obs
            if done:
                obs = new_episode()  # auto-reset: the next decision starts a fresh episode
            conn.send((obs, state["prompt"], disc_reward, n, done, truncated, final_obs, ep))
        elif cmd == "reseed":  # restart the random layout sequence - arg: (seed, layout dials)
            seed_, layout = arg
            rng = np.random.default_rng(seed_)
            para = Paraphraser("train", seed=seed_)
            layout_cfg = dials(layout)
            conn.send(True)
        elif cmd == "set_layout":  # change the layout mix for FUTURE episodes (curriculum level-up); no reseed
            layout = arg
            layout_cfg = dials(layout)
            conn.send(True)
        elif cmd == "close":
            conn.close()
            return


class ParallelNav:
    """N simulators in N processes, stepped together. One "decision" = up to ``k`` control steps.

    reset()            -> images (N,H,W,3) uint8, states (N,3) raw speeds, prompts [N]
    step(chunks (N,k,3) raw actions) -> dict with the next observations and, per env:
        reward  discounted sum of the step rewards inside the decision (gamma per 0.1 s step)
        n       how many steps were executed (k, or fewer when the episode ended)
        done / truncated  (truncated = hit the 60 s limit; PPO then bootstraps from final_obs)
        final_obs  observation at the end of the decision (before any auto-reset)
        episode  stats dict for episodes that just ended, else None
    """

    def __init__(self, n: int, seed: int = 0, reward_cfg: RewardConfig | None = None, gamma: float = 0.99,
                 paraphrase: bool = True, layout: dict | None = None):
        """layout: changes to the layout dials (EnvConfig fields) for practice drives, e.g.
        {"p_goal_visible": 0.25, "p_hard_distractor": 0.8}. None/{} = the standard layouts."""
        ctx = mp.get_context("spawn")  # a fresh interpreter per worker: Genesis must not be forked
        self.conns, self.procs = [], []
        for i in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, seed * 1000 + i, asdict(reward_cfg or RewardConfig()), gamma, paraphrase, dict(layout or {})),
                            daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        self.n = n

    @staticmethod
    def _stack(obs_list):
        return (np.stack([o["image"] for o in obs_list]), np.stack([o["state"] for o in obs_list]).astype(np.float32))

    def reset(self):
        for c in self.conns:
            c.send(("reset", None))
        res = [c.recv() for c in self.conns]
        imgs, states = self._stack([r[0] for r in res])
        return imgs, states, [r[1] for r in res]

    def step(self, chunks):
        for c, ch in zip(self.conns, chunks):
            c.send(("step", np.asarray(ch, np.float32)))
        res = [c.recv() for c in self.conns]
        imgs, states = self._stack([r[0] for r in res])
        f_imgs, f_states = self._stack([r[6] for r in res])
        return {"images": imgs, "states": states, "prompts": [r[1] for r in res],
                "reward": np.array([r[2] for r in res], np.float32), "n": np.array([r[3] for r in res]),
                "done": np.array([r[4] for r in res]), "truncated": np.array([r[5] for r in res]),
                "final_images": f_imgs, "final_states": f_states, "episodes": [r[7] for r in res]}

    def reseed(self, seed: int, layout: dict | None = None):
        """Make every worker's sequence of layouts start again from ``seed`` (worker i uses seed*1000+i),
        using the layout dials ``layout`` (None/{} = standard layouts). Call ``reset()`` afterwards.
        Used for a fixed, repeatable test set (always standard layouts)."""
        for i, c in enumerate(self.conns):
            c.send(("reseed", (seed * 1000 + i, dict(layout or {}))))
        for c in self.conns:
            c.recv()

    def set_layout(self, layout: dict):
        """Change the layout mix for the episodes that start from now on (no reseed, no reset)."""
        for c in self.conns:
            c.send(("set_layout", dict(layout)))
        for c in self.conns:
            c.recv()

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
