"""The RL practice arena: the reward, and many simulators running in parallel CPU processes.

Reward, given after every 0.1 s control step:

    progress   +1 per metre the shortest path to the goal got shorter (-1 per metre longer)
    success    +2.5 when the car arrives and stops at the RIGHT object (the episode ends)
    time       -0.01 per step
    crash      -0.1 per step spent touching anything except the goal
    sight      +0.2 when the goal comes into view, -0.3 when it leaves the picture

"progress" is a difference of a quantity (the distance), so over an episode it adds up to (start - end): driving
back and forth cannot farm points ("potential-based shaping", Ng et al. 1999). "sight" is almost one too, but losing
the goal costs a little more than finding it gives back (-0.3 vs +0.2): looking away and back costs 0.1, so it can't
be farmed either, and drives that keep the goal in view are preferred on purpose.
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
    sight: float = 0.2            # when the goal comes into view
    sight_lost: float = -0.3      # when it leaves the picture (one-time; -0.3 + 0.2 = 0.1 net cost per loss)


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
                 "sight": c.sight * max(vis - self.vis, 0.0),             # goal came into view
                 "sight_lost": c.sight_lost * max(self.vis - vis, 0.0)}   # goal left the picture
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
def _worker(conn, seed: int, reward_cfg: dict, gamma: float, paraphrase: bool, layout: dict, detail: bool = False,
            overhead_view: bool = False, poses: bool = False, route_view: bool = False):
    from .data import Paraphraser
    from .env import NavEnv

    from .curriculum import LEVEL_SECONDS, make_level_spec
    from .route import route_facts
    from .tracker import track
    from .world import goal_in_view

    if overhead_view or route_view:   # notebook 06: two-colour roof marker + the overhead camera (front camera unaffected)
        from . import overhead
        env = NavEnv(backend="cpu", car_cfg=overhead.car_config(two_colour=True), overview_camera_res=(1024, 1024))
        jrng = np.random.default_rng(seed + 777)   # own stream: the layouts are the same as without the overhead view
    else:
        env = NavEnv(backend="cpu")

    def with_view(o):
        """Add the overhead picture, the car pose and the goal position to an observation (notebook 06 only)."""
        if overhead_view:
            j = overhead.sample_jitter(jrng)   # recorded: a calibrated ceiling camera knows its own mounting
            o = dict(o, overhead=overhead.render(env, j), jitter=j.astype(np.float32))
        if route_view:   # notebook 07: track the car in every overhead picture; send the picture itself only at a drive's start
            j = overhead.sample_jitter(jrng)
            pic = overhead.render(env, j)
            t = track(pic, j)
            o = dict(o, tracked=np.array([t["x"], t["y"], t["yaw"], float(t["ok"])], np.float32))
            if state.get("fresh"):
                o.update(fresh_pic=pic, fresh_jitter=j.astype(np.float32))
                state["fresh"] = False
        if overhead_view or poses:   # poses only: for simulated server help (no picture rendered)
            o = dict(o, pose=np.asarray(env.car.pose(), np.float32), goal_xy=np.asarray(env.spec.goal[2:4], np.float32),
                     goal_visible=bool(goal_in_view(env.cfg, env.grid, env._occ_no_goal, env.car.pose(), env.spec.goal[2:4])),
                     route=route_facts(env.grid, env.spec.dist_field, env.car.pose(), env.spec.goal[2:4]))
        return o
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
        state["fresh"] = True
        obs, info = env.reset(spec=spec)
        obs = with_view(obs)
        shape, colour = spec.goal[:2]
        state.update(tracker=RewardTracker(env, info, rcfg), ret=0.0, path=[], coll=[], seen=bool(info["goal_visible"]),
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
                if detail:
                    state["path"].append([round(float(v), 3) for v in info["pose"]])
                    if info["collision"]:
                        state.setdefault("coll", []).append(len(state["path"]) - 1)
                if info["goal_visible"]:
                    state["seen"] = True
                disc_reward += (gamma ** n) * r
                state["ret"] += r
                n += 1
                if term or trunc:
                    done, truncated = True, bool(trunc and not term)
                    ep = {"success": bool(info["success"]), "return": state["ret"], "seconds": info["t"] / 10,
                          "collision_steps": info["n_collision_steps"], "goal": env.spec.goal_name,
                          "start_visible": env.spec.start_visible, "lookalike": env.spec.hard_distractors,
                          "level": state["level"], "parts": dict(state["tracker"].totals),
                          "goal_visible_end": bool(info["goal_visible"]), "goal_seen": bool(state.get("seen", False)),
                          "seed": int(env.spec.seed)}
                    if detail:   # everything needed to draw the drive on the map (carsim.evaluate.plot_room_drives)
                        ep.update(path=state["path"], prompt=state["prompt"], collisions=state.get("coll", []))
                    break
            obs = with_view(obs) if not done else obs
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
                 paraphrase: bool = True, layout: dict | None = None, detail: bool = False, overhead_view: bool = False,
                 poses: bool = False, route_view: bool = False):
        """layout: changes to the layout dials (EnvConfig fields) for practice drives, e.g.
        {"p_goal_visible": 0.25, "p_hard_distractor": 0.5}. None/{} = the standard layouts.
        detail: also return each finished drive's path (x, y, yaw every 0.1 s) and prompt (for the test maps)."""
        ctx = mp.get_context("spawn")  # a fresh interpreter per worker: Genesis must not be forked
        self.conns, self.procs = [], []
        for i in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, seed * 1000 + i, asdict(reward_cfg or RewardConfig()), gamma, paraphrase, dict(layout or {}), detail,
                                                  overhead_view, poses, route_view),
                            daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        self.n = n

    @staticmethod
    def _stack(obs_list):
        return (np.stack([o["image"] for o in obs_list]), np.stack([o["state"] for o in obs_list]).astype(np.float32))

    @staticmethod
    def views(obs_list):
        """Notebook 06 (overhead_view=True): overhead pictures (N,1024,1024,3), car poses (N,3), goal positions (N,2)."""
        out = {k: np.stack([o[k] for o in obs_list]) for k in ("overhead", "jitter", "pose", "goal_xy", "goal_visible", "route", "tracked")
               if k in obs_list[0]}
        if "tracked" in obs_list[0]:   # notebook 07: the overhead picture of simulators that just started a drive (else None)
            out["fresh"] = [(o["fresh_pic"], o["fresh_jitter"]) if "fresh_pic" in o else None for o in obs_list]
        return out

    def reset(self):
        for c in self.conns:
            c.send(("reset", None))
        res = [c.recv() for c in self.conns]
        imgs, states = self._stack([r[0] for r in res])
        self.last_views = self.views([r[0] for r in res])
        self.last_prompts = [r[1] for r in res]
        return imgs, states, [r[1] for r in res]

    def step(self, chunks):
        for c, ch in zip(self.conns, chunks):
            c.send(("step", np.asarray(ch, np.float32)))
        res = [c.recv() for c in self.conns]
        imgs, states = self._stack([r[0] for r in res])
        f_imgs, f_states = self._stack([r[6] for r in res])
        self.last_views = self.views([r[0] for r in res])
        self.last_prompts = [r[1] for r in res]
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
