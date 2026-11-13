"""A scripted "expert" driver that cheats: it knows the whole map and the shortest path.

It is called *privileged* because it uses information the real policy will never have (the map,
the exact goal position). We use it to:
  * prove every episode is solvable and that success detection works,
  * get an upper-bound baseline for evaluation (how good can a driver be?),
  * (optionally, later) generate extra demonstrations.
It only drives forward and turns (no strafing), like a careful human would.
"""

import math

import numpy as np

from .world import shortest_path


class PathFollowingExpert:
    def __init__(self, env, lookahead: float = 0.3, v_max: float = 0.6, k_turn: float = 3.0, finish: str = "standard"):
        """finish="standard": the original expert (top speed 0.6, brakes over the last 0.5 m of the path, stops ~0.30 m
        from the goal). finish="human": a human driver's speed profile - top speed 0.8 until ~0.5 m, then
        ~0.5, ~0.2, and stop ~0.12 m from the goal's surface (human demos: median stop 0.14 m)."""
        if finish == "human":
            v_max = 0.8
        self.env, self.lookahead, self.v_max, self.k_turn, self.finish = env, lookahead, v_max, k_turn, finish
        self._backup_steps = 0
        self._finishing = False

    def _human_finish(self, d_surface: float, goal_err: float) -> np.ndarray:
        """Close to the goal: face it, then drive in with the human speed profile and stop at STOP_AT."""
        STOP_AT = 0.12
        wz = float(np.clip(self.k_turn * goal_err, -1.5, 1.5))
        if d_surface <= STOP_AT:
            return np.zeros(3)
        # human medians: 0.8 above 0.5 m, 0.5 at 0.2-0.35 m, 0.2 below 0.2 m
        v = 0.8 if d_surface > 0.5 else 0.5 if d_surface > 0.25 else 0.2
        return np.array([v * max(0.0, math.cos(goal_err)) ** 3, 0.0, wz])

    def act(self, info: dict | None = None) -> np.ndarray:
        """Next (vx, vy, wz). Pass the latest ``info`` from env.step so the expert can react to bumps."""
        env, spec = self.env, self.env.spec
        if info is not None and info.get("collision") and self._backup_steps == 0:
            self._backup_steps = 5  # bumped into something: reverse for 0.5 s, then re-plan
        if self._backup_steps > 0:
            self._backup_steps -= 1
            return np.array([-0.25, 0.0, 0.0])
        x, y, yaw = env.car.pose()
        _s, _c, gx, gy = spec.goal
        d_surface = env.goal_surface_distance()
        field = spec.dist_field
        i, j = env.grid.cell(x, y)
        goal_err = (math.atan2(gy - y, gx - x) - yaw + math.pi) % (2 * math.pi) - math.pi
        if self.finish == "human" and (self._finishing or (np.isfinite(field[i, j]) and field[i, j] <= 0.2)):
            # within 0.2 m (along the path) of the goal region: straight in. Latched: closer to the object than the planner's
            # safety margin its map says "blocked", so the planner must not take over again.
            self._finishing = True
            return self._human_finish(d_surface, goal_err)
            return self._human_finish(d_surface, goal_err)
        if field[i, j] == 0.0:
            # Inside the goal region: by construction, simply facing the goal puts the bumper within the
            # success distance. So turn on the spot (safe: obstacles are >= car radius away), then stop.
            if abs(goal_err) > 0.08:
                return np.array([0.0, 0.0, float(np.clip(self.k_turn * goal_err, -1.0, 1.0))])
            if d_surface > 0.95 * env.cfg.success_dist:
                return np.array([0.1, 0.0, 0.0])  # creep the last few centimetres
            return np.zeros(3)
        if not np.isfinite(field[i, j]):  # hugging an obstacle: use the nearest reachable cell
            sl = field[max(0, i - 6): i + 7, max(0, j - 6): j + 7]
            if np.isfinite(sl).any():
                a, b = np.unravel_index(np.nanargmin(np.where(np.isfinite(sl), sl, np.nan)), sl.shape)
                x, y = env.grid.X[max(0, i - 6) + a, max(0, j - 6) + b], env.grid.Y[max(0, i - 6) + a, max(0, j - 6) + b]
        path = shortest_path(env.grid, field, x, y)
        # "Carrot" point: walk `lookahead` metres along the path (or its end, inside the goal region).
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1) if len(path) > 1 else np.zeros(0)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        tx, ty = path[min(np.searchsorted(cum, self.lookahead), len(path) - 1)]
        err = (math.atan2(ty - y, tx - x) - yaw + math.pi) % (2 * math.pi) - math.pi
        wz = float(np.clip(self.k_turn * err, -1.5, 1.5))
        slow_down = min(1.0, (cum[-1] + 0.1) / 0.5)  # brake when approaching the end of the path
        if self.finish == "human":
            slow_down = 1.0   # the human finish (above) does the braking, from 0.5 m
        vx = self.v_max * max(0.0, math.cos(err)) ** 3 * slow_down
        return np.array([vx, 0.0, wz])


class SearchThenGoExpert:
    """A scripted demonstrator whose behaviour a camera-only robot CAN copy (the training data, notebook 02):

      1. goal not in view -> rotate on the spot, always to the LEFT (a consistent, learnable habit; training does not
         mirror these drives, see notebook 03), even if a decoy is in view;
      2. goal in view once -> from then on, drive to it along the shortest path (PathFollowingExpert).
      3. a full turn without seeing it (hidden behind an obstacle from the start) -> drive along the route towards it
         until it comes into view, then as 2. (Like a person who has looked all around and walks somewhere else.)

    PathFollowingExpert alone turns straight towards a hidden goal: from the camera's point of view that is magic and
    cannot be learned. "In view" uses the simulator's exact visibility check (it is never fooled by a lookalike).
    """

    def __init__(self, env, search_wz: float = 1.0, finish: str = "standard"):
        self.env, self.search_wz, self.finish = env, search_wz, finish
        self.go = PathFollowingExpert(env, finish=finish)
        self.found = False
        self.turned = 0.0   # radians rotated while searching

    def reset(self):
        self.found, self.turned = False, 0.0
        self.go = PathFollowingExpert(self.env, finish=self.finish)

    def act(self, info: dict | None = None) -> np.ndarray:
        from .world import goal_in_view

        env = self.env
        if not self.found:
            _s, _c, gx, gy = env.spec.goal
            self.found = bool(goal_in_view(env.cfg, env.grid, env._occ_no_goal, env.car.pose(), (gx, gy)))
        if not self.found and self.turned < 2 * math.pi:
            self.turned += self.search_wz / env.sim_cfg.control_hz
            return np.array([0.0, 0.0, self.search_wz])
        return self.go.act(info)   # found it, or looked all around in vain: head along the route
