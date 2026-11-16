"""Route help (notebook 06): instead of "the goal is over there", tell the car "drive to these points next" - points
along the shortest obstacle-free path to the goal, re-planned from wherever the car is now.

This is the "perfect route" version: the path comes from the simulator's own map (the same distance field the demo
driver, carsim/expert.py PathFollowingExpert, follows with a 0.3 m "carrot"). It is the upper bound for a real server
that would find the obstacles in the overhead picture and plan on that.

The 9 numbers (same size as the facts, so the same guidance adapter type is used):
    3 points on the path, 0.3 / 0.6 / 1.0 m ahead along it, each as (forward, left) metres in the CAR's frame
    the remaining path length (/4 m)
    the goal's direction from the car (cos, sin)  - for the last stretch, where the path ends at the goal region
"""

from __future__ import annotations

import math

import numpy as np

from .world import shortest_path

AHEAD = (0.3, 0.6, 1.0)          # metres along the path
XY_SCALE = 4.0


def route_facts(grid, field, pose, goal_xy) -> np.ndarray:
    x, y, yaw = (float(v) for v in pose)
    gx, gy = (float(v) for v in goal_xy)
    i, j = grid.cell(x, y)
    sx, sy = x, y
    if not np.isfinite(field[i, j]):          # hugging an obstacle: start from the nearest reachable cell (as the expert)
        sl = field[max(0, i - 6): i + 7, max(0, j - 6): j + 7]
        if np.isfinite(sl).any():
            a, b = np.unravel_index(np.nanargmin(np.where(np.isfinite(sl), sl, np.nan)), sl.shape)
            sx, sy = grid.X[max(0, i - 6) + a, max(0, j - 6) + b], grid.Y[max(0, i - 6) + a, max(0, j - 6) + b]
    path = shortest_path(grid, field, sx, sy)
    if len(path) < 2:                          # inside the goal region (or no path): every point = the goal
        path = np.array([[sx, sy], [gx, gy]])
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for d in AHEAD:
        px, py = path[min(np.searchsorted(cum, d), len(path) - 1)]
        dx, dy = px - x, py - y
        out += [c * dx + s * dy, -s * dx + c * dy]          # world -> car frame (forward, left)
    remain = grid.lookup(field, sx, sy)
    remain = cum[-1] if not np.isfinite(remain) else remain
    b = math.atan2(gy - y, gx - x) - yaw
    out += [remain / XY_SCALE, math.cos(b), math.sin(b)]
    return np.asarray(out, np.float32)


# ------------------------------------------------------------------------------------------------------
# The REAL server's route: its own obstacle map (carsim/server_map.py) + its own goal position (the facts head),
# then the same planner as the simulator's map (inflate obstacles by the car's size, shortest path).
# ------------------------------------------------------------------------------------------------------
def goal_shape_from_prompt(prompt: str) -> str:
    """The server knows the instruction: which shape is the goal."""
    from .world import SHAPE_WORDS
    words = prompt.lower().replace(".", " ").split()
    for shape, names in SHAPE_WORDS.items():
        if any(n in words for n in names):
            return shape
    return "cube"


def planned_field(grid, blocked80: np.ndarray, goal_shape: str, gx: float, gy: float) -> np.ndarray:
    """Shortest-path distance field on the server's map. blocked80: (80, 80) bool at 10 cm. The goal object itself is
    usually on the map too (as in the simulator's map). If the server's goal position puts the goal region inside
    blocked cells (position error), the reach around the goal is widened step by step (up to +0.6 m)."""
    from dataclasses import replace
    from .world import Grid, geodesic_field
    occ = np.repeat(np.repeat(np.asarray(blocked80, bool), 2, 0), 2, 1) | grid.walls()
    for extra in (0.0, 0.2, 0.4, 0.6):
        g = grid if extra == 0 else Grid(replace(grid.cfg, success_dist=grid.cfg.success_dist + extra))
        field, _ = geodesic_field(g, occ, goal_shape, gx, gy)
        if np.isfinite(field).any():
            return field
    return field


def server_route_facts(grid, blocked80, prompt, goal_xy, pose) -> np.ndarray:
    """The 9 route numbers, planned on the server's map from the server's goal position (the car's pose is NOT taken
    from the simulator either: pass the server's estimate)."""
    field = planned_field(grid, blocked80, goal_shape_from_prompt(prompt), float(goal_xy[0]), float(goal_xy[1]))
    if not np.isfinite(field).any():                         # no path at all on the server's map: straight at the goal
        x, y, yaw = (float(v) for v in pose)
        b = math.atan2(goal_xy[1] - y, goal_xy[0] - x) - yaw
        out = []
        for d in AHEAD:
            out += [d * math.cos(b), d * math.sin(b)]
        return np.asarray(out + [math.hypot(goal_xy[0] - x, goal_xy[1] - y) / XY_SCALE, math.cos(b), math.sin(b)], np.float32)
    return route_facts(grid, field, pose, goal_xy)
