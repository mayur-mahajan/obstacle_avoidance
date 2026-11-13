"""Episode design: *what* is in the arena and *where*, decided with plain numpy (no simulator).

Separating "sample an episode" (this file) from "show it in Genesis" (env.py) has two benefits:
  * We can generate and check thousands of layouts in seconds, without rendering anything.
  * Every episode is a small, reproducible recipe (an ``EpisodeSpec``) that can be stored in the
    dataset next to the video, so any episode can be rebuilt exactly later.

Main ideas implemented here:
  * Goal objects: 3 shapes x 6 colours. Some distractors deliberately share a word with the goal
    ("red cube" next to "red sphere"), so the model must understand BOTH words of the prompt.
  * Obstacles: neutral colours only, so a colour word always refers to a goal object.
  * Reachability: an occupancy grid + shortest-path search guarantees every episode is solvable,
    and gives the *geodesic* (walking) distance to the goal from every point in the arena.
  * Visibility: half of the episodes start with the goal in view, half without (the car must search).
"""

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# ------------------------------------------------------------------------------------------------
# Vocabulary: goal shapes, colours and the words used for them in prompts
# ------------------------------------------------------------------------------------------------
SHAPES = ("sphere", "cube", "cylinder")
COLORS = {
    "red": (0.85, 0.10, 0.10),
    "green": (0.10, 0.65, 0.15),
    "blue": (0.10, 0.30, 0.90),
    "yellow": (0.95, 0.85, 0.10),
    "purple": (0.55, 0.15, 0.75),
    "orange": (1.00, 0.50, 0.00),
}
SHAPE_WORDS = {"sphere": ["sphere", "ball"], "cube": ["cube", "block"], "cylinder": ["cylinder"]}
PROMPT_TEMPLATES = [
    "drive to the {c} {s}",
    "go to the {c} {s}",
    "find the {c} {s}",
    "navigate to the {c} {s}",
    "head to the {c} {s}",
    "move to the {c} {s}",
    "go find the {c} {s}",
]
# Size of each goal shape (metres). All goals are ~30 cm tall, about twice the camera height.
GOAL_RADIUS = {"sphere": 0.15, "cube": 0.15, "cylinder": 0.15}  # cube: half edge length
GOAL_HEIGHT = {"sphere": 0.30, "cube": 0.30, "cylinder": 0.35}

# ------------------------------------------------------------------------------------------------
# Obstacle pool: neutral (grey/white/stone) colours only, never a goal colour
# ------------------------------------------------------------------------------------------------
_NEUTRALS = [(0.55, 0.55, 0.55), (0.80, 0.80, 0.78), (0.30, 0.30, 0.32), (0.70, 0.68, 0.62), (0.92, 0.92, 0.92), (0.45, 0.44, 0.42)]
OBSTACLE_POOL = (
    # kind, size: box=(sx, sy, sz) / cylinder=(radius, height)
    [("box", s) for s in [(0.4, 0.4, 0.5), (0.6, 0.35, 0.45), (0.8, 0.4, 0.5), (0.5, 0.5, 0.35), (0.35, 0.35, 0.6), (1.0, 0.25, 0.4)]]
    + [("cylinder", s) for s in [(0.15, 0.7), (0.2, 0.6), (0.25, 0.5), (0.18, 0.8), (0.12, 0.6)]]
    + [("box", s) for s in [(1.5, 0.1, 0.5), (2.0, 0.1, 0.5), (1.2, 0.1, 0.45), (1.8, 0.12, 0.5), (1.0, 0.1, 0.5)]]  # wall segments
)
OBSTACLE_COLORS = [_NEUTRALS[i % len(_NEUTRALS)] for i in range(len(OBSTACLE_POOL))]

# Visual variety: floor 0 = the default checker plane; the others are solid-colour floor covers.
FLOOR_COLORS = [None, (0.62, 0.62, 0.60), (0.66, 0.60, 0.50), (0.27, 0.29, 0.32)]
WALL_COLORS = [(0.75, 0.72, 0.65), (0.85, 0.85, 0.88), (0.50, 0.52, 0.55)]


@dataclass
class EnvConfig:
    arena_size: float = 8.0
    wall_thickness: float = 0.1
    wall_height: float = 0.5
    n_goal_objects: tuple = (3, 6)  # goal + distractors, inclusive range
    n_obstacles: tuple = (5, 15)
    p_hard_distractor: float = 0.5  # chance of distractors that share a word with the goal
    p_goal_visible: float = 0.5  # chance the episode starts with the goal in view
    min_start_goal_dist: float = 2.0  # metres, straight line
    placement_gap: float = 0.15  # minimum free space between placed things
    # Success: the front bumper within success_dist of the goal's surface AND (almost) stopped.
    success_dist: float = 0.30
    success_speed: float = 0.10  # m/s
    success_yaw_rate: float = 0.30  # rad/s
    max_episode_s: float = 60.0
    # Planning grid used for reachability and geodesic distances.
    grid_res: float = 0.05
    car_radius: float = 0.276  # radius of the circle the car sweeps when rotating on the spot
    plan_margin: float = 0.08  # extra clearance the planner keeps (grid error, corner cutting)
    car_half_length: float = 0.20  # centre -> front bumper
    visible_half_angle_deg: float = 30.0  # horizontal half-FOV is 37.5 deg; keep a margin


@dataclass
class EpisodeSpec:
    """Everything needed to rebuild one episode exactly."""

    seed: int
    goal_index: int  # index into ``objects``
    objects: list  # [(shape, colour, x, y)]  - goal + distractors
    obstacles: list  # [(pool_index, x, y, yaw)]
    start: tuple  # (x, y, yaw)
    prompt: str
    floor_index: int
    wall_index: int
    light: dict  # {"azimuth", "elevation", "intensity"}
    hard_distractors: bool
    start_visible: bool  # is the goal actually visible from the start pose?
    geodesic_start: float  # shortest collision-free path length from start to the goal region (m)
    dist_field: np.ndarray = field(repr=False, default=None)  # geodesic distance per grid cell (m)

    @property
    def goal(self):
        return self.objects[self.goal_index]

    @property
    def goal_name(self) -> str:
        shape, colour, *_ = self.goal
        return f"{colour} {shape}"


# ------------------------------------------------------------------------------------------------
# Grid helpers
# ------------------------------------------------------------------------------------------------
class Grid:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.res = cfg.grid_res
        self.half = cfg.arena_size / 2
        self.n = int(round(cfg.arena_size / self.res))
        c = -self.half + (np.arange(self.n) + 0.5) * self.res
        self.X, self.Y = np.meshgrid(c, c, indexing="ij")  # cell centres; index [i, j] = (x, y)

    def cell(self, x: float, y: float) -> tuple[int, int]:
        i = int(np.clip((x + self.half) / self.res, 0, self.n - 1))
        j = int(np.clip((y + self.half) / self.res, 0, self.n - 1))
        return i, j

    def lookup(self, field_: np.ndarray, x: float, y: float) -> float:
        return float(field_[self.cell(x, y)])

    # Shape rasterisation -------------------------------------------------------------------------
    def circle(self, x, y, r):
        return (self.X - x) ** 2 + (self.Y - y) ** 2 <= r * r

    def rect(self, x, y, sx, sy, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = self.X - x, self.Y - y
        u, v = c * dx + s * dy, -s * dx + c * dy  # cell centres in the rectangle's own frame
        return (np.abs(u) <= sx / 2) & (np.abs(v) <= sy / 2)

    def walls(self):
        inner = self.half - self.cfg.wall_thickness / 2
        return (np.abs(self.X) >= inner) | (np.abs(self.Y) >= inner)


def object_mask(grid: Grid, shape: str, x: float, y: float) -> np.ndarray:
    r = GOAL_RADIUS[shape]
    return grid.rect(x, y, 2 * r, 2 * r, 0.0) if shape == "cube" else grid.circle(x, y, r)


def obstacle_mask(grid: Grid, pool_index: int, x: float, y: float, yaw: float) -> np.ndarray:
    kind, size = OBSTACLE_POOL[pool_index]
    if kind == "box":
        return grid.rect(x, y, size[0], size[1], yaw)
    return grid.circle(x, y, size[0])


def surface_distance(shape: str, gx: float, gy: float, px: float, py: float) -> float:
    """Distance from point (px, py) to the surface of a goal object (2-D footprint)."""
    r = GOAL_RADIUS[shape]
    if shape == "cube":
        dx, dy = max(abs(px - gx) - r, 0.0), max(abs(py - gy) - r, 0.0)
        return math.hypot(dx, dy)
    return max(math.hypot(px - gx, py - gy) - r, 0.0)


def occupancy(grid: Grid, spec_objects, spec_obstacles, skip_object: int | None = None) -> np.ndarray:
    occ = grid.walls()
    for k, (shape, _c, x, y) in enumerate(spec_objects):
        if k != skip_object:
            occ |= object_mask(grid, shape, x, y)
    for idx, x, y, yaw in spec_obstacles:
        occ |= obstacle_mask(grid, idx, x, y, yaw)
    return occ


def geodesic_field(grid: Grid, occ: np.ndarray, goal_shape: str, gx: float, gy: float) -> tuple[np.ndarray, np.ndarray]:
    """Shortest collision-free path length from every cell to the goal region.

    1. *Inflate* obstacles by the car's radius (+ a safety margin): if the car's centre stays out of
       the inflated area, its body never touches anything (even while rotating). This turns "plan for
       a car" into the much simpler "plan for a point".
    2. The goal region = free cells close enough to the goal for a success.
    3. Dijkstra's algorithm on the 8-connected grid gives the distance from every cell to the
       nearest goal-region cell. Unreachable cells get +inf.
    Returns (distance field in metres, blocked mask).
    """
    cfg = grid.cfg
    free_dist = ndimage.distance_transform_edt(~occ) * grid.res
    blocked = free_dist < cfg.car_radius + cfg.plan_margin
    gd = np.vectorize(lambda px, py: surface_distance(goal_shape, gx, gy, px, py))(grid.X, grid.Y)
    region = (~blocked) & (gd <= cfg.car_half_length + cfg.success_dist)

    n = grid.n
    idx = np.arange(n * n).reshape(n, n)
    rows, cols, wts = [], [], []
    for di, dj in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        a = idx[max(0, -di): n - max(0, di), max(0, -dj): n - max(0, dj)]
        b = idx[max(0, di): n + min(0, di), max(0, dj): n + min(0, dj)]
        ok = ~blocked.ravel()[a.ravel()] & ~blocked.ravel()[b.ravel()]
        w = grid.res * math.hypot(di, dj)
        rows += [a.ravel()[ok], b.ravel()[ok]]
        cols += [b.ravel()[ok], a.ravel()[ok]]
        wts += [np.full(ok.sum() * 2, w)]
    graph = csr_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(n * n, n * n))
    sources = np.flatnonzero(region.ravel())
    if len(sources) == 0:
        return np.full((n, n), np.inf), blocked
    dist = dijkstra(graph, directed=False, indices=sources, min_only=True)
    return dist.reshape(n, n), blocked


def shortest_path(grid: Grid, dist: np.ndarray, x: float, y: float, max_steps: int = 5000) -> np.ndarray:
    """Follow the distance field downhill from (x, y) to the goal region. Returns (k, 2) points."""
    i, j = grid.cell(x, y)
    pts = [(grid.X[i, j], grid.Y[i, j])]
    for _ in range(max_steps):
        if dist[i, j] == 0 or not np.isfinite(dist[i, j]):
            break
        best = (dist[i, j], i, j)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                a, b = i + di, j + dj
                if 0 <= a < grid.n and 0 <= b < grid.n and dist[a, b] < best[0]:
                    best = (dist[a, b], a, b)
        if (best[1], best[2]) == (i, j):
            break
        _, i, j = best
        pts.append((grid.X[i, j], grid.Y[i, j]))
    return np.array(pts)


def line_of_sight(grid: Grid, occ: np.ndarray, x0, y0, x1, y1) -> bool:
    """True if the straight segment between the two points crosses no occupied cell."""
    n = max(2, int(math.hypot(x1 - x0, y1 - y0) / (grid.res / 2)))
    for t in np.linspace(0, 1, n)[1:-1]:
        if occ[grid.cell(x0 + t * (x1 - x0), y0 + t * (y1 - y0))]:
            return False
    return True


def goal_in_view(cfg: EnvConfig, grid: Grid, occ_without_goal: np.ndarray, pose, goal_xy) -> bool:
    """Is the goal inside the camera's horizontal field of view and not hidden behind something?"""
    x, y, yaw = pose
    bearing = math.atan2(goal_xy[1] - y, goal_xy[0] - x)
    err = (bearing - yaw + math.pi) % (2 * math.pi) - math.pi
    if abs(err) > math.radians(cfg.visible_half_angle_deg):
        return False
    return line_of_sight(grid, occ_without_goal, x, y, *goal_xy)


# ------------------------------------------------------------------------------------------------
# Episode sampling
# ------------------------------------------------------------------------------------------------
def _choose_objects(rng: np.random.Generator, cfg: EnvConfig):
    colours = list(COLORS)
    n = int(rng.integers(cfg.n_goal_objects[0], cfg.n_goal_objects[1] + 1))
    goal = (SHAPES[rng.integers(3)], colours[rng.integers(len(colours))])
    chosen = [goal]
    hard = bool(rng.random() < cfg.p_hard_distractor)
    if hard:  # at least one distractor that shares exactly one word with the goal
        same_colour = [(s, goal[1]) for s in SHAPES if s != goal[0]]
        same_shape = [(goal[0], c) for c in colours if c != goal[1]]
        options = [same_colour[rng.integers(len(same_colour))], same_shape[rng.integers(len(same_shape))]]
        k = 1 if n == 3 else int(rng.integers(1, 3))
        for o in rng.permutation(len(options))[:k]:
            if len(chosen) < n:
                chosen.append(options[o])
    rest = [(s, c) for s in SHAPES for c in colours if (s, c) not in chosen]
    for o in rng.permutation(len(rest))[: n - len(chosen)]:
        chosen.append(rest[o])
    return chosen, hard


def sample_episode(seed: int, cfg: EnvConfig | None = None, max_tries: int = 50) -> EpisodeSpec:
    """Sample a random, solvable episode. The same seed always gives the same episode.
    Seeds >= carsim.words.WORDS_SEED_BASE are words layouts (see carsim/words.py)."""
    cfg = cfg or EnvConfig()
    if seed >= 10**12:
        from .words import words_spec
        return words_spec(seed, cfg)
    rng = np.random.default_rng(seed)
    grid = Grid(cfg)
    lim = cfg.arena_size / 2 - cfg.wall_thickness / 2 - 0.35  # keep things off the walls

    for _ in range(max_tries):
        # 1) Place goal objects and obstacles without overlaps -------------------------------------
        combos, hard = _choose_objects(rng, cfg)
        occ = grid.walls()
        gap_cells = int(math.ceil(cfg.placement_gap / grid.res))

        def try_place(mask_fn):
            for _ in range(100):
                x, y = rng.uniform(-lim, lim, size=2)
                yaw = float(rng.uniform(0, math.pi))
                m = mask_fn(x, y, yaw)
                if not (ndimage.binary_dilation(m, iterations=gap_cells) & occ).any():
                    return x, y, yaw, m
            return None

        objects, obstacles, ok = [], [], True
        for shape, colour in combos:
            r = try_place(lambda x, y, yaw: object_mask(grid, shape, x, y))
            if r is None:
                ok = False
                break
            objects.append((shape, colour, float(r[0]), float(r[1])))
            occ |= r[3]
        n_obs = int(rng.integers(cfg.n_obstacles[0], cfg.n_obstacles[1] + 1))
        for pool_idx in rng.permutation(len(OBSTACLE_POOL))[:n_obs] if ok else []:
            r = try_place(lambda x, y, yaw: obstacle_mask(grid, int(pool_idx), x, y, yaw))
            if r is not None:  # if one obstacle doesn't fit we simply use one fewer
                obstacles.append((int(pool_idx), float(r[0]), float(r[1]), float(r[2])))
                occ |= r[3]
        if not ok:
            continue

        # 2) Reachability: geodesic distance field to the goal --------------------------------------
        g_shape, _gc, gx, gy = objects[0]
        dist, blocked = geodesic_field(grid, occ, g_shape, gx, gy)

        # 3) Start pose: reachable, far enough, with the requested visibility ------------------------
        want_visible = bool(rng.random() < cfg.p_goal_visible)
        occ_no_goal = occupancy(grid, objects, obstacles, skip_object=0)
        cand = np.argwhere(np.isfinite(dist) & ~blocked & (np.hypot(grid.X - gx, grid.Y - gy) >= cfg.min_start_goal_dist))
        start = None
        for k in rng.permutation(len(cand))[:200]:
            i, j = cand[k]
            x, y = float(grid.X[i, j]), float(grid.Y[i, j])
            bearing = math.atan2(gy - y, gx - x)
            if want_visible:
                yaw = bearing + math.radians(rng.uniform(-20, 20))
            else:  # goal behind or to the side: the car has to turn to find it
                yaw = bearing + math.radians(rng.uniform(60, 300))
            yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
            visible = goal_in_view(cfg, grid, occ_no_goal, (x, y, yaw), (gx, gy))
            if visible == want_visible:
                start = (x, y, yaw)
                break
        if start is None:
            continue

        # 4) Language and looks ----------------------------------------------------------------------
        shape_word = SHAPE_WORDS[g_shape][rng.integers(len(SHAPE_WORDS[g_shape]))]
        prompt = PROMPT_TEMPLATES[rng.integers(len(PROMPT_TEMPLATES))].format(c=objects[0][1], s=shape_word)
        light = {
            "azimuth": float(rng.uniform(0, 2 * math.pi)),
            "elevation": float(rng.uniform(math.radians(35), math.radians(70))),
            "intensity": float(rng.uniform(3.5, 6.5)),
        }
        # Shuffle objects so the goal isn't always index 0 (avoids accidental "first object" shortcuts).
        order = rng.permutation(len(objects))
        objects = [objects[o] for o in order]
        return EpisodeSpec(
            seed=seed,
            goal_index=int(np.flatnonzero(order == 0)[0]),
            objects=objects,
            obstacles=obstacles,
            start=start,
            prompt=prompt,
            floor_index=int(rng.integers(len(FLOOR_COLORS))),
            wall_index=int(rng.integers(len(WALL_COLORS))),
            light=light,
            hard_distractors=hard,
            start_visible=want_visible,
            geodesic_start=grid.lookup(dist, start[0], start[1]),
            dist_field=dist,
        )
    raise RuntimeError(f"Could not sample a valid episode for seed {seed}")
