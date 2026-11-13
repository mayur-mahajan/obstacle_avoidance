"""Matplotlib drawings of episodes seen from above (for the notebooks)."""

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge

from .world import COLORS, GOAL_RADIUS, OBSTACLE_COLORS, OBSTACLE_POOL, EnvConfig, Grid, shortest_path

_MARK = {"sphere": "o", "cube": "s", "cylinder": "h"}


def _rect_corners(x, y, sx, sy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = np.array([[-sx, -sy], [sx, -sy], [sx, sy], [-sx, sy]]) / 2
    return pts @ np.array([[c, s], [-s, c]]) + [x, y]


def draw_episode(spec, ax=None, cfg: EnvConfig | None = None, show_field=False, show_path=True,
                 trajectory=None, title=None, fov_deg=75.0):
    cfg = cfg or EnvConfig()
    grid = Grid(cfg)
    ax = ax or plt.gca()
    h = cfg.arena_size / 2
    if show_field and spec.dist_field is not None:
        f = np.where(np.isfinite(spec.dist_field), spec.dist_field, np.nan)
        im = ax.imshow(f.T, origin="lower", extent=(-h, h, -h, h), cmap="viridis_r", alpha=0.85)
        plt.colorbar(im, ax=ax, fraction=0.046, label="geodesic distance to goal (m)")
    ax.add_patch(Rectangle((-h, -h), 2 * h, 2 * h, fill=False, lw=3, color="#8a8170"))
    for idx, x, y, yaw in spec.obstacles:
        kind, size = OBSTACLE_POOL[idx]
        col = OBSTACLE_COLORS[idx]
        if kind == "box":
            ax.add_patch(Polygon(_rect_corners(x, y, size[0], size[1], yaw), color=col, ec="k", lw=0.6))
        else:
            ax.add_patch(Circle((x, y), size[0], color=col, ec="k", lw=0.6))
    for k, (shape, colour, x, y) in enumerate(spec.objects):
        r = GOAL_RADIUS[shape]
        if shape == "cube":
            ax.add_patch(Rectangle((x - r, y - r), 2 * r, 2 * r, color=COLORS[colour], ec="k", lw=0.8))
        elif shape == "sphere":
            ax.add_patch(Circle((x, y), r, color=COLORS[colour], ec="k", lw=0.8))
        else:
            ax.add_patch(Polygon([(x + r * math.cos(a), y + r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 7)[:-1]],
                                 color=COLORS[colour], ec="k", lw=0.8))
        if k == spec.goal_index:
            ax.add_patch(Circle((x, y), r + cfg.car_half_length + cfg.success_dist, fill=False, ls="--", color="k", lw=1))
            ax.plot(x, y, "*", color="white", ms=9, mec="k")
    sx, sy, syaw = spec.start
    half_fov = math.degrees(math.atan(math.tan(math.radians(fov_deg) / 2)))  # square image: same h/v FOV
    ax.add_patch(Wedge((sx, sy), 1.5, math.degrees(syaw) - half_fov, math.degrees(syaw) + half_fov, color="gold", alpha=0.25))
    ax.arrow(sx, sy, 0.4 * math.cos(syaw), 0.4 * math.sin(syaw), width=0.06, color="k", length_includes_head=True)
    if show_path and spec.dist_field is not None:
        p = shortest_path(grid, spec.dist_field, sx, sy)
        ax.plot(p[:, 0], p[:, 1], ":", color="#c0504d", lw=1.8, label=f"shortest path {spec.geodesic_start:.1f} m")
    if trajectory is not None:
        ax.plot(trajectory[:, 0], trajectory[:, 1], "-", color="#2f6db5", lw=1.6, label="driven")
    ax.set_xlim(-h - 0.2, h + 0.2); ax.set_ylim(-h - 0.2, h + 0.2); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(title if title is not None else f'"{spec.prompt}"\n(goal {"visible" if spec.start_visible else "hidden"} at start)', fontsize=9)
    return ax
