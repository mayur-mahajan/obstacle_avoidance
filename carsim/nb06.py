"""Helpers for notebook 06: running the pipeline steps, training charts, test tables, luck tests and the route pictures.

Everything here reads files the scripts write (runs/...), so the notebook's charts can be redrawn at any time without
retraining anything.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from functools import lru_cache
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
FRAMES = RUNS / "overhead_frames" / "route_follow"
TESTS = RUNS / "car06_tests"


# ---------------------------------------------------------------------------------------------------------------
# Running a pipeline step (skipped when its results already exist)
# ---------------------------------------------------------------------------------------------------------------
def run(args, outputs, force: bool = False, log=None, show_last: int = 6):
    """Run ``python args...`` from the project folder unless every path in ``outputs`` exists already.
    Long steps print only their last lines; the full output goes to ``log`` (if given)."""
    outputs = [Path(o) if Path(o).is_absolute() else ROOT / o for o in outputs]
    if not force and outputs and all(o.exists() for o in outputs):
        print("already done:", ", ".join(str(o.relative_to(ROOT)) for o in outputs))
        return None
    t0 = time.time()
    print("running:", "python", " ".join(map(str, args)))
    r = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True)
    out = (r.stdout + r.stderr).splitlines()
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        Path(log).write_text("\n".join(out))
    keep = [l for l in out if not any(w in l for w in ("Warning", "warn(", "Loading weights", "[Genesis]"))]
    print("\n".join(keep[-show_last:]))
    print(f"({(time.time() - t0) / 60:.1f} min, exit code {r.returncode})")
    return r.returncode


def read_jsonl(path):
    """The JSON lines of a training log (other lines are skipped)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


# ---------------------------------------------------------------------------------------------------------------
# Training charts
# ---------------------------------------------------------------------------------------------------------------
def plot_car_training(logs: dict, fig=None):
    """logs: {label: path to runs/car06_logs/<M>_s0.log}. Left: training loss. Right: validation loss WITH help
    (solid) and WITHOUT help (dashed) - if the car uses the help, solid must sit clearly below dashed."""
    import matplotlib.pyplot as plt
    fig = fig or plt.figure(figsize=(13, 4))
    a, b = fig.subplots(1, 2)
    for k, (label, path) in enumerate(logs.items()):
        rows, c = read_jsonl(path), f"C{k}"
        a.plot([r["step"] for r in rows], [r["driving_loss"] for r in rows], color=c, label=label)
        v = [r for r in rows if "val_loss_with_help" in r]
        if v:
            b.plot([r["step"] for r in v], [r["val_loss_with_help"] for r in v], "-o", color=c, ms=3, label=f"{label}: with help")
            b.plot([r["step"] for r in v], [r["val_loss_no_help"] for r in v], "--", color=c, alpha=0.7, label=f"{label}: no help")
    a.set_title("training loss (copying the driver)"); a.set_xlabel("step"); a.legend(fontsize=8)
    b.set_title("validation loss: with help (solid) vs without (dashed)"); b.set_xlabel("step"); b.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def plot_map_training(log, fig=None):
    """The obstacle-map reader: overlap, blocked cells found, and how many of its 'blocked' marks are right."""
    import matplotlib.pyplot as plt
    fig = fig or plt.figure(figsize=(13, 3.6))
    axs = fig.subplots(1, 3)
    st = [r["step"] for r in log]
    v = [r for r in log if "val_rooms_iou" in r]
    for ax, key, title in zip(axs, ("iou", "recall", "precision"),
                              ("overlap with the true map (IoU)", "blocked cells found (recall)",
                               "'blocked' marks that are right (precision)")):
        ax.plot(st, [r[f"train_{key}"] for r in log], label="training pictures", alpha=0.7)
        ax.plot([r["step"] for r in v], [r[f"val_rooms_{key}"] for r in v], "-o", ms=3, label="unseen rooms")
        ax.plot([r["step"] for r in v], [r[f"val_drive_{key}"] for r in v], "--s", ms=3, label="unseen driving rooms")
        ax.set_ylim(0, 1); ax.set_title(title); ax.set_xlabel("step"); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Test results: tables, luck tests, the "box in the way" split
# ---------------------------------------------------------------------------------------------------------------
def episodes(name, mode="help"):
    """The 128 test drives of runs/car06_tests/<name>.json in one mode (help / no_help / other_room)."""
    return json.load(open(TESTS / f"{name}.json"))[mode]["episodes"]


@lru_cache(maxsize=None)
def _detour(seed):
    """Shortest obstacle-free route from the start / straight distance to the goal (>= 1.0: a box is in the way)."""
    from .world import sample_episode
    s = sample_episode(int(seed))
    x, y, _ = s.start; gx, gy = s.goal[2:4]
    return s.geodesic_start / max(math.hypot(gx - x, gy - y), 1e-6)


def _stuck(e):
    return (not e["success"]) and e["collision_steps"] > 20


def summary(rows):
    """rows: [(label, file, mode)] -> table: success (all / goal hidden / box in the way), touching, time, failures."""
    import pandas as pd
    out = []
    for label, name, mode in rows:
        E = episodes(name, mode)
        hid = [e for e in E if not e["start_visible"]]
        box = [e for e in E if _detour(e["seed"]) >= 1.0]
        ok = [e["seconds"] for e in E if e["success"]]
        okh = [e["seconds"] for e in hid if e["success"]]
        fail = [e for e in E if not e["success"]]
        out.append({"car": label, "success": np.mean([e["success"] for e in E]),
                    "goal hidden at start": np.mean([e["success"] for e in hid]),
                    "box in the way": np.mean([e["success"] for e in box]),
                    "touched something": np.mean([e["collision_steps"] > 0 for e in E]),
                    "s per success": np.mean(ok) if ok else np.nan, "s per success, hidden": np.mean(okh) if okh else np.nan,
                    "failures": len(fail), "...stuck": sum(_stuck(e) for e in fail)})
    df = pd.DataFrame(out).set_index("car")
    pct = ["success", "goal hidden at start", "box in the way", "touched something"]
    return df.style.format({**{c: "{:.0%}" for c in pct}, "s per success": "{:.1f}", "s per success, hidden": "{:.1f}"})


def sign_test(w, l):
    n = w + l
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n) if n else 1.0


def paired(a, b, which="all"):
    """Same rooms, drive by drive: a, b = (file, mode). which: all / hidden / box. -> 'wins:losses, p'."""
    A = {(e["env"], e["k"]): e for e in episodes(*a)}
    B = {(e["env"], e["k"]): e for e in episodes(*b)}
    keep = {"all": lambda e: True, "hidden": lambda e: not e["start_visible"],
            "box": lambda e: _detour(e["seed"]) >= 1.0}[which]
    ks = [k for k in A if keep(A[k])]
    w = sum(A[k]["success"] and not B[k]["success"] for k in ks)
    l = sum(B[k]["success"] and not A[k]["success"] for k in ks)
    p = sign_test(w, l)
    return f"{w}:{l}  p = {p:.3f}" + ("  (real)" if p < 0.05 else "  (could be luck)")


def paired_table(pairs):
    """pairs: [(label, a, b)] -> table of wins:losses on all rooms / goal hidden / box in the way."""
    import pandas as pd
    return pd.DataFrame([{"comparison": lab, "all rooms": paired(a, b), "goal hidden": paired(a, b, "hidden"),
                          "box in the way": paired(a, b, "box")} for lab, a, b in pairs]).set_index("comparison")


def results_chart(rows, ax=None):
    """Grouped bars: success on all rooms / goal hidden / box in the way."""
    import matplotlib.pyplot as plt
    ax = ax or plt.figure(figsize=(13, 4)).gca()
    groups = ["all rooms", "goal hidden at start", "box in the way"]
    w = 0.8 / len(rows)
    for k, (label, name, mode) in enumerate(rows):
        E = episodes(name, mode)
        vals = [np.mean([e["success"] for e in E]), np.mean([e["success"] for e in E if not e["start_visible"]]),
                np.mean([e["success"] for e in E if _detour(e["seed"]) >= 1.0])]
        bars = ax.bar(np.arange(3) + (k - (len(rows) - 1) / 2) * w, vals, w, label=label)
        ax.bar_label(bars, [f"{v:.0%}" for v in vals], fontsize=7)
    ax.set_xticks(range(3), [f"{g}\n({n} rooms)" for g, n in zip(groups, _group_sizes(rows[0][1]))])
    ax.set_ylim(0, 1.1); ax.set_ylabel("success"); ax.legend(fontsize=8, ncol=len(rows), loc="lower center", bbox_to_anchor=(0.5, 1.0))
    return ax


def _group_sizes(name):
    E = episodes(name)
    return [len(E), sum(not e["start_visible"] for e in E), sum(_detour(e["seed"]) >= 1.0 for e in E)]


def detour_thirds(rows):
    """Success and stuck drives in the third of rooms needing the least / middle / most detour."""
    import pandas as pd
    d = np.array([_detour(e["seed"]) for e in episodes(rows[0][1])])
    q = np.quantile(d, [1 / 3, 2 / 3])
    out = []
    for label, name, mode in rows:
        E = episodes(name, mode)
        r = {"car": label}
        for lo, hi, g in [(0, q[0], "least detour"), (q[0], q[1], "middle"), (q[1], 99, "most detour")]:
            L = [e for e in E if lo <= _detour(e["seed"]) < hi]
            r[g] = f"{np.mean([e['success'] for e in L]):.0%} ({sum(_stuck(e) for e in L)} stuck)"
        out.append(r)
    return pd.DataFrame(out).set_index("car")


# ---------------------------------------------------------------------------------------------------------------
# Accuracy of each server part (on the 15,494 driving frames; all parts out-of-sample there)
# ---------------------------------------------------------------------------------------------------------------
def _ang(a, b):
    return np.abs(np.degrees((a - b + np.pi) % (2 * np.pi) - np.pi))


def localisation_table():
    """Car position / heading and goal position: learned server vs classic tracking (+ colour for the goal)."""
    import pandas as pd
    meta = np.load(FRAMES / "meta.npz"); pose = meta["pose"]
    fp, ft = np.load(FRAMES / "facts_pred.npy"), np.load(FRAMES / "facts_true.npy")
    tp, gr = np.load(FRAMES / "tracker_pose.npy"), np.load(FRAMES / "goal_refined.npy")
    srv_pos = np.hypot(fp[:, 0] * 4 - pose[:, 0], fp[:, 1] * 4 - pose[:, 1])
    srv_head = _ang(np.arctan2(fp[:, 3], fp[:, 2]), pose[:, 2])
    ok = tp[:, 3] > 0
    trk_pos = np.hypot(tp[ok, 0] - pose[ok, 0], tp[ok, 1] - pose[ok, 1]); trk_head = _ang(tp[ok, 2], pose[ok, 2])
    g_true = ft[:, 4:6] * 4
    g_srv = np.hypot(*(fp[:, 4:6] * 4 - g_true).T); g_ref = np.hypot(*(gr[:, :2] - g_true).T)
    row = lambda name, found, e, unit, extra="": {"what": name, "found": f"{found:.1%}", "typical error": f"{np.median(e):.2f} {unit}",
                                                 "90% of frames within": f"{np.percentile(e, 90):.2f} {unit}", "note": extra}
    return pd.DataFrame([
        row("car position: learned server", 1.0, srv_pos, "m"),
        row("car position: marker tracking", ok.mean(), trk_pos, "m"),
        row("car heading: learned server", 1.0, srv_head, "deg", f"backwards (>90 deg): {np.mean(srv_head > 90):.1%}"),
        row("car heading: marker tracking", ok.mean(), trk_head, "deg", f"backwards (>90 deg): {np.mean(trk_head > 90):.1%}"),
        row("goal position: learned server", 1.0, g_srv, "m"),
        row("goal position: server + colour", gr[:, 2].mean(), g_ref, "m", f"> 0.6 m off: {np.mean(g_ref > 0.6):.1%}"),
    ]).set_index("what")


def route_error_table():
    """How far the real routes' next point (0.6 m ahead) points away from the perfect route's."""
    import pandas as pd
    t = np.load(FRAMES / "route_true.npy")
    rows = []
    for name, f in [("learned server for everything", "route_pred.npy"),
                    ("tracker + learned map + colour-refined goal (used)", "route_tracked.npy"),
                    ("learned map, but TRUE car + goal (map's own limit)", "route_map_only.npy")]:
        r = np.load(FRAMES / f)
        e = _ang(np.arctan2(r[:, 3], r[:, 2]), np.arctan2(t[:, 3], t[:, 2]))
        rows.append({"route from": name, "typical error": f"{np.median(e):.0f} deg", "off by > 45 deg": f"{np.mean(e > 45):.1%}",
                     "off by > 90 deg": f"{np.mean(e > 90):.1%}"})
    return pd.DataFrame(rows).set_index("route from")


# ---------------------------------------------------------------------------------------------------------------
# Pictures
# ---------------------------------------------------------------------------------------------------------------
_ENV = {}


def _env():
    if "env" not in _ENV:
        from . import overhead
        from .env import NavEnv
        _ENV["env"] = NavEnv(car_cfg=overhead.car_config(two_colour=True), overview_camera_res=(1024, 1024), backend="cpu")
    return _ENV["env"]


def overhead_and_front(seed: int, jitter=None):
    """(overhead picture 1024 px, the car's front-camera picture) at a room's start pose."""
    from . import overhead
    from .world import sample_episode
    env = _env(); env.reset(spec=sample_episode(seed))
    return overhead.render(env, jitter), env._obs()["image"]


def _path(grid, field, x, y):
    """The planner's path from (x, y); if the car is inside an obstacle's safety margin, from the nearest free cell
    (as carsim/route.py route_facts does)."""
    from .world import shortest_path
    i, j = grid.cell(float(x), float(y))
    if not np.isfinite(field[i, j]):
        sl = field[max(0, i - 6): i + 7, max(0, j - 6): j + 7]
        if np.isfinite(sl).any():
            a, b = np.unravel_index(np.nanargmin(np.where(np.isfinite(sl), sl, np.nan)), sl.shape)
            x, y = grid.X[max(0, i - 6) + a, max(0, j - 6) + b], grid.Y[max(0, i - 6) + a, max(0, j - 6) + b]
    return shortest_path(grid, field, float(x), float(y))


def route_figure(frame: int, map_ckpt=RUNS / "checkpoints" / "server_map_s0.pt", fig=None):
    """One driving frame: (1) the overhead picture with what the tracker found, (2) the true obstacle map with the
    perfect route, (3) the server's learned map with the real route it planned."""
    import matplotlib.pyplot as plt
    import torch
    from . import overhead
    from .data import load_training_data
    from .route import goal_shape_from_prompt, planned_field
    from .server_map import MapHead, label_map
    from .tracker import floor_to_pixel
    from .world import EnvConfig, Grid, sample_episode, shortest_path
    meta = np.load(FRAMES / "meta.npz")
    _p, table, _f, _e = load_training_data(RUNS / "datasets" / "route_follow_prep.json")
    seed, pose, jit, prompt = int(meta["seed"][frame]), meta["pose"][frame], meta["jitter"][frame], table.iloc[frame]["task"]
    spec = sample_episode(seed); grid = Grid(EnvConfig())
    env = _env(); env.reset(spec=spec); env.car.reset(*map(float, pose))
    pic = overhead.render(env, jit)
    tp, gr = np.load(FRAMES / "tracker_pose.npy")[frame], np.load(FRAMES / "goal_refined.npy")[frame]
    head = MapHead(); head.load_state_dict(torch.load(map_ckpt, map_location="cpu")["state"]); head.eval()
    feats = np.load(FRAMES / "feats.npy", mmap_mode="r")
    with torch.no_grad():
        pred = (head(torch.from_numpy(np.array(feats[frame:frame + 1])).float())[0].sigmoid() > 0.5).numpy()
    truth = label_map(spec, grid)
    path_true = _path(grid, spec.dist_field, pose[0], pose[1])
    fld = planned_field(grid, pred, goal_shape_from_prompt(prompt), float(gr[0]), float(gr[1]))
    path_real = _path(grid, fld, tp[0], tp[1]) if np.isfinite(fld).any() else np.zeros((0, 2))
    fig = fig or plt.figure(figsize=(15, 5))
    a, b, c = fig.subplots(1, 3)
    a.imshow(pic)
    px = floor_to_pixel([[tp[0], tp[1], 0.11], [tp[0] + 0.4 * math.cos(tp[2]), tp[1] + 0.4 * math.sin(tp[2]), 0.11],
                         [gr[0], gr[1], 0.2]], jit)
    a.annotate("", px[1], px[0], arrowprops=dict(color="yellow", width=2, headwidth=8))
    a.plot(*px[2], "x", color="yellow", ms=14, mew=3)
    a.set_title(f'"{prompt}"\noverhead picture: tracked car (arrow) and goal (x)'); a.axis("off")
    ext = [-4, 4, -4, 4]
    for ax, m, path, title, col in [(b, truth, path_true, "true obstacles + perfect route", "lime"),
                                    (c, pred, path_real, "server's learned map + real route", "orange")]:
        ax.imshow(m.T, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1.6)
        if len(path):
            ax.plot(path[:, 0], path[:, 1], color=col, lw=2.5)
        ax.arrow(pose[0], pose[1], 0.35 * math.cos(pose[2]), 0.35 * math.sin(pose[2]), width=0.05, color="tab:blue")
        ax.plot(*spec.goal[2:4], "*", color="red", ms=16)
        ax.set_title(title); ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect("equal")
    c.contour(truth.T, levels=[0.5], origin="lower", extent=ext, colors="red", linewidths=0.6)
    c.text(-3.9, -3.85, "red outline = true obstacles", color="red", fontsize=8)
    fig.tight_layout()
    return fig


def map_versions_table(logs: dict):
    """{label: route log of scripts/server_route_frames.py} -> map quality and route quality side by side."""
    import pandas as pd
    rows = []
    for label, path in logs.items():
        text = Path(path).read_text()
        rep = json.loads(text[text.rfind("\n{") + 1:])
        full = next(v for k, v in rep.items() if k.startswith("tracker pose"))
        rows.append({"map": label, "blocked cells found (driving rooms)": f"{rep['map_recall']:.0%}",
                     "'blocked' marks right": f"{rep['map_precision']:.0%}",
                     "route points touching an obstacle": f"{rep['route point touches a real obstacle (this route)']:.2f}%",
                     "route points off > 45 deg (real system)": f"{full['over45_pct']:.1f}%",
                     "... with true car + goal (map's limit)": f"{rep['server map only (true goal + pose)']['over45_pct']:.1f}%"})
    return pd.DataFrame(rows).set_index("map")
