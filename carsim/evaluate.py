"""Charts and statistics for the driving tests (scripts/rl_tests.py), used by notebooks 04 and 05.

The two tests (always the same drives, so different models can be compared drive by drive):
  words test ("which")   empty floor, objects 2-3 m away; "reached" = front bumper within 0.35 m of an object
      alone          one object in view                                 -> can it drive to an object at all?   (30)
      pair           two objects in view; every layout is driven twice, asking for each object in turn
                     -> does it follow the WORDS?  50% = coin flip                                            (180)
                     decoy kinds: same colour (must read the shape word), same shape (must read the colour
                     word), unrelated (either word works)
      behind         goal behind the car, nothing in front               -> does it search?                    (30)
      behind+decoy   goal behind, a decoy in front                       -> does it settle for the decoy?      (30)
  standard test          128 ordinary random rooms with obstacles, 60 s each -> success, goal hidden at start, touching

Every rate is shown with its 95% range: the interval in which the true rate lies, given that a test of n drives has
luck in it (Clopper-Pearson interval). Two models are compared drive by drive with a sign test (`paired`).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCORECARDS = ROOT / "runs" / "rl" / "scorecards"
DIAGNOSIS = ROOT / "runs" / "diagnosis"
COLOURS = ["#2f6db5", "#e8853d", "#5a9e5a", "#9467bd", "#8c564b"]
OBJ_COLOURS = {"red": "r", "green": "g", "blue": "b", "yellow": "y", "orange": "orange", "purple": "purple",
               "white": "lightgrey", "black": "k", "cyan": "c", "pink": "pink"}


def load(tag, test):
    return json.load(open(SCORECARDS / f"{tag}_{test}.json"))


def rate(xs):
    """(rate, low, high): the share of True and its 95% range."""
    from scipy.stats import binomtest
    xs = np.asarray(xs, bool)
    if not len(xs):
        return float("nan"), float("nan"), float("nan")
    ci = binomtest(int(xs.sum()), len(xs)).proportion_ci()
    return float(xs.mean()), float(ci.low), float(ci.high)


def words_results(tag):
    """{test name: list of True/False per drive} for the words test."""
    d = load(tag, "which")["drives"]
    out = {"alone": [r["went"] == "goal" for r in d if r["test"] == "alone"],
           "pair: asked-for object": [r["went"] == "goal" for r in d if r["test"] == "pair"]}
    for kind, name in (("same_colour", "pair, same colour\n(read the shape)"), ("same_shape", "pair, same shape\n(read the colour)"),
                       ("unrelated", "pair, unrelated")):
        out[name] = [r["went"] == "goal" for r in d if r["test"] == "pair" and r["kind"] == kind]
    out["behind"] = [r["went"] == "goal" for r in d if r["test"] == "behind"]
    out["behind + decoy"] = [r["went"] == "goal" for r in d if r["test"] == "behind+decoy"]
    return out


def standard_results(tag):
    eps = load(tag, "standard")["episodes"]
    return {"all rooms": [e["success"] for e in eps],
            "goal visible at start": [e["success"] for e in eps if e["start_visible"]],
            "goal hidden at start": [e["success"] for e in eps if not e["start_visible"]],
            "touched something": [e["collision_steps"] > 0 for e in eps]}


def _bars(ax, groups, models, title, ref=None, ref_label=None):
    names = list(groups[models[0]])
    w = 0.8 / len(models)
    for k, m in enumerate(models):
        vals = [rate(groups[m][n]) for n in names]
        x = np.arange(len(names)) + (k - (len(models) - 1) / 2) * w
        ax.bar(x, [v[0] for v in vals], w, color=COLOURS[k % len(COLOURS)], label=m,
               yerr=np.array([[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]]), capsize=3, error_kw={"lw": 1})
        for xi, v in zip(x, vals):
            ax.text(xi, 0.02, f"{v[0]:.0%}", ha="center", va="bottom", fontsize=7, color="w", rotation=90)
    ax.set_xticks(range(len(names)), [f"{n}\n({len(groups[models[0]][n])})" for n in names], fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_title(title, fontsize=10)
    if ref is not None:
        ax.axhline(ref, ls="--", c="k", lw=0.8, label=ref_label)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))


def words_chart(models: dict, ax=None):
    """models = {display name: scorecard tag}. Bars = share of drives that reached the asked-for object; whiskers =
    95% range; (n) = number of drives; dashed line = coin flip for the pair tests."""
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(14, 4))[1]
    _bars(ax, {m: words_results(t) for m, t in models.items()}, list(models),
          "Words test: reached the object that was ASKED for", 0.5, "coin flip (pair tests)")
    return ax


def standard_chart(models: dict, ax=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(9, 4))[1]
    _bars(ax, {m: standard_results(t) for m, t in models.items()}, list(models),
          "Standard test: 128 ordinary rooms (60 s each)")
    return ax


def summary_table(models: dict):
    """One row per model: the headline numbers, as a pandas table of percentages."""
    import pandas as pd
    rows = {}
    for m, t in models.items():
        w, s = words_results(t), standard_results(t)
        side = [r["went_side"] == "left" for r in load(t, "which")["drives"] if r["test"] == "pair" and r["went_side"]]
        rows[m] = {"alone": np.mean(w["alone"]), "pair: asked-for object": np.mean(w["pair: asked-for object"]),
                   "behind + decoy": np.mean(w["behind + decoy"]), "pair: went left (50% = no habit)": np.mean(side),
                   "standard: success": np.mean(s["all rooms"]), "standard: touched something": np.mean(s["touched something"])}
    return pd.DataFrame(rows).T


def paired(tag_a, tag_b, test="which", name_a=None, name_b=None):
    """Drive-by-drive comparison on the SAME drives. Counts the drives only A got right and only B got right; if
    the models were equally good, each such drive would be a coin flip between A and B, so p = the chance of a
    split at least this uneven by luck alone (two-sided sign test). p < 0.05: very unlikely to be luck."""
    from scipy.stats import binomtest
    if test == "which":
        ga, gb = (words_results(t)["pair: asked-for object"] for t in (tag_a, tag_b))
        what = "pair drives (asked-for object)"
    else:
        ga, gb = (standard_results(t)["all rooms"] for t in (tag_a, tag_b))
        what = "standard rooms"
    only_a = sum(a and not b for a, b in zip(ga, gb)); only_b = sum(b and not a for a, b in zip(ga, gb))
    p = binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0
    na, nb = name_a or tag_a, name_b or tag_b
    verdict = "very unlikely to be luck" if p < 0.05 else "could easily be luck"
    return (f"{what}: {na} {np.mean(ga):.0%} vs {nb} {np.mean(gb):.0%} | only {na} right: {only_a}, only {nb} right: "
            f"{only_b} | p = {p:.3f} ({verdict})")


def _goal_visible_end(e):
    """Was the goal in the picture when the drive ended? Recorded directly by newer test runs; for older ones it is
    worked out from the sight reward (which was +0.2 per gain, -0.2 per loss)."""
    if "goal_visible_end" in e:
        return bool(e["goal_visible_end"])
    return int(e["start_visible"]) + round(e["parts"]["sight"] / 0.2) > 0


def failure_table(models: dict):
    """Standard test, in detail: where does each model succeed and fail? One column per model.
    seconds touching = time spent in contact with an obstacle or wall (0.1 s per step)."""
    import pandas as pd
    rows = {}
    for m, t in models.items():
        eps = load(t, "standard")["episodes"]
        fail = [e for e in eps if not e["success"]]
        n = len(eps)
        rows[m] = {
            "success: all rooms": f"{np.mean([e['success'] for e in eps]):.0%}",
            "success: goal visible at start": f"{np.mean([e['success'] for e in eps if e['start_visible']]):.0%}",
            "success: goal hidden at start": f"{np.mean([e['success'] for e in eps if not e['start_visible']]):.0%}",
            "drives that touched something": f"{np.mean([e['collision_steps'] > 0 for e in eps]):.0%}",
            "seconds touching per drive (mean)": f"{np.mean([e['collision_steps'] for e in eps]) / 10:.1f}",
            "failed drives": f"{len(fail)} of {n}",
            "... stuck: 10 s or more touching": f"{sum(e['collision_steps'] >= 100 for e in fail)}",
            "... goal out of view at the end": f"{sum(not _goal_visible_end(e) for e in fail)}",
            "seconds to reach the goal (median, successes)": f"{np.median([e['seconds'] for e in eps if e['success']]):.1f}",
        }
    return pd.DataFrame(rows)


def _room_detail(tag):
    p = DIAGNOSIS / f"{tag}_standard_detail.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run scripts/rl_tests.py standard ... --detail {p}")
    return json.load(open(p))


def failure_kinds(tag):
    """Sorts the failed standard-test drives (all are 60 s timeouts) into one group each, from their paths:
      stuck on an obstacle   10 s or more touching an obstacle or wall
      parked at another object   ended within 0.6 m of a goal-type object that is not the goal (and not stuck)
      never saw the goal     the goal was never in the picture during the drive (and none of the above)
      saw it, then lost it   the goal was in the picture at some point but not at the end (and none of the above)
      goal in view at the end   still had it in view when time ran out (slow, or circling near it)
    Returns {group: number of drives}."""
    from .world import sample_episode
    groups = {k: 0 for k in ("stuck on an obstacle", "parked at another object", "never saw the goal",
                             "saw it, then lost it", "goal in view at the end")}
    for e in _room_detail(tag):
        if e["success"]:
            continue
        spec = sample_episode(e["seed"])
        x, y = e["path"][-1][:2]
        near_other = any(math.hypot(ox - x, oy - y) < 0.6 for k, (_s, _c, ox, oy) in enumerate(spec.objects) if k != spec.goal_index)
        ever_seen = e["goal_seen"]
        if e["collision_steps"] >= 100:
            g = "stuck on an obstacle"
        elif near_other:
            g = "parked at another object"
        elif e["goal_visible_end"]:
            g = "goal in view at the end"
        elif not ever_seen:
            g = "never saw the goal"
        else:
            g = "saw it, then lost it"
        groups[g] += 1
    return groups


def failure_kinds_chart(models: dict, ax=None):
    """Stacked bars: how each model's failed standard-test drives failed (needs the --detail logs)."""
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(8, 4))[1]
    kinds = {m: failure_kinds(t) for m, t in models.items()}
    names = list(next(iter(kinds.values())))
    bottom = np.zeros(len(models))
    for k, n in enumerate(names):
        vals = np.array([kinds[m][n] for m in models])
        ax.bar(list(models), vals, bottom=bottom, color=COLOURS[k % len(COLOURS)], label=n)
        for i, v in enumerate(vals):
            if v:
                ax.text(i, bottom[i] + v / 2, str(v), ha="center", va="center", color="w", fontsize=9)
        bottom += vals
    ax.set_ylabel("failed drives (of 128)"); ax.set_title("Standard test: how the failed drives failed", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return ax


def plot_room_drives(tag, only="failed", n=8, cols=4):
    """Maps of standard-test drives (needs the --detail log). Blue = the path driven, red dots = where it was touching
    something, red dotted = the shortest path, white star + dashed circle = the goal and its 'reached' line, yellow wedge
    = what the camera saw at the start. only = "failed" | "succeeded" | "all"."""
    import matplotlib.pyplot as plt
    from .viz import draw_episode
    from .world import sample_episode
    det = [e for e in _room_detail(tag) if only == "all" or e["success"] == (only == "succeeded")][:n]
    if not det:
        print("no such drives"); return None
    rows = math.ceil(len(det) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.4 * rows), squeeze=False)
    for ax, e in zip(axs.flat, det):
        p = np.array(e["path"])
        draw_episode(sample_episode(e["seed"]), ax=ax, trajectory=p[:, :2],
                     title=f'"{e["prompt"]}"\ngoal {"visible" if e["start_visible"] else "hidden"} at start, '
                           f'touching {e["collision_steps"] / 10:.0f} s, goal {"in view" if e["goal_visible_end"] else "out of view"} at the end')
        c = e.get("collisions") or []
        if c:
            ax.plot(p[c, 0], p[c, 1], ".", color="red", ms=3)
        ax.title.set_fontsize(7)
    for ax in list(axs.flat)[len(det):]:
        ax.axis("off")
    fig.suptitle(f"{tag}: standard-test drives ({only})", fontsize=10)
    fig.tight_layout()
    return fig


def plot_drives(tag, test="pair", only="wrong", n=12, cols=6):
    """Top-down paths of words-test drives (needs the --detail log). only = "wrong" | "right" | "all".
    o = start, x = end, thick black ring = the asked-for object, dotted ring = the 'reached' line (0.35 m)."""
    import matplotlib.pyplot as plt
    det = json.load(open(DIAGNOSIS / f"{tag}_which_detail.json"))
    kinds = ["alone"] * 30 + ["pair"] * 180 + ["behind"] * 30 + ["behind+decoy"] * 30
    sel = [(i, d) for i, d in enumerate(det) if kinds[i] == test and
           (only == "all" or (only == "wrong") == (d["reached"] != d["goal_index"]))][:n]
    if not sel:
        print("no such drives"); return None
    rows = math.ceil(len(sel) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(2.8 * cols, 2.9 * rows), squeeze=False)
    for ax, (i, d) in zip(axs.flat, sel):
        p = np.array(d["path"])
        ax.plot(p[:, 0], p[:, 1], "-", c="k", lw=1)
        ax.plot(*p[0, :2], "o", c="k", ms=4); ax.plot(*p[-1, :2], "x", c="m", ms=7)
        for k, (shape, colour, ox, oy) in enumerate(d["objects"]):
            mark = {"sphere": "o", "cube": "s", "cylinder": "h"}.get(shape, "o")
            ax.plot(ox, oy, mark, ms=10, color=OBJ_COLOURS.get(colour, "grey"),
                    mec="k" if k == d["goal_index"] else "none", mew=2.5)
            ax.add_patch(plt.Circle((ox, oy), 0.5, fill=False, ls=":", color="grey"))
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        went = "nothing" if d["reached"] is None else ("asked-for" if d["reached"] == d["goal_index"] else "other object")
        ax.set_title(f'"{d["prompt"]}"\n-> reached {went}', fontsize=7)
    for ax in list(axs.flat)[len(sel):]:
        ax.axis("off")
    fig.suptitle(f"{tag}: {test} drives ({only}). o = start, x = end, black outline = asked-for object", fontsize=9)
    fig.tight_layout()
    return fig
