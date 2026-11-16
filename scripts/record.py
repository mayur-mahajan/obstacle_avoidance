"""Record driving demonstrations into a LeRobot dataset.

Run from the project root (with the project's Python, e.g. `uv run python` or .venv/bin/python):
    .venv/bin/python scripts/record.py                          # you drive, default dataset runs/datasets/human_demos
    .venv/bin/python scripts/record.py --target 50              # stop automatically after 50 kept episodes
    .venv/bin/python scripts/record.py --expert --episodes 20   # the scripted expert drives (no windows, fast)
    .venv/bin/python scripts/record.py --expert --cpu --search-go --finish human --words --words-mix search \
        --dataset runs/datasets/expert_sgc_words_1 --episodes 0 --seed-list assets/scripted_layouts/words_1.jsonl
                       # the scripted training data (notebook 02, section 10): "words" layouts, where only the
                       # instruction tells which object is the goal (carsim/words.py). A side-by-side pair is always
                       # followed by the SAME layout asking for the other object.

Each episode goes through three phases:
  READY     the sim is paused. Read the prompt; press T to retype it if you like. Start driving to start
            recording (idle time before your first key press is NOT recorded).
  DRIVING   every 0.1 s one frame (camera image, speed, your command) is buffered in memory.
            Ends on success (stopped within 30 cm of the goal) or after 60 s.
  REVIEW    the sim is paused again. K = keep (writes the episode to disk), D = discard.
            Only successful episodes can be kept. The HUD shows how much of the episode you
            stood still ("pauses") and strafed - see notebook 02 section 7 for why that matters.

Keys (the 3-D "Genesis" window must have keyboard focus):
  arrows / Q / E / Space   drive, as in teleop.py
  T  type a new prompt (READY only)        K  keep episode (REVIEW)        D  discard episode (REVIEW)
  R  restart this layout (drops the current attempt)
  N  skip to a new layout (drops the current attempt)
  Esc  save everything kept so far and quit
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")  # OpenCV's bundled Qt has no Wayland plugin; use X11
import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import genesis as gs  # noqa: E402
from genesis.vis.keybindings import Key, KeyAction, Keybind  # noqa: E402

from carsim.env import NavEnv  # noqa: E402
from carsim.expert import PathFollowingExpert, SearchThenGoExpert  # noqa: E402
from carsim.recording import EpisodeRecord, append_log, make_frame, open_dataset, save_episode  # noqa: E402
from carsim.teleop import KeyboardCommander  # noqa: E402
from carsim.words import MIXES, words_kind, words_seed  # noqa: E402


class SeedPicker:
    """Next layout seed. Normal: a random seed. --words: a words seed, and after the first half of a
    pair (asking for object A) the second half (same layout, asking for object B)."""

    def __init__(self, rng, words: bool, mix: str = "standard"):
        self.rng, self.words, self.mix, self.twin = rng, words, mix, None

    def next(self) -> int:
        if not self.words:
            return int(self.rng.integers(2**31 - 1))
        if self.twin is not None:
            seed, self.twin = self.twin, None
        else:
            seed = words_seed(int(self.rng.integers(2**40)), mix=self.mix)
            if words_kind(seed) == "pair":
                self.twin = seed + 1
        return seed

    @staticmethod
    def describe(seed: int) -> str:
        if seed < 10**12:
            return ""
        kind = words_kind(seed)
        return f"[{kind}{' - 2nd half: same layout, other object' if kind == 'pair' and seed % 2 else ''}]"

GREEN, RED, YELLOW, WHITE = (80, 255, 80), (60, 60, 255), (0, 220, 255), (255, 255, 255)


# ---------------------------------------------------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------------------------------------------------
def put(img, text, y, colour=WHITE, size=0.6, x=8, thick=1):
    """Text with a black outline. The outline is the same text drawn in black, shifted by a couple
    of pixels in 8 directions, with the SAME thickness. (A thicker black copy doesn't work: in
    OpenCV 5 thicker strokes also make the letters wider, so the copy sticks out as a ghost.)"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    r = 2 if size > 0.8 else 1
    for dx in (-r, 0, r):
        for dy in (-r, 0, r):
            if dx or dy:
                cv2.putText(img, text, (x + dx, y + dy), font, size, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, size, colour, thick, cv2.LINE_AA)


def show_fpv(rgb, header, footer=(), border=None, scale=3):
    img = cv2.resize(rgb[..., ::-1], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    if border is not None:
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), border, 10)
    for k, (text, colour) in enumerate(header):
        put(img, text, 26 + 24 * k, colour)
    for k, (text, colour, size) in enumerate(footer):
        put(img, text, img.shape[0] // 2 + 10 + 36 * k, colour, size, x=24, thick=2 if size > 0.8 else 1)
    cv2.imshow("FPV", img)
    cv2.waitKey(1)


def ask_prompt(current: str, goal_colour: str, goal_shape: str) -> str:
    """Pop up a text box to retype the prompt; returns the (possibly unchanged) prompt.

    The box runs in a separate process (see prompt_dialog.py for why), preferably with the system
    Python: the Tk library bundled with uv's Python crashes when drawing text on some Linux setups."""
    import shutil
    import subprocess

    script = str(Path(__file__).with_name("prompt_dialog.py"))
    for py in dict.fromkeys(p for p in ("/usr/bin/python3", shutil.which("python3"), sys.executable) if p):
        res = subprocess.run([py, script, current, goal_colour, goal_shape], capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    print("  (could not open the prompt window - keeping the suggested prompt)", flush=True)
    return current


def chase_camera(viewer, pose, dist=1.6, height=0.9):
    import genesis.utils.geom as gu

    x, y, yaw = pose
    pos = (x - dist * math.cos(yaw), y - dist * math.sin(yaw), height)
    lookat = (x + 0.6 * math.cos(yaw), y + 0.6 * math.sin(yaw), 0.1)
    viewer.set_camera_pose(pose=gu.pos_lookat_up_to_T(np.array(pos), np.array(lookat), np.array([0.0, 0.0, 1.0])))


def style_hud(frames, cmd):
    """Live driving-style line: a PAUSED warning while standing still, plus running pause/strafe shares."""
    pause, strafe = driving_style(frames)
    line = f"pauses {pause:4.0%}   strafing {strafe:4.0%}"
    if np.abs(cmd).max() < 1e-3:
        return [("PAUSED - keep moving!  " + line, YELLOW)]
    return [(line, YELLOW if pause > PAUSE_HINT else WHITE)]


class Flags:
    """One-shot flags set from key callbacks (which run on the viewer's thread)."""

    def __init__(self):
        self.keep = self.discard = self.edit_prompt = False

    def set(self, name):
        setattr(self, name, True)

    def pop(self, name) -> bool:
        v = getattr(self, name)
        setattr(self, name, False)
        return v


# ---------------------------------------------------------------------------------------------------
PAUSE_HINT = 0.10  # REVIEW suggests discarding when more than this fraction of the episode was standing still


def driving_style(frames) -> tuple[float, float]:
    """(pause fraction, strafe fraction) of an episode.
    A *pause* is a frame whose action is all zeros: standing still mid-episode. These teach the model
    that stopping is a fine answer anywhere, which can make it freeze, so we want few of them.
    A *strafe* frame uses sideways velocity (|vy| > 0.01 m/s)."""
    if not frames:
        return 0.0, 0.0
    a = np.array([f["action"] for f in frames])
    return float((np.abs(a).max(axis=1) < 1e-3).mean()), float((np.abs(a[:, 1]) > 0.01).mean())


def episode_summary(env, info, record, outcome):
    geo = env.spec.geodesic_start
    pause, strafe = driving_style(record.frames)
    return {
        "seed": record.seed,
        "goal": env.spec.goal_name,
        "suggested_prompt": record.suggested_prompt,
        "prompt": record.prompt,
        "outcome": outcome,  # success / timeout / restarted / skipped / quit
        "n_frames": len(record.frames),
        "duration_s": len(record.frames) / env.sim_cfg.control_hz,
        "collision_steps": int(info["n_collision_steps"]),
        "path_length": float(info["path_length"]),
        "shortest_path": float(geo),
        "spl": float(geo / max(geo, info["path_length"])) if outcome == "success" else 0.0,
        "start_visible": bool(env.spec.start_visible),
        "hard_distractors": bool(env.spec.hard_distractors),
        "pause_frac": pause,
        "strafe_frac": strafe,
    }


def run_expert(env, ds, args, rng):
    """The scripted expert drives. Successful, collision-free episodes are kept automatically."""
    kept = tried = 0
    picker = SeedPicker(rng, args.words, args.words_mix)
    seeds = None
    if args.seed_list:   # same layouts as an earlier dataset (only what the expert does changes)
        seeds = [json.loads(l)["seed"] for l in open(args.seed_list) if json.loads(l).get("kept")]
        args.episodes = min(args.episodes, len(seeds)) if args.episodes else len(seeds)
    while kept < args.episodes:
        if seeds is not None and tried >= len(seeds):
            break
        seed = seeds[tried] if seeds is not None else picker.next()
        obs, info = env.reset(seed)
        expert = (SearchThenGoExpert(env, finish=args.finish) if args.search_go else PathFollowingExpert(env, finish=args.finish))
        record = EpisodeRecord(seed, env.spec.prompt, env.spec.prompt)
        while True:
            action = expert.act(info)
            record.frames.append(make_frame(obs, action, info, env.spec, record.prompt))
            obs, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        tried += 1
        row = episode_summary(env, info, record, "success" if terminated else "timeout")
        row["source"] = "expert_search_go" if args.search_go else "expert"
        row["kept"] = bool(terminated and info["n_collision_steps"] == 0)
        if row["kept"]:
            row["episode_index"] = save_episode(ds, record)
            kept += 1
        append_log(args.dataset, row)
        print(f"[{tried:3d}] seed {seed:10d}  {row['outcome']:7s}  {row['duration_s']:5.1f}s  "
              f"collisions {row['collision_steps']:2d}  {'KEPT' if row['kept'] else 'dropped'}  ({kept}/{args.episodes})",
              flush=True)


def run_human(env, ds, args, rng):
    driver = KeyboardCommander(env.car_cfg, env.sim_cfg)
    flags = Flags()
    binds = driver.keybinds() + [
        Keybind("keep", Key.K, KeyAction.PRESS, callback=flags.set, args=("keep",)),
        Keybind("discard", Key.D, KeyAction.PRESS, callback=flags.set, args=("discard",)),
        Keybind("edit_prompt", Key.T, KeyAction.PRESS, callback=flags.set, args=("edit_prompt",)),
    ]
    env.scene.viewer.register_keybinds(*binds)
    print(__doc__.split("Keys")[1])

    kept = ds.meta.total_episodes
    start_kept = kept
    period = 1.0 / env.sim_cfg.control_hz

    picker = SeedPicker(rng, args.words, args.words_mix)

    def new_layout(seed=None):
        seed = picker.next() if seed is None else seed
        obs, info = env.reset(seed)
        driver.command[:] = 0.0
        driver.held.clear()
        for f in ("reset_requested", "new_episode_requested", "stop_requested"):
            setattr(driver, f, False)
        for f in ("keep", "discard", "edit_prompt"):
            flags.pop(f)
        print(f"\n[seed {seed}] {SeedPicker.describe(seed)} goal: {env.spec.goal_name}   prompt: \"{env.spec.prompt}\"", flush=True)
        return obs, info, EpisodeRecord(seed, env.spec.prompt, env.spec.prompt)

    obs, info, record = new_layout(args.seed)
    phase = "READY"
    outcome = None
    next_tick = time.perf_counter()
    try:
        while env.scene.viewer.is_alive() and not driver.quit_requested:
            progress = f"kept {kept}" + (f"/{args.target}" if args.target else "") + f"  (this session {kept - start_kept})"
            # ---------- keys that work in every phase ----------
            if driver.reset_requested or driver.new_episode_requested:
                same = driver.reset_requested
                if phase in ("DRIVING", "REVIEW") and record.frames:  # the attempt in progress is dropped
                    reason = outcome if phase == "REVIEW" else ("restarted" if same else "skipped")
                    append_log(args.dataset, {**episode_summary(env, info, record, reason), "source": "human", "kept": False})
                    print("  attempt dropped", flush=True)
                typed = record.prompt
                obs, info, record = new_layout(record.seed if same else None)
                if same:  # R keeps a prompt you typed for this layout
                    record.prompt = env.spec.prompt = typed
                phase, outcome = "READY", None

            if phase == "READY":
                if flags.pop("edit_prompt"):
                    shape, colour = env.spec.goal[:2]
                    record.prompt = ask_prompt(record.prompt, colour, shape)
                    env.spec.prompt = record.prompt  # so the HUD shows it
                    driver.held.clear()
                    print(f"  prompt set to: \"{record.prompt}\"  (click the 3-D window again to drive)", flush=True)
                if np.any(driver.target() != 0):
                    phase = "DRIVING"
                else:
                    show_fpv(obs["image"],
                             [(f'GOAL: "{record.prompt}"', WHITE), (progress, WHITE)],
                             [("READY - drive to start recording", YELLOW, 0.75),
                              ("T: retype prompt   N: other layout", WHITE, 0.6)])
                    # The sim is paused here, so nothing else updates the 3-D window. Where the viewer has no thread of
                    # its own (macOS) it only redraws and receives key presses inside update(): without this call the
                    # window is frozen and ignores the keys in READY.
                    env.scene.viewer.update(force=True)
                    time.sleep(1 / 30)
                    next_tick = time.perf_counter()
                    continue

            if phase == "DRIVING":
                cmd = driver.update()
                record.frames.append(make_frame(obs, cmd, info, env.spec, record.prompt))
                obs, terminated, truncated, info = env.step(cmd)
                chase_camera(env.scene.viewer, info["pose"])
                show_fpv(obs["image"],
                         [(f'GOAL: "{record.prompt}"', WHITE),
                          (f"REC  t {info['t'] / 10:4.1f}s  dist {info['goal_surface_dist']:.2f} m  "
                           f"collisions {info['n_collision_steps']}", RED),
                          (f"vx {cmd[0]:+.2f}  vy {cmd[1]:+.2f}  wz {cmd[2]:+.2f}", WHITE),
                          *style_hud(record.frames, cmd)],
                         border=RED if info["collision"] else None)
                if terminated or truncated:
                    outcome = "success" if terminated else "timeout"
                    phase = "REVIEW"
                    flags.pop("keep"), flags.pop("discard")
                    s = episode_summary(env, info, record, outcome)
                    print(f"  -> {outcome} after {s['duration_s']:.1f} s, path {s['path_length']:.1f} m "
                          f"(shortest {s['shortest_path']:.1f} m), collision steps {s['collision_steps']}, "
                          f"pauses {s['pause_frac']:.0%}, strafing {s['strafe_frac']:.0%}", flush=True)
                next_tick += period
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:  # fell behind: don't burst-step to catch up
                    next_tick = time.perf_counter()
                continue

            if phase == "REVIEW":
                s = episode_summary(env, info, record, outcome)
                ok = outcome == "success"
                lines = [("SUCCESS!" if ok else "TIME UP", GREEN if ok else RED, 1.4),
                         (f"{s['duration_s']:.1f} s   path {s['path_length']:.1f} m (best {s['shortest_path']:.1f})", WHITE, 0.6),
                         (f"collision steps: {s['collision_steps']}", RED if s["collision_steps"] else WHITE, 0.6),
                         (f"pauses: {s['pause_frac']:.0%}   strafing: {s['strafe_frac']:.0%}",
                          YELLOW if s["pause_frac"] > PAUSE_HINT else WHITE, 0.6)]
                if ok:
                    lines.append(("K: keep    D: discard    R: retry layout", YELLOW, 0.7))
                    if s["collision_steps"]:
                        lines.append(("(it bumped into something: consider D)", YELLOW, 0.55))
                    elif s["pause_frac"] > PAUSE_HINT:
                        lines.append((f"(stood still {s['pause_frac']:.0%} of the time: consider D)", YELLOW, 0.55))
                else:
                    lines.append(("failed episodes are not kept - D or N to go on", YELLOW, 0.6))
                show_fpv(obs["image"], [(f'GOAL: "{record.prompt}"', WHITE), (progress, WHITE)], lines)
                env.scene.viewer.update(force=True)  # sim paused: keep the 3-D window alive for the K / D keys (macOS)
                keep, discard = flags.pop("keep"), flags.pop("discard")
                if (keep and ok) or discard:
                    row = {**s, "source": "human", "kept": bool(keep and ok)}
                    if row["kept"]:
                        show_fpv(obs["image"], [(f'GOAL: "{record.prompt}"', WHITE)], [("saving...", YELLOW, 1.0)])
                        row["episode_index"] = save_episode(ds, record)
                        kept += 1
                        print(f"  KEPT as episode {row['episode_index']}  ({progress.split('(')[0].strip()} -> {kept})", flush=True)
                    else:
                        print("  discarded", flush=True)
                    append_log(args.dataset, row)
                    if args.target and kept >= args.target and picker.twin is None:  # never stop between the two halves of a pair
                        print(f"\nTarget of {args.target} kept episodes reached.", flush=True)
                        break
                    obs, info, record = new_layout()
                    phase = "READY"
                time.sleep(1 / 30)
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        print("Interrupted.")
    except gs.GenesisException as e:  # closing the 3-D window ends the session
        print(f"Viewer closed ({e}).")
    finally:
        if phase in ("DRIVING", "REVIEW") and record.frames:
            append_log(args.dataset, {**episode_summary(env, info, record, outcome or "quit"), "source": "human",
                                      "kept": False, "note": "session ended before keep/discard"})
        cv2.destroyAllWindows()
    print(f"Session finished: {kept - start_kept} episodes kept this session, {kept} in total.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=ROOT / "runs" / "datasets" / "human_demos",
                    help="dataset folder (created if missing, appended to if it exists)")
    ap.add_argument("--repo-id", default=None, help="LeRobot dataset id (default: local/<folder name>)")
    ap.add_argument("--target", type=int, default=0, help="human mode: stop after this many kept episodes in total")
    ap.add_argument("--seed", type=int, default=None, help="seed for the random layouts (default: random)")
    ap.add_argument("--expert", action="store_true", help="let the scripted expert drive instead of you")
    ap.add_argument("--episodes", type=int, default=10, help="expert mode: number of episodes to keep")
    ap.add_argument("--cpu", action="store_true", help="expert mode: physics on the CPU (several recorders can run in parallel)")
    ap.add_argument("--search-go", action="store_true",
                    help="expert mode: rotate left until the goal is in view, then drive to it (copyable by a camera-only robot)")
    ap.add_argument("--words-mix", default="standard", choices=list(MIXES),
                    help="--words scene mix: standard (40%% pairs) or search (50%% goal-behind-decoy scenes)")
    ap.add_argument("--finish", default="standard", choices=["standard", "human"],
                    help="expert mode: standard (stops ~0.30 m, top speed 0.6) or human (a human speed profile, stops ~0.12 m)")
    ap.add_argument("--seed-list", type=Path, default=None,
                    help="expert mode: drive exactly these layouts - the kept seeds of an earlier *_attempts.jsonl")
    ap.add_argument("--words", action="store_true", help="layouts where the words decide the goal (carsim/words.py)")
    args = ap.parse_args()
    args.repo_id = args.repo_id or f"local/{args.dataset.name}"

    print("Building scene (the first run compiles GPU kernels, this can take ~1 minute)...", flush=True)
    env = NavEnv(show_viewer=not args.expert, **({"backend": "cpu"} if args.cpu else {}))
    ds = open_dataset(args.dataset, args.repo_id)
    print(f"Dataset {args.dataset}: {ds.meta.total_episodes} episodes / {ds.meta.total_frames} frames so far", flush=True)
    rng = np.random.default_rng(args.seed)
    try:
        if args.expert:
            run_expert(env, ds, args, rng)
        else:
            run_human(env, ds, args, rng)
    finally:
        ds.finalize()  # writes the parquet footers - without this the files are unreadable
        print(f"Dataset now has {ds.meta.total_episodes} episodes / {ds.meta.total_frames} frames.", flush=True)


if __name__ == "__main__":
    main()
