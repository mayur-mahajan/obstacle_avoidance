"""Drive the mecanum car with the keyboard through random navigation episodes.

Run from the project root:
    .venv/bin/python scripts/teleop.py                    # random episodes
    .venv/bin/python scripts/teleop.py --seed 7           # start from a specific episode
    .venv/bin/python scripts/teleop.py --log runs/drive_log.npz   # also save commands/poses for plotting

Two windows open:
  * "Genesis" 3-D viewer with a chase camera  -> give it keyboard focus to drive
  * "FPV"  what the car's camera sees (what the AI model will see), with the goal prompt and status
An episode ends on success (bumper within 30 cm of the goal and stopped) or after 60 s.
(Recording datasets and typing your own prompt come in notebook 02.)
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")  # OpenCV's bundled Qt has no Wayland plugin; use X11
import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import genesis as gs  # noqa: E402
import genesis.utils.geom as gu  # noqa: E402

from carsim.env import NavEnv  # noqa: E402
from carsim.teleop import HELP_TEXT, KeyboardCommander  # noqa: E402


def chase_camera(viewer, pose, dist=1.6, height=0.9):
    x, y, yaw = pose
    pos = (x - dist * math.cos(yaw), y - dist * math.sin(yaw), height)
    lookat = (x + 0.6 * math.cos(yaw), y + 0.6 * math.sin(yaw), 0.1)
    # Pass a full pose built with the *world* up axis. (Passing only pos/lookat makes Genesis reuse
    # the camera's previous up-vector, so the view slowly rolls as the chase camera swings around.)
    pose = gu.pos_lookat_up_to_T(np.array(pos), np.array(lookat), np.array([0.0, 0.0, 1.0]))
    viewer.set_camera_pose(pose=pose)


def put(img, text, y, colour=(255, 255, 255), size=0.6, x=8, thick=1):
    """Outlined text: black copies shifted a few pixels in 8 directions, same thickness (in OpenCV 5 a
    thicker black copy would also be wider and show up as a misaligned ghost)."""
    r = 2 if size > 0.8 else 1
    for dx in (-r, 0, r):
        for dy in (-r, 0, r):
            if dx or dy:
                cv2.putText(img, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, colour, thick, cv2.LINE_AA)


def fpv_window(rgb, command, info, banner=None, scale=3):
    img = cv2.resize(rgb[..., ::-1], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    if info["collision"]:
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 0, 255), 10)
    lines = [
        f'GOAL: "{info["prompt"]}"',
        f"t {info['t'] / 10:5.1f}s   dist {info['goal_surface_dist']:.2f} m   collisions {info['n_collision_steps']}",
        f"vx {command[0]:+.2f}  vy {command[1]:+.2f}  wz {command[2]:+.2f}",
    ]
    for k, line in enumerate(lines):
        put(img, line, 24 + 22 * k)
    if banner:
        put(img, banner, img.shape[0] // 2, colour=(80, 255, 80), size=1.6, x=40, thick=2)
    cv2.imshow("FPV", img)
    cv2.waitKey(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None, help="seed of the first episode (default: random)")
    ap.add_argument("--log", type=str, default=None, help="save commands and poses to this .npz file")
    args = ap.parse_args()

    print("Building scene (the first run compiles GPU kernels, this can take ~1 minute)...")
    env = NavEnv(show_viewer=True)
    driver = KeyboardCommander(env.car_cfg, env.sim_cfg)
    env.scene.viewer.register_keybinds(*driver.keybinds())
    print(HELP_TEXT)

    rng = np.random.default_rng(args.seed)
    seed = args.seed if args.seed is not None else int(rng.integers(2**31 - 1))
    obs, info = env.reset(seed)
    print(f"[episode seed {seed}] {info['prompt']}")

    log = {"t": [], "command": [], "target": [], "pose": [], "velocity": [], "episode": [], "seed": []}
    episode_no = 0
    period = 1.0 / env.sim_cfg.control_hz
    t0 = time.time()
    next_tick = time.perf_counter()
    try:
        while env.scene.viewer.is_alive() and not driver.quit_requested:
            if driver.reset_requested or driver.new_episode_requested:
                if driver.new_episode_requested:
                    seed = int(rng.integers(2**31 - 1))
                obs, info = env.reset(seed)
                driver.command[:] = 0.0
                driver.held.clear()
                driver.reset_requested = driver.new_episode_requested = False
                episode_no += 1
                print(f"[episode seed {seed}] {info['prompt']}")
            target = driver.target()
            cmd = driver.update()
            obs, terminated, truncated, info = env.step(cmd)
            chase_camera(env.scene.viewer, info["pose"])
            banner = "SUCCESS!" if terminated else ("TIME UP" if truncated else None)
            fpv_window(obs["image"], cmd, info, banner)
            if args.log:
                log["t"].append(time.time() - t0)
                log["command"].append(cmd)
                log["target"].append(target)
                log["pose"].append(info["pose"])
                log["velocity"].append(info["velocity"])
                log["episode"].append(episode_no)
                log["seed"].append(seed)
            if terminated or truncated:
                print(f"  -> {'success' if terminated else 'timeout'} after {info['t'] / 10:.1f} s, "
                      f"path {info['path_length']:.1f} m (shortest {env.spec.geodesic_start:.1f} m), "
                      f"collision steps {info['n_collision_steps']}")
                time.sleep(1.5)
                driver.new_episode_requested = True
            next_tick += period
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:  # we fell behind (e.g. a slow frame): don't try to "catch up" with a burst of steps
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        print("Interrupted.")
    except gs.GenesisException as e:  # closing the viewer window ends the loop
        print(f"Viewer closed ({e}).")
    finally:
        cv2.destroyAllWindows()
        if args.log and log["t"]:
            Path(args.log).parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.log, **{k: np.asarray(v) for k, v in log.items()})
            print(f"Saved {len(log['t'])} control steps to {args.log}")


if __name__ == "__main__":
    main()
