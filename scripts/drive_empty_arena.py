"""Drive the mecanum car with the keyboard in the simple arena of notebook 00 (3 fixed landmarks).

Run from the project root:
    .venv/bin/python scripts/drive_empty_arena.py            # just drive
    .venv/bin/python scripts/drive_empty_arena.py --log runs/drive_log.npz   # also save commands/poses for plotting

Two windows open:
  * "Genesis" 3-D viewer with a chase camera  -> give it keyboard focus to drive
  * "FPV"  what the car's camera sees (this is what the AI model will see)
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

from carsim.arena import add_demo_objects, add_walled_arena  # noqa: E402
from carsim.car import MecanumCar  # noqa: E402
from carsim.config import CarConfig, SimConfig  # noqa: E402
from carsim.teleop import HELP_TEXT, KeyboardCommander  # noqa: E402

START_POSE = (-2.5, 0.0, 0.0)


def chase_camera(viewer, pose, dist=1.6, height=0.9):
    x, y, yaw = pose
    pos = (x - dist * math.cos(yaw), y - dist * math.sin(yaw), height)
    lookat = (x + 0.6 * math.cos(yaw), y + 0.6 * math.sin(yaw), 0.1)
    # Pass a full pose built with the *world* up axis. (Passing only pos/lookat makes Genesis reuse
    # the camera's previous up-vector, so the view slowly rolls as the chase camera swings around.)
    pose = gu.pos_lookat_up_to_T(np.array(pos), np.array(lookat), np.array([0.0, 0.0, 1.0]))
    viewer.set_camera_pose(pose=pose)


def fpv_window(rgb: np.ndarray, command: np.ndarray, scale: int = 3) -> None:
    img = cv2.resize(rgb[..., ::-1], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    hud = f"vx {command[0]:+.2f} m/s  vy {command[1]:+.2f} m/s  wz {command[2]:+.2f} rad/s"
    cv2.putText(img, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("FPV", img)
    cv2.waitKey(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, default=None, help="save commands and poses to this .npz file")
    args = ap.parse_args()

    car_cfg, sim_cfg = CarConfig(), SimConfig()
    from carsim.device import sim_backend
    gs.init(backend=sim_backend(), logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=sim_cfg.sim_dt),
        viewer_options=gs.options.ViewerOptions(
            res=(1100, 750), camera_fov=50, enable_default_keybinds=False, realtime_factor=None
        ),
        show_viewer=True,
        renderer=gs.renderers.Rasterizer(),
    )
    add_walled_arena(scene)
    add_demo_objects(scene)
    car = MecanumCar(scene, car_cfg, sim_cfg)
    print("Building scene (the first run compiles GPU kernels, this can take ~1 minute)...")
    scene.build()
    car.on_built()
    car.reset(*START_POSE)

    driver = KeyboardCommander(car_cfg, sim_cfg)
    scene.viewer.register_keybinds(*driver.keybinds())
    print(HELP_TEXT)

    log = {"t": [], "command": [], "target": [], "pose": [], "velocity": []}
    # We pace the *control loop* ourselves (exactly control_hz ticks per second of wall-clock time)
    # instead of letting Genesis pace each physics step: the camera rendering and display take time
    # too, and only pacing the whole loop gives a steady 10 Hz in the recorded data.
    period = 1.0 / sim_cfg.control_hz
    t0 = time.time()
    next_tick = time.perf_counter()
    try:
        while scene.viewer.is_alive() and not driver.quit_requested:
            if driver.reset_requested:
                car.reset(*START_POSE)
                driver.command[:] = 0.0
                driver.reset_requested = False
            target = driver.target()
            cmd = driver.update()
            car.step(*cmd)
            chase_camera(scene.viewer, car.pose())
            fpv_window(car.render_fpv(), cmd)
            if args.log:
                log["t"].append(time.time() - t0)
                log["command"].append(cmd)
                log["target"].append(target)
                log["pose"].append(car.pose())
                log["velocity"].append(car.body_velocity())
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
