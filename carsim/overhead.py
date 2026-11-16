"""The overhead ("plan view") camera of notebook 06: a fixed camera high above the near wall, tilted down ~60 degrees.

It plays the role of a ceiling camera in a real room. Its picture is processed OFF the car (by a server), and only a
small summary is sent to the car. The camera never moves with the car; the car is found in its picture by the cyan
roof arrow (CarConfig.roof_marker), which also shows which way the car faces.

Jitter: a real ceiling camera is never mounted at exactly the same spot, and a model trained on one exact view can
memorise pixel positions. So every rendered picture moves the camera by up to +-0.3 m and its aim point by up to
+-0.2 m (a few degrees) at random. The jitter is recorded with each picture so experiments are reproducible.
"""

from __future__ import annotations

import numpy as np

RES = 512                              # pixels; smolVLA resizes every camera picture to 512x512 anyway
POS = np.array([0.0, -5.0, 9.5])       # metres: above the near (-y) wall, 9.5 m up ("steep60", chosen 2026-11-14)
LOOKAT = np.array([0.0, 0.4, 0.0])     # aimed just past the room centre so the far wall stays in frame
FOV = 55                               # vertical field of view (degrees); fixed by NavEnv's overview camera
POS_JITTER = 0.3                       # +- metres on x, y, z
LOOKAT_JITTER = 0.2                    # +- metres on x, y


def sample_jitter(rng: np.random.Generator) -> np.ndarray:
    """6 numbers: camera position offset (x, y, z) and aim-point offset (x, y, 0)."""
    return np.concatenate([rng.uniform(-POS_JITTER, POS_JITTER, 3),
                           [*rng.uniform(-LOOKAT_JITTER, LOOKAT_JITTER, 2), 0.0]])


MARKER_FRONT = (1.0, 0.1, 0.6)         # hot pink front chevron (CarConfig.roof_marker_front), cyan shaft behind


def car_config(two_colour: bool = True):
    """The car as notebook 06 renders it: roof marker on; two colours (pink front, cyan back) unless asked otherwise."""
    from .config import CarConfig
    return CarConfig(roof_marker=True, roof_marker_front=MARKER_FRONT if two_colour else None)


MAX_OBJECTS = 6                        # rooms have 3-6 coloured objects (EnvConfig.n_goal_objects)


def random_free_pose(env, spec, rng: np.random.Generator, clearance: float = 0.3):
    """A random car pose (x, y, yaw) with no wall, object or obstacle within ``clearance`` metres. Used for the
    server's practice pictures: the car can be anywhere, facing anywhere (no driving needed)."""
    from .world import occupancy
    occ = occupancy(env.grid, spec.objects, spec.obstacles)
    lim = env.cfg.arena_size / 2 - 0.4
    while True:
        x, y = rng.uniform(-lim, lim, 2)
        if not (occ & env.grid.circle(x, y, clearance)).any():
            return float(x), float(y), float(rng.uniform(-np.pi, np.pi))


def object_table(spec) -> np.ndarray:
    """The room's coloured objects as a fixed-size array (MAX_OBJECTS, 5): shape index, colour index, x, y, valid.
    Saved with every practice picture, so the training question ("where is the <colour> <shape>?") can be picked
    fresh each time the picture is used."""
    from .world import COLORS, SHAPES
    t = np.zeros((MAX_OBJECTS, 5), np.float32)
    for k, (shape, colour, x, y) in enumerate(spec.objects):
        t[k] = [SHAPES.index(shape), list(COLORS).index(colour), x, y, 1.0]
    return t


def facts(pose, obj_xy) -> dict:
    """The simple facts the server must read from the picture (all from the simulator, free labels):
    car position and heading (room frame), the named object's position, and where that object is FROM THE CAR
    (bearing: + = to the car's left; distance). Works on numpy arrays of shape (..., 3) and (..., 2)."""
    x, y, yaw = pose[..., 0], pose[..., 1], pose[..., 2]
    ox, oy = obj_xy[..., 0], obj_xy[..., 1]
    bearing = (np.arctan2(oy - y, ox - x) - yaw + np.pi) % (2 * np.pi) - np.pi
    return {"car_xy": np.stack([x, y], -1), "heading": np.stack([np.cos(yaw), np.sin(yaw)], -1),
            "obj_xy": np.stack([ox, oy], -1), "rel_dir": np.stack([np.cos(bearing), np.sin(bearing)], -1),
            "rel_dist": np.hypot(ox - x, oy - y)[..., None], "bearing": bearing}


def direction_class(bearing):
    """0 ahead (within 45 deg), 1 left, 2 behind, 3 right."""
    d = np.degrees(bearing)
    return np.where(np.abs(d) < 45, 0, np.where((d >= 45) & (d < 135), 1, np.where(np.abs(d) >= 135, 2, 3)))


def render(env, jitter=None) -> np.ndarray:
    """Overhead picture (RES x RES x 3 uint8) of the env's current state. ``env`` must be a NavEnv built with
    ``overview_camera_res=(RES, RES)`` (and, for notebook 06, ``CarConfig(roof_marker=True)``)."""
    j = np.zeros(6) if jitter is None else np.asarray(jitter, dtype=float)
    env.scene.visualizer.update(force=True)       # the renderer must see the car where it is now
    return env.render_overview(POS + j[:3], LOOKAT + j[3:], up=(0.0, 0.0, 1.0))
