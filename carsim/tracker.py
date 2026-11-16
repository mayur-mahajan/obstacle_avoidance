"""Classic marker tracking of the car in the overhead picture (notebook 06): no learning, just colour and geometry.

The roof marker is the only hot-pink (front chevron) and cyan (back shaft) thing in the room. So:
  1. find the pink and the cyan pixels (colour thresholds in HSV: hue = which colour, saturation = how vivid)
  2. take the centre of each blob in the picture
  3. turn each picture point into a point on the floor plane at the marker's height, using the camera's position
     and aim (a ray from the camera through that pixel, cut by the plane z = marker height)
  4. heading = direction from the cyan centre to the pink centre (front and back have different colours, so it can't
     come out backwards); car centre = the two centres shifted by their known offsets on the roof
This is how multi-robot labs localise robots from an overhead camera (e.g. the Robotarium, 1609.04730: overhead camera
+ printed markers). Learned models are kept for what they are good at (which object, where the obstacles are).

Calibration: a real ceiling camera is fixed and calibrated once. Our pictures move the camera a little at random
(jitter); pass that jitter for a "calibrated" camera, or None to assume the nominal mounting (uncalibrated).
"""

from __future__ import annotations

import math

import numpy as np

from . import overhead

MARKER_Z = 0.113                 # m above the floor (roof of the chassis; CarConfig: chassis_z + height/2 + 3 mm)
CYAN_OFFSET = -0.05              # m along the car's length: centre of the cyan shaft
PINK_OFFSET = 0.067              # m: centre of the pink chevron (tip 0.12 m, arms 0.15 m at 45 deg)


def _camera(jitter, res):
    j = np.zeros(6) if jitter is None else np.asarray(jitter, float)
    pos, look = overhead.POS + j[:3], overhead.LOOKAT + j[3:]
    f = look - pos; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    fpx = 0.5 * res / math.tan(math.radians(overhead.FOV) / 2)
    return pos, f, r, u, fpx


def pixel_to_floor(uv, jitter=None, res: int = 1024, z: float = MARKER_Z):
    """Picture points (n, 2) (column, row) -> room points (n, 2) on the plane at height z."""
    pos, f, r, u, fpx = _camera(jitter, res)
    uv = np.atleast_2d(np.asarray(uv, float))
    xc, yc = (uv[:, 0] - res / 2) / fpx, -(uv[:, 1] - res / 2) / fpx
    d = xc[:, None] * r + yc[:, None] * u + f                      # ray directions
    t = (z - pos[2]) / d[:, 2]
    return (pos + t[:, None] * d)[:, :2]


def _masks(pic):
    import cv2
    hsv = cv2.cvtColor(np.ascontiguousarray(pic), cv2.COLOR_RGB2HSV)  # OpenCV hue: 0-180 (= degrees / 2)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1], hsv[..., 2]
    pink = (h >= 155) & (h <= 175) & (s >= 110) & (v >= 90)        # (1.0, 0.1, 0.6): hue ~330 deg
    cyan = (h >= 80) & (h <= 100) & (s >= 110) & (v >= 70)          # (0.0, 0.95, 0.95): hue 180 deg
    return pink, cyan


def _blobs(mask, min_px: int = 4):
    import cv2
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [(int(stats[i, cv2.CC_STAT_AREA]), cent[i]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_px]


def track(pic, jitter=None):
    """-> dict(ok, x, y, yaw, n_pink, n_cyan). ok=False when the marker isn't found (e.g. hidden behind a box)."""
    res = pic.shape[0]
    pink, cyan = _masks(pic)
    pb, cb = _blobs(pink), _blobs(cyan)
    out = {"ok": False, "x": np.nan, "y": np.nan, "yaw": np.nan, "n_pink": int(pink.sum()), "n_cyan": int(cyan.sum())}
    if not pb or not cb:
        return out
    # the marker's two halves are next to each other: the closest (pink, cyan) pair, preferring big blobs
    best = min(((np.linalg.norm(p[1] - c[1]) - 0.2 * math.sqrt(p[0] + c[0]), p, c) for p in pb for c in cb),
               key=lambda t: t[0])
    _, p, c = best
    if np.linalg.norm(p[1] - c[1]) > 60:                            # too far apart to be one marker
        return out
    (px, py), (cx, cy) = pixel_to_floor(np.stack([p[1], c[1]]), jitter, res)
    yaw = math.atan2(py - cy, px - cx)
    ux, uy = math.cos(yaw), math.sin(yaw)
    x = 0.5 * ((px - PINK_OFFSET * ux) + (cx - CYAN_OFFSET * ux))
    y = 0.5 * ((py - PINK_OFFSET * uy) + (cy - CYAN_OFFSET * uy))
    out.update(ok=True, x=x, y=y, yaw=yaw)
    return out


# ------------------------------------------------------------------------------------------------------
# The goal object: the server's model says WHICH object (it reads the instruction and gives a rough position);
# colour + geometry say exactly WHERE. Goal objects come in vivid colours, unlike the grey floors and obstacles.
# ------------------------------------------------------------------------------------------------------
HUES = {"red": [(0, 7), (173, 180)], "orange": [(10, 20)], "yellow": [(22, 33)], "green": [(50, 75)],
        "blue": [(104, 121)], "purple": [(128, 150)]}          # OpenCV hue (degrees / 2)
OBJECT_Z = 0.2                   # m: the visible part of a 30 cm object seen from above-front (top + upper side)


def colour_from_prompt(prompt: str) -> str | None:
    words = prompt.lower().replace(".", " ").split()
    return next((c for c in HUES if c in words), None)


def locate_goal(pic, jitter, prompt, rough_xy, car_xy=None, max_jump: float = 1.2):
    """Refine the server's rough goal position: the blob of the instruction's colour closest to it (lookalikes share the
    colour but not the place). Returns ((x, y), found)."""
    import cv2
    colour = colour_from_prompt(prompt)
    if colour is None:
        return tuple(rough_xy), False
    hsv = cv2.cvtColor(np.ascontiguousarray(pic), cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1], hsv[..., 2]
    m = np.zeros(h.shape, bool)
    for lo, hi in HUES[colour]:
        m |= (h >= lo) & (h <= hi)
    m &= (s >= 100) & (v >= 60)
    blobs = _blobs(m, min_px=30)
    if not blobs:
        return tuple(rough_xy), False
    pts = pixel_to_floor(np.stack([b[1] for b in blobs]), jitter, pic.shape[0], z=OBJECT_Z)
    if car_xy is not None:                                           # the car's blue body is not a blue goal
        keep = np.hypot(pts[:, 0] - car_xy[0], pts[:, 1] - car_xy[1]) > 0.35
        pts = pts[keep]
        if not len(pts):
            return tuple(rough_xy), False
    d = np.hypot(pts[:, 0] - rough_xy[0], pts[:, 1] - rough_xy[1])
    k = int(np.argmin(d))
    if d[k] > max_jump:
        return tuple(rough_xy), False
    return (float(pts[k, 0]), float(pts[k, 1])), True


def floor_to_pixel(xyz, jitter=None, res: int = 1024):
    """Room points (n, 3) -> picture points (n, 2) (column, row): the inverse of pixel_to_floor (for drawing)."""
    pos, f, r, u, fpx = _camera(jitter, res)
    p = np.atleast_2d(np.asarray(xyz, float)) - pos
    xc, yc, zc = p @ r, p @ u, p @ f
    return np.stack([res / 2 + fpx * xc / zc, res / 2 - fpx * yc / zc], -1)
