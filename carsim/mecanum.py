"""Mecanum-wheel maths and the car's URDF (robot description file).

Two pieces live here:

1. ``wheel_speeds`` - the standard mecanum *inverse kinematics*: given the body velocity we want
   (vx forward, vy left, wz counter-clockwise), how fast must each wheel spin? On a real robot this
   is exactly the function you would run on the microcontroller. In our simulation the body is
   driven directly (see car.py) and this function only animates the wheels, but keeping it here
   documents how the command maps onto real hardware.

2. ``build_urdf`` - writes the robot description. The base is a chain of three joints
   (slide-x, slide-y, rotate-z) so the chassis can move in any planar direction, which is exactly
   the freedom a mecanum base gives you.
"""

import math
import os
from pathlib import Path

import numpy as np

from .config import CarConfig

# Wheel order used everywhere: front-left, front-right, rear-left, rear-right.
WHEEL_NAMES = ("fl", "fr", "rl", "rr")


def wheel_speeds(vx: float, vy: float, wz: float, cfg: CarConfig) -> np.ndarray:
    """Mecanum inverse kinematics -> wheel angular speeds in rad/s (positive = rolls forward).

    Each wheel's rollers sit at 45 degrees, so a wheel can only push the car along a diagonal.
    Adding/subtracting the four diagonals gives forward, sideways and rotation:

        w_fl = (vx - vy - k*wz) / r        w_fr = (vx + vy + k*wz) / r
        w_rl = (vx + vy - k*wz) / r        w_rr = (vx - vy + k*wz) / r      with k = lx + ly
    """
    k = cfg.wheelbase_half + cfg.track_half
    r = cfg.wheel_radius
    return np.array(
        [vx - vy - k * wz, vx + vy + k * wz, vx + vy - k * wz, vx - vy + k * wz], dtype=np.float64
    ) / r


def body_to_world(vx: float, vy: float, yaw: float) -> tuple[float, float]:
    """Rotate a velocity from the car's frame (forward/left) into the world frame (x/y)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return c * vx - s * vy, s * vx + c * vy


# ------------------------------------------------------------------------------------------------
# URDF generation
# ------------------------------------------------------------------------------------------------


def _rpy_from_matrix(R: np.ndarray) -> tuple[float, float, float]:
    """Rotation matrix -> URDF roll/pitch/yaw (fixed-axis X, then Y, then Z)."""
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def _inertial(mass: float, ixx: float, iyy: float, izz: float) -> str:
    return (
        f'<inertial><origin xyz="0 0 0"/><mass value="{mass}"/>'
        f'<inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" iyz="0" izz="{izz}"/></inertial>'
    )


def _box_inertial(m: float, x: float, y: float, z: float) -> str:
    return _inertial(m, m * (y * y + z * z) / 12, m * (x * x + z * z) / 12, m * (x * x + y * y) / 12)


def _wheel_link(name: str, cfg: CarConfig, roller_sign: int) -> str:
    """A wheel = a hub plus N angled rollers. Visual geometry only: no collision shapes, because the
    chassis is moved by the base joints, not by wheel-ground friction (see notebook 00)."""
    r, w = cfg.wheel_radius, cfg.wheel_width
    roller_r = 0.012
    visuals = [
        # Hub: a cylinder whose axis is the wheel axle (link y-axis).
        f'<visual><origin xyz="0 0 0" rpy="{math.pi / 2} 0 0"/>'
        f'<geometry><cylinder radius="{r - 2 * roller_r:.4f}" length="{w * 0.8:.4f}"/></geometry>'
        f'<material name="hub"><color rgba="0.95 0.75 0.1 1"/></material></visual>'
    ]
    for i in range(cfg.n_rollers):
        th = 2 * math.pi * i / cfg.n_rollers
        radial = np.array([math.cos(th), 0.0, math.sin(th)])
        tangent = np.array([-math.sin(th), 0.0, math.cos(th)])
        # Roller axis: 45 deg between the axle (y) and the rim tangent. The sign decides the
        # "handedness" of the wheel; front-left/rear-right are mirrored w.r.t. front-right/rear-left.
        axis = (np.array([0.0, 1.0, 0.0]) + roller_sign * tangent) / math.sqrt(2)
        R = np.stack([radial, np.cross(axis, radial), axis], axis=1)  # columns = new x, y, z
        rpy = _rpy_from_matrix(R)
        p = (r - roller_r) * radial
        visuals.append(
            f'<visual><origin xyz="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" '
            f'rpy="{rpy[0]:.4f} {rpy[1]:.4f} {rpy[2]:.4f}"/>'
            f'<geometry><cylinder radius="{roller_r}" length="{w * 1.1:.4f}"/></geometry>'
            f'<material name="roller"><color rgba="0.15 0.15 0.15 1"/></material></visual>'
        )
    inert = _inertial(cfg.wheel_mass, 1e-4, 2e-4, 1e-4)
    return f'<link name="{name}">{inert}{"".join(visuals)}</link>'


def build_urdf(cfg: CarConfig, out_path: str | Path) -> Path:
    """Write the car URDF to ``out_path`` and return the path."""
    L, W, H = cfg.chassis_length, cfg.chassis_width, cfg.chassis_height
    zc = cfg.chassis_z
    tiny = _inertial(0.05, 1e-4, 1e-4, 1e-4)  # massless-ish helper links for the slide joints

    parts = ['<?xml version="1.0"?>', '<robot name="mecanum_car">']
    # --- Holonomic base: world_anchor -(slide x)-> base_x -(slide y)-> base_y -(yaw)-> chassis ---
    parts += [
        f'<link name="world_anchor">{tiny}</link>',
        f'<link name="base_x">{tiny}</link>',
        f'<link name="base_y">{tiny}</link>',
        '<joint name="base_x_joint" type="prismatic"><parent link="world_anchor"/><child link="base_x"/>'
        '<axis xyz="1 0 0"/><limit lower="-100" upper="100" effort="1000" velocity="10"/></joint>',
        '<joint name="base_y_joint" type="prismatic"><parent link="base_x"/><child link="base_y"/>'
        '<axis xyz="0 1 0"/><limit lower="-100" upper="100" effort="1000" velocity="10"/></joint>',
        f'<joint name="base_yaw_joint" type="continuous"><parent link="base_y"/><child link="chassis"/>'
        f'<origin xyz="0 0 {zc}"/><axis xyz="0 0 1"/></joint>',
    ]
    # --- Chassis: the only part that collides with the world ---
    chassis_vis = [
        f'<visual><geometry><box size="{L} {W} {H}"/></geometry>'
        f'<material name="body"><color rgba="0.2 0.45 0.8 1"/></material></visual>',
        # Red bumper so "front" is obvious from the outside.
        f'<visual><origin xyz="{L / 2 - 0.01} 0 {H / 2 + 0.005}"/><geometry><box size="0.02 {W * 0.8} 0.01"/>'
        f'</geometry><material name="bumper"><color rgba="0.85 0.1 0.1 1"/></material></visual>',
        # Camera mast + camera box.
        f'<visual><origin xyz="{cfg.cam_forward - 0.02} 0 {(cfg.cam_height - zc) / 2 + H / 4}"/>'
        f'<geometry><box size="0.015 0.015 {cfg.cam_height - zc - H / 2:.4f}"/></geometry>'
        f'<material name="mast"><color rgba="0.1 0.1 0.1 1"/></material></visual>',
        f'<visual><origin xyz="{cfg.cam_forward - 0.02} 0 {cfg.cam_height - zc}"/>'
        f'<geometry><box size="0.03 0.05 0.03"/></geometry>'
        f'<material name="mast"><color rgba="0.1 0.1 0.1 1"/></material></visual>',
    ]
    if cfg.roof_marker:
        # Notebook 06: a big cyan arrow on the roof, pointing forward, so an OVERHEAD camera can see where the car
        # faces (the 2 cm bumper is invisible from 8 m up). Cyan is no goal, obstacle, floor or wall colour, and it
        # also stops the blue body looking like a "blue" goal from above. It sits behind the camera mast, so the
        # car's own front camera never sees it. Shaft along the body + a chevron (two bars at +-45 deg) at the front.
        z, cyan = H / 2 + 0.003, '<material name="marker"><color rgba="0.0 0.95 0.95 1"/></material>'
        fr = cfg.roof_marker_front
        front = cyan if fr is None else f'<material name="marker_front"><color rgba="{fr[0]} {fr[1]} {fr[2]} 1"/></material>'
        tip, arm = 0.12, 0.15                      # chevron tip (m ahead of the centre) and arm length
        d = arm / 2 * math.cos(math.pi / 4)        # arm centre offset from the tip, along x and y
        chassis_vis += [
            f'<visual><origin xyz="{-0.05:.4f} 0 {z:.4f}"/><geometry><box size="0.26 0.07 0.006"/></geometry>{cyan}</visual>',
            f'<visual><origin xyz="{tip - d:.4f} {d:.4f} {z:.4f}" rpy="0 0 {-math.pi / 4:.4f}"/>'
            f'<geometry><box size="{arm} 0.05 0.006"/></geometry>{front}</visual>',
            f'<visual><origin xyz="{tip - d:.4f} {-d:.4f} {z:.4f}" rpy="0 0 {math.pi / 4:.4f}"/>'
            f'<geometry><box size="{arm} 0.05 0.006"/></geometry>{front}</visual>',
        ]
    # The collision box covers the wheels' footprint, so the car's "bumper" matches what you see.
    coll_L = 2 * (cfg.wheelbase_half + cfg.wheel_radius)
    coll_W = 2 * cfg.track_half + cfg.wheel_width
    parts.append(
        f'<link name="chassis">{_box_inertial(cfg.chassis_mass, L, W, H)}{"".join(chassis_vis)}'
        f'<collision><origin xyz="0 0 0.005"/><geometry><box size="{coll_L:.4f} {coll_W:.4f} {H + 0.04:.4f}"/>'
        f"</geometry></collision></link>"
    )
    # --- Wheels (visual + spinning, no collision) ---
    signs = {"fl": +1, "fr": -1, "rl": -1, "rr": +1}
    for name in WHEEL_NAMES:
        x = cfg.wheelbase_half if name[0] == "f" else -cfg.wheelbase_half
        y = cfg.track_half if name[1] == "l" else -cfg.track_half
        parts.append(_wheel_link(f"wheel_{name}", cfg, signs[name]))
        parts.append(
            f'<joint name="wheel_{name}_joint" type="continuous"><parent link="chassis"/>'
            f'<child link="wheel_{name}"/><origin xyz="{x} {y} {cfg.wheel_radius - zc}"/>'
            f'<axis xyz="0 1 0"/></joint>'
        )
    parts.append("</robot>")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Parallel simulator workers all call this on the same file; write to a private temp file and rename it into place
    # so that no worker ever reads a half-written URDF.
    tmp = out_path.with_name(f"{out_path.name}.{os.getpid()}.tmp")
    tmp.write_text("\n".join(parts))
    os.replace(tmp, out_path)
    return out_path
