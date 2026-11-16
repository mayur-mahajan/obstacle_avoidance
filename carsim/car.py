"""The simulated mecanum car: Genesis entity + velocity controller + first-person camera.

How the car moves (the "kinematic holonomic base" decision):
    Genesis has no mecanum-wheel model and its friction is the same in every direction, so we do
    not simulate wheel/ground contact. Instead the chassis hangs off three motorised joints
    (slide-x, slide-y, rotate-z). Every physics step we turn the commanded body velocity
    (vx, vy, wz) into world-frame joint velocities and a velocity controller drives the joints.
    Because the controller can only push with a limited force, obstacles still stop the car: it
    behaves like a robot with strong but not infinite motors.
"""

import math
from pathlib import Path

import numpy as np

import genesis as gs

from .config import CarConfig, SimConfig
from .mecanum import WHEEL_NAMES, body_to_world, build_urdf, wheel_speeds

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"


def _camera_offset(cfg: CarConfig) -> np.ndarray:
    """4x4 pose of the camera relative to the chassis centre.

    Genesis cameras follow the OpenGL convention: the camera looks along its own -z axis and +y is
    "up" in the image. We map that onto the car: look forward (+x of the car), up = +z, then tilt
    the camera down by ``cam_pitch_deg``.
    """
    look_forward = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # Rotating about the camera's x-axis by a *negative* angle tips the viewing direction (-z) down.
    p = -math.radians(cfg.cam_pitch_deg)
    tilt_down = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    T = np.eye(4)
    T[:3, :3] = look_forward @ tilt_down
    T[:3, 3] = [cfg.cam_forward, 0.0, cfg.cam_height - cfg.chassis_z]
    return T


class MecanumCar:
    """Create with the scene *before* ``scene.build()``, then call ``on_built()`` after it."""

    def __init__(self, scene: "gs.Scene", cfg: CarConfig | None = None, sim_cfg: SimConfig | None = None):
        self.cfg = cfg or CarConfig()
        self.sim_cfg = sim_cfg or SimConfig()
        self.scene = scene
        name = ("mecanum_car_marker2.urdf" if self.cfg.roof_marker_front else "mecanum_car_marker.urdf") if self.cfg.roof_marker else "mecanum_car.urdf"
        urdf = build_urdf(self.cfg, ASSET_DIR / name)
        self.entity = scene.add_entity(gs.morphs.URDF(file=str(urdf), fixed=True))
        self.fpv = scene.add_camera(
            res=self.cfg.cam_res, fov=self.cfg.cam_fov_deg, near=0.02, far=30.0, GUI=False
        )
        self.command = np.zeros(3)  # last applied (vx, vy, wz) in the car's own frame

    # ---------------------------------------------------------------------------------------------
    def on_built(self) -> None:
        """Look up joint indices and configure the motors. Needs a built scene."""
        e = self.entity
        self.base_dofs = [e.get_joint(n).dofs_idx_local[0] for n in ("base_x_joint", "base_y_joint", "base_yaw_joint")]
        self.wheel_dofs = [e.get_joint(f"wheel_{n}_joint").dofs_idx_local[0] for n in WHEEL_NAMES]
        c = self.cfg
        e.set_dofs_kv(np.array([c.kv_linear, c.kv_linear, c.kv_yaw]), self.base_dofs)
        lim = np.array([c.max_force, c.max_force, c.max_torque])
        e.set_dofs_force_range(-lim, lim, self.base_dofs)
        e.set_dofs_kv(np.full(4, 0.5), self.wheel_dofs)
        e.set_dofs_force_range(np.full(4, -5.0), np.full(4, 5.0), self.wheel_dofs)
        self.chassis = e.get_link("chassis")
        self.fpv.attach(self.chassis, _camera_offset(c))

    # ---------------------------------------------------------------------------------------------
    def reset(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> None:
        """Teleport the car to a pose and stop it."""
        q = np.zeros(self.entity.n_dofs)
        q[self.base_dofs] = [x, y, yaw]
        self.entity.set_dofs_position(q, zero_velocity=True)
        self.command[:] = 0.0
        self.fpv.move_to_attach()

    def pose(self) -> np.ndarray:
        """(x, y, yaw) in the world frame."""
        return self.entity.get_dofs_position(self.base_dofs).cpu().numpy().astype(np.float64)

    def body_velocity(self) -> np.ndarray:
        """Measured (vx, vy, wz) in the car's own frame - what the car is *actually* doing."""
        vxw, vyw, wz = self.entity.get_dofs_velocity(self.base_dofs).cpu().numpy().astype(np.float64)
        yaw = self.pose()[2]
        vx, vy = body_to_world(vxw, vyw, -yaw)  # rotating by -yaw goes world -> body
        return np.array([vx, vy, wz])

    # ---------------------------------------------------------------------------------------------
    def step(self, vx: float, vy: float, wz: float, on_substep=None) -> None:
        """Hold the body-frame command (vx, vy, wz) for one *control* period (several physics steps).

        The command is clipped to the motor limits. The body->world rotation is recomputed every
        physics step because the car's heading changes while it turns. ``on_substep`` (optional) is
        called after every physics step, e.g. to check for collisions.
        """
        c = self.cfg
        vx = float(np.clip(vx, -c.max_vx, c.max_vx))
        vy = float(np.clip(vy, -c.max_vy, c.max_vy))
        wz = float(np.clip(wz, -c.max_wz, c.max_wz))
        self.command[:] = (vx, vy, wz)
        wheels = wheel_speeds(vx, vy, wz, c)
        for _ in range(self.sim_cfg.substeps):
            yaw = float(self.entity.get_dofs_position([self.base_dofs[2]])[0])
            vxw, vyw = body_to_world(vx, vy, yaw)
            self.entity.control_dofs_velocity(np.array([vxw, vyw, wz]), self.base_dofs)
            self.entity.control_dofs_velocity(wheels, self.wheel_dofs)
            self.scene.step()
            if on_substep is not None:
                on_substep()
        self.fpv.move_to_attach()

    def render_fpv(self) -> np.ndarray:
        """First-person RGB image, shape (H, W, 3), dtype uint8."""
        rgb, *_ = self.fpv.render(rgb=True)
        return np.asarray(rgb)
