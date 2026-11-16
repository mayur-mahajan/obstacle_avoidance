"""Central place for every tunable number of the simulated car.

Keeping these in dataclasses (instead of scattering magic numbers through the code) means the
notebooks can print them, explain them and override them in one line.
"""

from dataclasses import dataclass


@dataclass
class CarConfig:
    # ---- Chassis (metres / kilograms) ------------------------------------------------------------
    chassis_length: float = 0.40  # along the car's forward (x) axis
    chassis_width: float = 0.26  # along the car's left (y) axis
    chassis_height: float = 0.06
    chassis_mass: float = 3.0
    # ---- Wheels ------------------------------------------------------------------------------------
    wheel_radius: float = 0.05
    wheel_width: float = 0.04
    wheel_mass: float = 0.15
    wheelbase_half: float = 0.15  # "lx": distance from centre to front/rear axle
    track_half: float = 0.17  # "ly": distance from centre to left/right wheel plane
    n_rollers: int = 10  # mecanum rollers per wheel (visual only)
    # ---- FPV camera --------------------------------------------------------------------------------
    cam_forward: float = 0.17  # mounted near the front bumper...
    cam_height: float = 0.16  # ...this high above the ground
    cam_pitch_deg: float = 10.0  # tilted down a little so the floor ahead is visible
    cam_fov_deg: float = 75.0  # vertical field of view
    cam_res: tuple = (256, 256)  # (width, height) in pixels
    # ---- Roof marker (notebook 06) -----------------------------------------------------------------
    roof_marker: bool = False  # cyan arrow on the roof so an overhead camera sees the heading; front camera can't see it
    # colour of the arrow's FRONT chevron; None = cyan like the rest (first design). Two colours (2026-11-14): the overhead
    # server read the all-cyan arrow BACKWARDS in ~6% of pictures; a hot-pink front makes front and back unmistakable.
    roof_marker_front: tuple | None = None
    # ---- Velocity limits (what the "motors" can do) -----------------------------------------------
    max_vx: float = 0.8  # m/s forward/backward
    max_vy: float = 0.6  # m/s sideways (strafing)
    max_wz: float = 1.5  # rad/s rotation (~86 deg/s)
    # ---- Base velocity controller ------------------------------------------------------------------
    kv_linear: float = 400.0  # gain of the velocity controller on the x/y slides
    kv_yaw: float = 40.0  # gain on the yaw hinge
    max_force: float = 60.0  # N   -> the car can't push harder than this (obstacles stop it)
    max_torque: float = 15.0  # N*m

    @property
    def chassis_z(self) -> float:
        """Height of the chassis centre above the ground (chassis sits between the wheels)."""
        return self.wheel_radius + 0.03


@dataclass
class SimConfig:
    sim_dt: float = 0.01  # physics step: 100 Hz
    control_hz: float = 10.0  # policy / keyboard decisions per second
    # Smoothing of keyboard commands. A real motor can't jump from 0 to full speed instantly, and
    # smooth labels are much easier for a neural network to imitate than on/off jumps.
    accel_linear: float = 1.5  # m/s^2 when speeding up
    decel_linear: float = 3.0  # m/s^2 when a key is released (braking is faster)
    accel_yaw: float = 4.0  # rad/s^2
    decel_yaw: float = 8.0

    @property
    def substeps(self) -> int:
        """Physics steps per control step (10 Hz control on 100 Hz physics -> 10)."""
        return round(1.0 / (self.control_hz * self.sim_dt))
