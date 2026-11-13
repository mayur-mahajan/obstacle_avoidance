"""Turn keyboard presses into smooth velocity commands.

Why not send "full speed" the instant a key goes down? Two reasons:
  * Physics realism - a real motor needs time to accelerate.
  * Learnability - in imitation learning the network copies the recorded commands. On/off jumps
    (0 -> 0.8 -> 0) are hard to predict from one camera frame, because the frame where you pressed the
    key looks almost identical to the frame before it. A ramp makes the commands a smooth function
    of time, which is much easier to imitate.
"""

from dataclasses import dataclass, field

import numpy as np

from genesis.vis.keybindings import Key, KeyAction, Keybind

from .config import CarConfig, SimConfig

# key -> (axis index, direction). Axis 0 = vx (forward), 1 = vy (left), 2 = wz (counter-clockwise)
DRIVE_KEYS = {
    Key.UP: (0, +1),
    Key.DOWN: (0, -1),
    Key.LEFT: (2, +1),  # rotate left (counter-clockwise seen from above)
    Key.RIGHT: (2, -1),  # rotate right
    Key.Q: (1, +1),  # strafe left
    Key.E: (1, -1),  # strafe right
}

HELP_TEXT = """
  Arrow Up / Down   : drive forward / backward
  Arrow Left / Right: rotate left / right
  Q / E             : strafe left / right   (mecanum sideways motion)
  Space             : emergency stop (zero velocity immediately)
  R                 : restart the current episode (same layout)
  N                 : new random episode
  Esc               : quit
  (Click the 3-D viewer window first, it must have keyboard focus.)
"""


def ramp(current: np.ndarray, target: np.ndarray, dt: float, sim_cfg: SimConfig) -> np.ndarray:
    """Move ``current`` toward ``target`` without exceeding the acceleration limits.

    Speeding up (|target| > |current| in the same direction) uses the gentler 'accel' limit; slowing
    down or reversing uses the stronger 'decel' limit, like braking.
    """
    out = current.copy()
    for i in range(3):
        linear = i < 2
        speeding_up = abs(target[i]) > abs(current[i]) and np.sign(target[i]) * np.sign(current[i]) >= 0
        if linear:
            a = sim_cfg.accel_linear if speeding_up else sim_cfg.decel_linear
        else:
            a = sim_cfg.accel_yaw if speeding_up else sim_cfg.decel_yaw
        out[i] = current[i] + np.clip(target[i] - current[i], -a * dt, a * dt)
    return out


@dataclass
class KeyboardCommander:
    car_cfg: CarConfig
    sim_cfg: SimConfig
    held: set = field(default_factory=set)
    command: np.ndarray = field(default_factory=lambda: np.zeros(3))
    reset_requested: bool = False
    new_episode_requested: bool = False
    quit_requested: bool = False
    stop_requested: bool = False

    # -------------------------------------------------------------------------------------------
    def keybinds(self) -> list[Keybind]:
        """Keybinds to register on the Genesis viewer. PRESS adds a key to the held-set, RELEASE
        removes it, so we always know exactly which keys are down right now."""
        binds = []
        for key in DRIVE_KEYS:
            binds.append(Keybind(f"drive_{key}_down", key, KeyAction.PRESS, callback=self.held.add, args=(key,)))
            binds.append(Keybind(f"drive_{key}_up", key, KeyAction.RELEASE, callback=self.held.discard, args=(key,)))
        binds.append(Keybind("car_reset", Key.R, KeyAction.PRESS, callback=self._flag, args=("reset_requested",)))
        binds.append(Keybind("new_episode", Key.N, KeyAction.PRESS, callback=self._flag, args=("new_episode_requested",)))
        binds.append(Keybind("car_stop", Key.SPACE, KeyAction.PRESS, callback=self._flag, args=("stop_requested",)))
        binds.append(Keybind("car_quit", Key.ESCAPE, KeyAction.PRESS, callback=self._flag, args=("quit_requested",)))
        return binds

    def _flag(self, name: str) -> None:
        setattr(self, name, True)

    # -------------------------------------------------------------------------------------------
    def target(self) -> np.ndarray:
        """Velocity the driver is asking for right now (before ramping)."""
        c = self.car_cfg
        limits = np.array([c.max_vx, c.max_vy, c.max_wz])
        t = np.zeros(3)
        for key in list(self.held):
            axis, sign = DRIVE_KEYS[key]
            t[axis] += sign
        return np.clip(t, -1, 1) * limits

    def update(self) -> np.ndarray:
        """Call once per control step. Returns the smoothed command to send to the car."""
        if self.stop_requested:
            self.command[:] = 0.0
            self.stop_requested = False
            self.held.clear()
        else:
            self.command = ramp(self.command, self.target(), 1.0 / self.sim_cfg.control_hz, self.sim_cfg)
        return self.command.copy()
