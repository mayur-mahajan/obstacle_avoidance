"""The navigation environment: a Genesis scene + the car + episode logic (reset / step / success).

API in the style of Gymnasium, the standard RL interface:
    obs, info = env.reset(seed)                 # start a new episode
    obs, terminated, truncated, info = env.step(action)   # action = (vx, vy, wz)
(The reward is added in notebook 05, where we design it.)

Genesis limitation that shapes this file: after ``scene.build()`` you cannot add, remove, recolour
or resize objects. So we build *pools* once - every goal object (3 shapes x 6 colours), every
obstacle, several floor covers and wall sets - and each episode moves the ones it needs into the
arena and "parks" the rest far underground, out of sight.
"""

import math

import numpy as np
import genesis as gs
import genesis.utils.geom as gu

from .arena import add_walled_arena  # noqa: F401  (kept for notebook 00)
from .car import MecanumCar
from .config import CarConfig, SimConfig
from .world import (
    COLORS, FLOOR_COLORS, GOAL_HEIGHT, GOAL_RADIUS, OBSTACLE_COLORS, OBSTACLE_POOL, SHAPES, WALL_COLORS,
    EnvConfig, EpisodeSpec, Grid, goal_in_view, occupancy, sample_episode, surface_distance,
)

PARK_Z = -5.0  # parked objects sit 5 m underground


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


class NavEnv:
    def __init__(self, show_viewer: bool = False, car_cfg: CarConfig | None = None, sim_cfg: SimConfig | None = None,
                 env_cfg: EnvConfig | None = None, viewer_res=(1100, 750), overview_camera_res=None,
                 ambient_light: float = 0.3, backend: str = "gpu"):
        """backend: where the physics runs. "cpu" is ~2x faster for this small scene and scales across
        CPU cores when several environments run in parallel processes (measured on a 20-core CPU: GPU 80 steps/s,
        and parallel GPU copies do not add up; CPU 150 steps/s each, ~730 total with 6 processes; results
        identical). Genesis allows one backend per process, chosen by whichever environment is created first."""
        self.car_cfg, self.sim_cfg, self.cfg = car_cfg or CarConfig(), sim_cfg or SimConfig(), env_cfg or EnvConfig()
        self.grid = Grid(self.cfg)
        if not gs._initialized:
            from .device import sim_backend
            gs.init(backend=sim_backend("cpu" if backend == "cpu" else "gpu"), logging_level="warning")
        viewer_opts = gs.options.ViewerOptions(res=viewer_res, camera_fov=50, enable_default_keybinds=False,
                                               realtime_factor=None)
        self.scene = gs.Scene(sim_options=gs.options.SimOptions(dt=self.sim_cfg.sim_dt), viewer_options=viewer_opts,
                              show_viewer=show_viewer, renderer=gs.renderers.Rasterizer(),
                              # Ambient light lifts shadowed sides of objects; Genesis' default (0.1) makes
                              # goal colours on the shaded side too dark to tell apart (e.g. red -> maroon).
                              vis_options=gs.options.VisOptions(ambient_light=(ambient_light,) * 3))
        self._park_slot = 0
        self._build_pools()
        self.car = MecanumCar(self.scene, self.car_cfg, self.sim_cfg)
        # Optional extra camera for notebooks (third-person / overhead pictures); not seen by the model.
        self.overview = (self.scene.add_camera(res=overview_camera_res, pos=(0, -9, 7), lookat=(0, 0, 0), fov=55)
                         if overview_camera_res else None)
        self.scene.build()
        self.car.on_built()
        self._light_node = self._find_light_node()
        self.spec: EpisodeSpec | None = None

    # ------------------------------------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------------------------------------
    def _park_pos(self, z_extra=0.0):
        self._park_slot += 1
        return (60.0 + 3.0 * (self._park_slot % 40), 60.0 + 3.0 * (self._park_slot // 40), PARK_Z + z_extra)

    def _add(self, morph_cls, color, **kw):
        return self.scene.add_entity(morph_cls(pos=self._park_pos(), fixed=True, **kw),
                                     surface=gs.surfaces.Default(color=color))

    def _build_pools(self):
        s = self.cfg.arena_size
        self.scene.add_entity(gs.morphs.Plane())  # floor variant 0: the default checker plane
        self.floors = [None] + [self._add(gs.morphs.Box, c, size=(s, s, 0.01)) for c in FLOOR_COLORS[1:]]
        t, h = self.cfg.wall_thickness, self.cfg.wall_height
        self.wall_sets = []
        for c in WALL_COLORS:
            self.wall_sets.append([self._add(gs.morphs.Box, c, size=sz)
                                   for sz in [(s, t, h), (s, t, h), (t, s, h), (t, s, h)]])
        self.goal_pool = {}
        for shape in SHAPES:
            for name, rgb in COLORS.items():
                r, hgt = GOAL_RADIUS[shape], GOAL_HEIGHT[shape]
                if shape == "sphere":
                    e = self._add(gs.morphs.Sphere, rgb, radius=r)
                elif shape == "cube":
                    e = self._add(gs.morphs.Box, rgb, size=(2 * r, 2 * r, hgt))
                else:
                    e = self._add(gs.morphs.Cylinder, rgb, radius=r, height=hgt)
                self.goal_pool[(shape, name)] = e
        self.obstacle_pool = []
        for (kind, size), rgb in zip(OBSTACLE_POOL, OBSTACLE_COLORS):
            if kind == "box":
                self.obstacle_pool.append(self._add(gs.morphs.Box, rgb, size=size))
            else:
                self.obstacle_pool.append(self._add(gs.morphs.Cylinder, rgb, radius=size[0], height=size[1]))
        self._park_positions = {}  # entity idx -> original parking spot

    def _find_light_node(self):
        """The scene's directional light lives in the renderer's internal scene graph. Genesis has no
        public API to change lights after build, so we reach into it here (isolated in one place in case
        a Genesis update moves it). Returns None if not found - lighting then simply stays fixed."""
        try:
            ctx = self.scene.visualizer.context
            pyr = next(v for v in vars(ctx).values() if type(v).__name__ == "Scene" and v is not self.scene)
            self._pyr_scene = pyr
            return next(n for n in pyr.nodes if n.light is not None)
        except Exception:
            return None

    # ------------------------------------------------------------------------------------------------
    # Episode control
    # ------------------------------------------------------------------------------------------------
    def _place(self, entity, x, y, z, yaw=0.0):
        entity.set_pos(np.array([x, y, z]))
        entity.set_quat(_yaw_quat(yaw))

    def _park(self, entity):
        if entity.idx not in self._park_positions:
            p = entity.get_pos().cpu().numpy()
            self._park_positions[entity.idx] = p if p[2] < PARK_Z / 2 else np.array(self._park_pos())
        entity.set_pos(self._park_positions[entity.idx])

    def reset(self, seed: int | None = None, spec: EpisodeSpec | None = None):
        """Start a new episode: sample (or reuse) a layout, move pool objects into place, reset the car."""
        if spec is None:
            seed = int(np.random.randint(0, 2**31 - 1)) if seed is None else seed
            spec = sample_episode(seed, self.cfg)
        self.spec = spec
        # Park everything, then bring in what this episode uses.
        for e in [f for f in self.floors if f is not None] + [w for ws in self.wall_sets for w in ws] \
                + list(self.goal_pool.values()) + self.obstacle_pool:
            self._park(e)
        floor = self.floors[spec.floor_index]
        if floor is not None:
            self._place(floor, 0.0, 0.0, -0.002)  # top surface 3 mm above the checker plane
        h, t, wh = self.cfg.arena_size / 2, self.cfg.wall_thickness, self.cfg.wall_height
        for w, (x, y) in zip(self.wall_sets[spec.wall_index], [(0, h), (0, -h), (h, 0), (-h, 0)]):
            self._place(w, x, y, wh / 2)
        for shape, colour, x, y in spec.objects:
            self._place(self.goal_pool[(shape, colour)], x, y, GOAL_HEIGHT[shape] / 2)
        for idx, x, y, yaw in spec.obstacles:
            kind, size = OBSTACLE_POOL[idx]
            self._place(self.obstacle_pool[idx], x, y, (size[2] if kind == "box" else size[1]) / 2, yaw)
        self._set_light(spec.light)
        self.goal_entity = self.goal_pool[spec.goal[:2]]
        self._occ_no_goal = occupancy(self.grid, spec.objects, spec.obstacles, skip_object=spec.goal_index)
        self.car.reset(*spec.start)
        self.scene.visualizer.update(force=True)  # make the renderer see the moved objects
        self.t = 0
        self.n_collision_steps = 0
        self.path_length = 0.0
        self._last_xy = np.array(spec.start[:2])
        obs = self._obs()
        return obs, self._info(obs_collision=False)

    def _set_light(self, light: dict):
        if self._light_node is None:
            return
        az, el = light["azimuth"], light["elevation"]
        direction = -np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])  # pointing down
        pose = np.eye(4)
        gu.z_up_to_R(-direction, out=pose[:3, :3])
        self._pyr_scene.set_pose(self._light_node, pose)
        self._light_node.light.intensity = light["intensity"]

    # ------------------------------------------------------------------------------------------------
    def step(self, action):
        """Apply (vx, vy, wz) for one control period (0.1 s). Returns obs, terminated, truncated, info."""
        collided = self._step_car_with_contacts(*action)
        self.t += 1
        self.n_collision_steps += int(collided)
        xy = self.car.pose()[:2]
        self.path_length += float(np.linalg.norm(xy - self._last_xy))
        self._last_xy = xy
        obs = self._obs()
        info = self._info(obs_collision=collided)
        terminated = info["success"]
        truncated = (not terminated) and self.t >= self.max_steps
        return obs, terminated, truncated, info

    @property
    def max_steps(self) -> int:
        return int(self.cfg.max_episode_s * self.sim_cfg.control_hz)

    def _step_car_with_contacts(self, vx, vy, wz) -> bool:
        """``car.step`` plus: did the car touch anything other than the goal object?

        Genesis only reports contacts from the most recent physics step, so we check after each of the
        10 physics steps of this control period.
        """
        car_links = {l.idx for l in self.car.entity.links}
        goal_links = {l.idx for l in self.goal_entity.links}
        hit = [False]

        def check():
            if hit[0]:
                return
            con = self.car.entity.get_contacts(exclude_self_contact=True)
            for a, b in zip(con["link_a"].tolist(), con["link_b"].tolist()):
                if (b if a in car_links else a) not in goal_links:
                    hit[0] = True
                    return

        self.car.step(vx, vy, wz, on_substep=check)
        return hit[0]

    # ------------------------------------------------------------------------------------------------
    def _obs(self):
        return {"image": self.car.render_fpv(), "state": self.car.body_velocity().astype(np.float32)}

    def render_overview(self, pos, lookat, up=(0.0, 0.0, 1.0)) -> np.ndarray:
        """Picture from the optional overview camera placed at ``pos`` looking at ``lookat``.
        (For a straight-down view pass e.g. ``up=(0, 1, 0)``: 'up' must not be parallel to the view.)"""
        self.overview.set_pose(pos=np.asarray(pos, dtype=float), lookat=np.asarray(lookat, dtype=float),
                               up=np.asarray(up, dtype=float))
        return np.asarray(self.overview.render()[0])

    def goal_surface_distance(self, pose=None) -> float:
        """Distance from the car's front bumper to the goal object's surface (metres)."""
        x, y, yaw = self.car.pose() if pose is None else pose
        bx, by = x + self.cfg.car_half_length * math.cos(yaw), y + self.cfg.car_half_length * math.sin(yaw)
        shape, _c, gx, gy = self.spec.goal
        return surface_distance(shape, gx, gy, bx, by)

    def _info(self, obs_collision: bool) -> dict:
        pose = self.car.pose()
        vel = self.car.body_velocity()
        d = self.goal_surface_distance(pose)
        speed = float(np.hypot(vel[0], vel[1]))
        success = d <= self.cfg.success_dist and speed <= self.cfg.success_speed and abs(vel[2]) <= self.cfg.success_yaw_rate
        _s, _c, gx, gy = self.spec.goal
        return {
            "prompt": self.spec.prompt,
            "goal": self.spec.goal_name,
            "t": self.t,
            "pose": pose,
            "velocity": vel,
            "collision": bool(obs_collision),
            "n_collision_steps": self.n_collision_steps,
            "goal_surface_dist": d,
            "geodesic_dist": self.grid.lookup(self.spec.dist_field, pose[0], pose[1]),
            "goal_visible": goal_in_view(self.cfg, self.grid, self._occ_no_goal, pose, (gx, gy)),
            "path_length": self.path_length,
            "success": bool(success),
        }
