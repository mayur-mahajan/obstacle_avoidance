"""Static world geometry. For now: a floor and four walls (obstacles and goals come in notebook 01)."""

import genesis as gs


def add_walled_arena(scene: "gs.Scene", size: float = 8.0, wall_height: float = 0.5, wall_thickness: float = 0.1):
    """A square arena of ``size`` x ``size`` metres centred on the origin."""
    scene.add_entity(gs.morphs.Plane())
    half = size / 2
    wall_color = (0.75, 0.72, 0.65)
    specs = [
        ((0, half, wall_height / 2), (size, wall_thickness, wall_height)),
        ((0, -half, wall_height / 2), (size, wall_thickness, wall_height)),
        ((half, 0, wall_height / 2), (wall_thickness, size, wall_height)),
        ((-half, 0, wall_height / 2), (wall_thickness, size, wall_height)),
    ]
    for pos, sz in specs:
        scene.add_entity(gs.morphs.Box(pos=pos, size=sz, fixed=True), surface=gs.surfaces.Default(color=wall_color))


def add_demo_objects(scene: "gs.Scene"):
    """A few coloured landmarks so there is something to drive toward in the first test drive."""
    objs = [
        (gs.morphs.Sphere(radius=0.15, pos=(2.5, 0.0, 0.15), fixed=True), (0.9, 0.1, 0.1)),
        (gs.morphs.Box(size=(0.3, 0.3, 0.3), pos=(1.5, 2.0, 0.15), fixed=True), (0.1, 0.7, 0.1)),
        (gs.morphs.Cylinder(radius=0.15, height=0.4, pos=(-1.5, -2.0, 0.2), fixed=True), (0.1, 0.3, 0.9)),
    ]
    for morph, color in objs:
        scene.add_entity(morph, surface=gs.surfaces.Default(color=color))
