"""Small, explicit Genesis scene factory for MicroDuck validation."""
from __future__ import annotations


def make_scene(show_viewer: bool = False, dt: float = 0.005):
    import genesis as gs
    return gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.65, -0.85, 0.45), camera_lookat=(0.0, 0.0, 0.12), camera_fov=45,
        ),
        show_viewer=show_viewer,
    )


def add_ground(scene):
    import genesis as gs
    return scene.add_entity(gs.morphs.Plane())

