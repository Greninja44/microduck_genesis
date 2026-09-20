"""Load the unmodified upstream MicroDuck walking MJCF into Genesis."""
from __future__ import annotations

from pathlib import Path


JOINT_NAMES = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
HOME_POSE = (0.0, -0.0872664626, -0.457924, -0.004940, 0.452984,
             0.3490658504, 0.3490658504, 0.0, 0.0,
             0.0, 0.0872664626, 0.457924, 0.004940, -0.452984)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def mjcf_path() -> Path:
    path = project_root() / "upstream/microduck_rl/src/mjlab_microduck/robot/microduck/robot_walk.xml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing upstream walking MJCF: {path}")
    return path


def add_microduck(scene, *, visualize_contact: bool = False):
    """Add the current walking model directly, retaining its mesh-relative paths."""
    import genesis as gs
    return scene.add_entity(gs.morphs.MJCF(file=str(mjcf_path())), visualize_contact=visualize_contact)


def actuated_dof_indices(robot) -> list[int]:
    return [robot.get_joint(name).dofs_idx_local[0] for name in JOINT_NAMES]


def set_home_pose(robot) -> list[int]:
    indices = actuated_dof_indices(robot)
    robot.set_dofs_position(HOME_POSE, dofs_idx_local=indices)
    return indices
