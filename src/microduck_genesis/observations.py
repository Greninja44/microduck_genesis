"""The actor observation contract of ``Mjlab-Velocity-Flat-MicroDuck``.

This module deliberately contains no implicit concatenation: the order is the
deployment ABI shared by the MicroDuck policy family.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

from .robot import JOINT_NAMES, HOME_POSE


@dataclass(frozen=True)
class ObservationField:
    name: str
    size: int
    description: str


ACTOR_FIELDS = (
    ObservationField("base_ang_vel", 3, "base-frame angular velocity (rad/s)"),
    ObservationField("projected_gravity", 3, "world gravity projected into base frame"),
    ObservationField("joint_pos_rel", 14, "servo position minus HOME_POSE (rad)"),
    ObservationField("joint_vel_rel", 14, "servo velocity (rad/s)"),
    ObservationField("last_action", 14, "previous raw policy action"),
    ObservationField("twist_command", 3, "[vx, vy, yaw_rate] in base frame"),
    ObservationField("head_pose_command", 4, "neck_pitch, head_pitch, head_yaw, head_roll deltas"),
    ObservationField("body_pose_command", 6, "[x, y, z, roll, pitch, yaw] deltas"),
)
EXPECTED_OBSERVATION_DIM = sum(field.size for field in ACTOR_FIELDS)
assert EXPECTED_OBSERVATION_DIM == 61


def layout_lines() -> list[str]:
    lines, index = [], 0
    for field in ACTOR_FIELDS:
        lines.append(f"index {index}:{index + field.size:<2} {field.name:<18} {field.description}")
        index += field.size
    lines.append(f"total dimension: {index}")
    return lines


def validate_observation(observation: torch.Tensor, *, batch: int | None = None) -> torch.Tensor:
    if observation.ndim != 2 or observation.shape[-1] != EXPECTED_OBSERVATION_DIM:
        raise AssertionError(f"expected (N, {EXPECTED_OBSERVATION_DIM}) actor observation, got {tuple(observation.shape)}")
    if batch is not None and observation.shape[0] != batch:
        raise AssertionError(f"expected {batch} observations, got {observation.shape[0]}")
    if not torch.isfinite(observation).all():
        bad = (~torch.isfinite(observation)).nonzero()[:8].tolist()
        raise FloatingPointError(f"actor observation contains NaN/Inf at {bad}")
    return observation


def build_actor_observation(*, base_ang_vel: torch.Tensor, projected_gravity: torch.Tensor,
                            joint_pos: torch.Tensor, joint_vel: torch.Tensor,
                            last_action: torch.Tensor, command: torch.Tensor,
                            encoder_bias: torch.Tensor | None = None) -> torch.Tensor:
    """Build the upstream 61D actor vector in its exact term insertion order."""
    tensors = (base_ang_vel, projected_gravity, joint_pos, joint_vel, last_action, command)
    n = base_ang_vel.shape[0]
    expected = (3, 3, len(JOINT_NAMES), len(JOINT_NAMES), len(JOINT_NAMES), 13)
    if any(x.ndim != 2 or x.shape[0] != n or x.shape[1] != d for x, d in zip(tensors, expected, strict=True)):
        raise AssertionError("invalid observation component shape or servo ordering")
    home = torch.as_tensor(HOME_POSE, dtype=joint_pos.dtype, device=joint_pos.device)
    sensed_pos = joint_pos + (encoder_bias if encoder_bias is not None else 0.0)
    obs = torch.cat((base_ang_vel, projected_gravity, sensed_pos - home, joint_vel, last_action, command), dim=-1)
    return validate_observation(obs, batch=n)
