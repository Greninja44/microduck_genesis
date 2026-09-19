"""Explicit policy-action to Genesis-servo mapping."""
from __future__ import annotations

from dataclasses import dataclass
import torch

from .robot import JOINT_NAMES, HOME_POSE

ACTION_DIM = len(JOINT_NAMES)
ACTION_SCALE = 1.0  # JointPositionActionCfg.scale in upstream velocity cfg.


@dataclass(frozen=True)
class ActionMapping:
    policy_index: int
    joint_name: str
    genesis_dof_index: int


def resolve_action_mapping(robot) -> tuple[ActionMapping, ...]:
    mappings = tuple(ActionMapping(i, name, int(robot.get_joint(name).dofs_idx_local[0])) for i, name in enumerate(JOINT_NAMES))
    if len({m.genesis_dof_index for m in mappings}) != ACTION_DIM:
        raise AssertionError("Genesis resolved duplicate servo DOF indices")
    if tuple(m.joint_name for m in mappings) != JOINT_NAMES:
        raise AssertionError("policy joint order differs from upstream JOINT_NAMES")
    return mappings


def action_to_targets(actions: torch.Tensor, *, device: torch.device | None = None) -> torch.Tensor:
    if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
        raise AssertionError(f"expected action shape (N, {ACTION_DIM}), got {tuple(actions.shape)}")
    if not torch.isfinite(actions).all():
        raise FloatingPointError("policy actions contain NaN/Inf")
    # Upstream JointPositionAction uses default joint positions as its offset.
    home = torch.as_tensor(HOME_POSE, dtype=actions.dtype, device=device or actions.device)
    return home + ACTION_SCALE * actions
