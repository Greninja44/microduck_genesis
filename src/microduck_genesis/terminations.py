"""Termination checks ported where Genesis exposes equivalent state."""
from __future__ import annotations
import torch

def compute_terminations(*, base_pos: torch.Tensor, projected_gravity: torch.Tensor, state_tensors: tuple[torch.Tensor, ...], episode_steps: torch.Tensor, max_episode_steps: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Parent velocity cfg owns exact bad_orientation threshold; local mjlab source
    # does not vendor it. Genesis uses the common 60-degree gravity criterion.
    tilted = projected_gravity[:, 2] > -0.5
    below_ground = base_pos[:, 2] < 0.04
    nan_state = ~torch.stack([torch.isfinite(x).all(dim=tuple(range(1, x.ndim))) for x in state_tensors]).all(dim=0)
    timeout = episode_steps >= max_episode_steps
    terms = {"bad_orientation": tilted | below_ground, "nan_state": nan_state, "time_out": timeout}
    return torch.stack(tuple(terms.values())).any(dim=0), terms
