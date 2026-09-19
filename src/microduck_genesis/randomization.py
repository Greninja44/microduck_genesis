"""Per-environment DR state; exact only for encoder noise in Genesis 1.4.1."""
from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class RandomizationState:
    encoder_bias: torch.Tensor
    imu_angle: torch.Tensor
    friction_scale: torch.Tensor
    armature_scale: torch.Tensor
    mass_scale: torch.Tensor

def make_randomization(num_envs: int, device: torch.device) -> RandomizationState:
    return RandomizationState(*(torch.zeros(num_envs, d, device=device) if d > 1 else torch.ones(num_envs, d, device=device) for d in (14, 3, 1, 1, 1)))

def reset_randomization(state: RandomizationState, ids: torch.Tensor, generator: torch.Generator) -> None:
    device = state.encoder_bias.device
    state.encoder_bias[ids] = -.015 + .03 * torch.rand((len(ids), 14), device=device, generator=generator)
    state.imu_angle[ids] = (torch.rand((len(ids), 3), device=device, generator=generator) * 2 - 1) * (6.0 * torch.pi / 180)
    state.friction_scale[ids] = .9 + .2 * torch.rand((len(ids), 1), device=device, generator=generator)
    state.armature_scale[ids] = .9 + .2 * torch.rand((len(ids), 1), device=device, generator=generator)
    state.mass_scale[ids] = .95 + .1 * torch.rand((len(ids), 1), device=device, generator=generator)
