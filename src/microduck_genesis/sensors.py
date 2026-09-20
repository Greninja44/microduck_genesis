"""Seeded actor sensor noise/delay model recovered from MJLab velocity config."""
from __future__ import annotations
import torch

class ActorSensorModel:
    """Per-environment observation history with upstream ranges.

    The caller owns clean state and decides whether to enable this model. Noise
    is additive uniform noise; joint velocity has a fixed one-control-step lag,
    and IMU fields support a zero/one-step history selection.
    """
    def __init__(self, num_envs, device, seed=0):
        self.device=torch.device(device); self.gen=torch.Generator(device=self.device).manual_seed(seed)
        self.prev_ang=torch.zeros(num_envs,3,device=self.device); self.prev_grav=torch.zeros(num_envs,3,device=self.device); self.prev_vel=torch.zeros(num_envs,14,device=self.device)
    def reset(self, ids=None):
        if ids is None: self.prev_ang.zero_(); self.prev_grav.zero_(); self.prev_vel.zero_()
        else: self.prev_ang[ids]=0; self.prev_grav[ids]=0; self.prev_vel[ids]=0
    def apply(self, ang, grav, qvel, *, enabled=True):
        if not enabled: out=(ang,grav,self.prev_vel.clone())
        else:
            # The upstream delay sampler is 0–1 for IMU and exactly 1 for qdot.
            out=(ang + torch.empty_like(ang).uniform_(-.03,.03,generator=self.gen),
                 grav + torch.empty_like(grav).uniform_(-.01,.01,generator=self.gen),
                 self.prev_vel + torch.empty_like(qvel).uniform_(-.25,.25,generator=self.gen))
        self.prev_ang.copy_(ang); self.prev_grav.copy_(grav); self.prev_vel.copy_(qvel)
        return out
