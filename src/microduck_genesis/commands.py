"""Upstream velocity/head/body command sampler with deterministic torch RNG."""
from __future__ import annotations
import torch

TWIST_RANGES = ((-0.4, 0.4), (-0.3, 0.3), (-1.0, 1.0))
HEAD_RANGES_INITIAL = ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))
BODY_RANGES = ((-0.005, 0.005), (-0.005, 0.005), (-0.005, 0.005), (-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05))

class CommandGenerator:
    def __init__(self, num_envs: int, device: torch.device, seed: int = 0, evaluation: str | None = None):
        self.num_envs, self.device, self.evaluation = num_envs, device, evaluation
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.command = torch.zeros(num_envs, 13, device=device)
        self.resample(torch.arange(num_envs, device=device))

    def _uniform(self, ids: torch.Tensor, ranges: tuple[tuple[float, float], ...]) -> torch.Tensor:
        lo = torch.tensor([x[0] for x in ranges], device=self.device)
        hi = torch.tensor([x[1] for x in ranges], device=self.device)
        return lo + (hi - lo) * torch.rand((len(ids), len(ranges)), device=self.device, generator=self.generator)

    def resample(self, ids: torch.Tensor) -> torch.Tensor:
        if self.evaluation:
            values = {"stand": (0., 0., 0.), "forward": (.2, 0., 0.), "backward": (-.2, 0., 0.), "turn_left": (0., 0., .6), "turn_right": (0., 0., -.6)}
            if self.evaluation not in values: raise ValueError(f"unsupported evaluation command: {self.evaluation}")
            self.command[ids] = 0
            self.command[ids, :3] = torch.tensor(values[self.evaluation], device=self.device)
            return self.command
        self.command[ids, :3] = self._uniform(ids, TWIST_RANGES)
        # Upstream: 2% exact standing and 15% turn-in-place, sampled command-only.
        standing = torch.rand(len(ids), device=self.device, generator=self.generator) < .02
        turning = torch.rand(len(ids), device=self.device, generator=self.generator) < .15
        self.command[ids[standing], :3] = 0
        if turning.any():
            tid = ids[turning]; sign = torch.where(torch.rand(len(tid), device=self.device, generator=self.generator) < .5, -1., 1.)
            self.command[tid, :2] = 0; self.command[tid, 2] = sign * (.4 + .6 * torch.rand(len(tid), device=self.device, generator=self.generator))
        self.command[ids, 3:7] = self._uniform(ids, HEAD_RANGES_INITIAL)
        self.command[ids, 7:13] = self._uniform(ids, BODY_RANGES)
        return self.command
