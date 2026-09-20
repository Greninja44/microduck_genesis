import math
import torch

from microduck_genesis.env import MicroDuckGenesisEnv
from microduck_genesis.terminations import compute_terminations


def _quat(axis: str, angle_deg: float) -> torch.Tensor:
    a = math.radians(angle_deg) / 2.0
    q = torch.tensor([math.cos(a), 0.0, 0.0, 0.0])
    q[{'roll': 1, 'pitch': 2, 'yaw': 3}[axis]] = math.sin(a)
    return q


def _terminated(quat: torch.Tensor) -> tuple[bool, float]:
    gravity = MicroDuckGenesisEnv._projected_gravity(quat.reshape(1, 4))
    z = torch.zeros(1, 14)
    done, _ = compute_terminations(
        base_pos=torch.tensor([[0.0, 0.0, 0.125]]),
        projected_gravity=gravity,
        state_tensors=(z,), episode_steps=torch.zeros(1, dtype=torch.long),
        max_episode_steps=1000,
    )
    return bool(done[0]), float(gravity[0, 2])


def test_termination_orientation_cases():
    assert _terminated(torch.tensor([1.0, 0.0, 0.0, 0.0])) == (False, -1.0)
    assert _terminated(_quat('roll', 20.0))[0] is False
    assert _terminated(_quat('pitch', 20.0))[0] is False
    assert _terminated(_quat('pitch', 100.0))[0] is True
    assert _terminated(_quat('pitch', -100.0))[0] is True
    assert _terminated(_quat('roll', 100.0))[0] is True
