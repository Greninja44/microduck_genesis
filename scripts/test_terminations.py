#!/usr/bin/env python3
"""Print the Genesis/MJLab orientation termination cases."""
from __future__ import annotations
import math
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from microduck_genesis.env import MicroDuckGenesisEnv
from microduck_genesis.terminations import compute_terminations

def quat(axis, deg):
    a = math.radians(deg) / 2
    q = torch.tensor([math.cos(a), 0., 0., 0.])
    q[{'roll': 1, 'pitch': 2, 'yaw': 3}[axis]] = math.sin(a)
    return q

def check(name, q, expected):
    g = MicroDuckGenesisEnv._projected_gravity(q.reshape(1, 4))
    z = torch.zeros(1, 14)
    done, terms = compute_terminations(base_pos=torch.tensor([[0., 0., .125]]),
        projected_gravity=g, state_tensors=(z,), episode_steps=torch.zeros(1, dtype=torch.long), max_episode_steps=1000)
    angle = math.degrees(math.acos(float((-g[0, 2]).clamp(-1, 1))))
    actual = bool(done[0])
    print(f'{name}: projected_gravity={g[0].tolist()} orientation_deg={angle:.2f} '
          f'expected={expected} actual={actual} bad_orientation={bool(terms["bad_orientation"][0])}')
    assert actual == expected

def main():
    check('HOME_POSE', torch.tensor([1., 0., 0., 0.]), False)
    check('small_roll', quat('roll', 20), False)
    check('small_pitch', quat('pitch', 20), False)
    check('pitch_65deg', quat('pitch', 65), False)
    check('pitch_75deg', quat('pitch', 75), True)
    check('fallen_forward', quat('pitch', 100), True)
    check('fallen_backward', quat('pitch', -100), True)
    check('fallen_left', quat('roll', 100), True)
    check('fallen_right', quat('roll', -100), True)

if __name__ == '__main__': main()
