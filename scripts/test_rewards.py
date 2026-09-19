#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
def main():
    import torch
    from microduck_genesis.env import MicroDuckGenesisEnv
    p=argparse.ArgumentParser(); p.add_argument('--num-envs',type=int,default=1); p.add_argument('--device',default='cpu'); a=p.parse_args()
    env=MicroDuckGenesisEnv(a.num_envs, device=a.device)
    for _ in range(3):
        _, total, _, info=env.step(torch.zeros(a.num_envs,14,device=env.device))
        summed=torch.stack(tuple(info['reward_terms'].values())).sum(0)
        assert torch.isfinite(total).all() and torch.allclose(total,summed)
    print('PASS: finite rewards; total equals sum of weighted reward terms')
if __name__=='__main__': main()
