#!/usr/bin/env python3
"""Compare repeated train/eval observations from identical states."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--device',default='cuda'); a=p.parse_args()
    from microduck_genesis.env import MicroDuckGenesisEnv
    env=MicroDuckGenesisEnv(1,headless=True,device=a.device,seed=321,evaluation_command='forward',randomization=False,domain_randomization=False,sensor_noise=False,sensor_delay=False); obs=env.reset();
    for label in ('reset','step1','step2','step5'):
        if label!='reset':
            for _ in range(1 if label in ('step1','step2') else 3): obs,_,_,_=env.step(torch.zeros(1,14,device=env.device))
        first=obs.clone(); second=env._obs().clone(); err=(first-second).abs().max().item(); print(label,'max_error=',err); assert err < 1e-6
    print('PASS train/eval observation parity with noise and delay disabled')
if __name__=='__main__': main()
