#!/usr/bin/env python3
"""One-env, no-PPO RL validation with per-step diagnostics."""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

def main():
    import torch
    from microduck_genesis.env import MicroDuckGenesisEnv
    p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=1000); p.add_argument('--device',default='cuda'); a=p.parse_args()
    env=MicroDuckGenesisEnv(1,headless=True,device=a.device,seed=11)
    out=Path('logs/single_env_smoke.csv'); out.parent.mkdir(exist_ok=True)
    fields=['step','obs_norm','action_norm','reward','base_x','base_y','base_z','joint0','joint0_vel','contacts','done']
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); obs=env.reset()
        for step in range(a.steps):
            action=torch.zeros((1,14),device=env.device)
            obs,reward,done,info=env.step(action)
            q,qd,pos,*_=env._state(); contact=env._contact_state()[2]
            for value in (obs,action,reward,q,qd,pos): assert torch.isfinite(value).all(), f'nonfinite at step {step}'
            w.writerow({'step':step,'obs_norm':float(torch.linalg.vector_norm(obs)),'action_norm':0.,'reward':float(reward[0]),'base_x':float(pos[0,0]),'base_y':float(pos[0,1]),'base_z':float(pos[0,2]),'joint0':float(q[0,0]),'joint0_vel':float(qd[0,0]),'contacts':int(contact.sum()),'done':int(done[0])})
    print(f'PASS: {a.steps} GPU RL steps, finite obs/actions/rewards/states, CSV={out}')
if __name__=='__main__': main()
