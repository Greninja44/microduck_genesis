#!/usr/bin/env python3
from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
def main():
 import torch
 from microduck_genesis.env import MicroDuckGenesisEnv
 from microduck_genesis.ppo import ActorCritic,RunningNorm
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--viewer',action='store_true');p.add_argument('--steps',type=int,default=1000);p.add_argument('--device',default='cuda');a=p.parse_args()
 env=MicroDuckGenesisEnv(1,headless=not a.viewer,device=a.device,evaluation_command='forward',randomization=False); state=torch.load(a.checkpoint,map_location=env.device);m=ActorCritic().to(env.device);n=RunningNorm(61).to(env.device);m.load_state_dict(state['model']);n.load_state_dict(state['norm']);o=env.reset()
 for _ in range(a.steps):
  with torch.no_grad(): act=m.actor(n(o));o,r,d,_=env.step(act)
 print('evaluation complete')
if __name__=='__main__':main()
