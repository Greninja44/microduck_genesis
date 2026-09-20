"""Replay actions from logs/mujoco_reference.npz in Genesis."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from microduck_genesis.env import MicroDuckGenesisEnv

def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',default='logs/mujoco_reference.npz'); p.add_argument('--output',default='logs/genesis_replay.npz'); p.add_argument('--device',default='cuda'); a=p.parse_args()
 ref=np.load(a.input); actions=torch.as_tensor(ref['action'],dtype=torch.float32)
 env=MicroDuckGenesisEnv(1,True,a.device,seed=123,randomization=False,evaluation_command='stand'); env.reset(); rows=[]
 for action in actions:
  obs,reward,done,info=env.step(action.reshape(1,14).to(env.device)); q,qd,pos,quat,vel,ang=env._state(); rows.append((q,qd,pos,quat,vel,ang,reward,done))
 def stack(i): return torch.cat([x[i].detach().cpu() for x in rows],dim=0).numpy()
 Path(a.output).parent.mkdir(exist_ok=True); np.savez_compressed(a.output,time=ref['time'],action=ref['action'],joint_pos=stack(0),joint_vel=stack(1),base_pos=stack(2),base_quat=stack(3),base_lin_vel=stack(4),base_ang_vel=stack(5),reward=stack(6),done=stack(7)); print(f'wrote {a.output}')
if __name__=='__main__': main()
