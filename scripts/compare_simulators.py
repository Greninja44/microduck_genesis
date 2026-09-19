#!/usr/bin/env python3
"""Write a Genesis baseline trace; MuJoCo column stays blank when unavailable."""
from __future__ import annotations
import csv,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
def main():
 import torch
 from microduck_genesis.env import MicroDuckGenesisEnv
 out=Path('logs/simulator_comparison.csv');out.parent.mkdir(exist_ok=True)
 env=MicroDuckGenesisEnv(1,device='cpu',randomization=False);env.reset()
 with out.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['time_s','genesis_base_z','genesis_joint_0','genesis_joint_vel_0','mujoco_base_z','mujoco_joint_0','note']);w.writeheader()
  for i in range(100):
   env.step(torch.zeros(1,14,device=env.device));q,qd,pos,*_=env._state();w.writerow({'time_s':i*env.control_dt,'genesis_base_z':float(pos[0,2]),'genesis_joint_0':float(q[0,0]),'genesis_joint_vel_0':float(qd[0,0]),'note':'MuJoCo runtime/config is not installed in this project'})
 print(out)
if __name__=='__main__':main()
