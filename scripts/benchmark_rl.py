#!/usr/bin/env python3
"""Fresh batched RL-environment benchmark; one process per requested count."""
from __future__ import annotations
import argparse, os, subprocess, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
def main():
 import psutil, torch
 from microduck_genesis.env import MicroDuckGenesisEnv
 p=argparse.ArgumentParser();p.add_argument('--num-envs',type=int,required=True);p.add_argument('--steps',type=int,default=20);p.add_argument('--device',default='cuda');a=p.parse_args()
 env=MicroDuckGenesisEnv(a.num_envs,headless=True,device=a.device,seed=4,randomization=False); obs=env.reset(); actions=torch.zeros(a.num_envs,14,device=env.device)
 for _ in range(2): obs,*_=env.step(actions)
 if env.device.type=='cuda': torch.cuda.synchronize()
 start=time.perf_counter()
 for _ in range(a.steps): obs,*_=env.step(actions)
 if env.device.type=='cuda': torch.cuda.synchronize()
 elapsed=time.perf_counter()-start
 try: smi=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
 except Exception: smi='unavailable'
 print({'num_envs':a.num_envs,'control_steps':a.steps,'control_fps':a.steps/elapsed,'sim_fps':a.num_envs*a.steps*env.decimation/elapsed,'torch_alloc_mib':torch.cuda.memory_allocated()/2**20 if env.device.type=='cuda' else 0,'torch_reserved_mib':torch.cuda.memory_reserved()/2**20 if env.device.type=='cuda' else 0,'ram_mib':psutil.Process().memory_info().rss/2**20,'nvidia_smi':smi},flush=True)
if __name__=='__main__': main()
