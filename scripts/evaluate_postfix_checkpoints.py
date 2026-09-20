#!/usr/bin/env python3
"""Compare deterministic forward behavior for fresh checkpoints 0/1..5."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))

@torch.no_grad()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--device',default='cuda'); p.add_argument('--steps',type=int,default=100); p.add_argument('--checkpoint-dir',default='checkpoints/postfix'); a=p.parse_args()
 from microduck_genesis.env import MicroDuckGenesisEnv
 from microduck_genesis.ppo import ActorCritic,RunningNorm
 env=MicroDuckGenesisEnv(1,True,a.device,seed=900,evaluation_command='forward',randomization=False,domain_randomization=False,sensor_noise=True,sensor_delay=True); dev=env.device
 paths=[None]+[Path(a.checkpoint_dir)/f'model_{i:04d}.pt' for i in range(1,6)]
 for it,path in enumerate(paths):
  model=ActorCritic().to(dev); norm=RunningNorm(61).to(dev)
  if path is None: norm.eval()
  else:
   s=torch.load(path,map_location=dev,weights_only=False); model.load_state_dict(s['model']); norm.load_state_dict(s['norm'])
  model.eval(); env.reset(); env.commands.command.zero_(); env.commands.command[:,0]=.2; obs=env._obs(); x0=env._state()[2][0,0].item(); vs=[]; mus=[]; alive=0; reason='horizon'
  for k in range(a.steps):
   mu=model.actor(norm(obs)); mus.append(mu.abs().max().item())
   try: obs,r,d,info=env.step(mu)
   except FloatingPointError as e: reason=str(e); break
   vs.append(env._state()[4][0,0].item()); alive=k+1
   if bool(d[0]): reason=','.join(n for n,v in info['termination_terms'].items() if bool(v[0])); break
  dist=env._state()[2][0,0].item()-x0; print(f'iteration={it} checkpoint={path} survival={alive*env.control_dt:.3f} distance={dist:.6f} mean_vx={sum(vs)/len(vs) if vs else 0:.6f} tracking_rmse={(sum((v-.2)**2 for v in vs)/len(vs))**.5 if vs else 0:.6f} max_actor_mean={max(mus) if mus else 0:.6f} termination={reason}')
if __name__=='__main__': main()
