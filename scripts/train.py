#!/usr/bin/env python3
"""Short/medium training CLI. Default values intentionally fit a 6 GB GPU."""
from __future__ import annotations
import argparse, os, sys, time, subprocess
import psutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
def main():
 import torch
 from microduck_genesis.env import MicroDuckGenesisEnv
 from microduck_genesis.ppo import ActorCritic,RunningNorm
 p=argparse.ArgumentParser(); p.add_argument('--num-envs',type=int,default=256); p.add_argument('--device',default='cuda'); p.add_argument('--seed',type=int,default=0); p.add_argument('--headless',action='store_true'); p.add_argument('--iterations',type=int,default=5); p.add_argument('--checkpoint'); p.add_argument('--resume',action='store_true'); p.add_argument('--steps-per-env',type=int,default=24); p.add_argument('--evaluation-command',choices=('stand','forward','backward','turn_left','turn_right')); p.add_argument('--no-randomization',action='store_true'); a=p.parse_args()
 env=MicroDuckGenesisEnv(a.num_envs,a.headless,a.device,a.seed,evaluation_command=a.evaluation_command,randomization=not a.no_randomization); dev=env.device; model=ActorCritic().to(dev); norm=RunningNorm(61).to(dev); opt=torch.optim.Adam(model.parameters(),lr=1e-3)
 if a.resume and a.checkpoint:
  state=torch.load(a.checkpoint,map_location=dev); model.load_state_dict(state['model']); norm.load_state_dict(state['norm']); opt.load_state_dict(state['optimizer'])
 obs=env.reset(); Path('checkpoints').mkdir(exist_ok=True); Path('runs').mkdir(exist_ok=True)
 for it in range(a.iterations):
  env.rewarder.set_training_steps(it*a.steps_per_env); rollout=[]; started=time.perf_counter()
  for _ in range(a.steps_per_env):
   norm.update(obs)
   with torch.no_grad(): act,lp,val=model.act(norm(obs))
   nxt,rew,done,_=env.step(act); rollout.append((obs,act,lp,val,rew,done)); obs=nxt
  with torch.no_grad(): last=model.critic(norm(obs)).squeeze(-1)
  returns=[]; gae=torch.zeros(a.num_envs,device=dev)
  for _,_,_,val,rew,done in reversed(rollout):
   not_done=(~done).float(); gae=rew+.99*last*not_done-val+.99*.95*not_done*gae; returns.append(gae+val); last=val
  returns=torch.stack(list(reversed(returns)), dim=0)
  O=torch.stack([x[0] for x in rollout]); A=torch.stack([x[1] for x in rollout]); LP=torch.stack([x[2] for x in rollout]); V=torch.stack([x[3] for x in rollout]); R=returns
  O,A,LP,V,R=[x.reshape(-1,*x.shape[2:]) if x.ndim > 2 else x.reshape(-1) for x in (O,A,LP,V,R)]
  adv=(R-V); adv=(adv-adv.mean())/(adv.std()+1e-8); idx=torch.randperm(len(O),device=dev); losses=[]
  for _ in range(5):
   for batch in idx.chunk(4):
    logp,ent,value=model.evaluate(norm(O[batch]),A[batch]); ratio=(logp-LP[batch]).exp(); policy=-torch.minimum(ratio*adv[batch],torch.clamp(ratio,.8,1.2)*adv[batch]).mean(); value_loss=torch.maximum((value-R[batch]).square(),(torch.clamp(value,V[batch]-.2,V[batch]+.2)-R[batch]).square()).mean(); loss=policy+value_loss-.01*ent.mean(); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); losses.append((policy.item(),value_loss.item(),ent.mean().item()))
  elapsed=time.perf_counter()-started; mem=torch.cuda.max_memory_allocated(dev)/2**20 if dev.type=='cuda' else 0.; mean=[sum(x[i] for x in losses)/len(losses) for i in range(3)]
  try: smi=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
  except Exception: smi='unavailable'
  print(f'iteration={it} fps={a.num_envs*a.steps_per_env/elapsed:.1f} mean_reward={torch.stack([x[4] for x in rollout]).mean().item():.3f} policy_loss={mean[0]:.4f} value_loss={mean[1]:.4f} entropy={mean[2]:.3f} vram_mib={mem:.0f} ram_mib={psutil.Process().memory_info().rss/2**20:.0f} nvidia_smi={smi}')
  torch.save({'model':model.state_dict(),'norm':norm.state_dict(),'optimizer':opt.state_dict(),'iteration':it},f'checkpoints/model_{it:04d}.pt')
if __name__=='__main__': main()
