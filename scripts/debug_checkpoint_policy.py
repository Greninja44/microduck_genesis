#!/usr/bin/env python3
"""Static and short-rollout diagnostics for a trained checkpoint.

This intentionally performs no optimizer updates and does not save a new
checkpoint. It compares checkpoint load parity and deterministic means against
raw observations collected through the normal environment path.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))

def stats(name,x):
    x=x.detach().float(); print(f'{name}: shape={tuple(x.shape)} min={x.min().item():.6g} max={x.max().item():.6g} mean={x.mean().item():.6g} std={x.std().item():.6g}')

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='checkpoints/postfix/model_0005.pt'); p.add_argument('--device',default='cuda'); p.add_argument('--samples',type=int,default=1000); p.add_argument('--command',choices=('stand','forward'),default='forward'); a=p.parse_args()
    from microduck_genesis.env import MicroDuckGenesisEnv
    from microduck_genesis.ppo import ActorCritic,RunningNorm
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    env=MicroDuckGenesisEnv(1,headless=True,device=a.device,seed=123,evaluation_command='forward',randomization=False,domain_randomization=False,sensor_noise=True,sensor_delay=True); dev=env.device
    model_a=ActorCritic().to(dev); model_a.load_state_dict(state['model']); model_a.eval(); model_b=ActorCritic().to(dev); model_b.load_state_dict(state['model']); model_b.eval(); norm=RunningNorm(61).to(dev); norm.load_state_dict(state['norm']); norm.eval()
    print('architecture:',model_a)
    diffs=[(k,(model_a.state_dict()[k]-model_b.state_dict()[k]).abs().max().item()) for k in model_a.state_dict()]; print('checkpoint_parameter_max_diff:',max(v for _,v in diffs)); print('checkpoint_keys_match:',list(model_a.state_dict())==list(model_b.state_dict())); print('norm_mean_std:',norm.mean.mean().item(),torch.sqrt(norm.var+1e-8).mean().item()); print('log_std:',model_a.log_std.min().item(),model_a.log_std.max().item())
    synthetic=torch.zeros(1,61,device=dev); stats('synthetic_actor_mean',model_a.actor(norm(synthetic))); print('synthetic_A_B_max_diff:',(model_a.actor(norm(synthetic))-model_b.actor(norm(synthetic))).abs().max().item())
    obs=env.reset(); raw=[]; means=[]; sampled=[]
    for i in range(a.samples):
        raw.append(obs.detach().cpu()); mu=model_a.actor(norm(obs)); means.append(mu.detach().cpu())
        act,_,_=model_a.act(norm(obs),env.critic_obs); sampled.append(act.detach().cpu())
        try: obs,rew,done,info=env.step(act)
        except FloatingPointError as e: print('rollout_aborted_at',i,type(e).__name__,e); break
    raw=torch.cat(raw); means=torch.cat(means); sampled=torch.cat(sampled); stats('rollout_observations',raw); stats('rollout_deterministic_means',means); stats('rollout_sampled_actions',sampled); print('mean_abs_mu:',means.abs().mean().item(),'p95_abs_mu:',torch.quantile(means.abs(),.95).item(),'p99_abs_mu:',torch.quantile(means.abs(),.99).item(),'max_abs_mu:',means.abs().max().item()); print('mean_abs_sampled:',sampled.abs().mean().item(),'max_abs_sampled:',sampled.abs().max().item())
    # Exact deterministic evaluation sequence, including delay buffers.
    obs=env.reset(); env.commands.command.zero_(); env.commands.command[:,0]=.2 if a.command=='forward' else 0.
    print('deterministic_forward_sequence:')
    for i in range(5):
        mu=model_a.actor(norm(obs)); stats(f'step_{i}_obs',obs); stats(f'step_{i}_mu',mu); print('step',i,'mu=',mu[0].detach().cpu().tolist(),'sensor_prev_ang_mean=',env.sensors.prev_ang.mean().item(),'sensor_prev_vel_mean=',env.sensors.prev_vel.mean().item(),'last_action_mean=',env.last_actions.mean().item())
        try: obs,rew,done,info=env.step(mu)
        except FloatingPointError as e: print('deterministic_aborted_at',i,type(e).__name__,e); break
if __name__=='__main__': main()
