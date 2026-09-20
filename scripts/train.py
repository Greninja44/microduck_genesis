#!/usr/bin/env python3
"""Small PPO runner with explicit post-fix numerical diagnostics."""
from __future__ import annotations
import argparse, csv, math, sys, time
from pathlib import Path
import psutil
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

def finite(name, value):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f'{name} contains NaN/Inf')

def main():
    from microduck_genesis.env import MicroDuckGenesisEnv
    from microduck_genesis.ppo import ActorCritic, RunningNorm
    p=argparse.ArgumentParser(); p.add_argument('--num-envs',type=int,default=256); p.add_argument('--device',default='cuda'); p.add_argument('--seed',type=int,default=0); p.add_argument('--headless',action='store_true'); p.add_argument('--iterations',type=int,default=5); p.add_argument('--checkpoint'); p.add_argument('--resume',action='store_true'); p.add_argument('--steps-per-env',type=int,default=24); p.add_argument('--evaluation-command',choices=('stand','forward','backward','turn_left','turn_right'),default='forward'); p.add_argument('--no-randomization',action='store_true'); p.add_argument('--checkpoint-dir',default='checkpoints'); p.add_argument('--metrics',default='runs/postfix_validation/training_metrics.csv'); a=p.parse_args()
    env=MicroDuckGenesisEnv(a.num_envs,a.headless,a.device,a.seed,evaluation_command=a.evaluation_command,randomization=not a.no_randomization,domain_randomization=not a.no_randomization); dev=env.device
    model=ActorCritic().to(dev); norm=RunningNorm(61).to(dev); critic_norm=RunningNorm(76).to(dev); opt=torch.optim.Adam(model.parameters(),lr=1e-3); start=0
    if a.resume:
        if not a.checkpoint: raise ValueError('--resume requires --checkpoint')
        state=torch.load(a.checkpoint,map_location=dev,weights_only=False); model.load_state_dict(state['model']); norm.load_state_dict(state['norm']); critic_norm.load_state_dict(state.get('critic_norm',critic_norm.state_dict())); opt.load_state_dict(state['optimizer']); start=int(state.get('iteration',0))
    elif a.checkpoint: raise ValueError('checkpoint requires --resume')
    ckdir=Path(a.checkpoint_dir); ckdir.mkdir(parents=True,exist_ok=True); mp=Path(a.metrics); mp.parent.mkdir(parents=True,exist_ok=True)
    fields=['iteration','mean_reward','episode_length','termination_rate','mean_vx','tracking_rmse','mean_abs_actor_mean','max_abs_actor_mean','mean_abs_action','max_abs_action','action_std','mean_abs_bam_target','max_abs_bam_target','mean_abs_bam_torque','max_abs_bam_torque','torque_saturation_fraction','policy_loss','value_loss','entropy','gradient_norm','actor_parameter_norm','critic_parameter_norm','log_std_min','log_std_max','ppo_fps','vram_mib','ram_mib']; new=not mp.exists() or mp.stat().st_size==0; mf=mp.open('a',newline=''); writer=csv.DictWriter(mf,fieldnames=fields); writer.writeheader() if new else None
    obs=env.reset()
    for local in range(a.iterations):
        iteration=start+local+1; env.rewarder.set_training_steps((iteration-1)*a.steps_per_env); rollout=[]; started=time.perf_counter(); av=[]; am=[]; tg=[]; tv=[]; vv=[]; dv=[]
        for _ in range(a.steps_per_env):
            finite('actor observation',obs); finite('critic observation',env.critic_obs)
            norm.update(obs); critic_norm.update(env.critic_obs)
            with torch.no_grad():
                mu=model.actor(norm(obs)); act,lp,val=model.act(norm(obs),critic_norm(env.critic_obs))
            finite('actor mean',mu); finite('sampled action',act); nxt,rew,done,info=env.step(act); finite('reward',rew); finite('next observation',nxt); finite('critic observation',info['critic_obs'])
            from microduck_genesis.actions import action_to_targets
            av.append(act.detach()); am.append(mu.detach()); tg.append(action_to_targets(act).detach()); tv.append(env.last_bam_torque.detach().clone()); _,_,_,_,vel,_=env._state(); finite('base state',vel); vv.append(vel[:,0].detach()); dv.append(done.detach()); rollout.append((obs,env.critic_obs.clone(),act,lp,val,rew,done,info['termination_terms']['time_out'].detach())); obs=nxt
        with torch.no_grad(): last=model.critic(critic_norm(env.critic_obs)).squeeze(-1); finite('critic value',last)
        returns=[]; gae=torch.zeros(a.num_envs,device=dev)
        for _,_,_,_,val,rew,done,time_out in reversed(rollout):
            rew=rew+.99*val*time_out.float(); nd=(~done).float(); gae=rew+.99*last*nd-val+.99*.95*nd*gae; returns.append(gae+val); last=val
        R=torch.stack(list(reversed(returns))); O=torch.stack([x[0] for x in rollout]); C=torch.stack([x[1] for x in rollout]); A=torch.stack([x[2] for x in rollout]); LP=torch.stack([x[3] for x in rollout]); V=torch.stack([x[4] for x in rollout]); O,C,A,LP,V,R=[x.reshape(-1,*x.shape[2:]) if x.ndim>2 else x.reshape(-1) for x in (O,C,A,LP,V,R)]
        adv=(R-V); adv=(adv-adv.mean())/(adv.std()+1e-8); idx=torch.randperm(len(O),device=dev); losses=[]; grads=[]
        for _ in range(5):
            for batch in idx.chunk(4):
                logp,ent,value=model.evaluate(norm(O[batch]),A[batch],critic_norm(C[batch])); ratio=(logp-LP[batch]).exp(); pol=-torch.minimum(ratio*adv[batch],torch.clamp(ratio,.8,1.2)*adv[batch]).mean(); vl=torch.maximum((value-R[batch]).square(),(torch.clamp(value,V[batch]-.2,V[batch]+.2)-R[batch]).square()).mean(); loss=pol+vl-.01*ent.mean(); finite('policy loss',pol); finite('value loss',vl); finite('entropy',ent); opt.zero_grad(); loss.backward(); g=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); finite('gradient norm',torch.as_tensor(g)); opt.step(); losses.append((pol.item(),vl.item(),ent.mean().item())); grads.append(float(g))
        elapsed=time.perf_counter()-started; mean=[sum(x[i] for x in losses)/len(losses) for i in range(3)]; acts=torch.cat(av); actor_means=torch.cat(am); targets=torch.cat(tg); torques=torch.cat(tv); vxs=torch.cat(vv); dones=torch.cat(dv); finite('action diagnostics',acts); finite('actor mean diagnostics',actor_means); finite('target diagnostics',targets); finite('torque diagnostics',torques); sat=(torques.abs()>=.99*.834).float().mean().item(); cmd=env.commands.command[:,0].mean(); metrics={'iteration':iteration,'mean_reward':torch.stack([x[5] for x in rollout]).mean().item(),'episode_length':(1-dones.float().mean().item())*a.steps_per_env,'termination_rate':dones.float().mean().item(),'mean_vx':vxs.mean().item(),'tracking_rmse':torch.sqrt(torch.square(vxs-cmd).mean()).item(),'mean_abs_actor_mean':actor_means.abs().mean().item(),'max_abs_actor_mean':actor_means.abs().max().item(),'mean_abs_action':acts.abs().mean().item(),'max_abs_action':acts.abs().max().item(),'action_std':acts.std().item(),'mean_abs_bam_target':targets.abs().mean().item(),'max_abs_bam_target':targets.abs().max().item(),'mean_abs_bam_torque':torques.abs().mean().item(),'max_abs_bam_torque':torques.abs().max().item(),'torque_saturation_fraction':sat,'policy_loss':mean[0],'value_loss':mean[1],'entropy':mean[2],'gradient_norm':sum(grads)/len(grads),'actor_parameter_norm':math.sqrt(sum(float(x.detach().square().sum()) for x in model.actor.parameters())),'critic_parameter_norm':math.sqrt(sum(float(x.detach().square().sum()) for x in model.critic.parameters())),'log_std_min':model.log_std.min().item(),'log_std_max':model.log_std.max().item(),'ppo_fps':a.num_envs*a.steps_per_env/elapsed,'vram_mib':torch.cuda.max_memory_allocated(dev)/2**20 if dev.type=='cuda' else 0,'ram_mib':psutil.Process().memory_info().rss/2**20}; writer.writerow(metrics); mf.flush(); print(' '.join(f'{k}={v:.6g}' if isinstance(v,float) else f'{k}={v}' for k,v in metrics.items())); torch.save({'model':model.state_dict(),'norm':norm.state_dict(),'critic_norm':critic_norm.state_dict(),'optimizer':opt.state_dict(),'iteration':iteration},ckdir/f'model_{iteration:04d}.pt')
    mf.close()
if __name__=='__main__': main()
