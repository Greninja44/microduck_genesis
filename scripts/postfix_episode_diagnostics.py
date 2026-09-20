#!/usr/bin/env python3
"""Episode, failure-state, reward, and stochastic-vs-mean diagnostics.

No optimizer updates are performed. The script uses the existing model_0005
checkpoint only as an inference policy.
"""
from __future__ import annotations
import argparse, csv, math, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
FIELDS=[('base_ang_vel',3),('projected_gravity',3),('joint_pos_rel',14),('joint_vel',14),('last_action',14),('twist',3),('head_pose',4),('body_pose',6)]

def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='checkpoints/postfix/model_0005.pt'); p.add_argument('--device',default='cuda'); p.add_argument('--num-envs',type=int,default=64); p.add_argument('--trials',type=int,default=20); a=p.parse_args()
 from microduck_genesis.env import MicroDuckGenesisEnv
 from microduck_genesis.ppo import ActorCritic,RunningNorm
 state=torch.load(a.checkpoint,map_location='cpu',weights_only=False); env=MicroDuckGenesisEnv(a.num_envs,True,a.device,seed=456,evaluation_command='forward',randomization=False,domain_randomization=False,sensor_noise=True,sensor_delay=True); dev=env.device
 model=ActorCritic().to(dev); model.load_state_dict(state['model']); model.eval(); norm=RunningNorm(61).to(dev); norm.load_state_dict(state['norm']); critic_norm=RunningNorm(76).to(dev); critic_norm.load_state_dict(state['critic_norm']); norm.eval(); critic_norm.eval()
 print('physics_dt',env.physics_dt,'control_dt',env.control_dt,'decimation',env.decimation,'max_episode_steps',env.max_episode_steps,'max_episode_seconds',env.max_episode_steps*env.control_dt,'num_envs',a.num_envs,'steps_per_rollout',24)
 # Reset-state distribution: 1000 fixed validation resets (randomization off).
 rolls=[]; pitches=[]; angs=[]; qoffs=[]; qds=[]
 for _ in range(16):
  env.reset(); q,qd,pos,quat,vel,ang=env._state(); rolls.append(torch.atan2(2*(quat[:,0]*quat[:,1]+quat[:,2]*quat[:,3]),1-2*(quat[:,1]**2+quat[:,2]**2)).cpu()); pitches.append(torch.asin((2*(quat[:,0]*quat[:,2]-quat[:,3]*quat[:,1])).clamp(-1,1)).cpu()); angs.append(ang.cpu()); qoffs.append((q-torch.tensor(__import__('microduck_genesis.robot',fromlist=['HOME_POSE']).HOME_POSE,device=dev)).cpu()); qds.append(qd.cpu())
 print('reset_samples',sum(x.numel() for x in rolls),'roll_std',torch.cat(rolls).std().item(),'pitch_std',torch.cat(pitches).std().item(),'ang_vel_std',torch.cat(angs).std().item(),'joint_offset_std',torch.cat(qoffs).std().item(),'joint_vel_std',torch.cat(qds).std().item())
 # One fresh stochastic rollout, collecting reset causes, episode durations, observations, rewards, values.
 obs=env.reset(); training_obs=[]; durations=[]; causes={'bad_orientation':0,'time_out':0,'nan_state':0}; ep=torch.zeros(a.num_envs,dtype=torch.long,device=dev); term_values=[]; training_values=[]
 for step in range(24):
  training_obs.append(obs.detach().cpu()); training_values.append(model.critic(critic_norm(env.critic_obs)).detach().cpu())
  with torch.no_grad(): act,_,_=model.act(norm(obs),critic_norm(env.critic_obs))
  try: nxt,rew,done,info=env.step(act)
  except FloatingPointError as e: print('stochastic_rollout_abort',step,e); break
  term_values.append({k:v.detach().cpu() for k,v in info['reward_terms'].items()}); terms=info['termination_terms']
  for k,v in terms.items(): causes[k]+=int(v.sum().item())
  ep+=1; ids=done.nonzero().flatten(); durations += [int(x)+1 for x in ep[ids].cpu().tolist()]; ep[ids]=0; obs=nxt
 training_obs=torch.cat(training_obs); training_values=torch.cat(training_values); torch.save(training_obs,ROOT/'runs/postfix_debug/training_observations.pt'); print('training_observations',training_obs.shape,'training_value_mean',training_values.mean().item(),'training_value_minmax',training_values.min().item(),training_values.max().item()); print('reset_causes_raw',causes,'episode_durations_steps',durations)
 if durations:
  d=torch.tensor(durations,dtype=torch.float); print('episode_duration_quantiles_steps',torch.quantile(d,torch.tensor([0,.25,.5,.75,.95,1.])).tolist(),'seconds', (torch.quantile(d,torch.tensor([0,.25,.5,.75,.95,1.]))*env.control_dt).tolist())
 # Reward inventory from this rollout.
 for k in sorted(term_values[0]):
  x=torch.cat([t[k] for t in term_values]); print('reward_term',k,'mean',x.mean().item(),'std',x.std().item(),'abs_fraction',x.abs().mean().item()/(sum(torch.cat([t[j] for t in term_values]).abs().mean().item() for j in term_values[0])+1e-9))
 # Deterministic forward failure trace for env 0.
 env.reset(); env.commands.command.zero_(); env.commands.command[:,0]=.2; obs=env._obs(); ref_mean=training_obs.mean(0); ref_std=training_obs.std(0).clamp_min(1e-5); out=ROOT/'runs/postfix_debug/forward_failure_trace.csv'; out.parent.mkdir(parents=True,exist_ok=True); names=['step','time','actual_vx','base_z','roll','pitch','ang_vel_norm','gravity_x','gravity_y','gravity_z','reward','done','bad_orientation','time_out','nan_state','critic_value']; names += [f'obs_{i}' for i in range(61)] + [f'mu_{i}' for i in range(14)] + [f'q_{i}' for i in range(14)] + [f'qd_{i}' for i in range(14)] + [f'torque_{i}' for i in range(14)] + ['foot_left_contact','foot_right_contact']
 f=out.open('w',newline=''); w=csv.DictWriter(f,fieldnames=names); w.writeheader(); first_ood=None; failure='';
 for step in range(100):
  q,qd,pos,quat,vel,ang=env._state(); gravity=env._projected_gravity(quat); mu=model.actor(norm(obs)); cv=model.critic(critic_norm(env.critic_obs)); z=((obs[0].cpu()-ref_mean)/ref_std); field_scores=[]; off=0
  for n,d in FIELDS: field_scores.append((n,float(z[off:off+d].abs().max()))); off+=d
  if first_ood is None and max(s for _,s in field_scores)>3: first_ood=(step,field_scores,obs[0].detach().cpu())
  foot_contact=env._contact_state()[2]
  row={'step':step,'time':step*env.control_dt,'actual_vx':float(vel[0,0]),'base_z':float(pos[0,2]),'roll':float(torch.atan2(2*(quat[0,0]*quat[0,1]+quat[0,2]*quat[0,3]),1-2*(quat[0,1]**2+quat[0,2]**2))),'pitch':float(torch.asin((2*(quat[0,0]*quat[0,2]-quat[0,3]*quat[0,1])).clamp(-1,1))),'ang_vel_norm':float(torch.linalg.vector_norm(ang[0])),'gravity_x':float(gravity[0,0]),'gravity_y':float(gravity[0,1]),'gravity_z':float(gravity[0,2]),'reward':0.0,'critic_value':float(cv[0]),'foot_left_contact':int(foot_contact[0,0]),'foot_right_contact':int(foot_contact[0,1])}; row.update({f'obs_{i}':float(v) for i,v in enumerate(obs[0])}); row.update({f'mu_{i}':float(v) for i,v in enumerate(mu[0])}); row.update({f'q_{i}':float(v) for i,v in enumerate(q[0])}); row.update({f'qd_{i}':float(v) for i,v in enumerate(qd[0])}); row.update({f'torque_{i}':float(v) for i,v in enumerate(env.last_bam_torque[0])}); w.writerow(row)
  try: obs,rew,done,info=env.step(mu)
  except FloatingPointError as e: failure=f'action_or_reward_failure_step_{step}: {e}'; break
  if bool(done[0]):
   print('deterministic_done_step',step+1,'terms',{k:bool(v[0]) for k,v in info['termination_terms'].items()}); break
 f.close(); print('forward_trace',out,'steps',step+1,'failure',failure)
 if first_ood: print('first_ood_step',first_ood[0],'time',first_ood[0]*env.control_dt,'fields',first_ood[1])
 else: print('first_ood_step none')
 # Deterministic and stochastic survival distributions, 20 trials each, reusing this scene.
 def trials(deterministic):
  surv=[]; dist=[]
  for trial in range(20):
   env.reset(); env.commands.command.zero_(); env.commands.command[:,0]=.2; o=env._obs(); x0=env._state()[2][0,0].item(); alive=0
   for k in range(150):
    with torch.no_grad():
     if deterministic: aa=model.actor(norm(o))
     else: aa,_,_=model.act(norm(o),critic_norm(env.critic_obs))
    try: o,rr,dd,ii=env.step(aa)
    except FloatingPointError: break
    alive=k+1; 
    if bool(dd[0]): break
   surv.append(alive*env.control_dt); dist.append(env._state()[2][0,0].item()-x0)
  return torch.tensor(surv),torch.tensor(dist)
 for det in (True,False) if a.trials else ():
  s,d=trials(det); print('deterministic' if det else 'stochastic','survival_quantiles',torch.quantile(s,torch.tensor([.25,.5,.75])).tolist(),'mean_distance',d.mean().item(),'falls_or_abort',int((s<150*env.control_dt).sum()))
 print('trace_training_nearest_distance', 'not computed')

if __name__=='__main__': main()
