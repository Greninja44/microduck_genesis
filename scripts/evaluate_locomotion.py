"""Persist deterministic batched forward-locomotion checkpoint metrics."""
import argparse, json, math, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
def q(x,p): return float(torch.quantile(x,torch.tensor(p,device=x.device)))
@torch.no_grad()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True); p.add_argument('--seeds',type=int,default=20); p.add_argument('--steps',type=int,default=1000); p.add_argument('--device',default='cuda'); a=p.parse_args()
 from microduck_genesis.env import MicroDuckGenesisEnv
 from microduck_genesis.ppo import ActorCritic,RunningNorm
 e=MicroDuckGenesisEnv(a.seeds,True,a.device,seed=20260921,evaluation_command='forward',randomization=False,domain_randomization=False,sensor_noise=False,sensor_delay=False); s=torch.load(a.checkpoint,map_location=e.device,weights_only=False); m=ActorCritic().to(e.device); n=RunningNorm(61).to(e.device); m.load_state_dict(s['model']); n.load_state_dict(s['norm']); o=e.reset(); e.commands.command.zero_(); e.commands.command[:,0]=.2
 _,_,start,_,_,_=e._state(); alive=torch.ones(a.seeds,dtype=torch.bool,device=e.device); survival=torch.full((a.seeds,),a.steps*e.control_dt,device=e.device); final=start.clone(); vals={k:[] for k in ('vx','roll','pitch','height','mu','action','torque')}; reasons={}
 for step in range(a.steps):
  mu=m.actor(n(o)); act=mu; o,r,d,info=e.step(act); pos,quat,vel=info['terminal_pos'],info['terminal_quat'],info['terminal_vel']; w,x,y,z=quat.unbind(-1); roll=torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y)).abs(); pitch=torch.asin((2*(w*y-z*x)).clamp(-1,1)).abs(); ix=alive
  for k,v in [('vx',vel[:,0]),('roll',roll),('pitch',pitch),('height',pos[:,2]),('mu',mu.abs().mean(-1)),('action',act.abs().mean(-1)),('torque',e.last_bam_torque.abs().mean(-1))]: vals[k].append(v[ix])
  hit=ix&d; final[hit]=pos[hit]; survival[hit]=(step+1)*e.control_dt
  for k,v in info['termination_terms'].items(): reasons[k]=reasons.get(k,0)+int((hit&v).sum())
  alive&=~d
  if not alive.any(): break
 final[alive]=e._state()[2][alive]; cat={k:torch.cat(v) for k,v in vals.items()}; dist=final[:,0]-start[:,0]; out={'checkpoint':a.checkpoint,'seeds':a.seeds,'survival':{'median':q(survival,.5),'p25':q(survival,.25),'p75':q(survival,.75)},'distance':{'median':q(dist,.5),'p25':q(dist,.25),'p75':q(dist,.75)},'mean_vx':float(cat['vx'].mean()),'tracking_rmse':float(torch.sqrt(((cat['vx']-.2)**2).mean())),'mean_abs_roll':float(cat['roll'].mean()),'mean_abs_pitch':float(cat['pitch'].mean()),'p95_abs_pitch':q(cat['pitch'],.95),'max_abs_pitch':float(cat['pitch'].max()),'mean_height':float(cat['height'].mean()),'min_height':float(cat['height'].min()),'mean_abs_actor_mean':float(cat['mu'].mean()),'max_abs_actor_mean':float(cat['mu'].max()),'mean_abs_action':float(cat['action'].mean()),'max_abs_action':float(cat['action'].max()),'mean_abs_torque':float(cat['torque'].mean()),'max_abs_torque':float(e.last_bam_torque.abs().max()),'torque_saturation_fraction':float((e.last_bam_torque.abs()>=.99*.834).float().mean()),'termination_reasons':reasons}
 Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
