"""Record a deterministic, zero-action upstream MJLab trace.

Run with `.venv-mjlab/bin/python` and `PYTHONPATH=upstream/microduck_rl/src`.
The reference environment is kept separate from the Genesis runtime.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg

def main():
 p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=100); p.add_argument('--output',default='logs/mujoco_reference.npz'); a=p.parse_args()
 cfg=make_microduck_velocity_env_cfg(); cfg.scene.num_envs=1; cfg.seed=123; cfg.scene.terrain.terrain_type='plane'; cfg.scene.terrain.terrain_generator=None
 # Disable stochastic events and observation noise for the deterministic trace.
 for term in cfg.events.values():
  if term.mode in ('startup','reset','interval'): term.func = lambda *args, **kwargs: None
 for group in cfg.observations.values():
  for term in group.terms.values(): term.noise=None; term.delay_min_lag=0; term.delay_max_lag=0
 env=ManagerBasedRlEnv(cfg,device='cpu'); reset=env.reset(); actions=torch.zeros((1,14),device='cpu')
 asset=env.scene['robot']; names=tuple(asset.joint_names)
 joint_ids=[names.index(n) for n in ('left_hip_yaw','left_hip_roll','left_hip_pitch','left_knee','left_ankle','neck_pitch','head_pitch','head_yaw','head_roll','right_hip_yaw','right_hip_roll','right_hip_pitch','right_knee','right_ankle')]
 rows=[]
 for i in range(a.steps):
  out=env.step(actions); obs,reward,terminated,truncated,info=out[:5]
  asset=env.scene['robot']; d=asset.data
  torque=getattr(d,'actuator_force',torch.zeros((1,14))).detach().cpu().numpy()
  rows.append({'time':i*float(env.step_dt),'action':actions.numpy().copy(),'joint_pos':d.joint_pos[:,joint_ids].numpy().copy(),'joint_vel':d.joint_vel[:,joint_ids].numpy().copy(),'base_pos':d.root_link_pos_w.numpy().copy(),'base_quat':d.root_link_quat_w.numpy().copy(),'base_lin_vel':d.root_link_lin_vel_w.numpy().copy(),'base_ang_vel':d.root_link_ang_vel_w.numpy().copy(),'actuator_torque':torque.copy(),'reward':reward.numpy().copy(),'terminated':terminated.numpy().copy(),'truncated':truncated.numpy().copy()})
 Path(a.output).parent.mkdir(exist_ok=True)
 np.savez_compressed(a.output, **{k:np.concatenate([r[k] for r in rows],axis=0) if k!='time' else np.array([r[k] for r in rows]) for k in rows[0]})
 print(f'wrote {a.output} ({a.steps} control steps)')
if __name__=='__main__': main()
