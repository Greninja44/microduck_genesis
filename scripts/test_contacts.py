"""Isolated Genesis contact API diagnostic for MicroDuck feet and self-contact."""
import argparse, sys, torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from microduck_genesis.env import MicroDuckGenesisEnv

def main():
 p=argparse.ArgumentParser(); p.add_argument('--device',default='cuda'); a=p.parse_args()
 env=MicroDuckGenesisEnv(1,True,a.device,randomization=False)
 assert len(env.foot_link_ids)==2
 for _ in range(4):
  c=env._contact_state(); foot_pos,foot_vel,contact,self_collision,force=c
  assert contact.shape==(1,2) and force.shape[-1]==3
  assert torch.isfinite(force).all()
  env.scene.step()
 print('CONTACT API PASS', 'foot_links=',env.foot_link_ids, 'contact=',contact.tolist(), 'force_norm=',torch.linalg.vector_norm(force,dim=-1).tolist())
if __name__=='__main__': main()
