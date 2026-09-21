import argparse, csv, sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from microduck_genesis.env import MicroDuckGenesisEnv
def main():
 p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=1000); p.add_argument('--device',default='cuda'); a=p.parse_args()
 env=MicroDuckGenesisEnv(1,True,a.device,seed=11); obs=env.reset(); mx=0.; resets=0
 Path('logs').mkdir(exist_ok=True)
 with open('logs/final_bam_smoke.csv','w',newline='') as f:
  w=csv.writer(f); w.writerow(['step','reward','torque_max','done','base_z'])
  for i in range(a.steps):
   action=torch.zeros((1,14),device=env.device); obs,r,d,info=env.step(action); c=info['critic_obs']; mx=max(mx,float(env.last_bam_torque.abs().max())); resets += int(d[0])
   assert obs.shape==(1,61) and c.shape==(1,76) and torch.isfinite(obs).all() and torch.isfinite(c).all() and torch.isfinite(r).all() and torch.isfinite(env.last_bam_torque).all()
   w.writerow([i,float(r[0]),float(env.last_bam_torque.abs().max()),int(d[0]),float(env._state()[2][0,2])])
 print(f'PASS steps={a.steps} max_abs_bam_torque={mx:.6f} terminations={resets}')
if __name__=='__main__': main()
