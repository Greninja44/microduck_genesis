#!/usr/bin/env python3
import argparse, torch, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from microduck_genesis.env import MicroDuckGenesisEnv
def main():
    p=argparse.ArgumentParser(); p.add_argument('--samples',type=int,default=10000); p.add_argument('--device',default='cuda'); a=p.parse_args()
    env=MicroDuckGenesisEnv(256,True,a.device,seed=17,sensor_noise=False,sensor_delay=False,domain_randomization=False); rows=[]
    for _ in range((a.samples+env.num_envs-1)//env.num_envs):
        env.reset(); *_,pos,quat,_,_=env._state(); yaw=2*torch.atan2(quat[:,3],quat[:,0]); rows.append(torch.cat((pos[:,:2],pos[:,2:3],yaw[:,None]),-1).detach().cpu())
    x=torch.cat(rows)[:a.samples]; print('field min max mean std')
    for i,n in enumerate(('x','y','z','yaw')): print(n,*[f'{v:.6f}' for v in (x[:,i].min(),x[:,i].max(),x[:,i].mean(),x[:,i].std(unbiased=False))])
    assert x[:,0].min()>=-.5 and x[:,0].max()<=.5 and x[:,1].min()>=-.5 and x[:,1].max()<=.5 and x[:,2].min()>=.12 and x[:,2].max()<=.13
    print('RESET DISTRIBUTION PASS')
if __name__=='__main__': main()
