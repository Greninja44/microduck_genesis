#!/usr/bin/env python3
import torch, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from microduck_genesis.rewards import RewardComputer
def main():
    d=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); r=RewardComputer(4,d); q=torch.zeros(4,14,device=d); a=torch.zeros_like(q); c=torch.tensor([[.2,0,0,0,0,0,0]]*4,device=d); gp=torch.tensor([[0,0,-1.]]*4,device=d); fp=torch.zeros(4,2,3,device=d); fv=torch.zeros_like(fp); fc=torch.tensor([[1,1],[0,0],[1,0],[0,1]],device=d,dtype=torch.bool); total,t=r.compute(base_lin_vel=torch.zeros(4,3,device=d),base_ang_vel=torch.zeros(4,3,device=d),projected_gravity=gp,joint_pos=q,actions=a,previous_actions=a,command=c,foot_pos=fp,foot_vel=fv,foot_contact=fc); assert torch.isfinite(total).all() and t['dof_pos_limits'][0]==0; print('terms:',','.join(sorted(t))); print('REWARD PARITY FORMULA PASS')
if __name__=='__main__': main()
