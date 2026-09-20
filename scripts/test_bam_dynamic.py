"""Compare upstream BAM output with the command emitted by the Genesis env."""
from pathlib import Path
import csv, sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from microduck_genesis.bam import bam_motor_torque, bam_torque, bam_voltage
from microduck_genesis.env import MicroDuckGenesisEnv
from microduck_genesis.actions import action_to_targets

def main():
 env=MicroDuckGenesisEnv(1,True,'cuda',seed=3,randomization=False,sensor_noise=False,sensor_delay=False)
 q=torch.tensor([[0.1]*14],device=env.device); qd=torch.tensor([[0.2]*14],device=env.device); target=torch.tensor([[0.15]*14],device=env.device)
 got=env.compute_bam_torque(target,q,qd)
 vin=torch.full_like(q,7.4); ref=bam_torque(target,q,qd,vin)
 rows=[]
 for j,(a,b) in enumerate(zip(ref[0].tolist(),got[0].tolist())):
  e=abs(a-b); rows.append({'joint':j,'q':q[0,j].item(),'qd':qd[0,j].item(),'target':target[0,j].item(),'upstream_tau':a,'genesis_commanded_tau':b,'abs_error':e})
 Path('logs').mkdir(exist_ok=True)
 with open('logs/bam_dynamic_parity.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
 print('max_dynamic_command_error=',max(r['abs_error'] for r in rows))
 assert max(r['abs_error'] for r in rows) < 1e-6
if __name__=='__main__': main()
