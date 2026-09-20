"""Compute short-horizon MuJoCo/Genesis trace metrics."""
from pathlib import Path
import csv, numpy as np
def quat_angle(a,b):
 d=np.abs(np.sum(a*b,axis=-1)); return 2*np.arccos(np.clip(d,-1,1))
def main():
 m=np.load('logs/mujoco_reference.npz'); g=np.load('logs/genesis_replay.npz'); Path('logs').mkdir(exist_ok=True)
 fields=[('joint_pos','joint_pos_mae'),('joint_vel','joint_vel_mae'),('base_lin_vel','base_lin_vel_mae'),('base_ang_vel','base_ang_vel_mae')]
 rows=[]
 for n in (1,5,10,25,50,100):
  row={'control_steps':n}
  for key,name in fields: row[name]=float(np.mean(np.abs(m[key][n-1]-g[key][n-1])))
  row['base_orientation_rad']=float(quat_angle(m['base_quat'][n-1],g['base_quat'][n-1]))
  rows.append(row)
 with open('logs/trajectory_parity.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
 print(rows)
if __name__=='__main__': main()
