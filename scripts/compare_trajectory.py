"""Compute short-horizon MuJoCo/Genesis trace metrics."""
from pathlib import Path
import csv, numpy as np
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
HOME_POSE=(0.0,-0.0872664626,-0.457924,-0.004940,0.452984,0.3490658504,0.3490658504,0.0,0.0,0.0,0.0872664626,0.457924,0.004940,-0.452984)
def quat_angle(a,b):
 d=np.abs(np.sum(a*b,axis=-1)); return 2*np.arccos(np.clip(d,-1,1))
def main():
 m=np.load('logs/mujoco_reference.npz'); g=np.load('logs/genesis_replay.npz'); Path('logs').mkdir(exist_ok=True)
 mujoco_joint_pos=m['joint_pos'] + np.asarray(HOME_POSE, dtype=np.float32)
 fields=[('joint_pos','joint_pos_mae'),('joint_vel','joint_vel_mae'),('base_lin_vel','base_lin_vel_mae'),('base_ang_vel','base_ang_vel_mae')]
 rows=[]
 horizons=tuple(n for n in (1,5,10,25,50,100) if n <= min(len(m['time']),len(g['time'])))
 for n in horizons:
  row={'control_steps':n}
  for key,name in fields:
   lhs=mujoco_joint_pos[n-1] if key=='joint_pos' else m[key][n-1]
   row[name]=float(np.mean(np.abs(lhs-g[key][n-1])))
  row['base_orientation_rad']=float(quat_angle(m['base_quat'][n-1],g['base_quat'][n-1]))
  rows.append(row)
 with open('logs/trajectory_parity.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
 print(rows)
if __name__=='__main__': main()
