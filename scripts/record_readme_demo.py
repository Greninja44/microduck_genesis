"""Record real MicroDuck Genesis media for the README (not a policy rollout)."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',default='docs/assets');a=p.parse_args();out=ROOT/a.output_dir;out.mkdir(parents=True,exist_ok=True)
 import genesis as gs
 from microduck_genesis.robot import HOME_POSE,add_microduck,set_home_pose
 from microduck_genesis.scene import add_ground,make_scene
 gs.init(backend=gs.gpu,seed=7); scene=make_scene(dt=.005);add_ground(scene);robot=add_microduck(scene)
 cam=scene.add_camera(res=(640,480),pos=(.62,-.82,.42),lookat=(0,0,.12),fov=42)
 side=scene.add_camera(res=(640,480),pos=(.0,-.92,.22),lookat=(0,0,.12),fov=42)
 scene.build(); ids=set_home_pose(robot);robot.set_dofs_kp(np.full(14,.7),dofs_idx_local=ids);robot.set_dofs_kv(np.full(14,.06),dofs_idx_local=ids)
 home=np.asarray(HOME_POSE);frames=[]
 for i in range(75):
  t=i/15; target=home.copy();target[[2,3,4,11,12,13]]+=np.array([.06,.10,-.06,-.06,-.10,.06])*math.sin(2*math.pi*.45*t)
  for _ in range(4): robot.control_dofs_position(target,dofs_idx_local=ids);scene.step()
  frame=cam.render()[0];frames.append(frame)
  if i==12:imageio.imwrite(out/'microduck_genesis.png',frame)
  if i==42:imageio.imwrite(out/'microduck_genesis_side.png',side.render()[0])
 imageio.mimsave(out/'microduck_genesis.gif',frames,fps=15,loop=0)
 rows=[json.loads((ROOT/f'runs/learning20_post_bam_fix/evaluations/iter_{i:04d}.json').read_text()) for i in (0,5,10,15,20)];x=[0,5,10,15,20]
 fig,ax=plt.subplots(1,3,figsize=(10,3));
 for a_,k,title in zip(ax,('survival','mean_vx','mean_abs_pitch'),('Median survival (s)','Mean vx (m/s)','Mean |pitch| (rad)')):
  y=[r[k]['median'] if isinstance(r[k],dict) else r[k] for r in rows];a_.plot(x,y,'o-',color='#2563a6');a_.set(title=title,xlabel='PPO iteration');a_.grid(alpha=.25)
 fig.tight_layout();fig.savefig(out/'learning20_evaluation.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
