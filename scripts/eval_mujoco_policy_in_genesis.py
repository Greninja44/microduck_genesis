#!/usr/bin/env python3
"""Evaluate a supplied upstream 61D/14D ONNX policy without adapting its ABI."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--onnx',required=True);p.add_argument('--steps',type=int,default=1000);p.add_argument('--device',default='cpu');a=p.parse_args()
 try: import onnxruntime as ort
 except ImportError as e: raise SystemExit('onnxruntime is not installed; install it and supply a real exported MicroDuck policy.') from e
 import torch
 from microduck_genesis.env import MicroDuckGenesisEnv
 s=ort.InferenceSession(a.onnx,providers=['CPUExecutionProvider']); inputs=s.get_inputs()
 if len(inputs)!=1 or inputs[0].shape[-1] != 61: raise SystemExit(f'not the expected feed-forward MicroDuck policy: inputs={[(x.name,x.shape) for x in inputs]}')
 env=MicroDuckGenesisEnv(1,device=a.device,evaluation_command='forward',randomization=False); obs=env.reset(); falls=0; rewards=[]
 for step in range(a.steps):
  act=s.run(None,{inputs[0].name:obs.detach().cpu().numpy()})[0]
  obs,reward,done,info=env.step(torch.as_tensor(act,device=env.device,dtype=torch.float32));rewards.append(float(reward[0]));falls+=int(done[0])
  if done[0]: print(f'fall/termination at {step*env.control_dt:.2f}s: {[k for k,v in info["termination_terms"].items() if bool(v[0])]}')
 print({'survival_s':a.steps*env.control_dt,'mean_reward':sum(rewards)/len(rewards),'falls':falls})
if __name__=='__main__':main()
