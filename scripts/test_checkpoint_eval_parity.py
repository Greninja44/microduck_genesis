#!/usr/bin/env python3
"""Regression test for actor architecture, checkpoint, and normalization parity."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='checkpoints/postfix/model_0005.pt'); p.add_argument('--device',default='cpu'); a=p.parse_args()
    from microduck_genesis.ppo import ActorCritic,RunningNorm
    state=torch.load(a.checkpoint,map_location=a.device,weights_only=False); a1=ActorCritic().to(a.device); a2=ActorCritic().to(a.device); a1.load_state_dict(state['model']); a2.load_state_dict(state['model']); n1=RunningNorm(61).to(a.device); n2=RunningNorm(61).to(a.device); n1.load_state_dict(state['norm']); n2.load_state_dict(state['norm']); x=torch.randn(17,61,device=a.device); y1=a1.actor(n1(x)); y2=a2.actor(n2(x)); diff=(y1-y2).abs().max().item(); pd=max((a1.state_dict()[k]-a2.state_dict()[k]).abs().max().item() for k in a1.state_dict()); nd=max((n1.state_dict()[k]-n2.state_dict()[k]).abs().max().item() for k in n1.state_dict()); assert pd==0 and nd==0 and diff==0, (pd,nd,diff); print(f'PASS checkpoint parameter max diff={pd:.3g} normalization max diff={nd:.3g} actor output max diff={diff:.3g} obs_dim=61 architecture=ActorCritic(61->14)')
if __name__=='__main__': main()
