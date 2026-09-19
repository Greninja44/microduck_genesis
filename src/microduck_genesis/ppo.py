"""Small PyTorch PPO backend matching the upstream feed-forward configuration."""
from __future__ import annotations
import torch
from torch import nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, obs_dim=61, action_dim=14):
        super().__init__()
        def net(out):
            layers=[]; n=obs_dim
            for width in (512,256,128): layers += [nn.Linear(n,width),nn.ELU()]; n=width
            layers.append(nn.Linear(n,out)); return nn.Sequential(*layers)
        self.actor, self.critic = net(action_dim), net(1); self.log_std=nn.Parameter(torch.zeros(action_dim))
    def distribution(self, obs): return Normal(self.actor(obs), self.log_std.exp())
    def act(self, obs):
        d=self.distribution(obs); a=d.sample(); return a,d.log_prob(a).sum(-1),self.critic(obs).squeeze(-1)
    def evaluate(self, obs, actions):
        d=self.distribution(obs); return d.log_prob(actions).sum(-1),d.entropy().sum(-1),self.critic(obs).squeeze(-1)

class RunningNorm(nn.Module):
    def __init__(self, dim): super().__init__(); self.register_buffer('mean',torch.zeros(dim)); self.register_buffer('var',torch.ones(dim)); self.register_buffer('count',torch.tensor(1e-4))
    @torch.no_grad()
    def update(self,x):
        mean=x.mean(0); var=x.var(0,unbiased=False); count=torch.tensor(float(x.shape[0]),device=x.device); delta=mean-self.mean; total=self.count+count
        self.mean.add_(delta*count/total); self.var.copy_((self.var*self.count+var*count+delta.square()*self.count*count/total)/total); self.count.copy_(total)
    def forward(self,x): return (x-self.mean)/(self.var+1e-8).sqrt()
