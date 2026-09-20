"""Privileged critic ABI recovered from upstream velocity_env_cfg."""
from __future__ import annotations
import torch
from dataclasses import dataclass
from .robot import HOME_POSE

@dataclass(frozen=True)
class CriticField:
    name: str
    size: int
    source: str

# mjlab term insertion order. Sensor terms are represented by Genesis contact
# and kinematic equivalents; the actor never receives these fields.
CRITIC_FIELDS = (
    CriticField("base_lin_vel",3,"root link velocity"), CriticField("base_ang_vel",3,"IMU sensor"),
    CriticField("projected_gravity",3,"true base gravity"), CriticField("joint_pos_rel",14,"unbiased encoders"),
    CriticField("joint_vel_rel",14,"true joint velocity"), CriticField("last_action",14,"previous action"),
    CriticField("twist_command",3,"generated command"), CriticField("foot_height",2,"terrain ray sensor"),
    CriticField("foot_air_time",2,"contact sensor"), CriticField("foot_contact",2,"contact sensor"),
    CriticField("foot_contact_forces",6,"contact force sensor"), CriticField("head_pose_command",4,"generated command"),
    CriticField("body_pose_command",6,"generated command"),
)
CRITIC_OBSERVATION_DIM = sum(x.size for x in CRITIC_FIELDS)
assert CRITIC_OBSERVATION_DIM == 76

def build_critic_observation(*, base_lin_vel, base_ang_vel, projected_gravity,
                             joint_pos, joint_vel, last_action, command,
                             foot_height, foot_air_time, foot_contact,
                             foot_contact_forces, encoder_bias=None):
    home=torch.as_tensor(HOME_POSE,dtype=joint_pos.dtype,device=joint_pos.device)
    pos=joint_pos-home  # critic is unbiased, unlike actor
    if foot_contact_forces.shape[-1] != 6:
        foot_contact_forces=foot_contact_forces.reshape(joint_pos.shape[0],-1)[...,:6]
    out=torch.cat((base_lin_vel,base_ang_vel,projected_gravity,pos,joint_vel,last_action,
                   command[...,:3],foot_height,foot_air_time,foot_contact,foot_contact_forces,
                   command[...,3:7],command[...,7:13]),dim=-1)
    if out.shape[-1] != CRITIC_OBSERVATION_DIM or not torch.isfinite(out).all():
        raise AssertionError(f"invalid critic observation {tuple(out.shape)}")
    return out
