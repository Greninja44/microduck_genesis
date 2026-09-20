import torch
from microduck_genesis.actions import ACTION_DIM, action_to_targets
from microduck_genesis.commands import CommandGenerator
from microduck_genesis.observations import EXPECTED_OBSERVATION_DIM, build_actor_observation, validate_observation
from microduck_genesis.observations import ACTOR_FIELDS
from microduck_genesis.critic_observations import CRITIC_FIELDS, CRITIC_OBSERVATION_DIM, build_critic_observation
from microduck_genesis.robot import JOINT_NAMES
from microduck_genesis.rewards import RewardComputer
from microduck_genesis.terminations import compute_terminations

def test_observation_dimension_order_and_nan_guard():
 z=torch.zeros(2,14);obs=build_actor_observation(base_ang_vel=torch.ones(2,3),projected_gravity=torch.zeros(2,3),joint_pos=z,joint_vel=z,last_action=z,command=torch.zeros(2,13))
 assert obs.shape==(2,EXPECTED_OBSERVATION_DIM) and torch.all(obs[:,:3]==1)
 obs[0,0]=float('nan')
 try: validate_observation(obs);assert False
 except FloatingPointError: pass
def test_actions_and_deterministic_commands():
 assert action_to_targets(torch.zeros(3,ACTION_DIM)).shape==(3,ACTION_DIM)
 assert ACTION_DIM == 14
 from microduck_genesis.actions import ACTION_SCALE
 assert ACTION_SCALE == 1.0 and len(JOINT_NAMES) == ACTION_DIM
 a=CommandGenerator(4,torch.device('cpu'),123).command;b=CommandGenerator(4,torch.device('cpu'),123).command;assert torch.equal(a,b)

def test_critic_contract_and_privilege_boundary():
 assert EXPECTED_OBSERVATION_DIM == 61 and CRITIC_OBSERVATION_DIM == 76
 assert tuple(x.name for x in ACTOR_FIELDS) == ('base_ang_vel','projected_gravity','joint_pos_rel','joint_vel_rel','last_action','twist_command','head_pose_command','body_pose_command')
 assert tuple(x.name for x in CRITIC_FIELDS)[:3] == ('base_lin_vel','base_ang_vel','projected_gravity')
 z=lambda n: torch.zeros(2,n)
 c=build_critic_observation(base_lin_vel=z(3),base_ang_vel=z(3),projected_gravity=z(3),joint_pos=z(14),joint_vel=z(14),last_action=z(14),command=z(13),foot_height=z(2),foot_air_time=z(2),foot_contact=z(2),foot_contact_forces=z(6))
 assert c.shape==(2,76)
def test_rewards_and_termination():
 r=RewardComputer(2,torch.device('cpu'));z=torch.zeros(2,14);total,terms=r.compute(base_lin_vel=torch.zeros(2,3),base_ang_vel=torch.zeros(2,3),projected_gravity=torch.tensor([[0.,0.,-1.],[0.,0.,-1.]]),joint_pos=torch.tensor([(.0,-.0872664626,-.457924,-.004940,.452984,.3490658504,.3490658504,0.,0.,0.,.0872664626,.457924,.004940,-.452984)]*2),actions=z,previous_actions=z,command=torch.zeros(2,13));assert torch.allclose(total,torch.stack(tuple(terms.values())).sum(0))
    d,t=compute_terminations(base_pos=torch.tensor([[0.,0.,.12],[0.,0.,.12]]),projected_gravity=torch.tensor([[0.,0.,1.],[0.,0.,1.]]),state_tensors=(z,),episode_steps=torch.tensor([0,1000]),max_episode_steps=1000);assert not d[0] and d[1] and t['time_out'][1]
