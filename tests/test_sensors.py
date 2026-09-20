import torch
from microduck_genesis.sensors import ActorSensorModel
def test_sensor_noise_seed_and_ranges():
 a=ActorSensorModel(3,'cpu',7); b=ActorSensorModel(3,'cpu',7); z=torch.zeros(3,3); q=torch.zeros(3,14)
 ao,ag,av=a.apply(z,z,q); bo,bg,bv=b.apply(z,z,q)
 assert torch.equal(ao,bo) and torch.equal(ag,bg) and torch.equal(av,bv)
 assert ao.abs().max() <= .03 and ag.abs().max() <= .01 and av.abs().max() <= .25
def test_sensor_reset_is_per_environment():
 s=ActorSensorModel(3,'cpu',1); s.apply(torch.ones(3,3),torch.ones(3,3),torch.ones(3,14)); s.reset(torch.tensor([1])); assert torch.all(s.prev_vel[1]==0) and torch.all(s.prev_vel[[0,2]]==1)
