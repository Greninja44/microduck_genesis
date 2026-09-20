"""Genesis batched MicroDuck velocity environment.

One Genesis Scene is built with ``n_envs``; no Python loop creates worlds.
"""
from __future__ import annotations
import torch
from .actions import ACTION_DIM, action_to_targets, resolve_action_mapping
from .commands import CommandGenerator
from .observations import build_actor_observation, validate_observation
from .critic_observations import build_critic_observation
from .randomization import make_randomization, reset_randomization
from .rewards import RewardComputer
from .robot import HOME_POSE, add_microduck
from .scene import add_ground, make_scene
from .terminations import compute_terminations
from .sensors import ActorSensorModel
from .bam import bam_torque

class MicroDuckGenesisEnv:
    observation_dim, action_dim = 61, ACTION_DIM
    physics_dt, control_dt, decimation = .005, .02, 4
    def __init__(self, num_envs: int = 1, headless: bool = True, device: str = 'cuda', seed: int = 0,
                 evaluation_command: str | None = None, max_episode_seconds: float = 20., randomization: bool = True,
                 domain_randomization: bool | None = None, sensor_noise: bool = True, sensor_delay: bool = True):
        import genesis as gs
        self.num_envs, self.seed = num_envs, seed
        self.randomization_enabled = randomization if domain_randomization is None else domain_randomization
        self.sensor_noise_enabled, self.sensor_delay_enabled = sensor_noise, sensor_delay
        self.actuator_mode = "bam_torque"
        requested_cuda = device.startswith('cuda') and torch.cuda.is_available()
        self.device = torch.device('cuda' if requested_cuda else 'cpu')
        gs.init(backend=gs.gpu if requested_cuda else gs.cpu, seed=seed)
        self.scene = make_scene(show_viewer=not headless, dt=self.physics_dt); self.ground = add_ground(self.scene)
        self.robot = add_microduck(self.scene, visualize_contact=not headless); self.scene.build(n_envs=num_envs, env_spacing=(1., 1.))
        self.mapping = resolve_action_mapping(self.robot); self.servo_ids = [m.genesis_dof_index for m in self.mapping]
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.commands = CommandGenerator(num_envs, self.device, seed, evaluation_command)
        self.dr = make_randomization(num_envs, self.device); self.rewarder = RewardComputer(num_envs, self.device, self.control_dt)
        self.last_actions = torch.zeros(num_envs, ACTION_DIM, device=self.device); self.episode_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.max_episode_steps = round(max_episode_seconds / self.control_dt)
        self._joint_vel_lag = torch.zeros_like(self.last_actions)
        self.sensors = ActorSensorModel(num_envs, self.device, seed)
        self._foot_air_time = torch.zeros(num_envs, 2, device=self.device)
        self.critic_obs = torch.zeros(num_envs, 76, device=self.device)
        self.last_bam_torque = torch.zeros(num_envs, ACTION_DIM, device=self.device)
        # The upstream names are sites/geoms (`left_foot`, `right_foot`). The
        # Genesis MJCF importer exposes their parent rigid links as
        # `ankle_left` and `ankle_right`.
        self.foot_link_ids = [self.robot.get_link(name).idx_local for name in ('ankle_left', 'ankle_right')]
        self.reset()

    def _state(self):
        q = self.robot.get_dofs_position(self.servo_ids); qd = self.robot.get_dofs_velocity(self.servo_ids)
        pos, quat = self.robot.get_pos(), self.robot.get_quat(); vel, ang = self.robot.get_vel(relative=True), self.robot.get_ang()
        return tuple(x.to(self.device) if isinstance(x, torch.Tensor) else torch.as_tensor(x, device=self.device) for x in (q, qd, pos, quat, vel, ang))

    @staticmethod
    def _projected_gravity(quat: torch.Tensor) -> torch.Tensor:
        # Genesis uses wxyz quaternions for MJCF state. Rotate world gravity
        # [0, 0, -1] by q^-1.  This matches MJLab's
        # ``asset.data.projected_gravity_b``: upright is [0, 0, -1].
        w, x, y, z = quat.unbind(-1)
        return torch.stack((2*(w*y-x*z), -2*(y*z+w*x), -(1-2*(x*x+y*y))), -1)

    def _obs(self):
        q, qd, pos, quat, vel, ang = self._state()
        gravity = self._projected_gravity(quat)
        sensed_ang, sensed_gravity, sensed_vel = self.sensors.apply(
            ang, gravity, qd, noise=self.sensor_noise_enabled,
            delay=self.sensor_delay_enabled)
        obs = build_actor_observation(base_ang_vel=sensed_ang, projected_gravity=sensed_gravity, joint_pos=q,
            joint_vel=sensed_vel, last_action=self.last_actions, command=self.commands.command, encoder_bias=self.dr.encoder_bias if self.randomization_enabled else None)
        self._joint_vel_lag.copy_(sensed_vel)
        self._last_state = (q, qd, pos, quat, vel, ang, gravity)
        return obs

    def _contact_state(self):
        """Return batched foot positions/velocities/contact and self-contact."""
        foot_pos = self.robot.get_links_pos(self.foot_link_ids)
        foot_vel = self.robot.get_links_vel(self.foot_link_ids)
        force = self.robot.get_links_net_contact_force()
        foot_force = force[:, self.foot_link_ids]
        foot_contact = torch.linalg.vector_norm(foot_force, dim=-1) > 1e-3
        contacts = self.robot.get_contacts(with_entity=self.robot, exclude_self_contact=False, is_padded=True)
        valid = contacts['valid_mask']
        self_collision = valid.any(dim=-1)
        return foot_pos, foot_vel, foot_contact, self_collision, foot_force

    def compute_bam_torque(self, targets: torch.Tensor, q: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
        """Return the exact vectorized command sent to Genesis force control."""
        vin = torch.full_like(q, 7.4)
        torque = bam_torque(targets, q, qd, vin, friction_scale=self.dr.friction_scale)
        if torque.shape != (self.num_envs, ACTION_DIM) or not torch.isfinite(torque).all():
            raise AssertionError("invalid BAM torque command")
        return torque

    def reset(self, env_ids: torch.Tensor | None = None):
        ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int32) if env_ids is None else env_ids.to(self.device, torch.int32)
        if len(ids) == 0: return self._obs()
        home = torch.tensor(HOME_POSE, device=self.device).expand(len(ids), -1)
        self.robot.set_dofs_position(home, dofs_idx_local=self.servo_ids, envs_idx=ids, zero_velocity=True)
        self.robot.set_dofs_velocity(torch.zeros_like(home), dofs_idx_local=self.servo_ids, envs_idx=ids)
        # Match mjlab reset_root_state_uniform exactly for the MicroDuck
        # velocity task: x/y ∈ [-.5,.5], z ∈ [.12,.13], yaw ∈ [-pi,pi],
        # with zero roll/pitch and zero root velocity. This is upstream reset
        # distribution, independent of optional model-field randomization.
        u = torch.rand((len(ids), 4), device=self.device, generator=self.generator)
        xy = -0.5 + u[:, :2]
        z = 0.12 + 0.01 * u[:, 2]
        yaw = -torch.pi + 2.0 * torch.pi * u[:, 3]
        self.robot.set_pos(torch.cat((xy, z[:, None]), dim=-1), envs_idx=ids, zero_velocity=True)
        quat = torch.stack((torch.cos(yaw/2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw/2)), -1)
        self.robot.set_quat(quat, envs_idx=ids, zero_velocity=True)
        self.last_actions[ids] = 0; self._joint_vel_lag[ids] = 0; self._foot_air_time[ids] = 0; self.episode_steps[ids] = 0; self.rewarder.reset(ids); self.sensors.reset(ids); self.last_bam_torque[ids] = 0
        self.commands.resample(ids)
        if self.randomization_enabled: reset_randomization(self.dr, ids, self.generator)
        return self._obs()

    def step(self, actions: torch.Tensor):
        if actions.shape != (self.num_envs, ACTION_DIM): raise AssertionError(f'expected {(self.num_envs, ACTION_DIM)}, got {tuple(actions.shape)}')
        actions = actions.to(self.device); targets = action_to_targets(actions)
        if self.actuator_mode != "bam_torque":
            raise RuntimeError("only bam_torque is permitted in the training environment")
        for _ in range(self.decimation):
            q_now = self.robot.get_dofs_position(self.servo_ids)
            qd_now = self.robot.get_dofs_velocity(self.servo_ids)
            torque = self.compute_bam_torque(targets, q_now, qd_now)
            self.last_bam_torque.copy_(torque)
            self.robot.control_dofs_force(torque, dofs_idx_local=self.servo_ids)
            self.scene.step()
        q, qd, pos, quat, vel, ang = self._state(); gravity = self._projected_gravity(quat)
        self.episode_steps += 1
        foot_pos, foot_vel, foot_contact, self_collision, foot_force = self._contact_state()
        rewards, terms = self.rewarder.compute(base_lin_vel=vel, base_ang_vel=ang, projected_gravity=gravity, joint_pos=q, actions=actions, previous_actions=self.last_actions, command=self.commands.command, foot_pos=foot_pos, foot_vel=foot_vel, foot_contact=foot_contact, self_collision=self_collision)
        dones, termination_terms = compute_terminations(base_pos=pos, projected_gravity=gravity, state_tensors=(q,qd,pos,quat,vel,ang), episode_steps=self.episode_steps, max_episode_steps=self.max_episode_steps)
        self.last_actions.copy_(actions)
        self._foot_air_time += self.control_dt
        self._foot_air_time[foot_contact] = 0.0
        done_ids = dones.nonzero().flatten(); terminal_obs = self._obs(); self.reset(done_ids)
        obs = self._obs(); validate_observation(obs, batch=self.num_envs)
        q, qd, pos, quat, vel, ang, gravity = self._last_state
        # Genesis has no terrain ray sensor in this environment; foot z relative
        # to the plane is the closest physically meaningful equivalent.
        foot_height = foot_pos[..., 2] - 0.0
        self.critic_obs = build_critic_observation(base_lin_vel=vel, base_ang_vel=ang,
            projected_gravity=gravity, joint_pos=q, joint_vel=qd, last_action=self.last_actions,
            command=self.commands.command, foot_height=foot_height, foot_air_time=self._foot_air_time,
            foot_contact=foot_contact.float(), foot_contact_forces=foot_force)
        return obs, rewards, dones, {'reward_terms': terms, 'termination_terms': termination_terms, 'terminal_observation': terminal_obs, 'critic_obs': self.critic_obs}
