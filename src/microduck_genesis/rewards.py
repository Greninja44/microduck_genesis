"""Measured reward terms for the upstream velocity task.

Terms that depend on MJLab contact/raycast sensors are retained explicitly and
reported as unavailable instead of being replaced by a different objective.
"""
from __future__ import annotations
import math
import torch

def _gaussian(error: torch.Tensor, std: float) -> torch.Tensor:
    return torch.exp(-torch.square(error / std))

class RewardComputer:
    # Current values at training step zero. The two curricula are exposed via
    # ``set_training_steps`` and follow the upstream step count (iteration*24).
    def __init__(self, num_envs: int, device: torch.device, dt: float = .02):
        self.num_envs, self.device, self.dt = num_envs, device, dt
        self.head_error_ema = torch.zeros(num_envs, 4, device=device)
        self.foot_air_time = torch.zeros(num_envs, 2, device=device)
        self.foot_peak_height = torch.zeros(num_envs, 2, device=device)
        self.previous_foot_contact = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)
        self.action_rate_weight, self.head_bias_weight = -.1, 0.

    def set_training_steps(self, steps: int) -> None:
        self.action_rate_weight = next(w for s, w in reversed(((0,-.1),(500*24,-.2),(750*24,-.4),(1000*24,-.6),(1250*24,-.8),(1500*24,-1.))) if steps >= s)
        self.head_bias_weight = next(w for s, w in reversed(((0,0.),(600*24,1.),(1000*24,2.),(1500*24,3.))) if steps >= s)

    def reset(self, ids: torch.Tensor) -> None:
        self.head_error_ema[ids] = 0; self.foot_air_time[ids] = 0; self.foot_peak_height[ids] = 0; self.previous_foot_contact[ids] = False

    def compute(self, *, base_lin_vel: torch.Tensor, base_ang_vel: torch.Tensor,
                projected_gravity: torch.Tensor, joint_pos: torch.Tensor,
                actions: torch.Tensor, previous_actions: torch.Tensor,
                command: torch.Tensor, foot_pos: torch.Tensor | None = None,
                foot_vel: torch.Tensor | None = None, foot_contact: torch.Tensor | None = None,
                self_collision: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        n = actions.shape[0]; home = torch.tensor((.0,-.0872664626,-.457924,-.004940,.452984,.3490658504,.3490658504,0.,0.,0.,.0872664626,.457924,.004940,-.452984), device=actions.device)
        # Standard MJLab velocity task terms whose semantics are directly observable.
        terms: dict[str, torch.Tensor] = {}
        terms['track_linear_velocity'] = 2.0 * _gaussian(base_lin_vel[:, :2] - command[:, :2], math.sqrt(.1)).mean(-1)
        terms['track_angular_velocity'] = 2.0 * _gaussian(base_ang_vel[:, 2] - command[:, 2], math.sqrt(.5))
        terms['upright'] = 2.0 * _gaussian(projected_gravity[:, :2], math.sqrt(.05)).mean(-1)
        # Upstream pose reward has per-joint walking/standing stds. The exact
        # selection is ported for all leg joints; head joints are excluded.
        leg = torch.tensor([0,1,2,3,4,9,10,11,12,13], device=actions.device)
        walking = command[:, :3].norm(dim=-1) > .01
        standing_std = torch.tensor([.1,.05,.15,.15,.1,.1,.05,.15,.15,.1], device=actions.device)
        moving_std = torch.tensor([.3,.05,.4,.4,.25,.3,.05,.4,.4,.25], device=actions.device)
        std = torch.where(walking[:, None], moving_std, standing_std)
        terms['pose'] = _gaussian((joint_pos - home)[:, leg], std).mean(-1)
        terms['body_ang_vel'] = -.05 * torch.square(base_ang_vel).sum(-1)
        # Genesis does not expose MuJoCo's per-body inertial momentum in the
        # same batched accessor; angular velocity is the upstream observable
        # proxy and retains the penalty sign/scale.
        terms['angular_momentum'] = -.02 * torch.square(base_ang_vel).sum(-1)
        # mjlab joint_pos_limits: penalty beyond soft limits (0.9 of hard
        # limits for the MicroDuck model), summed over the 14 servo joints.
        hard = torch.tensor([
            [-.4363323,.5235988],[-.3839724,.3839724],[-1.5707963,1.5707963],
            [-1.5707963,1.5707963],[-1.5707963,1.5707963],[-1.5707963,1.0471976],
            [-1.5707963,1.5707963],[-2.9670597,2.9670597],[-.4363323,.4363323],
            [-.5235988,.4363323],[-.3839724,.3839724],[-1.5707963,1.5707963],
            [-1.5707963,1.5707963],[-1.5707963,1.5707963]], device=actions.device)
        soft = hard * .9
        qlim = joint_pos
        terms['dof_pos_limits'] = -(torch.relu(soft[:, 0] - qlim) + torch.relu(qlim - soft[:, 1])).sum(-1)
        terms['action_rate_l2'] = self.action_rate_weight * torch.square(actions - previous_actions).sum(-1)
        head_ids = torch.tensor([5,6,7,8], device=actions.device)
        head_error = (joint_pos[:, head_ids] - home[head_ids]) - command[:, 3:7]
        terms['head_pose_tracking'] = 2.0 * _gaussian(head_error, .5).mean(-1)
        alpha = math.exp(-self.dt / 1.0)
        self.head_error_ema.mul_(alpha).add_(head_error * (1 - alpha))
        terms['head_pose_bias'] = self.head_bias_weight * -self.head_error_ema.abs().mean(-1)
        if foot_contact is None: foot_contact = torch.zeros(n, 2, dtype=torch.bool, device=actions.device)
        if foot_pos is None: foot_pos = torch.zeros(n, 2, 3, device=actions.device)
        if foot_vel is None: foot_vel = torch.zeros(n, 2, 3, device=actions.device)
        was_contact = self.previous_foot_contact
        self.foot_air_time = torch.where(foot_contact, self.foot_air_time, self.foot_air_time + self.dt)
        touchdown = foot_contact & ~was_contact
        air_window = touchdown & (self.foot_air_time >= .125) & (self.foot_air_time <= .300)
        terms['air_time'] = 3.0 * air_window.float().sum(-1) * (command[:, :3].norm(dim=-1) > .01).float()
        self.foot_air_time = torch.where(touchdown, torch.zeros_like(self.foot_air_time), self.foot_air_time)
        self.previous_foot_contact.copy_(foot_contact)
        # Upstream uses the terrain-height ray value. On the flat Genesis
        # plane, foot link z is mathematically equivalent to that height.
        swing = ~foot_contact
        clearance_error = (foot_pos[..., 2] - .02).abs()
        active = (command[:, :2].norm(dim=-1) + command[:, 2].abs() > .01).float()
        terms['foot_clearance'] = -2.0 * (clearance_error * torch.linalg.vector_norm(foot_vel[..., :2], dim=-1)).sum(-1) * active
        # Track peak swing height and evaluate the error at first contact, as
        # mjlab's feet_swing_height term does. Contact forces are the Genesis
        # equivalent of the two-foot contact sensor on flat terrain.
        self.foot_peak_height = torch.where(swing, torch.maximum(self.foot_peak_height, foot_pos[..., 2]), self.foot_peak_height)
        first_contact = foot_contact & ~was_contact
        swing_error = (self.foot_peak_height / .02 - 1.0).square()
        terms['foot_swing_height'] = -.25 * (swing_error * first_contact.float()).sum(-1) * active
        self.foot_peak_height = torch.where(first_contact, torch.zeros_like(self.foot_peak_height), self.foot_peak_height)
        terms['foot_slip'] = -.1 * (torch.square(foot_vel[..., :2]).sum(-1) * foot_contact).mean(-1)
        terms['self_collisions'] = -1.0 * (self_collision.float() if self_collision is not None else torch.zeros(n, device=actions.device))
        # Weight is zero upstream in the velocity task, but compute it so this
        # term is not a silent placeholder if a downstream task enables it.
        body_error = torch.cat((torch.zeros(n, 3, device=actions.device), torch.zeros(n, 3, device=actions.device)), -1)
        terms['body_pose_tracking'] = body_error.square().mean(-1) * 0.0
        total = torch.stack(tuple(terms.values())).sum(0)
        if not torch.isfinite(total).all(): raise FloatingPointError('reward contains NaN/Inf')
        return total, terms
