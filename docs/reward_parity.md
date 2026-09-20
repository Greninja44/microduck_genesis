# Upstream velocity reward inventory

Recovered from `upstream/microduck_rl/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`
using the locked MJLab reference environment.

| Reward | Weight | Required upstream state | Genesis status |
|---|---:|---|---|
| track_linear_velocity | 2.0 | base-frame velocity and twist command, std²=0.1 | CLOSE APPROXIMATION |
| track_angular_velocity | 2.0 | yaw velocity and twist command, std²=0.5 | CLOSE APPROXIMATION |
| upright | 2.0 | trunk orientation, std²=0.05 | CLOSE APPROXIMATION |
| pose | 1.0 | leg joint error, command-dependent per-joint std | APPROXIMATED |
| body_ang_vel | -0.05 | trunk angular velocity | CLOSE APPROXIMATION |
| angular_momentum | -0.02 | root angular momentum sensor | APPROXIMATED |
| dof_pos_limits | -1.0 | joint limits | APPROXIMATED |
| action_rate_l2 | -0.1 | current/previous action | MATCHED |
| air_time | 3.0 | foot contact sensor, window 0.125–0.300 s | APPROXIMATED |
| foot_clearance | -2.0 | terrain ray sensor, target 0.02 m | APPROXIMATED |
| foot_swing_height | -0.25 | terrain ray + contact sensor, target 0.02 m | APPROXIMATED |
| foot_slip | -0.1 | contact and foot tangential velocity | APPROXIMATED |
| self_collisions | -1.0 | subtree contact sensor | APPROXIMATED |
| head_pose_tracking | 2.0 | commanded head joints, std=0.5 | CLOSE APPROXIMATION |
| body_pose_tracking | 0.0 | body pose command | INTENTIONALLY DISABLED upstream |
| head_pose_bias | 0.0 | EMA head command error | INTENTIONALLY DISABLED at stage 0 |

No reward term silently returns a constant in the Genesis reward path. The
sensor-dependent terms are individually logged and marked approximated because
Genesis does not expose MJLab's sensor manager/raycast reductions.
