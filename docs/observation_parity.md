# Observation parity

| Field | MuJoCo/MJLab implementation | Genesis implementation | Status | Notes |
|---|---|---|---|---|
| `base_ang_vel` | actor IMU frame, rad/s; delay 0–1 (updated every 64 control steps), noise ±0.03, 6° mounting DR | `RigidEntity.get_ang()` | APPROXIMATED | Physical frame matches; sensor model pending. |
| `projected_gravity` | inverse base quaternion applied to world gravity `[0,0,-1]`; actor delay 0–1; noise ±0.01; same IMU misalignment | inverse wxyz quaternion applied to `[0,0,-1]` | APPROXIMATE | Frame, units, order, and sign are matched; sensor delay/noise/misalignment remain. |
| `joint_pos_rel` | 14 non-`passive_*` joints, `joint_pos - default_joint_pos`; actor encoder bias ±0.015 rad | 14 explicit Genesis DOFs, subtracts `HOME_POSE`, adds per-env bias | APPROXIMATE | Joint order and radians match; Genesis state is the physical joint rather than BAM output-side encoder semantics. |
| `joint_vel` | 14 non-passive joints, rad/s, exactly one control-step lag, noise ±0.25 | previous control-step Genesis velocity | APPROXIMATE | Lag and units match; noise is not currently injected. |
| `last_action` | previous 14 raw policy outputs | previous 14 action tensor | MATCHED | No filtering; upstream policies are unfiltered. |
| `twist` | command manager `[vx,vy,yaw_rate]`, base-frame m/s/rad/s | seedable command tensor | APPROXIMATE | Sampling matches ranges and turn bucket; no world-heading update is needed for command-only mode. |
| `head_pose` | 4 deltas `[neck_pitch,head_pitch,head_yaw,head_roll]`, rad, 2–5 s resampling | 4 seedable command values | APPROXIMATE | Initial ranges/order match; curriculum resampling is not yet implemented. |
| `body_pose` | 6 deltas `[x,y,z,roll,pitch,yaw]`, m/rad, 2–5 s resampling | 6 seedable command values | APPROXIMATE | Order and initial ranges match; curriculum resampling is not yet implemented. |

There is no actor frame stacking or recurrent state. The actor dimension is 61.
The reference critic ABI is 76D: `[base_lin_vel(3), base_ang_vel(3),
projected_gravity(3), joint_pos(14), joint_vel(14), last_action(14),
twist(3), foot_height(2), foot_air_time(2), foot_contact(2),
foot_contact_forces(6), head_pose(4), body_pose(6)]`. Genesis now keeps this
stream separate in `critic_observations.py`; foot height/air time are closest
equivalents because Genesis does not expose the MJLab sensor manager.

No normalization, clipping, or unit conversion is applied to the 61 raw fields
by `observations.py`; upstream normalization is performed by rsl_rl's running
normalizer and baked into exported ONNX. A Genesis checkpoint must use the
same normalizer state before deployment.
