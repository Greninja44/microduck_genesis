# Episode and PPO parity

| Parameter | Upstream MJLab velocity task | Genesis | Status |
|---|---:|---:|---|
| physics timestep | 0.005 s | 0.005 s | MATCHED |
| control decimation | 4 | 4 | MATCHED |
| control timestep | 0.020 s | 0.020 s | MATCHED |
| episode duration | 20.0 s | 20.0 s | MATCHED |
| max control steps | ceil(20/0.02) = 1000 | 1000 | MATCHED |
| orientation termination | `acos(-projected_gravity.z) > 70°` | equivalent `z > -cos(70°)` | MATCHED |
| timeout | separate `time_out=True` truncation | reported separately and bootstrapped in PPO returns | MATCHED |
| initial joint position | default pose + zero offset | HOME_POSE | MATCHED |
| initial joint velocity | zero | zero | MATCHED |
| base z | uniform 0.12–0.13 m | fixed 0.125 m | APPROXIMATED |
| base x/y/yaw | reset randomization | fixed zero | APPROXIMATED |
| base velocity | upstream reset event | zero | APPROXIMATED |
| command range | x/y ±1.0, yaw ±0.5 in base task; MicroDuck override is task-specific | fixed forward validation command +0.2 m/s | INTENTIONAL VALIDATION MODE |
| rollout horizon | 24 control steps/env | 24 control steps/env | MATCHED |

The current post-fix validation deliberately disables domain randomization and
uses a fixed forward command. It therefore does not exercise the upstream
reset/event distribution. Timeout bootstrap follows rsl_rl: time-limit
transitions receive `gamma * V` before GAE, while true terminations do not.
