# MuJoCo / Genesis static parity (phase 1)

The source of truth below is the current upstream walking task:
`upstream/microduck_rl/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`
and its direct MJCF asset `robot/microduck/robot_walk.xml` (commit `cb70b79`).
This is a configuration comparison only, not a claim of physical parity.

| Field | MuJoCo / mjlab source | Genesis phase-1 configuration | Status |
|---|---|---|---|
| Robot asset | `robot_walk.xml`, MJCF, `meshdir="assets"` | Same unmodified MJCF loaded directly | MATCHED |
| Floating base | `trunk_base_freejoint` | Preserved by MJCF import | MATCHED |
| Servo ordering | 14 joints, left leg, head, right leg | Same named lookup/order in `robot.py` | MATCHED |
| Joint axes and hard limits | XML hinge axes/ranges | Source MJCF import; runtime limits printed | MATCHED |
| Link mass / inertia | XML `<inertial>` values (15 bodies) | Direct source MJCF import | MATCHED |
| Visual meshes | 38 STL mesh assets, original paths/scales | Direct source MJCF import | MATCHED |
| Collision geometry | Walking model has foot collision meshes plus local self-collision definitions | Imported from MJCF; cross-entity filtering differs (Genesis limitation) | APPROXIMATED |
| Gravity | MuJoCo default `(0, 0, -9.81)` | Genesis default gravity; test validates falling toward plane | MATCHED |
| Simulation timestep | Velocity source documents `0.005 s` | `SimOptions(dt=0.005)` | MATCHED |
| Policy control timestep | 50 Hz / `0.02 s` | No policy in phase 1 | NOT YET MATCHED |
| XML PD | `kp=0.55`, `kv=0`, force ±0.96, ctrl ±10 | Small diagnostic uses kp 0.55, kv 0.053 only on one joint | APPROXIMATED |
| Training actuator | BAM M6 voltage model, `kp_fw=200`, delay 3..6 ticks | Genesis built-in position PD only | NOT YET MATCHED |
| Action scale | `JointPositionActionCfg.scale=1.0` | No action interface / RL step yet | NOT YET MATCHED |
| Contact friction | RL uses foot friction 1.0, condim 3; other named collision geoms condim 1 | Plane / MJCF import only; exact Genesis contact material mapping not tuned | NOT YET MATCHED |
| Solver iterations/contact capacity | MuJoCo/mjlab-specific | Genesis solver defaults | NOT APPLICABLE |
| Default pose | HOME/STAND2 values set by current task configuration | Same 14 servo HOME values injected after build | MATCHED |
| Observations | 61D actor: 48 proprioception + twist 3 + head pose 4 + body pose 6 | No observations (intentionally no RL) | NOT YET MATCHED |

## Important limitation

Genesis World documents that global MJCF options (including timestep, integrator,
and constraint solver) are ignored during MJCF loading because they belong to the
Genesis scene. The project explicitly supplies the source task's 0.005 s timestep
at the scene level. Contact behavior and BAM actuator dynamics remain future
parity work, after basic import validation.
