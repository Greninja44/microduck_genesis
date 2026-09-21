# Reward parity

The active MicroDuck velocity configuration is derived from
`upstream/microduck_rl/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`.
Genesis keeps every term separate in `RewardComputer` and exposes it through
`info["reward_terms"]`.

| term | upstream weight | Genesis weight | status |
|---|---:|---:|---|
| track linear velocity | 2.0 | 2.0 | MATCHED |
| track angular velocity | 2.0 | 2.0 | MATCHED |
| upright | 2.0 | 2.0 | MATCHED |
| pose | 1.0 | 1.0 | MATCHED |
| body angular velocity | -0.05 | -0.05 | MATCHED |
| angular momentum | -0.02 | -0.02 | APPROXIMATED |
| dof position limits | -1.0 | -1.0 | MATCHED |
| action rate | -0.1 curriculum | -0.1 curriculum | MATCHED |
| air time | 3.0, [.125,.300], command > .01 | same | APPROXIMATED |
| foot clearance | -2.0, target .02, command > .01 | same on plane | EQUIVALENT_ON_FLAT |
| foot swing height | -.25, target .02, command > .01 | same on plane | EQUIVALENT_ON_FLAT |
| foot slip | -.1, command > .01 | -.1 | EQUIVALENT_ON_FLAT |
| soft landing | removed by upstream velocity config | disabled | MATCHED |
| self collisions | -1.0 | -1.0 | APPROXIMATED |
| head pose tracking | 2.0 | 2.0 | MATCHED |
| head pose bias | curriculum | curriculum | APPROXIMATED |
| body pose | 0.0 | 0.0 | MATCHED |

`soft_landing` exists in generic MJLab but is explicitly removed by the
current MicroDuck velocity task. Flat-plane foot height is equivalent to the
upstream terrain-height ray value for this task; rough-terrain ray parity is
deferred.

Status counts: MATCHED 11, EQUIVALENT_ON_FLAT 3, APPROXIMATED 4, MISSING 0.
