# Post-fix frame and sign audit

The upstream MJLab termination implementation is
`mjlab.envs.mdp.terminations.bad_orientation`:

```python
projected_gravity = asset.data.projected_gravity_b
return torch.acos(-projected_gravity[:, 2]).abs() > limit_angle
```

MJLab's projected gravity is the world gravity vector expressed in the base
frame, so an identity quaternion is `[0, 0, -1]`. The Genesis adapter now uses
the same convention. With the equivalent 60-degree limit, Genesis tests
`projected_gravity[:, 2] > -0.5`, which is algebraically the same condition.

The related quantities are checked as follows:

| Quantity | Upstream convention | Genesis convention | Status |
|---|---|---|---|
| projected gravity | body-frame world gravity, upright z = -1 | wxyz quaternion rotation of `[0, 0, -1]`, upright z = -1 | MATCHED |
| upright reward | Gaussian on x/y projected gravity | same x/y Gaussian | MATCHED |
| forward velocity | base-frame linear velocity x | `get_vel(relative=True)[..., 0]` | MATCHED |
| command tracking | command twist x/y against base-frame velocity | same tensor components | MATCHED |
| base angular velocity | base-frame angular velocity | Genesis angular velocity accessor | APPROXIMATED |
| orientation termination | `acos(-g_z) > limit_angle` | `g_z > -0.5` for 60 degrees | MATCHED |

Regression cases cover identity, 20-degree roll/pitch, and 100-degree
forward/backward/left/right falls.
