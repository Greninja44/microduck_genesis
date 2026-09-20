# MuJoCo/MJLab → Genesis parity scorecard

| Feature | Status | Evidence / limitation |
|---|---|---|
| robot geometry, joint order, limits | MATCHED | upstream walking MJCF imported unchanged; 14 explicit actuated joints |
| initial pose | MATCHED | `HOME_POSE` copied from upstream constants |
| physics/control timestep | MATCHED | 5 ms physics, 20 ms control, four-step decimation |
| action ABI | MATCHED | 14D, target is home plus raw action, scale 1 |
| actor observation ABI | CLOSE APPROXIMATION | exact 61D order; sensor noise, delay, and IMU mounting DR pending |
| critic observation ABI | CLOSE APPROXIMATION | exact upstream 76D term order; Genesis sensor fields are equivalents |
| BAM motor equations | MATCHED | numerical unit test max error 1.11e-16 Nm |
| BAM friction constraint | APPROXIMATED | Genesis signed Coulomb/viscous torque lacks MuJoCo static-friction solver |
| contacts | APPROXIMATED | link net forces and padded contact API; site/geom sensor semantics differ |
| raycasts / terrain height | APPROXIMATED | flat-plane foot z used; no MJLab terrain ray sensor |
| reward terms | CLOSE APPROXIMATION | all velocity, pose, contact, slip, clearance, and self-collision terms represented; formulas differ for sensor terms |
| sensor noise/delay | MISSING | clean Genesis state currently feeds actor |
| model-field domain randomization | MISSING | episode bookkeeping exists; mass/COM/friction field writes not ported |
| termination | CLOSE APPROXIMATION | timeout, 70° orientation gate, height and NaN checks; upstream also terrain bounds/sensor NaN |
| reset independence | MATCHED | vectorized per-environment reset |
| command generation | CLOSE APPROXIMATION | ranges/order/seed match; curriculum not ported |
| MuJoCo deterministic reference | NOT TESTED | MJLab imports/configuration work; paired simulation harness not complete |
| policy transfer | NOT TESTED | no compatible official checkpoint located |

The scorecard is intentionally qualitative; no aggregate numerical score is
claimed.
