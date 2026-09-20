# BAM actuator parity

The locked upstream environment uses `better-actuator-models` at commit
`62bd8ce12154340be97e06f7f41a0ca8f116d967`, `xl330/m6.json`, and
`kp_fw=200`.  Its actuator configuration is in
`upstream/microduck_rl/src/mjlab_microduck/robot/microduck_constants.py`.

For each control target, firmware computes

`duty = clip((q_target - q_encoder) * kp_fw, -1, 1)` and `V = vin * duty`.

The motor torque is

`tau_motor = Kt*V/R - Kt^2*qd/R`,

with `Kt=0.3660134969`, `R=2.8113923539`, and armature
`0.0018077433 kg m²`.  Upstream samples `vin` in `[6.5, 8.2] V`, applies a
lag of 3–6 control steps, and uses a load voltage drop gain in `[0, 0.2]`
with a 6 V floor.  The deployed task uses the FrictionDRBam subclass, which
multiplies the velocity-independent friction budget by an episode scale.

The M6 budget is `base + Stribeck + directional load + quadratic load`, where
`S=exp(-(abs(qd)/2.8903721)^8.6832599)`.  The exact coefficients are captured
in `src/microduck_genesis/bam.py`.

MuJoCo writes this budget to per-world `dof_frictionloss` and `dof_damping`;
its constraint solver clips the stopping torque. Genesis has no equivalent
per-step static-friction constraint API, so Genesis currently applies the same
motor equation and a signed Coulomb plus viscous torque. Motor equations are
**EXACT**; contact/friction solver semantics are **APPROXIMATED**.
