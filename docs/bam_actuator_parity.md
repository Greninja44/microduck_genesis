# BAM actuator parity

The upstream source is `upstream/microduck_rl/src/mjlab_microduck/actuator/friction_dr_bam.py`.
It subclasses `bam.mjlab.BamActuator`; the BAM dependency is pinned in the
upstream `uv.lock` to commit `62bd8ce...` on the `mjlab_frictionloss` branch,
but is not installed in this Genesis environment.

The local subclass confirms these equations and hooks:

* `friction_scale` multiplies the complete BAM velocity-independent friction
  budget returned by `_compute_friction_budget` (Coulomb, Stribeck, and
  load-dependent terms); viscous velocity-proportional friction is nominal.
* `BacklashEncoderBamActuator` closes the position loop on
  `q_servo + q_passive_backlash`, while motor velocity remains motor-side.
* Upstream MicroDuck config uses model `m6`, firmware `kp_fw=200`, voltage
  range 6.5–8.2 V, load voltage-drop gain 0–0.2, voltage floor 6.0 V, and
  actuator delay 3–6 physics steps. The regular MJCF position actuator is not
  the training actuator.

Genesis currently applies `control_dofs_position` at each of four 5 ms steps
   per 20 ms control action. This preserves the control interval and joint
   targets but does not implement BAM voltage, load sag, friction, backlash
   encoder feedback, or delay. The correct classification is
   **APPROXIMATED**, and current zero-shot transfer/training results must not
   be interpreted as BAM-equivalent.

Implementing exact BAM requires either installing the pinned `better-actuator-
models` package and translating its torque equation, or a Genesis force-control
callback that reproduces the voltage/current/friction state per environment.
