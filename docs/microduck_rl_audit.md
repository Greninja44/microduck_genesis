# Current MicroDuck velocity-task audit

This audit is of the local upstream source at `upstream/microduck_rl`, task
`Mjlab-Velocity-Flat-MicroDuck`, registered in
`src/mjlab_microduck/tasks/__init__.py`. It is the walking task, not one of the
stand-up, roller, spin, or pose tasks.

The actor is feed-forward: `MicroduckRlCfg` in
`src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py` has actor hidden
layers `(512, 256, 128)`, ELU activation, scalar Gaussian initial std `1.0`,
and observation normalization enabled. There is no observation history, frame
stacking, LSTM, GRU, recurrent hidden state, or actor privileged observation.
The critic is a separate feed-forward `(512,256,128)` ELU model. It has
privileged base linear velocity and sensor terms inherited from MJLab's
`make_velocity_env_cfg`; it is critic-only.

Actor ABI, in source insertion order: 3D `base_ang_vel`, 3D
`projected_gravity`, 14D `joint_pos` relative to default, 14D `joint_vel`,
14D previous action, then command `[twist(3), head_pose(4), body_pose(6)]`.
This is 61D. `AGENTS.md` explicitly preserves this shared deployment ABI.
Actor sensor noise is respectively angular velocity ±0.03, gravity ±0.01,
joint position ±0.001, and joint velocity ±0.25. Angular velocity/gravity use
0–1 control-step delay; joint velocity has exactly one control-step delay.
Joint positions have a constant encoder bias ±0.015 rad in actor observations.

The 14 outputs use `JointPositionActionCfg` with `scale=1.0`, in
`src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py:271`; the action is a
joint-position delta around the default/home pose. The ordered servos and home
pose are defined in `src/mjlab_microduck/robot/microduck_constants.py` and
`src/microduck_genesis/robot.py`: left leg 0–4, neck/head 5–8, right leg 9–13.
Upstream uses BAM M6 voltage actuators (firmware kp 200, voltage 6.5–8.2 V,
delay 3–6 physics lags), rather than the XML position actuator. The task runs
at 50 Hz (`NUM_STEPS_PER_ENV=24` rollout steps); the MJCF comment and task
source state a 0.005 s simulator step, giving a 0.02 s control interval.

Commands come from `VelocityCommandCommandOnly` and `UniformPoseCommand` in
`src/mjlab_microduck/tasks/mdp.py`. Twist ranges are vx [-0.4,0.4], vy
[-0.3,0.3], yaw [-1,1]. Two percent are exact stand commands. Fifteen percent
are turn-in-place, linear command zero and yaw magnitude [0.4,1.0]. Head
commands resample in 2–5 s and begin at ±[.05,.05,.07,.015]; body commands
are 6D ranges ±[.005,.005,.005,.05,.05,.05]. Both commands resample in 2–5 s.

Rewards configured directly in the velocity cfg: pose +1; upright +2 with
std sqrt(.05); air time +3, window [.125,.300] s; linear tracking +2, std
sqrt(.1); yaw tracking +2, std sqrt(.5); body angular velocity -.05; angular
momentum -.02; foot slip -.1; self collision -1; action-rate L2 -.1 initially,
curriculum to -1 at 1500×24 steps; head pose tracking +2 (std .5); body pose
tracking 0; and head-pose-bias 0 initially, then +1/+2/+3 at
600/1000/1500×24. Foot clearance and foot swing height are inherited MJLab
terms with target height .02 m. Exact inherited base-template formulae and
termination thresholds are not present in this local checkout, so they are
identified rather than guessed.

Terminations explicitly added upstream are `nan_state`, checking every joint,
root pose/velocity, and foot-contact forces (`mdp.py:1105`). Other task
terminations are inherited from `mjlab.tasks.velocity.make_velocity_env_cfg`;
that dependency is not vendored locally. Genesis therefore documents its
orientation/height fallback as an approximation.

Enabled randomization: trunk CoM reset ±3 mm curriculum to ±15 mm; head-body
CoM reset ±3 mm curriculum to ±10 mm; trunk mass/inertia startup scale
[.95,1.05]; BAM friction reset [.9,1.1]; armature reset [.9,1.1]; per-episode
encoder bias ±.015 rad; actor-only IMU orientation up to 6 degrees; foot
friction [.7,1.3]; velocity pushes every 3–6 s with x/y velocity changes
[-.3,.3]. KP/KD/damping/base orientation randomization are disabled.

PPO is rsl_rl through MJLab's `VelocityOnPolicyRunner`, 24 rollout steps, 5
epochs, 4 minibatches, LR 1e-3 adaptive schedule, gamma .99, lambda .95,
clip .2, entropy .01, value coefficient 1, clipped value loss, max grad norm
1. Save interval 250. Upstream export in `src/mjlab_microduck/export.py` bakes
the observation normalizer into ONNX.

| Feature | MuJoCo/MJLab implementation | Genesis implementation | Status |
|---|---|---|---|
| Actor observation ABI | exact 61D contract | `observations.py` | MATCHED |
| Actions/joint order | 14 position deltas, scale 1 | `actions.py` | MATCHED |
| Commands | twist/head/body sampling | `commands.py` | MATCHED |
| Actor/critic architecture | feed-forward 512/256/128 ELU | `ppo.py` | MATCHED |
| PPO settings | rsl_rl | compact PyTorch PPO | APPROXIMATED |
| BAM actuator/delay | M6 voltage actuator | Genesis position PD | APPROXIMATED |
| Sensor noise/delay | configured upstream | encoder bias and layout only | APPROXIMATED |
| Contact/raycast rewards | MJLab contact/ray sensors | explicit zero/unavailable terms | MISSING |
| CoM/mass/friction/armature DR | per-env MuJoCo model fields | recorded state only | MISSING |
| NaN termination | exact finite state + contact force | finite state | APPROXIMATED |
| Critic privileged observations | base velocity/contact/raycast | actor-only PPO input | MISSING |
| Recurrent state | none | none | NOT APPLICABLE |
