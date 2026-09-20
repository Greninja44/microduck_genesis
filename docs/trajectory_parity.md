# Short-horizon trajectory parity

The deterministic trace workflow is:

```bash
PYTHONPATH=upstream/microduck_rl/src .venv-mjlab/bin/python scripts/reference_mujoco_trace.py
PYTHONPATH=src .venv/bin/python scripts/replay_mujoco_actions_genesis.py --device cuda
python scripts/compare_trajectory.py
```

`logs/mujoco_reference.npz` is the authoritative upstream trace and
`logs/genesis_replay.npz` contains the same five recorded actions replayed in
Genesis. The current trace is intentionally five steps; longer horizons have
not been fabricated. The corrected joint-position comparison adds the upstream
home pose before comparing against Genesis absolute joint positions.

Measured errors:

| Horizon | Joint position MAE | Joint velocity MAE | Base linear velocity MAE | Base angular velocity MAE | Orientation error |
|---:|---:|---:|---:|---:|---:|
| 1 step | 7.6e-13 rad | 2.3e-9 rad/s | 0.0654 m/s | 2.7e-8 rad/s | 0 rad |
| 5 steps | 0.0624 rad | 0.9434 rad/s | 0.0194 m/s | 0.8298 rad/s | 0.1283 rad |

The one-step state is effectively aligned. Divergence is visible by five steps,
consistent with actuator/contact/integration differences between simulators.
`logs/trajectory_parity.csv` contains the same values.

Genesis headless replay still performs a one-time visualizer/kernel build in
Genesis 1.4.1, even when no viewer is requested. The replay itself completed
successfully after removing contact visualization from headless entities.

Terrain height is used by the active flat-ground task's privileged critic and
foot rewards. Genesis currently uses foot height above the plane as the closest
equivalent to MJLab's two-ray foot-height sensor; this is classified as
APPROXIMATED, while no terrain scan is exposed to the actor.
