# Short-horizon trajectory parity

The deterministic trace workflow is:

```bash
PYTHONPATH=upstream/microduck_rl/src .venv-mjlab/bin/python scripts/reference_mujoco_trace.py
PYTHONPATH=src .venv/bin/python scripts/replay_mujoco_actions_genesis.py --device cuda
python scripts/compare_trajectory.py
```

`logs/mujoco_reference.npz` is the authoritative upstream trace and
`logs/genesis_replay.npz` contains the same recorded action sequence replayed in
Genesis. `logs/trajectory_parity.csv` reports 1, 5, 10, 25, 50, and 100 control
step errors using mean absolute state errors and quaternion angular distance.

Terrain height is used by the active flat-ground task's privileged critic and
foot rewards. Genesis currently uses foot height above the plane as the closest
equivalent to MJLab's two-ray foot-height sensor; this is classified as
APPROXIMATED, while no terrain scan is exposed to the actor.
