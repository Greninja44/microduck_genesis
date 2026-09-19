# MuJoCo ↔ Genesis differences

`scripts/compare_simulators.py` writes `logs/simulator_comparison.csv` with
Genesis state under a zero-action sequence. The local project has no installed
MuJoCo/MJLab runtime or trained policy, so it cannot populate a valid paired
trace yet; blank MuJoCo fields are deliberate.

Known material differences are: MuJoCo training uses the custom BAM M6 voltage
actuator and 3–6 physics-step actuator delay; Genesis uses its imported model
with Genesis position PD. MuJoCo rewards use explicit ContactSensor and
TerrainHeightSensor measurements, while the current Genesis API path exposes
no equivalent per-foot sensor in this environment. Contact solver, friction,
integration, and imported mesh inertias must be measured by a paired trace
before interpreting zero-shot transfer or locomotion quality.
