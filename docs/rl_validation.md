# GPU RL validation

The original CUDA failure was environmental: the sandbox denied NVIDIA device
access. The venv already contained CUDA-enabled `torch 2.9.1+cu126` and all
CUDA runtime wheels. With host GPU access, `nvidia-smi` reports an NVIDIA
GeForce RTX 4050 Laptop GPU, driver 596.49, CUDA driver 13.2; PyTorch reports
CUDA available and the 2048×2048 CUDA matmul passes.

Genesis 1.4.1 reports `backend=gs.cuda` and executes the imported MJCF on the
RTX 4050. The one-environment smoke ran 1000 control steps (4000 physics
steps), produced finite observations/actions/rewards/state, and wrote
`logs/single_env_smoke.csv`. 1, 8, and 64 environment smoke checks pass with
independent batched shapes and weighted reward sums.

The RL environment now uses vectorized BAM force control by default. Each 20 ms
policy action is converted to `HOME_POSE + action`; BAM torque is recomputed at
each of the four 5 ms physics substeps and sent through Genesis
`control_dofs_force`. The previous position-control path is rejected by an
actuator-mode guard.

The earlier five-iteration PPO smoke at 64 environments completed and saved five
checkpoints. Mean reward was 4.930 → 5.171. The deterministic 64-environment
forward-command overfit test completed ten iterations with mean reward
5.113 → 5.700. After wiring the actor sensor model, a bounded one-iteration
final-path smoke completed at reward 5.058, policy loss -0.0834, value loss
25.1883, and entropy 19.776.

With BAM force control enabled, a bounded 64-environment PPO run completed four
iterations before the execution window ended. Rewards were 5.810 → 5.843;
policy, value, and entropy metrics remained finite. A short one-environment
force-control smoke also passed without NaN/Inf state or torque values.

The final one-environment path passes 1000 control steps with sensor model,
privileged critic, rewards, contacts, terminations, and finite-state checks.

Genesis replay is complete for the five-step MuJoCo reference. Cold startup is
about 18–25 seconds and warm replay startup is about 30–40 seconds because
Genesis 1.4.1 still builds the visualizer/kernel path during headless scene
creation. Replay metrics are recorded in `docs/trajectory_parity.md`.

| environments | RL sim FPS (physics steps/s) | PPO FPS (env control steps/s) | nvidia-smi used | process RAM |
|---:|---:|---:|---:|---:|
| 64 | 13,000–20,000 in smoke; 64-env PPO varies with warmup | 64-env smoke 162–851 | 924 MiB during smoke | ~2.7 GiB |
| 128 | 8,543 | 556 | 859 / 887 MiB | 2.57 / 2.69 GiB |
| 256 | 16,662 | 928 | 925 / 1000 MiB | 2.30 / 2.67 GiB |
| 512 | 38,339 | 1,338 | 963 / 1413 MiB | 2.30 / 2.66 GiB |
| 1024 | 81,709 | 4,326 | 989 / 1112 MiB | 2.33 / 2.67 GiB |

The first proper training experiment should use **256 environments**. It leaves
substantial measured VRAM headroom while giving a stable 928 PPO env-steps/s;
512 is safe for this compact trainer but leaves less room for future privileged
critic tensors and exact actuator state.

MuJoCo/MJLab policy transfer was **NOT TESTED**: no compatible trained ONNX or
checkpoint exists locally. The paired five-step MuJoCo/Genesis trace is now
**TESTED**. A 100-step trace, 100-iteration learning experiment, checkpoint
evaluation, and visual policy evaluation remain outstanding.

Known remaining approximations are Genesis static-friction solver semantics,
exact MuJoCo contact/raycast reward formulas, model-field domain randomization,
and dynamic torque capture from Genesis's position-control interface. Actor
sensor noise/delay is now wired; IMU mounting randomization and some model-field
randomization remain deferred. These are documented individually in the parity
docs and are why this result is a validated Genesis pipeline rather than a
claim of simulator or sim-to-real parity.
