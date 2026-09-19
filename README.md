# MicroDuck Genesis

Genesis implementation work for the current upstream
`Mjlab-Velocity-Flat-MicroDuck` task. The audited actor ABI is 61D and the
policy has 14 joint-position actions. See [the audit](docs/microduck_rl_audit.md)
before using a checkpoint.

Install using the existing project environment:

```bash
cd /home/batman/microduck_genesis
source .venv/bin/activate
```

Genesis and model checks:

```bash
python scripts/check_system.py
python scripts/test_genesis.py
PYTHONPATH=src python scripts/inspect_observations.py
PYTHONPATH=src python scripts/run_microduck.py --headless
```

Run a one-environment RL reward smoke check, then increase only after it is
stable:

```bash
PYTHONPATH=src python scripts/test_rewards.py --num-envs 1 --device cpu
PYTHONPATH=src python scripts/test_rewards.py --num-envs 8 --device cuda
PYTHONPATH=src python scripts/test_rewards.py --num-envs 64 --device cuda
```

Start a deliberately short PPO pipeline check:

```bash
PYTHONPATH=src python scripts/train.py --num-envs 64 --headless --device cuda --iterations 5
```

Resume or evaluate a local checkpoint:

```bash
PYTHONPATH=src python scripts/train.py --num-envs 256 --headless --device cuda --iterations 100 --checkpoint checkpoints/model_0004.pt --resume
PYTHONPATH=src python scripts/evaluate.py --checkpoint checkpoints/model_0004.pt --viewer --device cuda
```

If an actual exported upstream 61D/14D feed-forward ONNX policy is made
available locally, test it without ABI adaptation:

```bash
PYTHONPATH=src python scripts/eval_mujoco_policy_in_genesis.py --onnx /path/to/policy.onnx --device cuda
```

For capacity checks, run `PYTHONPATH=src python scripts/benchmark_genesis.py`.
The target RTX 4050 has 6 GB VRAM: begin training at 256 environments, then
test 512 before considering 1024. Do not use measured simulator-only capacity
as the training count; leave VRAM for PPO.

Troubleshooting: under WSL, run the helper scripts normally; they re-exec with
`/usr/lib/wsl/lib` when needed. If `torch.cuda.is_available()` is false, use
the CPU smoke path only. Do not treat CPU validation as an RTX throughput or
memory result.
