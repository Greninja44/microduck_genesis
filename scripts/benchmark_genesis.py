#!/usr/bin/env python3
"""Conservative Genesis batch benchmark; never advances after a failed/low-headroom run."""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def ensure_wsl_cuda_loader() -> None:
    path, current = "/usr/lib/wsl/lib", os.environ.get("LD_LIBRARY_PATH", "")
    if os.path.isdir(path) and path not in current.split(":") and not os.environ.get("MICRODUCK_WSL_REEXEC"):
        env = {**os.environ, "LD_LIBRARY_PATH": f"{path}:{current}" if current else path, "MICRODUCK_WSL_REEXEC": "1"}
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def nvidia_memory_mib() -> str:
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip()
    except Exception:
        return ""


def main() -> int:
    ensure_wsl_cuda_loader()
    import genesis as gs
    import torch
    from microduck_genesis.robot import add_microduck
    from microduck_genesis.scene import add_ground, make_scene
    if not torch.cuda.is_available():
        print("SKIP: CUDA is unavailable; a CPU benchmark is not a valid RTX 4050 capacity measurement.")
        return 0
    gs.init(backend=gs.gpu)
    output = ROOT / "logs/genesis_benchmark.csv"
    rows = []
    for n_envs in (1, 64, 128, 256, 512, 1024):
        try:
            scene = make_scene(dt=0.005)
            add_ground(scene)
            add_microduck(scene)
            scene.build(n_envs=n_envs, env_spacing=(1.0, 1.0))
            steps = 100
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(steps): scene.step()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            free, total = torch.cuda.mem_get_info()
            row = {"env_count": n_envs, "steps": steps, "elapsed_s": f"{seconds:.4f}", "sim_steps_per_s": f"{n_envs*steps/seconds:.1f}", "vram_used_mib": nvidia_memory_mib(), "vram_free_mib": f"{free/2**20:.0f}", "vram_total_mib": f"{total/2**20:.0f}"}
            rows.append(row)
            print(row)
            if free / total < 0.20:
                print("STOP: remaining VRAM below 20% safety threshold.")
                break
        except Exception as exc:
            print(f"STOP: {n_envs} environments failed: {type(exc).__name__}: {exc}")
            break
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["env_count", "steps", "elapsed_s", "sim_steps_per_s", "vram_used_mib", "vram_free_mib", "vram_total_mib"])
        writer.writeheader(); writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
