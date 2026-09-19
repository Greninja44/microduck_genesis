#!/usr/bin/env python3
"""Print a non-mutating host and Python/CUDA diagnostic."""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

import psutil


def command(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"UNAVAILABLE: {exc}"


def main() -> int:
    print("== System ==")
    print(f"platform: {platform.platform()}")
    print(f"python: {sys.version.split()[0]} ({sys.executable})")
    print(f"pip_on_path: {shutil.which('pip') or 'UNAVAILABLE'}")
    print(f"pip_in_active_venv: {command([sys.executable, '-m', 'pip', '--version'])}")
    print(f"uv: {shutil.which('uv') or 'UNAVAILABLE'}")
    print(f"ram_available_gib: {psutil.virtual_memory().available / 2**30:.2f}")
    disk = psutil.disk_usage(Path.cwd())
    print(f"disk_free_gib: {disk.free / 2**30:.2f}")
    print("== NVIDIA ==")
    print(command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu", "--format=csv,noheader"]))
    print("== PyTorch ==")
    try:
        import torch
        print(f"torch: {torch.__version__}")
        print(f"torch_cuda_build: {torch.version.cuda}")
        available = torch.cuda.is_available()
        print(f"torch_cuda_available: {available}")
        if available:
            props = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info(0)
            print(f"torch_gpu: {props.name}")
            print(f"torch_vram_total_mib: {total / 2**20:.0f}")
            print(f"torch_vram_free_mib: {free / 2**20:.0f}")
    except Exception as exc:  # diagnostic must not hide import failures
        print(f"torch: UNAVAILABLE ({type(exc).__name__}: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
