#!/usr/bin/env python3
"""Headless Genesis World smoke test with clear diagnostics."""
from __future__ import annotations

import traceback
import os
import sys


def ensure_wsl_cuda_loader() -> None:
    """Re-exec so dlopen sees WSL's CUDA driver path from process startup."""
    path = "/usr/lib/wsl/lib"
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if os.path.isdir(path) and path not in current.split(":") and not os.environ.get("MICRODUCK_WSL_REEXEC"):
        env = {**os.environ, "LD_LIBRARY_PATH": f"{path}:{current}" if current else path, "MICRODUCK_WSL_REEXEC": "1"}
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def main() -> int:
    ensure_wsl_cuda_loader()
    try:
        import genesis as gs
        import torch
        backend = gs.gpu if torch.cuda.is_available() else gs.cpu
        print(f"Genesis version: {gs.__version__}")
        print(f"PyTorch: {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
        print(f"Requested backend: {backend}")
        gs.init(backend=backend)
        print(f"Initialized backend: {gs.backend}; device: {gs.device}")
        scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.005), show_viewer=False)
        scene.add_entity(gs.morphs.Plane())
        scene.add_entity(gs.morphs.Box(pos=(0.0, 0.0, 0.35), size=(0.1, 0.1, 0.1)))
        scene.build()
        for _ in range(100):
            scene.step()
        print("PASS: initialized, built plane+box scene, and completed 100 headless steps.")
        return 0
    except Exception:
        print("FAIL: Genesis smoke test failed")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
