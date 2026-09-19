"""WSL CUDA loader compatibility, imported automatically by Python's site module."""
from __future__ import annotations

import os

_WSL_CUDA = "/usr/lib/wsl/lib"
if os.path.isdir(_WSL_CUDA):
    previous = os.environ.get("LD_LIBRARY_PATH", "")
    if _WSL_CUDA not in previous.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{_WSL_CUDA}:{previous}" if previous else _WSL_CUDA
