#!/usr/bin/env python3
"""Passive MicroDuck run plus a deliberately small single-joint PD diagnostic."""
from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def ensure_wsl_cuda_loader() -> None:
    path, current = "/usr/lib/wsl/lib", os.environ.get("LD_LIBRARY_PATH", "")
    if os.path.isdir(path) and path not in current.split(":") and not os.environ.get("MICRODUCK_WSL_REEXEC"):
        env = {**os.environ, "LD_LIBRARY_PATH": f"{path}:{current}" if current else path, "MICRODUCK_WSL_REEXEC": "1"}
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def finite(values) -> bool:
    return bool(np.isfinite(host(values)).all())


def host(values) -> np.ndarray:
    """Convert Genesis' CPU or CUDA tensor return values for diagnostic printing."""
    import torch
    if isinstance(values, torch.Tensor):
        return values.detach().to(device="cpu").numpy()
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    return np.asarray(values)


def main() -> int:
    ensure_wsl_cuda_loader()
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--headless", action="store_true", help="run without a viewer (default)")
    mode.add_argument("--viewer", action="store_true", help="show Genesis viewer")
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    try:
        import genesis as gs
        import torch
        from microduck_genesis.robot import HOME_POSE, JOINT_NAMES, add_microduck, actuated_dof_indices, set_home_pose
        from microduck_genesis.scene import add_ground, make_scene

        gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu)
        print(f"backend={gs.backend} device={gs.device} viewer={args.viewer}")
        scene = make_scene(show_viewer=args.viewer, dt=0.005)
        add_ground(scene)
        robot = add_microduck(scene)
        scene.build()
        servo_ids = set_home_pose(robot)
        lower_limits, upper_limits = robot.get_dofs_limit(servo_ids)
        limits = np.stack((host(lower_limits), host(upper_limits)), axis=-1)
        print(f"joint_names={list(JOINT_NAMES)}")
        print(f"entity_dofs={robot.n_dofs}; actuated_servo_dofs={len(servo_ids)}")
        print(f"servo_limits={host(limits).tolist()}")
        print(f"initial_base_pos={host(robot.get_pos()).tolist()} initial_base_quat={host(robot.get_quat()).tolist()}")

        # Passive settling: holding no walking targets. The HOME pose is injected once only.
        for step in range(args.steps):
            scene.step()
            if step % 100 == 0 or step == args.steps - 1:
                pos, quat = robot.get_pos(), robot.get_quat()
                q = robot.get_dofs_position(servo_ids)
                print(f"passive step={step:04d} pos={host(pos).round(5).tolist()} quat={host(quat).round(5).tolist()} finite={finite(q) and finite(pos) and finite(quat)}")
                if not (finite(q) and finite(pos) and finite(quat)):
                    raise RuntimeError("NaN/Inf detected in MicroDuck state")

        # A small, bounded position/PD diagnostic on left_hip_yaw only, not locomotion.
        tested_name, tested_id, rest = JOINT_NAMES[0], servo_ids[0], HOME_POSE[0]
        lower, upper = limits[0]
        target = float(np.clip(rest + math.radians(3.0), lower + 0.01, upper - 0.01))
        robot.set_dofs_kp(np.array([0.55]), dofs_idx_local=[tested_id])
        robot.set_dofs_kv(np.array([0.053]), dofs_idx_local=[tested_id])
        for step in range(100):
            robot.control_dofs_position(np.array([target]), dofs_idx_local=[tested_id])
            scene.step()
            if step in (0, 49, 99):
                actual = float(host(robot.get_dofs_position([tested_id])).reshape(-1)[0])
                vel = float(host(robot.get_dofs_velocity([tested_id])).reshape(-1)[0])
                print(f"control joint={tested_name} target={target:.5f} actual={actual:.5f} velocity={vel:.5f} error={target-actual:.5f}")
        print("PASS: MJCF import, passive simulation, finite-state check, and bounded PD joint diagnostic completed.")
        return 0
    except Exception:
        print("FAIL: MicroDuck Genesis run failed")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
